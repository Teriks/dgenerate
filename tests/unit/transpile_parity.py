"""Compare transpiled scripts with the config interpreter. Does not generate images."""

import enum
import io
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

import dgenerate
import dgenerate.arguments
import dgenerate.batchprocess.configrunner as configrunner
import dgenerate.invoker as invoker
import dgenerate.messages as messages
import dgenerate.plugin as plugin
import dgenerate.renderloop as renderloop
import dgenerate.renderloopconfig as renderloopconfig
import dgenerate.textprocessing as textprocessing

import dgenerate.transpile as transpile_config

IMAGE = Path(tempfile.gettempdir()) / '_dgenerate_adversarial.png'
_OBJECT_ID = re.compile(r' at 0x[0-9A-Fa-f]+>')
_LINE_WRAP = re.compile(r'^Error on line -?\d+:\s*')
_ON_LINE = re.compile(r'\bon line -?\d+\b')
_LINE_CITE = re.compile(r' on line -?\d+')
_CONTINUATION = re.compile(
    r'^Error inside template continuation from line -?\d+ to line -?\d+:\n'
)
# Fields argparse puts on DgenerateArguments. A relaxed script builds a
# RenderLoopConfig and leaves them unset, which is the same as the default.
_PARSER_DEFAULTS = {
    'plugin_module_paths': [],
    'verbose': False,
    'global_config': None,
}


def _deterministic_seeds(n):
    return list(range(1, int(n) + 1))


def _write_png(path: Path):
    if path.exists():
        return
    import struct
    import zlib

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = zlib.compress(b'\x00\x00\x00\x00')
    path.write_bytes(b''.join([
        b'\x89PNG\r\n\x1a\n',
        chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)),
        chunk(b'IDAT', raw),
        chunk(b'IEND', b''),
    ]))


def _norm(value, seen=None):
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return (type(value).__name__, value.name)
    if isinstance(value, Path):
        return value.as_posix()
    identity = id(value)
    if identity in seen:
        return '<cycle>'
    if isinstance(value, dict):
        seen.add(identity)
        return {str(key): _norm(item, seen) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        seen.add(identity)
        return [_norm(item, seen) for item in value]
    if isinstance(value, set):
        seen.add(identity)
        return sorted((_norm(item, seen) for item in value), key=repr)
    if hasattr(value, '__dict__'):
        seen.add(identity)
        data = {
            key: _norm(item, seen)
            for key, item in vars(value).items()
            if not key.startswith('_') and not callable(item)
        }
        return (type(value).__name__, data)
    return repr(value)


def _snapshot(config):
    names = []
    for cls in reversed(type(config).__mro__):
        names.extend(getattr(cls, '__annotations__', {}))
    seen = set()
    data = {}
    for name in names:
        if name in seen or name.startswith('_'):
            continue
        seen.add(name)
        try:
            value = getattr(config, name)
        except Exception:
            continue
        if callable(value):
            continue
        data[name] = _norm(value)
    return data


def _parse_without_check(args):
    parsed, unknown = dgenerate.arguments.parse_known_args(
        list(args), throw=True, log_error=False, no_model=False, no_help=True,
    )
    if unknown:
        raise ValueError('unknown arguments: ' + ' '.join(unknown))
    return parsed


def _plant(loop):
    loop._written_images = io.StringIO(str(IMAGE) + '\n')
    loop._written_animations = io.StringIO(str(IMAGE) + '\n')


def _failure(exc):
    text = str(exc).strip()
    if not text:
        text = exc.__class__.__name__
    lines = []
    for line in text.splitlines():
        stripped = _LINE_WRAP.sub('', line.strip())
        if stripped:
            lines.append(stripped)
    return _OBJECT_ID.sub(' at 0x0>', '\n'.join(lines))


class _Trace:
    def __init__(self):
        self.events = []

    def log(self, *args, level=messages.INFO, **kwargs):
        text = ' '.join(str(arg) for arg in args)
        if text.startswith('Processing Arguments'):
            return
        if text.startswith('\\env "'):
            text = '\\env'
        self.events.append((level, _OBJECT_ID.sub(' at 0x0>', text)))

    def invoke(self, config):
        self.events.append(('invoke', _snapshot(config)))


def _events_of(trace, exc):
    events = list(trace.events)
    if exc is not None:
        events.append(('error', _failure(exc)))
    return events


def _short(value):
    text = repr(value)
    return text if len(text) <= 500 else text[:500] + '...'


def _relax_error(text):
    """Drop the line numbers and continuation wrapper that relaxed scripts omit."""
    return _ON_LINE.sub('on line 0', _CONTINUATION.sub('', text))


def _field_value(key, value, relaxed):
    if relaxed and value is None and key in _PARSER_DEFAULTS:
        return _PARSER_DEFAULTS[key]
    return value


def _diff(left, right, relaxed=False):
    if left == right:
        return None
    length = max(len(left), len(right))
    for index in range(length):
        a = left[index] if index < len(left) else '<missing>'
        b = right[index] if index < len(right) else '<missing>'
        if a == b:
            continue
        if (
            relaxed
            and isinstance(a, tuple) and isinstance(b, tuple)
            and a and b and a[0] == 'error' and b[0] == 'error'
            and _relax_error(a[1]) == _relax_error(b[1])
        ):
            continue
        if (
            relaxed
            and isinstance(a, tuple) and isinstance(b, tuple)
            and len(a) > 1 and len(b) > 1
            and a[0] == b[0]
            and isinstance(a[1], str) and isinstance(b[1], str)
            and _LINE_CITE.sub('', a[1]) == _LINE_CITE.sub('', b[1])
        ):
            continue
        if isinstance(a, tuple) and isinstance(b, tuple) and a and b and a[0] == 'invoke' and b[0] == 'invoke':
            changed = [
                key for key in sorted(set(a[1]) | set(b[1]))
                if _field_value(key, a[1].get(key), relaxed) != _field_value(key, b[1].get(key), relaxed)
            ]
            if not changed:
                continue
            detail = [
                f'{key}: config={a[1].get(key)!r} script={b[1].get(key)!r}'
                for key in changed[:4]
            ]
            return 'invoke ' + ', '.join(changed[:8]) + ' | ' + ' ; '.join(detail)
        return f'event {index} config={_short(a)} script={_short(b)}'
    return None


CASES = {
    'setp-listcomp': r'''
\setp x [i for i in range(3)]
\print {{ x }}
''',
    'setp-set-literal': r'''
\setp x {1, 2}
\print {{ x }}
''',
    'setp-lambda': r'''
\setp x lambda: 1
\print {{ x() }}
''',
    'setp-fstring': r'''
\setp x f'{1 + 1}'
\print {{ x }}
''',
    'setp-walrus': r'''
\setp x (y := 3)
\print {{ x }}
''',
    'setp-bytes': r'''
\setp x b'hi'
\print {{ x }}
''',
    'setp-adjacent-strings': r'''
\setp x 'a' 'b'
\print {{ x }}
''',
    'setp-leading-zero': r'''
\setp x 01
\print {{ x }}
''',
    'setp-dunder-import': r'''
\setp x __import__('math').pi
\print {{ x }}
''',
    'setp-last-images': r'''
\setp x last_images
\print {{ x }}
''',
    'setp-last-model': r'''
\setp x last_model_path
\print {{ x }}
''',
    'setp-divide-zero': r'''
\setp x 1 / 0
\print survived
''',
    'setp-modulo': r'''
\setp x 10 % 3
\print {{ x }}
''',
    'setp-xor': r'''
\setp x 5 ^ 1
\print {{ x }}
''',
    'setp-not': r'''
\setp x not 0
\print {{ x }}
''',
    'setp-chained': r'''
\setp x 1 < 2 < 3
\print {{ x }}
''',
    'setp-slice': r'''
\setp x (1, 2, 3)[1:]
\print {{ x }}
''',
    'setp-none': r'''
\setp x None is None
\print {{ x }}
''',
    'setp-append': r'''
\setp box [1]
\setp ignored box.append(2)
\print {{ box }}
''',
    'set-reserved-len': r'''
\set len 5
\print {{ len('ab') }}
''',
    'set-reserved-quote': r'''
\set quote no
\print {{ quote('ab') }}
''',
    'set-reserved-glob': r'''
\set glob 1
\print {{ glob }}
''',
    'set-reserved-env': r'''
\set env 1
\print {{ env.PATH }}
''',
    'set-bad-name': r'''
\set 1bad value
\print survived
''',
    'set-only': r'''
\set
\print survived
''',
    'setp-only': r'''
\setp
\print survived
''',
    'sete-only': r'''
\sete
\print survived
''',
    'unset-only': r'''
\unset
\print survived
''',
    'unset-missing': r'''
\unset missing
\print survived
''',
    'unset-twice': r'''
\set x 1
\unset x
\unset x
\print survived
''',
    'print-only': r'''
\print
''',
    'echo-only': r'''
\echo
''',
    'env-stuck': r'''
\envFOO
\print survived
''',
    'print-stuck': r'''
\printf
\print survived
''',
    'unset-env-only': r'''
\unset_env
\print survived
''',
    'unset-env-quoted': r'''
\env ADV_A=1
\env ADV_B=2
\unset_env "ADV_A ADV_B"
\print {{ env.ADV_A }}|{{ env.ADV_B }}
''',
    'sete-quotes': r'''
\sete files "a b" c
\print {{ files }}
''',
    'set-quotes-stay': r'''
\set name "quoted"
\print {{ name }}
''',
    'set-percent': r'''
\set name %NOT_A_REAL_VAR_ZZZ%tail
\print [{{ name }}]
''',
    'set-dollar': r'''
\set name $NOT_A_REAL_VAR_ZZZ
\print [{{ name }}]
''',
    'echo-percent': r'''
\echo [%NOT_A_REAL_VAR_ZZZ%]
''',
    'echo-dollar': r'''
\echo [$NOT_A_REAL_VAR_ZZZ]
''',
    'echo-home': r'''
\echo [~]
''',
    'echo-unterminated': r'''
\echo "unterminated
\print survived
''',
    'echo-hash-quoted': r'''
\echo "hello # world"
''',
    'echo-escaped-hash': r'''
\echo hello\ \#\ world
''',
    'env-equals': r'''
\env ADV_KEY=a=b=c
\print [{{ env.ADV_KEY }}]
''',
    'env-no-value': r'''
\env ADV_BARE
\print [{{ env.ADV_BARE }}]
''',
    'env-empty-name': r'''
\env =novalue
\print survived
''',
    'env-illegal': r'''
\env ADV-BAD=1
\print survived
''',
    'delayed-set': r'''
{% for n in ['a', 'b'] %}
\set token {{ '{{ n }}' }}
\print {{ token }}
{% endfor %}
''',
    'jinja-set': r'''
{% set x = 5 %}
\print {{ x }}
''',
    'jinja-undefined': r'''
\print [{{ no_such_name }}]
''',
    'jinja-strict-attr': r'''
\print [{{ env.no_such_attr_zz }}]
''',
    'jinja-raw': r'''
{% raw %}
\print {{ not_expanded }}
{% endraw %}
''',
    'jinja-comment': r'''
{# hidden #}
\print visible
''',
    'jinja-for-unpack': r'''
{% for a, b in [('x', 1), ('y', 2)] %}
\print {{ a }}={{ b }}
{% endfor %}
''',
    'jinja-elif': r'''
\set n 2
{% if n == 1 %}
\print one
{% elif n == 2 %}
\print two
{% else %}
\print other
{% endif %}
''',
    'jinja-filter': r'''
\setp items ['a', 'b', 'c']
\print {{ items | join(', ') }}
''',
    'jinja-concat': r'''
\print {{ 'a' ~ 'b' }}
''',
    'false-block': r'''
{% if false %}
\print hidden
{% endif %}
\print after
''',
    'gen-seeds': r'''
\gen_seeds seeds 4
\print {{ seeds }}
''',
    'gen-seeds-bad': r'''
\gen_seeds onlyone
\print survived
''',
    'version-major': '''#! dgenerate 1.0.0\n\\print ok\n''',
    'version-minor': '''#! dgenerate 6.9.0\n\\print ok\n''',
    'reset-lineno': '''\\set\n\\reset_lineno\n\\set\n''',
    'comment-in-continuation': r'''
\setp x \
    # ignored
    2 + 2
\print {{ x }}
''',
    'tail-comment-keeps-backslash': r'''
\set greeting \
    # tail
    hello
\print [{{ greeting }}]
''',
    'hash-in-value': r'''
\set greeting hello # world
\print [{{ greeting }}]
''',
    'escaped-hash-value': r'''
\set greeting hello \# world
\print [{{ greeting }}]
''',
    'crlf': '\r\n'.join([r'\set x hello', r'\print [{{ x }}]', '']),
    'blank-and-comment': r'''

# just a comment
\print still-here
''',
    'unknown-directive': r'''
\not_a_directive a b
\print survived
''',
    'exit-code': r'''
\exit 3
\print survived
''',
    'pwd': r'''
\pwd
''',
    'exec-print': r'''
\exec print('from-exec')
''',
    'import-math': r'''
\import math
\print {{ math.pi }}
''',
    'import-as': r'''
\import math as m
\print {{ m.pi }}
''',
    'bad-invocation': r'''
--not-a-real-option
\print survived
''',
    'static-invocation': r'''
stabilityai/sd-turbo --prompts "a cat" --seeds 2 --inference-steps 1 --output-size 512
''',
    'static-two-prompts': r'''
stabilityai/sd-turbo --prompts "a cat" "a dog" --seeds 1
''',
    'omitted-seeds': r'''
stabilityai/sd-turbo --prompts "a cat"
''',
    'tilde-invocation': r'''
~/missing-model --prompts "a cat" --seeds 1
''',
    'glob-echo': r'''
\echo *.txt
''',
    'glob-sete': r'''
\sete files *.txt
\print {{ files }}
''',
    'template-glob': r'''
\print {{ glob.glob('*.txt') | join(' ') }}
''',
    'set-then-unset-print': r'''
\set color blue
\print {{ color }}
\unset color
\print [{{ color }}]
''',
    'setp-template-func': r'''
\setp x quote('a b')
\print {{ x }}
''',
    'setp-gen-seeds': r'''
\setp seeds gen_seeds(3)
\print {{ seeds }}
''',
    'nested-quotes': r'''
\echo "it's \""
''',
    'env-in-setp-name': r'''
\env ADV_NAME=color
\set $ADV_NAME blue
\print {{ color }}
''',
    'continuation-dash': r'''
stabilityai/sd-turbo \
    --prompts "a cat" \
    --seeds
    - 1
    - 2
''',
    'post-processor-comments': r'''
stabilityai/sd-turbo --prompts "a cat" --seeds 1 --post-processors \
    # comment
    resize; size=512
''',
    'print-expr-space': r'''
\setp x 3
\print {{ x }} {{ x }}
''',
    'sete-glob-zero': r'''
\sete files *.does-not-exist
\print survived
''',
    'unicode-name': r'''
\set café 1
\print survived
''',
    'setp-statement': r'''
\setp x 1; y = 2
\print survived
''',
    'if-inline-jinja': r'''
\setp n 1
\print {{ 'yes' if n == 1 else 'no' }}
''',
    'for-inline-in-print': r'''
\print {% for i in [1, 2] %}{{ i }}{% endfor %}
''',
    'reset-then-bad': '\\print a\n\\reset_lineno\n\\set 1bad v\n',
    'after-run-last': '''stabilityai/sd-turbo --prompts "a cat" --seeds 1
\\print {{ last_model_path }}
\\print {{ last_seeds }}
\\print {{ last_images }}
''',
    'setp-underscore': '\\setp x 1_000\n\\print {{ x }}\n',
    'setp-dict': "\\setp x {'a': 1}['a']\n\\print {{ x }}\n",
    'setp-dict-comp': '\\setp x {n: n for n in range(2)}\n\\print {{ x }}\n',
    'setp-genexp': '\\setp x list(i for i in range(2))\n\\print {{ x }}\n',
    'env-expand-setp': '\\env ADV_N=4\n\\setp x $ADV_N + 1\n\\print {{ x }}\n',
    'import-then-attr': '\\import math as m\n\\setp x m.pi\n\\print {{ x }}\n',
    'jinja-upper': "\\print {{ 'ab'|upper }}\n",
    'jinja-default': "\\print {{ missing_zz | default('z') }}\n",
    'jinja-none-concat': "\\print {{ 'a' ~ none }}\n",
    'sete-tilde': '\\sete home ~\n\\print {{ home[0] }}\n',
    'echo-empty': '\\echo ""\n',
    'unset-builtin': '\\unset len\n\\print survived\n',
    'version-tail-comment': '#! dgenerate 1.0.0 # note\n\\print ok\n',
    'version-exact': '#! dgenerate 6.0.0\n\\print ok\n',
    'raw-broken-after': '\\print before\n{% raw %}\n\\print inside\n',
    'unclosed-expr': '\\print {{ 1 +\n',
    'setp-getcwd': '\\setp x os.getcwd()\n\\print {{ x }}\n',
    'setp-glob': "\\setp x sorted(glob.glob('*.txt'))\n\\print {{ x }}\n",
    'jinja-if-expr': "\\print {{ 'yes' if 0 else 'no' }}\n",
    'duplicate-seeds': 'stabilityai/sd-turbo --prompts "a cat" --seeds 1 --seeds 2\n\\print survived\n',
    'pushd-popd': '\\pushd .\n\\pwd\n\\popd\n\\pwd\n',
    'filter-block': '{% filter upper %}hi{% endfilter %}\n\\print after\n',
    'with-block': '{% with x = 1 %}{{ x }}{% endwith %}\n',
    'set-then-setp': '\\set a 2\n\\setp b a + 3\n\\print {{ b }}\n',
    'method-call': '\\setp x (2).bit_count()\n\\print {{ x }}\n',
    'yield-expr': '\\setp x (yield 1)\n\\print survived\n',
    'slash-names': '\\print {{ 7 // 2 }}\n\\setp x 7 // 2\n\\print {{ x }}\n',
    'upscaler-help': '\\prompt_upscaler_help dynamicprompts\n',
    'templates-help-name': '\\templates_help glob\n',
    'quoted-tilde': '\\echo "~"\n\\echo \'~\'\n',
    'prompt-mentions-help': 'stabilityai/sd-turbo --prompts "see --help now" --seeds 1\n',
    'setp-last-after-run': 'stabilityai/sd-turbo --prompts "a cat" --seeds 1\n\\setp seen last_images\n\\print {{ seen }}\n',
    'env-through-jinja': '\\env ADV_WORD=hello\n\\print [{{ env.ADV_WORD }}]\n',
    'jinja-list-filter': '\\print {{ range(3) | list }}\n',
    'setp-float': '\\setp x 1/2\n\\print {{ x }}\n',
    'setp-sci': '\\setp x 1e2\n\\print {{ x }}\n',
    'nested-if-set': '{% if 1 %}\n{% if 0 %}\n\\print no\n{% else %}\n\\set v inside\n{% endif %}\n{% endif %}\n\\print {{ v }}\n',
    'line-number-blanks': '\n\n\\set 1bad name\n',
    'help-text-value': '\\set text --help\n\\print {{ text }}\n',
    'invocation-tail-comment': 'stabilityai/sd-turbo --prompts "a cat" --seeds 1 # tail\n',
    'is-defined': '''{% if token is defined %}
\\print yes
{% else %}
\\print no
{% endif %}
\\set token hi
{% if token is defined %}
\\print yes
{% else %}
\\print no
{% endif %}
{% if token is undefined %}
\\print missing
{% else %}
\\print present
{% endif %}
''',
    'loop-index': '''{% for i in [1, 2] %}
\\print {{ loop.index }}-{{ loop.last }}
{% endfor %}
{% for i in [] %}
\\print gone
{% else %}
\\print empty
{% endfor %}
''',
    'jinja-filters': '''{% if items | length %}
\\print yes
{% else %}
\\print no
{% endif %}
{% if missing | default(false) %}
\\print yes
{% else %}
\\print no
{% endif %}
\\print [{{ ' ab '|trim }}]
''',
    'setp-env-call': '''\\env ADVERSARIAL_SIZE=512x768
\\setp sized scale_size("%ADVERSARIAL_SIZE%", 2)
\\print {{ sized }}
''',
    'double-space-print': '\\print hello  world\n\\echo "a b"  c\n',
    'line-inside-for': '{% for i in [1] %}\n\\set 1bad name\n{% endfor %}\n',
    'templated-invocation': '''\\set model stabilityai/sd-turbo
\\set prompt a cat
\\set output_size 512x512
\\set upscaler_model 4x-UltraSharp
{{ model }} --prompts {{ quote(prompt) }} --inference-steps 1 --seeds 1 --output-size {{ output_size }} --seed-image-processors upscaler;model={{ upscaler_model }};scale=1
''',
    'quoted-adapter-seed': '''\\setp images ['a.png', 'b.png']
\\setp images ' + '.join(images)
stabilityai/sd-turbo --prompts hi --seeds 1 --image-seeds "adapter: {{ images }}, face.png"
''',
}


def _mutate(source: str, kind: str) -> str:
    if kind == 'indent':
        return '\n'.join(('    ' + line) if line.strip() else line for line in source.splitlines())
    if kind == 'trailing-space':
        return '\n'.join(line + '   ' for line in source.splitlines())
    if kind == 'blank-between':
        return '\n\n'.join(source.splitlines())
    return source


MUTATIONS = {
    'setp-modulo': 'trailing-space',
    'jinja-elif': 'blank-between',
    'hash-in-value': 'trailing-space',
    'static-invocation': 'indent',
    'comment-in-continuation': 'trailing-space',
}


def _prepare_dir(path: Path):
    for name in ('a.txt', 'b.txt', 'c.txt'):
        (path / name).write_text(name, encoding='utf-8')


def _run_side(source: str, work: Path, which: str, **transpile_options):
    trace = _Trace()
    previous = os.getcwd()
    saved_env = dict(os.environ)
    original_log = messages.log
    original_invoke = invoker.invoke_dgenerate
    original_run = renderloop.RenderLoop.run
    original_parse = dgenerate.arguments.parse_args
    script = (
        transpile_config.transpile(source, **transpile_options)
        if which == 'script' else None
    )
    exc = None
    messages.log = trace.log
    dgenerate.arguments.parse_args = lambda args, **kwargs: _parse_without_check(args)

    def fake_invoke(args, render_loop=None, throw=False, **kwargs):
        if any(str(arg).startswith('-') and 'help' in str(arg) for arg in args):
            return original_invoke(args, render_loop=render_loop, throw=throw, **kwargs)
        try:
            parsed = _parse_without_check(list(args))
        except Exception as error:
            messages.error(f'dgenerate: error: {error}')
            if throw:
                raise
            return 1
        if render_loop is not None:
            render_loop.config = parsed
        trace.invoke(parsed)
        if render_loop is not None:
            _plant(render_loop)
        return 0

    def fake_run(self):
        trace.invoke(self.config)
        _plant(self)

    try:
        os.chdir(work)
        if which == 'config':
            invoker.invoke_dgenerate = fake_invoke
            runner = configrunner.ConfigRunner(throw=False, version=dgenerate.__version__)
            runner.run_string(source)
        else:
            renderloop.RenderLoop.run = fake_run
            sandbox = {'print': lambda *args, **kwargs: trace.log(*args)}
            exec(compile(script, '<adversarial>', 'exec'), sandbox, sandbox)
    except SystemExit as error:
        trace.events.append(('exit', error.code))
    except Exception as error:
        exc = error
    finally:
        os.chdir(previous)
        os.environ.clear()
        os.environ.update(saved_env)
        messages.log = original_log
        invoker.invoke_dgenerate = original_invoke
        renderloop.RenderLoop.run = original_run
        dgenerate.arguments.parse_args = original_parse
    return _events_of(trace, exc)


def iter_cases():
    """Adversarial configs, including the whitespace mutations."""
    cases = dict(CASES)
    for name, kind in MUTATIONS.items():
        cases[f'{name}__{kind}'] = _mutate(CASES[name], kind)
    return cases.items()


def _config_failed(events) -> bool:
    for event in events:
        if event and (event[0] == 'error' or event[0] == messages.ERROR):
            return True
    return False


def _restore_plugins(saved):
    plugin.PLUGIN_PATHS.clear()
    plugin.PLUGIN_PATHS.update(saved)


def compare(source: str, work: Path, **transpile_options):
    """Return a difference summary, or None when the script matches the config.

    A config the interpreter rejects is not a script. Translation stops with
    ParseError, and that agrees with the interpreter when the config fails too.
    ``transpile_options`` are passed to ``transpile`` for the script side.
    """
    saved_plugins = set(plugin.PLUGIN_PATHS)
    try:
        try:
            config_events = _run_side(source, work, 'config')
            try:
                script_events = _run_side(source, work, 'script', **transpile_options)
            except transpile_config.ParseError as error:
                if _config_failed(config_events):
                    return None
                return f'transpile rejected a config that runs: {error}'
            return _diff(
                config_events, script_events,
                relaxed=transpile_options.get('relaxed', False),
            )
        except Exception:
            return traceback.format_exc().splitlines()[-1]
    finally:
        _restore_plugins(saved_plugins)


def main(echo=True):
    _write_png(IMAGE)
    renderloopconfig.gen_seeds = _deterministic_seeds
    renderloop.gen_seeds = _deterministic_seeds
    work = Path(tempfile.mkdtemp(prefix='dgen-adversarial-'))
    _prepare_dir(work)
    cases = list(iter_cases())
    broken = []
    try:
        for index, (name, source) in enumerate(cases, start=1):
            difference = compare(source, work)
            if difference:
                broken.append((name, difference))
                if echo:
                    print(f'BREAK {name}: {difference}')
            if echo and index % 20 == 0:
                print(f'[{index}/{len(cases)}] broken {len(broken)}')
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if echo:
        print(f'cases {len(cases)} broken {len(broken)}')
    return broken


_STUBBED = {
    'image_process', 'rm', 'rmdir', 'mv', 'cp', 'exec', 'download',
    'prompt_upscale',
}


def run_examples(echo=True, **transpile_options):
    import dgenerate.batchprocess.configrunnerbuiltins as builtins_module

    _write_png(IMAGE)
    renderloopconfig.gen_seeds = _deterministic_seeds
    renderloop.gen_seeds = _deterministic_seeds
    original_download = builtins_module.download
    original_init = configrunner.ConfigRunner.__init__
    builtins_module.download = lambda *_args, **_kwargs: str(IMAGE)

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for name in list(self.directives):
            if name in _STUBBED:
                def run(argv, directive_name=name):
                    return 0
                self.directives[name] = run

    configrunner.ConfigRunner.__init__ = init
    broken = []
    files = sorted(Path('examples').rglob('*.dgen'))
    try:
        for index, path in enumerate(files, start=1):
            source = path.read_text(encoding='utf-8')
            try:
                config_events = _run_side(source, path.parent, 'config')
                script_events = _run_side(
                    source, path.parent, 'script', **transpile_options)
                difference = _diff(
                    config_events, script_events,
                    relaxed=transpile_options.get('relaxed', False),
                )
            except Exception:
                difference = traceback.format_exc().splitlines()[-1]
            if difference:
                broken.append((str(path), difference))
                if echo:
                    print(f'BREAK {path}: {difference}')
            if echo and index % 50 == 0:
                print(f'[{index}/{len(files)}] broken {len(broken)}')
    finally:
        configrunner.ConfigRunner.__init__ = original_init
        builtins_module.download = original_download
    if echo:
        print(f'examples {len(files)} broken {len(broken)}')
    return broken


if __name__ == '__main__':
    broken = run_examples() if '--examples' in sys.argv else main()
    sys.exit(1 if broken else 0)
