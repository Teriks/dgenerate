import json
import os
import re
import subprocess
import sys
import tempfile
import time

import dgenerate
import dgenerate.assistant.corpus as _corpus
import dgenerate.assistant.index as _index
import dgenerate.batchprocess.util as _b_util
import dgenerate.assistant.catalog as _catalog
import dgenerate.assistant.models as _models
import dgenerate.assistant.prompt as _prompt

# The directory holding the dgenerate package, so the check subprocess imports the same dgenerate.
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(dgenerate.__file__)))


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def deep_check(config: str, run_dir: str) -> dict:
    """
    Run :mod:`dgenerate.assistant.check` in a subprocess so the config runner's
    global state and working directory changes stay out of this process.
    """
    fd, path = tempfile.mkstemp(suffix='.dgen')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(config)
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(p for p in (_PACKAGE_PARENT, env.get('PYTHONPATH')) if p)
        result = subprocess.run([sys.executable, '-m', 'dgenerate.assistant.check', path],
                                cwd=run_dir, env=env, capture_output=True, text=True, timeout=600)
        lines = result.stdout.strip().splitlines()
        if result.returncode != 0 or not lines:
            return {'ok': None, 'errors': [], 'warnings': [],
                    'failure': (result.stderr.strip() or 'no output')[-2000:]}
        return json.loads(lines[-1])
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return {'ok': None, 'errors': [], 'warnings': [], 'failure': str(e)}
    finally:
        os.unlink(path)


def _problem(entry: dict) -> str:
    line = entry.get('line')
    return f'line {line}: {entry["message"]}' if line else entry['message']


def _guessed_files(warnings: list[dict], files: list[_prompt.RequestFile]) -> list[str]:
    """
    Missing files that are neither request inputs nor placeholders, usually a
    guess at the name of something an earlier invocation writes.
    """
    known = {p for f in files for p in (f.written, f.config_path)}
    problems = []
    for entry in warnings:
        match = re.search(r'"([^"]+)" does not exist', entry['message'])
        if not match:
            continue
        path = match.group(1).split('latents:', 1)[-1].strip()
        if path in known or path.replace('\\', '/').startswith('path/to/'):
            continue
        if path.lower().endswith(('.safetensors', '.gguf', '.ckpt', '.bin', '.pt', '.pth')):
            continue
        known.add(match.group(1))
        problems.append(_problem(entry) + ' If an earlier invocation writes it, use {{ quote(last_images) }} '
                                          'or {{ quote(last_animations) }} instead of its file name. If the '
                                          'user must supply it, use a path/to/ placeholder.')
    return problems


# "generate an image and then expand it": the request asks for the input to be generated first.
_GENERATE_THEN = re.compile(r'\b(generate|create|make|render)\b.*\b(then|afterwards|after that)\b',
                            re.IGNORECASE | re.DOTALL)


def _chainable_placeholders(request: str, config: str, warnings: list[dict], raised: set[str]) -> list[str]:
    """
    Placeholders that stand for an image generated earlier in the config, or that the request
    asks to generate first, with the edit that chains them. Each is raised once, since it may
    be a file the user supplies.
    """
    problems = []
    for entry in warnings:
        path = entry.get('placeholder')
        if not path or path in raised:
            continue
        after = entry.get('after_line')
        if after is None:
            if not _GENERATE_THEN.search(request):
                continue
            raised.add(path)
            problems.append(_problem(entry) + ' The request asks to generate this image first, but no '
                                              'invocation before this line generates it. Add an invocation '
                                              'that generates it from a prompt at the start of the config, '
                                              'then use {{ quote(first(last_images)) }} for it instead of '
                                              f'"{path}", for example "\\set input_image '
                                              '{{ quote(first(last_images)) }}" directly after that invocation.')
            continue
        raised.add(path)
        where = f'the invocation that ends on line {after}'
        setter = re.search(rf'^[ \t]*\\set[ \t]+(\w+)[ \t]+["\']?{re.escape(path)}["\']?[ \t]*$', config, re.MULTILINE)
        if setter:
            fix = (f'Delete the line "{setter.group(0).strip()}" and add the line '
                   f'"\\set {setter.group(1)} {{{{ quote(first(last_images)) }}}}" directly after {where}, '
                   f'so {{{{ {setter.group(1)} }}}} is the image that invocation writes.')
        else:
            fix = f'Replace "{path}" with {{{{ quote(first(last_images)) }}}}, used directly after {where}.'
        problems.append(_problem(entry) + ' The request does not supply this file, it is the image generated '
                                          'earlier in the config. ' + fix)
    return problems


def create_parser(prog: str) -> _b_util.DirectiveArgumentParser:
    parser = _b_util.DirectiveArgumentParser(
        prog=prog,
        description='Write a dgenerate config from a plain language request, using a local '
                    'Qwen model with retrieval over the dgenerate examples and documentation.')
    parser.add_argument('request', nargs='*',
                        help='What the config should do. Read from stdin when omitted.')
    parser.add_argument('-o', '--output', help='Write the config to this file instead of stdout. '
                                               'File paths in the request are relative to the current '
                                               'directory and are rewritten relative to this file.')
    parser.add_argument('--model', default=_models.DEFAULT_CHAT_MODEL,
                        help='Chat model, a .gguf path or org/repo/file.gguf. Default: %(default)s')
    parser.add_argument('--embed-model', default=_models.DEFAULT_EMBED_MODEL,
                        help='Embedding model, a .gguf path or org/repo/file.gguf. '
                             'The models in the Generate Config menu have a packaged index. '
                             'Any other Qwen3-Embedding model builds an index on first use. '
                             'Default: %(default)s')
    parser.add_argument('--ctx', type=int, default=32768, help='Chat context size in tokens. Default: %(default)s')
    parser.add_argument('--gpu-layers', type=int, default=-1,
                        help='Layers to put on the GPU, -1 for automatic, 0 for CPU only. Default: %(default)s')
    parser.add_argument('--think', action='store_true', help='Let the model reason before answering. Slower.')
    parser.add_argument('--reasoning-effort', choices=_catalog.REASONING_EFFORTS,
                        default=_catalog.DEFAULT_REASONING_EFFORT,
                        help='How long to think when --think is set. Qwen3.8 otherwise uses extra high. '
                             'Default: %(default)s')
    parser.add_argument('--temperature', type=float, default=0.3, help='Default: %(default)s')
    parser.add_argument('--max-tokens', type=int, default=0,
                        help='Maximum reply tokens. 0 uses the rest of the context window and does not '
                             'increase memory use. Default: %(default)s')
    parser.add_argument('--no-check', action='store_true', help='Skip validating the config with dgenerate.')
    parser.add_argument('--max-repairs', type=int, default=2,
                        help='How many times to ask the model to fix a config dgenerate rejects. Default: %(default)s')
    parser.add_argument('--show-context', action='store_true', help='Print the retrieved context to stderr.')
    parser.add_argument('--offline', action='store_true', help='Only use models already in the Hugging Face cache.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show llama.cpp output.')
    return parser


def main(argv: list[str], prog: str = 'assistant', local_files_only: bool = False) -> int:
    parser = create_parser(prog)
    args = parser.parse_args(argv)
    if parser.return_code is not None:
        return parser.return_code
    args.offline = args.offline or local_files_only
    if not _catalog.xllamacpp_installed():
        log('error: xllamacpp is not installed. Install it with: pip install dgenerate[xllamacpp]')
        return 1
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

    version = dgenerate.__version__

    request = ' '.join(args.request).strip()
    if not request:
        if sys.stdin.isatty():
            log('Describe the config you want, then press Ctrl+Z and Enter (Ctrl+D on Linux or macOS):')
        request = sys.stdin.read().strip()
        if not request:
            log('error: no request given.')
            return 2

    output = os.path.abspath(args.output) if args.output else None
    run_dir = os.path.dirname(output) if output else os.getcwd()
    os.makedirs(run_dir, exist_ok=True)

    try:
        log('Loading the embedding model')
        embedder = _models.Embedder(_models.resolve_gguf(args.embed_model, args.offline),
                                    gpu_layers=args.gpu_layers, verbose=args.verbose)
        index = _index.load_index(embedder, log=log)
        files = _prompt.request_files(request, os.getcwd(), run_dir)
        query = _corpus.strip_media_names(request)
        hits = index.search(query, embedder.embed_query(query), has_inputs=_corpus.wants_inputs(request))
        del embedder

        for f in files:
            if not f.exists:
                log(f'warning: "{f.written}" does not exist relative to {os.getcwd()}')
        context = _prompt.build_context(request, hits, index)
        if args.show_context:
            log(context)
        log(f'Retrieved {len(context)} characters of context')

        log(f'Loading {os.path.basename(args.model)}')
        chat = _models.ChatModel(_models.resolve_gguf(args.model, args.offline), n_ctx=args.ctx,
                                 gpu_layers=args.gpu_layers, think=args.think, effort=args.reasoning_effort,
                                 verbose=args.verbose)

        messages = [
            {'role': 'system', 'content': _prompt.system_message(version)},
            {'role': 'user', 'content': _prompt.user_message(request, context, files)},
        ]

        config, report = None, None
        raised_placeholders = set()
        for attempt in range(args.max_repairs + 1):
            limit = _models.reply_token_limit(args.ctx, args.max_tokens)
            log('Writing the config' if attempt == 0 else f'Fixing the config (attempt {attempt})')
            log('  up to the rest of the context window' if limit < 0 else f'  up to {limit} tokens')
            started = time.time()
            reply = chat.complete(messages, max_tokens=limit, temperature=args.temperature)
            log(f'  done in {time.time() - started:.1f}s')
            messages.append({'role': 'assistant', 'content': reply})
            config = _prompt.apply_request_files(_prompt.extract_config(reply, version), files)

            if args.no_check:
                break

            problems = _prompt.inputs_used_as_output(config, files) + _prompt.split_prompts(config)
            suggested = []
            for option, matches in _prompt.unknown_options(config, index.known_options):
                hint = f', did you mean {" or ".join(matches)}?' if matches else '.'
                problems.append(f'{option} is not a dgenerate option{hint}')
                suggested += matches

            if not problems:
                log('Checking the config with dgenerate')
                report = deep_check(config, run_dir)
                if report.get('ok') is None:
                    log(f'warning: could not run the dgenerate check: {report.get("failure")}')
                    break
                problems = [_problem(e) for e in report['errors']] + _guessed_files(report['warnings'], files)
                if not files:
                    problems += _chainable_placeholders(request, config, report['warnings'], raised_placeholders)
                suggested = [o for p in problems for o in _prompt.options_in(p) if index.argument(o)]

            if not problems:
                break
            for p in problems:
                log(f'  {p}')
            if attempt == args.max_repairs:
                log('warning: the config still has problems, review it before running.')
                break
            messages.append({'role': 'user', 'content': _prompt.repair_message(problems, suggested, index)})

        if report:
            for warning in report.get('warnings', []):
                log(f'note: {_problem(warning)}')
            if report.get('ok'):
                log(f'dgenerate accepted the config ({report["invocations"]} invocation(s)).')

    except (_models.ModelError, _index.AssistantIndexError) as e:
        log(f'error: {e}')
        return 1

    if output:
        with open(output, 'w', encoding='utf-8', newline='\n') as f:
            f.write(config)
        log(f'Wrote {output}')
    else:
        sys.stdout.write(config)
    return 0
