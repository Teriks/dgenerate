# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import codecs
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
import dgenerate.assistant.link_tools as _link_tools

# The directory holding the dgenerate package, so the check subprocess imports the same dgenerate.
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(dgenerate.__file__)))


def log(*args, style='plain'):
    sys.stderr.write(_models.progress_style(style) + ' '.join(str(arg) for arg in args) + '\n')
    sys.stderr.flush()


def _control_code(value: str) -> str:
    """Turn ``\\033`` and ``\\x1b`` into the characters they name. A real ESC is kept."""
    if '\\' not in value:
        return value
    cooked = value.replace('\\e', '\\x1b').replace('\\E', '\\x1b')
    try:
        return codecs.decode(cooked.encode('ascii'), 'unicode_escape')
    except UnicodeError:
        return value


def log_model_open(title: str) -> str:
    """The title and its ``=`` rule. A blank line is left for the model to open."""
    log(title)
    rule = '=' * len(title)
    log(rule)
    return rule


def log_model_close(rule: str) -> None:
    """A blank line then ``=`` after the model's own output."""
    log('')
    log(rule)


def _context_too_small(n_ctx: int, needed: int, *, retrieved: bool) -> str:
    if retrieved:
        detail = 'This prompt is about'
        when = ' once the examples are included'
    else:
        detail = 'The instructions need about'
        when = ' before any examples are retrieved'
    return (
        f'error: the context window is {n_ctx} tokens. {detail} {needed} tokens{when}, '
        'including room for the reply. Raise --ctx.')


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


def _supported_model(value: str, specs: tuple[str, ...], kind: str) -> str:
    if value in specs:
        return value
    choices = '\n'.join(f'  {spec}' for spec in specs)
    raise argparse.ArgumentTypeError(
        f'"{value}" is not a supported {kind} model. Choose one of:\n{choices}')


def _supported_chat_model(value: str) -> str:
    return _supported_model(value, _catalog.chat_model_specs(), 'chat')


def _supported_embed_model(value: str) -> str:
    return _supported_model(value, _catalog.embed_model_specs(), 'embedding')


def _model_choices_help(intro: str, models) -> str:
    """Help text that lists every compatible model, one per line."""
    lines = [
        intro,
        '',
        'NOWRAP!',
    ]
    lines.extend(f'{spec} ({size} GB, {note})' for spec, size, note in models)
    lines.append('')
    lines.append('Default: %(default)s')
    return '\n'.join(lines)


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
    parser.add_argument('-mdl', '--model', default=_models.DEFAULT_CHAT_MODEL, type=_supported_chat_model,
                        help=_model_choices_help(
                            'Chat model. One of these compatible Qwen 3.8 27B GGUFs, smallest download first.',
                            _catalog.CHAT_MODELS))
    parser.add_argument('-emb', '--embed-model', default=_models.DEFAULT_EMBED_MODEL, type=_supported_embed_model,
                        help=_model_choices_help(
                            'Embedding model. One of these compatible Qwen3-Embedding GGUFs. '
                            'Each one has a packaged index.',
                            _catalog.EMBED_MODELS))
    parser.add_argument('-ctx', '--ctx', type=int, default=32768, help='Chat context size in tokens. Default: %(default)s')
    parser.add_argument('-gpu', '--gpu-layers', type=int, default=-1,
                        help='Layers to put on the GPU, -1 for automatic, 0 for CPU only. Default: %(default)s')
    parser.add_argument('-ed', '--edit', metavar='FILE',
                        help='Revise this existing config instead of writing a new one. '
                             'The request describes the change.')
    parser.add_argument('-thk', '--think', action='store_true', help='Let the model reason before answering. Slower.')
    parser.add_argument('-eff', '--reasoning-effort', choices=_catalog.REASONING_EFFORTS,
                        default=_catalog.DEFAULT_REASONING_EFFORT,
                        help='How long to think when --think is set. Qwen3.8 otherwise uses extra high. '
                             'Default: %(default)s')
    # None means "not passed". The value is filled from the Qwen3.8 card after --think is known.
    parser.add_argument('-temp', '--temperature', type=float, default=None,
                        help='Sampling temperature. Default: 0.7, or 1.0 with --think.')
    parser.add_argument('-tp', '--top-p', type=float, default=None,
                        help='Default: 0.8, or 0.95 with --think.')
    parser.add_argument('-tk', '--top-k', type=int, default=None,
                        help='Default: 20.')
    parser.add_argument('-mp', '--min-p', type=float, default=None,
                        help='Default: 0.0.')
    parser.add_argument('-pres', '--presence-penalty', type=float, default=None,
                        help='Default: 1.5, or 0.0 with --think.')
    parser.add_argument('-rpen', '--repeat-penalty', type=float, default=None,
                        help='Repetition penalty. Default: 1.0.')
    parser.add_argument('-mtok', '--max-tokens', type=int, default=0,
                        help='Maximum reply tokens. 0 uses the rest of the context window and does not '
                             'increase memory use. Default: %(default)s')
    parser.add_argument('-nc', '--no-check', action='store_true', help='Skip validating the config with dgenerate.')
    parser.add_argument('-mrep', '--max-repairs', type=int, default=2,
                        help='How many times to ask the model to fix a config dgenerate rejects. Default: %(default)s')
    parser.add_argument('-sctx', '--show-context', action='store_true', help='Print the retrieved context to stderr.')
    parser.add_argument('-ofm', '--offline-mode', action='store_true',
                        help='Only use models already in the Hugging Face cache. '
                             'The "fetch_link" tool lists Hugging Face files already in that cache and does not call the Hub. '
                             'A quant that is not cached is not listed, so the example file is kept. '
                             'The "fetch_link" tool does not fetch a CivitAI link or any other URL. '
                             'A CivitAI page is not in the local cache, so the config is written from the request '
                             'and the retrieved examples.')
    parser.add_argument('-nlt', '--no-link-tools', action='store_true',
                        help='Do not offer the "fetch_link" tool; the model cannot request CivitAI or Hugging Face lookups.')
    parser.add_argument('-ppl', '--progress-plain', default='', type=_control_code, metavar='CODE',
                        help='Terminal color before status text. Off by default. '
                             r'Reset with \033[0m.')
    parser.add_argument('-pth', '--progress-thought', default='', type=_control_code, metavar='CODE',
                        help=r'Terminal color before reasoning. Off by default. Green is \033[32m.')
    parser.add_argument('-ptk', '--progress-talk', default='', type=_control_code, metavar='CODE',
                        help=r"Terminal color before the model's reply. Off by default. Cyan is \033[36m.")
    parser.add_argument('-ptl', '--progress-tool', default='', type=_control_code, metavar='CODE',
                        help=r'Terminal color before a tool call. Off by default. Yellow is \033[33m.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show llama.cpp output.')
    return parser


def sampling_from_args(args) -> dict:
    """Sampling for this invocation. A flag that was omitted keeps the recommended value."""
    return _catalog.resolve_sampling(
        args.think,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repeat_penalty=args.repeat_penalty)


def main(argv: list[str], prog: str = 'assistant', local_files_only: bool = False) -> int:
    parser = create_parser(prog)
    args = parser.parse_args(argv)
    if parser.return_code is not None:
        return parser.return_code
    _models.set_progress_styles(
        args.progress_plain, args.progress_thought, args.progress_tool, args.progress_talk)
    args.offline = args.offline_mode or local_files_only
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

    editor = None
    if args.edit:
        try:
            with open(args.edit, encoding='utf-8') as handle:
                editor = handle.read()
        except OSError as e:
            log(f'error: could not read the config to edit: {e}')
            return 1
        if not editor.strip():
            log('error: the config to edit is empty.')
            return 2

    files = _prompt.request_files(request, os.getcwd(), run_dir)
    link_text = request + ('\n' + editor if editor else '')
    request_links = _link_tools.find_link_urls(link_text)
    # Tools replace streaming. The config check is always offered. fetch_link stays
    # available offline and reads the Hugging Face cache.
    use_link_tools = not args.no_link_tools
    use_check_tool = not args.no_check
    tools = _link_tools.tool_definitions(fetch=use_link_tools, check=use_check_tool)
    machine = _prompt.machine_report()
    system = _prompt.system_message(
        version, link_tools=use_link_tools, machine=machine, check_tool=use_check_tool)
    # The instructions are the smallest prompt this command sends. Retrieved
    # examples are added after the embedding model, and checked again there.
    reserve = args.max_tokens if args.max_tokens > 0 else _models.REPLY_RESERVE
    tool_text = json.dumps(tools) if tools else ''
    needed = _models.context_tokens_needed(
        system, _prompt.user_message(request, '', files, editor=editor), tool_text, reserve=reserve)
    if needed > args.ctx:
        log(_context_too_small(args.ctx, needed, retrieved=False))
        return 1

    try:
        log('Loading the embedding model')
        embedder = _models.Embedder(_models.resolve_gguf(args.embed_model, args.offline),
                                    gpu_layers=args.gpu_layers, verbose=args.verbose)
        index = _index.load_index(embedder, log=log)
        query = _corpus.strip_media_names(request)
        if editor:
            mentioned = ' '.join(re.findall(r'--model-type\s+(\S+)', editor))
            if mentioned:
                query = f'{query} {mentioned}'.strip()
        hits = index.search(query, embedder.embed_query(query), has_inputs=_corpus.wants_inputs(request))
        del embedder

        for f in files:
            if not f.exists:
                log(f'warning: "{f.written}" does not exist relative to {os.getcwd()}')
        context = _prompt.build_context(request, hits, index)
        if args.show_context:
            log(context)
        log(f'Retrieved {len(context)} characters of context')

        if use_link_tools and request_links:
            log(f'Link tool enabled ({len(request_links)} URL(s) in the request). The model can call fetch_link.')

        user = _prompt.user_message(request, context, files, editor=editor)
        needed = _models.context_tokens_needed(system, user, tool_text, reserve=reserve)
        if needed > args.ctx:
            log(_context_too_small(args.ctx, needed, retrieved=True))
            return 1

        log(f'Loading {os.path.basename(args.model)}')
        chat = _models.ChatModel(_models.resolve_gguf(args.model, args.offline), n_ctx=args.ctx,
                                 gpu_layers=args.gpu_layers, think=args.think, effort=args.reasoning_effort,
                                 verbose=args.verbose)

        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]

        def log_tool(line: str) -> None:
            text = line if line.startswith('  ->') else f'Tool {line}'
            log(text, style='tool')

        config, report = None, None
        raised_placeholders = set()
        baseline = set()
        if editor:
            base_report = deep_check(editor, run_dir)
            baseline = {entry['message'] for entry in base_report.get('errors', [])}

        def keep_new(problems: list[str]) -> list[str]:
            if not baseline:
                return problems
            kept = []
            for problem in problems:
                text = re.sub(r'^line \d+:\s*', '', problem)
                if text not in baseline:
                    kept.append(problem)
            if problems and not kept:
                log('Ignoring check errors that were already in the editor.')
            return kept

        sampling = sampling_from_args(args)
        elapsed = None
        closed = False
        accepted = False
        for attempt in range(args.max_repairs + 1):
            limit = _models.reply_token_limit(args.ctx, args.max_tokens)
            action = 'Writing the config' if attempt == 0 else f'Fixing the config (attempt {attempt})'
            room = 'up to the rest of the context window' if limit < 0 else f'up to {limit} tokens'
            rule = log_model_open(f'{action}: {room}')
            started = time.time()
            try:
                reply = chat.complete(messages, max_tokens=limit, tools=tools or None,
                                      on_tool=log_tool if tools else None, **sampling)
            finally:
                elapsed = time.time() - started
                if getattr(chat, 'showed', False):
                    log_model_close(rule)
            messages.append({'role': 'assistant', 'content': reply})
            draft, used_reply = _prompt.take_draft(config, reply, version)
            if not used_reply:
                log('warning: the repair reply was not a config, so the previous draft was kept.')
            config = _prompt.apply_request_files(draft, files)

            if args.no_check:
                break

            problems = keep_new(_prompt.inputs_used_as_output(config, files) + _prompt.split_prompts(config))
            suggested = []
            for option, matches in _prompt.unknown_options(config, index.known_options):
                hint = f', did you mean {" or ".join(matches)}?' if matches else '.'
                problems.append(f'{option} is not a dgenerate option{hint}')
                suggested += matches

            if not problems and getattr(chat, 'config_was_submitted', False):
                report = {
                    'ok': True,
                    'errors': [],
                    'warnings': [],
                    'invocations': chat.submitted_invocations,
                }
            elif not problems:
                log('Checking the config with dgenerate')
                report = deep_check(config, run_dir)
                if report.get('ok') is None:
                    log(f'Done in {elapsed:.1f}s')
                    closed = True
                    log(f'warning: could not run the dgenerate check: {report.get("failure")}')
                    break
                problems = keep_new(
                    [_problem(e) for e in report['errors']] + _guessed_files(report['warnings'], files))
                if not files:
                    problems += _chainable_placeholders(request, config, report['warnings'], raised_placeholders)
                suggested = [o for p in problems for o in _prompt.options_in(p) if index.argument(o)]

            if not problems:
                accepted = True
                break
            log(f'Done in {elapsed:.1f}s')
            closed = True
            for p in problems:
                log(f'  {p}')
            if attempt == args.max_repairs:
                log('warning: the config still has problems, review it before running.')
                break
            messages.append({'role': 'user', 'content': _prompt.repair_message(
                problems, suggested, index, config or '')})

        if report:
            for warning in report.get('warnings', []):
                log(f'note: {_problem(warning)}')
        if accepted and report and report.get('ok'):
            log(f'Done in {elapsed:.1f}s: dgenerate accepted the config '
                f'({report["invocations"]} invocation(s)).')
        elif elapsed is not None and not closed:
            log(f'Done in {elapsed:.1f}s')

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
