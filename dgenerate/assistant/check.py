"""
Validate a dgenerate config without loading any models.

The config runs through dgenerate's own batch processor, so templates,
directives, and line continuations are handled exactly as dgenerate handles
them. Invocations are captured instead of executed and then validated with
dgenerate's argument parser. Directives with side effects are replaced by
no-ops.

Run as ``python -m dgenerate.assistant.check config.dgen`` in the directory the
config will run from. Prints a JSON report on stdout.
"""

import contextlib
import io
import json
import os
import re
import sys

import collections.abc

from dgenerate.assistant.corpus import media_kind

# Directives that load models, touch the file system, run programs, or end the process.
_STUBBED_DIRECTIVES = (
    'image_process', 'prompt_upscale', 'exec', 'mv', 'cp', 'mkdir', 'rmdir', 'rm',
    'save_modules', 'use_modules', 'clear_modules', 'clear_object_cache', 'clear_device_cache',
    'import_plugins', 'exit',
)

_ASSIGN_DIRECTIVE = re.compile(
    r'^[ \t]*\\(set[ep]?|gen_seeds|download)[ \t]+(\w+)', re.MULTILINE)
_FOR_VAR = re.compile(r'\{%[ \t]+for[ \t]+(\w+)[ \t]+in[ \t]+')
_JINJA_EXPR = re.compile(r'\{\{(.*?)\}\}', re.DOTALL)
# Official in-block form: {{ '{{ name }}' }} or {{ '{{ quote(name) }}' }} renders
# to {{ name }} after the continuation, so \set in the block is visible.
_DELAYED_ESCAPE = re.compile(
    r"""\{\{[ \t]*(['\"])\{\{.*?\}\}[ \t]*\1[ \t]*\}\}""", re.DOTALL)
_LAST_STAR = re.compile(r'\b(last_images|last_animations)\b')
# {% if token.strip() %} wrapping work, not {% if not token.strip() %} \exit.
_POSITIVE_TOKEN_IF = re.compile(
    r'\{%[ \t]*if[ \t]+(?!not\b).*\b(civit_ai_token|civitai_token|hf_token|\btoken\b)',
    re.IGNORECASE | re.DOTALL)
_WORK_IN_BLOCK = re.compile(
    r'^[ \t]*(\\(set[ep]?|download)[ \t]+\w+|{{|[\w.-]+/[\w.-]+)', re.MULTILINE)
_PRINT_EXAMPLE = re.compile(r'\\print\b[^\n]*\bexample\b', re.IGNORECASE)
# sd-embed: (phrase), ((phrase)), (phrase:1.3), [phrase]. word+ is compel, so C++ is not a weight.
_SD_EMBED_WEIGHT = re.compile(
    r'\(\([^()\n]+\)\)|\([^()\n]+:\s*\d+(?:\.\d+)?\)|\([^()\n]+\)|\[[^\[\]\n]+\]')
_COMPEL_WEIGHT = re.compile(
    r'(?<!\w)\w+\+\+?(?!\w)|\([^()\n]+\)\s*(?:\+\+?|\d+(?:\.\d+)?)')
_FLUX_NO_NEGATIVE = frozenset({
    'flux', 'flux-fill', 'flux-kontext', 'flux2', 'flux2-klein-kv'})

_MISSING_FILE = re.compile(r'(does not exist|not found|no such file|could not find)', re.IGNORECASE)

_MEDIA_EXT = re.compile(
    r'\.(png|jpe?g|webp|gif|bmp|tiff?|mp4|webm|mov|mkv|avi|apng|wav|mp3|flac|ogg|txt|safetensors|gguf|pt)$',
    re.IGNORECASE)


def _template_continuations(text: str) -> list[tuple[int, str]]:
    """
    Blocks that start with ``{`` and run as one Jinja render, the same way
    :py:class:`dgenerate.batchprocess.BatchProcessor` collects them.
    """
    import jinja2
    import dgenerate.batchprocess.jinjabalancechecker as _jinjabalancechecker

    blocks = []
    lines = text.splitlines(keepends=True)
    i = 0
    line_no = 0
    while i < len(lines):
        line_no += 1
        if lines[i].lstrip().startswith('{'):
            lexer = _jinjabalancechecker.JinjaBalanceChecker(jinja2.Environment())
            start = line_no
            chunk = lines[i]
            try:
                lexer.put_source(lines[i].lstrip())
                while not lexer.is_balanced() and i + 1 < len(lines):
                    i += 1
                    line_no += 1
                    chunk += lines[i]
                    lexer.put_source(lines[i].lstrip())
            except Exception:
                i += 1
                continue
            blocks.append((start, chunk))
        i += 1
    return blocks


def _jinja_uses(expr: str, name: str) -> bool:
    return re.search(rf'(?<![\w.]){re.escape(name)}(?![\w])', expr) is not None


def _unescaped_jinja_matches(block: str):
    """
    ``{{ expr }}`` spans that are not the delayed-escape
    ``{{ '{{ expr }}' }}`` form from the download examples.
    """
    escapes = [m.span() for m in _DELAYED_ESCAPE.finditer(block)]
    for match in _JINJA_EXPR.finditer(block):
        if any(start <= match.start() < end for start, end in escapes):
            continue
        yield match


def _template_delayed_expansion(text: str) -> list[dict]:
    """
    Delayed expansion inside a ``{{% %}}`` continuation: the whole block is
    rendered before any directive in it runs. Nested ``if`` / ``for`` are
    still one continuation.
    """
    errors = []
    for start, block in _template_continuations(text):
        if not block.lstrip().startswith('{%'):
            continue
        if _POSITIVE_TOKEN_IF.search(block) and _WORK_IN_BLOCK.search(block):
            errors.append({
                'line': start,
                'message': (
                    'When CIVIT_AI_TOKEN or HF_TOKEN is required, \\set it at the top, then '
                    '{% if not token.strip() %} \\print ... \\exit {% endif %} once. Do not '
                    'wrap \\set, \\download, or the generation in {% if token %}. After the '
                    'exit, \\set model and invoke at the top level.'
                ),
            })
        loop_vars = set(_FOR_VAR.findall(block))
        assigned = []
        for match in _ASSIGN_DIRECTIVE.finditer(block):
            name = match.group(2)
            if name not in loop_vars:
                assigned.append((name, match.start(), match.group(1)))
        assigned_names = {name for name, _, _ in assigned}

        for expr_match in _unescaped_jinja_matches(block):
            expr, pos = expr_match.group(1), expr_match.start()
            line_start = block.rfind('\n', 0, pos) + 1
            assign_line = _ASSIGN_DIRECTIVE.match(block, line_start)
            for name, assign_at, directive in assigned:
                if name not in assigned_names or not _jinja_uses(expr, name):
                    continue
                if assign_line and assign_line.group(2) == name:
                    continue
                if pos < assign_at:
                    continue
                line = start + block[:pos].count('\n')
                errors.append({
                    'line': line,
                    'message': (
                        f'\\{directive} {name} and {{{{ {name} }}}} are in the same {{% %}} '
                        f'continuation. The whole block is rendered before any \\{directive} '
                        f'in it runs, so {{{{ {name} }}}} is empty. For a token, \\set it at '
                        f'the top and {{% if not token.strip() %}} \\exit {{% endif %}}, then '
                        f'\\{directive} {name} after that block. Otherwise move '
                        f'\\{directive} {name} above the {{% if %}} / {{% for %}}. Do not '
                        f'write {{{{ {name} }}}} for a name you set in this block.'
                    ),
                })
                break

        finished_invocation = False
        in_invocation = False
        for i, raw in enumerate(block.splitlines()):
            strip = raw.lstrip()
            if not strip or strip.startswith('#') or strip.startswith('{%'):
                if in_invocation:
                    finished_invocation = True
                    in_invocation = False
                continue
            if strip.startswith('\\'):
                if in_invocation:
                    finished_invocation = True
                    in_invocation = False
                continue
            if strip.startswith('-'):
                if finished_invocation and _LAST_STAR.search(strip):
                    errors.append({
                        'line': start + i,
                        'message': (
                            'last_images / last_animations inside this {{% %}} continuation is '
                            'the value from before the block. Invocations in the block have not '
                            'run yet, including earlier iterations of a {{% for %}}. Use the '
                            'loop variable for this item, or put the next step after the '
                            '{{% endfor %}} / {{% endif %}} so last_images can update.'
                        ),
                    })
                continue
            if in_invocation:
                finished_invocation = True
            if finished_invocation and _LAST_STAR.search(strip):
                errors.append({
                    'line': start + i,
                    'message': (
                        'last_images / last_animations inside this {{% %}} continuation is '
                        'the value from before the block. Invocations in the block have not '
                        'run yet. Put the next step after the {{% endfor %}} / {{% endif %}}.'
                    ),
                })
            in_invocation = True
    return errors


@contextlib.contextmanager
def _assume_media_exists():
    """
    Make dgenerate's argument parsers accept media files that do not exist yet, usually
    files an earlier step writes, so the rest of the invocation is still validated.
    Missing files are reported separately by :py:func:`_missing_local_files`.
    """
    exists, isfile = os.path.exists, os.path.isfile

    def assumed(path):
        return isinstance(path, str) and not exists(path) and _MEDIA_EXT.search(path) is not None

    os.path.exists = lambda path: exists(path) or assumed(path)
    os.path.isfile = lambda path: isfile(path) or assumed(path)
    try:
        yield
    finally:
        os.path.exists, os.path.isfile = exists, isfile


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            _assume_media_exists():
        yield

# URI arguments that name something inside a model repository, or a file written, rather than a local input.
_REMOTE_URI_KEYS = {'weight-name', 'subfolder', 'revision', 'variant', 'dtype', 'quantizer', 'output-file'}


def _placeholder_download(runner):
    def directive(args: collections.abc.Sequence[str]):
        if args:
            name = os.path.basename(args[1]) if len(args) > 1 else 'download'
            runner.template_variables[args[0]] = os.path.join('downloads', name or 'download')
        return 0

    return directive


def _option_value(argv: list[str], names: tuple[str, ...], default: str) -> str:
    for i, value in enumerate(argv[:-1]):
        if value in names:
            return argv[i + 1]
    return default


def _placeholder_outputs(argv: list[str]) -> tuple[str, str]:
    """
    Stand-in image and animation files for what an invocation would write,
    so later invocations can use ``last_images`` and ``last_animations``.
    """
    folder = _option_value(argv, ('-o', '--output-path'), 'output')
    image_format = _option_value(argv, ('-if', '--image-format'), 'png').strip('.')
    animation_format = _option_value(argv, ('-af', '--animation-format'), 'mp4').strip('.')
    return (f'{folder}/check_output.{image_format}'.replace('\\', '/'),
            f'{folder}/check_output.{animation_format}'.replace('\\', '/'))


def _missing_local_files(argv: list[str], generated: set[str]) -> list[str]:
    """
    Paths in option values that look like local media files and do not exist.
    """
    missing = []
    for value in argv:
        if value.startswith('-') or '://' in value:
            continue
        for piece in re.split(r'[;,|]', value):
            key, _, rest = piece.partition('=')
            if rest and key.strip() in _REMOTE_URI_KEYS:
                continue
            piece = (rest or key).strip().strip('"\'')
            if not _MEDIA_EXT.search(piece) or os.path.exists(piece) or piece.replace('\\', '/') in generated:
                continue
            # org/repo/file.ext is a Hugging Face path, not a local file.
            if piece.replace('\\', '/').count('/') >= 2 and not piece.startswith(('.', '/', '\\', 'path/to/')) \
                    and not os.path.splitdrive(piece)[0]:
                continue
            missing.append(piece)
    return missing


_REPO_ID = re.compile(r'^[\w.-]+/[\w.-]+$')


def _placeholders(argv: list[str]) -> list[str]:
    found = []
    for value in argv:
        for piece in re.split(r'[;,|]', value):
            piece = piece.partition('=')[2] or piece
            piece = piece.strip().strip('"\'').replace('\\', '/')
            if piece.startswith('path/to/') and piece not in found:
                found.append(piece)
    return found


def _repo_problem(repo: str, cache: dict) -> str | None:
    """
    An error message if ``repo`` looks like a Hugging Face repo id and does not exist.
    Network failures are not reported.
    """
    if repo in cache:
        return cache[repo]
    cache[repo] = None
    if not _REPO_ID.match(repo) or os.path.exists(repo):
        return None
    import huggingface_hub
    try:
        if huggingface_hub.repo_exists(repo):
            return None
        similar = [m.id for m in huggingface_hub.HfApi().list_models(
            search=repo.split('/')[1], sort='downloads', limit=3)]
    except Exception:
        return None
    hint = f', did you mean {" or ".join(f"{s!r}" for s in similar)}?' if similar else '.'
    cache[repo] = f'The Hugging Face repo "{repo}" does not exist{hint}'
    return cache[repo]


# Options whose URIs name a model, either first ("repo;scale=0.5") or as model= ("AutoencoderKL;model=repo").
_MODEL_URI_OPTIONS = (
    'sdxl_refiner_uri', 's_cascade_decoder_uri', 'unet_uri', 'second_model_unet_uri', 'transformer_uri',
    'vae_uri', 'lora_uris', 'ltx_ic_lora_uri', 'wan_second_transformer_uri', 'image_encoder_uri',
    'ip_adapter_uris', 'textual_inversion_uris',
    'text_encoder_uris', 'second_model_text_encoder_uris', 'controlnet_uris', 't2i_adapter_uris',
)


def _uri_repos(config) -> list[tuple[str, str | None, str | None]]:
    """
    (repo, weight-name, subfolder) for every Hugging Face repo named in a model URI option.
    """
    repos = []
    for name in _MODEL_URI_OPTIONS:
        value = getattr(config, name, None)
        for uri in [value] if isinstance(value, str) else (value or []):
            if not isinstance(uri, str):
                continue
            parts = [p.strip().strip('"\'') for p in uri.split(';')]
            args = {k.strip(): v.strip().strip('"\'') for k, _, v in (p.partition('=') for p in parts[1:] if '=' in p)}
            repo = parts[0] if _REPO_ID.match(parts[0]) else args.get('model', '')
            if _REPO_ID.match(repo):
                entry = (repo, args.get('weight-name') or None, args.get('subfolder') or None)
                if entry not in repos:
                    repos.append(entry)
    return repos


def _repo_file_list(repo: str, cache: dict) -> list[str] | None:
    if repo not in cache:
        cache[repo] = None
        try:
            import huggingface_hub
            cache[repo] = huggingface_hub.HfApi().list_repo_files(repo)
        except Exception:
            pass
    return cache[repo]


def _weight_name_problem(repo: str, weight_name: str | None, subfolder: str | None, cache: dict) -> str | None:
    if not weight_name or os.path.exists(repo):
        return None
    files = _repo_file_list(repo, cache)
    if not files:
        return None
    name = os.path.basename(str(weight_name).replace('\\', '/'))
    path = f'{subfolder.strip("/")}/{name}' if subfolder else name
    if path in files or any(f == name or f.endswith('/' + name) for f in files):
        return None
    weights = [f for f in files if f.endswith(('.safetensors', '.bin', '.pt', '.gguf'))]
    shown = ', '.join(weights[:12]) + (', ...' if len(weights) > 12 else '')
    return f'The Hugging Face repo "{repo}" has no file "{path}". Its weight files are: {shown or "none"}.'


def _model_path_problem(path: str) -> str | None:
    """
    An error message for a model path that is not a repo id, an existing path, a URL, or a file.
    """
    if not path or os.path.exists(path) or '://' in path or _REPO_ID.match(path):
        return None
    if os.path.splitext(path)[1].lower() in ('.safetensors', '.ckpt', '.bin', '.pt', '.pth', '.gguf'):
        return None
    if path.replace('\\', '/').count('/') < 2 or path.startswith(('.', '/', '\\', 'path/to/')) \
            or os.path.splitdrive(path)[0]:
        return None
    return (f'"{path}" is not a Hugging Face repo, a repo is exactly organization/name, '
            f'for example "{"/".join(path.replace(chr(92), "/").split("/")[-2:])}".')


def _variant_problem(repo: str, variant: str | None, cache: dict) -> str | None:
    """
    An error message if the Hugging Face repo ``repo`` has no weights for ``variant``.
    Network failures are not reported.
    """
    if not variant or not _REPO_ID.match(repo) or os.path.exists(repo):
        return None
    files = _repo_file_list(repo, cache)
    if not files:
        return None
    variants = sorted({m.group(1) for f in files
                       for m in [re.search(r'\.([\w-]+)\.(?:safetensors|bin)$', f.rsplit('/', 1)[-1])] if m})
    if variant in variants:
        return None
    available = f'Its variants are: {", ".join(variants)}.' if variants else \
        'It has no variants, remove --variant.'
    return f'The Hugging Face repo "{repo}" has no "{variant}" variant. {available}'


def _is_inpaint_unet(repo: str, cache: dict) -> bool:
    """
    Whether ``repo`` has a UNet taking the 9 input channels of inpainting
    (latents, mask, and masked image latents).
    """
    if repo not in cache:
        cache[repo] = False
        try:
            if os.path.isdir(repo):
                path = os.path.join(repo, 'unet', 'config.json')
            elif _REPO_ID.match(repo):
                import huggingface_hub
                path = huggingface_hub.hf_hub_download(repo, 'config.json', subfolder='unet')
            else:
                return False
            with open(path, encoding='utf-8') as f:
                cache[repo] = json.load(f).get('in_channels') == 9
        except Exception:
            pass
    return cache[repo]


def _has_mask(image_seeds: list[str] | None) -> bool:
    """
    Whether any seed is ``image;mask``. A template without ``;`` is not a mask;
    a parse failure is only treated as a mask when the seed contains ``;``.
    """
    import dgenerate.mediainput as _mediainput
    for seed in image_seeds or []:
        seed = str(seed)
        try:
            if _mediainput.parse_image_seed_uri(seed).mask_images:
                return True
        except Exception:
            if ';' in seed:
                return True
    return False


def _model_type_str(config) -> str:
    import dgenerate.pipelinewrapper.enums as _enums
    try:
        return _enums.get_model_type_string(config.model_type)
    except Exception:
        return str(getattr(config, 'model_type', '')).lower()


def _size_px(size) -> int | None:
    if not size:
        return None
    if isinstance(size, int):
        return size
    try:
        return max(int(size[0]), int(size[1]))
    except (TypeError, IndexError, ValueError):
        return None


def _runaway_comments(text: str) -> list[dict]:
    """
    A long streak of comment-only lines, usually the model looping instead of
    finishing the remaining invocations.
    """
    streak = 0
    start = 0
    for i, line in enumerate(text.splitlines()):
        if line.lstrip().startswith('#'):
            if streak == 0:
                start = i
            streak += 1
            if streak >= 20:
                return [{'line': start + 1,
                         'message': 'This config repeats the same comments instead of '
                                    'finishing the remaining invocations. Delete the '
                                    'repeated comments and write the actual generate / '
                                    '\\image_process / inpaint steps.'}]
        else:
            streak = 0
    return []


def _print_says_example(text: str) -> list[dict]:
    errors = []
    for i, line in enumerate(text.splitlines(), 1):
        if _PRINT_EXAMPLE.search(line):
            errors.append({
                'line': i,
                'message': '\\print should not call this an example. Write '
                           '"Set HF_TOKEN environmental variable or pass --auth-token." '
                           'or "Set CIVIT_AI_TOKEN environmental variable."',
            })
    return errors


def _has_weight_syntax(text: str, weighter: str) -> bool:
    weighter = str(weighter)
    if re.match(r'compel\b', weighter) and 'sdwui' not in weighter:
        return _COMPEL_WEIGHT.search(text) is not None
    return _SD_EMBED_WEIGHT.search(text) is not None


def _missing_weight_marks(config, weighter, names: tuple[str, ...]) -> bool:
    if not weighter:
        return False
    for name in names:
        for prompt in getattr(config, name, None) or []:
            positive = getattr(prompt, 'positive', None) or ''
            if positive.strip() and not _has_weight_syntax(positive, weighter):
                return True
    return False


_IMAGE_MODEL_TYPES = frozenset({
    'sd', 'sdxl', 'sd3', 'flux', 'flux-fill', 'flux-kontext', 'flux2', 'flux2-klein-kv',
    'z-image', 'z-image-omni', 'qwen-image', 'qwen-image-edit', 'qwen-image-layered', 'pix2pix',
    'sdxl-pix2pix', 'sd3-pix2pix', 'kolors', 'upscaler-x2', 'upscaler-x4',
    'if', 'ifs', 'ifs-img2img',
})
_NEED_SEEDS = frozenset({
    'flux-fill', 'flux-kontext', 'pix2pix', 'sdxl-pix2pix', 'sd3-pix2pix',
    'upscaler-x2', 'upscaler-x4', 'ifs-img2img',
})
_NEED_MASK_TYPES = frozenset({'flux-fill'})
_NEEDS_ORIGINAL = re.compile(r'\b(letterbox|outpaint-mask)\b', re.IGNORECASE)
_PATCHMATCH = re.compile(r'\bpatchmatch\b', re.IGNORECASE)
_MAKES_MASK = re.compile(r'\boutpaint-mask\b', re.IGNORECASE)
_LAST_ANIM_SEED = re.compile(r'--image-seeds[^\n]*last_animations')
_OUTPUT_FILE_ARG = re.compile(r'output-file\s*=\s*([^;]+)')
_LATENT_FORMATS = frozenset({'pt', 'pth', 'safetensors'})


_IMAGE_PROCESSOR_OPTIONS = ('seed_image_processors', 'mask_image_processors',
                            'control_image_processors', 'last_frame_image_processors',
                            'reference_image_processors', 'adapter_image_processors',
                            'wan_pose_image_processors', 'wan_face_image_processors',
                            'wan_driving_image_processors', 'wan_background_image_processors',
                            'post_processors')
_LATENTS_PROCESSOR_OPTIONS = ('latents_processors', 'latents_post_processors', 'img2img_latents_processors')
_PROMPT_WEIGHTER_OPTIONS = ('prompt_weighter_uri', 'second_model_prompt_weighter_uri')
_PROMPT_UPSCALER_OPTIONS = ('prompt_upscaler_uri', 'second_model_prompt_upscaler_uri', 'second_prompt_upscaler_uri',
                            'second_model_second_prompt_upscaler_uri', 'third_prompt_upscaler_uri')


class _UriChecker:
    """
    Validates plugin URIs with the loaders dgenerate uses, without creating the plugins.
    """

    def __init__(self):
        import dgenerate.imageprocessors as _imageprocessors
        import dgenerate.latentsprocessors as _latentsprocessors
        import dgenerate.promptupscalers as _promptupscalers
        import dgenerate.promptweighters as _promptweighters
        self.image_processors = _imageprocessors.ImageProcessorLoader()
        self.latents_processors = _latentsprocessors.LatentsProcessorLoader()
        self.prompt_weighters = _promptweighters.PromptWeighterLoader()
        self.prompt_upscalers = _promptupscalers.PromptUpscalerLoader()
        self.loaders = (self.image_processors, self.latents_processors,
                        self.prompt_weighters, self.prompt_upscalers)

    def load_plugin_modules(self, paths):
        for loader in self.loaders:
            try:
                loader.load_plugin_modules(paths)
            except Exception:
                pass

    @staticmethod
    def _uris(value) -> list[str]:
        if not value:
            return []
        values = [value] if isinstance(value, str) else list(value)
        return [v for v in values if v and v != '+']

    @staticmethod
    def _problems(loader, uris, **kwargs) -> list[str]:
        problems = []
        for uri in uris:
            try:
                loader.check_uri(uri, **kwargs)
            except Exception as e:
                problems.append(f'{str(e).strip()} (in "{uri}")')
        return problems

    def image_process(self, uris) -> list[str]:
        return self._problems(self.image_processors, self._uris(uris))

    def invocation(self, config) -> list[str]:
        problems = []
        for name in _IMAGE_PROCESSOR_OPTIONS:
            problems += self._problems(self.image_processors, self._uris(getattr(config, name, None)))
        for name in _LATENTS_PROCESSOR_OPTIONS:
            problems += self._problems(self.latents_processors, self._uris(getattr(config, name, None)),
                                       model_type=config.model_type)
        for name in _PROMPT_WEIGHTER_OPTIONS:
            problems += self._problems(self.prompt_weighters, self._uris(getattr(config, name, None)),
                                       model_type=config.model_type, dtype=config.dtype)
        for name in _PROMPT_UPSCALER_OPTIONS:
            problems += self._problems(self.prompt_upscalers, self._uris(getattr(config, name, None)))
        return problems


def check_config(text: str) -> dict:
    import dgenerate.arguments as _arguments
    import dgenerate.image_process.arguments as _image_process_arguments
    import dgenerate.batchprocess as _batchprocess
    import dgenerate.messages as _messages
    import dgenerate.pipelinewrapper.help as _help
    import dgenerate.pipelinewrapper.schedulers as _schedulers

    _messages.messages_to_null()
    _messages.errors_to_null()

    errors: list[dict] = []
    warnings: list[dict] = []
    invocations: list[dict] = []
    errors.extend(_template_delayed_expansion(text))
    errors.extend(_runaway_comments(text))
    errors.extend(_print_says_example(text))
    if _LAST_ANIM_SEED.search(text) and not re.search(r'--model-type\s+(ltx|wan|wan-animate)\b', text):
        errors.append({
            'line': None,
            'message': '--image-seeds uses last_animations, but this config has no video '
                       'step. Image models and image-to-video take last_images or '
                       'the user photo, never last_animations.',
        })

    runner = _batchprocess.ConfigRunner(throw=True)
    generated: set[str] = set()
    directive_inputs: list[dict] = []
    uri_checker = _UriChecker()
    last_from_process = False
    current_last_images: list[str] = []
    last_animation_path: str | None = None
    prev_was_video = False
    process_outputs: set[str] = set()
    mask_outputs: set[str] = set()
    last_process_kind: str | None = None
    prev_image_format: str | None = None
    prev_size: int | None = None
    debug_files: set[str] = set()

    def note_placeholders(argv):
        for path in _placeholders(argv):
            if invocations:
                warnings.append({'line': runner.current_line, 'placeholder': path,
                                 'after_line': invocations[-1]['line'],
                                 'message': f'"{path}" is a placeholder used after an invocation '
                                            f'that generates images.'})
            else:
                warnings.append({'line': runner.current_line, 'placeholder': path, 'after_line': None,
                                 'message': f'"{path}" is a placeholder for an input file.'})

    def invoker(command_line, argv):
        nonlocal last_from_process, current_last_images, last_animation_path, last_process_kind
        if any(a in ('-h', '--help') or (a.startswith('--') and a.endswith('-help')) for a in argv):
            # Help directives such as \image_processor_help run dgenerate with a help option.
            return 0
        note_placeholders(argv)
        invocations.append({'line': runner.current_line, 'argv': list(argv)})
        try:
            with _quiet():
                runner.render_loop.config = _arguments.parse_args(list(argv), throw=True, log_error=False)
            runner.template_variables.update(runner._generate_template_variables())
        except (Exception, SystemExit):
            # Reported when the invocation is validated below.
            pass
        image, animation = _placeholder_outputs(list(argv))
        generated.update((image, animation))
        runner.template_variables['last_images'] = [image]
        runner.template_variables['last_animations'] = [animation]
        last_from_process = False
        current_last_images = [image]
        last_animation_path = animation
        last_process_kind = None
        return 0

    runner.invoker = invoker
    directive_calls = []

    def stub(args):
        directive_calls.append(args)
        return 0

    def image_process(args):
        nonlocal last_from_process, current_last_images, last_process_kind
        directive_calls.append(args)
        args = list(args)
        note_placeholders(args)
        inputs = []
        for arg in args:
            if arg.startswith('-'):
                break
            inputs.append(arg)
        directive_inputs.append({'line': runner.current_line, 'inputs': inputs,
                                 'generated': set(generated)})
        last_set = {p.replace('\\', '/') for p in current_last_images}
        rendered = [i.replace('\\', '/') for i in inputs]
        joined = ' '.join(args)
        uses_last = last_from_process and (
            any(i in last_set for i in rendered) or any('last_images' in i for i in inputs))
        on_process_output = any(i.replace('\\', '/') in process_outputs for i in inputs)
        on_mask = any(i.replace('\\', '/') in mask_outputs for i in inputs)
        if _NEEDS_ORIGINAL.search(joined) and (uses_last or on_process_output or on_mask):
            errors.append({
                'line': runner.current_line,
                'message': '\\image_process letterbox / outpaint-mask needs the generated '
                           'image, not a previous \\image_process output (usually the mask). '
                           'Save the generated file with "\\set input_image '
                           '{{ quote(first(last_images)) }}" after the invocation that creates '
                           'it, and pass {{ input_image }} to every \\image_process that needs '
                           'that image.',
            })
        if _PATCHMATCH.search(joined) and (on_mask or (uses_last and last_process_kind == 'mask')):
            errors.append({
                'line': runner.current_line,
                'message': '\\image_process patchmatch is reading the mask as its input. '
                           'Letterbox the generated image, then patchmatch that letterboxed '
                           'file with mask=mask.png. Do not pass the mask as the '
                           '\\image_process input.',
            })
        try:
            with _quiet():
                parsed = _image_process_arguments.parse_args(
                    args, help_name='\\image_process', throw=True, log_error=False)
            if parsed is not None:
                for uri in parsed.processors or []:
                    for key, value in re.findall(r'(?:^|;)\s*(mask|image)\s*=\s*["\']?([^;"\']+)', uri):
                        if value.strip().replace('\\', '/') in (i.replace('\\', '/') for i in inputs):
                            errors.append({
                                'line': runner.current_line,
                                'message': f'\\image_process processes "{value.strip()}" with itself as {key}=. '
                                           f'last_images is replaced by every invocation and \\image_process, so '
                                           f'it no longer holds the generated image. Save the image with '
                                           f'"\\set input_image {{{{ quote(first(last_images)) }}}}" directly after '
                                           f'the invocation that generates it, and use {{{{ input_image }}}} '
                                           f'as the input of each \\image_process.'})
                if parsed.plugin_module_paths:
                    uri_checker.load_plugin_modules(parsed.plugin_module_paths)
                for problem in uri_checker.image_process(parsed.processors):
                    errors.append({'line': runner.current_line, 'message': f'\\image_process: {problem}'})
        except Exception as e:
            if not _MISSING_FILE.search(str(e)):
                errors.append({'line': runner.current_line, 'message': f'\\image_process: {str(e).strip()}'})
        output = _option_value(args, ('-o', '--output'), '')
        if not output:
            inputs = [a for a in args if not a.startswith('-')]
            output = f'{os.path.splitext(inputs[0])[0]}_processed.png' if inputs else 'processed.png'
        output = output.replace('\\', '/')
        generated.add(output)
        process_outputs.add(output)
        if _MAKES_MASK.search(joined):
            mask_outputs.add(output)
            last_process_kind = 'mask'
        elif _NEEDS_ORIGINAL.search(joined) and not _PATCHMATCH.search(joined):
            last_process_kind = 'letterbox'
        else:
            last_process_kind = 'other'
        runner.template_variables['last_images'] = [output]
        runner.template_variables['last_animations'] = [output] if media_kind(output) == 'video' else []
        last_from_process = True
        current_last_images = [output]
        return 0

    for name in _STUBBED_DIRECTIVES:
        if name in runner.directives:
            runner.directives[name] = stub
    runner.directives['image_process'] = image_process
    runner.directives['download'] = _placeholder_download(runner)
    runner.template_functions['download'] = lambda url, *a, **kw: os.path.join('downloads', os.path.basename(url))

    image_size = runner.template_functions['image_size']
    image_width = runner.template_functions['image_width']
    image_height = runner.template_functions['image_height']
    scale_size = runner.template_functions['scale_size']

    def placeholder_image_size(file: str, format_size: bool = True):
        if os.path.exists(file):
            return image_size(file, format_size)
        return '512x512' if format_size else (512, 512)

    def placeholder_image_width(file: str) -> int:
        if os.path.exists(file):
            return image_width(file)
        return 512

    def placeholder_image_height(file: str) -> int:
        if os.path.exists(file):
            return image_height(file)
        return 512

    def placeholder_scale_size(size: str | tuple, scale=1, format_size: bool = True):
        try:
            return scale_size(size, scale, format_size)
        except Exception:
            if isinstance(size, str) and not os.path.exists(size):
                try:
                    return scale_size((512, 512), scale, format_size)
                except Exception:
                    return (512, 512) if not format_size else '512x512'
            raise

    runner.template_functions['image_size'] = placeholder_image_size
    runner.template_functions['image_width'] = placeholder_image_width
    runner.template_functions['image_height'] = placeholder_image_height
    runner.template_functions['scale_size'] = placeholder_scale_size

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runner.run_string(text)
    except SystemExit:
        pass
    except Exception as e:
        errors.append({'line': runner.current_line, 'message': f'{type(e).__name__}: {e}'})

    if not invocations and not directive_calls and not errors:
        errors.append({'line': None, 'message': 'The config contains no dgenerate invocation.'})

    for call in directive_inputs:
        for path in _missing_local_files(call['inputs'], call['generated']):
            warnings.append({'line': call['line'], 'message': f'Local file "{path}" does not exist.'})

    repo_problems: dict = {}
    repo_files: dict = {}
    inpaint_unets: dict = {}
    prev_animation: str | None = None
    for invocation in invocations:
        argv = invocation['argv']
        line = invocation['line']
        for value in argv:
            if value.replace('\\', '/') in debug_files:
                errors.append({
                    'line': line,
                    'message': f'"{value}" was written with output-file= on a processor, which '
                               f'is a debug dump. It is not an input for a later invocation. '
                               f'Keep --control-image-processors on the invocation that uses '
                               f'--control-nets, or use last_images for the generated image.',
                })
            for match in _OUTPUT_FILE_ARG.finditer(value):
                debug_files.add(match.group(1).strip().strip('"\'').replace('\\', '/'))
        try:
            with _quiet():
                config = _arguments.parse_args(argv, throw=True, log_error=False)
            if config.plugin_module_paths:
                uri_checker.load_plugin_modules(config.plugin_module_paths)
            for problem in uri_checker.invocation(config):
                errors.append({'line': line, 'message': problem})
            for repo, weight_name, subfolder in _uri_repos(config):
                uri_problem = _repo_problem(repo, repo_problems) or \
                    _weight_name_problem(repo, weight_name, subfolder, repo_files)
                if uri_problem:
                    errors.append({'line': line, 'message': uri_problem})
            model_path = config.model_path or ''
            if model_path.lower().endswith('.gguf'):
                errors.append({
                    'line': line,
                    'message': 'A .gguf file is a transformer or UNet replacement, not the '
                               'model path. Use the Hugging Face repo as the first line and '
                               '--transformer path/to/file.gguf (see the Flux, Flux.2, SD3, '
                               'Z-Image, Qwen-Image, LTX, and Wan GGUF examples).',
                })
            problem = _model_path_problem(model_path) or \
                _repo_problem(model_path, repo_problems) or \
                _variant_problem(model_path, config.variant, repo_files)
            if problem:
                errors.append({'line': line, 'message': problem})
            mt = _model_type_str(config)
            seeds = [str(s) for s in (config.image_seeds or [])]
            if _missing_weight_marks(config, config.prompt_weighter_uri,
                                     ('prompts', 'second_prompts', 'third_prompts')) \
                    or _missing_weight_marks(config, config.second_model_prompt_weighter_uri,
                                             ('second_model_prompts', 'second_model_second_prompts')):
                errors.append({
                    'line': line,
                    'message': '--prompt-weighter is set but the prompt has no weighting '
                               'syntax, so it does nothing. With sd-embed write '
                               '(creatures:1.3) or ((creatures)); with compel write '
                               'creatures+ or (creatures)++. Drop --prompt-weighter if '
                               'you do not want weights.',
                })
            if mt in _FLUX_NO_NEGATIVE:
                for prompt in config.prompts or []:
                    if getattr(prompt, 'negative', None):
                        errors.append({
                            'line': line,
                            'message': f'--model-type {mt} does not use a negative prompt. '
                                       f'Remove the ; and everything after it from --prompts.',
                        })
                        break
            if _is_inpaint_unet(model_path, inpaint_unets) or mt in _NEED_MASK_TYPES \
                    or 'inpaint' in model_path.lower():
                if not _has_mask(config.image_seeds):
                    errors.append({
                        'line': line,
                        'message': f'"{model_path or mt}" is an inpainting model, it needs '
                                   f'--image-seeds with an image and a mask, like '
                                   f'"image.png;mask.png". Generate from a prompt alone with a '
                                   f'model that is not for inpainting.',
                    })
            if mt in _NEED_SEEDS and not seeds:
                errors.append({
                    'line': line,
                    'message': f'--model-type {mt} needs --image-seeds.',
                })
            if mt in _IMAGE_MODEL_TYPES:
                for seed in seeds:
                    if 'last_animations' in seed or (
                            prev_animation and prev_animation.replace('\\', '/')
                            in seed.replace('\\', '/')):
                        errors.append({
                            'line': line,
                            'message': f'--model-type {mt} is an image model. Use last_images '
                                       f'or the user photo as --image-seeds, not last_animations.',
                        })
                        break
            if mt in ('ltx', 'wan') and not prev_was_video:
                for seed in seeds:
                    if 'last_animations' in seed or (
                            prev_animation and prev_animation.replace('\\', '/')
                            in seed.replace('\\', '/')):
                        errors.append({
                            'line': line,
                            'message': 'Image-to-video uses last_images (the still), not '
                                       'last_animations. last_animations is a video file from '
                                       'the previous step.',
                        })
                        break
            if prev_image_format in _LATENT_FORMATS and any(
                    a in ('-iss', '--image-seed-strengths') for a in argv):
                if seeds and not any(s.startswith('latents:') for s in seeds):
                    errors.append({
                        'line': line,
                        'message': 'The previous invocation wrote latents (--image-format pt). '
                                   '--image-seed-strengths is for img2img on images. Write a '
                                   'PNG or JPEG in the first step (omit --image-format pt), then '
                                   'img2img with --image-seed-strengths. For a latent handoff to '
                                   'a refiner, use --denoising-start instead of strengths.',
                    })
            if 'refiner' in model_path.lower() and seeds and not config.denoising_start:
                if not any(s.startswith('latents:') or s.endswith('.pt') for s in seeds):
                    errors.append({
                        'line': line,
                        'message': 'The SDXL refiner as its own invocation needs the previous '
                                   'step to write latents (--image-format pt --denoising-end) '
                                   'and this step to use --denoising-start. A PNG --image-seeds '
                                   'without --denoising-start is ordinary img2img. Prefer '
                                   '--sdxl-refiner on the base invocation.',
                    })
            cur_size = _size_px(config.output_size)
            if mt.startswith('upscaler'):
                if not seeds:
                    errors.append({
                        'line': line,
                        'message': 'An upscaler needs --image-seeds of the image to enlarge.',
                    })
                if prev_size and cur_size and cur_size < prev_size:
                    errors.append({
                        'line': line,
                        'message': f'The previous image is about {prev_size}px. --output-size '
                                   f'{cur_size} shrinks it before the {mt} upscaler, so the '
                                   f'result is not larger. Set --output-size to the previous '
                                   f'image size, or omit it. --output-size is the size fed in, '
                                   f'not the final size.',
                    })
                factor = 4 if mt == 'upscaler-x4' else 2
                prev_size = (cur_size or prev_size or 0) * factor or prev_size
            elif cur_size:
                prev_size = cur_size
            for uri in (config.scheduler_uri or []) + (config.second_model_scheduler_uri or []):
                if _help.scheduler_is_help(uri):
                    errors.append({
                        'line': line,
                        'message': '--schedulers help only prints scheduler names; it is not a '
                                   'generation. Remove this invocation or use a real scheduler '
                                   'class name.',
                    })
                    continue
                try:
                    _schedulers.check_scheduler_uri(uri)
                except _schedulers.SchedulerLoadError as e:
                    errors.append({'line': line, 'message': str(e).strip()})
            prev_image_format = config.image_format
            prev_was_video = mt in ('ltx', 'wan', 'wan-animate')
            _img, prev_animation = _placeholder_outputs(list(argv))
        except SystemExit:
            errors.append({'line': line, 'message': 'dgenerate exited while parsing the invocation.'})
        except Exception as e:
            entry = {'line': line, 'message': _explain_parse_error(str(e).strip())}
            if not _MISSING_FILE.search(entry['message']):
                errors.append(entry)
            elif not any(path in entry['message'] for path in generated):
                warnings.append(entry)
        for path in _missing_local_files(argv, generated):
            warnings.append({'line': line, 'message': f'Local file "{path}" does not exist.'})

    return {
        'ok': not errors,
        'errors': errors,
        'warnings': warnings,
        'invocations': len(invocations),
    }


def _explain_parse_error(message: str) -> str:
    """
    Argparse's missing ``model_path`` means this invocation is only options.

    A blank line, or a wrapped line that does not start with ``-``, ended the
    invocation that holds the model path. The next ``--`` line is then its own
    command.
    """
    bare = message.removeprefix('dgenerate: error: ').strip()
    if bare == 'the following arguments are required: model_path':
        return ('This line was run on its own, with no model path. A blank line, or a '
                'wrapped line that does not start with -, ended the invocation above. '
                'Put this line back with that invocation. Do not wrap --prompts, and '
                'leave a {{ }} expression on the line it already occupies.')
    return message


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print('usage: python -m dgenerate.assistant.check CONFIG', file=sys.stderr)
        return 2
    with open(argv[0], encoding='utf-8') as f:
        text = f.read()
    report = check_config(text)
    sys.stdout.write(json.dumps(report) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
