import collections.abc
import dataclasses
import difflib
import enum
import functools
import os
import re
import types
import typing

import dgenerate.assistant.corpus as _corpus
import dgenerate.assistant.index as _index
import dgenerate.textprocessing as _textprocessing

CONTEXT_BUDGET = 40000
MAX_EXAMPLES = 4
MAX_GUIDES = 8
MAX_GUIDE_CHARS = 2800
MAX_EXAMPLE_CHARS = 6000
MAX_ARGUMENT_CHARS = 1200
MAX_REFERENCE_CHARS = 1500

_OPTION = re.compile(r'(?<![\w-])(--?[a-zA-Z][a-zA-Z0-9-]*)')
_OPTION_LINE = re.compile(r'^\s*(--?[a-zA-Z][a-zA-Z0-9-]*)', re.MULTILINE)
_SCHEDULER = re.compile(r'\b([A-Z]\w*Scheduler)\b')
_FENCE = re.compile(r'^\s*```')
_MEDIA_PATH = re.compile(r'(?<![\w/:.])(?:\.{1,2}[/\\])*(?:examples[/\\])?media[/\\]([\w.-]+)')

_FILE_EXT = (r'png|jpe?g|webp|gif|bmp|tiff?|mp4|webm|mov|mkv|avi|apng|wav|mp3|flac|ogg|'
             r'txt|json|safetensors|gguf|ckpt|pt|bin')
# A quoted path, or an unquoted token ending in a known file extension.
_REQUEST_FILE = re.compile(
    rf'"([^"\n]+\.(?:{_FILE_EXT}))"|\'([^\'\n]+\.(?:{_FILE_EXT}))\'|'
    rf'(?<![\w"\'])((?:[A-Za-z]:)?[^\s"\',;()]+\.(?:{_FILE_EXT}))(?![\w])',
    re.IGNORECASE)

SHEBANG = '#! /usr/bin/env dgenerate --file'

SYSTEM_PROMPT = """\
You write dgenerate config scripts. dgenerate is a command line tool for \
generating images, video, and audio with diffusion models. A config (.dgen file) \
is a script that runs one or more dgenerate invocations.

Config syntax:
- The first two lines are "{shebang}" and "#! dgenerate {version}".
- Lines starting with # are comments.
- An invocation starts with the model path on its own line. Each following line \
starting with - is an option of that invocation. A blank line or a line that does \
not start with - ends it.
- Values containing spaces or ; are quoted: --prompts "a red fox in the snow; blurry, low quality".
- A prompt is split at its first ; into a positive and a negative prompt. Everything \
before the ; is what the output should show: subject, action, style, lighting, camera, \
quality. Everything after it is only what the output must not show, like "blurry, \
low quality, distorted". Separate phrases with commas, never with ;. A prompt has at \
most one ;, and none if no negative prompt is needed. Models that do not use a \
negative prompt (check the examples) get no ; at all.
- Directives start with a backslash, for example \\set name value, \\setp name expr, \\print text.
- Jinja2 templates work anywhere: {{{{ variable }}}}, {{% if %}}...{{% endif %}}, {{% for %}}...{{% endfor %}}.
- A line that starts with {{% is one template continuation (a heredoc): \
the whole {{% if %}}...{{% endif %}} or {{% for %}}...{{% endfor %}} \
is rendered first, then the result is run. Nested if/for stay in that \
same render. \\set, \\setp, \\sete, \\download, and \\gen_seeds inside \
the block have not run yet, so {{{{ name }}}} in the same block is \
empty. last_images inside the block is the list from before the \
block, including later iterations of a {{% for %}}. When CIVIT_AI_TOKEN \
or HF_TOKEN is required, \\set it on the first lines, then one early \
exit: {{% if not token.strip() %}} \\print Set HF_TOKEN environmental \
variable or pass --auth-token. \\exit {{% endif %}} (CivitAI: \
CIVIT_AI_TOKEN, no --auth-token). After that, \\set model and the \
invocation are at the top level. Never wrap \\set, \\download, or a \
generation in {{% if token %}}. Official .dgen files say "to run this \
example" because they are examples; you are writing a config the user \
will run, so that \\print is a real instruction, never an example. \
When the repo is gated, write exactly: \\print Set HF_TOKEN environmental \
variable or pass --auth-token. When the config downloads from civitai.com, \
write exactly: \\print Set CIVIT_AI_TOKEN environmental variable. \
A Hugging Face repo does not use CIVIT_AI_TOKEN. Public checkpoints get \
no HF_TOKEN block. Put a per-item loop variable on the option line \
({{% for image in last_images %}} on --image-seeds) when you only \
need to expand a list.
- Environment variables expand as $VAR, ${{VAR}} or %VAR%.

Rules:
- Use only options, URI arguments, model types, and plugin names that appear in the \
reference material. Never invent an option. If something is not supported, say so in \
a # comment instead of guessing.
- Base the config on the closest example and keep its model, --model-type, --dtype, \
offloading, and required settings unless the request asks otherwise. Follow the \
documented constraints on output size, frame counts, and guidance.
- Copy Hugging Face repo names (org/name) exactly as they appear in the reference material. \
Never invent a repo or change its organization or name.
- Inpainting models (repos with "inpainting" in the name) only work with --image-seeds \
"image.png;mask.png". A step that generates from a prompt alone needs a regular model of the \
same family, for example stabilityai/stable-diffusion-xl-base-1.0 for SDXL.
- HF_TOKEN is only for gated checkpoints. Omit it for public repos, even when \
the closest example has a token block. Public: SD 1.5, SD 2.1, SDXL (base, \
refiner, inpainting), FLUX.1-schnell, Kolors, Stable Cascade, the upscalers, \
pix2pix. Gated: FLUX.1-dev, FLUX.1-Fill-dev, FLUX.1-Kontext-dev, SD3, SD3.5, \
LTX-2.5, LTX-Video, DeepFloyd IF. For a gated repo, \\set token %HF_TOKEN%, \
then {{% if not token.strip() and not '--auth-token' in injected_args %}} \
\\print Set HF_TOKEN environmental variable or pass --auth-token. \
\\exit {{% endif %}}. Do not wrap the generation. Do not write "example" \
in that \\print. CIVIT_AI_TOKEN is only for a civitai.com download link. \
Omit it when every model is a Hugging Face repo or a local file. For a \
CivitAI link: \\set civit_ai_token %CIVIT_AI_TOKEN%, then \
{{% if not civit_ai_token.strip() %}} \\print Set CIVIT_AI_TOKEN \
environmental variable. \\exit {{% endif %}}. No --auth-token check.
- Paths marked (input file) in the request are inputs, for example for --image-seeds. \
Copy them exactly. They are never output paths: --output-path is a new directory name \
for the results. Paths in the examples that start with path/to/ are placeholders. If the request needs an input file it did not name, use a \
placeholder like path/to/input.png and say in a # comment that it must be replaced.
- To feed one invocation's results into a later one (refine, upscale, img2img, image to video, \
\\image_process), use the template variables set after every invocation: {{{{ last_images }}}} \
and {{{{ last_animations }}}} list the files it wrote. Write --image-seeds {{{{ quote(last_images) }}}}, \
or {{{{ quote(first(last_images)) }}}} for a single file, and "latents: {{{{ quote(last_images) }}}}" \
when the earlier invocation used --image-format pt. They are replaced by the next invocation, so \
save a result for later with \\set name {{{{ quote(first(last_images)) }}}}. Never guess the file \
names dgenerate writes. Every other option of the previous invocation is a last_ variable too, for \
example --prompts {{{{ format_prompt(last_prompts) }}}}. Use only variables and functions listed in \
the template reference.
- The prompts in the examples are short placeholders. Write a new, detailed prompt for the \
request in the style its model understands (see the prompt guide in the reference material), \
keeping everything the user asked for. Text that should appear in the image goes in single \
quotes inside the prompt, like a sign that says 'OPEN'. \
Prompt weighting ((word:1.3), word+) only works with --prompt-weighter. \
Without it those marks are literal characters. --prompt-weighter with a \
plain sentence and no (phrase:1.3) / ((phrase)) / word+ does nothing; \
write those marks on the subjects that matter, or omit the weighter. \
sd-embed is Automatic1111 / CivitAI syntax and the default, including \
SD3. compel is InvokeAI word+ / word++ syntax. Use a weighter when the \
user emphasizes a subject or pastes weighted syntax; do not decorate \
every quality word. Flux, flux-fill, and flux-kontext never take a \
negative prompt: no ; in --prompts.
- Add short # comments explaining the choices that matter. Never repeat the same \
comment. If you are unsure how to finish a step, write the invocation anyway; \
do not stall in a comment loop.
- A .gguf file is only a --transformer (or --unet) replacement. The first line \
of a Flux invocation is still the Hugging Face repo, never the .gguf path.
- last_images is the stills the last step wrote. last_animations is its video \
file. Image models (including Flux Kontext and Flux Fill) and image-to-video \
always take last_images or the user's photo, never last_animations.
- output-file= on a processor only writes a debug image. Do not load that path \
in a later invocation; keep --control-image-processors on the ControlNet step.
- --image-format pt writes latents. Do not follow it with --image-seed-strengths; \
that option is img2img on a PNG/JPEG. Hires fix is generate a small image, then \
img2img larger. A two-step SDXL refine writes latents plus --denoising-end, then \
the refiner repo with --denoising-start. Prefer --sdxl-refiner on the base step.
- --output-size on an upscaler is the size fed in, not the final size. Do not \
shrink a 1024 image to 256 before x4.
- --schedulers help only prints names. Never use it as a generation.
- Reply with exactly one ```dgen code block containing the complete config, and nothing after it.
"""

_CLIP_GUIDE = (
    'Comma separated phrases, most important first: the subject and what it is doing, the setting, '
    'the medium or art style, lighting, color, camera, lens or composition, then a few quality words '
    'like "highly detailed, sharp focus". Keep it under about 60 words unless --prompt-weighter is '
    'set (sd-embed or compel), which also lifts the 77-token CLIP cutoff. To emphasize a subject, '
    'use --prompt-weighter sd-embed and (subject:1.3), or compel and subject+. Without a weighter, '
    'parentheses are plain text. Add a negative prompt with the usual defects, like "blurry, low '
    'quality, deformed, extra fingers, watermark, text".')
_SENTENCE_GUIDE = (
    'Two to four natural sentences that describe the scene in detail, then the style and lighting. '
    'A negative prompt is optional and short.')
_EDIT_GUIDE = (
    'An instruction that states the change and what to keep, like "replace the background with a snowy '
    'forest, keep the person\'s face and pose unchanged". No negative prompt.')

# How to write --prompts for each --model-type, added next to the request for the model types
# of the chosen examples.
PROMPT_GUIDES = {
    'sd': _CLIP_GUIDE,
    'sdxl': _CLIP_GUIDE,
    'kolors': _CLIP_GUIDE + ' Kolors also understands full sentences.',
    's-cascade': _CLIP_GUIDE,
    'sd3': _SENTENCE_GUIDE,
    'if': _SENTENCE_GUIDE,
    'flux': ('Natural sentences describing the subject, its appearance and action, the setting, '
             'composition, lighting, mood, and the style or camera. No negative prompt.'),
    'flux-fill': 'Describe the whole image as it should look, including what fills the masked area. '
                 'No negative prompt.',
    'flux-kontext': _EDIT_GUIDE,
    'pix2pix': _EDIT_GUIDE,
    'sdxl-pix2pix': _EDIT_GUIDE,
    'sd3-pix2pix': _EDIT_GUIDE,
    'ltx': ('One paragraph of three to six plain sentences in the order things happen: the main action '
            'first, then specific movements and gestures, how the people and objects look, the setting, '
            'the camera angle and movement, and the lighting and color. Describe only what fits in the '
            'clip\'s length. --video-lengths is that length in seconds. '
            'LTX-2.5 (Lightricks/LTX-2.5-Diffusers) also makes audio, so end with a sentence '
            'about what is heard, like the rain, footsteps, or music, and put spoken words in single quotes. '
            'A negative prompt is optional, like "worst quality, inconsistent motion, blurry, jittery, '
            'distorted".'),
    'upscaler-x2': 'A short description of what the image shows.',
    'upscaler-x4': 'A short description of what the image shows.',
}

# Always in the context, so the model does not have to guess a repo when the examples use another family.
MODEL_TABLE = """\
### reference: models
The standard Hugging Face repo for each model family. Use the one the request names, and the \
examples for everything else it needs.
- Stable Diffusion 1.5: stable-diffusion-v1-5/stable-diffusion-v1-5, --model-type sd (the default). Public, no HF_TOKEN.
- Stable Diffusion 2.1: sd2-community/stable-diffusion-2-1, --model-type sd. Public, no HF_TOKEN.
- SDXL: stabilityai/stable-diffusion-xl-base-1.0, --model-type sdxl --variant fp16 --dtype float16. \
Public, no HF_TOKEN. Refine in the same invocation with --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0
- SDXL inpainting (needs --image-seeds "image;mask"): diffusers/stable-diffusion-xl-1.0-inpainting-0.1, \
--model-type sdxl --variant fp16 --dtype float16. Public, no HF_TOKEN.
- Stable Diffusion 3 Medium: stabilityai/stable-diffusion-3-medium-diffusers, --model-type sd3 --variant fp16. Gated, needs HF_TOKEN.
- Stable Diffusion 3.5: stabilityai/stable-diffusion-3.5-large or stabilityai/stable-diffusion-3.5-medium, --model-type sd3. Gated, needs HF_TOKEN.
- Stable Cascade: stabilityai/stable-cascade-prior, --model-type s-cascade, \
with --s-cascade-decoder "stabilityai/stable-cascade;dtype=float16". Public, no HF_TOKEN.
- Flux Schnell: black-forest-labs/FLUX.1-schnell, 4 steps, guidance 0, --model-type flux --dtype bfloat16. Public, no HF_TOKEN.
- Flux Dev: black-forest-labs/FLUX.1-dev, --model-type flux --dtype bfloat16. Gated, needs HF_TOKEN.
- Flux inpainting and outpainting: black-forest-labs/FLUX.1-Fill-dev, --model-type flux-fill. Gated, needs HF_TOKEN.
- Flux image editing: black-forest-labs/FLUX.1-Kontext-dev, --model-type flux-kontext. Gated, needs HF_TOKEN.
- Kolors: Kwai-Kolors/Kolors-diffusers, --model-type kolors --variant fp16. Public, no HF_TOKEN.
- DeepFloyd IF: DeepFloyd/IF-I-M-v1.0, --model-type if --variant fp16. Gated, needs HF_TOKEN.
- Animate a still or make a clip: Lightricks/LTX-2.5-Diffusers, --model-type ltx, \
--guidance-scales 1, --model-sequential-offload, --animation-format mp4. Gated, needs HF_TOKEN. \
--video-lengths is seconds. When the user names a duration, use that many seconds, \
however long. When they do not, use 4. Describe the sound in the prompt. \
Lightricks/LTX-Video is the older video-only model; use it only when the user names it.
- 4x upscaling with diffusion: stabilityai/stable-diffusion-x4-upscaler, --model-type upscaler-x4 --variant fp16. Public, no HF_TOKEN.
- 2x latent upscaling: stabilityai/sd-x2-latent-upscaler, --model-type upscaler-x2. Public, no HF_TOKEN.
- Instruction editing: timbrooks/instruct-pix2pix (--model-type pix2pix), \
diffusers/sdxl-instructpix2pix-768 (--model-type sdxl-pix2pix). Public, no HF_TOKEN.
Only use a --variant that the examples use with that repo."""

_MODEL_TYPE = re.compile(r'--model-type\s+"?([\w-]+)')

REPAIR_PROMPT = """\
dgenerate rejected the config:

{problems}

{references}Fix these problems and reply with the complete corrected config in one ```dgen code block."""


def options_in(text: str) -> list[str]:
    seen = []
    for option in _OPTION.findall(text):
        if option not in seen:
            seen.append(option)
    return seen


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + '\n[... truncated]'


def _section(chunk: _corpus.Chunk, limit: int) -> str:
    return f'### {chunk.kind}: {chunk.title}\n{_truncate(chunk.text, limit)}'


@dataclasses.dataclass
class RequestFile:
    written: str
    """The path as the user wrote it."""
    config_path: str
    """The path to put in the config, relative to the directory the config runs from."""
    exists: bool


def request_files(request: str, cwd: str, run_dir: str) -> list[RequestFile]:
    """
    File paths named in the request. Relative paths are taken as relative to
    ``cwd`` and rewritten relative to ``run_dir``. URLs are left out.
    """
    files = []
    for match in _REQUEST_FILE.finditer(request):
        written = next(g for g in match.groups() if g)
        if '://' in written or any(f.written == written for f in files):
            continue
        full = os.path.normpath(os.path.join(cwd, os.path.expanduser(written)))
        if os.path.isabs(written):
            config_path = written
        else:
            try:
                config_path = os.path.relpath(full, run_dir)
            except ValueError:
                config_path = full
            if config_path.replace('\\', '/').startswith('../../../'):
                config_path = full
        files.append(RequestFile(written, config_path.replace('\\', '/'), os.path.exists(full)))
    return files


def apply_request_files(config: str, files: list[RequestFile]) -> str:
    """
    Replace request paths the model copied verbatim with their run directory relative form.
    """
    for f in files:
        if f.written != f.config_path:
            config = re.sub(rf'(?<![\w./\\-]){re.escape(f.written)}(?![\w.-])',
                            lambda _: f.config_path, config)
    return config


def placeholder_media_paths(text: str) -> str:
    """
    Turn ``../../media/file`` paths in examples into ``path/to/file`` so the
    model does not treat the example media as files the user has.
    """
    return _MEDIA_PATH.sub(lambda m: f'path/to/{m.group(1)}', text)


_WEIGHT_REQUEST = re.compile(
    r'prompt[- ]weight|\bsd-embed\b|\bcompel\b|\bweight(?:ed|ing)\b', re.IGNORECASE)
_FLUX_REQUEST = re.compile(r'\bflux\b', re.IGNORECASE)
_COMPEL_REQUEST = re.compile(r'\bcompel\b', re.IGNORECASE)
_GATED_HF = re.compile(
    r'flux\.1-dev|\bflux[ -]?dev\b|flux[ -]?fill|flux\.1-fill|\bkontext\b|'
    r'\bsd3\b|stable diffusion 3|\bltx\b|\bdeepfloyd\b',
    re.IGNORECASE)
_PUBLIC_HF = re.compile(
    r'\bsdxl\b|stable diffusion xl|\bsd\s*1\.?5\b|\bschnell\b|\bkolors\b|'
    r'\bcascade\b|\bpix2pix\b|\bupscal',
    re.IGNORECASE)
_CIVITAI_REQUEST = re.compile(r'\bcivit\s*\.?ai\b|\bcivitai\.com\b', re.IGNORECASE)


def wants_prompt_weighting(request: str) -> bool:
    """Whether the request asked for sd-embed, compel, or prompt weighting."""
    return bool(_WEIGHT_REQUEST.search(request))


def _example_source(chunk: _corpus.Chunk) -> str:
    return (chunk.source or '').replace('\\', '/')


def first_draft_recipes(request: str) -> str:
    """
    Rules placed after the retrieved examples so the first draft does not copy
    their ``\\print`` wording, a Flux negative, or a weighter with no marks.
    """
    lines = [
        '### write the config from these rules',
        'The examples and the manual say "to run this example" because those files are examples.',
        'Do not copy that sentence.',
        'HF_TOKEN is only for a gated checkpoint. A public repo gets no token block, '
        'even when an example for a different repo has one.',
        'Public, no HF_TOKEN: SD 1.5, SD 2.1, SDXL, FLUX.1-schnell, Kolors, Stable Cascade, upscalers, pix2pix.',
        'Gated, HF_TOKEN required: FLUX.1-dev, FLUX.1-Fill-dev, FLUX.1-Kontext-dev, SD3, SD3.5, '
        'LTX-2.5, LTX-Video, DeepFloyd IF.',
        'For a gated repo the \\print is exactly:',
        r'    \print Set HF_TOKEN environmental variable or pass --auth-token.',
        'CIVIT_AI_TOKEN is only for a civitai.com link. A Hugging Face repo or a local file does not use it.',
        'That \\print is exactly: \\print Set CIVIT_AI_TOKEN environmental variable.',
        'Neither token \\print contains the word "example".',
    ]
    if re.search(r'\b(animat\w*|video|ltx|clip)\b', request, re.IGNORECASE):
        lines += [
            'Animate with Lightricks/LTX-2.5-Diffusers, not Lightricks/LTX-Video, unless the user names LTX-Video.',
            '--guidance-scales 1, --model-sequential-offload, --animation-format mp4.',
            '--video-lengths is seconds. If the user names a duration, use that many seconds. '
            'If they do not, use 4. Do not turn a frame count such as 97 into the length unless they asked for 97 seconds.',
            'Image-to-video uses --image-seeds {{ quote(first(last_images)) }} and describes the sound.',
        ]
    if _GATED_HF.search(request):
        lines.append('This request uses a gated checkpoint. Include the HF_TOKEN early-exit.')
    elif _PUBLIC_HF.search(request):
        lines.append('This request uses a public checkpoint. Do not add HF_TOKEN.')
    if _CIVITAI_REQUEST.search(request):
        lines.append('This request downloads from CivitAI. Include the CIVIT_AI_TOKEN early-exit.')
    else:
        lines.append('This request has no CivitAI link. Do not add CIVIT_AI_TOKEN.')
    if _FLUX_REQUEST.search(request):
        lines.append(
            'This request uses Flux. --prompts is the description only: no ";", no negative prompt.')
    if wants_prompt_weighting(request):
        if _COMPEL_REQUEST.search(request):
            lines += [
                'This request asks for compel weighting. Write the marks on this request\'s subjects:',
                '--prompt-weighter compel',
                '--prompts "(main subject)+ , (second subject)++"',
                'word+ and (phrase)+ are compel. A plain sentence with --prompt-weighter does nothing.',
            ]
        else:
            lines += [
                'This request asks for prompt weighting. Write sd-embed marks on this request\'s subjects:',
                '--prompt-weighter sd-embed',
                '--prompts "((requested style)) of a (main subject:1.3), (second subject:1.2), setting"',
                '(phrase:1.3) and ((phrase)) are sd-embed. word+ is compel, not sd-embed. '
                'A plain sentence with --prompt-weighter does nothing.',
            ]
    return '\n'.join(lines)


def prefer_weighting_examples(request: str, examples: list[_corpus.Chunk],
                              index: _index.Index) -> list[_corpus.Chunk]:
    """
    Put a prompt-weighting example first when the request asks for weights,
    so the first draft copies ``(phrase:1.3)`` instead of a plain sentence.
    """
    if not wants_prompt_weighting(request):
        return examples
    already = [c for c in examples if 'prompt_weighting' in _example_source(c)]
    rest = [c for c in examples if 'prompt_weighting' not in _example_source(c)]
    if already:
        return already + rest
    want_flux = 'flux' in request.lower()
    want_compel = bool(re.search(r'\bcompel\b', request, re.IGNORECASE))
    best, best_score = None, -1
    for chunk in index.chunks:
        if chunk.kind != 'example':
            continue
        src = _example_source(chunk)
        if 'prompt_weighting' not in src:
            continue
        score = 0
        if want_flux and 'flux' in src:
            score += 3
        if want_compel and 'compel' in src:
            score += 2
        if not want_compel and 'sd-embed' in src:
            score += 2
        if 'llm4gen' in src:
            score -= 2
        if score > best_score:
            best, best_score = chunk, score
    if best is None:
        return examples
    return [best] + [c for c in rest if _example_source(c) != _example_source(best)][:MAX_EXAMPLES - 1]


def _kind(hint) -> str:
    """
    A short name for a template variable's type.
    """
    args = [a for a in typing.get_args(hint) if a is not type(None)]
    if typing.get_origin(hint) in (typing.Union, types.UnionType) and len(args) == 1:
        hint = args[0]
    origin = typing.get_origin(hint) or hint
    if origin is bool:
        return 'bool'
    if origin is str:
        return 'str'
    if origin in (int, float):
        return 'number'
    if origin is tuple:
        return 'tuple'
    if origin is dict:
        return 'dict'
    if isinstance(origin, type) and issubclass(origin, enum.Enum):
        return 'enum'
    if isinstance(origin, type) and issubclass(origin, collections.abc.Iterable):
        return 'list'
    return 'value'


@functools.cache
def template_reference() -> str:
    """
    Every template variable and function a config can use, read from dgenerate's config runner.
    """
    import dgenerate.batchprocess as _batchprocess

    runner = _batchprocess.ConfigRunner()
    variables = runner._generate_template_variables_with_types()
    last = ', '.join(f'{name} ({_kind(hint)})' for name, (hint, _) in variables.items() if name.startswith('last_'))
    other = ', '.join(name for name in variables if not name.startswith('last_'))
    functions = ', '.join(sorted(runner.template_functions))
    return (
        '### reference: template variables and functions\n'
        'After every invocation, each of its options is available to later lines as a last_ variable. '
        'Options that take several values are lists, unset options are None or empty. '
        'last_images and last_animations list the files it wrote, \\image_process sets them too.\n'
        f'{last}\n\n'
        f'Other variables: {other}\n\n'
        f'Template functions: {functions}\n\n'
        'Common uses:\n'
        '--image-seeds {{ quote(last_images) }}\n'
        '--image-seeds {{ quote(first(last_animations)) }}\n'
        '--prompts {{ format_prompt(last_prompts) }}\n'
        "--seeds {{ last_seeds | join(' ') }} --seeds-to-images (pairs each seed with the image it made, "
        'without --seeds-to-images every seed runs on every image)\n'
        '\\set still {{ quote(first(last_images)) }}\n'
        '\\image_process {{ quote(first(last_images)) }} --output upscaled.png '
        '--processors upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/'
        'realesr-general-x4v3.pth')


def _matching_guides(request: str, hits: list[_index.SearchHit], index: _index.Index) -> list[_corpus.Chunk]:
    """
    Guide sections that match the request, retrieved hits first, then title-word overlap.
    """
    chosen, seen = [], set()
    for chunk in index.chunks:
        if chunk.kind != 'guide' or chunk.id in seen:
            continue
        title = chunk.title.lower()
        if 'first draft' in title or (
                wants_prompt_weighting(request) and 'prompt weighting' in title):
            seen.add(chunk.id)
            chosen.append(chunk)
    for hit in hits:
        if hit.chunk.kind == 'guide' and hit.chunk.id not in seen:
            seen.add(hit.chunk.id)
            chosen.append(hit.chunk)
    query = set(_index.tokenize(request))
    extra = []
    for chunk in index.chunks:
        if chunk.kind != 'guide' or chunk.id in seen:
            continue
        extra.append((len(query & set(_index.tokenize(chunk.title))), chunk))
    extra.sort(key=lambda item: item[0], reverse=True)
    for overlap, chunk in extra:
        if overlap < 2 or len(chosen) >= MAX_GUIDES:
            break
        chosen.append(chunk)
    return chosen[:MAX_GUIDES]


def build_context(request: str, hits: list[_index.SearchHit], index: _index.Index) -> str:
    """
    Pick the closest examples, help for every option and scheduler they and the request use,
    the template reference, and the best matching documentation, within
    :data:`CONTEXT_BUDGET` characters.
    """
    examples, references = [], []
    seen_sources = set()
    for hit in hits:
        chunk = hit.chunk
        if chunk.kind == 'example':
            if chunk.source not in seen_sources and len(examples) < MAX_EXAMPLES:
                seen_sources.add(chunk.source)
                examples.append(chunk)
        elif chunk.kind != 'guide':
            references.append(chunk)

    examples = prefer_weighting_examples(request, examples, index)
    sections = [placeholder_media_paths(_section(c, MAX_EXAMPLE_CHARS)) for c in examples]
    for chunk in _matching_guides(request, hits, index):
        sections.append(_section(chunk, MAX_GUIDE_CHARS))
    sections.append(template_reference())
    used = sum(len(s) for s in sections)

    included = set()
    wanted_options = options_in(request)
    for chunk in examples:
        wanted_options += [o for o in _OPTION_LINE.findall(chunk.text) if o not in wanted_options]
    wanted = [index.argument(o) for o in wanted_options]
    for text in [request] + [c.text for c in examples]:
        wanted += [index.scheduler(name) for name in _SCHEDULER.findall(text)]
    for chunk in wanted:
        if chunk is None or chunk.id in included:
            continue
        section = _section(chunk, MAX_ARGUMENT_CHARS)
        if used + len(section) > CONTEXT_BUDGET * 0.8:
            break
        included.add(chunk.id)
        sections.append(section)
        used += len(section)

    for chunk in references:
        if chunk.id in included:
            continue
        section = _section(chunk, MAX_REFERENCE_CHARS)
        if used + len(section) > CONTEXT_BUDGET:
            continue
        included.add(chunk.id)
        sections.append(section)
        used += len(section)

    sections.append(MODEL_TABLE)

    guide = prompt_guide(examples)
    if guide:
        sections.append(guide)

    return '\n\n'.join(sections)


def prompt_guide(examples: list[_corpus.Chunk]) -> str:
    """
    How to write the prompt for the model types the examples use. Invocations
    without --model-type are Stable Diffusion.
    """
    model_types = []
    for chunk in examples:
        for model_type in _MODEL_TYPE.findall(chunk.text) or ['sd']:
            if model_type in PROMPT_GUIDES and model_type not in model_types:
                model_types.append(model_type)
    if not model_types:
        return ''
    lines = [f'- --model-type {t}: {PROMPT_GUIDES[t]}' for t in model_types]
    return '### reference: prompt guide\nHow to write --prompts for each model type:\n' + '\n'.join(lines)


def user_message(request: str, context: str, files: list[RequestFile]) -> str:
    for f in files:
        request = re.sub(rf'(?<![\w./\\-]){re.escape(f.written)}(?![\w.-])',
                         lambda _: f'"{f.config_path}" (input file)', request)
    return (f'Reference material from the dgenerate documentation and examples:\n\n{context}\n\n'
            f'Request:\n{request}\n\n{first_draft_recipes(request)}')


def inputs_used_as_output(config: str, files: list[RequestFile]) -> list[str]:
    """
    Request input files the model wrote into an output option.
    """
    problems = []
    for match in re.finditer(r'^\s*(-o|-op|--output-path|--output-prefix)\s+(.+)$', config, re.MULTILINE):
        value = match.group(2).strip().strip('"\'')
        for f in files:
            if value in (f.config_path, f.written):
                problems.append(f'{match.group(1)} is set to the input file {value}. Use a new '
                                f'output directory name instead, for example a short name for the result.')
    return problems


_POSITIVE_QUALITY = re.compile(
    r'\b(highly detailed|sharp focus|masterpiece|best quality|high quality|ultra detailed|'
    r'high resolution|[48]k)\b', re.IGNORECASE)


def split_prompts(config: str) -> list[str]:
    """
    Prompt values that put positive text in the negative prompt: more than one ``;``,
    or quality words after the ``;``.
    """
    problems = []
    for match in re.finditer(r'(?<![\w-])(--[\w-]*prompts)[ \t]+(.+)$', config, re.MULTILINE):
        try:
            values = _textprocessing.shell_parse(
                match.group(2), expand_home=False, expand_vars=False, expand_glob=False)
        except _textprocessing.ShellParseSyntaxError:
            values = [match.group(2)]
        for value in values:
            if value.startswith('-'):
                break
            text = re.sub(r'<[^<>]*>', '', value)
            if text.count(';') > 1:
                problems.append(
                    f'{match.group(1)} "{value}" has more than one ;. Only the first ; is the split, '
                    f'so everything after it becomes the negative prompt. Join the positive phrases '
                    f'with commas, and after the one ; keep only what the output must not show.')
                continue
            negative = text.partition(';')[2]
            quality = sorted({m.group(1).lower() for m in _POSITIVE_QUALITY.finditer(negative)})
            if quality:
                problems.append(
                    f'{match.group(1)} "{value}" puts {", ".join(quality)} after the ;, in the negative '
                    f'prompt, so the model avoids them. Move them before the ; and keep only defects '
                    f'like "blurry, low quality" after it.')
    return problems


def system_message(version: str) -> str:
    return SYSTEM_PROMPT.format(shebang=SHEBANG, version=version)


def extract_config(reply: str, version: str) -> str:
    """
    Pull the config out of a model reply and make sure it starts with the shebang lines.
    """
    lines = reply.strip('\n').splitlines()
    # Everything from the first fence to the last, because replies sometimes nest one fence inside another.
    fences = [i for i, line in enumerate(lines) if _FENCE.match(line)]
    if fences:
        lines = lines[fences[0] + 1:fences[-1] if len(fences) > 1 else None]
        lines = [line for line in lines if not _FENCE.match(line)]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and lines[0].startswith('#!'):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return '\n'.join([SHEBANG, f'#! dgenerate {version}', ''] + lines).rstrip() + '\n'


def unknown_options(config: str, known: list[str]) -> list[tuple[str, list[str]]]:
    """
    Option lines whose option is not a dgenerate option, with close matches.
    """
    known_set = set(known)
    result = []
    in_directive = False
    for line in config.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(('#', '{%', '{#')):
            continue
        match = _OPTION_LINE.match(line)
        if match is None:
            # Options continuing a \directive line belong to the directive, which the full check validates.
            in_directive = stripped.startswith('\\')
            continue
        option = match.group(1)
        if in_directive:
            continue
        if option not in known_set and option not in (o for o, _ in result):
            result.append((option, difflib.get_close_matches(option, known, n=3, cutoff=0.6)))
    return result


def repair_message(problems: list[str], suggested: list[str], index: _index.Index) -> str:
    references = []
    for option in suggested:
        chunk = index.argument(option)
        if chunk is not None:
            references.append(_section(chunk, MAX_ARGUMENT_CHARS))
    reference_text = ('Help for options that may be what you meant:\n\n' + '\n\n'.join(references) + '\n\n') \
        if references else ''
    return REPAIR_PROMPT.format(problems='\n'.join(f'- {p}' for p in problems), references=reference_text)
