import dataclasses
import glob
import json
import os
import re

# Chunk sizes are in characters. Qwen tokenizes English RST at roughly 3 to 4 characters per token.
MAX_DOC_CHARS = 2400
MAX_EXAMPLE_CHARS = 9000

_RST_UNDERLINE = re.compile(r'^([=\-~^"\'`#*+.:_])\1{2,}\s*$')

_MEDIA_KINDS = {
    **dict.fromkeys(('png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff'), 'image'),
    **dict.fromkeys(('gif', 'apng', 'mp4', 'webm', 'mov', 'mkv', 'avi'), 'video'),
    **dict.fromkeys(('wav', 'mp3', 'flac', 'ogg'), 'audio'),
}
_MEDIA_FILE = re.compile(
    r'(?:[A-Za-z]:)?[^\s"\';,=<>()]*?([\w.-]+)\.(' + '|'.join(_MEDIA_KINDS) + r')\b', re.IGNORECASE)
_PROMPT_VALUES = re.compile(r'(--[\w-]*prompts)((?:[ \t]+(?:"[^"\n]*"|\'[^\'\n]*\'))+)')
_OUTPUT_VALUES = re.compile(r'(?<![\w-])(--output-path|--output-prefix|-o|-op)[ \t]+("[^"\n]*"|\'[^\'\n]*\'|\S+)')
_OUTPUT_FILE = re.compile(r'output-file=("[^"\n]*"|\'[^\'\n]*\'|[^\s;"\']+)')
_WORD = re.compile(r'[a-z0-9]+')
_INPUT_TECHNIQUE = re.compile(
    r'\b(?:in|out)paint|\bimg2img\b|\bimage[- ]to[- ]image\b|\bimage[- ]seeds?\b|\bcontrol[- ]?nets?\b|'
    r'\bt2i\b|\badapters?\b|\bface[- ]?swap|\bupscal|\bpix2pix\b|\bkontext\b|\bvideo[- ]to[- ]video\b|'
    r'\bextend|\bic[- ]lora\b|\bstyle transfer\b|\bmy (?:image|photo|picture|video|clip)s?\b',
    re.IGNORECASE)


def media_kind(path: str) -> str | None:
    """
    ``image``, ``video``, or ``audio`` for a media file path.
    """
    return _MEDIA_KINDS.get(os.path.splitext(path)[1][1:].lower())


def uses_media(text: str) -> bool:
    """
    Whether text names a media file other than an ``output-file`` path.
    """
    return _MEDIA_FILE.search(_OUTPUT_FILE.sub('', text)) is not None


def wants_inputs(request: str) -> bool:
    """
    Whether a request names input media or a technique that transforms an input.
    """
    return uses_media(request) or _INPUT_TECHNIQUE.search(request) is not None


def strip_media_names(text: str) -> str:
    """
    Replace media file paths with their kind, so a file name does not match on its subject.
    """
    return _MEDIA_FILE.sub(lambda m: _MEDIA_KINDS[m.group(2).lower()], text)


def strip_subjects(text: str) -> tuple[str, set[str]]:
    """
    Remove what an example depicts rather than how it works: prompt text, media
    file names, and output paths. Returns the text and the removed subject words.
    """
    subjects, stems = set(), set()

    def keep_option(m):
        subjects.update(_WORD.findall(m.group(2).lower()))
        return m.group(1)

    def drop(m):
        subjects.update(_WORD.findall(m.group(1).lower()))
        return ''

    def media(m):
        stem = m.group(1).lower()
        stems.add(stem)
        subjects.update(_WORD.findall(stem))
        return _MEDIA_KINDS[m.group(2).lower()]

    text = _PROMPT_VALUES.sub(keep_option, text)
    text = _OUTPUT_FILE.sub(drop, text)
    text = _OUTPUT_VALUES.sub(keep_option, text)
    text = _MEDIA_FILE.sub(media, text)
    for stem in stems:
        text = re.sub(rf'(?<![\w-]){re.escape(stem)}(?![\w-])', '', text, flags=re.IGNORECASE)
    return text, subjects

# console schema file name: (chunk kind, key holding the help text, None when the value is the help text)
_SCHEMAS = {
    'arguments.json': ('argument', None),
    'directives.json': ('directive', None),
    'functions.json': ('function', None),
    'imageprocessors.json': ('image processor', 'PROCESSOR_HELP'),
    'latentsprocessors.json': ('latents processor', 'PROCESSOR_HELP'),
    'promptupscalers.json': ('prompt upscaler', 'PROMPT_UPSCALER_HELP'),
    'promptweighters.json': ('prompt weighter', 'PROMPT_WEIGHTER_HELP'),
    'quantizers.json': ('quantizer', 'QUANTIZER_HELP'),
    'submodels.json': ('sub model URI', 'SUBMODEL_HELP'),
    'karrasschedulers.json': ('scheduler', None),
    'mediaformats.json': ('media formats', None),
}

_MEDIA_FORMAT_LABELS = {
    'images-in': 'Image formats dgenerate reads, for --image-seeds and image processors',
    'images-out': 'Image formats dgenerate writes, for --image-format',
    'videos-in': 'Video and animation formats dgenerate reads, for --image-seeds',
    'videos-out': 'Animation formats dgenerate writes, for --animation-format',
}

_SCHEDULER_NAMES = {
    'DDIMScheduler': 'DDIM',
    'DEISMultistepScheduler': 'DEIS',
    'DPMSolverMultistepScheduler': 'DPM++ 2M, or DPM++ 2M SDE with algorithm-type=sde-dpmsolver++',
    'DPMSolverSDEScheduler': 'DPM++ SDE',
    'DPMSolverSinglestepScheduler': 'DPM++ 2S',
    'EulerAncestralDiscreteScheduler': 'Euler a',
    'EulerDiscreteScheduler': 'Euler',
    'FlowMatchEulerDiscreteScheduler': 'the flow matching scheduler of SD3, Flux, and LTX',
    'HeunDiscreteScheduler': 'Heun',
    'KDPM2AncestralDiscreteScheduler': 'DPM2 a',
    'KDPM2DiscreteScheduler': 'DPM2',
    'LCMScheduler': 'LCM, for LCM LoRAs and LCM UNets',
    'LMSDiscreteScheduler': 'LMS',
    'PNDMScheduler': 'PNDM or PLMS',
    'UniPCMultistepScheduler': 'UniPC',
}

# The manual's copy of --help repeats the option help that arguments.json provides.
_SKIPPED_SECTIONS = {'Help Output'}


@dataclasses.dataclass
class Chunk:
    id: str
    kind: str
    title: str
    source: str
    text: str

    def search_text(self) -> str:
        """
        The text retrieval matches against. Example subjects are left out so a
        request that mentions a subject does not pull in whatever technique the
        example with that subject happens to show.
        """
        text, subjects = strip_subjects(self.text)
        title = self.title
        if self.kind == 'example':
            folder, name = os.path.split(self.source)
            words = [w for w in _WORD.findall(os.path.splitext(name)[0].lower())
                     if w != 'config' and w not in subjects]
            title = ' '.join([folder] + words) + self.title[len(self.source):]
        return f'{self.kind}: {title}\n{text}'


def source_files(repo: str) -> list[str]:
    """
    Every file the corpus is built from, used to tell when the index is stale.
    """
    files = sorted(glob.glob(os.path.join(repo, 'examples', '**', '*.dgen'), recursive=True))
    files += [p for p in (os.path.join(repo, 'docs', 'manual.rst'),
                          os.path.join(repo, 'FEATURE_TABLE.rst'),
                          os.path.join(repo, 'dgenerate', 'assistant', 'data', 'guide.rst'))
              if os.path.isfile(p)]
    schemas = os.path.join(repo, 'dgenerate', 'console', 'schemas')
    files += [os.path.join(schemas, name) for name in _SCHEMAS if os.path.isfile(os.path.join(schemas, name))]
    return files


def _rel(repo: str, path: str) -> str:
    return os.path.relpath(path, repo).replace('\\', '/')


def _split_long(text: str, limit: int) -> list[str]:
    """
    Split on blank lines into parts no longer than ``limit`` where possible.
    """
    if len(text) <= limit:
        return [text]

    parts, current = [], ''
    for paragraph in re.split(r'\n\s*\n', text):
        if current and len(current) + len(paragraph) + 2 > limit:
            parts.append(current)
            current = ''
        current = f'{current}\n\n{paragraph}' if current else paragraph
        while len(current) > limit:
            parts.append(current[:limit])
            current = current[limit:]
    if current.strip():
        parts.append(current)
    return parts


def _example_chunks(repo: str) -> list[Chunk]:
    chunks = []
    for path in sorted(glob.glob(os.path.join(repo, 'examples', '**', '*.dgen'), recursive=True)):
        with open(path, encoding='utf-8', errors='replace') as f:
            text = f.read().strip()
        if not text:
            continue
        rel = _rel(repo, path)
        parts = _split_long(text, MAX_EXAMPLE_CHARS)
        for i, part in enumerate(parts):
            suffix = f' (part {i + 1} of {len(parts)})' if len(parts) > 1 else ''
            chunks.append(Chunk(id=f'example:{rel}:{i}', kind='example', title=rel + suffix, source=rel, text=part))
    return chunks


def _rst_sections(text: str) -> list[tuple[list[str], str]]:
    """
    Split RST into ``(heading path, body)`` pairs. Heading levels follow the
    order in which underline characters first appear, as docutils does.
    """
    lines = text.splitlines()
    levels: list[str] = []
    path: list[str] = []
    sections = []
    body: list[str] = []

    def flush():
        content = '\n'.join(body).strip()
        if content:
            sections.append((list(path), content))
        body.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ''
        underline = _RST_UNDERLINE.match(nxt)
        if line.strip() and not line.startswith(' ') and underline and len(nxt.rstrip()) >= len(line.rstrip()):
            flush()
            char = underline.group(1)
            if char not in levels:
                levels.append(char)
            level = levels.index(char)
            path[:] = path[:level] + [line.strip()]
            i += 2
            continue
        body.append(line)
        i += 1
    flush()
    return sections


def _rst_chunks(repo: str, path: str, kind: str, limit: int = MAX_DOC_CHARS) -> list[Chunk]:
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    rel = _rel(repo, path)
    chunks = []
    for n, (headings, body) in enumerate(_rst_sections(text)):
        if headings and headings[0] in _SKIPPED_SECTIONS:
            continue
        title = ' > '.join(headings) if headings else rel
        for i, part in enumerate(_split_long(body, limit)):
            chunks.append(Chunk(id=f'{kind}:{rel}:{n}:{i}', kind=kind, title=title, source=rel, text=part))
    return chunks


def _uri_arguments(schema: dict) -> dict[str, dict]:
    return {name: spec for name, spec in schema.items() if isinstance(spec, dict) and 'types' in spec}


def _uri_value(value) -> str:
    return f'"{value}"' if isinstance(value, str) else str(value)


def _uri_argument_line(name: str, spec: dict) -> str:
    types = ' | '.join('str' if t.startswith('typing.Literal') else t for t in spec['types'])
    if spec.get('optional'):
        types += ' | None'
    line = f'{name}: {types}'
    if 'default' in spec:
        line += f' = {_uri_value(spec["default"])}'
    if spec.get('options'):
        line += ' (one of: ' + ', '.join(str(o) for o in spec['options']) + ')'
    return line


def _scheduler_text(name: str, schema: dict) -> str:
    arguments = _uri_arguments(schema)
    lines = [f'    {_uri_argument_line(n, s)}' for n, s in arguments.items()]
    about = [f'Also known as {_SCHEDULER_NAMES[name]}.'] if name in _SCHEDULER_NAMES else []
    if 'use-karras-sigmas' in arguments:
        about.append(f'The Karras version is "{name};use-karras-sigmas=true".')
    return (f'{name} for --scheduler or --second-model-scheduler, arguments are optional:\n'
            f'--scheduler "{name};argument=value;argument=value"\n' +
            ''.join(line + '\n' for line in about) +
            'arguments:\n' + '\n'.join(lines))


def _allowed_values(schema: dict) -> str:
    """
    The choices of every URI argument that has a fixed set, which the help text does not always list.
    """
    lines = [f'    {name}: ' + ', '.join(str(o) for o in spec['options'])
             for name, spec in _uri_arguments(schema).items() if spec.get('options')]
    return '\n\nallowed values:\n' + '\n'.join(lines) if lines else ''


def _schema_chunks(repo: str) -> list[Chunk]:
    schemas = os.path.join(repo, 'dgenerate', 'console', 'schemas')
    chunks = []
    for name, (kind, help_key) in _SCHEMAS.items():
        path = os.path.join(schemas, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        rel = _rel(repo, path)
        if kind == 'media formats':
            text = '\n\n'.join(f'{_MEDIA_FORMAT_LABELS.get(key, key)}:\n' + ', '.join(value)
                               for key, value in data.items())
            chunks.append(Chunk(id=f'{kind}:0', kind=kind, title='supported media formats', source=rel, text=text))
            continue
        for key, value in data.items():
            if kind == 'scheduler':
                text = _scheduler_text(key, value)
            elif help_key is None:
                text = value
            else:
                text = value.get(help_key, '')
                if isinstance(text, str) and text.strip():
                    text = text.strip() + _allowed_values(value)
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            if kind == 'argument':
                text = f'{key}\n{text}'
            for i, part in enumerate(_split_long(text, MAX_DOC_CHARS * 2)):
                chunks.append(Chunk(id=f'{kind}:{key}:{i}', kind=kind, title=key, source=rel, text=part))
    return chunks


def collect(repo: str) -> list[Chunk]:
    """
    Build every retrieval chunk for a dgenerate checkout.
    """
    chunks = _example_chunks(repo)
    manual = os.path.join(repo, 'docs', 'manual.rst')
    if os.path.isfile(manual):
        chunks += _rst_chunks(repo, manual, 'manual')
    features = os.path.join(repo, 'FEATURE_TABLE.rst')
    if os.path.isfile(features):
        # A list-table split across chunks loses the header row that names each column.
        chunks += _rst_chunks(repo, features, 'features', limit=MAX_EXAMPLE_CHARS)
    chunks += _schema_chunks(repo)
    guide = os.path.join(repo, 'dgenerate', 'assistant', 'data', 'guide.rst')
    if os.path.isfile(guide):
        chunks += _rst_chunks(repo, guide, 'guide')
    return chunks
