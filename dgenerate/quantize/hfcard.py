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

"""Model card and Hugging Face upload for the quantize sub-command."""

import contextlib
import json
import os
import re

import huggingface_hub

import dgenerate.hfhub as _hfhub
import dgenerate.quantize.quantexport as _quantexport
import dgenerate.resources as _resources

# Copied from quantexport._TEXT_SIDECARS. Those files sit beside a text quant
# and belong in the upload when the quant files are at the output root.
_TEXT_SIDECARS = (
    'config.json',
    'generation_config.json',
    'preprocessor_config.json',
    'processor_config.json',
    'tokenizer_config.json',
    'tokenizer.json',
    'tokenizer.model',
    'special_tokens_map.json',
    'added_tokens.json',
    'vocab.json',
    'merges.txt',
    'chat_template.json',
    'chat_template.jinja',
)

_REPO_ID = re.compile(
    r'^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,95})'
    r'(?:/[A-Za-z0-9](?:[A-Za-z0-9._-]{0,95}))?$')


def normalize_repo_id(repo: str) -> str:
    """
    ``user/name`` from a repository id or a ``huggingface.co`` URL.

    :raises QuantExportError: The value is empty or not a repository id.
    """
    text = (repo or '').strip()
    for prefix in ('https://huggingface.co/', 'http://huggingface.co/', 'huggingface.co/'):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.split('?', 1)[0].split('#', 1)[0].strip('/')
    parts = [part for part in text.split('/') if part]
    if len(parts) >= 2:
        text = parts[0] + '/' + parts[1]
    elif parts:
        text = parts[0]
    else:
        text = ''
    if not _REPO_ID.match(text):
        raise _quantexport.QuantExportError(
            f'"{repo}" is not a Hugging Face repository id. '
            f'Pass user/name, or a huggingface.co URL for that repository.')
    return text


def quantize_command_line(model: str, *,
                          subfolders: list[str] | None = None,
                          formats: list[str] | None = None,
                          quantizer_uris: list[str] | None = None,
                          comfy: bool = False,
                          llama: bool = False,
                          output: str | None = None,
                          weight_name: str | None = None,
                          variant: str | None = None,
                          revision: str | None = None,
                          hf_upload: str | None = None,
                          hf_upload_card: str | None = None,
                          hf_private: bool = False,
                          offline: bool = False) -> str:
    """The quantize invocation, without a token and without ``--hf-upload-card-preview``."""
    parts = ['dgenerate', '--sub-command', 'quantize', _bash_quote(model)]
    if subfolders:
        parts.append('--subfolders')
        parts.extend(_bash_quote(name) for name in subfolders)
    if formats:
        parts.append('--formats')
        parts.extend(_bash_quote(name) for name in formats)
    if comfy:
        parts.append('--comfy')
    if llama:
        parts.append('--llama')
    if output:
        parts.extend(['--output', _bash_quote(output)])
    if quantizer_uris:
        parts.append('--quantizer')
        parts.extend(_bash_quote(uri) for uri in quantizer_uris)
    if weight_name:
        parts.extend(['--weight-name', _bash_quote(weight_name)])
    if variant:
        parts.extend(['--variant', _bash_quote(variant)])
    if revision:
        parts.extend(['--revision', _bash_quote(revision)])
    if offline:
        parts.append('--offline-mode')
    if hf_upload:
        parts.extend(['--hf-upload', _bash_quote(hf_upload)])
    if hf_upload_card is not None:
        # An empty value, and the old generate keyword, are the flag alone.
        if hf_upload_card.strip().lower() in {'', 'generate'}:
            parts.append('--hf-upload-card')
        else:
            parts.extend(['--hf-upload-card', _bash_quote(hf_upload_card)])
    if hf_private:
        parts.append('--hf-private')
    return ' '.join(parts)


class UploadCard:
    """One ``--hf-upload-card`` value."""

    def __init__(self, kind: str, value: str | None = None):
        self.kind = kind
        self.value = value


def parse_upload_card(text: str) -> UploadCard:
    """
    An empty value builds the card. ``file:PATH``, ``text:TEXT``, ``extra:TEXT``,
    or ``extra-file:PATH`` select another card. ``generate`` is the empty value.

    ``extra:`` and ``extra-file:`` keep the generated card and insert that text
    after the introduction.

    :raises QuantExportError: The value is not one of those forms.
    """
    forms = 'file:PATH, text:TEXT, extra:TEXT, or extra-file:PATH'
    if not isinstance(text, str):
        raise _quantexport.QuantExportError(f'--hf-upload-card must be {forms}, or have no value.')
    if not text.strip() or text.strip().lower() == 'generate':
        return UploadCard('generate')
    body = text.lstrip()
    lowered = body.lower()
    for kind, prefix in (
            ('extra-file', 'extra-file:'),
            ('extra', 'extra:'),
            ('file', 'file:'),
            ('text', 'text:')):
        if not lowered.startswith(prefix):
            continue
        value = body[len(prefix):]
        if kind in {'file', 'extra-file'}:
            value = value.strip()
        if not value.strip():
            raise _quantexport.QuantExportError(
                f'--hf-upload-card {prefix} needs a value after the colon.')
        return UploadCard(kind, value)
    raise _quantexport.QuantExportError(
        f'--hf-upload-card must be {forms}, or have no value.')


def read_card_text(card: UploadCard) -> str:
    """UTF-8 file text, or the text after ``text:`` or ``extra:``."""
    if card.kind in {'file', 'extra-file'}:
        try:
            with open(card.value, encoding='utf-8') as handle:
                body = handle.read()
        except UnicodeDecodeError as error:
            raise _quantexport.QuantExportError(
                f'"{card.value}" is not UTF-8 text: {error}') from error
        except OSError as error:
            raise _quantexport.QuantExportError(
                f'Could not read "{card.value}": {error}') from error
    else:
        body = card.value or ''
    if body and not body.endswith('\n'):
        body += '\n'
    return body


def read_provided_card(card: UploadCard) -> str:
    """The README text for ``file:`` or ``text:``."""
    return read_card_text(card)


def component_class(config: dict | None) -> str | None:
    """``_class_name``, the first architecture, or ``model_type``."""
    if not isinstance(config, dict):
        return None
    name = config.get('_class_name')
    if name:
        return str(name)
    architectures = config.get('architectures') or []
    if architectures:
        return str(architectures[0])
    model_type = config.get('model_type')
    return str(model_type) if model_type else None


def pipeline_class(directory: str | None) -> str | None:
    """``_class_name`` from ``model_index.json``, when that file is present."""
    if not directory:
        return None
    path = os.path.join(directory, 'model_index.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    name = data.get('_class_name') if isinstance(data, dict) else None
    return str(name) if name else None


def upload_layout(output_dir: str,
                  written: list[str],
                  include_card: bool = False) -> tuple[str, list[str] | None]:
    """
    ``(folder, allow_patterns)`` for the Hub upload.

    One quantizer repository and no loose files uploads that folder, so
    ``model_index.json`` is at the repository root. Anything else uploads
    the output directory, limited to the files this run wrote. ``README.md``
    is included only when a model card was requested.
    """
    directories = [path for path in written if os.path.isdir(path)]
    files = [path for path in written if os.path.isfile(path)]
    if len(directories) == 1 and not files:
        return directories[0], None
    patterns = ['README.md'] if include_card else []
    for path in written:
        relative = os.path.relpath(path, output_dir).replace('\\', '/')
        if os.path.isdir(path):
            patterns.append(relative + '/**')
        else:
            patterns.append(relative)
    if any(os.path.dirname(os.path.abspath(path)) == os.path.abspath(output_dir) for path in files):
        for name in _TEXT_SIDECARS:
            if os.path.isfile(os.path.join(output_dir, name)) and name not in patterns:
                patterns.append(name)
    return output_dir, patterns


def listed_files(root: str, patterns: list[str] | None) -> list[tuple[str, int]]:
    """Relative paths and sizes that an upload of ``root`` would send, excluding the model card."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in {'.git', '__pycache__', '.cache'}]
        for name in filenames:
            if name == 'README.md' or name.endswith('.pyc'):
                continue
            path = os.path.join(dirpath, name)
            relative = os.path.relpath(path, root).replace('\\', '/')
            if patterns is not None and not any(_pattern_matches(relative, pattern) for pattern in patterns):
                continue
            found.append((relative, os.path.getsize(path)))
    found.sort()
    return found


def build_model_card(*,
                     model: str,
                     repo_id: str | None,
                     revision: str | None,
                     subfolders: list[str] | None,
                     formats_requested: list[str] | None,
                     formats: list[str],
                     quantizers: list,
                     comfy: bool,
                     llama: bool,
                     variant: str | None,
                     weight_name: str | None,
                     command: str,
                     component_classes: list[tuple[str, str | None]] | None = None,
                     pipeline: str | None = None,
                     files: list[tuple[str, int]] | None = None,
                     extra: str | None = None,
                     preview: bool = False) -> str:
    """README.md for the quantized files. A preview has no measured file sizes."""
    source = _source(model)
    if pipeline is None and preview and os.path.isdir(model):
        pipeline = pipeline_class(model)
    if component_classes is None and preview:
        component_classes = _local_component_classes(model, subfolders)

    lines = ['---']
    if source.repo_id:
        lines.append(f'base_model: {_yaml(source.repo_id)}')
        lines.append('base_model_relation: quantized')
    library = _library_name(formats, quantizers)
    if library:
        lines.append(f'library_name: {library}')
    lines.append('tags:')
    for tag in _tags(formats, quantizers, library):
        lines.append(f'- {tag}')
    lines.append('---')
    lines.append('')
    title = repo_id or _title_from_model(source)
    lines.append(f'# {title}')
    lines.append('')
    lines.append(_lead(source, formats, quantizers))
    lines.append('')
    lines.append('The license of the source model applies to these weights.')
    lines.append('')
    if extra and extra.strip():
        lines.append(extra.strip())
        lines.append('')
    lines.extend(_source_section(
        source, revision, pipeline, component_classes, subfolders,
        formats_requested, variant, weight_name))
    lines.extend(_quant_section(formats, quantizers, comfy, llama, formats_requested))
    lines.append('## Command')
    lines.append('')
    lines.append('```bash')
    lines.append(command)
    lines.append('```')
    lines.append('')
    lines.extend(_files_section(formats, quantizers, comfy, llama, subfolders, files))
    return '\n'.join(lines).rstrip() + '\n'


def commit_message(model: str,
                   formats: list[str],
                   quantizers: list,
                   components: list[str]) -> str:
    """One-line Hub commit message. It does not include a token."""
    bits = []
    if formats:
        bits.append(', '.join(formats))
    bits.extend(spec.folder for spec in quantizers)
    what = ', '.join(bits) if bits else 'quant'
    target = ', '.join(components) if components else 'selected components'
    source = model if len(model) < 80 else os.path.basename(model.rstrip('/\\'))
    message = f'dgenerate quantize: {what} of {target} from {source}'
    if len(message) > 240:
        return message[:237] + '...'
    return message


def _operation_bytes(operation) -> int:
    """Bytes of one file the upload will send, from the local path or buffer."""
    source = getattr(operation, 'path_or_fileobj', None)
    if isinstance(source, (bytes, bytearray)):
        return len(source)
    if isinstance(source, (str, os.PathLike)):
        try:
            return os.path.getsize(source)
        except OSError:
            return 0
    return 0


def _attach_upload_bar(pipeline) -> None:
    """Follow the Hub upload with a tqdm bar.

    ``huggingface_hub`` draws this upload with its own stderr renderer, which
    only runs when stderr is a terminal and does not go through dgenerate's
    tqdm patch. A normal tqdm bar does. ``DGENERATE_FORCE_TQDM`` defaults to
    on, so the bar shows in an IDE terminal.
    """
    display = pipeline.display
    if huggingface_hub.utils.are_progress_bars_disabled():
        return
    from tqdm.auto import tqdm
    # The Hub renderer and the tqdm bar would fight over the same lines.
    display._tty = False
    total = sum(_operation_bytes(operation) for operation in pipeline.add_operations)
    bar = tqdm(
        total=total or None,
        desc='Uploading',
        unit='B',
        unit_scale=True,
        unit_divisor=1000,
        dynamic_ncols=True)
    # The callback that fills _xet_bytes is skipped when the display is inactive.
    display._active = True
    display._dgenerate_bar = bar

    def sync():
        with display._lock:
            done = display._xet_bytes
        if bar.total and done > bar.total:
            bar.total = int(done * 1.25) + 1
        step = done - (bar.n or 0)
        if step > 0:
            bar.update(step)

    def render():
        while not display._stop_event.wait(0.2):
            sync()

    def close():
        original_close()
        with display._lock:
            done = display._xet_bytes
        if bar.total and done > (bar.n or 0):
            bar.total = max(bar.total, done)
            bar.update(done - (bar.n or 0))
        if (not pipeline.abort_event.is_set() and bar.total
                and (bar.n or 0) < bar.total):
            bar.update(bar.total - (bar.n or 0))
        bar.close()

    original_close = display.close
    display._sync_upload_bar = sync
    display._render_loop = render
    display.close = close


@contextlib.contextmanager
def _show_upload_progress():
    """Attach the tqdm bar to Hub folder uploads made inside this block."""
    import huggingface_hub._upload_pipeline as pipeline
    original = pipeline._UploadPipeline.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _attach_upload_bar(self)

    pipeline._UploadPipeline.__init__ = init
    try:
        yield
    finally:
        pipeline._UploadPipeline.__init__ = original


def upload_quantized(folder: str,
                     repo_id: str,
                     token: str,
                     *,
                     private: bool,
                     commit_message_text: str,
                     allow_patterns: list[str] | None) -> str:
    """
    Create the model repository when needed and upload ``folder``.

    :return: ``https://huggingface.co/{repo_id}``
    :raises QuantExportError: The Hub refused the repository or the upload.
    """
    try:
        api = huggingface_hub.HfApi(token=token)
        api.create_repo(
            repo_id=repo_id,
            repo_type='model',
            private=True if private else None,
            exist_ok=True)
        with _show_upload_progress():
            api.upload_folder(
                folder_path=folder,
                repo_id=repo_id,
                repo_type='model',
                commit_message=commit_message_text,
                allow_patterns=allow_patterns,
                ignore_patterns=['**/__pycache__/**', '**/*.pyc', '**/.cache/**'])
    except Exception as error:
        message = str(error)
        if token and token in message:
            message = message.replace(token, '***')
        raise _quantexport.QuantExportError(
            f'Could not upload to "{repo_id}": {message}') from error
    return f'https://huggingface.co/{repo_id}'


class _Source:
    def __init__(self, label: str, repo_id: str | None, blob: _hfhub.HFBlobLink | None):
        self.label = label
        self.repo_id = repo_id
        self.blob = blob


def _source(model: str) -> _Source:
    blob = _hfhub.HFBlobLink.parse(model)
    if blob is not None:
        return _Source(blob.repo_id, blob.repo_id, blob)
    if _looks_like_repo_id(model) and not os.path.exists(model):
        return _Source(model, model, None)
    return _Source(model, None, None)


def _looks_like_repo_id(model: str) -> bool:
    if os.path.isabs(model) or '\\' in model or model.startswith('.'):
        return False
    return _REPO_ID.match(model) is not None and '/' in model


def _bash_quote(value: str) -> str:
    if value and all(character.isalnum() or character in '._-/:=+@' for character in value) \
            and not value.startswith('-'):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _yaml(value: str) -> str:
    if value and all(character.isalnum() or character in '._/-' for character in value):
        return value
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def _format_kind(fmt: str) -> str:
    if fmt in set(_quantexport.gguf_format_names()):
        return 'GGUF'
    if fmt in set(_quantexport.comfy_format_names()):
        return 'ComfyUI safetensors quant'
    return 'safetensors, every tensor stored in this dtype'


def _library_name(formats: list[str], quantizers: list) -> str | None:
    kinds = set()
    if quantizers:
        kinds.add('diffusers')
    for fmt in formats:
        kinds.add('gguf' if _format_kind(fmt) == 'GGUF' else 'other')
    if kinds == {'diffusers'}:
        return 'diffusers'
    if kinds == {'gguf'}:
        return 'gguf'
    return None


def _tags(formats: list[str], quantizers: list, library: str | None) -> list[str]:
    tags = ['dgenerate', 'quantization']
    if any(_format_kind(fmt) == 'GGUF' for fmt in formats):
        tags.append('gguf')
    if any(_format_kind(fmt) == 'ComfyUI safetensors quant' for fmt in formats):
        tags.append('comfyui')
    if any(spec.backend == 'sdnq' for spec in quantizers):
        tags.append('sdnq')
    if any(spec.backend == 'bnb' for spec in quantizers):
        tags.append('bitsandbytes')
    if library == 'diffusers' or any(spec.backend for spec in quantizers):
        if 'diffusers' not in tags:
            tags.append('diffusers')
    return tags


def _title_from_model(source: _Source) -> str:
    if source.repo_id:
        return source.repo_id.split('/')[-1]
    name = os.path.basename(source.label.rstrip('/\\')) or source.label
    return name


def _lead(source: _Source, formats: list[str], quantizers: list) -> str:
    parts = []
    gguf = [fmt for fmt in formats if _format_kind(fmt) == 'GGUF']
    comfy = [fmt for fmt in formats if _format_kind(fmt) == 'ComfyUI safetensors quant']
    plain = [fmt for fmt in formats if fmt not in gguf and fmt not in comfy]
    if gguf:
        parts.append('GGUF ' + _group(gguf))
    if comfy:
        parts.append('ComfyUI safetensors ' + _group(comfy))
    if plain:
        parts.append(_group(plain) + ' safetensors')
    for spec in quantizers:
        if spec.backend == 'sdnq':
            parts.append(f'SDNQ {spec.parsed.type}')
        else:
            parts.append(f'bitsandbytes {spec.parsed.bits}-bit')
    produced = _and_list(parts) if parts else 'A'
    target = _model_link(source)
    version = _resources.__version__
    return (
        f'{produced} quantization of {target}, written by '
        f'[dgenerate](https://github.com/Teriks/dgenerate) {version}.'
    )


def _group(names: list[str]) -> str:
    if len(names) > 4:
        return f'{len(names)} formats ({names[0]} through {names[-1]})'
    return _and_list(names)


def _and_list(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f'{parts[0]} and {parts[1]}'
    return ', '.join(parts[:-1]) + ', and ' + parts[-1]


def _model_link(source: _Source) -> str:
    if source.repo_id:
        return f'[{source.repo_id}](https://huggingface.co/{source.repo_id})'
    return f'`{source.label}`'


def _source_section(source, revision, pipeline, component_classes, subfolders,
                    formats_requested, variant, weight_name) -> list[str]:
    rows = [('Model', f'`{source.label}`')]
    if source.blob is not None:
        rows.append(('File', f'`{source.blob.subfolder + "/" if source.blob.subfolder else ""}'
                             f'{source.blob.weight_name}`'))
        if not revision:
            revision = source.blob.revision
    if revision:
        rows.append(('Revision', f'`{revision}`'))
    if pipeline:
        rows.append(('Pipeline', f'`{pipeline}`'))
    if component_classes:
        shown = []
        for name, klass in component_classes:
            shown.append(f'`{name}` (`{klass}`)' if klass else f'`{name}`')
        rows.append(('Components', ', '.join(shown)))
    elif subfolders:
        rows.append(('Components', ', '.join(f'`{name}`' for name in subfolders)))
    else:
        rows.append(('Components', 'every component that has weights'))
    if formats_requested:
        rows.append(('Formats', ', '.join(f'`{name}`' for name in formats_requested)))
    if variant:
        rows.append(('Variant', f'`{variant}`'))
    if weight_name:
        rows.append(('Weight file', f'`{weight_name}`'))
    lines = ['## Source', '', '| | |', '| --- | --- |']
    for label, value in rows:
        lines.append(f'| {label} | {value} |')
    lines.append('')
    return lines


def _quant_section(formats, quantizers, comfy, llama, formats_requested) -> list[str]:
    lines = ['## Quantization', '']
    if quantizers and not formats:
        lines.append(
            'Load the diffusers folder and do not pass `--quantizer`. '
            'The weights are already quantized. Components that were not selected '
            'are copied at full precision, including `model_index.json`.')
    elif quantizers and formats:
        lines.append(
            'An SDNQ or bitsandbytes folder loads with `--quantizer` left unset. '
            'A GGUF or Comfy safetensors file keeps the original repository as the model '
            'and is passed with the matching component option.')
    else:
        lines.append(
            'Keep the original repository as the model and pass the quant file '
            'with the matching component option, so the config still comes from the repository.')
    lines.append('')
    if comfy and quantizers:
        lines.append(
            'GGUF and Comfy safetensors files use ComfyUI tensor names (`--comfy`). '
            'SDNQ and bitsandbytes repositories keep diffusers tensor names.')
    elif llama and quantizers:
        lines.append(
            'GGUF files use llama.cpp tensor names (`--llama`). '
            'SDNQ and bitsandbytes repositories keep diffusers tensor names.')
    elif comfy:
        lines.append(
            'Tensor names follow the ComfyUI layout (`--comfy`). '
            'ComfyUI and ComfyUI-GGUF load the files that have a key map for that architecture. '
            'A component with no key map is an error.')
    elif llama:
        lines.append(
            'Tensor names follow the llama.cpp layout (`--llama`), '
            'the layout Unsloth text-encoder GGUFs use. dgenerate loads that layout.')
    else:
        lines.append('Tensor names are the ones the component stored.')
    lines.append('')
    if formats:
        lines.append('| Format | File |')
        lines.append('| --- | --- |')
        for fmt in formats:
            lines.append(f'| `{fmt}` | {_format_kind(fmt)} |')
        lines.append('')
        if formats_requested and list(formats_requested) != list(formats):
            lines.append(
                '`--formats` accepts short names and presets. '
                '`q4` is Q4_K. `q8` is Q8_0. `fp8` is float8_e4m3fn.')
            lines.append('')
    if any(spec.backend == 'sdnq' for spec in quantizers):
        lines.append(
            'SDNQ `type` is the weight dtype. `group-size` 0 uses one scale per tensor. '
            'A group size above 0 shares one scale across that many elements. '
            '`use-svd` keeps an SVDQuant low-rank residual. '
            '`quant-conv` quantizes convolutional layers. '
            '`quantized-matmul` and `quantized-matmul-conv` use quantized matrix multiplies.')
        lines.append('')
    if any(spec.backend == 'bnb' for spec in quantizers):
        lines.append(
            'bitsandbytes `bits` is 4 or 8. 8-bit uses LLM.int8(). '
            '4-bit uses `fp4` or `nf4`. `bits4-use-double-quant` quantizes the scales again. '
            '`bits4-compute-dtype` is the compute dtype and `bits4-quant-storage` is the packed storage dtype. '
            '`default` means the URI left that argument unset. '
            'bitsandbytes quantization runs on CUDA.')
        lines.append('')
    for spec in quantizers:
        lines.append(f'### `{spec.folder}`')
        lines.append('')
        lines.append(f'URI: `{spec.uri}`')
        lines.append('')
        lines.append('| Setting | Value |')
        lines.append('| --- | --- |')
        for setting, value in _quantizer_settings(spec):
            lines.append(f'| `{setting}` | `{value}` |')
        lines.append('')
    return lines


def _quantizer_settings(spec) -> list[tuple[str, str]]:
    parsed = spec.parsed
    if spec.backend == 'sdnq':
        return [
            ('backend', 'sdnq'),
            ('type', parsed.type),
            ('group-size', str(parsed.group_size)),
            ('quant-conv', _bool_text(parsed.quant_conv)),
            ('quantized-matmul', _bool_text(parsed.quantized_matmul)),
            ('quantized-matmul-conv', _bool_text(parsed.quantized_matmul_conv)),
            ('use-svd', _bool_text(parsed.use_svd)),
        ]
    rows = [
        ('backend', 'bitsandbytes'),
        ('bits', str(parsed.bits)),
    ]
    if parsed.bits == 4:
        rows.extend([
            ('bits4-quant-type', parsed.bits4_quant_type),
            ('bits4-use-double-quant', _bool_text(parsed.bits4_use_double_quant)),
            ('bits4-compute-dtype', parsed.bits4_compute_dtype or 'default'),
            ('bits4-quant-storage', parsed.bits4_quant_storage or 'default'),
        ])
    return rows


def _bool_text(value: bool) -> str:
    return 'true' if value else 'false'


def _files_section(formats, quantizers, comfy, llama, subfolders, files) -> list[str]:
    lines = ['## Files', '']
    if files is not None:
        if not files:
            lines.append('No files were listed.')
            lines.append('')
            return lines
        lines.append('| Path | Size |')
        lines.append('| --- | --- |')
        for relative, size in files:
            lines.append(f'| `{relative}` | {_size_text(size)} |')
        lines.append('')
        return lines
    predicted = _predicted_files(formats, quantizers, comfy, llama, subfolders)
    if predicted:
        for name in predicted:
            lines.append(f'- `{name}`')
        lines.append('')
    if not subfolders and formats:
        lines.append('`{component}` is the component folder, such as `transformer` or `text_encoder`.')
        lines.append('')
    return lines


def _predicted_files(formats, quantizers, comfy, llama, subfolders) -> list[str]:
    marker = '-comfy' if comfy else '-llama' if llama else ''
    names = []
    components = list(subfolders) if subfolders else ['{component}']
    gguf_names = set(_quantexport.gguf_format_names())
    if formats:
        for component in components:
            safe = component.replace('\\', '/').replace('/', '-').replace(' ', '_')
            for fmt in formats:
                suffix = '.gguf' if fmt in gguf_names else '.safetensors'
                names.append(f'{safe}-{fmt}{marker}{suffix}')
    for spec in quantizers:
        names.append(spec.folder + '/')
    return names


def _local_component_classes(model: str,
                             subfolders: list[str] | None) -> list[tuple[str, str | None]]:
    if not subfolders or not os.path.isdir(model):
        return []
    rows = []
    for name in subfolders:
        path = os.path.join(model, name, 'config.json')
        config = None
        if os.path.isfile(path):
            try:
                with open(path, encoding='utf-8') as handle:
                    config = json.load(handle)
            except (OSError, json.JSONDecodeError):
                config = None
        rows.append((name, component_class(config)))
    return rows


def _size_text(size: int) -> str:
    value = float(size)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1024 or unit == 'TB':
            if unit == 'B':
                return f'{int(value)} B'
            return f'{value:.1f} {unit}'
        value /= 1024
    return f'{size} B'


def _pattern_matches(relative: str, pattern: str) -> bool:
    if pattern.endswith('/**'):
        prefix = pattern[:-3]
        return relative == prefix or relative.startswith(prefix + '/')
    return relative == pattern
