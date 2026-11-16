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

import json
import os

import huggingface_hub

import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.batchprocess.util as _b_util
import dgenerate.quantize.quantexport as _quantexport
import dgenerate.quantize.repoquant as _repoquant
import dgenerate.subcommands.subcommand as _subcommand


def _acquire_repository(model: str,
                        subfolders: list[str] | None,
                        revision: str | None,
                        token: str | None,
                        local_files_only: bool,
                        weight_name: str | None = None) -> str:
    """A local diffusers folder, downloading a Hub repository into the cache when needed."""
    if os.path.isdir(model):
        return os.path.abspath(model)
    if os.path.isfile(model):
        raise _quantexport.QuantExportError(
            f'"{model}" is not a weight file. Pass a .safetensors, .bin, .pt, .pth, '
            f'or .ckpt file, or a diffusers folder.')
    if os.path.exists(model):
        raise _quantexport.QuantExportError(f'"{model}" is not a diffusers folder.')

    patterns = ['model_index.json']
    if subfolders:
        for name in subfolders:
            prefix = name.strip('/\\').replace('\\', '/')
            patterns.extend([
                f'{prefix}/config.json',
                f'{prefix}/*index.json',
                f'{prefix}/*.safetensors',
                f'{prefix}/*.bin',
                f'{prefix}/*.pt',
                f'{prefix}/*.pth',
                f'{prefix}/*.ckpt',
            ])
    else:
        patterns.extend([
            '**/config.json',
            '**/*index.json',
            '**/*.safetensors',
            '**/*.bin',
            '**/*.pt',
            '**/*.pth',
            '**/*.ckpt',
        ])
    if weight_name:
        relative = weight_name.replace('\\', '/').lstrip('/')
        patterns.append(relative)
        patterns.append(f'**/{relative}')
    try:
        return huggingface_hub.snapshot_download(
            repo_id=model,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=patterns)
    except (OSError,
            huggingface_hub.errors.HfHubHTTPError,
            huggingface_hub.errors.HFValidationError,
            huggingface_hub.errors.LocalEntryNotFoundError,
            huggingface_hub.errors.OfflineModeIsEnabled,
            huggingface_hub.errors.EntryNotFoundError) as error:
        raise _quantexport.QuantExportError(f'Could not read "{model}": {error}') from error


def _config_beside_weight(path: str,
                          blob: _hfhub.HFBlobLink | None,
                          token: str | None,
                          local_files_only: bool) -> dict | None:
    """``config.json`` next to a single weight file, including one fetched for a blob link."""
    beside = os.path.join(os.path.dirname(path), 'config.json')
    if os.path.isfile(beside):
        with open(beside, encoding='utf-8') as handle:
            return json.load(handle)
    if blob is None:
        return None
    try:
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=blob.repo_id,
            filename='config.json',
            subfolder=blob.subfolder or None,
            revision=blob.revision,
            local_files_only=local_files_only,
            token=token)
    except (OSError,
            huggingface_hub.errors.HfHubHTTPError,
            huggingface_hub.errors.HFValidationError,
            huggingface_hub.errors.LocalEntryNotFoundError,
            huggingface_hub.errors.OfflineModeIsEnabled,
            huggingface_hub.errors.EntryNotFoundError,
            huggingface_hub.errors.RepositoryNotFoundError,
            huggingface_hub.errors.RevisionNotFoundError):
        return None
    with open(downloaded, encoding='utf-8') as handle:
        return json.load(handle)


def _ensure_index_shards(path: str,
                         blob: _hfhub.HFBlobLink | None,
                         token: str | None,
                         local_files_only: bool) -> None:
    """Download the shards named by an index when the source is a blob link."""
    if not path.endswith('.index.json'):
        return
    with open(path, encoding='utf-8') as handle:
        weight_map = json.load(handle).get('weight_map') or {}
    folder = os.path.dirname(path)
    missing = []
    for shard in dict.fromkeys(weight_map.values()):
        if not os.path.isfile(os.path.join(folder, shard)):
            missing.append(shard)
    if not missing:
        return
    if blob is None:
        raise _quantexport.QuantExportError(f'"{path}" is missing shard "{missing[0]}".')
    try:
        for shard in missing:
            huggingface_hub.hf_hub_download(
                repo_id=blob.repo_id,
                filename=shard,
                subfolder=blob.subfolder or None,
                revision=blob.revision,
                local_files_only=local_files_only,
                token=token)
    except (OSError,
            huggingface_hub.errors.HfHubHTTPError,
            huggingface_hub.errors.HFValidationError,
            huggingface_hub.errors.LocalEntryNotFoundError,
            huggingface_hub.errors.OfflineModeIsEnabled,
            huggingface_hub.errors.EntryNotFoundError) as error:
        raise _quantexport.QuantExportError(
            f'Could not read shard "{missing[0]}" for "{path}": {error}') from error


def _acquire_single_weight(model: str,
                           token: str | None,
                           local_files_only: bool) -> tuple[str, str, dict | None]:
    """``(path, component name, config)`` for one weight file or blob link."""
    blob = _hfhub.HFBlobLink.parse(model)
    try:
        if model.lower().startswith(('http://', 'https://')):
            path = _hfhub.webcache_or_hf_blob_download(
                model, local_files_only=local_files_only, auth_token=token)
        else:
            path = os.path.abspath(model)
    except (OSError,
            huggingface_hub.errors.HfHubHTTPError,
            huggingface_hub.errors.HFValidationError,
            huggingface_hub.errors.LocalEntryNotFoundError,
            huggingface_hub.errors.OfflineModeIsEnabled,
            huggingface_hub.errors.EntryNotFoundError) as error:
        raise _quantexport.QuantExportError(f'Could not read "{model}": {error}') from error
    _ensure_index_shards(path, blob, token, local_files_only)
    name = _quantexport.component_name_for_file(
        path, blob.subfolder if blob is not None else None)
    config = _config_beside_weight(path, blob, token, local_files_only)
    return path, name, config


class QuantizeSubCommand(_subcommand.SubCommand):
    """
    Quantize a diffusers model into GGUF files, ComfyUI safetensors quants, and
    SDNQ or bitsandbytes repositories.

    List every format you want. Each one is written. --list-formats prints the
    GGUF types, Comfy safetensors quants, plain float16 and float32, short names,
    presets, and example file names. q4 means Q4_K. Q4_0 is requested by that name.

    The source is a diffusers folder, a Hugging Face repository, one weight file,
    or a Hugging Face blob link to one weight file. --subfolders picks component
    folders such as transformer and text_encoder. Without it, every component
    that has weights is quantized. --weight-name selects one file in that
    folder, such as model.fp16.safetensors.

    Tensor names stay as that component stored them, unless --comfy is set. --comfy
    writes the names ComfyUI and ComfyUI-GGUF load, for FLUX.1, SD3, Wan,
    Qwen-Image 2.1, Z-Image safetensors, and the GGUF text encoders T5, UMT5,
    Llama 3 / Mistral, Qwen2, Qwen3, and Gemma 3. Those files are named like
    transformer-Q4_K-comfy.gguf.

    Load a diffusers-named file with the matching component option and keep the
    original repository as the model:

    dgenerate --sub-command quantize black-forest-labs/FLUX.1-dev --subfolders transformer --formats q4 q8 fp8 -o quants

    dgenerate --sub-command quantize ./model --formats dgenerate comfy -o quants

    dgenerate --sub-command quantize --list-formats

    See: dgenerate --sub-command quantize --help
    """

    NAMES = ['quantize']

    def __init__(self, program_name='quantize', **kwargs):
        super().__init__(**kwargs)
        self._parser = _b_util.DirectiveArgumentParser(
            prog=program_name,
            description=(
                'Quantize a diffusers model into GGUF files, ComfyUI safetensors quants, '
                'and SDNQ or bitsandbytes repositories. Every format you list is written.'))

        self._parser.add_argument(
            'model', nargs='?',
            help='Diffusers folder, Hugging Face repository, weight file, or a blob link '
                 'to one weight file.')
        self._parser.add_argument(
            '-sbf', '--subfolders', nargs='+', default=None,
            help='Component folders to quantize, such as transformer or text_encoder. '
                 'Without this, every component that has weights is quantized.')
        self._parser.add_argument(
            '-f', '--formats', nargs='*', default=None,
            help='Formats and presets to write. Every name is written. '
                 '--list-formats prints the GGUF types, Comfy safetensors quants, '
                 'plain float16 and float32, short names, presets, and example file names. '
                 'q4 means Q4_K. Q4_0 is requested by that name.')
        naming = self._parser.add_mutually_exclusive_group()
        naming.add_argument(
            '--comfy', action='store_true',
            help='Write ComfyUI tensor names, so ComfyUI and ComfyUI-GGUF can load the files. '
                 'Diffusion models: FLUX.1, SD1, SDXL, SD3, Wan, and Qwen-Image. '
                 'FLUX.2 and Z-Image are safetensors, because ComfyUI-GGUF has no architecture for them. '
                 'GGUF text encoders: T5, UMT5, Llama 3 / Mistral, Qwen2, Qwen3, and Gemma 3. '
                 'A Qwen2-VL vision tower is written as mmproj beside the text encoder. '
                 'CLIP is safetensors. Files are named like transformer-Q4_K-comfy.gguf. '
                 'A component with no key map is an error.')
        naming.add_argument(
            '--llama', action='store_true',
            help='Write llama.cpp tensor names, the layout Unsloth text-encoder GGUFs use and '
                 'dgenerate loads. T5, UMT5, Llama, Qwen, and Gemma, including Gemma 4. '
                 'A full Gemma 4 model, such as the LTX-2.5 prompt enhancer, writes its '
                 'vision and audio towers as a CLIP mmproj (gemma4v and gemma4a) beside the language file. '
                 'Qwen3.5, including the FLUX.3 prompt upsampler, is written as qwen35. '
                 'Its vision tower stays a Qwen3-VL mmproj. '
                 'A Llama whose head counts are not 32 and 8 is included. Files are named like '
                 'text_encoder-Q4_K-llama.gguf. Cannot be combined with --comfy.')
        self._parser.add_argument(
            '-o', '--output',
            help='Directory to write. Files are named like transformer-Q4_K.gguf, '
                 'transformer-F16.gguf, transformer-F16-comfy.gguf, '
                 'text_encoder-F16-llama.gguf, text_encoder-float8_e4m3fn.safetensors, '
                 'or transformer-float16-comfy.safetensors. '
                 'A --quantizer URI writes a diffusers repository in a subdirectory '
                 'named from that URI, such as sdnq-uint4-g32-svd. An existing file '
                 'or repository of the same name is replaced.')
        self._parser.add_argument(
            '-q', '--quantizer', nargs='+', default=None,
            help='Write an SDNQ or bitsandbytes diffusers repository from a quantizer URI. '
                 'The URI is the same one passed to --quantizer when generating images. '
                 'Examples: "sdnq;type=uint4;group-size=32;quant-conv=true;use-svd=true", '
                 '"bnb;bits=4;bits4-quant-type=nf4;bits4-use-double-quant=true;bits4-compute-dtype=bfloat16". '
                 'Several URIs write several repositories. Quote the URI so the shell keeps the semicolons.')
        self._parser.add_argument(
            '--list-formats', action='store_true',
            help='Print GGUF types, Comfy safetensors quants, plain float16 and float32, '
                 'short names, and presets, then exit.')
        self._parser.add_argument(
            '-wn', '--weight-name', default=None,
            help='Weight file to read, such as model.fp16.safetensors. A path such as '
                 'text_encoder/model.fp16.safetensors selects that one file. The same '
                 'name is used in every --subfolders component.')
        self._parser.add_argument(
            '-var', '--variant', default=None,
            help='Weight variant to read, such as fp16 (diffusion_pytorch_model.fp16.safetensors).')
        self._parser.add_argument(
            '-rev', '--revision', default=None,
            help='Revision, when the model is a Hugging Face repository.')
        self._parser.add_argument(
            '-atk', '--auth-token', default=None,
            help='Hugging Face token for a gated repository.')
        self._parser.add_argument(
            '-ofm', '--offline-mode', action='store_true',
            help='Use weights already on disk. Do not download a repository.')

    def __call__(self) -> int:
        args = self._parser.parse_args(self.args)
        if self._parser.return_code is not None:
            return self._parser.return_code

        if args.list_formats:
            _messages.log(_quantexport.format_catalog())
            return 0

        if not args.model:
            raise self.argument_error(
                'A model folder, repository, weight file, or blob link is required.')
        if not args.formats and not args.quantizer:
            raise self.argument_error(
                'Pass --formats, for example --formats q4 q8 fp8, '
                'or --quantizer with an sdnq or bnb URI. --list-formats prints the file formats.')
        if not args.output:
            raise self.argument_error('Pass --output for the directory the quant files are written to.')

        try:
            file_formats = _quantexport.resolve_formats(args.formats) if args.formats else []
            quantizers = []
            seen_folders = set()
            for uri in args.quantizer or []:
                spec = _repoquant.parse_quantizer(uri)
                if spec.folder in seen_folders:
                    continue
                seen_folders.add(spec.folder)
                quantizers.append(spec)
            token = _hfhub.resolve_auth_token(args.auth_token)
            local_files_only = self.local_files_only or args.offline_mode
            written = []
            if _quantexport.single_weight_reference(args.model):
                if args.subfolders:
                    raise _quantexport.QuantExportError(
                        'A single weight file already selects one component. '
                        'Do not pass --subfolders.')
                if args.weight_name:
                    raise _quantexport.QuantExportError(
                        'A single weight file already names the weights. Do not pass --weight-name.')
                if args.variant:
                    raise _quantexport.QuantExportError(
                        'A single weight file already names the weights. Do not pass --variant.')
                if quantizers:
                    raise _quantexport.QuantExportError(
                        '--quantizer writes a diffusers repository and needs the whole model, '
                        'not one weight file.')
                path, name, config = _acquire_single_weight(
                    args.model, token, local_files_only)
                if file_formats:
                    state = _quantexport.load_component_state(
                        os.path.dirname(path), weight_name=os.path.basename(path))
                    count = len(state)
                    noun = 'tensor' if count == 1 else 'tensors'
                    _messages.log(
                        f'Quantizing {name} ({count} {noun}) to {", ".join(file_formats)}.')
                    written.extend(_quantexport.export_component(
                        state, file_formats, args.output, name, config,
                        comfy=args.comfy, llama=args.llama,
                        source_dir=os.path.dirname(path)))
                root = None
                components = []
            else:
                if args.weight_name and args.variant:
                    raise _quantexport.QuantExportError(
                        'Pass --weight-name or --variant, not both.')
                if quantizers and args.weight_name:
                    raise _quantexport.QuantExportError(
                        '--quantizer loads the component diffusers would load and honors --variant. '
                        '--weight-name applies to --formats.')
                root = _acquire_repository(
                    args.model,
                    args.subfolders,
                    args.revision,
                    token,
                    local_files_only,
                    args.weight_name)
                components = _quantexport.weight_targets(
                    root, args.subfolders, args.variant, args.weight_name)
            if file_formats and root is not None:
                for name, folder, weight_name in components:
                    state = _quantexport.load_component_state(
                        folder, args.variant, weight_name)
                    config = _quantexport.load_component_config(folder)
                    count = len(state)
                    noun = 'tensor' if count == 1 else 'tensors'
                    _messages.log(
                        f'Quantizing {name} ({count} {noun}) to {", ".join(file_formats)}.')
                    written.extend(_quantexport.export_component(
                        state, file_formats, args.output, name, config,
                        comfy=args.comfy, llama=args.llama, source_dir=folder))
            if quantizers and (args.comfy or args.llama):
                _messages.warning(
                    '--comfy renames GGUF and Comfy safetensors files. '
                    '--llama renames GGUF text encoders. '
                    'SDNQ and bitsandbytes repositories keep diffusers tensor names.')
            for spec in quantizers:
                written.append(_repoquant.export_pretrained_repo(
                    root,
                    [(name, folder) for name, folder, _weight in components],
                    spec, args.output, args.variant))
        except _quantexport.QuantExportError as error:
            raise self.argument_error(str(error)) from error

        if not written:
            raise self.argument_error('No quant files were written.')
        return 0
