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
import dgenerate.quantize.hfcard as _hfcard
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
    Qwen-Image 2.1, Z-Image, and MiniMax-H3 video and image safetensors, and
    the GGUF text encoders T5, UMT5, Llama 3 / Mistral, Qwen2, Qwen3, and Gemma 3.
    Those files are named like transformer-Q4_K-comfy.gguf.

    MiniMax-H3 video: ``--subfolders transformer`` is text and keyframes,
    ``transformer_ref`` is references, and ``text_encoder`` is Qwen3-VL.
    MiniMax-H3-Image: ``transformer`` is already Comfy names, and
    ``transformer_diffusers`` is the Diffusers copy with six output heads.
    ``--comfy`` writes both as the Comfy layout.

    Load a diffusers-named file with the matching component option and keep the
    original repository as the model:

    dgenerate --sub-command quantize black-forest-labs/FLUX.1-dev --subfolders transformer --formats q4 q8 fp8 -o quants

    dgenerate --sub-command quantize ./model --formats dgenerate comfy -o quants

    dgenerate --sub-command quantize --list-formats

    --hf-upload publishes the written files. --hf-upload-card adds a model card.
    --hf-upload-card-preview prints that card and does not quantize or upload.

    dgenerate --sub-command quantize ./model --formats q4 -o quants --hf-upload user/model-q4 --hf-upload-card

    dgenerate --sub-command quantize ./model --hf-upload-card-preview --quantizer "sdnq;type=uint4;group-size=32;use-svd=true"

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
                 'Diffusion models: FLUX.1, SD1, SDXL, SD3, Wan, Qwen-Image, and MiniMax-H3 '
                 'video and image, and LTX-Video. '
                 'FLUX.2, FLUX.3, Z-Image, LTX-2, Krea 2, Hunyuan-DiT, HunyuanVideo 1.5, '
                 'HunyuanImage 2.1, Kandinsky 5, and Stable Cascade are safetensors, because '
                 'ComfyUI-GGUF has no architecture for them. MiniMax-H3 video uses '
                 '--subfolders transformer (text and keyframes) or transformer_ref '
                 '(references). MiniMax-H3-Image transformer is already Comfy names; '
                 'transformer_diffusers is the Diffusers copy. '
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
            help='Hugging Face token for a gated repository and for --hf-upload. '
                 'When this is omitted, HF_TOKEN or a saved huggingface-cli login is used.')
        self._parser.add_argument(
            '--hf-upload', default=None, metavar='REPO',
            help='Upload the written files to this Hugging Face model repository. '
                 'Accepts a repo ID for example: user/repo-name. A huggingface.co URL for that '
                 'repository is accepted. A new repository is created when it does not '
                 'exist. It is public unless --hf-private is set or the account default '
                 'is private. An existing repository keeps its visibility. One --quantizer repository '
                 'and no --formats uploads that diffusers folder at the repo root, so it can be '
                 'loaded as a model. Several files upload under the names written to '
                 '--output. Pass --hf-upload-card to write a README.md model card. '
                 'The token is --auth-token, HF_TOKEN, or a saved login.')
        self._parser.add_argument(
            '--hf-upload-card', nargs='?', const='', default=None, metavar='CARD',
            help='README.md to write with the quant files, and to upload with --hf-upload. '
                 'Omitted, no model card is written. With no value, the card is built from '
                 'the source model, the formats, the quantizer settings, and the command. '
                 'extra:TEXT and extra-file:PATH insert that text into the generated card, '
                 'after the introduction. file:PATH uses that UTF-8 file as the whole card. '
                 'text:TEXT uses that text as the whole card. Quote the value when it contains '
                 'spaces, as in --hf-upload-card "extra:Use guidance 3.5."')
        self._parser.add_argument(
            '--hf-upload-card-preview', action='store_true',
            help='Print the model card and do not quantize or upload. Pass --output to also '
                 'write README.md there. With no --hf-upload-card, the preview is the generated '
                 'card. extra: and extra-file: are inserted into that card. file: and text: '
                 'are printed as the whole card. With --hf-upload, the card names that '
                 'repository and nothing is uploaded.')
        self._parser.add_argument(
            '--hf-private', action='store_true',
            help='Create the --hf-upload repository as private when it does not already exist. '
                 'An existing repository keeps its visibility.')
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
        if args.hf_private and not args.hf_upload:
            raise self.argument_error('--hf-private applies to --hf-upload.')

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
            repo_id = _hfcard.normalize_repo_id(args.hf_upload) if args.hf_upload else None
            card_spec = None if args.hf_upload_card is None else _hfcard.parse_upload_card(args.hf_upload_card)
            if (card_spec is not None and card_spec.kind in {'file', 'extra-file'}
                    and not os.path.isfile(card_spec.value)):
                raise _quantexport.QuantExportError(
                    f'"{card_spec.value}" is not a file. '
                    f'--hf-upload-card {card_spec.kind}: needs a UTF-8 file.')
            self._reject_incompatible_sources(args, quantizers)
            if repo_id and not args.hf_upload_card_preview and not token:
                raise _quantexport.QuantExportError(
                    'Pass --auth-token, set HF_TOKEN, or run huggingface-cli login '
                    'before --hf-upload.')
            if args.hf_upload_card_preview:
                card = self._readme_text(
                    args, file_formats, quantizers, repo_id, card_spec, preview=True)
                _messages.log(card.rstrip())
                if args.output:
                    os.makedirs(args.output, exist_ok=True)
                    readme = os.path.join(args.output, 'README.md')
                    with open(readme, 'w', encoding='utf-8', newline='\n') as handle:
                        handle.write(card)
                    _messages.log(f'Wrote {readme}.')
                _messages.log('Card preview: not quantizing and not uploading.')
                return 0
            if not args.output:
                raise self.argument_error(
                    'Pass --output for the directory the quant files are written to.')
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
                component_classes = [(name, _hfcard.component_class(config))]
            else:
                component_classes = []
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
                    component_classes.append((name, _hfcard.component_class(config)))
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
            if repo_id or card_spec is not None:
                if not component_classes:
                    for name, folder, _weight in components:
                        component_classes.append((
                            name, _hfcard.component_class(
                                _quantexport.load_component_config(folder))))
                self._publish(
                    args, file_formats, quantizers, repo_id, token, card_spec,
                    written, component_classes, _hfcard.pipeline_class(root))
        except _quantexport.QuantExportError as error:
            raise self.argument_error(str(error)) from error

        if not written:
            raise self.argument_error('No quant files were written.')
        return 0

    @staticmethod
    def _reject_incompatible_sources(args, quantizers) -> None:
        single = _quantexport.single_weight_reference(args.model)
        if not single:
            cleaned = str(args.model).split('?', 1)[0].split('#', 1)[0].lower()
            single = cleaned.endswith((
                '.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.index.json'))
        if single:
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
            return
        if args.weight_name and args.variant:
            raise _quantexport.QuantExportError(
                'Pass --weight-name or --variant, not both.')
        if quantizers and args.weight_name:
            raise _quantexport.QuantExportError(
                '--quantizer loads the component diffusers would load and honors --variant. '
                '--weight-name applies to --formats.')

    def _model_card(self, args, file_formats, quantizers, repo_id, *,
                    preview: bool,
                    component_classes=None,
                    pipeline=None,
                    files=None,
                    extra=None) -> str:
        return _hfcard.build_model_card(
            model=args.model,
            repo_id=repo_id,
            revision=args.revision,
            subfolders=args.subfolders,
            formats_requested=args.formats,
            formats=file_formats,
            quantizers=quantizers,
            comfy=args.comfy,
            llama=args.llama,
            variant=args.variant,
            weight_name=args.weight_name,
            command=_hfcard.quantize_command_line(
                args.model,
                subfolders=args.subfolders,
                formats=args.formats,
                quantizer_uris=args.quantizer,
                comfy=args.comfy,
                llama=args.llama,
                output=args.output,
                weight_name=args.weight_name,
                variant=args.variant,
                revision=args.revision,
                hf_upload=args.hf_upload,
                hf_upload_card=args.hf_upload_card,
                hf_private=args.hf_private,
                offline=args.offline_mode),
            component_classes=component_classes,
            pipeline=pipeline,
            files=files,
            extra=extra,
            preview=preview)

    def _readme_text(self, args, file_formats, quantizers, repo_id, card_spec, *,
                     preview: bool,
                     component_classes=None,
                     pipeline=None,
                     files=None) -> str:
        if card_spec is None or card_spec.kind in {'generate', 'extra', 'extra-file'}:
            extra = _hfcard.read_card_text(card_spec) if card_spec is not None and card_spec.kind != 'generate' else None
            return self._model_card(
                args, file_formats, quantizers, repo_id, preview=preview,
                component_classes=component_classes, pipeline=pipeline,
                files=files, extra=extra)
        return _hfcard.read_provided_card(card_spec)

    def _publish(self, args, file_formats, quantizers, repo_id, token, card_spec,
                 written, component_classes, pipeline) -> None:
        include_card = card_spec is not None
        folder, patterns = _hfcard.upload_layout(
            args.output, written, include_card=include_card)
        if include_card:
            if card_spec.kind in {'generate', 'extra', 'extra-file'}:
                files = _hfcard.listed_files(folder, patterns)
            else:
                files = None
            card = self._readme_text(
                args, file_formats, quantizers, repo_id, card_spec, preview=False,
                component_classes=component_classes, pipeline=pipeline, files=files)
            readme = os.path.join(folder, 'README.md')
            with open(readme, 'w', encoding='utf-8', newline='\n') as handle:
                handle.write(card)
            _messages.log(f'Wrote {readme}.')
        if not repo_id:
            return
        names = [name for name, _klass in component_classes]
        message = _hfcard.commit_message(args.model, file_formats, quantizers, names)
        # --offline-mode keeps the source download local. The upload the user
        # asked for still has to reach the Hub.
        with _hfhub.offline_mode_context(False):
            url = _hfcard.upload_quantized(
                folder, repo_id, token,
                private=args.hf_private,
                commit_message_text=message,
                allow_patterns=patterns)
        _messages.log(f'Uploaded {url}.')
