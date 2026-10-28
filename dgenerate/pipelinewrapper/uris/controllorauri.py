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

import typing

import diffusers
import torch

import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions
from dgenerate.pipelinewrapper.uris.lorauri import LoRAUri

_control_lora_uri_parser = _textprocessing.ConceptUriParser(
    'Control LoRA', ['scale', 'revision', 'subfolder', 'weight-name'])


class ControlLoRAUri:
    """
    Representation of a ``--control-loras`` URI.
    """

    NAMES = ['Control LoRA']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--control-loras')

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]}
    }

    @property
    def model(self) -> str:
        """Model path, Hugging Face slug, or file path."""
        return self._model

    @property
    def revision(self) -> _types.OptionalString:
        """Model repo revision."""
        return self._revision

    @property
    def subfolder(self) -> _types.OptionalPath:
        """Model repo subfolder."""
        return self._subfolder

    @property
    def weight_name(self) -> _types.OptionalName:
        """Weight file name inside a repository or folder."""
        return self._weight_name

    @property
    def scale(self) -> float:
        """Per-LoRA scale. Combined with ``--control-lora-fuse-scale`` when fusing."""
        return self._scale

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None,
                 scale: float = 1.0):
        self._model = model
        self._revision = revision
        self._subfolder = subfolder
        self._weight_name = weight_name
        self._scale = scale

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def as_lora_uri(self) -> LoRAUri:
        """Same weights as a ``--loras`` URI, for the Flux transformer loader."""
        return LoRAUri(
            model=self.model,
            revision=self.revision,
            subfolder=self.subfolder,
            weight_name=self.weight_name,
            scale=self.scale)

    @staticmethod
    def load_on_pipeline(pipeline: diffusers.DiffusionPipeline,
                         uris: typing.Iterable[typing.Union['ControlLoRAUri', str]],
                         fuse_scale: float = 1.0,
                         auth_token: _types.OptionalString = None,
                         local_files_only: bool = False):
        """
        Load Flux control LoRAs with the pipeline LoRA loader.

        Full-precision weights are fused. An already quantized transformer
        (GGUF, Comfy packed weights, bitsandbytes, SDNQ) keeps the adapters
        active instead of fusing into packed parameters.
        """
        parsed = [uri if isinstance(uri, ControlLoRAUri) else ControlLoRAUri.parse(uri)
                  for uri in uris]
        try:
            LoRAUri.load_on_pipeline(
                pipeline,
                [uri.as_lora_uri() for uri in parsed],
                fuse_scale=fuse_scale,
                auth_token=auth_token,
                local_files_only=local_files_only,
                fuse=True)
        except _exceptions.LoRAUriLoadError as e:
            raise _exceptions.ControlLoRAUriLoadError(str(e)) from e

    @staticmethod
    def controlnet_from_unet(unet: torch.nn.Module,
                             uris: typing.Iterable[typing.Union['ControlLoRAUri', str]],
                             fuse_scale: float = 1.0,
                             auth_token: _types.OptionalString = None,
                             local_files_only: bool = False) -> diffusers.ControlNetModel:
        """
        Clone ``unet`` into a ControlNet and fuse Stability-style control LoRAs.

        The clone has to see ordinary UNet weights. Packed or already quantized
        UNet checkpoints cannot be cloned. ``--quantizer`` quantizes the UNet
        after this fuse. The clone stays full precision unless ``controlnet``
        is included in ``--quantizer-map``.
        """
        from dgenerate.pipelinewrapper.pipelines import module_skips_group_offload

        if module_skips_group_offload(unet):
            raise _exceptions.ControlLoRAUriLoadError(
                'SD and SDXL control LoRAs are fused into a ControlNet cloned from '
                'the UNet before quantization. This UNet is already quantized or '
                'packed (GGUF, Comfy, bitsandbytes, or SDNQ), so it cannot be cloned. '
                'Load a full-precision checkpoint and pass --quantizer. The UNet is '
                'quantized after the clone is built. Add controlnet to --quantizer-map '
                'to quantize the clone as well.')

        parsed = [uri if isinstance(uri, ControlLoRAUri) else ControlLoRAUri.parse(uri)
                  for uri in uris]
        try:
            controlnet = diffusers.ControlNetModel.from_unet(unet)
        except Exception as e:
            raise _exceptions.ControlLoRAUriLoadError(
                f'Could not build a ControlNet from the UNet: {e}') from e

        names = []
        weights = []
        try:
            for index, uri in enumerate(parsed):
                model_path = _hfhub.download_non_hf_slug_model(
                    uri.model, local_files_only=local_files_only, auth_token=auth_token)
                adapter_name = f'control-{index}'
                controlnet.load_lora_adapter(
                    model_path,
                    weight_name=uri.weight_name,
                    subfolder=uri.subfolder,
                    revision=uri.revision,
                    token=auth_token,
                    local_files_only=local_files_only,
                    adapter_name=adapter_name,
                    prefix=None)
                names.append(adapter_name)
                weights.append(uri.scale)
                _messages.debug_log(f'Added Control LoRA: "{uri}"')
            controlnet.set_adapters(names, weights)
            scale = 1.0 if fuse_scale is None else fuse_scale
            controlnet.fuse_lora(lora_scale=scale, adapter_names=names)
            unload = getattr(controlnet, 'unload_lora', None)
            if not callable(unload):
                unload = getattr(controlnet, 'unload_lora_weights', None)
            if callable(unload):
                unload()
            # The clone is built in the UNet dtype, then the Stability loader
            # writes fine-tuned layers back as float32. The pipeline feeds the
            # ControlNet the UNet's dtype.
            controlnet = controlnet.to(dtype=next(unet.parameters()).dtype)
        except _exceptions.ControlLoRAUriLoadError:
            raise
        except Exception as e:
            raise _exceptions.ControlLoRAUriLoadError(
                f'error loading Control LoRAs: {e}') from e
        return controlnet

    @staticmethod
    def parse(uri: _types.Uri) -> 'ControlLoRAUri':
        """
        Parse a ``--control-loras`` URI.

        :param uri: ``--control-loras`` URI syntax
        :raise InvalidControlLoRAUriError:
        """
        try:
            parsed = _control_lora_uri_parser.parse(uri)
            scale = parsed.args.get('scale', 1.0)
            try:
                scale = float(scale)
            except ValueError:
                raise _exceptions.InvalidControlLoRAUriError(
                    f'Control LoRA "scale" must be a floating point number, received: {scale}')
            return ControlLoRAUri(
                model=parsed.concept,
                scale=scale,
                weight_name=parsed.args.get('weight-name', None),
                revision=parsed.args.get('revision', None),
                subfolder=parsed.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidControlLoRAUriError(e) from e
