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

import os.path
import inspect
import typing

import diffusers
import huggingface_hub
import torch

import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_lora_uri_parser = _textprocessing.ConceptUriParser(
    'LoRA', ['scale', 'revision', 'subfolder', 'weight-name', 'component'])
_LORA_COMPONENTS = ('transformer', 'transformer_2')


def _lora_targets_are_split(listed, adapter_names) -> bool:
    """True when some adapters are missing from a component that has others.

    Wan 2.2 loads the high-noise LoRA onto ``transformer`` and the low-noise
    LoRA onto ``transformer_2``. ``set_adapters`` would otherwise require every
    expert to contain every adapter name.
    """
    components = [name for name, adapters in listed.items() if adapters]
    if len(components) < 2:
        return False
    for adapter in adapter_names:
        homes = [name for name in components if adapter in listed.get(name, ())]
        if homes and len(homes) != len(components):
            return True
    return False


def _every_target_accepts_adapters(pipeline, listed, adapter_names) -> bool:
    """True when each component that holds these adapters implements ``set_adapters``."""
    saw_target = False
    for component, present in listed.items():
        if not any(name in present for name in adapter_names):
            continue
        saw_target = True
        module = getattr(pipeline, component, None)
        if not callable(getattr(module, 'set_adapters', None)):
            return False
    return saw_target


def _activate_loaded_loras(pipeline, adapter_names, adapter_weights, fuse, fuse_scale):
    listed = pipeline.get_list_adapters() or {}
    if not isinstance(listed, dict):
        listed = {}
    quantized = _quantized_adapter_components(pipeline, listed, adapter_names) if fuse else []
    if quantized:
        _messages.warning(
            'Leaving LoRA adapters active on quantized or GGUF weights '
            f'({", ".join(quantized)}). Packed, bitsandbytes, and SDNQ weights '
            'cannot take a fused write. The fuse scale is folded into those '
            'adapter weights.')
    # Hunyuan-DiT keeps ``set_adapters`` on the pipeline. Its text encoder is
    # not a PEFT module, so a per-component listing still fuses from here.
    if not (_lora_targets_are_split(listed, adapter_names)
            and _every_target_accepts_adapters(pipeline, listed, adapter_names)):
        if quantized:
            _activate_quantized_lora_targets(
                pipeline, listed, adapter_names, adapter_weights, quantized, fuse_scale)
            return
        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
        if fuse:
            _messages.debug_log(
                f'Fusing all LoRAs into pipeline with global scale: {fuse_scale}')
            pipeline.fuse_lora(adapter_names=adapter_names, lora_scale=fuse_scale)
        else:
            _commit_unfused_weight_updates(pipeline, fuse_scale)
        return

    _messages.debug_log(
        'Activating LoRAs on the components that received them.')
    quantized_names = set(quantized)
    for component, present in listed.items():
        names = [name for name in adapter_names if name in present]
        if not names:
            continue
        module = getattr(pipeline, component, None)
        if module is None:
            continue
        weights = [adapter_weights[adapter_names.index(name)] for name in names]
        if component in quantized_names:
            module.set_adapters(names, [weight * fuse_scale for weight in weights])
            continue
        module.set_adapters(names, weights)
        if fuse and hasattr(module, 'fuse_lora'):
            module.fuse_lora(lora_scale=fuse_scale, adapter_names=names)
    if not fuse:
        _commit_unfused_weight_updates(pipeline, fuse_scale)


def _commit_unfused_weight_updates(pipeline, fuse_scale):
    commit_weight_updates = getattr(pipeline, 'commit_lora_weight_updates', None)
    if callable(commit_weight_updates):
        commit_weight_updates(fuse_scale)
    _messages.debug_log(
        'Leaving LoRA adapters active without fusing them into the weights.')


def _collapse_zimage_lora_prefix(state_dict):
    """
    Kohya and Musubi write Z-Image LoRAs as ``lora_unet__layers_...``.

    Diffusers strips one ``lora_unet_`` and the leftover underscore becomes
    ``transformer..layers``. PEFT then looks for a module named ``.layers``.
    """
    if not any(key.startswith('transformer..') for key in state_dict):
        return state_dict
    return {
        key.replace('transformer..', 'transformer.', 1): value
        for key, value in state_dict.items()
    }


def _quantized_adapter_components(pipeline, listed, adapter_names) -> list[str]:
    """Components that received these adapters and cannot take a fused write."""
    from dgenerate.pipelinewrapper import pipelines as _pipelines

    components = getattr(pipeline, 'components', None)
    if not isinstance(components, dict):
        components = {}
    found = []
    for component, present in listed.items():
        if not any(name in present for name in adapter_names):
            continue
        module = getattr(pipeline, component, None)
        if module is None:
            module = components.get(component)
        if isinstance(module, torch.nn.Module) and _pipelines.module_skips_group_offload(module):
            found.append(component)
    return found


def _pipeline_has_quantized_modules(pipeline) -> bool:
    """True when any component keeps BnB, SDNQ, GGUF, or other packed weights."""
    components = getattr(pipeline, 'components', None)
    if not isinstance(components, dict):
        return False
    listed = {name: ['*'] for name in components}
    return bool(_quantized_adapter_components(pipeline, listed, ['*']))


def _activate_quantized_lora_targets(pipeline, listed, adapter_names, adapter_weights,
                                     quantized, fuse_scale):
    """
    Fuse full-precision components and leave packed ones as adapters.

    A quantized text encoder does not block fusion into a full-precision UNet
    or transformer. The fuse scale is folded into the packed component's
    adapter weights.
    """
    quantized = set(quantized)

    def weights_for(names, scale):
        return [adapter_weights[adapter_names.index(name)] * scale for name in names]

    if _every_target_accepts_adapters(pipeline, listed, adapter_names):
        for component, present in listed.items():
            names = [name for name in adapter_names if name in present]
            if not names:
                continue
            module = getattr(pipeline, component, None)
            if module is None:
                continue
            if component in quantized:
                module.set_adapters(names, weights_for(names, fuse_scale))
                continue
            module.set_adapters(names, weights_for(names, 1.0))
            if hasattr(module, 'fuse_lora'):
                module.fuse_lora(lora_scale=fuse_scale, adapter_names=names)
        return

    plain = []
    for component, present in listed.items():
        if component in quantized:
            continue
        if any(name in present for name in adapter_names):
            plain.append(component)
    pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
    if plain:
        pipeline.fuse_lora(
            components=plain, adapter_names=adapter_names, lora_scale=fuse_scale)
        for component in quantized:
            module = getattr(pipeline, component, None)
            present = listed.get(component) or []
            names = [name for name in adapter_names if name in present]
            if names and callable(getattr(module, 'set_adapters', None)):
                module.set_adapters(names, weights_for(names, fuse_scale))
        return
    pipeline.set_adapters(
        adapter_names,
        adapter_weights=[weight * fuse_scale for weight in adapter_weights])


def _pipeline_is_z_image(pipeline) -> bool:
    return pipeline.__class__.__name__.startswith('ZImage')


class LoRAUri:
    """
    Representation of a ``--loras`` uri
    """

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['LoRA']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--loras')

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]}
    }

    # ===

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug, file path
        """
        return self._model

    @property
    def revision(self) -> _types.OptionalString:
        """
        Model repo revision
        """
        return self._revision

    @property
    def subfolder(self) -> _types.OptionalPath:
        """
        Model repo subfolder
        """
        return self._subfolder

    @property
    def weight_name(self) -> _types.OptionalName:
        """
        Model weight-name
        """
        return self._weight_name

    @property
    def scale(self) -> float:
        """
        LoRA scale
        """
        return self._scale

    @property
    def component(self) -> _types.OptionalName:
        """
        Pipeline component that receives this LoRA.

        ``transformer_2`` is the Wan 2.2 low-noise expert. ``None`` loads the
        primary transformer.
        """
        return self._component

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None,
                 scale: float = 1.0,
                 component: _types.OptionalName = None):
        self._model = model
        self._scale = scale
        self._revision = revision
        self._subfolder = subfolder
        self._weight_name = weight_name
        self._component = component

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    @staticmethod
    def load_on_pipeline(pipeline: diffusers.DiffusionPipeline,
                         uris: typing.Iterable[typing.Union["LoRAUri", str]],
                         fuse_scale: float = 1.0,
                         auth_token: _types.OptionalString = None,
                         local_files_only: bool = False,
                         fuse: bool = True):
        """
        Load LoRA weights on to a pipeline using this URI


        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param uris: Iterable of :py:class:`LoRAUri` or ``str`` LoRA URIs to load
        :param fuse_scale: Global scale for the fused LoRAs, all LoRAs are fused together
            using their individual scale value, and then fused into the main model using this scale.
        :param auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug
        :param fuse: Merge the loaded adapters into the pipeline weights.
            When false, the adapters stay active through ``set_adapters``.

        :raises dgenerate.ModelNotFoundError: If the model could not be found.
        :raises dgenerate.pipelinewrapper.uris.exceptions.InvalidLoRAUriError: On URI parsing errors.
        :raises dgenerate.pipelinewrapper.uris.exceptions.LoRAUriLoadError: On loading errors.
        """
        def cache_all(e):
            if isinstance(e, _exceptions.InvalidLoRAUriError):
                raise e
            else:
                raise _exceptions.LoRAUriLoadError(
                    f'error loading LoRAs: {e}') from e

        with _hfhub.with_hf_errors_as_model_not_found(cache_all):
            LoRAUri._load_on_pipeline(pipeline,
                                      uris=uris,
                                      fuse_scale=fuse_scale,
                                      auth_token=auth_token,
                                      local_files_only=local_files_only,
                                      fuse=fuse)

    @staticmethod
    def _load_on_pipeline(pipeline: diffusers.DiffusionPipeline,
                          uris: typing.Iterable[typing.Union["LoRAUri", str]],
                          fuse_scale: float = 1.0,
                          auth_token: _types.OptionalString = None,
                          local_files_only: bool = False,
                          fuse: bool = True):

        if hasattr(pipeline, 'load_lora_weights'):
            adapter_names = []
            adapter_weights = []
            for adapter_index, lora_uri in enumerate(uris):

                if not isinstance(lora_uri, LoRAUri):
                    lora_uri = LoRAUri.parse(lora_uri)

                model_path = _hfhub.download_non_hf_slug_model(lora_uri.model, local_files_only=local_files_only, auth_token=auth_token)

                weight_name = lora_uri.weight_name

                if local_files_only and not os.path.exists(model_path):

                    subfolder = lora_uri.subfolder if lora_uri.subfolder else ''

                    probable_path_1 = os.path.join(
                        subfolder, 'pytorch_lora_weights.safetensors' if
                        weight_name is None else weight_name)

                    probable_path_2 = os.path.join(
                        subfolder, 'pytorch_lora_weights.bin')

                    file_path = huggingface_hub.try_to_load_from_cache(lora_uri.model,
                                                                       filename=probable_path_1,
                                                                       revision=lora_uri.revision)

                    if not isinstance(file_path, str):
                        file_path = huggingface_hub.try_to_load_from_cache(lora_uri.model,
                                                                           filename=probable_path_2,
                                                                           revision=lora_uri.revision)
                        if not isinstance(file_path, str):
                            raise RuntimeError(
                                f'LoRA model "{lora_uri.model}" '
                                'was not available in the local huggingface cache.')
                        else:
                            weight_name = 'pytorch_lora_weights.bin'
                    else:
                        if weight_name is None:
                            weight_name = 'pytorch_lora_weights.safetensors'

                    model_path = os.path.dirname(file_path)

                adapter_name = str(adapter_index)
                adapter_names.append(adapter_name)
                adapter_weights.append(lora_uri.scale)

                load_kwargs = dict(
                    revision=lora_uri.revision,
                    subfolder=lora_uri.subfolder,
                    weight_name=weight_name,
                    local_files_only=local_files_only,
                    token=auth_token,
                )
                try:
                    _load_one_lora(
                        pipeline, model_path, adapter_name,
                        lora_strength=lora_uri.scale,
                        load_into_transformer_2=lora_uri.component == 'transformer_2',
                        use_safetensors=True, **load_kwargs)
                except EnvironmentError:
                    # brute force, try for .bin files
                    _load_one_lora(
                        pipeline, model_path, adapter_name,
                        lora_strength=lora_uri.scale,
                        load_into_transformer_2=lora_uri.component == 'transformer_2',
                        **load_kwargs)

                if not pipeline.get_list_adapters():
                    raise RuntimeError(f'LoRA model "{lora_uri.model}" contained no usable weights.')

                _messages.debug_log(f'Added LoRA: "{lora_uri}" to pipeline: "{pipeline.__class__.__name__}"')

            _activate_loaded_loras(
                pipeline, adapter_names, adapter_weights, fuse, fuse_scale)
        else:
            raise RuntimeError(f'Pipeline: {pipeline.__class__.__name__} '
                               f'does not support loading LoRAs.')

    @staticmethod
    def parse(uri: _types.Uri) -> 'LoRAUri':
        """
        Parse a ``--loras`` uri and return an object representing its constituents

        :param uri: string with ``--loras`` uri syntax

        :raise InvalidLoRAUriError:

        :return: :py:class:`.LoRAUri`
        """
        try:
            r = _lora_uri_parser.parse(uri)

            scale = r.args.get('scale', 1.0)
            try:
                scale = float(scale)
            except ValueError:
                raise _exceptions.InvalidLoRAUriError(
                    f'LoRA "scale" must be a floating point number, received: {scale}')

            component = r.args.get('component', None)
            if component is not None and component not in _LORA_COMPONENTS:
                raise _exceptions.InvalidLoRAUriError(
                    'LoRA "component" must be "transformer" or "transformer_2", '
                    f'received: {component}')

            return LoRAUri(model=r.concept,
                           scale=scale,
                           weight_name=r.args.get('weight-name', None),
                           revision=r.args.get('revision', None),
                           subfolder=r.args.get('subfolder', None),
                           component=component)
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidLoRAUriError(e) from e


def _load_one_lora(
        pipeline, model_path, adapter_name, lora_strength=1.0,
        load_into_transformer_2=False, **kwargs):
    parameters = inspect.signature(pipeline.load_lora_weights).parameters
    if 'lora_strength' in parameters:
        kwargs['lora_strength'] = lora_strength
    if load_into_transformer_2:
        accepts_target = (
            'load_into_transformer_2' in parameters
            or any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                   for parameter in parameters.values()))
        if not accepts_target:
            raise RuntimeError(
                f'{pipeline.__class__.__name__} cannot load a LoRA onto transformer_2.')
        kwargs['load_into_transformer_2'] = True
    if not _pipeline_is_z_image(pipeline):
        pipeline.load_lora_weights(model_path, adapter_name=adapter_name, **kwargs)
        return

    state_dict, metadata = pipeline.lora_state_dict(
        model_path, return_lora_metadata=True, **kwargs)
    state_dict = _collapse_zimage_lora_prefix(state_dict)
    pipeline.load_lora_into_transformer(
        state_dict,
        transformer=pipeline.transformer,
        adapter_name=adapter_name,
        metadata=metadata,
        _pipeline=pipeline,
    )
