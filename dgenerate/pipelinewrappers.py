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
import inspect
import os
import typing

try:
    import jax
    import jax.numpy as jnp
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard
    from diffusers import FlaxStableDiffusionPipeline, FlaxStableDiffusionImg2ImgPipeline, \
        FlaxStableDiffusionInpaintPipeline, FlaxStableDiffusionControlNetPipeline, FlaxControlNetModel, \
        FlaxAutoencoderKL

    _have_jax_flax = True

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
except ImportError:
    _have_jax_flax = False

import enum
import torch
import numbers
from PIL import Image
from .textprocessing import ConceptModelPathParser, ConceptModelPathParseError, quote
from . import messages
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, \
    StableDiffusionInpaintPipelineLegacy, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, \
    StableDiffusionXLInpaintPipeline, StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline, \
    ControlNetModel, AutoencoderKL, AsymmetricAutoencoderKL, AutoencoderTiny, StableDiffusionControlNetPipeline, \
    StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetImg2ImgPipeline, \
    StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetImg2ImgPipeline, \
    StableDiffusionXLControlNetInpaintPipeline

_TORCH_MODEL_CACHE = dict()
_FLAX_MODEL_CACHE = dict()
_TORCH_CONTROL_NET_CACHE = dict()
_FLAX_CONTROL_NET_CACHE = dict()
_TORCH_VAE_CACHE = dict()
_FLAX_VAE_CACHE = dict()

DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 5
DEFAULT_IMAGE_SEED_STRENGTH = 0.8
DEFAULT_SDXL_HIGH_NOISE_FRACTION = 0.8
DEFAULT_X4_UPSCALER_NOISE_LEVEL = 20
DEFAULT_OUTPUT_WIDTH = 512
DEFAULT_OUTPUT_HEIGHT = 512


class SchedulerHelpException(Exception):
    pass


class InvalidSDXLRefinerPathError(Exception):
    pass


class InvalidVaePathError(Exception):
    pass


class InvalidControlNetPathError(Exception):
    pass


class InvalidSchedulerName(Exception):
    pass


class InvalidLoRAPathError(Exception):
    pass


class InvalidTextualInversionPathError(Exception):
    pass


_sdxl_refiner_path_parser = ConceptModelPathParser('SDXL Refiner', ['revision', 'variant', 'subfolder', 'dtype'])

_torch_vae_path_parser = ConceptModelPathParser('VAE', ['model', 'revision', 'variant', 'subfolder', 'dtype'])
_flax_vae_path_parser = ConceptModelPathParser('VAE', ['model', 'revision', 'subfolder', 'dtype'])

_torch_control_net_path_parser = ConceptModelPathParser('ControlNet',
                                                        ['scale', 'start', 'end', 'revision', 'variant', 'subfolder', 'dtype'])

_flax_control_net_path_parser = ConceptModelPathParser('ControlNet',
                                                       ['scale', 'revision', 'subfolder', 'dtype', 'from_torch'])

_lora_path_parser = ConceptModelPathParser('LoRA', ['scale', 'revision', 'subfolder', 'weight-name'])
_textual_inversion_path_parser = ConceptModelPathParser('Textual Inversion',
                                                        ['revision', 'subfolder', 'weight-name'])


class FlaxControlNetPath:
    def __init__(self, model, scale, revision, subfolder, dtype, from_torch):
        self.model = model
        self.revision = revision
        self.subfolder = subfolder
        self.dtype = dtype
        self.from_torch = from_torch
        self.scale = scale

    def load(self, device, flax_dtype_fallback, **kwargs):
        cache_by = kwargs
        cache_by['device'] = device
        cache_key = _function_cache_key(cache_by)

        cache_hit = _FLAX_CONTROL_NET_CACHE.get(cache_key)

        if cache_hit is not None:
            return cache_hit

        single_file_load_path = _is_single_file_model_load(self.model)

        if single_file_load_path:
            raise NotImplementedError('Flax --control-nets do not support single file loads from disk.')
        else:
            new_net: FlaxControlNetModel = \
                FlaxControlNetModel.from_pretrained(self.model,
                                                    revision=self.revision,
                                                    subfolder=self.subfolder,
                                                    dtype=flax_dtype_fallback if self.dtype is None else self.dtype,
                                                    from_pt=self.from_torch,
                                                    **kwargs)

        _FLAX_CONTROL_NET_CACHE[cache_key] = new_net
        return new_net


def parse_flax_control_net_path(path):
    try:
        r = _flax_control_net_path_parser.parse_concept_path(path)

        dtype = r.args.get('dtype', None)
        scale = r.args.get('scale', 1.0)
        from_torch = r.args.get('from_torch', None)

        if from_torch is not None:
            try:
                from_torch = bool(from_torch)
            except ValueError:
                raise InvalidControlNetPathError(
                    f'Flax Control Net from_torch must be undefined or boolean (true or false), received: {from_torch}')

        if dtype not in {'float32', 'float16', 'auto', None}:
            raise InvalidVaePathError(
                f'Flax Control Net dtype must be float32, float16, auto, or left undefined, received: {dtype}')

        try:
            scale = float(scale)
        except ValueError:
            raise InvalidControlNetPathError(
                f'Flax Control Net scale must be a floating point number, received {scale}')

        return FlaxControlNetPath(
            model=r.concept,
            revision=r.args.get('revision', None),
            subfolder=r.args.get('subfolder', None),
            scale=scale,
            dtype=_get_flax_dtype(dtype),
            from_torch=from_torch)

    except ConceptModelPathParseError as e:
        raise InvalidControlNetPathError(e)


class TorchControlNetPath:
    def __init__(self, model, scale, start, end, revision, variant, subfolder, dtype):
        self.model = model
        self.revision = revision
        self.variant = variant
        self.subfolder = subfolder
        self.dtype = dtype
        self.scale = scale
        self.start = start
        self.end = end

    def load(self, device, torch_dtype_fallback, **kwargs) -> ControlNetModel:
        cache_by = kwargs
        cache_by['device'] = device
        cache_key = _function_cache_key(cache_by)

        cache_hit = _TORCH_CONTROL_NET_CACHE.get(cache_key)

        if cache_hit is not None:
            return cache_hit

        single_file_load_path = _is_single_file_model_load(self.model)

        if single_file_load_path:
            new_net: ControlNetModel = \
                ControlNetModel.from_single_file(self.model,
                                                 revision=self.revision,
                                                 torch_dtype=torch_dtype_fallback if self.dtype is None else self.dtype,
                                                 **kwargs)
        else:
            new_net: ControlNetModel = \
                ControlNetModel.from_pretrained(self.model,
                                                revision=self.revision,
                                                variant=self.variant,
                                                subfolder=self.subfolder,
                                                torch_dtype=torch_dtype_fallback if self.dtype is None else self.dtype,
                                                **kwargs)

        _TORCH_CONTROL_NET_CACHE[cache_key] = new_net
        return new_net


def parse_torch_control_net_path(path) -> TorchControlNetPath:
    try:
        r = _torch_control_net_path_parser.parse_concept_path(path)

        dtype = r.args.get('dtype', None)
        scale = r.args.get('scale', 1.0)
        start = r.args.get('start', 0.0)
        end = r.args.get('end', 1.0)

        if dtype not in {'float32', 'float16', 'auto', None}:
            raise InvalidVaePathError(
                f'Torch ControlNet "dtype" must be float32, float16, auto, or left undefined, received: {dtype}')

        try:
            scale = float(scale)
        except ValueError:
            raise InvalidControlNetPathError(
                f'Torch ControlNet "scale" must be a floating point number, received: {scale}')

        try:
            start = float(start)
        except ValueError:
            raise InvalidControlNetPathError(
                f'Torch ControlNet "start" must be a floating point number, received: {start}')

        try:
            end = float(end)
        except ValueError:
            raise InvalidControlNetPathError(
                f'Torch ControlNet "end" must be a floating point number, received: {end}')

        if start > end:
            raise InvalidControlNetPathError(
                f'Torch ControlNet "start" must be less than or equal to "end".')

        return TorchControlNetPath(
            model=r.concept,
            revision=r.args.get('revision', None),
            variant=r.args.get('variant', None),
            subfolder=r.args.get('subfolder', None),
            dtype=_get_torch_dtype(dtype),
            scale=scale,
            start=start,
            end=end)

    except ConceptModelPathParseError as e:
        raise InvalidControlNetPathError(e)


class SDXLRefinerPath:
    def __init__(self, model, revision, variant, dtype, subfolder):
        self.model = model
        self.revision = revision
        self.variant = variant
        self.dtype = dtype
        self.subfolder = subfolder


def parse_sdxl_refiner_path(path) -> SDXLRefinerPath:
    try:
        r = _sdxl_refiner_path_parser.parse_concept_path(path)

        dtype = r.args.get('dtype', None)
        if dtype not in {'float32', 'float16', 'auto', None}:
            raise InvalidSDXLRefinerPathError(
                f'Torch SDXL refiner dtype must be float32, float16, auto, or left undefined, received: {dtype}')

        return SDXLRefinerPath(
            model=r.concept,
            revision=r.args.get('revision', None),
            variant=r.args.get('variant', None),
            dtype=_get_torch_dtype(dtype),
            subfolder=r.args.get('subfolder', None))
    except ConceptModelPathParseError as e:
        raise InvalidSDXLRefinerPathError(e)


class TorchVAEPath:
    def __init__(self, encoder, model, revision, variant, subfolder, dtype):
        self.encoder = encoder
        self.model = model
        self.revision = revision
        self.variant = variant
        self.dtype = dtype
        self.subfolder = subfolder


def parse_torch_vae_path(path) -> TorchVAEPath:
    try:
        r = _torch_vae_path_parser.parse_concept_path(path)

        model = r.args.get('model')
        if model is None:
            raise InvalidVaePathError('model argument for torch VAE specification must be defined.')

        dtype = r.args.get('dtype', None)
        if dtype not in {'float32', 'float16', 'auto', None}:
            raise InvalidVaePathError(
                f'Torch VAE dtype must be float32, float16, auto, or left undefined, received: {dtype}')

        return TorchVAEPath(encoder=r.concept,
                            model=model,
                            revision=r.args.get('revision', None),
                            variant=r.args.get('variant', None),
                            dtype=_get_torch_dtype(dtype),
                            subfolder=r.args.get('subfolder', None))
    except ConceptModelPathParseError as e:
        raise InvalidVaePathError(e)


class FlaxVAEPath:
    def __init__(self, encoder, model, revision, dtype, subfolder):
        self.encoder = encoder
        self.model = model
        self.revision = revision
        self.dtype = dtype
        self.subfolder = subfolder


def parse_flax_vae_path(path) -> FlaxVAEPath:
    try:
        r = _flax_vae_path_parser.parse_concept_path(path)

        model = r.args.get('model')
        if model is None:
            raise InvalidVaePathError('model argument for flax VAE specification must be defined.')

        dtype = r.args.get('dtype', None)
        if dtype not in {'float32', 'float16', 'auto', None}:
            raise InvalidVaePathError(
                f'Flax VAE dtype must be float32, float16, auto, or left undefined {dtype}')

        return FlaxVAEPath(encoder=r.concept,
                           model=model,
                           revision=r.args.get('revision', None),
                           dtype=_get_flax_dtype(dtype),
                           subfolder=r.args.get('subfolder', None))
    except ConceptModelPathParseError as e:
        raise InvalidVaePathError(e)


class LoRAPath:
    def __init__(self, model, scale, revision, subfolder, weight_name):
        self.model = model
        self.scale = scale
        self.revision = revision
        self.subfolder = subfolder
        self.weight_name = weight_name

    def load_on_pipeline(self, pipeline, **kwargs):
        if hasattr(pipeline, 'load_lora_weights'):
            pipeline.load_lora_weights(self.model,
                                       revision=self.revision,
                                       subfolder=self.subfolder,
                                       weight_name=self.weight_name,
                                       **kwargs)


def parse_lora_path(path) -> LoRAPath:
    try:
        r = _lora_path_parser.parse_concept_path(path)

        return LoRAPath(model=r.concept,
                        scale=float(r.args.get('scale', 1.0)),
                        weight_name=r.args.get('weight-name', None),
                        revision=r.args.get('revision', None),
                        subfolder=r.args.get('subfolder', None))
    except ConceptModelPathParseError as e:
        raise InvalidLoRAPathError(e)


class TextualInversionPath:
    def __init__(self, model, revision, subfolder, weight_name):
        self.model = model
        self.revision = revision
        self.subfolder = subfolder
        self.weight_name = weight_name

    def load_on_pipeline(self, pipeline, **kwargs):
        if hasattr(pipeline, 'load_textual_inversion'):
            pipeline.load_textual_inversion(self.model,
                                            revision=self.revision,
                                            subfolder=self.subfolder,
                                            weight_name=self.weight_name,
                                            **kwargs)


def parse_textual_inversion_path(path) -> TextualInversionPath:
    try:
        r = _textual_inversion_path_parser.parse_concept_path(path)

        return TextualInversionPath(model=r.concept,
                                    weight_name=r.args.get('weight-name', None),
                                    revision=r.args.get('revision', None),
                                    subfolder=r.args.get('subfolder', None))
    except ConceptModelPathParseError as e:
        raise InvalidTextualInversionPathError(e)


def _is_single_file_model_load(path):
    path, ext = os.path.splitext(path)

    if path.startswith('http://') or path.startswith('https://'):
        return True

    if os.path.isdir(path):
        return True

    if not ext:
        return False

    if ext in {'.pt', '.pth', '.bin', '.msgpack', '.ckpt', '.safetensors'}:
        return True

    return False


def _load_pytorch_vae(path,
                      torch_dtype_fallback,
                      use_auth_token) -> typing.Union[AutoencoderKL, AsymmetricAutoencoderKL, AutoencoderTiny]:
    parsed_concept = parse_torch_vae_path(path)

    if parsed_concept.dtype is None:
        parsed_concept.dtype = torch_dtype_fallback

    cache_key = _function_cache_key({'path': path,
                                     'use_auth_token': use_auth_token,
                                     'dtype': parsed_concept.dtype})

    cache_hit = _TORCH_VAE_CACHE.get(cache_key)
    if cache_hit is not None:
        return cache_hit

    encoder_name = parsed_concept.encoder

    if encoder_name == 'AutoencoderKL':
        encoder = AutoencoderKL
    elif encoder_name == 'AsymmetricAutoencoderKL':
        encoder = AsymmetricAutoencoderKL
    elif encoder_name == 'AutoencoderTiny':
        encoder = AutoencoderTiny
    else:
        raise InvalidVaePathError(f'Unknown VAE encoder class {encoder_name}')

    path = parsed_concept.model

    can_single_file_load = hasattr(encoder, 'from_single_file')
    single_file_load_path = _is_single_file_model_load(path)

    if single_file_load_path and not can_single_file_load:
        raise NotImplementedError(f'{encoder_name} is not capable of loading from a single file, '
                                  f'must be loaded from a huggingface repository slug or folder on disk.')

    if single_file_load_path:
        if parsed_concept.subfolder is not None:
            raise NotImplementedError('Single file VAE loads do not support the subfolder option.')

        if encoder is AutoencoderKL:
            # There is a bug in their cast
            vae = encoder.from_single_file(path, revision=parsed_concept.revision). \
                to(dtype=parsed_concept.dtype, non_blocking=False)
        else:
            vae = encoder.from_single_file(path,
                                           revision=parsed_concept.revision,
                                           torch_dtype=parsed_concept.dtype)

    else:
        vae = encoder.from_pretrained(path,
                                      revision=parsed_concept.revision,
                                      variant=parsed_concept.variant,
                                      torch_dtype=parsed_concept.dtype,
                                      subfolder=parsed_concept.subfolder,
                                      use_auth_token=use_auth_token)

    _TORCH_VAE_CACHE[cache_key] = vae
    return vae


def _load_flax_vae(path,
                   flax_dtype_fallback,
                   use_auth_token):
    parsed_concept = parse_flax_vae_path(path)

    if parsed_concept.dtype is None:
        parsed_concept.dtype = flax_dtype_fallback

    cache_key = _function_cache_key({'path': path,
                                     'use_auth_token': use_auth_token,
                                     'dtype': parsed_concept.dtype})

    cache_hit = _FLAX_VAE_CACHE.get(cache_key)
    if cache_hit is not None:
        return cache_hit

    encoder_name = parsed_concept.encoder

    if encoder_name == 'FlaxAutoencoderKL':
        encoder = FlaxAutoencoderKL
    else:
        raise InvalidVaePathError(f'Unknown VAE flax encoder class {encoder_name}')

    path = parsed_concept.model

    can_single_file_load = hasattr(encoder, 'from_single_file')
    single_file_load_path = _is_single_file_model_load(path)

    if single_file_load_path and not can_single_file_load:
        raise NotImplementedError(f'{encoder_name} is not capable of loading from a single file, '
                                  f'must be loaded from a huggingface repository slug or folder on disk.')

    if single_file_load_path:
        # in the future this will be supported?
        if parsed_concept.subfolder is not None:
            raise NotImplementedError('Single file VAE loads do not support the subfolder option.')
        vae = encoder.from_single_file(path,
                                       revision=parsed_concept.revision,
                                       dtype=parsed_concept.dtype)
    else:
        vae = encoder.from_pretrained(path,
                                      revision=parsed_concept.revision,
                                      dtype=parsed_concept.dtype,
                                      subfolder=parsed_concept.subfolder,
                                      use_auth_token=use_auth_token)

    _FLAX_VAE_CACHE[cache_key] = vae
    return vae


def _load_scheduler(pipeline, scheduler_name=None):
    if scheduler_name is None:
        return

    compatibles = pipeline.scheduler.compatibles

    if isinstance(pipeline, StableDiffusionLatentUpscalePipeline):
        # Seems to only work with this scheduler
        compatibles = [c for c in compatibles if c.__name__ == 'EulerDiscreteScheduler']

    if scheduler_name.lower() == 'help':
        messages.log('Compatible schedulers for this model are:', underline=True)
        for i in compatibles:
            messages.log(i.__name__)
        raise SchedulerHelpException()

    for i in compatibles:
        if i.__name__.endswith(scheduler_name):
            pipeline.scheduler = i.from_config(pipeline.scheduler.config)
            return

    raise InvalidSchedulerName(f'Scheduler named "{scheduler_name}" is not a valid compatible scheduler, '
                               f'options are:\n\n{chr(10).join(sorted(i.__name__.split(".")[-1] for i in compatibles))}')


def clear_model_cache():
    _TORCH_MODEL_CACHE.clear()
    _TORCH_CONTROL_NET_CACHE.clear()
    _FLAX_CONTROL_NET_CACHE.clear()
    _TORCH_VAE_CACHE.clear()
    _FLAX_VAE_CACHE.clear()
    _FLAX_MODEL_CACHE.clear()


def _disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


def _function_cache_key(args_dict):
    def value_hash(obj):
        if isinstance(obj, dict):
            return '{' + _function_cache_key(obj) + '}'
        elif obj is None or isinstance(obj, (str, numbers.Number)):
            return str(obj)
        else:
            return f'<{obj.__class__.__name__}:{str(id(obj))}>'

    return ','.join(f'{k}={value_hash(v)}' for k, v in sorted(args_dict.items()))


class _PipelineTypes:
    BASIC = 1
    IMG2IMG = 2
    INPAINT = 3


def _describe_pipeline_type(enum):
    return {_PipelineTypes.BASIC: 'txt2img', _PipelineTypes.IMG2IMG: 'img2img', _PipelineTypes.INPAINT: 'inpaint'}[enum]


def _create_torch_diffusion_pipeline(pipeline_type,
                                     model_type,
                                     model_path,
                                     revision,
                                     variant,
                                     dtype,
                                     model_subfolder=None,
                                     vae_path=None,
                                     lora_paths=None,
                                     textual_inversion_paths=None,
                                     control_net_paths=None,
                                     scheduler=None,
                                     safety_checker=False,
                                     auth_token=None,
                                     device='cuda',
                                     extra_args=None,
                                     model_cpu_offload=False,
                                     sequential_cpu_offload=False):
    cache_key = _function_cache_key(locals())
    catch_hit = _TORCH_MODEL_CACHE.get(cache_key)

    if catch_hit is not None:
        return catch_hit

    if model_type_is_upscaler(model_type):
        if pipeline_type != _PipelineTypes.IMG2IMG and scheduler.lower() != 'help':
            raise NotImplementedError(
                'Upscaler models only work with img2img generation, IE: --image-seeds (with no image masks).')

        pipeline_class = (StableDiffusionUpscalePipeline if model_type is ModelTypes.TORCH_UPSCALER_X4
                          else StableDiffusionLatentUpscalePipeline)
    else:
        sdxl = model_type == ModelTypes.TORCH_SDXL

        if pipeline_type == _PipelineTypes.BASIC:
            if control_net_paths:
                pipeline_class = StableDiffusionXLControlNetPipeline if sdxl else StableDiffusionControlNetPipeline
            else:
                pipeline_class = StableDiffusionXLPipeline if sdxl else StableDiffusionPipeline
        elif pipeline_type == _PipelineTypes.IMG2IMG:
            if control_net_paths:
                if sdxl:
                    pipeline_class = StableDiffusionXLControlNetImg2ImgPipeline
                else:
                    pipeline_class = StableDiffusionControlNetImg2ImgPipeline
            else:
                pipeline_class = StableDiffusionXLImg2ImgPipeline if sdxl else StableDiffusionImg2ImgPipeline
        elif pipeline_type == _PipelineTypes.INPAINT:
            if control_net_paths:
                if sdxl:
                    pipeline_class = StableDiffusionXLControlNetInpaintPipeline
                else:
                    pipeline_class = StableDiffusionControlNetInpaintPipeline
            else:
                pipeline_class = StableDiffusionXLInpaintPipeline if sdxl else StableDiffusionInpaintPipeline
        else:
            # Should be impossible
            raise NotImplementedError('Pipeline type not implemented.')

    if textual_inversion_paths:
        if model_type == ModelTypes.TORCH_UPSCALER_X2:
            raise NotImplementedError(
                'Model type torch-upscaler-x2 cannot be used with textual inversion models.')

        if isinstance(textual_inversion_paths, str):
            textual_inversion_paths = [textual_inversion_paths]

    if lora_paths is not None:
        if not isinstance(lora_paths, str):
            raise NotImplementedError('Using multiple LoRA models is currently not supported.')

        if model_type_is_upscaler(model_type):
            raise NotImplementedError(
                'LoRA models cannot be used with upscaler models.')

        lora_paths = [lora_paths]

    kwargs = {}

    torch_dtype = _get_torch_dtype(dtype)

    parsed_control_net_paths = []

    if scheduler is None or scheduler.lower() != 'help':
        # prevent waiting on this stuff just get the scheduler
        # help message for the main model

        if vae_path is not None:
            kwargs['vae'] = _load_pytorch_vae(vae_path,
                                              torch_dtype_fallback=torch_dtype,
                                              use_auth_token=auth_token)

        if control_net_paths:
            control_nets = None

            for control_net_path in control_net_paths:
                parsed_control_net_path = parse_torch_control_net_path(control_net_path)

                parsed_control_net_paths.append(parsed_control_net_path)

                new_net = parsed_control_net_path.load(use_auth_token=auth_token,
                                                       device=device,
                                                       torch_dtype_fallback=torch_dtype)

                if control_nets is not None:
                    if not isinstance(control_nets, list):
                        control_nets = [control_nets, new_net]
                    else:
                        control_nets.append(new_net)
                else:
                    control_nets = new_net

            kwargs['controlnet'] = control_nets

    if extra_args is not None:
        kwargs.update(extra_args)

    if _is_single_file_model_load(model_path):
        if model_subfolder is not None:
            raise NotImplementedError('Single file model loads do not support the subfolder option.')
        pipeline = pipeline_class.from_single_file(model_path,
                                                   revision=revision,
                                                   variant=variant,
                                                   torch_dtype=torch_dtype,
                                                   use_safe_tensors=model_path.endswith('.safetensors'),
                                                   **kwargs)
    else:
        pipeline = pipeline_class.from_pretrained(model_path,
                                                  revision=revision,
                                                  variant=variant,
                                                  torch_dtype=torch_dtype,
                                                  subfolder=model_subfolder,
                                                  use_auth_token=auth_token,
                                                  **kwargs)

    _load_scheduler(pipeline, scheduler)

    if textual_inversion_paths is not None:
        for inversion_path in textual_inversion_paths:
            parse_textual_inversion_path(inversion_path). \
                load_on_pipeline(pipeline, use_auth_token=auth_token)

    if lora_paths is not None:
        if control_net_paths is not None and model_type == ModelTypes.TORCH_SDXL:
            raise NotImplementedError(
                'LoRA currently cannot be used with Control Nets when using SDXL')

        for lora_path in lora_paths:
            parse_lora_path(lora_path). \
                load_on_pipeline(pipeline, use_auth_token=auth_token)

    if not safety_checker:
        if hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None:
            # If it's already None for some reason you'll get a call
            # to an unassigned feature_extractor by assigning it a value

            # The attribute will not exist for SDXL pipelines currently

            pipeline.safety_checker = _disabled_safety_checker

    pipeline._dgenerate_sequential_offload = sequential_cpu_offload
    pipeline._dgenerate_cpu_offload = model_cpu_offload

    if sequential_cpu_offload and 'cuda' in device:
        pipeline.enable_sequential_cpu_offload(device=device)
    elif model_cpu_offload and 'cuda' in device:
        pipeline.enable_model_cpu_offload(device=device)

    _TORCH_MODEL_CACHE[cache_key] = (pipeline, parsed_control_net_paths)
    return pipeline, parsed_control_net_paths


def _create_flax_diffusion_pipeline(pipeline_type,
                                    model_path,
                                    revision,
                                    dtype,
                                    model_subfolder=None,
                                    vae_path=None,
                                    control_net_paths=None,
                                    scheduler=None,
                                    safety_checker=False,
                                    auth_token=None,
                                    extra_args=None,
                                    device='cuda'):
    cache_key = _function_cache_key(locals())
    catch_hit = _FLAX_MODEL_CACHE.get(cache_key)

    if catch_hit is not None:
        return catch_hit

    has_control_nets = False
    if control_net_paths is not None:
        if len(control_net_paths) > 1:
            raise NotImplementedError('Flax does not support multiple --control-nets.')
        if len(control_net_paths) == 1:
            has_control_nets = True

    if pipeline_type == _PipelineTypes.BASIC:
        if has_control_nets:
            pipeline_class = FlaxStableDiffusionControlNetPipeline
        else:
            pipeline_class = FlaxStableDiffusionPipeline
    elif pipeline_type == _PipelineTypes.IMG2IMG:
        if has_control_nets:
            raise NotImplementedError('Flax does not support --image-seeds with --control-nets, use --control-images.')
        pipeline_class = FlaxStableDiffusionImg2ImgPipeline
    elif pipeline_type == _PipelineTypes.INPAINT:
        if has_control_nets:
            raise NotImplementedError('Flax does not support --image-seeds with --control-nets, use --control-images.')
        pipeline_class = FlaxStableDiffusionInpaintPipeline
    else:
        raise NotImplementedError('Pipeline type not implemented.')

    kwargs = {}
    vae_params = None
    control_net_params = None

    flax_dtype = _get_flax_dtype(dtype)

    parsed_control_net_paths = []

    if scheduler is None or scheduler.lower() != 'help':
        # prevent waiting on this stuff just get the scheduler
        # help message for the main model

        if vae_path is not None:
            vae_path, vae_params = _load_flax_vae(vae_path,
                                                  flax_dtype_fallback=flax_dtype,
                                                  use_auth_token=auth_token)
            kwargs['vae'] = vae_path

        if control_net_paths is not None:
            parsed_flax_control_net_path = parse_flax_control_net_path(control_net_paths[0])

            parsed_control_net_paths.append(parsed_flax_control_net_path)

            control_net, control_net_params = parse_flax_control_net_path(control_net_paths[0]) \
                .load(use_auth_token=auth_token, device=device, flax_dtype_fallback=flax_dtype)

            kwargs['controlnet'] = control_net

    if extra_args is not None:
        kwargs.update(extra_args)

    pipeline, params = pipeline_class.from_pretrained(model_path,
                                                      revision=revision,
                                                      dtype=flax_dtype,
                                                      subfolder=model_subfolder,
                                                      use_auth_token=auth_token,
                                                      **kwargs)

    if vae_params is not None:
        params['vae'] = vae_params

    if control_net_params is not None:
        params['controlnet'] = control_net_params

    _load_scheduler(pipeline, scheduler)

    if not safety_checker:
        pipeline.safety_checker = None

    _FLAX_MODEL_CACHE[cache_key] = (pipeline, params, parsed_control_net_paths)
    return pipeline, params, parsed_control_net_paths


def supported_model_types():
    if have_jax_flax():
        return ['torch', 'torch-sdxl', 'torch-upscaler-x2', 'torch-upscaler-x4', 'flax']
    else:
        return ['torch', 'torch-sdxl', 'torch-upscaler-x2', 'torch-upscaler-x4']


class ModelTypes(enum.Enum):
    TORCH = 1
    TORCH_SDXL = 2
    TORCH_UPSCALER_X2 = 3
    TORCH_UPSCALER_X4 = 4
    FLAX = 5


def model_type_is_upscaler(model_type: typing.Union[ModelTypes, str]):
    if isinstance(model_type, str):
        model_type = get_model_type_enum(model_type)

    return model_type in {ModelTypes.TORCH_UPSCALER_X2,
                          ModelTypes.TORCH_UPSCALER_X4}


def get_model_type_enum(id_str) -> ModelTypes:
    return {'torch': ModelTypes.TORCH,
            'torch-sdxl': ModelTypes.TORCH_SDXL,
            'torch-upscaler-x2': ModelTypes.TORCH_UPSCALER_X2,
            'torch-upscaler-x4': ModelTypes.TORCH_UPSCALER_X4,
            'flax': ModelTypes.FLAX}[id_str]


def have_jax_flax():
    return _have_jax_flax


def _get_flax_dtype(dtype):
    if dtype is None:
        return None

    if isinstance(dtype, jnp.dtype):
        return dtype

    return {'float16': jnp.bfloat16,
            'float32': jnp.float32,
            'float64': jnp.float64,
            'auto': None}[dtype.lower()]


def _get_torch_dtype(dtype) -> typing.Union[torch.dtype, None]:
    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    return {'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
            'auto': None}[dtype.lower()]


def _image_grid(imgs, rows, cols):
    w, h = imgs[0].size

    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
        img.close()

    return grid


class PipelineResultWrapper:
    def __init__(self, image):
        self.image = image
        self._dgenerate_opts = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.image is not None:
            self.image.close()
            self.image = None

    @property
    def dgenerate_config(self):
        from .__init__ import __version__

        model_path = self._dgenerate_opts[0]

        opts = self._dgenerate_opts[1:]

        config = f'#! dgenerate {__version__}\n\n'

        config += quote(model_path) + ' \\\n'

        for opt in opts[:-1]:
            config += f'{opt[0]} {opt[1]} \\\n'

        last = opts[-1]
        config += f'{last[0]} {last[1]}'

        return config

    @property
    def dgenerate_command(self):
        model_path = self._dgenerate_opts[0]
        opts = self._dgenerate_opts[1:]
        return f'dgenerate {model_path} {" ".join(f"{opt[0]} {opt[1]}" for opt in opts)}'

    @property
    def dgenerate_opts(self):
        return self._dgenerate_opts.copy()


def _gpu_id_from_cuda_device(device):
    parts = device.split(':', 1)
    return parts[1] if len(parts) == 2 else 0


class DiffusionPipelineWrapperBase:
    def __init__(self,
                 model_path,
                 dtype,
                 device='cuda',
                 model_type='torch',
                 revision=None,
                 variant=None,
                 model_subfolder=None,
                 vae_path=None,
                 lora_paths=None,
                 textual_inversion_paths=None,
                 control_net_paths=None,
                 sdxl_refiner_path=None,
                 scheduler=None,
                 safety_checker=False,
                 auth_token=None):

        self._model_subfolder = model_subfolder
        self._device = device
        self._model_type = model_type.strip().lower()
        self._model_path = model_path
        self._pipeline = None
        self._flax_params = None
        self._revision = revision
        self._variant = variant
        self._dtype = dtype
        self._device = device
        self._vae_path = vae_path
        self._safety_checker = safety_checker
        self._scheduler = scheduler
        self._lora_paths = lora_paths
        self._lora_scale = None
        self._textual_inversion_paths = textual_inversion_paths
        self._control_net_paths = control_net_paths
        self._parsed_control_net_paths = []
        self._sdxl_refiner_path = sdxl_refiner_path
        self._sdxl_refiner_pipeline = None
        self._auth_token = auth_token
        self._pipeline_type = None

        if sdxl_refiner_path is not None:
            parsed_path = parse_sdxl_refiner_path(sdxl_refiner_path)
            self._sdxl_refiner_path = parsed_path.model
            self._sdxl_refiner_revision = parsed_path.revision
            self._sdxl_refiner_variant = parsed_path.variant
            self._sdxl_refiner_dtype = parsed_path.dtype
            self._sdxl_refiner_subfolder = parsed_path.subfolder

        if lora_paths is not None:
            if model_type == 'flax':
                raise NotImplementedError('LoRA loading is not implemented for flax.')

            if not isinstance(lora_paths, str):
                raise NotImplementedError('Using multiple LoRA models is currently not supported.')

            self._lora_scale = parse_lora_path(lora_paths).scale

    @staticmethod
    def _pipeline_to(pipeline, device):
        if hasattr(pipeline, 'to'):
            if not pipeline._dgenerate_cpu_offload and \
                    not pipeline._dgenerate_sequential_offload:
                return pipeline.to(device)
            else:
                return pipeline
        return pipeline

    _LAST_CALLED_PIPE = None

    @staticmethod
    def _call_pipeline(pipeline, device, **kwargs):
        messages.log("Calling Pipeline:", pipeline.__class__,
                     f'Device: "{device}"',
                     'Args:',
                     {k: str(v) if len(str(v)) < 256 else v.__class__ for k, v in kwargs.items()},
                     level=messages.DEBUG)

        if pipeline is DiffusionPipelineWrapperBase._LAST_CALLED_PIPE:
            return pipeline(**kwargs)

        if DiffusionPipelineWrapperBase._LAST_CALLED_PIPE is not None:
            DiffusionPipelineWrapperBase._pipeline_to(
                DiffusionPipelineWrapperBase._LAST_CALLED_PIPE, 'cpu')

        DiffusionPipelineWrapperBase._pipeline_to(pipeline, device)
        r = pipeline(**kwargs)
        DiffusionPipelineWrapperBase._pipeline_to(pipeline, 'cpu')

        DiffusionPipelineWrapperBase._LAST_CALLED_PIPE = pipeline
        return r

    @property
    def revision(self):
        return self._revision

    @property
    def safety_checker(self):
        return self._safety_checker

    @property
    def variant(self):
        return self._variant

    @property
    def dtype(self):
        return self._dtype

    @property
    def textual_inversion_paths(self):
        return [self._textual_inversion_paths] if \
            isinstance(self._textual_inversion_paths, str) else self.textual_inversion_paths

    @property
    def control_net_paths(self):
        return [self._control_net_paths] if \
            isinstance(self._control_net_paths, str) else self._control_net_paths

    @property
    def device(self):
        return self._device

    @property
    def model_path(self):
        return self._model_path

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def sdxl_refiner_path(self):
        return self._sdxl_refiner_path

    @property
    def model_type(self):
        return self._model_type

    @property
    def model_subfolder(self):
        return self._model_subfolder

    @property
    def vae_path(self):
        return self._vae_path

    @property
    def lora_paths(self):
        return [self._lora_paths] if \
            isinstance(self._lora_paths, str) else self._lora_paths

    @property
    def auth_token(self):
        return self._auth_token

    def _reconstruct_dgenerate_opts(self, **kwargs):
        prompt = kwargs.get('prompt', None)
        negative_prompt = kwargs.get('negative_prompt', None)
        image = kwargs.get('image', None)
        control_image = kwargs.get('control_image', None)
        strength = kwargs.get('strength', None)
        upscaler_noise_level = kwargs.get('upscaler_noise_level', None)
        mask_image = kwargs.get('mask_image', None)
        seed = kwargs.get('seed')
        width = kwargs.get('width', None)
        height = kwargs.get('height', None)
        num_inference_steps = kwargs.get('num_inference_steps')
        guidance_scale = kwargs.get('guidance_scale')
        sdxl_high_noise_fraction = kwargs.get('sdxl_high_noise_fraction', None)
        sdxl_original_size = kwargs.get('sdxl_original_size', None)
        sdxl_target_size = kwargs.get('sdxl_target_size', None)

        if strength is not None:
            num_inference_steps = int(num_inference_steps * strength)
            guidance_scale = guidance_scale * strength

        opts = [self.model_path, ('--model-type', self._model_type), ('--dtype', self._dtype),
                ('--device', self._device), ('--inference-steps', num_inference_steps),
                ('--guidance-scales', guidance_scale), ('--seeds', seed)]

        if prompt is not None:
            if negative_prompt is not None:
                opts.append(('--prompts', f'"{prompt}; {negative_prompt}"'))
            else:
                opts.append(('--prompts', quote(prompt)))

        if self._revision is not None:
            opts.append(('--revision', self._revision))

        if self._variant is not None:
            opts.append(('--variant', self._variant))

        if self._model_subfolder is not None:
            opts.append(('--subfolder', quote(self._model_subfolder)))

        if self._vae_path is not None:
            opts.append(('--vae', quote(self._vae_path)))

        if self._sdxl_refiner_path is not None:
            opts.append(('--sdxl-refiner', quote(self._sdxl_refiner_path)))

        if self._lora_paths is not None:
            if isinstance(self._lora_paths, str):
                opts.append(('--lora', quote(self._lora_paths)))
            else:
                opts.append(('--lora', ' '.join(quote(p) for p in self._lora_paths)))

        if self._textual_inversion_paths is not None:
            if isinstance(self._textual_inversion_paths, str):
                opts.append(('--textual-inversions', quote(self._textual_inversion_paths)))
            else:
                opts.append(('--textual-inversions', ' '.join(quote(p) for p in self._textual_inversion_paths)))

        if self._control_net_paths is not None:
            if isinstance(self._control_net_paths, str):
                opts.append(('--control-nets', quote(self._control_net_paths)))
            else:
                opts.append(('--control-nets', ' '.join(quote(p) for p in self._control_net_paths)))

        if self._scheduler is not None:
            opts.append(('--scheduler', self._scheduler))

        if sdxl_high_noise_fraction is not None:
            opts.append(('--sdxl-high-noise-fractions', sdxl_high_noise_fraction))

        if sdxl_original_size is not None:
            opts.append(('--sdxl-original-size', sdxl_original_size))

        if sdxl_target_size is not None:
            opts.append(('--sdxl-target-size', sdxl_target_size))

        if width is not None and height is not None:
            opts.append(('--output-size', f'{width}x{height}'))
        elif width is not None:
            opts.append(('--output-size', f'{width}'))

        if image is not None:
            if hasattr(image, 'filename'):
                seed_args = []

                if mask_image is not None and hasattr(mask_image, 'filename'):
                    seed_args.append(f'mask={mask_image.filename}')
                if control_image is not None and hasattr(control_image, 'filename'):
                    seed_args.append(f'control={control_image.filename}')

                if not seed_args:
                    opts.append(('--image-seeds', quote(image.filename)))
                else:
                    opts.append(('--image-seeds',
                                 quote(image.filename + ';' + ';'.join(seed_args))))

                if strength is not None:
                    opts.append(('--image-seed-strengths', strength))

                if upscaler_noise_level is not None:
                    opts.append(('--upscaler-noise-levels', upscaler_noise_level))
        elif control_image is not None:
            if hasattr(control_image, 'filename'):
                opts.append(('--control-images', quote(control_image.filename)))

        return opts

    def _pipeline_defaults(self, user_args):
        args = dict()
        args['guidance_scale'] = float(user_args.get('guidance_scale', DEFAULT_GUIDANCE_SCALE))
        args['num_inference_steps'] = user_args.get('num_inference_steps', DEFAULT_INFERENCE_STEPS)

        if self._control_net_paths is not None:
            control_image = user_args['control_image']
            if self._pipeline_type == _PipelineTypes.BASIC:
                args['image'] = control_image
            elif self._pipeline_type == _PipelineTypes.IMG2IMG or \
                    self._pipeline_type == _PipelineTypes.INPAINT:
                args['image'] = user_args['image']
                args['control_image'] = control_image
                args['strength'] = float(user_args.get('strength', DEFAULT_IMAGE_SEED_STRENGTH))

            mask_image = user_args.get('mask_image')
            if mask_image is not None:
                args['mask_image'] = mask_image

            args['width'] = user_args.get('width', control_image.width)
            args['height'] = user_args.get('height', control_image.height)

        elif 'image' in user_args:
            image = user_args['image']
            args['image'] = image

            if model_type_is_upscaler(self._model_type):
                if get_model_type_enum(self._model_type) == ModelTypes.TORCH_UPSCALER_X4:
                    args['noise_level'] = int(user_args.get('upscaler_noise_level', DEFAULT_X4_UPSCALER_NOISE_LEVEL))
            else:
                args['strength'] = float(user_args.get('strength', DEFAULT_IMAGE_SEED_STRENGTH))

            mask_image = user_args.get('mask_image')
            if mask_image is not None:
                args['mask_image'] = mask_image
                args['width'] = image.size[0]
                args['height'] = image.size[1]
        else:
            args['height'] = user_args.get('height', DEFAULT_OUTPUT_HEIGHT)
            args['width'] = user_args.get('width', DEFAULT_OUTPUT_WIDTH)

        if self._lora_scale is not None:
            args['cross_attention_kwargs'] = {'scale': self._lora_scale}

        return args

    def _get_control_net_conditioning_scale(self):
        print(self._parsed_control_net_paths)
        if not self._parsed_control_net_paths:
            return 1.0
        return [p.scale for p in self._parsed_control_net_paths] if \
            len(self._parsed_control_net_paths) > 1 else self._parsed_control_net_paths[0].scale

    def _get_control_net_guidance_start(self):
        if not self._parsed_control_net_paths:
            return 0.0
        return [p.start for p in self._parsed_control_net_paths] if \
            len(self._parsed_control_net_paths) > 1 else self._parsed_control_net_paths[0].start

    def _get_control_net_guidance_end(self):
        if not self._parsed_control_net_paths:
            return 1.0
        return [p.end for p in self._parsed_control_net_paths] if \
            len(self._parsed_control_net_paths) > 1 else self._parsed_control_net_paths[0].end

    def _call_flax_control_net(self, default_args, user_args):
        device_count = jax.device_count()

        pipe: FlaxStableDiffusionControlNetPipeline = self._pipeline

        default_args['prng_seed'] = jax.random.split(jax.random.PRNGKey(user_args.get('seed', 0)), device_count)
        prompt_ids = pipe.prepare_text_inputs([user_args.get('prompt', '')] * device_count)

        negative_prompt = user_args.get('negative_prompt', None)
        if negative_prompt is not None:
            negative_prompt_ids = pipe.prepare_text_inputs([negative_prompt] * device_count)
        else:
            negative_prompt_ids = None

        processed_image = pipe.prepare_image_inputs([default_args.get('image')] * device_count)
        default_args.pop('image')

        p_params = replicate(self._flax_params)
        prompt_ids = shard(prompt_ids)
        negative_prompt_ids = shard(negative_prompt_ids)
        processed_image = shard(processed_image)

        default_args.pop('width', None)
        default_args.pop('height', None)

        images = DiffusionPipelineWrapperBase._call_pipeline(
            pipeline=self._pipeline,
            device=self.device,
            prompt_ids=prompt_ids,
            image=processed_image,
            params=p_params,
            neg_prompt_ids=negative_prompt_ids,
            controlnet_conditioning_scale=self._get_control_net_conditioning_scale(),
            jit=True, **default_args)[0]

        return PipelineResultWrapper(
            _image_grid(self._pipeline.numpy_to_pil(images.reshape((images.shape[0],) + images.shape[-3:])),
                        device_count, 1))

    def _flax_prepare_text_input(self, text):
        tokenizer = self._pipeline.tokenizer
        text_input = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids

    def _call_flax(self, default_args, user_args):
        for arg, val in user_args.items():
            if arg.startswith('sdxl') and val is not None:
                raise NotImplementedError(
                    f'{arg.replace("_", "-")}s may only be used with SDXL models.')

        if user_args.get('guidance_rescale') is not None:
            raise NotImplementedError('--guidance-rescales is not supported when using --model-type flax.')

        if hasattr(self._pipeline, 'controlnet'):
            return self._call_flax_control_net(default_args, user_args)

        device_count = jax.device_count()

        default_args['prng_seed'] = jax.random.split(jax.random.PRNGKey(user_args.get('seed', 0)), device_count)

        processed_masks = None

        negative_prompt = user_args.get('negative_prompt', None)

        if negative_prompt is not None:
            negative_prompt_ids = shard(
                self._flax_prepare_text_input([negative_prompt] * device_count))
        else:
            negative_prompt_ids = None

        if 'image' in default_args:
            if 'mask_image' in default_args:

                prompt_ids, processed_images, processed_masks = \
                    self._pipeline.prepare_inputs(prompt=[user_args.get('prompt', '')] * device_count,
                                                  image=[default_args['image']] * device_count,
                                                  mask=[default_args['mask_image']] * device_count)

                default_args['masked_image'] = shard(processed_images)
                default_args['mask'] = shard(processed_masks)

                # inpainting pipeline does not have a strength argument, simply ignore it
                default_args.pop('strength')

                default_args.pop('image')
                default_args.pop('mask_image')
            else:
                prompt_ids, processed_images = self._pipeline.prepare_inputs(
                    prompt=[user_args.get('prompt', '')] * device_count,
                    image=[default_args['image']] * device_count)
                default_args['image'] = shard(processed_images)

            default_args['width'] = processed_images[0].shape[2]
            default_args['height'] = processed_images[0].shape[1]
        else:
            prompt_ids = self._pipeline.prepare_inputs([user_args.get('prompt', '')] * device_count)

        images = DiffusionPipelineWrapperBase._call_pipeline(
            pipeline=self._pipeline,
            device=self._device,
            prompt_ids=shard(prompt_ids),
            neg_prompt_ids=negative_prompt_ids,
            params=replicate(self._flax_params),
            **default_args, jit=True)[0]

        return PipelineResultWrapper(
            _image_grid(self._pipeline.numpy_to_pil(images.reshape((images.shape[0],) + images.shape[-3:])),
                        device_count, 1))

    def _get_non_universal_pipeline_arg(self,
                                        pipeline,
                                        default_args,
                                        user_args,
                                        pipeline_arg_name,
                                        user_arg_name,
                                        option_name, default):

        if pipeline_arg_name in inspect.getfullargspec(pipeline.__call__).args:
            val = user_args.get(user_arg_name, default)
            default_args[pipeline_arg_name] = val
            return val

        else:
            val = user_args.get(user_arg_name, None)
            if val is not None:
                raise NotImplementedError(
                    f'{option_name} cannot be used with --model-type "{self._model_type}" in '
                    f'{_describe_pipeline_type(self._pipeline_type)} mode.')
            return None

    def _get_sdxl_conditioning_args(self, pipeline, default_args, user_args, user_prefix=None):
        if user_prefix:
            user_prefix += '_'
            option_prefix = user_prefix.replace('_', '-')

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'aesthetic_score', f'sdxl_{user_prefix}aesthetic_score',
                                             f'--sdxl-{option_prefix}aesthetic-scores', None)
        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'original_size', f'sdxl_{user_prefix}original_size',
                                             f'--sdxl-{option_prefix}original-sizes', None)
        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'target_size', f'sdxl_{user_prefix}target_size',
                                             f'--sdxl-{option_prefix}target-sizes', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'crops_coords_top_left',
                                             f'sdxl_{user_prefix}crops_coords_top_left',
                                             f'--sdxl-{option_prefix}crops-coords-top-left', (0, 0))

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_aesthetic_score',
                                             f'sdxl_{user_prefix}negative_aesthetic_score',
                                             f'--sdxl-{option_prefix}negative-aesthetic-scores', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_original_size',
                                             f'sdxl_{user_prefix}negative_original_size',
                                             f'--sdxl-{option_prefix}negative-original-sizes', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_target_size',
                                             f'sdxl_{user_prefix}negative_target_size',
                                             f'--sdxl-{option_prefix}negative-target-sizes', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_crops_coords_top_left',
                                             f'sdxl_{user_prefix}negative_crops_coords_top_left',
                                             f'--sdxl-{option_prefix}negative-crops-coords-top-left', (0, 0))

    def _pop_sdxl_conditioning_args(self, default_args):
        default_args.pop('aesthetic_score', None)
        default_args.pop('target_size', None)
        default_args.pop('original_size', None)
        default_args.pop('crops_coords_top_left', None)
        default_args.pop('negative_aesthetic_score', None)
        default_args.pop('negative_target_size', None)
        default_args.pop('negative_original_size', None)
        default_args.pop('negative_crops_coords_top_left', None)

    def _call_torch(self, default_args, user_args):
        model_type = get_model_type_enum(self._model_type)

        self._get_sdxl_conditioning_args(self._pipeline, default_args, user_args)

        self._get_non_universal_pipeline_arg(self._pipeline, default_args, user_args,
                                             'guidance_rescale', 'guidance_rescale',
                                             '--guidance-rescales', None)

        if model_type != ModelTypes.TORCH_UPSCALER_X2:
            # Does not take this argument, can only produce one image
            default_args['num_images_per_prompt'] = user_args.get('num_images_per_prompt', 1)

        default_args['generator'] = torch.Generator(device=self._device).manual_seed(user_args.get('seed', 0))
        default_args['prompt'] = user_args.get('prompt', '')
        default_args['negative_prompt'] = user_args.get('negative_prompt', None)

        if isinstance(self._pipeline, StableDiffusionInpaintPipelineLegacy):
            # Not necessary, will cause an error
            default_args.pop('width')
            default_args.pop('height')

        has_control_net = hasattr(self._pipeline, 'controlnet')
        sd_edit = has_control_net or isinstance(self._pipeline,
                                                StableDiffusionXLInpaintPipeline)

        if has_control_net:
            default_args['controlnet_conditioning_scale'] = \
                self._get_control_net_conditioning_scale()

            default_args['control_guidance_start'] = \
                self._get_control_net_guidance_start()

            default_args['control_guidance_end'] = \
                self._get_control_net_guidance_end()

        if self._sdxl_refiner_pipeline is None:
            return PipelineResultWrapper(
                DiffusionPipelineWrapperBase._call_pipeline(
                    pipeline=self._pipeline,
                    device=self._device, **default_args).images[0])

        high_noise_fraction = user_args.get('sdxl_high_noise_fraction',
                                            DEFAULT_SDXL_HIGH_NOISE_FRACTION)

        if sd_edit:
            i_start = dict()
            i_end = dict()
        else:
            i_start = {'denoising_start': high_noise_fraction}
            i_end = {'denoising_end': high_noise_fraction}

        image = DiffusionPipelineWrapperBase._call_pipeline(pipeline=self._pipeline,
                                                            device=self._device,
                                                            **default_args,
                                                            **i_end,
                                                            output_type='latent').images

        default_args['image'] = image

        if not isinstance(self._sdxl_refiner_pipeline, StableDiffusionXLInpaintPipeline):
            # Width / Height not necessary for any other refiner
            if not (isinstance(self._pipeline, StableDiffusionXLImg2ImgPipeline) and
                    isinstance(self._sdxl_refiner_pipeline, StableDiffusionXLImg2ImgPipeline)):
                # Width / Height does not get passed to img2img
                default_args.pop('width')
                default_args.pop('height')

        # refiner does not use LoRA
        default_args.pop('cross_attention_kwargs', None)

        # Or any of these
        self._pop_sdxl_conditioning_args(default_args)
        default_args.pop('guidance_rescale', None)
        default_args.pop('controlnet_conditioning_scale', None)
        default_args.pop('control_guidance_start', None)
        default_args.pop('control_guidance_end', None)

        self._get_sdxl_conditioning_args(self._sdxl_refiner_pipeline,
                                         default_args, user_args,
                                         user_prefix='refiner')

        self._get_non_universal_pipeline_arg(self._pipeline, default_args, user_args,
                                             'guidance_rescale', 'sdxl_refiner_guidance_rescale',
                                             '--sdxl-refiner-guidance-rescales', None)

        if sd_edit:
            strength = 1.0 - high_noise_fraction
            default_args['strength'] = strength

        pipe_result = PipelineResultWrapper(
            DiffusionPipelineWrapperBase._call_pipeline(pipeline=self._sdxl_refiner_pipeline, device=self._device,
                                                        **default_args, **i_start).images[0])

        return pipe_result

    def _lazy_init_pipeline(self, pipeline_type):
        if self._pipeline is not None:
            return

        self._pipeline_type = pipeline_type

        model_type = get_model_type_enum(self._model_type)

        if model_type == ModelTypes.TORCH_SDXL and self._textual_inversion_paths is not None:
            raise NotImplementedError('Textual inversion not supported for SDXL.')

        if model_type == ModelTypes.FLAX:
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed.')

            if self._textual_inversion_paths is not None:
                raise NotImplementedError('Textual inversion not supported for flax.')

            if self._pipeline_type != _PipelineTypes.BASIC and self._control_net_paths:
                raise NotImplementedError('Inpaint and Img2Img not supported for flax with ControlNet.')

            self._pipeline, self._flax_params, self._parsed_control_net_paths = \
                _create_flax_diffusion_pipeline(pipeline_type,
                                                self._model_path,
                                                revision=self._revision,
                                                dtype=self._dtype,
                                                vae_path=self._vae_path,
                                                control_net_paths=self._control_net_paths,
                                                scheduler=self._scheduler,
                                                safety_checker=self._safety_checker,
                                                auth_token=self._auth_token,
                                                device=self._device)

        elif self._sdxl_refiner_path is not None:
            if model_type != ModelTypes.TORCH_SDXL:
                raise NotImplementedError('Only Stable Diffusion XL models support refiners, '
                                          'please use --model-type torch-sdxl if you are trying to load an sdxl model.')
            self._pipeline, self._parsed_control_net_paths = \
                _create_torch_diffusion_pipeline(pipeline_type,
                                                 ModelTypes.TORCH_SDXL,
                                                 self._model_path,
                                                 model_subfolder=self._model_subfolder,
                                                 revision=self._revision,
                                                 variant=self._variant,
                                                 dtype=self._dtype,
                                                 vae_path=self._vae_path,
                                                 lora_paths=self._lora_paths,
                                                 control_net_paths=self._control_net_paths,
                                                 scheduler=self._scheduler,
                                                 safety_checker=self._safety_checker,
                                                 auth_token=self._auth_token,
                                                 device=self._device)

            refiner_pipeline_type = _PipelineTypes.IMG2IMG if pipeline_type is _PipelineTypes.BASIC else pipeline_type

            self._sdxl_refiner_pipeline = \
                _create_torch_diffusion_pipeline(refiner_pipeline_type,
                                                 ModelTypes.TORCH_SDXL,
                                                 self._sdxl_refiner_path,
                                                 model_subfolder=self._sdxl_refiner_subfolder,
                                                 revision=self._sdxl_refiner_revision,

                                                 variant=self._sdxl_refiner_variant if
                                                 self._sdxl_refiner_variant is not None else self._variant,

                                                 dtype=self._sdxl_refiner_dtype if
                                                 self._sdxl_refiner_dtype is not None else self._dtype,

                                                 scheduler=self._scheduler,
                                                 safety_checker=self._safety_checker,
                                                 auth_token=self._auth_token,
                                                 extra_args={'vae': self._pipeline.vae,
                                                             'text_encoder_2': self._pipeline.text_encoder_2})[0]

        else:
            offload = self._control_net_paths and model_type == ModelTypes.TORCH_SDXL

            self._pipeline, self._parsed_control_net_paths = \
                _create_torch_diffusion_pipeline(pipeline_type,
                                                 model_type,
                                                 self._model_path,
                                                 model_subfolder=self._model_subfolder,
                                                 revision=self._revision,
                                                 variant=self._variant,
                                                 dtype=self._dtype,
                                                 vae_path=self._vae_path,
                                                 lora_paths=self._lora_paths,
                                                 textual_inversion_paths=self._textual_inversion_paths,
                                                 control_net_paths=self._control_net_paths,
                                                 scheduler=self._scheduler,
                                                 safety_checker=self._safety_checker,
                                                 auth_token=self._auth_token,
                                                 device=self._device,
                                                 sequential_cpu_offload=offload)

    def __call__(self, **kwargs) -> PipelineResultWrapper:
        default_args = self._pipeline_defaults(kwargs)

        model_type = get_model_type_enum(self._model_type)

        if model_type == ModelTypes.FLAX:
            result = self._call_flax(default_args, kwargs)
        else:
            result = self._call_torch(default_args, kwargs)

        result._dgenerate_opts = self._reconstruct_dgenerate_opts(**kwargs)
        return result


class DiffusionPipelineWrapper(DiffusionPipelineWrapperBase):
    def __init__(self,
                 model_path,
                 dtype,
                 device='cuda',
                 model_type='torch',
                 revision=None,
                 variant=None,
                 model_subfolder=None,
                 vae_path=None,
                 lora_paths=None,
                 textual_inversion_paths=None,
                 control_net_paths=None,
                 sdxl_refiner_path=None,
                 scheduler=None,
                 safety_checker=False,
                 auth_token=None):
        super().__init__(
            model_path,
            dtype,
            device,
            model_type,
            revision,
            variant,
            model_subfolder,
            vae_path,
            lora_paths,
            textual_inversion_paths,
            control_net_paths,
            sdxl_refiner_path,
            scheduler,
            safety_checker,
            auth_token)

    def __call__(self, **kwargs) -> PipelineResultWrapper:
        self._lazy_init_pipeline(_PipelineTypes.BASIC)

        return super().__call__(**kwargs)


class DiffusionPipelineImg2ImgWrapper(DiffusionPipelineWrapperBase):
    def __init__(self,
                 model_path,
                 dtype,
                 device='cuda',
                 model_type='torch',
                 revision=None,
                 variant=None,
                 model_subfolder=None,
                 vae_path=None,
                 lora_paths=None,
                 textual_inversion_paths=None,
                 control_net_paths=None,
                 sdxl_refiner_path=None,
                 scheduler=None,
                 safety_checker=False,
                 auth_token=None):

        super().__init__(
            model_path,
            dtype,
            device,
            model_type,
            revision,
            variant,
            model_subfolder,
            vae_path,
            lora_paths,
            textual_inversion_paths,
            control_net_paths,
            sdxl_refiner_path,
            scheduler,
            safety_checker,
            auth_token)

    def __call__(self, **kwargs) -> PipelineResultWrapper:
        if 'mask_image' in kwargs:
            self._lazy_init_pipeline(_PipelineTypes.INPAINT)
        else:
            self._lazy_init_pipeline(_PipelineTypes.IMG2IMG)

        return super().__call__(**kwargs)
