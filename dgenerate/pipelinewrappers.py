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
import os

try:
    import jax
    import jax.numpy as jnp
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard
    from diffusers import FlaxStableDiffusionPipeline, FlaxStableDiffusionImg2ImgPipeline, \
        FlaxStableDiffusionInpaintPipeline, FlaxAutoencoderKL

    _have_jax_flax = True

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
except ImportError:
    _have_jax_flax = False

import torch
import numbers
from PIL import Image
from .textprocessing import ConceptModelPathParser, ConceptModelPathParseError, quote
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, \
    StableDiffusionInpaintPipelineLegacy, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, \
    StableDiffusionXLInpaintPipeline, StableDiffusionUpscalePipeline ,\
    AutoencoderKL, AsymmetricAutoencoderKL, AutoencoderTiny

_TORCH_MODEL_CACHE = dict()
_FLAX_MODEL_CACHE = dict()


class InvalidSDXLRefinerPathError(Exception):
    pass


class InvalidVaePathError(Exception):
    pass


class InvalidSchedulerName(Exception):
    pass


class InvalidLoRAPathError(Exception):
    pass


class InvalidTextualInversionPathError(Exception):
    pass


_sdxl_refiner_path_parser = ConceptModelPathParser('SDXL Refiner', ['revision', 'variant', 'subfolder' 'dtype'])
_torch_vae_path_parser = ConceptModelPathParser('VAE', ['model', 'revision', 'variant', 'subfolder', 'dtype'])
_flax_vae_path_parser = ConceptModelPathParser('VAE', ['model', 'revision', 'subfolder', 'dtype'])
_lora_path_parser = ConceptModelPathParser('LoRA', ['scale', 'revision', 'subfolder', 'weight-name'])
_textual_inversion_path_parser = ConceptModelPathParser('Textual Inversion',
                                                        ['revision', 'subfolder', 'weight-name'])


class SDXLRefinerPath:
    def __init__(self, model, revision, variant, dtype, subfolder):
        self.model = model
        self.revision = revision
        self.variant = variant
        self.dtype = dtype
        self.subfolder = subfolder


def parse_sdxl_refiner_path(path):
    try:
        r = _sdxl_refiner_path_parser.parse_concept_path(path)

        dtype = r.args.get('dtype', None)
        if dtype not in {'float32', 'float16', 'auto', None}:
            raise InvalidSDXLRefinerPathError(
                'Torch SDXL refiner dtype must be float32, float16, auto, or left undefined.')

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


def parse_torch_vae_path(path):
    try:
        r = _torch_vae_path_parser.parse_concept_path(path)

        model = r.args.get('model')
        if model is None:
            raise InvalidVaePathError('model argument for torch VAE specification must be defined.')

        dtype = r.args.get('dtype', None)
        if dtype not in {'float32', 'float16', 'auto', None}:
            raise InvalidVaePathError('Torch VAE dtype must be float32, float16, auto, or left undefined.')

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


def parse_flax_vae_path(path):
    try:
        r = _flax_vae_path_parser.parse_concept_path(path)

        model = r.args.get('model')
        if model is None:
            raise InvalidVaePathError('model argument for flax VAE specification must be defined.')

        dtype = r.args.get('dtype', None)
        if dtype not in {'float32', 'float16', 'auto', None}:
            raise InvalidVaePathError('Flax VAE dtype must be float32, float16, auto, or left undefined.')

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


def parse_lora_path(path):
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


def parse_textual_inversion_path(path):
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

    if len(ext) == 0:
        return False

    if ext in {'.pt', '.pth', '.bin', '.msgpack', '.ckpt', '.safetensors'}:
        return True

    return False


def _load_pytorch_vae(path,
                      torch_dtype_fallback,
                      use_auth_token):
    parsed_concept = parse_torch_vae_path(path)

    if parsed_concept.dtype is None:
        parsed_concept.dtype = torch_dtype_fallback

    encoder_name = parsed_concept.encoder

    if encoder_name == "AutoencoderKL":
        encoder = AutoencoderKL
    elif encoder_name == "AsymmetricAutoencoderKL":
        encoder = AsymmetricAutoencoderKL
    elif encoder_name == "AutoencoderTiny":
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
            return encoder.from_single_file(path, revision=parsed_concept.revision). \
                to(device=None, dtype=parsed_concept.dtype, non_blocking=False)
        else:
            return encoder.from_single_file(path,
                                            revision=parsed_concept.revision,
                                            torch_dtype=parsed_concept.dtype)

    else:
        return encoder.from_pretrained(path,
                                       revision=parsed_concept.revision,
                                       variant=parsed_concept.variant,
                                       torch_dtype=parsed_concept.dtype,
                                       subfolder=parsed_concept.subfolder,
                                       use_auth_token=use_auth_token)


def _load_flax_vae(path,
                   flax_dtype_fallback,
                   use_auth_token):
    parsed_concept = parse_torch_vae_path(path)

    if parsed_concept.dtype is None:
        parsed_concept.dtype = flax_dtype_fallback

    encoder_name = parsed_concept.encoder

    if encoder_name == "FlaxAutoencoderKL":
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
        return encoder.from_single_file(path,
                                        revision=parsed_concept.revision,
                                        dtype=parsed_concept.dtype)
    else:
        return encoder.from_pretrained(path,
                                       revision=parsed_concept.revision,
                                       dtype=parsed_concept.dtype,
                                       subfolder=parsed_concept.subfolder,
                                       use_auth_token=use_auth_token)


def _load_scheduler(pipeline, scheduler_name=None):
    if scheduler_name is None:
        return

    for i in pipeline.scheduler.compatibles:
        if i.__name__.endswith(scheduler_name):
            pipeline.scheduler = i.from_config(pipeline.scheduler.config)
            return

    raise InvalidSchedulerName(f'Scheduler named "{scheduler_name}" is not a valid compatible scheduler, '
                               f'options are:\n\n{chr(10).join(sorted(i.__name__.split(".")[-1] for i in pipeline.scheduler.compatibles))}')


def clear_model_cache():
    _TORCH_MODEL_CACHE.clear()
    _FLAX_MODEL_CACHE.clear()


def _disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


def _pipeline_cache_key(args_dict):
    def value_hash(obj):
        if isinstance(obj, dict):
            return '{' + _pipeline_cache_key(obj) + '}'
        elif obj is None or isinstance(obj, (str, numbers.Number)):
            return str(obj)
        else:
            return f'<{obj.__class__.__name__}:{str(id(obj))}>'

    return ','.join(f'{k}={value_hash(v)}' for k, v in sorted(args_dict.items()))


class _PipelineTypes:
    BASIC = 1
    IMG2IMG = 2
    INPAINT = 3


def _create_torch_diffusion_pipeline(pipeline_type,
                                     model_path,
                                     revision,
                                     variant,
                                     dtype,
                                     model_subfolder=None,
                                     vae_path=None,
                                     lora_paths=None,
                                     textual_inversion_paths=None,
                                     scheduler=None,
                                     safety_checker=False,
                                     sdxl=False,
                                     auth_token=None,
                                     extra_args=None):
    cache_key = _pipeline_cache_key(locals())

    if pipeline_type == _PipelineTypes.BASIC:
        pipeline_class = StableDiffusionXLPipeline if sdxl else StableDiffusionPipeline
    elif pipeline_type == _PipelineTypes.IMG2IMG:
        pipeline_class = StableDiffusionXLImg2ImgPipeline if sdxl else StableDiffusionImg2ImgPipeline
    elif pipeline_type == _PipelineTypes.INPAINT:
        pipeline_class = StableDiffusionXLInpaintPipeline if sdxl else StableDiffusionInpaintPipeline
    else:
        raise NotImplementedError('Pipeline type not implemented')

    catch_hit = _TORCH_MODEL_CACHE.get(cache_key)

    if catch_hit is None:
        kwargs = {}

        torch_dtype = _get_torch_dtype(dtype)

        if vae_path is not None:
            kwargs['vae'] = _load_pytorch_vae(vae_path,
                                              torch_dtype_fallback=torch_dtype,
                                              use_auth_token=auth_token)

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

        if textual_inversion_paths:
            if isinstance(textual_inversion_paths, str):
                textual_inversion_paths = [textual_inversion_paths]

            for inversion_path in textual_inversion_paths:
                parse_textual_inversion_path(inversion_path). \
                    load_on_pipeline(pipeline, use_auth_token=auth_token)

        if lora_paths is not None:
            if isinstance(lora_paths, str):
                parse_lora_path(lora_paths). \
                    load_on_pipeline(pipeline, use_auth_token=auth_token)
            else:
                raise NotImplementedError('Using multiple LoRA models is currently not supported.')

        if not safety_checker:
            if hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None:
                # If it's already None for some reason you'll get a call
                # to an unassigned feature_extractor by assigning it a value

                # The attribute will not exist for SDXL pipelines currently

                pipeline.safety_checker = _disabled_safety_checker

        _TORCH_MODEL_CACHE[cache_key] = pipeline
        return pipeline
    else:
        return catch_hit


def _create_flax_diffusion_pipeline(pipeline_type,
                                    model_path,
                                    revision,
                                    dtype,
                                    model_subfolder=None,
                                    vae_path=None,
                                    scheduler=None,
                                    safety_checker=False,
                                    auth_token=None,
                                    extra_args=None):
    cache_key = _pipeline_cache_key(locals())

    if pipeline_type == _PipelineTypes.BASIC:
        pipeline_class = FlaxStableDiffusionPipeline
    elif pipeline_type == _PipelineTypes.IMG2IMG:
        pipeline_class = FlaxStableDiffusionImg2ImgPipeline
    elif pipeline_type == _PipelineTypes.INPAINT:
        pipeline_class = FlaxStableDiffusionInpaintPipeline
    else:
        raise NotImplementedError('Pipeline type not implemented')

    catch_hit = _FLAX_MODEL_CACHE.get(cache_key)

    if catch_hit is None:
        kwargs = {}
        vae_params = None

        flax_dtype = _get_flax_dtype(dtype)

        if vae_path is not None:
            vae_path, vae_params = _load_flax_vae(vae_path,
                                                  flax_dtype_fallback=flax_dtype,
                                                  use_auth_token=auth_token)
            kwargs['vae'] = vae_path

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

        _load_scheduler(pipeline, scheduler)

        if not safety_checker:
            pipeline.safety_checker = None

        _FLAX_MODEL_CACHE[cache_key] = (pipeline, params)
        return pipeline, params
    else:
        return catch_hit


def supported_model_types():
    if have_jax_flax():
        return ['torch', 'torch-sdxl', 'flax']
    else:
        return ['torch', 'torch-sdxl']


def have_jax_flax():
    return _have_jax_flax


def _get_flax_dtype(dtype):
    if dtype is None:
        return None
    return {'float16': jnp.bfloat16,
            'float32': jnp.float32,
            'float64': jnp.float64,
            'auto': None}[dtype.lower()]


def _get_torch_dtype(dtype):
    if dtype is None:
        return None
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
        self._sdxl_refiner_path = sdxl_refiner_path
        self._sdxl_refiner_pipeline = None
        self._auth_token = auth_token

        if sdxl_refiner_path is not None:
            parsed_path = parse_sdxl_refiner_path(sdxl_refiner_path)
            self._sdxl_refiner_revision = parsed_path.revision
            self._sdxl_refiner_variant = parsed_path.variant
            self._sdxl_refiner_dtype = parsed_path.dtype
            self._sdxl_refiner_subfolder = parsed_path.subfolder

        if lora_paths is not None:
            if model_type == "flax":
                raise NotImplementedError("LoRA loading is not implemented for flax.")

            if not isinstance(lora_paths, str):
                raise NotImplementedError('Using multiple LoRA models is currently not supported.')

            self._lora_scale = parse_lora_path(lora_paths).scale

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
        strength = kwargs.get('strength', None)
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

        if image is not None and hasattr(image, 'filename'):
            if mask_image is not None and hasattr(mask_image, 'filename'):
                opts.append(('--image-seeds', f'"{image.filename};{mask_image.filename}"'))
            else:
                opts.append(('--image-seeds', quote(image.filename)))

            if strength is not None:
                opts.append(('--image-seed-strengths', strength))

        return opts

    def _pipeline_defaults(self, kwargs):
        args = dict()
        args['guidance_scale'] = float(kwargs.get('guidance_scale', 5))
        args['num_inference_steps'] = kwargs.get('num_inference_steps', 30)
        if 'image' in kwargs:
            image = kwargs['image']
            args['image'] = image
            args['strength'] = float(kwargs.get('strength', 0.8))
            mask_image = kwargs.get('mask_image')
            if mask_image is not None:
                args['mask_image'] = mask_image
                args['width'] = image.size[0]
                args['height'] = image.size[1]
        else:
            args['height'] = kwargs.get('height', 512)
            args['width'] = kwargs.get('width', 512)

        if self._lora_scale is not None:
            args['cross_attention_kwargs'] = {'scale': self._lora_scale}

        return args

    def _call_flax(self, args, kwargs):
        if kwargs.get('sdxl_original_size', None) is not None:
            raise NotImplementedError('original-size micro-conditioning may only be used with SDXL models.')
        if kwargs.get('sdxl_target_size', None) is not None:
            raise NotImplementedError('target-size micro-conditioning may only be used with SDXL models.')

        device_count = jax.device_count()

        args['prng_seed'] = jax.random.split(jax.random.PRNGKey(kwargs.get('seed', 0)), device_count)

        processed_masks = None

        if 'image' in args:
            if 'mask_image' in args:

                prompt_ids, processed_images, processed_masks = \
                    self._pipeline.prepare_inputs(prompt=[kwargs.get('prompt', '')] * device_count,
                                                  image=[args['image']] * device_count,
                                                  mask=[args['mask_image']] * device_count)

                args['masked_image'] = shard(processed_images)
                args['mask'] = shard(processed_masks)

                args.pop('strength')
                args.pop('image')
                args.pop('mask_image')
            else:
                prompt_ids, processed_images = self._pipeline.prepare_inputs(
                    prompt=[kwargs.get('prompt', '')] * device_count,
                    image=[args['image']] * device_count)
                args['image'] = shard(processed_images)

            args['width'] = processed_images[0].shape[2]
            args['height'] = processed_images[0].shape[1]
        else:
            prompt_ids = self._pipeline.prepare_inputs([kwargs.get('prompt', '')] * device_count)

        images = self._pipeline(prompt_ids=shard(prompt_ids), params=replicate(self._flax_params),
                                **args, jit=True)[0]

        return PipelineResultWrapper(
            _image_grid(self._pipeline.numpy_to_pil(images.reshape((images.shape[0],) + images.shape[-3:])),
                        device_count, 1))

    def _call_torch(self, args, kwargs):
        sdxl_original_size = kwargs.get('sdxl_original_size', None)
        sdxl_target_size = kwargs.get('sdxl_target_size', None)

        if self._model_type != 'torch-sdxl':
            if sdxl_original_size is not None:
                raise NotImplementedError('original-size micro-conditioning may only be used with SDXL models.')
            if sdxl_target_size is not None:
                raise NotImplementedError('target-size micro-conditioning may only be used with SDXL models.')
        else:
            if sdxl_original_size is not None:
                args['target_size'] = sdxl_target_size
            if sdxl_target_size is not None:
                args['original_size'] = sdxl_original_size

        args['num_images_per_prompt'] = kwargs.get('num_images_per_prompt', 1)
        args['generator'] = torch.Generator(device=self._device).manual_seed(kwargs.get('seed', 0))
        args['prompt'] = kwargs.get('prompt', '')
        args['negative_prompt'] = kwargs.get('negative_prompt', None)

        if isinstance(self._pipeline, StableDiffusionInpaintPipelineLegacy):
            # Not necessary, will cause an error
            args.pop('width')
            args.pop('height')

        if self._sdxl_refiner_pipeline is not None:
            high_noise_fraction = kwargs.get('sdxl_high_noise_fraction', 0.8)
            image = self._pipeline(**args,
                                   denoising_end=high_noise_fraction,
                                   output_type="latent").images

            args['image'] = image

            if not isinstance(self._sdxl_refiner_pipeline, StableDiffusionXLInpaintPipeline):
                # Width / Height not necessary for any other refiner
                if not (isinstance(self._pipeline, StableDiffusionXLImg2ImgPipeline) and
                        isinstance(self._sdxl_refiner_pipeline, StableDiffusionXLImg2ImgPipeline)):
                    # Width / Height does not get passed to img2img
                    args.pop('width')
                    args.pop('height')

            # refiner does not use LoRA
            args.pop('cross_attention_kwargs', None)

            return PipelineResultWrapper(
                self._sdxl_refiner_pipeline(**args, denoising_start=high_noise_fraction).images[0])

        else:
            return PipelineResultWrapper(self._pipeline(**args).images[0])

    def _lazy_init_pipeline(self, pipeline_type):
        if self._pipeline is not None:
            return

        if self._model_type == 'torch-sdxl' and self._textual_inversion_paths is not None:
            raise NotImplementedError('Textual inversion not supported for SDXL')

        if self._model_type == 'flax':
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed')

            if self._textual_inversion_paths is not None:
                raise NotImplementedError('Textual inversion not supported for flax')

            self._pipeline, self._flax_params = \
                _create_flax_diffusion_pipeline(pipeline_type,
                                                self._model_path,
                                                revision=self._revision,
                                                dtype=self._dtype,
                                                vae_path=self._vae_path,
                                                scheduler=self._scheduler,
                                                safety_checker=self._safety_checker,
                                                auth_token=self._auth_token)

        elif self._sdxl_refiner_path is not None:
            if self._model_type != 'torch-sdxl':
                raise NotImplementedError('Only Stable Diffusion XL models support refiners, '
                                          'please use --model-type torch-sdxl if you are trying to load an sdxl model.')

            self._pipeline = \
                _create_torch_diffusion_pipeline(pipeline_type,
                                                 self._model_path,
                                                 model_subfolder=self._model_subfolder,
                                                 sdxl=True,
                                                 revision=self._revision,
                                                 variant=self._variant,
                                                 dtype=self._dtype,
                                                 vae_path=self._vae_path,
                                                 lora_paths=self._lora_paths,
                                                 scheduler=self._scheduler,
                                                 safety_checker=self._safety_checker,
                                                 auth_token=self._auth_token)

            refiner_pipeline_type = _PipelineTypes.IMG2IMG if pipeline_type is _PipelineTypes.BASIC else pipeline_type

            self._sdxl_refiner_pipeline = \
                _create_torch_diffusion_pipeline(refiner_pipeline_type,
                                                 self._sdxl_refiner_path,
                                                 model_subfolder=self._sdxl_refiner_subfolder,
                                                 sdxl=True,

                                                 revision=self._sdxl_refiner_revision,

                                                 variant=self._sdxl_refiner_variant if
                                                 self._sdxl_refiner_variant is not None else self._variant,

                                                 dtype=self._sdxl_refiner_dtype if
                                                 self._sdxl_refiner_dtype is not None else self._dtype,

                                                 scheduler=self._scheduler,
                                                 safety_checker=self._safety_checker,
                                                 auth_token=self._auth_token,
                                                 extra_args={'vae': self._pipeline.vae,
                                                             'text_encoder_2': self._pipeline.text_encoder_2})

            if self._device.startswith('cuda'):
                gpu_id = _gpu_id_from_cuda_device(self._device)
                self._pipeline.enable_model_cpu_offload(gpu_id)
                self._sdxl_refiner_pipeline.enable_model_cpu_offload(gpu_id)

        else:
            self._pipeline = \
                _create_torch_diffusion_pipeline(pipeline_type,
                                                 self._model_path,
                                                 model_subfolder=self._model_subfolder,
                                                 sdxl=self._model_type == 'torch-sdxl',
                                                 revision=self._revision,
                                                 variant=self._variant,
                                                 dtype=self._dtype,
                                                 vae_path=self._vae_path,
                                                 lora_paths=self._lora_paths,
                                                 textual_inversion_paths=self._textual_inversion_paths,
                                                 scheduler=self._scheduler,
                                                 safety_checker=self._safety_checker,
                                                 auth_token=self._auth_token).to(self._device)

    def __call__(self, **kwargs):
        args = self._pipeline_defaults(kwargs)
        if self._model_type == 'flax':
            result = self._call_flax(args, kwargs)
        else:
            result = self._call_torch(args, kwargs)

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
            sdxl_refiner_path,
            scheduler,
            safety_checker,
            auth_token)

    def __call__(self, **kwargs):
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
            sdxl_refiner_path,
            scheduler,
            safety_checker,
            auth_token)

    def __call__(self, **kwargs):
        if 'mask_image' in kwargs:
            self._lazy_init_pipeline(_PipelineTypes.INPAINT)
        else:
            self._lazy_init_pipeline(_PipelineTypes.IMG2IMG)

        return super().__call__(**kwargs)
