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
from collections import namedtuple

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
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, \
    StableDiffusionInpaintPipelineLegacy, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, \
    StableDiffusionXLInpaintPipeline, \
    AutoencoderKL, AsymmetricAutoencoderKL, AutoencoderTiny

_TORCH_MODEL_CACHE = dict()
_FLAX_MODEL_CACHE = dict()


class InvalidVaePathError(Exception):
    pass


class InvalidSchedulerName(Exception):
    pass


def _is_single_file_model_load(path):
    ext = os.path.splitext(path)[1]
    if len(ext) == 0:
        return False

    try:
        int(ext)
    except ValueError:
        return False

    return True


def _load_pytorch_vae(path,
                      revision,
                      variant,
                      torch_dtype,
                      subfolder,
                      use_auth_token):

    parts = path.split(';', 1)

    if len(parts) != 2:
        raise InvalidVaePathError(f'VAE path must contain auto encoder class name and path to the encoder URL or file.')

    encoder_name = parts[0].strip()

    if encoder_name == "AutoencoderKL":
        encoder = AutoencoderKL
    elif encoder_name == "AsymmetricAutoencoderKL":
        encoder = AsymmetricAutoencoderKL
    elif encoder_name == "AutoencoderTiny":
        encoder = AutoencoderTiny
    else:
        raise InvalidVaePathError(f'Unknown VAE encoder class {encoder_name}')

    path = parts[1].strip()

    can_single_file_load = hasattr(encoder, 'from_single_file')
    single_file_load_path = _is_single_file_model_load(path)

    if single_file_load_path and not can_single_file_load:
        raise NotImplementedError(f'{encoder_name} is not capable of loading from a single file, '
                                  f'must be loaded from a huggingface repository slug or folder on disk.')

    if single_file_load_path:
        if subfolder is not None:
            raise NotImplementedError('Single file VAE loads do not support the subfolder option.')
        return encoder.from_single_file(path,
                                        revision=revision,
                                        torch_dtype=torch_dtype)
    else:
        return encoder.from_pretrained(path,
                                       revision=revision,
                                       variant=variant,
                                       torch_dtype=torch_dtype,
                                       subfolder=subfolder,
                                       use_auth_token=use_auth_token)


class InvalidLoRAPathError(Exception):
    pass


def _parse_lora(path):
    parts = path.split(';', 1)

    try:
        if len(parts) == 2:
            return parts[0].strip(), float(parts[1].strip())
        else:
            return parts[0].strip(), 1.0
    except Exception as e:
        raise InvalidLoRAPathError(e)


def _load_flax_vae(path,
                   revision,
                   flax_dtype,
                   subfolder,
                   use_auth_token):
    parts = path.split(';', 1)

    if len(parts) != 2:
        raise InvalidVaePathError(f'VAE path must contain auto encoder class name and path to the encoder URL or file.')

    encoder_name = parts[0].strip()

    if encoder_name == "FlaxAutoencoderKL":
        encoder = FlaxAutoencoderKL
    else:
        raise InvalidVaePathError(f'Unknown VAE flax encoder class {encoder_name}')

    path = parts[1].strip()

    if _is_single_file_model_load(path):
        if subfolder is not None:
            raise NotImplementedError('Single file VAE loads do not support the subfolder option.')
        return encoder.from_single_file(path,
                                        revision=revision,
                                        dtype=flax_dtype)
    else:
        return encoder.from_pretrained(path,
                                       revision=revision,
                                       dtype=flax_dtype,
                                       subfolder=subfolder,
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
                                     torch_dtype,
                                     model_subfolder=None,
                                     vae=None,
                                     vae_revision=None,
                                     vae_variant=None,
                                     vae_torch_dtype=None,
                                     vae_subfolder=None,
                                     lora=None,
                                     lora_weight_name=None,
                                     lora_revision=None,
                                     lora_subfolder=None,
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

        if vae is not None:
            kwargs['vae'] = _load_pytorch_vae(vae,
                                              revision=vae_revision,
                                              variant=vae_variant,
                                              torch_dtype=vae_torch_dtype if vae_torch_dtype is not None else torch_dtype,
                                              subfolder=vae_subfolder,
                                              use_auth_token=auth_token)

        if extra_args is not None:
            kwargs.update(extra_args)

        if _is_single_file_model_load(model_path):
            print(model_path)
            if model_subfolder is not None:
                raise NotImplementedError('Single file model loads do not support the subfolder option.')
            pipeline = pipeline_class.from_single_file(model_path,
                                                       revision=revision,
                                                       variant=variant,
                                                       torch_dtype=torch_dtype,
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

        if lora is not None:
            pipeline.load_lora_weights(lora,
                                       weight_name=lora_weight_name,
                                       revision=lora_revision,
                                       subfolder=lora_subfolder,
                                       use_auth_token=auth_token)

        if not safety_checker:
            pipeline.safety_checker = _disabled_safety_checker

        _TORCH_MODEL_CACHE[cache_key] = pipeline
        return pipeline
    else:
        return catch_hit


def _create_flax_diffusion_pipeline(pipeline_type,
                                    model_path,
                                    revision,
                                    flax_dtype,
                                    model_subfolder=None,
                                    vae=None,
                                    vae_revision=None,
                                    vae_flax_dtype=None,
                                    vae_subfolder=None,
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

        if vae is not None:
            kwargs['vae'] = _load_flax_vae(vae,
                                           revision=vae_revision,
                                           flax_dtype=vae_flax_dtype if vae_flax_dtype is not None else flax_dtype,
                                           subfolder=vae_subfolder,
                                           use_auth_token=auth_token)

        if extra_args is not None:
            kwargs.update(extra_args)

        pipeline, params = pipeline_class.from_pretrained(model_path,
                                                          revision=revision,
                                                          dtype=flax_dtype,
                                                          subfolder=model_subfolder,
                                                          use_auth_token=auth_token,
                                                          **kwargs)

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
    def __init__(self, images):
        self.images = images


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
                 vae=None,
                 vae_revision=None,
                 vae_variant=None,
                 vae_dtype=None,
                 vae_subfolder=None,
                 lora=None,
                 lora_weight_name=None,
                 lora_revision=None,
                 lora_subfolder=None,
                 scheduler=None,
                 safety_checker=False,
                 sdxl_refiner_path=None,
                 sdxl_refiner_revision=None,
                 sdxl_refiner_variant=None,
                 sdxl_refiner_dtype=None,
                 sdxl_refiner_subfolder=None,
                 auth_token=None):

        self._vae_subfolder = vae_subfolder
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
        self._vae = vae
        self._vae_revision = vae_revision
        self._vae_variant = vae_variant
        self._vae_dtype = vae_dtype
        self._safety_checker = safety_checker
        self._scheduler = scheduler
        self._lora = lora
        self._lora_weight_name = lora_weight_name
        self._lora_scale = None
        self._lora_revision = lora_revision
        self._lora_subfolder = lora_subfolder
        self._sdxl_refiner_path = sdxl_refiner_path
        self._sdxl_refiner_pipeline = None
        self._sdxl_refiner_revision = sdxl_refiner_revision
        self._sdxl_refiner_variant = sdxl_refiner_variant
        self._sdxl_refiner_dtype = sdxl_refiner_dtype
        self._sdxl_refiner_subfolder = sdxl_refiner_subfolder
        self._auth_token = auth_token

        if lora is not None:
            if model_type == "flax":
                raise NotImplementedError("LoRA loading is not implemented for flax.")
            self._lora, self._lora_scale = _parse_lora(lora)

    @property
    def vae_revision(self):
        return self._vae_revision

    @property
    def safety_checker(self):
        return self._safety_checker

    @property
    def dtype(self):
        return self._dtype

    @property
    def sdxl_refiner_dtype(self):
        return self._sdxl_refiner_dtype

    @property
    def sdxl_refiner_variant(self):
        return self._sdxl_refiner_variant

    @property
    def model_path(self):
        return self._model_path

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def flax_params(self):
        return self._flax_params

    @property
    def sdxl_refiner_path(self):
        return self._sdxl_refiner_path

    @property
    def revision(self):
        return self._revision

    @property
    def lora_weight_name(self):
        return self._lora_weight_name

    @property
    def lora_revision(self):
        return self._lora_revision

    @property
    def sdxl_refiner_pipeline(self):
        return self._sdxl_refiner_pipeline

    @property
    def variant(self):
        return self._variant

    @property
    def sdxl_refiner_subfolder(self):
        return self._sdxl_refiner_subfolder

    @property
    def device(self):
        return self._device

    @property
    def vae_subfolder(self):
        return self._vae_subfolder

    @property
    def sdxl_refiner_revision(self):
        return self._sdxl_refiner_revision

    @property
    def lora_scale(self):
        return self._lora_scale

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def vae_variant(self):
        return self._vae_variant

    @property
    def vae(self):
        return self._vae

    @property
    def model_type(self):
        return self._model_type

    @property
    def lora_subfolder(self):
        return self._lora_subfolder

    @property
    def model_subfolder(self):
        return self._model_subfolder

    @property
    def lora(self):
        return self._lora

    @property
    def vae_dtype(self):
        return self._vae_dtype

    @property
    def auth_token(self):
        return self._auth_token

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
            [_image_grid(self._pipeline.numpy_to_pil(images.reshape((images.shape[0],) + images.shape[-3:])),
                         device_count, 1)])

    def _call_torch(self, args, kwargs):
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
                self._sdxl_refiner_pipeline(**args, denoising_start=high_noise_fraction).images)

        else:
            return PipelineResultWrapper(self._pipeline(**args).images)

    def _lazy_init_pipeline(self, pipeline_type):
        if self._pipeline is not None:
            return

        if self._model_type == 'flax':
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed')

            self._pipeline, self._flax_params = \
                _create_flax_diffusion_pipeline(pipeline_type,
                                                self._model_path,
                                                revision=self._revision,
                                                flax_dtype=_get_flax_dtype(self._dtype),
                                                vae=self._vae,
                                                vae_revision=self._vae_revision,
                                                vae_subfolder=self._vae_subfolder,
                                                vae_flax_dtype=_get_flax_dtype(self._vae_dtype),
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
                                                 torch_dtype=_get_torch_dtype(self._dtype),
                                                 vae=self._vae,
                                                 vae_revision=self._vae_revision,
                                                 vae_variant=self._vae_variant,
                                                 vae_torch_dtype=_get_torch_dtype(self._vae_dtype),
                                                 vae_subfolder=self._vae_subfolder,
                                                 lora=self._lora,
                                                 lora_weight_name=self._lora_weight_name,
                                                 lora_revision=self._lora_revision,
                                                 lora_subfolder=self._lora_subfolder,
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

                                                 torch_dtype=_get_torch_dtype(self._sdxl_refiner_dtype) if
                                                 self._sdxl_refiner_dtype is not None else _get_torch_dtype(
                                                     self._dtype),

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
                                                 torch_dtype=_get_torch_dtype(self._dtype),
                                                 vae=self._vae,
                                                 vae_revision=self._vae_revision,
                                                 vae_variant=self._vae_variant,
                                                 vae_torch_dtype=_get_torch_dtype(self._vae_dtype),
                                                 vae_subfolder=self._vae_subfolder,
                                                 lora=self._lora,
                                                 lora_weight_name=self._lora_weight_name,
                                                 lora_revision=self._lora_revision,
                                                 lora_subfolder=self._lora_subfolder,
                                                 scheduler=self._scheduler,
                                                 safety_checker=self._safety_checker,
                                                 auth_token=self._auth_token).to(self._device)

    def __call__(self, **kwargs):
        args = self._pipeline_defaults(kwargs)
        if self._model_type == 'flax':
            return self._call_flax(args, kwargs)
        else:
            return self._call_torch(args, kwargs)


class DiffusionPipelineWrapper(DiffusionPipelineWrapperBase):
    def __init__(self,
                 model_path,
                 dtype,
                 device='cuda',
                 model_type='torch',
                 revision=None,
                 variant=None,
                 model_subfolder=None,
                 vae=None,
                 vae_revision=None,
                 vae_variant=None,
                 vae_dtype=None,
                 vae_subfolder=None,
                 lora=None,
                 lora_weight_name=None,
                 lora_revision=None,
                 lora_subfolder=None,
                 scheduler=None,
                 safety_checker=False,
                 sdxl_refiner_path=None,
                 sdxl_refiner_revision=None,
                 sdxl_refiner_variant=None,
                 sdxl_refiner_dtype=None,
                 sdxl_refiner_subfolder=None,
                 auth_token=None):
        super().__init__(
            model_path,
            dtype,
            device,
            model_type,
            revision,
            variant,
            model_subfolder,
            vae,
            vae_revision,
            vae_variant,
            vae_dtype,
            vae_subfolder,
            lora,
            lora_weight_name,
            lora_revision,
            lora_subfolder,
            scheduler,
            safety_checker,
            sdxl_refiner_path,
            sdxl_refiner_revision,
            sdxl_refiner_variant,
            sdxl_refiner_dtype,
            sdxl_refiner_subfolder,
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
                 vae=None,
                 vae_revision=None,
                 vae_variant=None,
                 vae_dtype=None,
                 vae_subfolder=None,
                 lora=None,
                 lora_weight_name=None,
                 lora_revision=None,
                 lora_subfolder=None,
                 scheduler=None,
                 safety_checker=False,
                 sdxl_refiner_path=None,
                 sdxl_refiner_revision=None,
                 sdxl_refiner_variant=None,
                 sdxl_refiner_dtype=None,
                 sdxl_refiner_subfolder=None,
                 auth_token=None):

        super().__init__(
            model_path,
            dtype,
            device,
            model_type,
            revision,
            variant,
            model_subfolder,
            vae,
            vae_revision,
            vae_variant,
            vae_dtype,
            vae_subfolder,
            lora,
            lora_weight_name,
            lora_revision,
            lora_subfolder,
            scheduler,
            safety_checker,
            sdxl_refiner_path,
            sdxl_refiner_revision,
            sdxl_refiner_variant,
            sdxl_refiner_dtype,
            sdxl_refiner_subfolder,
            auth_token)

    def __call__(self, **kwargs):
        if 'mask_image' in kwargs:
            self._lazy_init_pipeline(_PipelineTypes.INPAINT)
        else:
            self._lazy_init_pipeline(_PipelineTypes.IMG2IMG)

        return super().__call__(**kwargs)
