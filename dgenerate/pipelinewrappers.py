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

try:
    import jax
    import jax.numpy as jnp
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard
    from diffusers import FlaxStableDiffusionPipeline, FlaxStableDiffusionImg2ImgPipeline, \
        FlaxStableDiffusionInpaintPipeline, FlaxAutoencoderKL

    _have_jax_flax = True

    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
except ImportError:
    _have_jax_flax = False

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, \
    StableDiffusionInpaintPipelineLegacy, AutoencoderKL, AsymmetricAutoencoderKL, AutoencoderTiny

_TORCH_MODEL_CACHE = dict()
_FLAX_MODEL_CACHE = dict()

_TORCH_IMG2IMG_MODEL_CACHE = dict()
_FLAX_IMG2IMG_MODEL_CACHE = dict()

_TORCH_INPAINT_MODEL_CACHE = dict()
_FLAX_INPAINT_MODEL_CACHE = dict()


class InvalidVaePath(Exception):
    pass


class InvalidSchedulerName(Exception):
    pass


def _is_single_file_model_load(path):
    return path.endswith('.ckpt') or path.endswith('.safetensors')


def _is_single_file_vae_load(path):
    return path.endswith('.pt') or path.endswith('.pth') or path.endswith('.safetensors')


def _load_pytorch_vae(path):
    parts = path.split(';', 1)

    if len(parts) != 2:
        raise InvalidVaePath(f'VAE path must contain auto encoder class name and path to the encoder URL or file.')

    encoder_name = parts[0].strip().lower()

    if encoder_name == "AutoencoderKL":
        encoder = AutoencoderKL
    elif encoder_name == "AsymmetricAutoencoderKL":
        encoder = AsymmetricAutoencoderKL
    elif encoder_name == "AutoencoderTiny":
        encoder = AutoencoderTiny
    else:
        raise InvalidVaePath(f'Unknown VAE encoder class {encoder_name}')

    path = parts[1].strip()

    if _is_single_file_vae_load(path):
        return encoder.from_single_file(path)
    else:
        return encoder.from_pretrained(path)


def _load_flax_vae(path):
    parts = path.split(';', 1)

    if len(parts) != 2:
        raise InvalidVaePath(f'VAE path must contain auto encoder class name and path to the encoder URL or file.')

    encoder_name = parts[0].strip()

    if encoder_name == "FlaxAutoencoderKL":
        encoder = FlaxAutoencoderKL
    else:
        raise InvalidVaePath(f'Unknown VAE flax encoder class {encoder_name}')

    path = parts[1].strip()

    if _is_single_file_vae_load(path):
        return encoder.from_single_file(path)
    else:
        return encoder.from_pretrained(path)


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
    _TORCH_IMG2IMG_MODEL_CACHE.clear()
    _FLAX_IMG2IMG_MODEL_CACHE.clear()
    _TORCH_INPAINT_MODEL_CACHE.clear()
    _FLAX_INPAINT_MODEL_CACHE.clear()


def _disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


def _create_torch_diffusion_pipeline(model_path, revision, variant, torch_dtype, vae=None, scheduler=None,
                                     safety_checker=False):
    cache_key = model_path + revision + '' if variant is None else variant + str(torch_dtype)
    catch_hit = _TORCH_MODEL_CACHE.get(cache_key)

    if catch_hit is None:
        kwargs = {}

        if vae is not None:
            kwargs['vae'] = _load_pytorch_vae(vae)

        if _is_single_file_model_load(model_path):
            pipeline = StableDiffusionPipeline.from_single_file(model_path,
                                                                revision=revision,
                                                                variant=variant,
                                                                torch_dtype=torch_dtype,
                                                                **kwargs)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(model_path,
                                                               revision=revision,
                                                               variant=variant,
                                                               torch_dtype=torch_dtype,
                                                               **kwargs)

        _load_scheduler(pipeline, scheduler)

        if not safety_checker:
            pipeline.safety_checker = _disabled_safety_checker

        _TORCH_MODEL_CACHE[cache_key] = pipeline
        return pipeline
    else:
        return catch_hit


def _create_flax_diffusion_pipeline(model_path, revision, flax_dtype, vae=None, scheduler=None,
                                    safety_checker=False):
    cache_key = model_path + revision + str(flax_dtype)
    catch_hit = _FLAX_MODEL_CACHE.get(cache_key)

    if catch_hit is None:
        kwargs = {}

        if vae is not None:
            kwargs['vae'] = _load_flax_vae(vae)

        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(model_path,
                                                                       revision=revision,
                                                                       dtype=flax_dtype,
                                                                       **kwargs)

        _load_scheduler(pipeline, scheduler)

        # if not safety_checker:
        #    pipeline.safety_checker = _disabled_safety_checker

        _FLAX_MODEL_CACHE[cache_key] = (pipeline, params)
        return pipeline, params
    else:
        return catch_hit


def _create_torch_img2img_diffusion_pipeline(model_path, revision, variant, torch_dtype, vae=None, scheduler=None,
                                             safety_checker=False):
    cache_key = model_path + revision + '' if variant is None else variant + str(torch_dtype)
    catch_hit = _TORCH_IMG2IMG_MODEL_CACHE.get(cache_key)

    if catch_hit is None:
        kwargs = {}

        if vae is not None:
            kwargs['vae'] = _load_pytorch_vae(vae)

        if _is_single_file_model_load(model_path):
            pipeline = StableDiffusionImg2ImgPipeline.from_single_file(model_path,
                                                                       revision=revision,
                                                                       variant=variant,
                                                                       torch_dtype=torch_dtype,
                                                                       **kwargs)
        else:
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_path,
                                                                      revision=revision,
                                                                      variant=variant,
                                                                      torch_dtype=torch_dtype,
                                                                      **kwargs)

        _load_scheduler(pipeline, scheduler)

        if not safety_checker:
            pipeline.safety_checker = _disabled_safety_checker

        _TORCH_IMG2IMG_MODEL_CACHE[cache_key] = pipeline
        return pipeline
    else:
        return catch_hit


def _create_flax_img2img_diffusion_pipeline(model_path, revision, flax_dtype, vae=None, scheduler=None,
                                            safety_checker=False):
    cache_key = model_path + revision + str(flax_dtype)
    catch_hit = _FLAX_IMG2IMG_MODEL_CACHE.get(cache_key)

    if catch_hit is None:
        kwargs = {}

        if vae is not None:
            kwargs['vae'] = _load_flax_vae(vae)

        pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(model_path,
                                                                              revision=revision,
                                                                              dtype=flax_dtype,
                                                                              **kwargs)
        _load_scheduler(pipeline, scheduler)

        _FLAX_IMG2IMG_MODEL_CACHE[cache_key] = (pipeline, params)
        return pipeline, params
    else:
        return catch_hit


def _create_torch_inpaint_diffusion_pipeline(model_path, revision, variant, torch_dtype, vae=None, scheduler=None,
                                             safety_checker=False):
    cache_key = model_path + revision + '' if variant is None else variant + str(torch_dtype)
    catch_hit = _TORCH_INPAINT_MODEL_CACHE.get(cache_key)

    if catch_hit is None:
        kwargs = {}

        if vae is not None:
            kwargs['vae'] = _load_pytorch_vae(vae)

        if _is_single_file_model_load(model_path):
            pipeline = StableDiffusionInpaintPipeline.from_single_file(model_path,
                                                                       revision=revision,
                                                                       variant=variant,
                                                                       torch_dtype=torch_dtype,
                                                                       **kwargs)
        else:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_path,
                                                                      revision=revision,
                                                                      variant=variant,
                                                                      torch_dtype=torch_dtype,
                                                                      **kwargs)

        _load_scheduler(pipeline, scheduler)

        if not safety_checker:
            pipeline.safety_checker = _disabled_safety_checker

        _TORCH_INPAINT_MODEL_CACHE[cache_key] = pipeline
        return pipeline
    else:
        return catch_hit


def _create_flax_inpaint_diffusion_pipeline(model_path, revision, flax_dtype, vae=None, scheduler=None,
                                            safety_checker=False):
    cache_key = model_path + revision + str(flax_dtype)
    catch_hit = _FLAX_INPAINT_MODEL_CACHE.get(cache_key)

    if catch_hit is None:
        kwargs = {}

        if vae is not None:
            kwargs['vae'] = _load_flax_vae(vae)

        pipeline, params = FlaxStableDiffusionInpaintPipeline.from_pretrained(model_path,
                                                                              revision=revision,
                                                                              dtype=flax_dtype,
                                                                              **kwargs)

        _load_scheduler(pipeline, scheduler)

        _FLAX_INPAINT_MODEL_CACHE[cache_key] = (pipeline, params)
        return pipeline, params
    else:
        return catch_hit


def supported_model_types():
    if have_jax_flax():
        return {'torch', 'flax'}
    else:
        return {'torch'}


def have_jax_flax():
    return _have_jax_flax


def _pipeline_defaults(kwargs):
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
    return args


def _get_flax_dtype(dtype):
    return {'float16': jnp.bfloat16,
            'float32': jnp.float32,
            'float64': jnp.float64,
            'auto': None,
            None: None}[dtype.lower()]


def _get_torch_dtype(dtype):
    return {'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
            'auto': None,
            None: None}[dtype.lower()]


def _image_grid(imgs, rows, cols):
    w, h = imgs[0].size

    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
        img.close()

    return grid


def _call_flax(wrapper, args, kwargs):
    device_count = jax.device_count()

    args['prng_seed'] = jax.random.split(jax.random.PRNGKey(kwargs.get('seed', 0)), device_count)

    processed_masks = None

    if 'image' in args:
        if 'mask_image' in args:

            prompt_ids, processed_images, processed_masks = \
                wrapper._pipeline.prepare_inputs(prompt=[kwargs.get('prompt', '')] * device_count,
                                                 image=[args['image']] * device_count,
                                                 mask=[args['mask_image']] * device_count)

            args['masked_image'] = shard(processed_images)
            args['mask'] = shard(processed_masks)

            args.pop('strength')
            args.pop('image')
            args.pop('mask_image')
        else:
            prompt_ids, processed_images = wrapper._pipeline.prepare_inputs(
                prompt=[kwargs.get('prompt', '')] * device_count,
                image=[args['image']] * device_count)
            args['image'] = shard(processed_images)

        args['width'] = processed_images[0].shape[2]
        args['height'] = processed_images[0].shape[1]
    else:
        prompt_ids = wrapper._pipeline.prepare_inputs([kwargs.get('prompt', '')] * device_count)

    images = wrapper._pipeline(prompt_ids=shard(prompt_ids), params=replicate(wrapper._flax_params),
                               **args, jit=True)[0]

    return PipelineResultWrapper(
        [_image_grid(wrapper._pipeline.numpy_to_pil(images.reshape((images.shape[0],) + images.shape[-3:])),
                     device_count, 1)])


def _call_torch(wrapper, args, kwargs):
    args['num_images_per_prompt'] = kwargs.get('num_images_per_prompt', 1)
    args['generator'] = torch.Generator(device=wrapper._device).manual_seed(kwargs.get('seed', 0))
    args['prompt'] = kwargs.get('prompt', '')
    args['negative_prompt'] = kwargs.get('negative_prompt', None)

    if 'mask_image' in args:
        if isinstance(wrapper._pipeline, StableDiffusionInpaintPipelineLegacy):
            # Not necessary, will cause an error
            args.pop('width')
            args.pop('height')

    return PipelineResultWrapper(wrapper._pipeline(**args).images)


class PipelineResultWrapper:
    def __init__(self, images):
        self.images = images


class DiffusionPipelineWrapper:
    def __init__(self, model_path, dtype, device='cuda', model_type='torch', revision='main', variant=None, vae=None,
                 scheduler=None,
                 safety_checker=False):
        self._device = device
        self._model_type = model_type
        self._model_path = model_path
        self._pipeline = None
        self._flax_params = None
        self._revision = revision
        self._variant = variant
        self._dtype = dtype
        self._device = device
        self._vae = vae
        self._safety_checker = safety_checker
        self._scheduler = scheduler

    def _lazy_init_pipeline(self):
        if self._pipeline is not None:
            return

        if self._model_type == 'flax':
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed')

            self._pipeline, self._flax_params = \
                _create_flax_diffusion_pipeline(self._model_path,
                                                revision=self._revision,
                                                flax_dtype=_get_flax_dtype(self._dtype),
                                                vae=self._vae, scheduler=self._scheduler,
                                                safety_checker=self._safety_checker)
        else:
            self._pipeline = \
                _create_torch_diffusion_pipeline(self._model_path,
                                                 revision=self._revision, variant=self._variant,
                                                 torch_dtype=_get_torch_dtype(self._dtype),
                                                 vae=self._vae, scheduler=self._scheduler,
                                                 safety_checker=self._safety_checker).to(self._device)

    def __call__(self, **kwargs):
        args = _pipeline_defaults(kwargs)

        self._lazy_init_pipeline()

        if self._model_type == 'flax':
            return _call_flax(self, args, kwargs)
        else:
            return _call_torch(self, args, kwargs)


class DiffusionPipelineImg2ImgWrapper:
    def __init__(self, model_path, dtype, device='cuda', model_type='torch', revision='main', variant=None, vae=None,
                 scheduler=None,
                 safety_checker=False):
        self._device = device
        self._model_type = model_type.strip().lower()
        self._model_path = model_path
        self._revision = revision
        self._variant = variant
        self._dtype = dtype
        self._pipeline = None
        self._flax_params = None
        self._vae = vae
        self._safety_checker = safety_checker
        self._scheduler = scheduler

    def _lazy_init_img2img(self):
        if self._pipeline is not None:
            return

        if self._model_type == 'flax':
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed')

            self._pipeline, self._flax_params = \
                _create_flax_img2img_diffusion_pipeline(self._model_path,
                                                        revision=self._revision,
                                                        flax_dtype=_get_flax_dtype(self._dtype),
                                                        vae=self._vae, scheduler=self._scheduler,
                                                        safety_checker=self._safety_checker)
        else:
            self._pipeline = \
                _create_torch_img2img_diffusion_pipeline(self._model_path,
                                                         revision=self._revision, variant=self._variant,
                                                         torch_dtype=_get_torch_dtype(self._dtype),
                                                         vae=self._vae, scheduler=self._scheduler,
                                                         safety_checker=self._safety_checker).to(
                    self._device)

    def _lazy_init_intpaint(self):
        if self._pipeline is not None:
            return

        if self._model_type == 'flax':
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed')

            self._pipeline, self._flax_params = \
                _create_flax_inpaint_diffusion_pipeline(self._model_path,
                                                        revision=self._revision,
                                                        flax_dtype=_get_flax_dtype(self._dtype),
                                                        vae=self._vae, scheduler=self._scheduler,
                                                        safety_checker=self._safety_checker)
        else:
            self._pipeline = \
                _create_torch_inpaint_diffusion_pipeline(self._model_path,
                                                         revision=self._revision, variant=self._variant,
                                                         torch_dtype=_get_torch_dtype(self._dtype),
                                                         vae=self._vae, scheduler=self._scheduler,
                                                         safety_checker=self._safety_checker).to(
                    self._device)

    def __call__(self, **kwargs):
        args = _pipeline_defaults(kwargs)

        if 'mask_image' in args:
            self._lazy_init_intpaint()
        else:
            self._lazy_init_img2img()

        if self._model_type == 'flax':
            return _call_flax(self, args, kwargs)
        else:
            return _call_torch(self, args, kwargs)
