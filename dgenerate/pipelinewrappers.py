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
        FlaxStableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy

    _have_jax_flax = True

    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
except ImportError:
    _have_jax_flax = False

import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline


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


def _call_flax(wrapper, args, kwargs):
    args['prng_seed'] = jax.random.split(jax.random.PRNGKey(kwargs.get('seed', 0)), 1)

    processed_masks = None

    if 'image' in args:
        if 'mask_image' in args:
            prompt_ids, processed_images, processed_masks = \
                wrapper._pipeline.prepare_inputs(prompt=kwargs.get('prompt', ''),
                                                 image=args['image'],
                                                 mask=args['mask_image'])

            args['masked_image'] = shard(processed_images)
            args['mask'] = shard(processed_masks)

            args.pop('strength')
            args.pop('image')
            args.pop('mask_image')
        else:
            prompt_ids, processed_images = wrapper._pipeline.prepare_inputs(prompt=kwargs.get('prompt', ''),
                                                                            image=args['image'])
            args['image'] = shard(processed_images)

        args['width'] = processed_images[0].shape[2]
        args['height'] = processed_images[0].shape[1]
    else:
        prompt_ids = wrapper._pipeline.prepare_inputs([kwargs.get('prompt', '')])

    images = wrapper._pipeline(prompt_ids=shard(prompt_ids), params=replicate(wrapper._flax_params),
                               **args, jit=True)[0]
    return PipelineResultWrapper(
        wrapper._pipeline.numpy_to_pil(images.reshape((images.shape[0],) + images.shape[-3:])))


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
    def __init__(self, model_path, dtype, device='cuda', model_type='torch', revision='main'):
        self._device = device
        self._model_type = model_type
        self._model_path = model_path
        self._pipeline = None
        self._flax_params = None
        self._revision = revision
        self._dtype = dtype
        self._device = device

    def _lazy_init_pipeline(self):
        if self._pipeline is not None:
            return

        if self._model_type == 'flax':
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed')

            self._pipeline, self._flax_params = FlaxStableDiffusionPipeline.from_pretrained(self._model_path,
                                                                                            revision=self._revision,
                                                                                            dtype=_get_flax_dtype(
                                                                                                self._dtype))
        else:
            self._pipeline = DiffusionPipeline.from_pretrained(self._model_path,
                                                               torch_dtype=_get_torch_dtype(self._dtype),
                                                               revision=self._revision).to(self._device)

    def __call__(self, **kwargs):
        args = _pipeline_defaults(kwargs)

        self._lazy_init_pipeline()

        if self._model_type == 'flax':
            return _call_flax(self, args, kwargs)
        else:
            return _call_torch(self, args, kwargs)


class DiffusionPipelineImg2ImgWrapper:
    def __init__(self, model_path, dtype, device='cuda', model_type='torch', revision='main'):
        self._device = device
        self._model_type = model_type.strip().lower()
        self._model_path = model_path
        self._revision = revision
        self._dtype = dtype
        self._pipeline = None
        self._flax_params = None

    def _lazy_init_img2img(self):
        if self._pipeline is not None:
            return

        if self._model_type == 'flax':
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed')

            self._pipeline, self._flax_params = \
                FlaxStableDiffusionImg2ImgPipeline.from_pretrained(self._model_path,
                                                                   revision=self._revision,
                                                                   dtype=_get_flax_dtype(self._dtype))
        else:
            self._pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(self._model_path,
                                                                            torch_dtype=_get_torch_dtype(self._dtype),
                                                                            revision=self._revision).to(self._device)

    def _lazy_init_intpaint(self):
        if self._pipeline is not None:
            return

        if self._model_type == 'flax':
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed')

            self._pipeline, self._flax_params = \
                FlaxStableDiffusionInpaintPipeline.from_pretrained(self._model_path,
                                                                   revision=self._revision,
                                                                   dtype=_get_flax_dtype(self._dtype))
        else:
            self._pipeline = StableDiffusionInpaintPipeline.from_pretrained(self._model_path,
                                                                            torch_dtype=_get_torch_dtype(self._dtype),
                                                                            revision=self._revision).to(self._device)

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
