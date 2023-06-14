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
    from diffusers import FlaxStableDiffusionPipeline, FlaxStableDiffusionImg2ImgPipeline
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard

    _have_jax_flax = True
except ImportError:
    _have_jax_flax = False

import os
import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def supported_models():
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
        args['image'] = kwargs['image']
        args['strength'] = float(kwargs.get('strength', 0.8))
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

    if 'image' in args:
        prompt_ids, processed_images = wrapper._pipeline.prepare_inputs(prompt=kwargs.get('prompt', ''), image=args['image'])
        args['width'] = processed_images[0].shape[2]
        args['height'] = processed_images[0].shape[1]
        args['image'] = shard(processed_images)
    else:
        prompt_ids = wrapper._pipeline.prepare_inputs([kwargs.get('prompt', '')])

    images = wrapper._pipeline(prompt_ids=shard(prompt_ids), params=replicate(wrapper._params), **args, jit=True)[0]
    return PipelineResultWrapper(
        wrapper._pipeline.numpy_to_pil(images.reshape((images.shape[0],) + images.shape[-3:])))


def _call_torch(self, args, kwargs):
    args['num_images_per_prompt'] = kwargs.get('num_images_per_prompt', 1)
    args['generator'] = torch.Generator(device=self._device).manual_seed(kwargs.get('seed', 0))
    args['prompt'] = kwargs.get('prompt', '')
    args['negative_prompt'] = kwargs.get('negative_prompt', None)
    return PipelineResultWrapper(self._pipeline(**args).images)


class PipelineResultWrapper:
    def __init__(self, images):
        self.images = images


class DiffusionPipelineWrapper:
    def __init__(self, model_path, dtype, device='cuda', model_type='torch', revision='main'):
        self._device = device
        self._model_type = model_type
        if self._model_type == 'flax':
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed')

            self._pipeline, self._params = FlaxStableDiffusionPipeline.from_pretrained(model_path,
                                                                                       revision=revision,
                                                                                       dtype=_get_flax_dtype(dtype))
        else:
            self._pipeline = DiffusionPipeline.from_pretrained(model_path,
                                                               torch_dtype=_get_torch_dtype(dtype),
                                                               revision=revision).to(self._device)

    def __call__(self, **kwargs):
        args = _pipeline_defaults(kwargs)
        if self._model_type == 'flax':
            return _call_flax(self, args, kwargs)
        else:
            return _call_torch(self, args, kwargs)


class DiffusionPipelineImg2ImgWrapper:
    def __init__(self, model_path, dtype, device='cuda', model_type='torch', revision='main'):
        self._device = device
        self._model_type = model_type.strip().lower()
        if self._model_type == 'flax':
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed')

            self._pipeline, self._params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(model_path,
                                                                                              revision=revision,
                                                                                              dtype=_get_flax_dtype(
                                                                                                  dtype))
        else:
            self._pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_path,
                                                                            torch_dtype=_get_torch_dtype(dtype),
                                                                            revision=revision).to(self._device)

    def __call__(self, **kwargs):
        args = _pipeline_defaults(kwargs)
        if self._model_type == 'flax':
            return _call_flax(self, args, kwargs)
        else:
            return _call_torch(self, args, kwargs)
