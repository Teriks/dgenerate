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

import datetime
import itertools
import math
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Iterator

import torch
from PIL.PngImagePlugin import PngInfo

from . import messages, preprocessors
from .mediainput import iterate_image_seed, get_image_seed_info, MultiContextManager, \
    get_control_image_info, iterate_control_image, ImageSeed
from .mediaoutput import create_animation_writer, supported_animation_writer_formats
from .pipelinewrappers import DiffusionPipelineWrapper, DiffusionPipelineImg2ImgWrapper, supported_model_type_strings, \
    PipelineResultWrapper, model_type_is_upscaler, get_model_type_enum, ModelTypes, model_type_is_pix2pix, \
    model_type_is_sdxl, supported_model_type_enums
from .textprocessing import oxford_comma, long_text_wrap_width


class InvalidDeviceOrdinalException(Exception):
    pass


def is_valid_device_string(device, raise_ordinal=True):
    match = re.match(r'^(?:cpu|cuda(?::([0-9]+))?)$', device)
    if match:
        if match.lastindex:
            ordinal = int(match[1])
            valid_ordinal = ordinal < torch.cuda.device_count()
            if raise_ordinal and not valid_ordinal:
                raise InvalidDeviceOrdinalException(f'CUDA device ordinal {ordinal} is invalid, no such device exists.')
            return valid_ordinal
        return True
    return False


def _has_len(obj):
    try:
        len(obj)
        return True
    except TypeError:
        return False


class DiffusionArgContext:
    def __init__(self,
                 diffusion_args,
                 prompt,
                 sdxl_second_prompt,
                 sdxl_refiner_prompt,
                 sdxl_refiner_second_prompt,
                 seed,
                 image_seed_strength,
                 upscaler_noise_level,
                 sdxl_high_noise_fraction,
                 sdxl_aesthetic_score,
                 sdxl_original_size,
                 sdxl_target_size,
                 sdxl_crops_coords_top_left,
                 sdxl_negative_aesthetic_score,
                 sdxl_negative_original_size,
                 sdxl_negative_target_size,
                 sdxl_negative_crops_coords_top_left,
                 sdxl_refiner_aesthetic_score,
                 sdxl_refiner_original_size,
                 sdxl_refiner_target_size,
                 sdxl_refiner_crops_coords_top_left,
                 sdxl_refiner_negative_aesthetic_score,
                 sdxl_refiner_negative_original_size,
                 sdxl_refiner_negative_target_size,
                 sdxl_refiner_negative_crops_coords_top_left,
                 guidance_scale,
                 image_guidance_scale,
                 guidance_rescale,
                 inference_steps):
        self.args = diffusion_args
        self.prompt = prompt
        self.sdxl_second_prompt = sdxl_second_prompt
        self.sdxl_refiner_prompt = sdxl_refiner_prompt
        self.sdxl_refiner_second_prompt = sdxl_refiner_second_prompt
        self.seed = seed
        self.image_seed_strength = image_seed_strength
        self.upscaler_noise_level = upscaler_noise_level
        self.sdxl_high_noise_fraction = sdxl_high_noise_fraction
        self.sdxl_aesthetic_score = sdxl_aesthetic_score
        self.sdxl_original_size = sdxl_original_size
        self.sdxl_target_size = sdxl_target_size
        self.sdxl_crops_coords_top_left = sdxl_crops_coords_top_left
        self.sdxl_negative_aesthetic_score = sdxl_negative_aesthetic_score
        self.sdxl_negative_original_size = sdxl_negative_original_size
        self.sdxl_negative_target_size = sdxl_negative_target_size
        self.sdxl_negative_crops_coords_top_left = sdxl_negative_crops_coords_top_left
        self.sdxl_refiner_aesthetic_score = sdxl_refiner_aesthetic_score
        self.sdxl_refiner_original_size = sdxl_refiner_original_size
        self.sdxl_refiner_target_size = sdxl_refiner_target_size
        self.sdxl_refiner_crops_coords_top_left = sdxl_refiner_crops_coords_top_left
        self.sdxl_refiner_negative_aesthetic_score = sdxl_refiner_negative_aesthetic_score
        self.sdxl_refiner_negative_original_size = sdxl_refiner_negative_original_size
        self.sdxl_refiner_negative_target_size = sdxl_refiner_negative_target_size
        self.sdxl_refiner_negative_crops_coords_top_left = sdxl_refiner_negative_crops_coords_top_left
        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.guidance_rescale = guidance_rescale
        self.inference_steps = inference_steps

    @staticmethod
    def _describe_prompt(prompt_format, prompt_dict, pos_title, neg_title):
        if prompt_dict is None:
            return

        prompt_wrap_width = long_text_wrap_width()
        prompt_val = prompt_dict.get('prompt', None)
        if prompt_val:
            header = f'{pos_title}: '
            prompt_val = textwrap.fill(prompt_val, width=prompt_wrap_width - len(header),
                                       subsequent_indent=' ' * len(header))
            prompt_format.append(f'{header}"{prompt_val}"')

        prompt_val = prompt_dict.get('negative_prompt', None)
        if prompt_val:
            header = f'{neg_title}: '
            prompt_val = textwrap.fill(prompt_val,
                                       width=prompt_wrap_width - len(header),
                                       subsequent_indent=' ' * len(header))
            prompt_format.append(f'{header}"{prompt_val}"')

    def describe_parameters(self):
        prompt_format = []

        DiffusionArgContext._describe_prompt(
            prompt_format, self.prompt,
            "Prompt",
            "Negative Prompt")

        DiffusionArgContext._describe_prompt(
            prompt_format, self.sdxl_second_prompt,
            "SDXL Second Prompt",
            "SDXL Second Negative Prompt")

        DiffusionArgContext._describe_prompt(
            prompt_format, self.sdxl_refiner_prompt,
            "SDXL Refiner Prompt",
            "SDXL Refiner Negative Prompt")

        DiffusionArgContext._describe_prompt(
            prompt_format, self.sdxl_refiner_second_prompt,
            "SDXL Refiner Second Prompt",
            "SDXL Refiner Second Negative Prompt")

        prompt_format = '\n'.join(prompt_format)
        if prompt_format:
            prompt_format = '\n' + prompt_format

        inputs = [f'Seed: {self.seed}']

        descripts = [
            (self.image_seed_strength, "Image Seed Strength:"),
            (self.upscaler_noise_level, "Upscaler Noise Level:"),
            (self.sdxl_high_noise_fraction, "SDXL High Noise Fraction:"),
            (self.sdxl_aesthetic_score, "SDXL Aesthetic Score:"),
            (self.sdxl_original_size, "SDXL Original Size:"),
            (self.sdxl_target_size, "SDXL Target Size:"),
            (self.sdxl_crops_coords_top_left, "SDXL Top Left Crop Coords:"),
            (self.sdxl_negative_aesthetic_score, "SDXL Negative Aesthetic Score:"),
            (self.sdxl_negative_original_size, "SDXL Negative Original Size:"),
            (self.sdxl_negative_target_size, "SDXL Negative Target Size:"),
            (self.sdxl_negative_crops_coords_top_left, "SDXL Negative Top Left Crop Coords:"),
            (self.sdxl_refiner_aesthetic_score, "SDXL Refiner Aesthetic Score:"),
            (self.sdxl_refiner_original_size, "SDXL Refiner Original Size:"),
            (self.sdxl_refiner_target_size, "SDXL Refiner Target Size:"),
            (self.sdxl_refiner_crops_coords_top_left, "SDXL Refiner Top Left Crop Coords:"),
            (self.sdxl_refiner_negative_aesthetic_score, "SDXL Refiner Negative Aesthetic Score:"),
            (self.sdxl_refiner_negative_original_size, "SDXL Refiner Negative Original Size:"),
            (self.sdxl_refiner_negative_target_size, "SDXL Refiner Negative Target Size:"),
            (self.sdxl_refiner_negative_crops_coords_top_left, "SDXL Refiner Negative Top Left Crop Coords:"),
            (self.guidance_scale, "Guidance Scale:"),
            (self.image_guidance_scale, "Image Guidance Scale:"),
            (self.guidance_rescale, "Guidance Rescale:"),
            (self.inference_steps, "Inference Steps:")
        ]

        for prompt_val, desc in descripts:
            if prompt_val is not None:
                inputs.append(desc + ' ' + str(prompt_val))

        inputs = '\n'.join(inputs)

        return inputs + prompt_format


def _list_or_list_of_none(val):
    return val if val else [None]


def iterate_diffusion_args(prompts,
                           sdxl_second_prompts,
                           sdxl_refiner_prompts,
                           sdxl_refiner_second_prompts,
                           seeds,
                           image_seed_strengths,
                           upscaler_noise_levels,
                           sdxl_high_noise_fractions,
                           sdxl_aesthetic_scores,
                           sdxl_original_sizes,
                           sdxl_target_sizes,
                           sdxl_crops_coords_top_left,
                           sdxl_negative_aesthetic_scores,
                           sdxl_negative_original_sizes,
                           sdxl_negative_target_sizes,
                           sdxl_negative_crops_coords_top_left,
                           sdxl_refiner_aesthetic_scores,
                           sdxl_refiner_original_sizes,
                           sdxl_refiner_target_sizes,
                           sdxl_refiner_crops_coords_top_left,
                           sdxl_refiner_negative_aesthetic_scores,
                           sdxl_refiner_negative_original_sizes,
                           sdxl_refiner_negative_target_sizes,
                           sdxl_refiner_negative_crops_coords_top_left,
                           guidance_scales,
                           image_guidance_scales,
                           guidance_rescales,
                           inference_steps_list) -> Iterator[DiffusionArgContext]:
    diffusion_args = dict()

    for arg_set in itertools.product(
            prompts,
            _list_or_list_of_none(sdxl_second_prompts),
            _list_or_list_of_none(sdxl_refiner_prompts),
            _list_or_list_of_none(sdxl_refiner_second_prompts),
            seeds,
            _list_or_list_of_none(image_seed_strengths),
            _list_or_list_of_none(upscaler_noise_levels),
            _list_or_list_of_none(sdxl_high_noise_fractions),
            _list_or_list_of_none(sdxl_aesthetic_scores),
            _list_or_list_of_none(sdxl_original_sizes),
            _list_or_list_of_none(sdxl_target_sizes),
            _list_or_list_of_none(sdxl_crops_coords_top_left),
            _list_or_list_of_none(sdxl_negative_aesthetic_scores),
            _list_or_list_of_none(sdxl_negative_original_sizes),
            _list_or_list_of_none(sdxl_negative_target_sizes),
            _list_or_list_of_none(sdxl_negative_crops_coords_top_left),
            _list_or_list_of_none(sdxl_refiner_aesthetic_scores),
            _list_or_list_of_none(sdxl_refiner_original_sizes),
            _list_or_list_of_none(sdxl_refiner_target_sizes),
            _list_or_list_of_none(sdxl_refiner_crops_coords_top_left),
            _list_or_list_of_none(sdxl_refiner_negative_aesthetic_scores),
            _list_or_list_of_none(sdxl_refiner_negative_original_sizes),
            _list_or_list_of_none(sdxl_refiner_negative_target_sizes),
            _list_or_list_of_none(sdxl_refiner_negative_crops_coords_top_left),
            guidance_scales,
            _list_or_list_of_none(image_guidance_scales),
            _list_or_list_of_none(guidance_rescales),
            inference_steps_list
    ):
        idx = iter(range(0, len(arg_set)))

        # It is very important that the order here matches what
        # is above :)

        prompt = arg_set[next(idx)]
        sdxl_second_prompt = arg_set[next(idx)]
        sdxl_refiner_prompt = arg_set[next(idx)]
        sdxl_refiner_second_prompt = arg_set[next(idx)]
        seed = arg_set[next(idx)]
        image_seed_strength = arg_set[next(idx)]
        upscaler_noise_level = arg_set[next(idx)]
        sdxl_high_noise_fraction = arg_set[next(idx)]
        sdxl_aesthetic_score = arg_set[next(idx)]
        sdxl_original_size = arg_set[next(idx)]
        sdxl_target_size = arg_set[next(idx)]
        sdxl_crops_coords_top_left_v = arg_set[next(idx)]
        sdxl_negative_aesthetic_score = arg_set[next(idx)]
        sdxl_negative_original_size = arg_set[next(idx)]
        sdxl_negative_target_size = arg_set[next(idx)]
        sdxl_negative_crops_coords_top_left_v = arg_set[next(idx)]
        sdxl_refiner_aesthetic_score = arg_set[next(idx)]
        sdxl_refiner_original_size = arg_set[next(idx)]
        sdxl_refiner_target_size = arg_set[next(idx)]
        sdxl_refiner_crops_coords_top_left_v = arg_set[next(idx)]
        sdxl_refiner_negative_aesthetic_score = arg_set[next(idx)]
        sdxl_refiner_negative_original_size = arg_set[next(idx)]
        sdxl_refiner_negative_target_size = arg_set[next(idx)]
        sdxl_refiner_negative_crops_coords_top_left_v = arg_set[next(idx)]
        guidance_scale = arg_set[next(idx)]
        image_guidance_scale = arg_set[next(idx)]
        guidance_rescale = arg_set[next(idx)]
        inference_steps = arg_set[next(idx)]

        # Mode dependant

        if image_seed_strengths:
            diffusion_args['strength'] = image_seed_strength

        if upscaler_noise_levels:
            diffusion_args['upscaler_noise_level'] = upscaler_noise_level

        if guidance_rescales:
            diffusion_args['guidance_rescale'] = guidance_rescale

        if image_guidance_scales:
            diffusion_args['image_guidance_scale'] = image_guidance_scale

        # SDXL

        if sdxl_high_noise_fractions:
            diffusion_args['sdxl_high_noise_fraction'] = sdxl_high_noise_fraction

        # SDXL Main Model

        if sdxl_aesthetic_scores:
            diffusion_args['sdxl_aesthetic_score'] = sdxl_aesthetic_score

        if sdxl_original_sizes:
            diffusion_args['sdxl_original_size'] = sdxl_original_size

        if sdxl_target_sizes:
            diffusion_args['sdxl_target_size'] = sdxl_target_size

        if sdxl_crops_coords_top_left:
            diffusion_args['sdxl_crops_coords_top_left'] = sdxl_crops_coords_top_left_v

        # SDXL Main Model negatives

        if sdxl_negative_aesthetic_scores:
            diffusion_args['sdxl_negative_aesthetic_score'] = sdxl_negative_aesthetic_score

        if sdxl_negative_original_sizes:
            diffusion_args['sdxl_negative_original_size'] = sdxl_negative_original_size

        if sdxl_negative_target_sizes:
            diffusion_args['sdxl_negative_target_size'] = sdxl_negative_target_size

        if sdxl_negative_crops_coords_top_left:
            diffusion_args['sdxl_negative_crops_coords_top_left'] = sdxl_negative_crops_coords_top_left_v

        # SDXL Refiner Model

        if sdxl_refiner_aesthetic_scores:
            diffusion_args['sdxl_refiner_aesthetic_score'] = sdxl_refiner_aesthetic_score

        if sdxl_refiner_original_sizes:
            diffusion_args['sdxl_refiner_original_size'] = sdxl_refiner_original_size

        if sdxl_refiner_target_sizes:
            diffusion_args['sdxl_refiner_target_size'] = sdxl_refiner_target_size

        if sdxl_refiner_crops_coords_top_left:
            diffusion_args['sdxl_refiner_crops_coords_top_left'] = sdxl_refiner_crops_coords_top_left_v

        # SDXL Refiner Model negatives

        if sdxl_refiner_negative_aesthetic_scores:
            diffusion_args['sdxl_refiner_negative_aesthetic_score'] = sdxl_refiner_negative_aesthetic_score

        if sdxl_refiner_negative_original_sizes:
            diffusion_args['sdxl_refiner_negative_original_size'] = sdxl_refiner_negative_original_size

        if sdxl_refiner_negative_target_sizes:
            diffusion_args['sdxl_refiner_negative_target_size'] = sdxl_refiner_negative_target_size

        if sdxl_refiner_negative_crops_coords_top_left:
            diffusion_args[
                'sdxl_refiner_negative_crops_coords_top_left'] = sdxl_refiner_negative_crops_coords_top_left_v

        # Universals

        diffusion_args['num_inference_steps'] = (
            math.ceil(inference_steps / image_seed_strength if image_seed_strength > 0 else inference_steps)
            if image_seed_strengths else inference_steps)

        diffusion_args['guidance_scale'] = (
            (guidance_scale / image_seed_strength if image_seed_strength > 0 else guidance_scale)
            if image_seed_strengths else guidance_scale)

        diffusion_args.update(prompt)

        if sdxl_second_prompts:
            diffusion_args['sdxl_prompt_2'] = sdxl_second_prompt.get('prompt', None)
            diffusion_args['sdxl_negative_prompt_2'] = sdxl_second_prompt.get('negative_prompt', None)

        if sdxl_refiner_prompts:
            diffusion_args['sdxl_refiner_prompt'] = sdxl_refiner_prompt.get('prompt', None)
            diffusion_args['sdxl_refiner_negative_prompt'] = sdxl_refiner_prompt.get('negative_prompt', None)

        if sdxl_refiner_second_prompts:
            diffusion_args['sdxl_refiner_prompt_2'] = sdxl_refiner_second_prompt.get('prompt', None)
            diffusion_args['sdxl_refiner_negative_prompt_2'] = sdxl_refiner_second_prompt.get('negative_prompt', None)

        yield DiffusionArgContext(diffusion_args,
                                  prompt,
                                  sdxl_second_prompt,
                                  sdxl_refiner_prompt,
                                  sdxl_refiner_second_prompt,
                                  seed,
                                  image_seed_strength,
                                  upscaler_noise_level,
                                  sdxl_high_noise_fraction,
                                  sdxl_aesthetic_score,
                                  sdxl_original_size,
                                  sdxl_target_size,
                                  sdxl_crops_coords_top_left_v,
                                  sdxl_negative_aesthetic_score,
                                  sdxl_negative_original_size,
                                  sdxl_negative_target_size,
                                  sdxl_negative_crops_coords_top_left_v,
                                  sdxl_refiner_aesthetic_score,
                                  sdxl_refiner_original_size,
                                  sdxl_refiner_target_size,
                                  sdxl_refiner_crops_coords_top_left_v,
                                  sdxl_refiner_negative_aesthetic_score,
                                  sdxl_refiner_negative_original_size,
                                  sdxl_refiner_negative_target_size,
                                  sdxl_refiner_negative_crops_coords_top_left_v,
                                  guidance_scale,
                                  image_guidance_scale,
                                  guidance_rescale,
                                  inference_steps)


def _safe_len(lst):
    if lst is None:
        return 0
    return len(lst)


class DiffusionRenderLoop:
    def __init__(self):
        self._generation_step = -1
        self._frame_time_sum = 0
        self._last_frame_time = 0
        self._written_images = []
        self._written_animations = []

        self.model_path = None
        self.model_subfolder = None
        self.sdxl_refiner_path = None

        self.prompts = ['']
        self.sdxl_second_prompts = None
        self.sdxl_refiner_prompts = None
        self.sdxl_refiner_second_prompts = None

        self.seeds = [0]
        self.guidance_scales = [5]
        self.inference_steps = [30]

        self.image_seeds = None
        self.control_images = None
        self.image_seed_strengths = None
        self.upscaler_noise_levels = None
        self.guidance_rescales = None
        self.image_guidance_scales = None

        self.sdxl_high_noise_fractions = None
        self.sdxl_aesthetic_scores = None
        self.sdxl_original_sizes = None
        self.sdxl_target_sizes = None
        self.sdxl_crops_coords_top_left = None
        self.sdxl_negative_aesthetic_scores = None
        self.sdxl_negative_original_sizes = None
        self.sdxl_negative_target_sizes = None
        self.sdxl_negative_crops_coords_top_left = None

        self.sdxl_refiner_aesthetic_scores = None
        self.sdxl_refiner_original_sizes = None
        self.sdxl_refiner_target_sizes = None
        self.sdxl_refiner_crops_coords_top_left = None
        self.sdxl_refiner_negative_aesthetic_scores = None
        self.sdxl_refiner_negative_original_sizes = None
        self.sdxl_refiner_negative_target_sizes = None
        self.sdxl_refiner_negative_crops_coords_top_left = None

        self.vae_path = None
        self.vae_tiling = False
        self.vae_slicing = False

        self.lora_paths = None
        self.textual_inversion_paths = None
        self.control_net_paths = None

        self.scheduler = None
        self.safety_checker = False
        self.model_type = ModelTypes.TORCH
        self.device = 'cuda'
        self.dtype = 'auto'
        self.revision = 'main'
        self.variant = None
        self.output_size = None
        self.output_path = os.path.join(os.getcwd(), 'output')
        self.output_prefix = None
        self.output_overwrite = False
        self.output_configs = False
        self.output_metadata = False

        self.animation_format = 'mp4'
        self.frame_start = 0
        self.frame_end = None

        self.auth_token = None

        self.seed_image_preprocessors = None
        self.mask_image_preprocessors = None
        self.control_image_preprocessors = None

    @property
    def written_images(self):
        return self._written_images

    @property
    def written_animations(self):
        return self._written_animations

    def _enforce_state(self):
        if not _has_len(self.prompts):
            raise ValueError('DiffusionRenderLoop.prompts must have len')
        if not _has_len(self.inference_steps):
            raise ValueError('DiffusionRenderLoop.inference_steps must have len')
        if not _has_len(self.seeds):
            raise ValueError('DiffusionRenderLoop.seeds must have len')
        if not _has_len(self.guidance_scales):
            raise ValueError('DiffusionRenderLoop.guidance_scales must have len')

        if self.dtype not in {'float32', 'float16', 'auto'}:
            raise ValueError('DiffusionRenderLoop.torch_dtype must be float32, float16 or auto')
        if not isinstance(self.safety_checker, bool):
            raise ValueError('DiffusionRenderLoop.safety_checker must be True or False (bool)')
        if self.revision is not None and not isinstance(self.revision, str):
            raise ValueError('DiffusionRenderLoop.revision must be None or a string')
        if self.variant is not None and not isinstance(self.variant, str):
            raise ValueError('DiffusionRenderLoop.variant must be None or a string')
        if self.model_type not in supported_model_type_enums():
            raise ValueError(
                f'DiffusionRenderLoop.model_type must be one of: {oxford_comma(supported_model_type_strings(), "or")}')
        if self.model_path is None:
            raise ValueError('DiffusionRenderLoop.model_path must not be None')
        if self.model_subfolder is not None and not isinstance(self.model_subfolder, str):
            raise ValueError('DiffusionRenderLoop.model_subfolder must be None or str')
        if self.auth_token is not None and not isinstance(self.auth_token, str):
            raise ValueError('DiffusionRenderLoop.auth_token must be None or str')
        if self.lora_paths is not None and not isinstance(self.lora_paths, str):
            raise ValueError('DiffusionRenderLoop.lora_paths must be None or str')
        if self.sdxl_refiner_path is not None and not isinstance(self.sdxl_refiner_path, str):
            raise ValueError('DiffusionRenderLoop.sdxl_refiner_path must be None or a string')
        if self.vae_path is not None and not isinstance(self.vae_path, str):
            raise ValueError('DiffusionRenderLoop.vae_path must be a string: AutoencoderClass;model=PATH')
        if not isinstance(self.vae_tiling, bool):
            raise ValueError('DiffusionRenderLoop.vae_tiling must be True or False (bool)')
        if not isinstance(self.vae_slicing, bool):
            raise ValueError('DiffusionRenderLoop.vae_slicing must be True or False (bool)')
        if self.scheduler is not None and not isinstance(self.scheduler, str):
            raise ValueError('DiffusionRenderLoop.scheduler must be a string that names a compatible scheduler class')
        if self.output_path is None:
            raise ValueError('DiffusionRenderLoop.output_path must not be None')
        if self.output_prefix is not None and not isinstance(self.output_prefix, str):
            raise ValueError('DiffusionRenderLoop.output_prefix must be None or a str')
        if not isinstance(self.output_overwrite, bool):
            raise ValueError('DiffusionRenderLoop.output_overwrite must be bool')
        if not isinstance(self.output_configs, bool):
            raise ValueError('DiffusionRenderLoop.output_configs must be bool')
        if not isinstance(self.output_metadata, bool):
            raise ValueError('DiffusionRenderLoop.output_metadata must be bool')
        if not isinstance(self.device, str) or not is_valid_device_string(self.device):
            raise ValueError('DiffusionRenderLoop.device must be "cuda" or "cpu"')
        if not (isinstance(self.animation_format, str) or
                self.animation_format.lower() not in supported_animation_writer_formats()):
            raise ValueError(f'DiffusionRenderLoop.animation_format must be one of: '
                             f'{oxford_comma(supported_animation_writer_formats(), "or")}')
        if self.output_size is None and not self.image_seeds and not self.control_images:
            raise ValueError(
                'DiffusionRenderLoop.output_size must not be None when no image seeds or control images specified')
        if self.output_size is not None and not isinstance(self.output_size, tuple):
            raise ValueError('DiffusionRenderLoop.output_size must be None or a tuple')
        if not isinstance(self.frame_start, int) or not self.frame_start >= 0:
            raise ValueError('DiffusionRenderLoop.frame_start must be an integer value greater than or equal to zero')
        if self.frame_end is not None and (not isinstance(self.frame_end, int) or not self.frame_end >= 0):
            raise ValueError('DiffusionRenderLoop.frame_end must be an integer value greater than or equal to zero')
        if self.frame_end is not None and self.frame_start > self.frame_end:
            raise ValueError(
                'DiffusionRenderLoop.frame_start must be an integer value less than DiffusionRenderLoop.frame_end')

    @property
    def num_generation_steps(self):
        optional_factors = [
            self.sdxl_second_prompts,
            self.sdxl_refiner_prompts,
            self.sdxl_refiner_second_prompts,
            self.image_guidance_scales,
            self.textual_inversion_paths,
            self.control_net_paths,
            self.image_seeds,
            self.control_images,
            self.image_seed_strengths,
            self.upscaler_noise_levels,
            self.guidance_rescales,
            self.sdxl_high_noise_fractions,
            self.sdxl_aesthetic_scores,
            self.sdxl_original_sizes,
            self.sdxl_target_sizes,
            self.sdxl_crops_coords_top_left,
            self.sdxl_negative_aesthetic_scores,
            self.sdxl_negative_original_sizes,
            self.sdxl_negative_target_sizes,
            self.sdxl_negative_crops_coords_top_left,
            self.sdxl_refiner_aesthetic_scores,
            self.sdxl_refiner_original_sizes,
            self.sdxl_refiner_target_sizes,
            self.sdxl_refiner_crops_coords_top_left,
            self.sdxl_refiner_negative_aesthetic_scores,
            self.sdxl_refiner_negative_original_sizes,
            self.sdxl_refiner_negative_target_sizes,
            self.sdxl_refiner_negative_crops_coords_top_left,
        ]

        product = 1
        for i in optional_factors:
            product *= max(_safe_len(i), 1)

        return (product *
                len(self.prompts) *
                len(self.seeds) *
                len(self.guidance_scales) *
                len(self.inference_steps))

    @property
    def generation_step(self):
        return self._generation_step

    def _gen_filename(self, *args, ext):
        def _make_path(args, ext, dup_number=None):
            return os.path.join(self.output_path,
                                f'{self.output_prefix + "_" if self.output_prefix is not None else ""}' + '_'.
                                join(str(s).replace('.', '-') for s in args) + (
                                    '' if dup_number is None else f'_duplicate_{dup_number}') + '.' + ext)

        path = _make_path(args, ext)

        if self.output_overwrite:
            return path

        if not os.path.exists(path):
            return path

        duplicate_number = 1
        while os.path.exists(path):
            path = _make_path(args, ext, duplicate_number)
            duplicate_number += 1

        return path

    def _gen_filename_base(self, args_ctx):
        if args_ctx.upscaler_noise_level is not None:
            noise_entry = ('unl', args_ctx.upscaler_noise_level)
        elif args_ctx.image_seed_strength is not None:
            noise_entry = ('st', args_ctx.image_seed_strength)
        else:
            noise_entry = []

        guidance_entry = ['g', args_ctx.guidance_scale]

        if args_ctx.guidance_rescale is not None:
            guidance_entry += ['grs', args_ctx.guidance_rescale]

        if args_ctx.image_guidance_scale is not None:
            guidance_entry += ['igs', args_ctx.image_guidance_scale]

        args = ['s', args_ctx.seed,
                *noise_entry,
                *guidance_entry,
                'i', args_ctx.inference_steps]
        return args

    def _gen_animation_filename(self, args_ctx: DiffusionArgContext, generation_step, animation_format):
        args = ['ANIM', *self._gen_filename_base(args_ctx)]

        if args_ctx.sdxl_high_noise_fraction is not None:
            args += ['hnf', args_ctx.sdxl_high_noise_fraction]

        return self._gen_filename(*args, 'step', generation_step + 1, ext=animation_format)

    def _write_generation_result(self, filename, generation_result: PipelineResultWrapper, config_txt):
        if self.output_metadata:
            metadata = PngInfo()
            metadata.add_text("DgenerateConfig", config_txt)
            generation_result.image.save(filename, pnginfo=metadata)
        else:
            generation_result.image.save(filename)
        if self.output_configs:
            config_file_name = os.path.splitext(filename)[0] + '.txt'
            with open(config_file_name, "w") as config_file:
                config_file.write(config_txt)
            messages.log(
                f'Wrote Image File: {filename}\nWrote Config File: {config_file_name}', underline=True)
        else:
            messages.log(f'Wrote Image File: {filename}', underline=True)

    def _write_animation_frame(self, args_ctx: DiffusionArgContext, image_seed_obj,
                               generation_result: PipelineResultWrapper):
        args = self._gen_filename_base(args_ctx)

        if args_ctx.sdxl_high_noise_fraction is not None:
            args += ['hnf', args_ctx.sdxl_high_noise_fraction]

        filename = self._gen_filename(*args,
                                      'frame',
                                      image_seed_obj.frame_index + 1,
                                      'step',
                                      self._generation_step + 1,
                                      ext='png')
        config_txt = \
            generation_result.dgenerate_config + \
            f' \\\n--frame-start {image_seed_obj.frame_index} --frame-end {image_seed_obj.frame_index}'

        self._written_images.append(os.path.abspath(filename))
        self._write_generation_result(filename, generation_result, config_txt)

    def _write_image_seed_gen_image(self, args_ctx: DiffusionArgContext,
                                    generation_result: PipelineResultWrapper):
        args = self._gen_filename_base(args_ctx)

        if args_ctx.sdxl_high_noise_fraction is not None:
            args += ['hnf', args_ctx.sdxl_high_noise_fraction]

        filename = self._gen_filename(*args, 'step', self._generation_step + 1, ext='png')
        self._written_images.append(os.path.abspath(filename))
        self._write_generation_result(filename, generation_result, generation_result.dgenerate_config)

    def _write_prompt_only_image(self, args_ctx: DiffusionArgContext, generation_result: PipelineResultWrapper):
        args = ['s', args_ctx.seed,
                'g', args_ctx.guidance_scale,
                'i', args_ctx.inference_steps]

        if args_ctx.sdxl_high_noise_fraction is not None:
            args += ['hnf', args_ctx.sdxl_high_noise_fraction]

        filename = self._gen_filename(*args, 'step', self._generation_step + 1, ext='png')
        self._written_images.append(os.path.abspath(filename))
        self._write_generation_result(filename, generation_result, generation_result.dgenerate_config)

    def _pre_generation_step(self, args_ctx: DiffusionArgContext):
        self._last_frame_time = 0
        self._frame_time_sum = 0
        self._generation_step += 1

        desc = args_ctx.describe_parameters()

        messages.log(
            f'Generation step {self._generation_step + 1} / {self.num_generation_steps}\n'
            + desc, underline=True)

    def _pre_generation(self, args_ctx):
        pass

    def _animation_frame_pre_generation(self, args_ctx: DiffusionArgContext, image_seed_obj):
        if self._last_frame_time == 0:
            eta = 'tbd...'
        else:
            self._frame_time_sum += time.time() - self._last_frame_time
            eta_seconds = (self._frame_time_sum / image_seed_obj.frame_index) * (
                    image_seed_obj.total_frames - image_seed_obj.frame_index)
            eta = str(datetime.timedelta(seconds=eta_seconds))

        self._last_frame_time = time.time()
        messages.log(
            f'Generating frame {image_seed_obj.frame_index + 1} / {image_seed_obj.total_frames}, Completion ETA: {eta}',
            underline=True)

    def _with_image_seed_pre_generation(self, args_ctx: DiffusionArgContext, image_seed_obj):
        pass

    def _with_control_image_pre_generation(self, args_ctx: DiffusionArgContext, image_seed_obj):
        pass

    def _iterate_diffusion_args(self, **overrides):
        def ov(n, v):
            if not model_type_is_sdxl(self.model_type):
                if n.startswith('sdxl'):
                    return None
            else:
                if n.startswith('sdxl_refiner') and not self.sdxl_refiner_path:
                    return None

            if n in overrides:
                return overrides[n]
            return v

        yield from iterate_diffusion_args(
            prompts=ov('prompts', self.prompts),
            sdxl_second_prompts=ov('sdxl_second_prompts',
                                   self.sdxl_second_prompts),
            sdxl_refiner_prompts=ov('sdxl_refiner_prompts',
                                    self.sdxl_refiner_prompts),
            sdxl_refiner_second_prompts=ov('sdxl_refiner_second_prompts',
                                           self.sdxl_refiner_second_prompts),
            seeds=ov('seeds', self.seeds),
            image_seed_strengths=ov('image_seed_strengths', self.image_seed_strengths),
            guidance_scales=ov('guidance_scales', self.guidance_scales),
            image_guidance_scales=ov('image_guidance_scales', self.image_guidance_scales),
            guidance_rescales=ov('guidance_rescales', self.guidance_rescales),
            inference_steps_list=ov('inference_steps_list', self.inference_steps),
            sdxl_high_noise_fractions=ov('sdxl_high_noise_fractions', self.sdxl_high_noise_fractions),
            upscaler_noise_levels=ov('upscaler_noise_levels', self.upscaler_noise_levels),
            sdxl_aesthetic_scores=ov('sdxl_aesthetic_scores', self.sdxl_aesthetic_scores),
            sdxl_original_sizes=ov('sdxl_original_sizes', self.sdxl_original_sizes),
            sdxl_target_sizes=ov('sdxl_target_sizes', self.sdxl_target_sizes),
            sdxl_crops_coords_top_left=ov('sdxl_crops_coords_top_left', self.sdxl_crops_coords_top_left),
            sdxl_negative_aesthetic_scores=ov('sdxl_negative_aesthetic_scores', self.sdxl_negative_aesthetic_scores),
            sdxl_negative_original_sizes=ov('sdxl_negative_original_sizes', self.sdxl_negative_original_sizes),
            sdxl_negative_target_sizes=ov('sdxl_negative_target_sizes', self.sdxl_negative_target_sizes),
            sdxl_negative_crops_coords_top_left=ov('sdxl_negative_crops_coords_top_left',
                                                   self.sdxl_negative_crops_coords_top_left),
            sdxl_refiner_aesthetic_scores=ov('sdxl_refiner_aesthetic_scores', self.sdxl_refiner_aesthetic_scores),
            sdxl_refiner_original_sizes=ov('sdxl_refiner_original_sizes', self.sdxl_refiner_original_sizes),
            sdxl_refiner_target_sizes=ov('sdxl_refiner_target_sizes', self.sdxl_refiner_target_sizes),
            sdxl_refiner_crops_coords_top_left=ov('sdxl_refiner_crops_coords_top_left',
                                                  self.sdxl_refiner_crops_coords_top_left),
            sdxl_refiner_negative_aesthetic_scores=ov('sdxl_refiner_negative_aesthetic_scores',
                                                      self.sdxl_refiner_negative_aesthetic_scores),
            sdxl_refiner_negative_original_sizes=ov('sdxl_refiner_negative_original_sizes',
                                                    self.sdxl_refiner_negative_original_sizes),
            sdxl_refiner_negative_target_sizes=ov('sdxl_refiner_negative_target_sizes',
                                                  self.sdxl_refiner_negative_target_sizes),
            sdxl_refiner_negative_crops_coords_top_left=ov('sdxl_refiner_negative_crops_coords_top_left',
                                                           self.sdxl_refiner_negative_crops_coords_top_left))

    def run(self):
        self._enforce_state()

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self._written_images = []
        self._written_animations = []
        self._generation_step = -1
        self._frame_time_sum = 0
        self._last_frame_time = 0

        generation_steps = self.num_generation_steps

        if generation_steps == 0:
            messages.log(f'Options resulted in no generation steps, nothing to do.', underline=True)
            return

        messages.log(f'Beginning {generation_steps} generation steps...', underline=True)

        if self.image_seeds:
            self._render_with_image_seeds()
        elif self.control_images:
            self._render_with_control_images()
        else:
            diffusion_model = DiffusionPipelineWrapper(self.model_path,
                                                       model_subfolder=self.model_subfolder,
                                                       dtype=self.dtype,
                                                       device=self.device,
                                                       model_type=self.model_type,
                                                       revision=self.revision,
                                                       variant=self.variant,
                                                       vae_path=self.vae_path,
                                                       lora_paths=self.lora_paths,
                                                       textual_inversion_paths=self.textual_inversion_paths,
                                                       scheduler=self.scheduler,
                                                       safety_checker=self.safety_checker,
                                                       sdxl_refiner_path=self.sdxl_refiner_path,
                                                       auth_token=self.auth_token)

            sdxl_high_noise_fractions = self.sdxl_high_noise_fractions if self.sdxl_refiner_path is not None else None

            for args_ctx in self._iterate_diffusion_args(sdxl_high_noise_fractions=sdxl_high_noise_fractions,
                                                         image_seed_strengths=None,
                                                         upscaler_noise_levels=None):
                self._pre_generation_step(args_ctx)
                self._pre_generation(args_ctx)

                with diffusion_model(**args_ctx.args,
                                     seed=args_ctx.seed,
                                     width=self.output_size[0],
                                     height=self.output_size[1]) as generation_result:
                    self._write_prompt_only_image(args_ctx, generation_result)

    def _render_with_control_images(self):
        diffusion_model = DiffusionPipelineWrapper(self.model_path,
                                                   model_subfolder=self.model_subfolder,
                                                   dtype=self.dtype,
                                                   device=self.device,
                                                   model_type=self.model_type,
                                                   revision=self.revision,
                                                   variant=self.variant,
                                                   vae_path=self.vae_path,
                                                   vae_tiling=self.vae_tiling,
                                                   vae_slicing=self.vae_slicing,
                                                   lora_paths=self.lora_paths,
                                                   textual_inversion_paths=self.textual_inversion_paths,
                                                   control_net_paths=self.control_net_paths,
                                                   scheduler=self.scheduler,
                                                   safety_checker=self.safety_checker,
                                                   sdxl_refiner_path=self.sdxl_refiner_path,
                                                   auth_token=self.auth_token)

        sdxl_high_noise_fractions = self.sdxl_high_noise_fractions if self.sdxl_refiner_path is not None else None

        for control_image in self.control_images:

            messages.log(f'Processing Control Image: {control_image}', underline=True)

            arg_iterator = self._iterate_diffusion_args(sdxl_high_noise_fractions=sdxl_high_noise_fractions,
                                                        image_seed_strengths=None,
                                                        upscaler_noise_levels=None)

            control_image_info = get_control_image_info(control_image, self.frame_start, self.frame_end)

            def control_iterator_func():
                yield from iterate_control_image(
                    control_image,
                    self.frame_start,
                    self.frame_end,
                    self.output_size,
                    preprocessor=preprocessors.load(self.control_image_preprocessors))

            if control_image_info.is_animation:
                def get_extra_args(ci_obj: ImageSeed):
                    return {'control_image': ci_obj.image}

                self._render_animation(diffusion_model,
                                       arg_iterator,
                                       control_image_info.fps,
                                       seed_iterator_func=control_iterator_func,
                                       get_extra_args=get_extra_args)
                break

            for args_ctx in arg_iterator:
                self._pre_generation_step(args_ctx)
                with next(control_iterator_func()) as control_image_obj:
                    with control_image_obj as cimg_obj:
                        self._with_control_image_pre_generation(args_ctx, cimg_obj)

                        with control_image_obj.image, diffusion_model(**args_ctx.args,
                                                                      control_image=control_image_obj.image,
                                                                      seed=args_ctx.seed) as generation_result:
                            self._write_image_seed_gen_image(args_ctx, generation_result)

    def _render_with_image_seeds(self):
        diffusion_model = DiffusionPipelineImg2ImgWrapper(self.model_path,
                                                          model_subfolder=self.model_subfolder,
                                                          dtype=self.dtype,
                                                          device=self.device,
                                                          model_type=self.model_type,
                                                          revision=self.revision,
                                                          variant=self.variant,
                                                          vae_path=self.vae_path,
                                                          vae_tiling=self.vae_tiling,
                                                          vae_slicing=self.vae_slicing,
                                                          lora_paths=self.lora_paths,
                                                          textual_inversion_paths=self.textual_inversion_paths,
                                                          control_net_paths=self.control_net_paths,
                                                          scheduler=self.scheduler,
                                                          safety_checker=self.safety_checker,
                                                          sdxl_refiner_path=self.sdxl_refiner_path,
                                                          auth_token=self.auth_token)

        sdxl_high_noise_fractions = self.sdxl_high_noise_fractions if self.sdxl_refiner_path is not None else None

        image_seed_strengths = self.image_seed_strengths if \
            not (model_type_is_upscaler(self.model_type) or model_type_is_pix2pix(self.model_type)) else None

        upscaler_noise_levels = self.upscaler_noise_levels if \
            self.model_type == ModelTypes.TORCH_UPSCALER_X4 else None

        for image_seed in self.image_seeds:

            messages.log(f'Processing Image Seed: {image_seed}', underline=True)

            arg_iterator = self._iterate_diffusion_args(
                sdxl_high_noise_fractions=sdxl_high_noise_fractions,
                image_seed_strengths=image_seed_strengths,
                upscaler_noise_levels=upscaler_noise_levels
            )

            seed_info = get_image_seed_info(image_seed, self.frame_start, self.frame_end)

            def seed_iterator_func():
                yield from iterate_image_seed(
                    image_seed,
                    self.frame_start,
                    self.frame_end,
                    self.output_size,
                    seed_image_preprocessor=preprocessors.load(self.seed_image_preprocessors),
                    mask_image_preprocessor=preprocessors.load(self.mask_image_preprocessors),
                    control_image_preprocessor=preprocessors.load(self.control_image_preprocessors))

            if seed_info.is_animation:
                def get_extra_args(ims_obj: ImageSeed):
                    extra_args = {'image': ims_obj.image}
                    if ims_obj.mask_image is not None:
                        extra_args['mask_image'] = ims_obj.mask_image
                    if ims_obj.control_image is not None:
                        extra_args['control_image'] = ims_obj.control_image
                    return extra_args

                self._render_animation(diffusion_model,
                                       arg_iterator,
                                       seed_info.fps,
                                       seed_iterator_func=seed_iterator_func,
                                       get_extra_args=get_extra_args)
                break

            for args_ctx in arg_iterator:
                self._pre_generation_step(args_ctx)
                with next(seed_iterator_func()) as image_obj:
                    with image_obj as image_seed_obj:
                        self._with_image_seed_pre_generation(args_ctx, image_seed_obj)

                        if image_seed_obj.mask_image is not None:
                            args_ctx.args['mask_image'] = image_seed_obj.mask_image

                        if image_seed_obj.control_image is None:
                            if self.control_net_paths:
                                raise NotImplementedError(
                                    'Cannot use --control-nets without a control image, '
                                    'see --image-seeds and --control-images for information '
                                    'on specifying a control image.')

                        else:
                            args_ctx.args['control_image'] = image_seed_obj.control_image

                        with MultiContextManager([image_seed_obj.mask_image, image_seed_obj.control_image]), \
                                diffusion_model(**args_ctx.args,
                                                image=image_seed_obj.image,
                                                seed=args_ctx.seed) as generation_result:
                            self._write_image_seed_gen_image(args_ctx, generation_result)

    def _render_animation(self, diffusion_model, arg_iterator, fps, seed_iterator_func, get_extra_args):

        animation_format_lower = self.animation_format.lower()
        first_args_ctx = next(arg_iterator)

        out_filename = self._gen_animation_filename(first_args_ctx, self._generation_step + 1,
                                                    animation_format_lower)
        next_frame_terminates_anim = False

        with create_animation_writer(animation_format_lower, out_filename, fps) as video_writer:

            for args_ctx in itertools.chain([first_args_ctx], arg_iterator):
                self._pre_generation_step(args_ctx)

                if next_frame_terminates_anim:
                    next_frame_terminates_anim = False
                    video_writer.end(
                        new_file=self._gen_animation_filename(args_ctx, self._generation_step,
                                                              animation_format_lower))

                if self.output_configs:
                    anim_config_file_name = os.path.splitext(video_writer.filename)[0] + '.txt'
                    messages.log(
                        f'Writing Animation: {video_writer.filename}\nWriting Config File: {anim_config_file_name}',
                        underline=True)
                else:
                    messages.log(f'Writing Animation: {video_writer.filename}', underline=True)

                self._written_animations.append(os.path.abspath(video_writer.filename))

                for image_obj in seed_iterator_func():

                    with image_obj as image_seed_obj:
                        self._animation_frame_pre_generation(args_ctx, image_seed_obj)

                        extra_args = get_extra_args(image_seed_obj)

                        with diffusion_model(**args_ctx.args,
                                             **extra_args,
                                             seed=args_ctx.seed) as generation_result:
                            video_writer.write(generation_result.image)

                            if self.output_configs:
                                if not os.path.exists(anim_config_file_name):
                                    config_text = generation_result.dgenerate_config

                                    if self.frame_start is not None:
                                        config_text += f' \\\n--frame-start {self.frame_start}'

                                    if self.frame_end is not None:
                                        config_text += f' \\\n--frame-end {self.frame_end}'

                                    if self.animation_format is not None:
                                        config_text += f' \\\n--animation-format {self.animation_format}'

                                    with open(anim_config_file_name, "w") as config_file:
                                        config_file.write(config_text)

                            self._write_animation_frame(args_ctx, image_seed_obj, generation_result)

                        next_frame_terminates_anim = image_seed_obj.frame_index == (image_seed_obj.total_frames - 1)

                video_writer.end()
