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

from .mediainput import iterate_image_seed, get_image_seed_info, create_and_exif_orient_pil_img, MultiContextManager
from .mediaoutput import create_animation_writer, supported_animation_writer_formats
from .pipelinewrappers import DiffusionPipelineWrapper, DiffusionPipelineImg2ImgWrapper, supported_model_types, \
    PipelineResultWrapper, model_type_is_upscaler, get_model_type_enum, ModelTypes
from .textprocessing import oxford_comma, underline, long_text_wrap_width


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
    def __init__(self, diffusion_args, prompt, seed, image_seed_strength, guidance_scale, inference_steps,
                 sdxl_high_noise_fraction, upscaler_noise_level):
        self.args = diffusion_args
        self.prompt = prompt
        self.seed = seed
        self.image_seed_strength = image_seed_strength
        self.guidance_scale = guidance_scale
        self.inference_steps = inference_steps
        self.sdxl_high_noise_fraction = sdxl_high_noise_fraction
        self.upscaler_noise_level = upscaler_noise_level


def iterate_diffusion_args(prompts, control_images, seeds, image_seed_strengths, guidance_scales, inference_steps_list,
                           sdxl_high_noise_fractions, upscaler_noise_levels) -> Iterator[DiffusionArgContext]:
    diffusion_args = dict()

    has_control_images = control_images is not None and len(control_images) > 0
    has_image_seed = image_seed_strengths is not None and len(image_seed_strengths) > 0
    has_high_noise_fractions = sdxl_high_noise_fractions is not None and len(sdxl_high_noise_fractions) > 0
    has_upscaler_noise_levels = upscaler_noise_levels is not None and len(upscaler_noise_levels) > 0

    for prompt, control_image, seed, image_seed_strength, upscaler_noise_level, sdxl_high_noise_fraction, \
        guidance_scale, inference_steps in itertools.product(
        prompts,
        control_images if has_control_images else [None],
        seeds,
        image_seed_strengths if has_image_seed else [None],
        upscaler_noise_levels if has_upscaler_noise_levels else [None],
        sdxl_high_noise_fractions if has_high_noise_fractions else [None],
        guidance_scales,
        inference_steps_list
    ):

        if has_control_images:
            diffusion_args['control_image'] = control_image

        if has_image_seed:
            diffusion_args['strength'] = image_seed_strength

        if has_high_noise_fractions:
            diffusion_args['sdxl_high_noise_fraction'] = sdxl_high_noise_fraction

        if has_upscaler_noise_levels:
            diffusion_args['noise_level'] = upscaler_noise_level

        diffusion_args['num_inference_steps'] = (
            math.ceil(inference_steps / image_seed_strength if image_seed_strength > 0 else inference_steps)
            if has_image_seed else inference_steps)

        diffusion_args['guidance_scale'] = (
            (guidance_scale / image_seed_strength if image_seed_strength > 0 else guidance_scale)
            if has_image_seed else guidance_scale)

        diffusion_args.update(prompt)

        yield DiffusionArgContext(diffusion_args,
                                  prompt,
                                  seed,
                                  image_seed_strength,
                                  guidance_scale,
                                  inference_steps,
                                  sdxl_high_noise_fraction,
                                  upscaler_noise_level)


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
        self.sdxl_high_noise_fractions = []
        self.sdxl_original_size = None
        self.sdxl_target_size = None
        self.vae_path = None
        self.lora_paths = None
        self.textual_inversion_paths = None
        self.control_net_paths = None
        self.scheduler = None
        self.safety_checker = False
        self.model_type = 'torch'
        self.device = 'cuda'
        self.dtype = 'auto'
        self.revision = 'main'
        self.variant = None
        self.output_size = (512, 512)
        self.output_path = os.path.join(os.getcwd(), 'output')
        self.output_prefix = None
        self.output_overwrite = False
        self.output_configs = False
        self.output_metadata = False
        self.prompts = []
        self.seeds = [0]
        self.image_seeds = []
        self.control_images = []
        self.animation_format = 'mp4'
        self.frame_start = 0
        self.frame_end = None
        self.image_seed_strengths = []
        self.upscaler_noise_levels = []
        self.guidance_scales = []
        self.inference_steps = []
        self.auth_token = None

    @property
    def written_images(self):
        return self._written_images

    @property
    def written_animations(self):
        return self._written_animations

    def _enforce_state(self):
        if self.dtype not in {'float32', 'float16', 'auto'}:
            raise ValueError('DiffusionRenderLoop.torch_dtype must be float32, float16 or auto')
        if not isinstance(self.safety_checker, bool):
            raise ValueError('DiffusionRenderLoop.safety_checker must be True or False (bool)')
        if self.revision is not None and not isinstance(self.revision, str):
            raise ValueError('DiffusionRenderLoop.revision must be None or a string')
        if self.variant is not None and not isinstance(self.variant, str):
            raise ValueError('DiffusionRenderLoop.variant must be None or a string')
        if self.model_type not in supported_model_types():
            raise ValueError(
                f'DiffusionRenderLoop.model_type must be one of: {oxford_comma(supported_model_types(), "or")}')
        if self.model_path is None:
            raise ValueError('DiffusionRenderLoop.model_path must not be None')
        if self.model_subfolder is not None and not isinstance(self.model_subfolder, str):
            raise ValueError('DiffusionRenderLoop.model_subfolder must be None or str')
        if self.auth_token is not None and not isinstance(self.auth_token, str):
            raise ValueError('DiffusionRenderLoop.auth_token must be None or str')
        if self.lora_paths is not None and not isinstance(self.lora_paths, str):
            raise ValueError('DiffusionRenderLoop.lora_paths must be None or str')
        if self.textual_inversion_paths is not None and not \
                isinstance(self.textual_inversion_paths, str) and not _has_len(self.textual_inversion_paths):
            raise ValueError('DiffusionRenderLoop.textual_inversion_paths must be None or str or have len')
        if self.control_net_paths is not None and not \
                isinstance(self.control_net_paths, str) and not _has_len(self.control_net_paths):
            raise ValueError('DiffusionRenderLoop.control_net_paths must be None or str or have len')
        if self.sdxl_refiner_path is not None and not isinstance(self.sdxl_refiner_path, str):
            raise ValueError('DiffusionRenderLoop.sdxl_refiner_path must be None or a string')
        if self.vae_path is not None and not isinstance(self.vae_path, str):
            raise ValueError('DiffusionRenderLoop.vae_path must be a string: AutoencoderClass;PATH')
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
        if not _has_len(self.prompts):
            raise ValueError('DiffusionRenderLoop.prompts must have len')
        if not _has_len(self.seeds):
            raise ValueError('DiffusionRenderLoop.seeds must have len')
        if not _has_len(self.image_seeds):
            raise ValueError('DiffusionRenderLoop.image_seeds must have len')
        if not _has_len(self.control_images):
            raise ValueError('DiffusionRenderLoop.control_images must have len')
        if self.output_size is None and len(self.image_seeds) == 0:
            raise ValueError('DiffusionRenderLoop.output_size must not be None when no image seeds specified')
        if self.output_size is not None and not isinstance(self.output_size, tuple):
            raise ValueError('DiffusionRenderLoop.output_size must be None or a tuple')
        if self.sdxl_original_size is not None and not isinstance(self.sdxl_original_size, tuple):
            raise ValueError('DiffusionRenderLoop.sdxl_original_size must be None or a tuple')
        if self.sdxl_target_size is not None and not isinstance(self.sdxl_target_size, tuple):
            raise ValueError('DiffusionRenderLoop.sdxl_target_size must be None or a tuple')
        if not _has_len(self.image_seed_strengths):
            raise ValueError('DiffusionRenderLoop.image_seed_strengths must have len')
        if not _has_len(self.guidance_scales):
            raise ValueError('DiffusionRenderLoop.guidance_scales must have len')
        if not _has_len(self.inference_steps):
            raise ValueError('DiffusionRenderLoop.inference_steps must have len')
        if not _has_len(self.sdxl_high_noise_fractions):
            raise ValueError('DiffusionRenderLoop.sdxl_high_noise_fractions must have len')
        if not isinstance(self.frame_start, int) or not self.frame_start >= 0:
            raise ValueError('DiffusionRenderLoop.frame_start must be an integer value greater than or equal to zero')
        if self.frame_end is not None and (not isinstance(self.frame_end, int) or not self.frame_end >= 0):
            raise ValueError('DiffusionRenderLoop.frame_end must be an integer value greater than or equal to zero')
        if self.frame_end is not None and self.frame_start > self.frame_end:
            raise ValueError(
                'DiffusionRenderLoop.frame_start must be an integer value less than DiffusionRenderLoop.frame_end')

    @property
    def num_generation_steps(self):
        return (max(len(self.image_seeds), 1) *
                max(len(self.control_images), 1) *
                len(self.prompts) *
                len(self.seeds) *
                len(self.guidance_scales) *
                max(len(self.image_seed_strengths), 1) *
                max(len(self.upscaler_noise_levels), 1) *
                len(self.inference_steps) *
                max(len(self.sdxl_high_noise_fractions), 1))

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

        args = ['s', args_ctx.seed,
                *noise_entry,
                'g', args_ctx.guidance_scale,
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
            print(underline(
                f'Wrote Image File: {filename}\nWrote Config File: {config_file_name}'))
        else:
            print(underline(f'Wrote Image File: {filename}'))

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

        prompt_format = []

        prompt_wrap_width = long_text_wrap_width()

        val = args_ctx.prompt["prompt"]
        if val is not None and len(val) > 0:
            header = 'Prompt: '
            val = textwrap.fill(val, width=prompt_wrap_width - len(header),
                                subsequent_indent=' ' * len(header))
            prompt_format.append(f'{header}"{val}"')

        if 'negative_prompt' in args_ctx.prompt:
            val = args_ctx.prompt["negative_prompt"]
            if val is not None and len(val) > 0:
                header = 'Negative Prompt: '
                val = textwrap.fill(val,
                                    width=prompt_wrap_width - len(header),
                                    subsequent_indent=' ' * len(header))
                prompt_format.append(f'{header}"{val}"')

        prompt_format = '\n'.join(prompt_format)
        if len(prompt_format) > 0:
            prompt_format = '\n' + prompt_format

        inputs = [f'Seed: {args_ctx.seed}']

        if args_ctx.image_seed_strength is not None:
            inputs.append(f'Image Seed Strength: {args_ctx.image_seed_strength}')

        if args_ctx.upscaler_noise_level is not None:
            inputs.append(f'Upscaler Noise Level: {args_ctx.upscaler_noise_level}')

        inputs.append(f'Guidance Scale: {args_ctx.guidance_scale}')
        inputs.append(f'Inference Steps: {args_ctx.inference_steps}')

        if args_ctx.sdxl_high_noise_fraction is not None:
            inputs.append(f'SDXL High Noise Fraction: {args_ctx.sdxl_high_noise_fraction}')

        inputs = '\n'.join(inputs)

        print(underline(
            f'Generation step {self._generation_step + 1} / {self.num_generation_steps}\n'
            + inputs + prompt_format
        ))

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
        print(underline(
            f'Generating frame {image_seed_obj.frame_index + 1} / {image_seed_obj.total_frames}, Completion ETA: {eta}'))

    def _with_image_seed_pre_generation(self, args_ctx: DiffusionArgContext, image_seed_obj):
        pass

    def _load_control_images(self):
        if self.control_images is not None:
            images = []
            for i in self.control_images:
                images.append(create_and_exif_orient_pil_img(i, file_source=i))
            if len(images) == 0:
                return None
            else:
                return images
        else:
            return None

    def run(self):
        self._enforce_state()

        control_images = self._load_control_images()

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self._written_images = []
        self._written_animations = []
        self._generation_step = -1
        self._frame_time_sum = 0
        self._last_frame_time = 0

        generation_steps = self.num_generation_steps

        if generation_steps == 0:
            print(underline(f'Options resulted in no generation steps, nothing to do.'))
            return

        print(underline(f'Beginning {generation_steps} generation steps...'))

        if len(self.image_seeds) > 0:
            self._render_with_image_seeds(control_images)
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
                                                       control_net_paths=self.control_net_paths,
                                                       scheduler=self.scheduler,
                                                       safety_checker=self.safety_checker,
                                                       sdxl_refiner_path=self.sdxl_refiner_path,
                                                       auth_token=self.auth_token)

            sdxl_high_noise_fractions = self.sdxl_high_noise_fractions if self.sdxl_refiner_path is not None else None
            for args_ctx in iterate_diffusion_args(prompts=self.prompts,
                                                   control_images=control_images,
                                                   seeds=self.seeds,
                                                   image_seed_strengths=None,
                                                   guidance_scales=self.guidance_scales,
                                                   inference_steps_list=self.inference_steps,
                                                   sdxl_high_noise_fractions=sdxl_high_noise_fractions,
                                                   upscaler_noise_levels=None):
                self._pre_generation_step(args_ctx)
                self._pre_generation(args_ctx)

                with diffusion_model(**args_ctx.args,
                                     seed=args_ctx.seed,
                                     width=self.output_size[0],
                                     height=self.output_size[1],
                                     sdxl_original_size=self.sdxl_original_size,
                                     sdxl_target_size=self.sdxl_target_size) as generation_result:
                    self._write_prompt_only_image(args_ctx, generation_result)

    def _render_with_image_seeds(self, control_images):
        diffusion_model = DiffusionPipelineImg2ImgWrapper(self.model_path,
                                                          model_subfolder=self.model_subfolder,
                                                          dtype=self.dtype,
                                                          device=self.device,
                                                          model_type=self.model_type,
                                                          revision=self.revision,
                                                          variant=self.variant,
                                                          vae_path=self.vae_path,
                                                          lora_paths=self.lora_paths,
                                                          textual_inversion_paths=self.textual_inversion_paths,
                                                          control_net_paths=self.control_net_paths,
                                                          scheduler=self.scheduler,
                                                          safety_checker=self.safety_checker,
                                                          sdxl_refiner_path=self.sdxl_refiner_path,
                                                          auth_token=self.auth_token)

        for image_seed in self.image_seeds:

            print(underline(f'Processing Image Seed: {image_seed}'))

            sdxl_high_noise_fractions = self.sdxl_high_noise_fractions if self.sdxl_refiner_path is not None else None

            upscaler_noise_levels = self.upscaler_noise_levels if \
                get_model_type_enum(self.model_type) == ModelTypes.TORCH_UPSCALER_X4 else None

            image_seed_strengths = self.image_seed_strengths if not model_type_is_upscaler(self.model_type) else None

            arg_iterator = iterate_diffusion_args(prompts=self.prompts,
                                                  control_images=control_images,
                                                  seeds=self.seeds,
                                                  image_seed_strengths=image_seed_strengths,
                                                  guidance_scales=self.guidance_scales,
                                                  inference_steps_list=self.inference_steps,
                                                  sdxl_high_noise_fractions=sdxl_high_noise_fractions,
                                                  upscaler_noise_levels=upscaler_noise_levels)

            seed_info = get_image_seed_info(image_seed, self.frame_start, self.frame_end)

            if seed_info.is_animation:
                self._render_animation(image_seed, diffusion_model, arg_iterator, seed_info.fps)
                break

            for args_ctx in arg_iterator:
                self._pre_generation_step(args_ctx)
                with next(iterate_image_seed(image_seed, self.frame_start, self.frame_end,
                                             self.output_size)) as image_obj:
                    with image_obj as image_seed_obj:
                        self._with_image_seed_pre_generation(args_ctx, image_seed_obj)

                        if image_seed_obj.mask_image is not None:
                            args_ctx.args['mask_image'] = image_seed_obj.mask_image

                        if image_seed_obj.control_image is None:
                            if self.control_net_paths is not None and len(self.control_net_paths) > 0:
                                raise NotImplementedError(
                                    'Cannot use Control Nets without a control image, '
                                    'see --image-seeds and --control-images for information '
                                    'on specifying a control image.')

                        else:
                            args_ctx.args['control_image'] = image_seed_obj.control_image

                        with MultiContextManager([image_seed_obj.mask_image, image_seed_obj.control_image]), \
                                diffusion_model(**args_ctx.args,
                                                image=image_seed_obj.image,
                                                seed=args_ctx.seed,
                                                sdxl_original_size=self.sdxl_original_size,
                                                sdxl_target_size=self.sdxl_target_size) as generation_result:
                            self._write_image_seed_gen_image(args_ctx, generation_result)

    def _render_animation(self, image_seed, diffusion_model, arg_iterator, fps):
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
                    print(underline(
                        f'Writing Animation: {video_writer.filename}\nWriting Config File: {anim_config_file_name}'))
                else:
                    print(underline(f'Writing Animation: {video_writer.filename}'))

                self._written_animations.append(os.path.abspath(video_writer.filename))

                for image_obj in iterate_image_seed(image_seed, self.frame_start, self.frame_end,
                                                    self.output_size):
                    with image_obj as image_seed_obj:
                        self._animation_frame_pre_generation(args_ctx, image_seed_obj)

                        extra_args = {}
                        if image_seed_obj.mask_image is not None:
                            extra_args = {'mask_image': image_seed_obj.mask_image}

                        with diffusion_model(**args_ctx.args, **extra_args,
                                             seed=args_ctx.seed,
                                             image=image_seed_obj.image,
                                             sdxl_original_size=self.sdxl_original_size,
                                             sdxl_target_size=self.sdxl_target_size) as generation_result:
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
