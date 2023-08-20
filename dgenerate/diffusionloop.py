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

import re
import torch
import datetime
import itertools
import math
import os
import time
from pathlib import Path

from .mediainput import iterate_image_seed, get_image_seed_info
from .mediaoutput import create_animation_writer, supported_animation_writer_formats
from .pipelinewrappers import DiffusionPipelineWrapper, DiffusionPipelineImg2ImgWrapper, supported_model_types
from .textprocessing import oxford_comma, underline


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
    def __init__(self, diffusion_args, prompt, seed, image_seed_strength, guidance_scale, inference_steps):
        self.args = diffusion_args
        self.prompt = prompt
        self.seed = seed
        self.image_seed_strength = image_seed_strength
        self.guidance_scale = guidance_scale
        self.inference_steps = inference_steps


def iterate_diffusion_args(prompts, seeds, image_seed_strengths, guidance_scales, inference_steps_list):
    diffusion_args = dict()

    has_image_seed = image_seed_strengths is not None and len(image_seed_strengths) > 0

    for prompt, seed, image_seed_strength, guidance_scale, inference_steps in itertools.product(
            prompts,
            seeds,
            image_seed_strengths if has_image_seed else [None],
            guidance_scales,
            inference_steps_list):

        if has_image_seed:
            diffusion_args['strength'] = image_seed_strength

        diffusion_args['num_inference_steps'] = (
            math.ceil(inference_steps / image_seed_strength if image_seed_strength > 0 else inference_steps)
            if has_image_seed else inference_steps)

        diffusion_args['guidance_scale'] = (
            math.ceil(guidance_scale / image_seed_strength if image_seed_strength > 0 else guidance_scale)
            if has_image_seed else guidance_scale)

        diffusion_args.update(prompt)

        yield DiffusionArgContext(diffusion_args,
                                  prompt,
                                  seed,
                                  image_seed_strength,
                                  guidance_scale,
                                  inference_steps)


class DiffusionRenderLoop:
    def __init__(self):
        self._generation_step = -1
        self._frame_time_sum = 0
        self._last_frame_time = 0

        self.model_path = None
        self.vae = None
        self.scheduler = None
        self.safety_checker = False
        self.model_type = 'torch'
        self.device = 'cuda'
        self.dtype = 'float16'
        self.revision = 'main'
        self.variant = None
        self.output_size = (512, 512)
        self.output_path = os.path.join(os.getcwd(), 'output')
        self.prompts = []
        self.seeds = [0]
        self.image_seeds = []
        self.animation_format = 'mp4'
        self.frame_start = 0
        self.frame_end = None
        self.image_seed_strengths = []
        self.guidance_scales = []
        self.inference_steps = []

    def _enforce_state(self):
        if self.dtype not in {'float32', 'float16', 'auto'}:
            raise ValueError('DiffusionRenderLoop.torch_dtype must be float32, float16, or auto')
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
        if self.vae is not None and not isinstance(self.vae, str):
            raise ValueError('DiffusionRenderLoop.vae must be a string: AutoencoderClass;PATH')
        if self.scheduler is not None and not isinstance(self.scheduler, str):
            raise ValueError('DiffusionRenderLoop.scheduler must be a string that names a compatible scheduler class.')
        if self.output_path is None:
            raise ValueError('DiffusionRenderLoop.output_path must not be None')
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
        if self.output_size is None and len(self.image_seeds) == 0:
            raise ValueError('DiffusionRenderLoop.output_size must not be None when no image seeds specified')
        if not _has_len(self.image_seed_strengths):
            raise ValueError('DiffusionRenderLoop.seeds must have len')
        if not _has_len(self.guidance_scales):
            raise ValueError('DiffusionRenderLoop.guidance_scales must have len')
        if not _has_len(self.inference_steps):
            raise ValueError('DiffusionRenderLoop.inference_steps must have len')
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
                len(self.prompts) *
                len(self.seeds) *
                len(self.guidance_scales) *
                max(len(self.image_seed_strengths), 1) *
                len(self.inference_steps))

    @property
    def generation_step(self):
        return self._generation_step

    def _gen_filename(self, *args, ext):
        return os.path.join(self.output_path, '_'.join(str(s).replace('.', '-') for s in args) + '.' + ext)

    def _gen_animation_filename(self, args_ctx, generation_step, animation_format):
        return self._gen_filename('ANIM',
                                  's', args_ctx.seed,
                                  'st', args_ctx.image_seed_strength,
                                  'g', args_ctx.guidance_scale,
                                  'i', args_ctx.inference_steps,
                                  'step', generation_step + 1,
                                  ext=animation_format)

    def _write_animation_frame(self, args_ctx, image_seed_obj, img):
        filename = self._gen_filename('s', args_ctx.seed,
                                      'st', args_ctx.image_seed_strength,
                                      'g', args_ctx.guidance_scale,
                                      'i', args_ctx.inference_steps,
                                      'frame', image_seed_obj.frame_index + 1,
                                      'step', self._generation_step + 1,
                                      ext='png')
        img.save(filename)
        print(underline(f'Wrote File: {filename}'))

    def _write_image_seed_gen_image(self, args_ctx, img):
        filename = self._gen_filename('s', args_ctx.seed,
                                      'st', args_ctx.image_seed_strength,
                                      'g', args_ctx.guidance_scale,
                                      'i', args_ctx.inference_steps,
                                      'step', self._generation_step + 1,
                                      ext='png')
        img.save(filename)
        print(underline(f'Wrote File: {filename}'))

    def _write_prompt_only_image(self, args_ctx, img):
        filename = self._gen_filename('s', args_ctx.seed,
                                      'g', args_ctx.guidance_scale,
                                      'i', args_ctx.inference_steps,
                                      'step', self._generation_step + 1,
                                      ext='png')
        img.save(filename)
        print(underline(f'Wrote File: {filename}'))

    def _pre_generation_step(self, args_ctx):
        self._last_frame_time = 0
        self._frame_time_sum = 0
        self._generation_step += 1

        prompt_format = []

        val = args_ctx.prompt["prompt"]
        if val is not None and len(val) > 0:
            prompt_format.append(f'Prompt: "{val}"')

        if 'negative_prompt' in args_ctx.prompt:
            val = args_ctx.prompt["negative_prompt"]
            if val is not None and len(val) > 0:
                prompt_format.append(f'Negative Prompt: "{val}"')

        prompt_format = '\n'.join(prompt_format)
        if len(prompt_format) > 0:
            prompt_format = '\n' + prompt_format

        inputs = [f'Seed: {args_ctx.seed}']

        if args_ctx.image_seed_strength is not None:
            inputs.append(f'Image Seed Strength: {args_ctx.image_seed_strength}')

        inputs.append(f'Guidance Scale: {args_ctx.guidance_scale}')
        inputs.append(f'Inference Steps: {args_ctx.inference_steps}')

        inputs = '\n'.join(inputs)

        print(underline(
            f'Generation step {self._generation_step + 1} / {self.num_generation_steps}\n'
            + inputs + prompt_format
        ))

    def _pre_generation(self, args_ctx):
        pass

    def _animation_frame_pre_generation(self, args_ctx, image_seed_obj):
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

    def _with_image_seed_pre_generation(self, args_ctx, image_seed_obj):
        pass

    def run(self):
        self._enforce_state()

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self._generation_step = -1
        self._frame_time_sum = 0
        self._last_frame_time = 0

        generation_steps = self.num_generation_steps

        if generation_steps == 0:
            print(underline(f'Options resulted in no generation steps, nothing to do.'))
            return

        print(underline(f'Beginning {generation_steps} generation steps...'))

        if len(self.image_seeds) > 0:
            self._render_with_image_seeds()
        else:

            diffusion_model = DiffusionPipelineWrapper(self.model_path,
                                                       dtype=self.dtype,
                                                       device=self.device,
                                                       model_type=self.model_type,
                                                       revision=self.revision,
                                                       variant=self.variant,
                                                       vae=self.vae,
                                                       scheduler=self.scheduler,
                                                       safety_checker=self.safety_checker)

            for args_ctx in iterate_diffusion_args(self.prompts, self.seeds, [], self.guidance_scales,
                                                   self.inference_steps):
                self._pre_generation_step(args_ctx)
                self._pre_generation(args_ctx)
                with diffusion_model(**args_ctx.args,
                                     seed=args_ctx.seed,
                                     width=self.output_size[0],
                                     height=self.output_size[1]).images[0] as gen_img:
                    self._write_prompt_only_image(args_ctx, gen_img)

    def _render_with_image_seeds(self):
        diffusion_model = DiffusionPipelineImg2ImgWrapper(self.model_path,
                                                          dtype=self.dtype,
                                                          device=self.device,
                                                          model_type=self.model_type,
                                                          revision=self.revision,
                                                          variant=self.variant,
                                                          vae=self.vae,
                                                          scheduler=self.scheduler,
                                                          safety_checker=self.safety_checker)

        for image_seed in self.image_seeds:

            print(underline(f'Processing Image Seed: {image_seed}'))
            arg_iterator = iterate_diffusion_args(self.prompts, self.seeds, self.image_seed_strengths,
                                                  self.guidance_scales, self.inference_steps)

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
                            with image_seed_obj.mask_image as mask_image, \
                                    diffusion_model(**args_ctx.args,
                                                    image=image_seed_obj.image,
                                                    mask_image=mask_image,
                                                    seed=args_ctx.seed).images[0] as gen_img:
                                self._write_image_seed_gen_image(args_ctx, gen_img)
                        else:
                            with diffusion_model(**args_ctx.args,
                                                 image=image_seed_obj.image,
                                                 seed=args_ctx.seed).images[0] as gen_img:
                                self._write_image_seed_gen_image(args_ctx, gen_img)

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

                print(underline(f'Writing Animation: {video_writer.filename}'))

                for image_obj in iterate_image_seed(image_seed, self.frame_start, self.frame_end,
                                                    self.output_size):
                    with image_obj as image_seed_obj:
                        self._animation_frame_pre_generation(args_ctx, image_seed_obj)

                        if image_seed_obj.mask_image is not None:
                            with diffusion_model(**args_ctx.args,
                                                 seed=args_ctx.seed,
                                                 image=image_seed_obj.image,
                                                 mask_image=image_seed_obj.mask_image).images[0] as gen_img:
                                video_writer.write(gen_img)
                                self._write_animation_frame(args_ctx, image_seed_obj, gen_img)
                        else:
                            with diffusion_model(**args_ctx.args,
                                                 seed=args_ctx.seed,
                                                 image=image_seed_obj.image).images[0] as gen_img:
                                video_writer.write(gen_img)
                                self._write_animation_frame(args_ctx, image_seed_obj, gen_img)

                        next_frame_terminates_anim = image_seed_obj.frame_index == (image_seed_obj.total_frames - 1)

                video_writer.end()
