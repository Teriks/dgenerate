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

__version__ = '1.1.0'

import sys


def _run_loop():
    import warnings
    warnings.filterwarnings('ignore')

    import re
    import os
    import shlex
    import torch
    import jinja2
    import textwrap
    import diffusers
    import transformers

    from .args import parse_args
    from .textprocessing import underline, long_text_wrap_width, quote, unquote
    from .diffusionloop import DiffusionRenderLoop

    from .pipelinewrappers import clear_model_cache, InvalidVaePathError, \
        InvalidSchedulerName, InvalidLoRAPathError, \
        InvalidTextualInversionPathError, InvalidSDXLRefinerPathError, \
        SchedulerHelpException

    from .preprocessors import ImagePreprocessorArgumentError, ImagePreprocessorNotFoundError, image_preprocessor_help

    from .mediainput import ImageSeedParseError, ImageSeedSizeMismatchError

    from . import messages

    import dgenerate.preprocessors.loader

    # The above modules take long enough to import that they must be in here in
    # order to handle keyboard interrupts without issues

    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    diffusers.logging.set_verbosity(diffusers.logging.CRITICAL)

    def parse_and_run(with_args):
        if with_args:
            if with_args[0] == '--image-preprocessor-help':
                sys.exit(image_preprocessor_help(with_args[1:]))

        arguments = parse_args(with_args)

        render_loop = DiffusionRenderLoop()
        render_loop.model_path = arguments.model_path
        render_loop.model_subfolder = arguments.subfolder
        render_loop.model_type = arguments.model_type
        render_loop.revision = arguments.revision
        render_loop.variant = arguments.variant
        render_loop.device = arguments.device
        render_loop.dtype = arguments.dtype
        render_loop.output_size = arguments.output_size
        render_loop.output_path = arguments.output_path
        render_loop.output_prefix = arguments.output_prefix
        render_loop.output_overwrite = arguments.output_overwrite
        render_loop.output_configs = arguments.output_configs
        render_loop.output_metadata = arguments.output_metadata

        render_loop.prompts = arguments.prompts
        render_loop.sdxl_second_prompts = arguments.sdxl_second_prompts
        render_loop.sdxl_refiner_prompts = arguments.sdxl_refiner_prompts
        render_loop.sdxl_refiner_second_prompts = arguments.sdxl_refiner_second_prompts

        render_loop.seeds = arguments.seeds
        render_loop.image_seeds = arguments.image_seeds
        render_loop.animation_format = arguments.animation_format
        render_loop.frame_start = arguments.frame_start
        render_loop.frame_end = arguments.frame_end
        render_loop.image_seed_strengths = arguments.image_seed_strengths
        render_loop.upscaler_noise_levels = arguments.upscaler_noise_levels
        render_loop.guidance_scales = arguments.guidance_scales
        render_loop.image_guidance_scales = arguments.image_guidance_scales
        render_loop.guidance_rescales = arguments.guidance_rescales
        render_loop.inference_steps = arguments.inference_steps

        render_loop.vae_path = arguments.vae
        render_loop.vae_tiling = arguments.vae_tiling
        render_loop.vae_slicing = arguments.vae_slicing

        render_loop.lora_paths = arguments.lora
        render_loop.textual_inversion_paths = arguments.textual_inversions
        render_loop.control_net_paths = arguments.control_nets
        render_loop.scheduler = arguments.scheduler
        render_loop.safety_checker = arguments.safety_checker
        render_loop.auth_token = arguments.auth_token

        render_loop.sdxl_refiner_path = arguments.sdxl_refiner
        render_loop.sdxl_refiner_scheduler = arguments.sdxl_refiner_scheduler
        render_loop.sdxl_high_noise_fractions = arguments.sdxl_high_noise_fractions
        render_loop.sdxl_refiner_inference_steps = arguments.sdxl_refiner_inference_steps
        render_loop.sdxl_refiner_guidance_scales = arguments.sdxl_refiner_guidance_scales
        render_loop.sdxl_refiner_guidance_rescales = arguments.sdxl_refiner_guidance_rescales

        render_loop.sdxl_aesthetic_scores = arguments.sdxl_aesthetic_scores
        render_loop.sdxl_original_sizes = arguments.sdxl_original_sizes
        render_loop.sdxl_target_sizes = arguments.sdxl_target_sizes
        render_loop.sdxl_crops_coords_top_left = arguments.sdxl_crops_coords_top_left

        render_loop.sdxl_negative_aesthetic_scores = arguments.sdxl_negative_aesthetic_scores
        render_loop.sdxl_negative_original_sizes = arguments.sdxl_negative_original_sizes
        render_loop.sdxl_negative_target_sizes = arguments.sdxl_negative_target_sizes
        render_loop.sdxl_negative_crops_coords_top_left = arguments.sdxl_negative_crops_coords_top_left

        render_loop.sdxl_refiner_aesthetic_scores = arguments.sdxl_refiner_aesthetic_scores
        render_loop.sdxl_refiner_original_sizes = arguments.sdxl_refiner_original_sizes
        render_loop.sdxl_refiner_target_sizes = arguments.sdxl_refiner_target_sizes
        render_loop.sdxl_refiner_crops_coords_top_left = arguments.sdxl_refiner_crops_coords_top_left

        render_loop.sdxl_refiner_negative_aesthetic_scores = arguments.sdxl_refiner_negative_aesthetic_scores
        render_loop.sdxl_refiner_negative_original_sizes = arguments.sdxl_refiner_negative_original_sizes
        render_loop.sdxl_refiner_negative_target_sizes = arguments.sdxl_refiner_negative_target_sizes
        render_loop.sdxl_refiner_negative_crops_coords_top_left = arguments.sdxl_refiner_negative_crops_coords_top_left

        render_loop.seed_image_preprocessors = arguments.seed_image_preprocessors
        render_loop.mask_image_preprocessors = arguments.mask_image_preprocessors
        render_loop.control_image_preprocessors = arguments.control_image_preprocessors

        dgenerate.preprocessors.loader.SEARCH_MODULES = arguments.plugin_modules

        if arguments.verbose:
            messages.LEVEL = messages.DEBUG
        else:
            # enable setting and unsetting in batch processing
            messages.LEVEL = messages.INFO

        # run the render loop
        try:
            try:
                render_loop.run()
            except SchedulerHelpException:
                pass
        except (ImageSeedParseError,
                ImageSeedSizeMismatchError,
                InvalidSDXLRefinerPathError,
                InvalidVaePathError,
                InvalidLoRAPathError,
                InvalidTextualInversionPathError,
                InvalidSchedulerName,
                ImagePreprocessorArgumentError,
                ImagePreprocessorNotFoundError,
                torch.cuda.OutOfMemoryError,
                NotImplementedError,
                EnvironmentError) as e:
            messages.log(f'Error: {e}', level=messages.ERROR)
            sys.exit(1)

        return {'last_image':
                    quote(render_loop.written_images[-1])
                    if render_loop.written_images else [],
                'last_images':
                    [quote(s) for s in render_loop.written_images],
                'last_animation':
                    quote(render_loop.written_animations[-1])
                    if render_loop.written_animations else [],
                'last_animations':
                    [quote(s) for s in render_loop.written_animations]}

    if not sys.stdin.isatty():
        template_args = {
            'last_image': '',
            'last_images': [],
            'last_animation': '',
            'last_animations': []
        }

        jinja_env = jinja2.Environment()
        jinja_env.globals['unquote'] = unquote
        jinja_env.filters['unquote'] = unquote
        jinja_env.globals['quote'] = quote
        jinja_env.filters['quote'] = quote

        continuation = ''

        for line_idx, line in enumerate(sys.stdin):
            line = line.strip()
            if line == '':
                continue
            if line.startswith('#'):
                versioning = re.match(r'#!\s+dgenerate\s+([0-9]+\.[0-9]+\.[0-9]+)', line)
                if versioning:
                    config_file_version = versioning.group(1)
                    cur_major_version = int(__version__.split('.')[0])
                    config_major_version = int(config_file_version.split('.')[0])
                    if cur_major_version != config_major_version:
                        messages.log(
                            f'WARNING: Failed version check on line {line_idx}, running an '
                            f'incompatible version of dgenerate! You are running version {__version__} '
                            f'and the config file specifies the required version: {config_file_version}'
                            , underline=True, level=messages.WARNING)
                continue

            if line.endswith('\\'):
                continuation += ' ' + line.rstrip(' \\')
            else:
                args = (continuation + ' ' + line).lstrip()

                if args.startswith('\\print'):
                    args = args.split(' ', 1)
                    if len(args) == 2:
                        messages.log(jinja_env.from_string(os.path.expandvars(args[1]))
                                     .render(**template_args))
                    continuation = ''
                    continue
                if args.startswith('\\clear_model_cache'):
                    clear_model_cache()
                    continuation = ''
                    continue

                templated_cmd = jinja_env. \
                    from_string(os.path.expandvars(args)).render(**template_args)

                extra_args = sys.argv[1:]

                shlexed = shlex.split(templated_cmd) + extra_args

                for idx, extra_arg in enumerate(extra_args):
                    if any(c.isspace() for c in extra_arg):
                        extra_args[idx] = quote(extra_arg)

                header = 'Processing Arguments: '
                args_wrapped = textwrap.fill(templated_cmd + ' ' + ' '.join(extra_args),
                                             width=long_text_wrap_width() - len(header),
                                             break_long_words=False,
                                             break_on_hyphens=False,
                                             subsequent_indent=' ' * len(header))

                messages.log(header + args_wrapped, underline=True)

                try:
                    template_args = parse_and_run(shlexed)
                except Exception as e:
                    messages.log(f'Error in input config file line: {line_idx}',
                                 level=messages.ERROR, underline=True)
                    raise e

                continuation = ''
    else:
        parse_and_run(sys.argv[1:])


def main():
    try:
        _run_loop()
    except KeyboardInterrupt:
        print('Aborting due to keyboard interrupt!')
        sys.exit(1)
