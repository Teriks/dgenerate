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

    import torch
    import diffusers
    import transformers

    from .args import parse_args
    from .textprocessing import quote, unquote
    from .diffusionloop import DiffusionRenderLoop

    from .pipelinewrappers import InvalidVaePathError, \
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

    def parse_args_and_run(with_args):
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

        if sys.stdin.isatty():
            # No templating occurs, this is a terminal
            return None

        # Templating is going to occur

        def jinja_prompt(prompts):
            if not prompts:
                # Completely undefined
                return [{'positive': '', 'negative': ''}]
            else:
                # inside prompt values might be None, don't want that in
                # the jinja2 template because it might be annoying
                # to work with. Also abstract the internal representation
                # of the prompt dictionary to something with friendlier
                # names

                return [{'positive': p.get('prompt', ''),
                         'negative': p.get('negative_prompt', '')} for p in prompts]

        # Return the template environment for pipelining
        return {'last_prompt': jinja_prompt(render_loop.prompts)[-1],
                'last_prompts': jinja_prompt(render_loop.prompts),
                'last_sdxl_second_prompt': jinja_prompt(render_loop.sdxl_second_prompts)[-1],
                'last_sdxl_second_prompts': jinja_prompt(render_loop.sdxl_second_prompts),
                'last_sdxl_refiner_prompt': jinja_prompt(render_loop.sdxl_refiner_prompts)[-1],
                'last_sdxl_refiner_prompts': jinja_prompt(render_loop.sdxl_refiner_prompts),
                'last_sdxl_refiner_second_prompt': jinja_prompt(render_loop.sdxl_refiner_second_prompts)[-1],
                'last_sdxl_refiner_second_prompts': jinja_prompt(render_loop.sdxl_refiner_second_prompts),
                'last_image':
                    quote(render_loop.written_images[-1])
                    if render_loop.written_images else [],
                'last_images':
                    [quote(s) for s in render_loop.written_images],
                'last_animation':
                    quote(render_loop.written_animations[-1])
                    if render_loop.written_animations else [],
                'last_animations':
                    [quote(s) for s in render_loop.written_animations]}

    arguments = sys.argv[1:]
    if not sys.stdin.isatty():
        # Not a terminal, batch process STDIN
        from . batchprocess import process_config, BatchProcessSyntaxException
        try:
            process_config(file_stream=sys.stdin,
                           injected_args=arguments,
                           version_string=__version__,
                           invocation_runner=parse_args_and_run)
        except BatchProcessSyntaxException as e:
            messages.log(f'Config Syntax Error: {e}', level=messages.ERROR)
            sys.exit(1)
    else:
        parse_args_and_run(arguments)


def main():
    try:
        _run_loop()
    except KeyboardInterrupt:
        print('Aborting due to keyboard interrupt!')
        sys.exit(1)
