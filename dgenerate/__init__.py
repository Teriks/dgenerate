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

__version__ = "1.0.0"

import sys


def run_diffusion():
    import re
    import os
    import shlex
    import torch
    import warnings
    import textwrap
    import diffusers
    import transformers

    from .args import parse_args
    from .textprocessing import underline, long_text_wrap_width
    from .diffusionloop import DiffusionRenderLoop

    from .pipelinewrappers import clear_model_cache, InvalidVaePathError, \
        InvalidSchedulerName, InvalidLoRAPathError, \
        InvalidTextualInversionPathError, InvalidSDXLRefinerPathError

    from .mediainput import ImageSeedParseError, MaskImageSizeMismatchError

    # The above modules take long enough to import that they must be in here in
    # order to handle keyboard interrupts without issues

    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    diffusers.logging.set_verbosity(diffusers.logging.CRITICAL)

    def parse_and_run(with_args=None):
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
        render_loop.seeds = arguments.seeds
        render_loop.image_seeds = arguments.image_seeds
        render_loop.animation_format = arguments.animation_format
        render_loop.frame_start = arguments.frame_start
        render_loop.frame_end = arguments.frame_end
        render_loop.image_seed_strengths = arguments.image_seed_strengths
        render_loop.guidance_scales = arguments.guidance_scales
        render_loop.inference_steps = arguments.inference_steps
        render_loop.vae_path = arguments.vae
        render_loop.lora_paths = arguments.lora
        render_loop.textual_inversion_paths = arguments.textual_inversions
        render_loop.scheduler = arguments.scheduler
        render_loop.safety_checker = arguments.safety_checker
        render_loop.sdxl_refiner_path = arguments.sdxl_refiner
        render_loop.sdxl_high_noise_fractions = arguments.sdxl_high_noise_fractions
        render_loop.sdxl_original_size = arguments.sdxl_original_size
        render_loop.sdxl_target_size = arguments.sdxl_target_size
        render_loop.auth_token = arguments.auth_token

        # run the render loop
        try:
            render_loop.run()
        except (ImageSeedParseError,
                MaskImageSizeMismatchError,
                InvalidSDXLRefinerPathError,
                InvalidVaePathError,
                InvalidLoRAPathError,
                InvalidTextualInversionPathError,
                InvalidSchedulerName,
                torch.cuda.OutOfMemoryError,
                NotImplementedError,
                EnvironmentError) as e:
            print("Error:", e, file=sys.stderr)
            sys.exit(1)

    continuation = ''
    first_line = True
    if not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line == '':
                first_line = False
                continue
            if line.startswith('#'):
                if first_line:
                    versioning = re.match(r'#!\s+dgenerate\s+([0-9]+\.[0-9]+\.[0-9]+)', line)
                    if versioning:
                        config_file_version = versioning.group(1)
                        cur_major_version = int(__version__.split('.')[0])
                        config_major_version = int(config_file_version.split('.')[0])
                        if cur_major_version != config_major_version:
                            print(underline(
                                'WARNING: Ingested configuration file is written for an '
                                f'incompatible version of dgenerate! You are using version {__version__} '
                                f'and the config file was written for version {config_file_version}',
                                ), file=sys. stderr)

                first_line = False
                continue
            if line.startswith('\\clear_model_cache'):
                clear_model_cache()
                first_line = False
                continue

            if line.endswith('\\'):
                continuation += ' ' + line.rstrip(' \\')
            else:
                args = (continuation + ' ' + line).lstrip()

                header = "Processing Arguments: "
                args_wrapped = textwrap.fill(args,
                                             width=long_text_wrap_width() - len(header),
                                             subsequent_indent=' ' * len(header))

                print(underline(header + args_wrapped))
                parse_and_run(shlex.split(os.path.expandvars(args)))
                continuation = ''

            first_line = False
    else:
        parse_and_run()


def main():
    try:
        run_diffusion()
    except KeyboardInterrupt:
        print("Aborting due to keyboard interrupt!")
        sys.exit(1)
