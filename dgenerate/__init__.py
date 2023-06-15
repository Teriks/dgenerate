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

__version__ = "0.3.2"

import sys
import warnings

import diffusers
import transformers

from .args import parse_args
from .diffusionloop import DiffusionRenderLoop
from .mediainput import ImageSeedParseError

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity(transformers.logging.CRITICAL)
diffusers.logging.set_verbosity(diffusers.logging.CRITICAL)


def main():
    arguments = parse_args()

    render_loop = DiffusionRenderLoop()
    render_loop.model_path = arguments.model_path
    render_loop.model_type = arguments.model_type
    render_loop.revision = arguments.revision
    render_loop.device = arguments.device
    render_loop.dtype = arguments.dtype
    render_loop.output_size = arguments.output_size
    render_loop.output_path = arguments.output_path
    render_loop.prompts = arguments.prompts
    render_loop.seeds = arguments.seeds
    render_loop.image_seeds = arguments.image_seeds
    render_loop.animation_format = arguments.animation_format
    render_loop.frame_start = arguments.frame_start
    render_loop.frame_end = arguments.frame_end
    render_loop.image_seed_strengths = arguments.image_seed_strengths
    render_loop.guidance_scales = arguments.guidance_scales
    render_loop.inference_steps = arguments.inference_steps

    # ============================
    # ============================

    # run the render loop
    try:
        render_loop.run()
    except ImageSeedParseError as e:
        print("Error:", e, file=sys.stderr)
        exit(1)
