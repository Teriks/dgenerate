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

DEFAULT_INFERENCE_STEPS = 30
"""
Default value for inference steps.
"""

DEFAULT_GUIDANCE_SCALE = 5
"""
Default value for guidance scale.
"""

DEFAULT_IMAGE_SEED_STRENGTH = 0.8
"""
Default image seed strength for img2img.
"""

DEFAULT_IMAGE_GUIDANCE_SCALE = 1.5
"""
Default image guidance scale for pix2pix.
"""

DEFAULT_SDXL_HIGH_NOISE_FRACTION = 0.8
"""
Default SDXL high noise fraction.
"""

DEFAULT_X4_UPSCALER_NOISE_LEVEL = 20
"""
Default x4 upscaler noise level.
"""

DEFAULT_OUTPUT_WIDTH = 512
"""
Default output width for txt2img.
"""

DEFAULT_OUTPUT_HEIGHT = 512
"""
Default output height for txt2img.
"""

DEFAULT_SDXL_OUTPUT_WIDTH = 1024
"""
Default output width for SDXL txt2img.
"""

DEFAULT_SDXL_OUTPUT_HEIGHT = 1024
"""
Default output height for SDXL txt2img.
"""

DEFAULT_FLOYD_IF_OUTPUT_WIDTH = 64
"""
Default output width for Deep Floyd IF txt2img first stage.
"""

DEFAULT_FLOYD_IF_OUTPUT_HEIGHT = 64
"""
Default output height for Deep Floyd IF txt2img first stage.
"""

DEFAULT_SEED = 0
"""
Default RNG seed.
"""
