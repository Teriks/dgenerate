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

IMAGE_PROCESSOR_CUDA_MEMORY_CONSTRAINTS = ['processor_size > (available * 0.70)']
"""
Cache constraint expressions for when to attempt to fully clear cuda VRAM before 
moving an image processor on to a cuda device, syntax provided via
:py:func:`dgenerate.memory.cuda_memory_constraints`

If any of these constraints are met, an effort is made to clear modules off the GPU 
which are cached for fast repeat usage but are okay to flush, prior to moving
an image processor to the GPU.

The only available extra variable is: ``pipeline_size`` (the estimated size 
of the image processor module that needs to enter VRAM, in bytes)
"""

IMAGE_PROCESSOR_MEMORY_CONSTRAINTS = ['processor_size > (available * 0.70)']
"""
Cache constraint expressions for when to attempt to fully clear CPU side ram before 
the initial loading of an image processor module into ram, syntax provided via
:py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, an effort is made to clear modules out of 
cpu side ram which are cached for fast repeat usage but are okay to flush,
prior to loading an image processor model.

The only available extra variable is: ``pipeline_size`` (the estimated size 
of the image processor module that needs to enter ram, in bytes)
"""
