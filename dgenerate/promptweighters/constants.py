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


PROMPT_WEIGHTER_CUDA_MEMORY_CONSTRAINTS = ['memory_required > (available * 0.70)']
"""
Cache constraint expressions for when to attempt to fully clear cuda VRAM 
upon a prompt weighter plugin requesting a device memory fence, syntax 
provided via :py:func:`dgenerate.memory.cuda_memory_constraints`

If any of these constraints are met, an effort is made to clear modules off a GPU 
which are cached for fast repeat usage but are okay to flush.

The only available extra variable is: ``memory_required``, which is the
amount of memory the prompt weighter plugin requested to fence the device
for.
"""

PROMPT_WEIGHTER_CACHE_GC_CONSTRAINTS = ['weighter_size > (available * 0.70)']
"""
Cache constraint expressions for when to attempt to fully clear CPU side ram before 
the initial loading of a prompt weighter module into ram, syntax provided via
:py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, an effort is made to clear objects out of 
cpu side ram which are cached for fast repeat usage but are okay to flush,
prior to loading a prompt weighter model.

The only available extra variable is: ``weighter_size`` (the estimated size 
of the prompt weighter module that needs to enter ram, in bytes)
"""

PROMPT_WEIGHTER_CACHE_MEMORY_CONSTRAINTS = ['weighter_size > (available * 0.70)']
"""
Cache constraint expressions for when to attempt to clear the prompt weighter
cache before bringing a new prompt weighter online, this cache caches prompt 
weighter objects for reuse. :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, all prompt weighter
objects are cleared from the CPU cache.

The only available extra variable is: ``weighter_size`` (the estimated size 
of the prompt weighter module that needs to enter ram, in bytes)
"""
