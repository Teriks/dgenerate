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

import dgenerate.globalconfig

PROMPT_WEIGHTER_GPU_MEMORY_CONSTRAINTS = ['memory_required > (available * 0.70)']
"""
Cache constraint expressions for when to attempt to clear cuda VRAM
upon a prompt weighter plugin calling :py:meth:`dgenerate.promptweighters.PromptWeighter.memory_guard_device`
on a cuda device, syntax provided via :py:func:`dgenerate.memory.gpu_memory_constraints`

If any of these constraints are met, an effort is made to clear modules off a GPU
which are cached for fast repeat usage but are okay to flush.

The only available extra variable is: ``memory_required``, which is the
amount of memory the prompt weighter plugin requested to be available.
"""

PROMPT_WEIGHTER_CACHE_GC_CONSTRAINTS = ['memory_required > (available * 0.70)']
"""
Cache constraint expressions for when to attempt to clear objects out of any CPU side cache
upon a prompt weighter plugin calling :py:meth:`dgenerate.promptweighters.PromptWeighter.memory_guard_device`
on the cpu, syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, an effort is made to clear 
objects out of any named CPU side cache.

The only available extra variable is: ``memory_required``, which is the
amount of memory the prompt weighter plugin requested to be available.
"""

PROMPT_WEIGHTER_CACHE_MEMORY_CONSTRAINTS = ['memory_required > (available * 0.70)']
"""
Cache constraint expressions for when to attempt to clear specifically the prompt weighter 
object cache upon a prompt weighter plugin calling :py:meth:`dgenerate.promptweighters.PromptWeighter.memory_guard_device`
on the cpu, syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, an effort is made to clear objects 
out of the prompt weighter object cache.

Available extra variables are: ``memory_required``, which is the
amount of memory the prompt weighter plugin requested to be available,
and ``cache_size`` which is the current size of the prompt weighter object cache.
"""

dgenerate.globalconfig.register_all()
