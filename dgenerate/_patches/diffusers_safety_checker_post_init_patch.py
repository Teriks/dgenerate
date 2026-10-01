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

from diffusers.pipelines.deepfloyd_if.safety_checker import IFSafetyChecker
from diffusers.pipelines.deprecated.stable_diffusion_safe.safety_checker import (
    SafeStableDiffusionSafetyChecker,
)

# Transformers 5 sets all_tied_weights_keys in PreTrainedModel.post_init().
# These checkers call PreTrainedModel.__init__ and then never post_init, so
# from_pretrained crashes while moving missing keys off the meta device.
def _call_post_init(cls):
    original = cls.__init__
    if getattr(original, '_dgenerate_calls_post_init', False):
        return

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if not hasattr(self, 'all_tied_weights_keys'):
            self.post_init()

    __init__._dgenerate_calls_post_init = True
    cls.__init__ = __init__


_call_post_init(IFSafetyChecker)
_call_post_init(SafeStableDiffusionSafetyChecker)
