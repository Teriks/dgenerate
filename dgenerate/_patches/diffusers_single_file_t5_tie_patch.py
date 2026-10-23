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

import functools

import diffusers.loaders.single_file as _single_file
import diffusers.loaders.single_file_utils as _single_file_utils


# Diffusers builds a single file T5 (SD3 all-in-one checkpoints) on the meta
# device, where Transformers 5 ties encoder.embed_tokens to shared. The
# checkpoint holds only shared.weight, and loading it replaces that
# parameter, which unties the embedding and leaves it a meta tensor. Moving
# the text encoder to a device then fails. Tie it again after loading.
def _retie(create):
    if getattr(create, '_dgenerate_retie', False):
        return create

    @functools.wraps(create)
    def create_diffusers_t5_model_from_checkpoint(*args, **kwargs):
        model = create(*args, **kwargs)
        encoder = getattr(model, 'encoder', None)
        embed_tokens = getattr(encoder, 'embed_tokens', None)
        shared = getattr(model, 'shared', None)
        if (embed_tokens is not None and shared is not None
                and embed_tokens.weight.device.type == 'meta'
                and shared.weight.device.type != 'meta'):
            embed_tokens.weight = shared.weight
        return model

    create_diffusers_t5_model_from_checkpoint._dgenerate_retie = True
    return create_diffusers_t5_model_from_checkpoint


_patched = _retie(_single_file_utils.create_diffusers_t5_model_from_checkpoint)
_single_file_utils.create_diffusers_t5_model_from_checkpoint = _patched
_single_file.create_diffusers_t5_model_from_checkpoint = _patched
