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

import inspect

from transformers.models.clip.modeling_clip import CLIPTextModel

# Transformers 5 flattened CLIPTextModel: embeddings, encoder, and
# final_layer_norm live on the model, and the nested text_model module is gone.
# Diffusers clip-skip still calls text_encoder.text_model.final_layer_norm.
# The alias is the model itself, so that lookup hits the real norm.
#
# Single-file loading keeps "text_model." on checkpoint keys when hasattr
# text_model is true, then copies a weight only if that name is in
# state_dict(). The flattened state_dict has no such prefix, so the weights
# stay meta. Hide the alias for that one call so Diffusers strips the prefix.
_init_source = inspect.getsource(CLIPTextModel.__init__)
if 'self.final_layer_norm' in _init_source and 'self.text_model' not in _init_source:
    def text_model(self):
        return self

    CLIPTextModel.text_model = property(text_model)

    def _call_without_flattened_text_model_alias(func, /, *args, **kwargs):
        prop = CLIPTextModel.__dict__.get('text_model')
        if not isinstance(prop, property):
            return func(*args, **kwargs)
        del CLIPTextModel.text_model
        try:
            return func(*args, **kwargs)
        finally:
            CLIPTextModel.text_model = prop

    def _patch_single_file_clip_loader(module):
        original = getattr(module, 'create_diffusers_clip_model_from_ldm', None)
        if original is None or getattr(original, '_dgenerate_hides_clip_text_model_alias', False):
            return

        def create_diffusers_clip_model_from_ldm(*args, **kwargs):
            return _call_without_flattened_text_model_alias(original, *args, **kwargs)

        create_diffusers_clip_model_from_ldm._dgenerate_hides_clip_text_model_alias = True
        module.create_diffusers_clip_model_from_ldm = create_diffusers_clip_model_from_ldm

    import diffusers.loaders.single_file as _single_file
    import diffusers.loaders.single_file_utils as _single_file_utils

    _patch_single_file_clip_loader(_single_file_utils)
    _patch_single_file_clip_loader(_single_file)
