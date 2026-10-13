# Copyright (c) 2026, Teriks
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

def patch_controlnet_text_projection(cn_model):
    """
    Apply Kolors' ChatGLM text projection before ``ControlNetModel.forward``.

    diffusers does not run ``encoder_hid_proj`` inside ControlNet, and Kolors
    ControlNets expect 4096-wide ChatGLM states to be projected to 2048.
    Each call binds that projection, and the original forward, to the model
    it was given. A wrapper defined in a loop over several ControlNets would
    otherwise call only the last model's forward.

    :param cn_model: ControlNet module to wrap
    :return: the forward method that was replaced
    """
    original_forward = cn_model.forward

    def _forward(*args, **kwargs):
        encoder_hidden_states = kwargs["encoder_hidden_states"]
        if cn_model.encoder_hid_proj is not None and cn_model.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = encoder_hidden_states.to(cn_model.encoder_hid_proj.weight.device)
            encoder_hidden_states = cn_model.encoder_hid_proj(encoder_hidden_states)
        kwargs.pop("encoder_hidden_states")
        return original_forward(*args, encoder_hidden_states=encoder_hidden_states, **kwargs)

    cn_model.forward = _forward
    return original_forward
