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

"""
Diffusers single file loader fixes shared by every single file transformer
load: dense, GGUF, and ComfyUI quantized ``.safetensors`` alike.

Registers transformers Diffusers can build but not load from a single file,
and replaces checkpoint conversions that miss tensors of newer releases.
"""

import functools

COMFY_PREFIX = 'model.diffusion_model.'


def strip_comfy_prefix(checkpoint: dict) -> dict:
    """Drop the Comfy ``model.diffusion_model.`` key prefix, if present."""
    if not any(key.startswith(COMFY_PREFIX) for key in checkpoint):
        return checkpoint
    renamed = {}
    for key, value in checkpoint.items():
        new_key = key[len(COMFY_PREFIX):] if key.startswith(COMFY_PREFIX) else key
        renamed[new_key] = value
    return renamed


def rename_ltx25_prompt_adaln(checkpoint: dict) -> dict:
    """
    LTX 2.5 checkpoints keep the original ``*_adaln_single`` prefix. Diffusers
    drops ``_single`` for the prompt embeddings and maps the main ones itself,
    but its converter (as of 0.41) does not know the prompt ones.
    """
    renamed = {}
    for key, value in checkpoint.items():
        new_key = key.replace('audio_prompt_adaln_single.', 'audio_prompt_adaln.')
        new_key = new_key.replace('prompt_adaln_single.', 'prompt_adaln.')
        renamed[new_key] = value
    return renamed


def convert_ltx2_transformer_checkpoint(checkpoint: dict, **kwargs) -> dict:
    """Diffusers' LTX 2 conversion, plus the LTX 2.5 prompt modulation names."""
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers
    return convert_ltx2_transformer_to_diffusers(rename_ltx25_prompt_adaln(checkpoint), **kwargs)


def convert_hunyuan_video15_transformer_checkpoint(checkpoint: dict, **kwargs) -> dict:
    """Convert original Hunyuan Video weights to the Diffusers 1.5 names."""
    from diffusers.loaders.single_file_utils import convert_hunyuan_video_transformer_to_diffusers

    converted = convert_hunyuan_video_transformer_to_diffusers(
        strip_comfy_prefix(checkpoint), **kwargs)
    renamed = {}
    vision_projection = {
        'vision_in.proj.0.': 'image_embedder.norm_in.',
        'vision_in.proj.1.': 'image_embedder.linear_1.',
        'vision_in.proj.3.': 'image_embedder.linear_2.',
        'vision_in.proj.4.': 'image_embedder.norm_out.',
    }
    byt5_projection = {
        'byt5_in.net.0.proj.': 'context_embedder_2.linear_1.',
        'byt5_in.net.2.': 'context_embedder_2.linear_2.',
        'byt5_in.fc3.': 'context_embedder_2.linear_3.',
        'byt5_in.layernorm.': 'context_embedder_2.norm.',
    }
    for name, value in converted.items():
        for source, target in byt5_projection.items():
            if name.startswith(source):
                name = target + name[len(source):]
                break
        if name.startswith('time_text_embed.'):
            name = 'time_embed.' + name[len('time_text_embed.'):]
        elif name.startswith('cond_type_embedding.'):
            name = 'cond_type_embed.' + name[len('cond_type_embedding.'):]
        else:
            for source, target in vision_projection.items():
                if name.startswith(source):
                    name = target + name[len(source):]
                    break
        renamed[name] = value
    return renamed


@functools.cache
def register_single_file_classes() -> None:
    """Install dgenerate's entries in Diffusers' single file loadable class table."""
    import diffusers
    import diffusers.loaders.single_file_model as single_file_model

    loadable = single_file_model.SINGLE_FILE_LOADABLE_CLASSES

    # Diffusers provides the Hunyuan 1.5 architecture but (as of 0.41) does
    # not expose it through the single-file loader. Its source checkpoint
    # converter is reusable once the 1.5-only component names are remapped.
    if hasattr(diffusers, 'HunyuanVideo15Transformer3DModel'):
        loadable.setdefault(
            'HunyuanVideo15Transformer3DModel', {
                'checkpoint_mapping_fn': convert_hunyuan_video15_transformer_checkpoint,
                'default_subfolder': 'transformer',
            })

    # Kandinsky 5 is likewise not registered. Its checkpoints use the Diffusers
    # tensor names, so they load unconverted.
    if hasattr(diffusers, 'Kandinsky5Transformer3DModel'):
        loadable.setdefault(
            'Kandinsky5Transformer3DModel', {
                'checkpoint_mapping_fn': lambda checkpoint, **kwargs: checkpoint,
                'default_subfolder': 'transformer',
            })

    # LTX 2.5 adds prompt modulation tensors Diffusers' LTX 2 conversion
    # leaves unrenamed, so they would stay on the meta device.
    ltx2 = loadable.get('LTX2VideoTransformer3DModel')
    if ltx2 is not None:
        ltx2['checkpoint_mapping_fn'] = convert_ltx2_transformer_checkpoint
