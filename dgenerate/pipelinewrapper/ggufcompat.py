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
Detect Flux.2, Qwen-Image, Z-Image, LTX-2, LTX 2.5, Kandinsky 5, HunyuanImage 2.1, and Wan GGUF
checkpoints and adapt ComfyUI layouts so Diffusers can load them without a manual config.

Text-encoder GGUF files (T5, UMT5, Qwen, Mistral, Gemma) are read here too. Their
llama.cpp tensor names are renamed to Hugging Face names, and the packed weights
stay packed for ``--text-encoders``.

Packed linear matmuls on CUDA, ROCm, and XPU unpack in Triton and use a PyTorch
GEMM. Triton is imported only when that matmul runs.
"""

import contextlib
import math
import os

import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat

_COMFY_PREFIX = _singlefilecompat.COMFY_PREFIX
_strip_comfy_prefix = _singlefilecompat.strip_comfy_prefix

# img_in / qkv hidden width -> public Diffusers transformer repo.
# 4B and 9B Klein share Flux.2 key names with dev, so the width is the
# only thing that stops Diffusers from building the 32B dev module.
_FLUX2_BY_HIDDEN = {
    3072: 'black-forest-labs/FLUX.2-klein-4B',
    4096: 'black-forest-labs/FLUX.2-klein-base-9B',
    6144: 'black-forest-labs/FLUX.2-dev',
}

_KANDINSKY5_BY_HIDDEN = {
    1792: 'kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers',
    4096: 'kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers',
}

_KANDINSKY6_BY_HIDDEN = {
    # Hidden size does not separate a distilled checkpoint from the full one.
    # The published GGUFs are the 50-step Lite and Pro transformers.
    1792: 'kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers',
    4096: 'kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers',
}

_FLUX3_DISTILLED = 'black-forest-labs/flux-3-image-klein'
_FLUX3_BASE = 'black-forest-labs/flux-3-image-klein-base'

_patches_installed = False


class GGUFLayout:
    """Where a detected GGUF transformer should be loaded from."""

    def __init__(self, kind: str, config_repo: str, subfolder: str = 'transformer',
                 comfy: bool = False, adapt: bool = False):
        self.kind = kind
        self.config_repo = config_repo
        self.subfolder = subfolder
        self.comfy = comfy
        self.adapt = adapt


def _shape(value) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def _has(shapes: dict, *names: str) -> bool:
    return any(name in shapes for name in names)


def _lookup_shape(shapes: dict, *names: str):
    for name in names:
        if name in shapes:
            return _shape(shapes[name])
    return None


def _flux2_hidden(shapes: dict) -> int | None:
    img_in = _lookup_shape(
        shapes, 'img_in.weight', _COMFY_PREFIX + 'img_in.weight')
    if img_in is not None and len(img_in) == 2:
        if img_in[0] == 128:
            return img_in[1]
        if img_in[1] == 128:
            return img_in[0]
    qkv = _lookup_shape(
        shapes,
        'double_blocks.0.img_attn.qkv.weight',
        _COMFY_PREFIX + 'double_blocks.0.img_attn.qkv.weight')
    if qkv is not None:
        return qkv[0]
    return None


def detect_gguf_layout(shapes: dict[str, tuple[int, ...]]) -> GGUFLayout | None:
    """
    Identify a GGUF transformer from tensor names and logical shapes.

    ``shapes`` maps a tensor name to its logical shape. Comfy files often
    prefix names with ``model.diffusion_model.``.
    """
    comfy = any(name.startswith(_COMFY_PREFIX) for name in shapes)

    # LTX-2 and LTX-2.5 both have audio_patchify_proj. The keyframe embedding
    # and prompt AdaLN are 2.5-only; without them the file is the 19B LTX-2
    # transformer and needs Lightricks/LTX-2 (Gemma 3, 2-layer connectors).
    if _has(shapes, 'keyframes_abs_pos_embedding',
            'prompt_adaln_single.linear.weight',
            'prompt_adaln_single.emb.timestep_embedder.linear_1.weight',
            _COMFY_PREFIX + 'keyframes_abs_pos_embedding',
            _COMFY_PREFIX + 'prompt_adaln_single.linear.weight',
            _COMFY_PREFIX + 'prompt_adaln_single.emb.timestep_embedder.linear_1.weight'):
        return GGUFLayout(
            'ltx-2.5', 'Lightricks/LTX-2.5-Diffusers', comfy=comfy, adapt=True)
    if _has(shapes, 'audio_patchify_proj.weight',
            'av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.weight',
            _COMFY_PREFIX + 'audio_patchify_proj.weight',
            _COMFY_PREFIX + 'av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.weight'):
        return GGUFLayout(
            'ltx-2', 'Lightricks/LTX-2', comfy=comfy, adapt=True)

    # Motif single-stream blocks carry a text cross-attention Flux does not.
    if _has(shapes, 'single_transformer_blocks.0.cross_attn.to_q.weight',
            _COMFY_PREFIX + 'single_transformer_blocks.0.cross_attn.to_q.weight'):
        return GGUFLayout(
            'motif-video', 'Motif-Technologies/Motif-Video-2B',
            comfy=comfy, adapt=False)

    # HunyuanVideo 1.5 shares HunyuanImage embedder names. The condition-type
    # embedding is 1.5-only, so it has to win before the HunyuanImage check.
    if _has(shapes, 'cond_type_embedding.weight', 'cond_type_embed.weight',
            _COMFY_PREFIX + 'cond_type_embedding.weight',
            _COMFY_PREFIX + 'cond_type_embed.weight'):
        return GGUFLayout(
            'hyvideo-15',
            'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v',
            comfy=comfy, adapt=False)

    if _has(shapes, 'cap_embedder.0.weight', 'layers.0.adaLN_modulation.0.weight',
            _COMFY_PREFIX + 'cap_embedder.0.weight',
            _COMFY_PREFIX + 'layers.0.adaLN_modulation.0.weight'):
        return GGUFLayout(
            'z-image-turbo', 'Tongyi-MAI/Z-Image-Turbo', comfy=comfy, adapt=False)

    if _has(shapes, 'txtfusion.projector.weight',
            _COMFY_PREFIX + 'txtfusion.projector.weight',
            'first.weight', _COMFY_PREFIX + 'first.weight'):
        return GGUFLayout(
            'krea2', 'krea/Krea-2-Raw', comfy=comfy, adapt=True)

    # Flux also has double_blocks img_attn qkv. Require a HunyuanImage-only embedder.
    if _has(shapes, 'txt_in.input_embedder.weight', 'img_in.proj.weight',
            _COMFY_PREFIX + 'txt_in.input_embedder.weight',
            _COMFY_PREFIX + 'img_in.proj.weight',
            'model.model.txt_in.input_embedder.weight',
            'model.model.img_in.proj.weight'):
        import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat
        config_repo = _singlefilecompat.hunyuanimage_gguf_config_repo(shapes)
        return GGUFLayout('hyimage21', config_repo, comfy=comfy, adapt=True)

    flux3 = _detect_flux3_layout(shapes, comfy)
    if flux3 is not None:
        return flux3

    if _has(shapes, 'single_stream_modulation.lin.weight',
            _COMFY_PREFIX + 'single_stream_modulation.lin.weight'):
        hidden = _flux2_hidden(shapes)
        repo = _FLUX2_BY_HIDDEN.get(hidden)
        if repo is None:
            return None
        if hidden == 3072:
            kind = 'flux2-klein-4b'
        elif hidden == 4096:
            kind = 'flux2-klein-9b'
        else:
            kind = 'flux2-dev'
        return GGUFLayout(kind, repo, comfy=comfy, adapt=False)

    if _has(shapes, 'txt_in.text_norm.weight',
            _COMFY_PREFIX + 'txt_in.text_norm.weight'):
        return GGUFLayout(
            'qwen-image-21', 'Qwen/Qwen-Image-2.1', comfy=comfy, adapt=False)

    if _has(shapes, 'time_text_embed.timestep_embedder.linear_1.weight',
            _COMFY_PREFIX + 'time_text_embed.timestep_embedder.linear_1.weight'):
        # Layered adds a learned embedding the base and edit transformers lack.
        # Edit shares the base module, so it keeps the Qwen-Image config.
        layered = _has(
            shapes,
            'time_text_embed.addition_t_embedding.weight',
            _COMFY_PREFIX + 'time_text_embed.addition_t_embedding.weight')
        if layered:
            # Always adapt: Diffusers only dequantizes GGUFLinear, so the
            # addition_t Embedding keeps packed BF16/quant bytes and returns
            # width 6144 instead of inner_dim 3072 at runtime.
            return GGUFLayout(
                'qwen-image-layered', 'Qwen/Qwen-Image-Layered',
                comfy=comfy, adapt=True)
        return GGUFLayout(
            'qwen-image', 'Qwen/Qwen-Image', comfy=comfy, adapt=comfy)

    kandinsky6 = _detect_kandinsky6_layout(shapes, comfy)
    if kandinsky6 is not None:
        return kandinsky6

    minimax_image = _detect_minimax_h3_image_layout(shapes, comfy)
    if minimax_image is not None:
        return minimax_image
    minimax = _detect_minimax_h3_layout(shapes, comfy)
    if minimax is not None:
        return minimax

    if _has(shapes, 'visual_transformer_blocks.0.visual_modulation.out_layer.weight',
            _COMFY_PREFIX + 'visual_transformer_blocks.0.visual_modulation.out_layer.weight'):
        # Always adapt: files store some unquantized tensors with extra
        # singleton dimensions, and some store quantized weights flattened.
        bias = _lookup_shape(
            shapes, 'visual_transformer_blocks.0.self_attention.to_query.bias',
            _COMFY_PREFIX + 'visual_transformer_blocks.0.self_attention.to_query.bias')
        hidden = math.prod(bias) if bias is not None else None
        return GGUFLayout(
            'kandinsky5',
            _KANDINSKY5_BY_HIDDEN.get(hidden, _KANDINSKY5_BY_HIDDEN[1792]),
            comfy=comfy, adapt=True)

    wan = _detect_wan_layout(shapes, comfy)
    if wan is not None:
        return wan

    return None


def _detect_flux3_layout(shapes: dict, comfy: bool) -> GGUFLayout | None:
    """FLUX.3 Klein, in Comfy names (``emb_in``) or Diffusers names (``x_embedders``)."""
    comfy_layout = _has(
        shapes,
        'emb_in.image.weight',
        'content_mode_blocks.image.0.q_proj.weight',
        'content_mode_blocks.image.0.attn.q_proj.weight',
        _COMFY_PREFIX + 'emb_in.image.weight',
        _COMFY_PREFIX + 'content_mode_blocks.image.0.q_proj.weight')
    diffusers_layout = _has(
        shapes,
        'x_embedders.image.weight',
        'mode_blocks.image.0.attn.to_q.weight',
        _COMFY_PREFIX + 'x_embedders.image.weight',
        _COMFY_PREFIX + 'mode_blocks.image.0.attn.to_q.weight')
    if not comfy_layout and not diffusers_layout:
        return None
    distilled = _has(
        shapes,
        'ref_time_in.in_layer.weight',
        'target_timestep_embedder.linear_1.weight',
        _COMFY_PREFIX + 'ref_time_in.in_layer.weight',
        _COMFY_PREFIX + 'target_timestep_embedder.linear_1.weight')
    return GGUFLayout(
        'flux3',
        _FLUX3_DISTILLED if distilled else _FLUX3_BASE,
        comfy=comfy,
        adapt=comfy or comfy_layout)


def _kandinsky6_hidden(shapes: dict) -> int | None:
    """
    Hidden width of a Kandinsky 6 transformer.

    ``visual_embeddings.in_layer.weight`` is the natural place to read it, but
    a Q4_K file stores that matrix as packed bytes, so neither dimension is
    1792 or 4096. The bias and the video attention projection stay at the real
    width.
    """
    names = (
        'visual_embeddings.in_layer.bias',
        'video_text_embeddings.in_layer.bias',
        'video_text_transformer_blocks.0.attn.to_q.bias',
        'video_text_transformer_blocks.0.attn.to_query.bias',
        'video_text_transformer_blocks.0.attn.to_q.weight',
        'video_text_transformer_blocks.0.attn.to_query.weight',
        'visual_token_type_embeddings.weight',
        'visual_embeddings.in_layer.weight',
    )
    for name in names:
        weight = _lookup_shape(shapes, name, _COMFY_PREFIX + name)
        if weight is None:
            continue
        for dim in weight:
            if dim in _KANDINSKY6_BY_HIDDEN:
                return dim
    return None


def _detect_kandinsky6_layout(shapes: dict, comfy: bool) -> GGUFLayout | None:
    """Kandinsky 6 video+audio. The audio embedder is absent from Kandinsky 5."""
    if not _has(
            shapes,
            'audio_embeddings.in_layer.weight',
            'video_text_transformer_blocks.0.attn.to_q.weight',
            'audio_out_layer.proj.weight',
            _COMFY_PREFIX + 'audio_embeddings.in_layer.weight',
            _COMFY_PREFIX + 'video_text_transformer_blocks.0.attn.to_q.weight',
            _COMFY_PREFIX + 'audio_out_layer.proj.weight'):
        return None
    hidden = _kandinsky6_hidden(shapes)
    # The published Pro GGUF stores four modulation weights transposed, with a
    # trailing size-1 dimension. Align those before Diffusers builds the module.
    return GGUFLayout(
        'kandinsky6',
        _KANDINSKY6_BY_HIDDEN.get(hidden, _KANDINSKY6_BY_HIDDEN[1792]),
        comfy=comfy,
        adapt=True)


def _orient_wan_patch(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return a Wan patch embedding in PyTorch order, ``(out, in, k...)``.

    city96 GGUFs store the conv as ``(k..., in, out)``, so a 14B text-to-video
    weight is ``(2, 2, 1, 16, 5120)`` rather than ``(5120, 16, 1, 2, 2)``.
    Diffusers reverses it while loading. Detection has to use the same order,
    or the leading 2 is read as the hidden width and the file is treated as
    image-to-video.
    """
    shape = _shape(shape)
    if len(shape) < 2:
        return shape
    widths = (1536, 3072, 5120)
    if shape[0] in widths:
        return shape
    if shape[-1] in widths:
        return tuple(reversed(shape))
    return shape


def _wan_patch_embedding(shapes: dict):
    patch = _lookup_shape(
        shapes, 'patch_embedding.weight',
        _COMFY_PREFIX + 'patch_embedding.weight')
    if patch is None:
        return None
    return _orient_wan_patch(patch)


def _detect_minimax_h3_layout(shapes: dict, comfy: bool) -> GGUFLayout | None:
    """MiniMax-H3. Comfy names use ``video_patch_proj`` and fused ``qkv_proj``."""
    diffusers_layout = _has(
        shapes,
        'audio_proj_in.weight',
        'audio_proj_out.weight',
        _COMFY_PREFIX + 'audio_proj_in.weight',
        _COMFY_PREFIX + 'audio_proj_out.weight')
    native_layout = _has(
        shapes,
        'video_patch_proj.weight',
        'audio_patch_proj.weight',
        'blocks.0.attn.qkv_proj.weight',
        _COMFY_PREFIX + 'video_patch_proj.weight',
        _COMFY_PREFIX + 'audio_patch_proj.weight',
        _COMFY_PREFIX + 'blocks.0.attn.qkv_proj.weight')
    # Image files share video_patch_proj and qkv_proj. Audio is what makes
    # this the video transformer.
    if not _has(
            shapes,
            'audio_patch_proj.weight',
            'audio_proj_in.weight',
            'audio_proj_out.weight',
            _COMFY_PREFIX + 'audio_patch_proj.weight',
            _COMFY_PREFIX + 'audio_proj_in.weight',
            _COMFY_PREFIX + 'audio_proj_out.weight'):
        return None
    if not diffusers_layout and not native_layout:
        return None
    return GGUFLayout(
        'minimax-h3',
        'MiniMaxAI/MiniMax-H3',
        comfy=comfy or native_layout,
        adapt=True)


def _detect_minimax_h3_image_layout(shapes: dict, comfy: bool) -> GGUFLayout | None:
    """MiniMax-H3-Image. The Comfy file is the video DiT without audio, plus one fused output."""
    if _has(shapes,
            'audio_patch_proj.weight',
            'audio_proj_in.weight',
            _COMFY_PREFIX + 'audio_patch_proj.weight',
            _COMFY_PREFIX + 'audio_proj_in.weight'):
        return None
    native = (
        _has(shapes, 'video_patch_proj.weight', _COMFY_PREFIX + 'video_patch_proj.weight')
        and _has(shapes, 'blocks.0.attn.qkv_proj.weight',
                 _COMFY_PREFIX + 'blocks.0.attn.qkv_proj.weight'))
    diffusers_layout = _has(shapes, 'pdd_proj_out.0.weight')
    if not native and not diffusers_layout:
        return None
    return GGUFLayout(
        'minimax-h3-image',
        'MiniMaxAI/MiniMax-H3-Image',
        comfy=comfy or native,
        adapt=True)


def _detect_wan_layout(shapes: dict, comfy: bool) -> GGUFLayout | None:
    if not _has(shapes, 'head.modulation', 'patch_embedding.weight',
                _COMFY_PREFIX + 'head.modulation',
                _COMFY_PREFIX + 'patch_embedding.weight'):
        return None
    if _has(shapes, 'vace_blocks.0.after_proj.bias',
            _COMFY_PREFIX + 'vace_blocks.0.after_proj.bias'):
        patch = _wan_patch_embedding(shapes)
        if patch is not None and patch[0] == 1536:
            return GGUFLayout(
                'wan-vace-1.3B', 'Wan-AI/Wan2.1-VACE-1.3B-diffusers',
                comfy=comfy, adapt=False)
        return GGUFLayout(
            'wan-vace-14B', 'Wan-AI/Wan2.1-VACE-14B-diffusers',
            comfy=comfy, adapt=False)
    if _has(shapes, 'motion_encoder.dec.direction.weight',
            _COMFY_PREFIX + 'motion_encoder.dec.direction.weight'):
        return GGUFLayout(
            'wan-animate-14B', 'Wan-AI/Wan2.2-Animate-14B-Diffusers',
            comfy=comfy, adapt=True)
    # Official / Comfy Animate-2 wraps each block as blocks.N.block.X.
    # Plain Wan transformers use blocks.N.self_attn / blocks.N.cross_attn
    # without that extra ``.block.`` level — do not key off self_attn alone.
    if _has_wan_animate_2_nested_block(shapes):
        return GGUFLayout(
            'wan-animate-2-14B', 'Wan-AI/Wan2.2-Animate-2-14B-Diffusers',
            comfy=comfy, adapt=False)
    # FLF2V adds img_emb.emb_pos. Diffusers names it pos_embed and leaves the
    # original key unloaded, so the parameter stays a meta tensor.
    if _has(shapes, 'img_emb.emb_pos',
            _COMFY_PREFIX + 'img_emb.emb_pos',
            'condition_embedder.image_embedder.pos_embed',
            _COMFY_PREFIX + 'condition_embedder.image_embedder.pos_embed'):
        return GGUFLayout(
            'wan-flf2v-14B', 'Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers',
            comfy=comfy, adapt=True)
    patch = _wan_patch_embedding(shapes)
    if patch is None:
        return None
    width = patch[0]
    channels = patch[1] if len(patch) > 1 else None
    if width == 1536:
        return GGUFLayout(
            'wan-t2v-1.3B', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
            comfy=comfy, adapt=False)
    if width == 3072:
        return GGUFLayout(
            'wan-ti2v-5B', 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
            comfy=comfy, adapt=False)
    if width == 5120 and channels == 36:
        # Wan 2.1 I2V keeps a CLIP image embedder (``img_emb``). Wan 2.2 I2V
        # A14B uses the same 5120 x 36 patch and drops that embedder. FLF2V
        # also has ``img_emb`` and was returned above via ``emb_pos``.
        if _has_wan_image_embedder(shapes):
            return GGUFLayout(
                'wan-i2v-14B', 'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers',
                comfy=comfy, adapt=False)
        return GGUFLayout(
            'wan-i2v-a14b', 'Wan-AI/Wan2.2-I2V-A14B-Diffusers',
            comfy=comfy, adapt=False)
    if width == 5120:
        return GGUFLayout(
            'wan-t2v-14B', 'Wan-AI/Wan2.1-T2V-14B-Diffusers',
            comfy=comfy, adapt=False)
    return None


def _has_wan_image_embedder(shapes: dict) -> bool:
    """True when a Wan file still has the Wan 2.1 CLIP image embedder."""
    for name in shapes:
        key = name[len(_COMFY_PREFIX):] if name.startswith(_COMFY_PREFIX) else name
        if key.startswith('diffusion_model.'):
            key = key[len('diffusion_model.'):]
        if 'img_emb.' in key or 'image_embedder.' in key:
            return True
    return False


def _has_wan_animate_2_nested_block(shapes: dict) -> bool:
    """True when keys use the Animate-2 ``blocks.N.block.*`` nesting."""
    for name in shapes:
        key = name[len(_COMFY_PREFIX):] if name.startswith(_COMFY_PREFIX) else name
        if key.startswith('blocks.') and '.block.' in key:
            return True
    return False


def read_gguf_shapes(path: str) -> dict[str, tuple[int, ...]]:
    """Logical tensor shapes from a GGUF header. Does not copy weights."""
    import gguf
    reader = gguf.GGUFReader(path)
    return {tensor.name: _shape(tensor.shape) for tensor in reader.tensors}


def _align_tensor(param, target_shape: tuple[int, ...]):
    """
    Make one checkpoint tensor match a Diffusers parameter.

    Comfy LTX 2.5 stores unquantized tables transposed, and
    ``keyframes_abs_pos_embedding`` without the leading 1. Quantized
    linears stay packed; a flag tells the GGUF linear to transpose
    after dequantization so the file can stay quantized.
    """
    target_shape = tuple(int(item) for item in target_shape)
    quant_shape = getattr(param, 'quant_shape', None)
    if quant_shape is not None:
        quant_shape = _shape(quant_shape)
        if quant_shape == target_shape:
            return param
        if len(quant_shape) == 1 and len(target_shape) == 2                 and quant_shape[0] == math.prod(target_shape):
            return _unflatten_quantized(param, target_shape)
        if (len(quant_shape) == 2 and len(target_shape) == 2
                and quant_shape == tuple(reversed(target_shape))):
            param._dgenerate_comfy_transpose = True
            return param
        # A row whose length is not a quant block is stored as
        # (numel / block, block). The values are still the original row-major
        # weight, so dequantize and restore the module shape.
        if math.prod(quant_shape) == math.prod(target_shape):
            from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
            return dequantize_gguf_tensor(param).reshape(target_shape).contiguous()
        return param

    current = _shape(param.shape)
    if current == target_shape:
        return param
    if param.numel() != math.prod(target_shape):
        return param
    squeezed = tuple(dim for dim in current if dim != 1)
    if squeezed and squeezed != current:
        param = param.reshape(squeezed)
        current = squeezed
        if current == target_shape:
            return param
    if tuple(d for d in current if d != 1) == tuple(d for d in target_shape if d != 1):
        # Only singleton dimensions differ.
        return param.reshape(target_shape)
    if param.ndim == 2 and target_shape == tuple(reversed(current)):
        return param.transpose(0, 1).contiguous()
    return param


def _unflatten_quantized(param, target_shape: tuple[int, int]):
    """
    Restore the rows of a quantized weight stored flattened to one dimension.

    Quant blocks run along the last dimension, so when a row holds a whole
    number of blocks the flat bytes are already in row-major order.
    """
    from diffusers.quantizers.gguf.utils import GGML_QUANT_SIZES, GGUFParameter

    block_size, _ = GGML_QUANT_SIZES[param.quant_type]
    rows, columns = target_shape
    if columns % block_size != 0:
        return param
    packed = param.as_tensor() if hasattr(param, 'as_tensor') else param
    return GGUFParameter(packed.reshape(rows, -1), quant_type=param.quant_type)


def _target_shapes(config_repo: str, subfolder: str, model_class, token, local_files_only):
    from accelerate import init_empty_weights
    config = model_class.load_config(
        config_repo,
        subfolder=subfolder,
        token=token,
        local_files_only=local_files_only)
    with init_empty_weights():
        model = model_class.from_config(config)
    return {name: _shape(param.shape) for name, param in model.state_dict().items()}


def adapt_ltx25_checkpoint(checkpoint: dict, config_repo: str, subfolder: str,
                           token=None, local_files_only: bool = False) -> dict:
    """Rename an LTX 2.5 Comfy/original GGUF and fix transposed tensors."""
    import diffusers
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers

    checkpoint = _strip_comfy_prefix(checkpoint)
    converted = convert_ltx2_transformer_to_diffusers(
        _singlefilecompat.rename_ltx25_prompt_adaln(checkpoint))
    targets = _target_shapes(
        config_repo, subfolder, diffusers.LTX2VideoTransformer3DModel,
        token, local_files_only)
    aligned = {}
    for name, param in converted.items():
        if name in targets:
            aligned[name] = _align_tensor(param, targets[name])
    missing = [name for name in targets if name not in aligned]
    if missing:
        _messages.debug_log(
            f'LTX GGUF is missing {len(missing)} Diffusers tensors, '
            f'for example {missing[:4]}.')
    return aligned


def _is_gguf_parameter(value) -> bool:
    return getattr(value, 'quant_type', None) is not None


def _slicable_tensor(value):
    """
    Plain tensor that convert_wan_transformer_to_diffusers can slice.

    A packed GGUFParameter keeps ``quant_type`` through ``__torch_function__``,
    so ``linear1_kv`` splits and motion-encoder bias indexes become garbage.
    """
    if not _is_gguf_parameter(value):
        return value
    from diffusers.quantizers.gguf import utils as gguf_utils
    if value.quant_type in gguf_utils.UNQUANTIZED_TYPES:
        tensor = value.as_tensor()
        shape = getattr(value, 'quant_shape', None)
        if shape is not None and tuple(tensor.shape) != tuple(shape):
            try:
                return tensor.view(tuple(shape))
            except RuntimeError:
                return tensor
        return tensor
    return gguf_utils.dequantize_gguf_tensor(value)


def _wan_animate_hostile_key(key: str) -> bool:
    return '.linear1_kv.' in key or (
        'motion_encoder.enc.net_app.convs.' in key and '.bias' in key)


def adapt_wan_animate_checkpoint(checkpoint: dict) -> dict:
    """Dequantize Wan-Animate tensors that Diffusers splits or indexes."""
    checkpoint = _strip_comfy_prefix(checkpoint)
    return {
        key: _slicable_tensor(value) if _wan_animate_hostile_key(key) else value
        for key, value in checkpoint.items()
    }


_WAN_FLF_POS_EMBED = 'condition_embedder.image_embedder.pos_embed'
_WAN_FLF_POS_SOURCE = 'img_emb.emb_pos'
_WAN_FLF_POS_SHAPE = (1, 514, 1280)


def _wan_checkpoint_needs_convert(keys) -> bool:
    """Original Wan names still need convert_wan_transformer_to_diffusers."""
    for key in keys:
        if (key.startswith('img_emb.') or key.startswith('time_embedding.')
                or '.self_attn.' in key or '.cross_attn.' in key):
            return True
    return False


def _materialize_wan_pos_embed(param):
    """
    Plain ``(1, 514, 1280)`` table for ``image_embedder.pos_embed``.

    The original checkpoint stores this as ``img_emb.emb_pos``. A packed
    GGUFParameter cannot be copied onto the meta parameter.
    """
    if _is_gguf_parameter(param):
        tensor = _gguf_float_tensor(param)
    elif isinstance(param, torch.Tensor):
        tensor = param.detach().contiguous()
    else:
        return param
    current = _shape(tensor.shape)
    target = _WAN_FLF_POS_SHAPE
    if current == target:
        return tensor
    if current == (target[2], target[1]):
        return tensor.transpose(0, 1).reshape(target).contiguous()
    if current == target[1:]:
        return tensor.reshape(target)
    if current == (target[0], target[2], target[1]):
        return tensor.transpose(1, 2).contiguous()
    if tensor.numel() == math.prod(target):
        return tensor.reshape(target).contiguous()
    return tensor


def assign_wan_flf_pos_embed(checkpoint: dict) -> dict:
    """Copy ``img_emb.emb_pos`` onto ``condition_embedder.image_embedder.pos_embed``.

    ``convert_wan_transformer_to_diffusers`` leaves the original name in place.
    The Diffusers parameter is then never loaded and stays a meta tensor, and
    moving the module fails with "Cannot copy out of meta tensor".
    """
    source = checkpoint.pop(_WAN_FLF_POS_SOURCE, None)
    if source is None:
        source = checkpoint.get(_WAN_FLF_POS_EMBED)
    if source is None:
        return checkpoint
    checkpoint[_WAN_FLF_POS_EMBED] = _materialize_wan_pos_embed(source)
    _messages.debug_log(
        'Mapped Wan FLF2V img_emb.emb_pos onto '
        f'{_WAN_FLF_POS_EMBED} {_WAN_FLF_POS_SHAPE}.')
    return checkpoint


def adapt_wan_flf_checkpoint(checkpoint: dict) -> dict:
    """Map ``img_emb.emb_pos`` onto the Diffusers first-last image embedding."""
    checkpoint = _strip_comfy_prefix(dict(checkpoint))
    if _wan_checkpoint_needs_convert(checkpoint):
        from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
        checkpoint = convert_wan_transformer_to_diffusers(checkpoint)
    return assign_wan_flf_pos_embed(checkpoint)


def adapt_qwen_checkpoint(checkpoint: dict) -> dict:
    """Drop the Comfy ``model.diffusion_model.`` prefix. Names are otherwise Diffusers names."""
    return _strip_comfy_prefix(checkpoint)


def adapt_krea2_checkpoint(checkpoint: dict, config_repo: str, subfolder: str,
                           token=None, local_files_only: bool = False) -> dict:
    """Convert Krea 2 GGUF keys and align unquantized singleton tensors."""
    import diffusers
    from diffusers.loaders.single_file_utils import convert_krea2_transformer_checkpoint_to_diffusers

    checkpoint = _strip_comfy_prefix(checkpoint)
    converted = convert_krea2_transformer_checkpoint_to_diffusers(checkpoint)
    targets = _target_shapes(
        config_repo, subfolder, diffusers.Krea2Transformer2DModel,
        token, local_files_only)
    return {
        name: _align_tensor(param, targets[name])
        for name, param in converted.items()
        if name in targets
    }


def _check_quantized_decodes(checkpoint: dict, label: str) -> None:
    """Fail clearly when a quantized weight does not decode to finite values."""
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

    quantized = [value for value in checkpoint.values() if _is_gguf_parameter(value)]
    if not quantized:
        return
    sample = min(quantized, key=lambda value: value.numel())
    if not torch.isfinite(dequantize_gguf_tensor(sample)).all():
        raise ValueError(
            f'{label} GGUF quantized weights decode to NaN/Inf values. The file '
            f'is corrupt or was written with a nonstandard block layout.')


def adapt_flux3_checkpoint(checkpoint: dict) -> dict:
    """Rename a ComfyUI / BFL FLUX.3 transformer to Diffusers names.

    ``final_layer.image.adaLN_modulation.1.weight`` is stored packed in some
    GGUF files. The Diffusers converter swaps its scale and shift halves, so
    that one tensor is dequantized first. A ComfyUI fused ``qkv`` is split
    into ``to_q``, ``to_k``, and ``to_v``.
    """
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

    stripped = {}
    for key, value in checkpoint.items():
        if key.startswith(_COMFY_PREFIX):
            key = key[len(_COMFY_PREFIX):]
        stripped[key] = value
    adaln = 'final_layer.image.adaLN_modulation.1.weight'
    if adaln in stripped and _is_gguf_parameter(stripped[adaln]):
        stripped[adaln] = dequantize_gguf_tensor(stripped[adaln])
    return _singlefilecompat.convert_flux3_transformer_checkpoint(stripped)


def _adapt_named_checkpoint(checkpoint: dict, config_repo: str, subfolder: str,
                            model_class, label: str, token, local_files_only) -> dict:
    """Align a GGUF whose keys are already Diffusers names."""
    checkpoint = _strip_comfy_prefix(checkpoint)
    _check_quantized_decodes(checkpoint, label)
    targets = _target_shapes(
        config_repo, subfolder, model_class, token, local_files_only)
    return {
        name: _align_tensor(param, targets[name]) if name in targets else param
        for name, param in checkpoint.items()
    }


def adapt_kandinsky5_checkpoint(checkpoint: dict, config_repo: str, subfolder: str,
                                token=None, local_files_only: bool = False) -> dict:
    """Align Kandinsky 5 GGUF tensors. Keys are already Diffusers names."""
    import diffusers

    return _adapt_named_checkpoint(
        checkpoint, config_repo, subfolder, diffusers.Kandinsky5Transformer3DModel,
        'Kandinsky 5', token, local_files_only)


def adapt_kandinsky6_checkpoint(checkpoint: dict, config_repo: str, subfolder: str,
                                token=None, local_files_only: bool = False) -> dict:
    """Align Kandinsky 6 GGUF tensors. Keys are already Diffusers names."""
    import diffusers

    return _adapt_named_checkpoint(
        checkpoint, config_repo, subfolder, diffusers.Kandinsky6Transformer3DModel,
        'Kandinsky 6', token, local_files_only)


def adapt_minimax_h3_checkpoint(checkpoint: dict, config_repo: str, subfolder: str,
                                token=None, local_files_only: bool = False) -> dict:
    """Comfy MiniMax-H3 names become Diffusers names. Diffusers names are aligned."""
    import dgenerate.pipelinewrapper.minimaxh3 as _minimax
    import diffusers

    checkpoint = _strip_comfy_prefix(dict(checkpoint))
    if _minimax._is_comfy_state(checkpoint):
        return _minimax.comfy_to_diffusers(checkpoint)
    return _adapt_named_checkpoint(
        checkpoint, config_repo, subfolder, diffusers.MiniMaxH3Transformer3DModel,
        'MiniMax-H3', token, local_files_only)


def adapt_minimax_h3_image_checkpoint(checkpoint: dict, config_repo: str, subfolder: str,
                                      token=None, local_files_only: bool = False) -> dict:
    """Comfy MiniMax-H3-Image names become the image transformer's per-step heads."""
    import dgenerate.pipelinewrapper.minimaxh3 as _minimax
    import diffusers

    checkpoint = _strip_comfy_prefix(dict(checkpoint))
    if _minimax._is_comfy_state(checkpoint):
        return _minimax.convert_minimax_h3_image_checkpoint(checkpoint)
    return _adapt_named_checkpoint(
        checkpoint, config_repo, subfolder, diffusers.MiniMaxH3ImageTransformer2DModel,
        'MiniMax-H3-Image', token, local_files_only)


_QWEN_LAYERED_ADDITION_T = 'time_text_embed.addition_t_embedding.weight'


def materialize_unquantized_parameters(module) -> None:
    """Replace BF16, F16, and F32 GGUF parameters with plain floating tensors.

    Quantized linears stay packed. Norms, biases, and other full-precision
    tensors are stored as GGUF bytes, and ops such as RMSNorm read that byte
    width as the parameter shape.
    """
    import gguf

    plain = {
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    }
    for child in module.modules():
        for name, param in list(child.named_parameters(recurse=False)):
            if getattr(param, 'quant_type', None) not in plain:
                continue
            decoded = as_float_tensor(param).detach()
            child._parameters[name] = torch.nn.Parameter(decoded, requires_grad=False)


def as_float_tensor(value, dtype=None):
    """Plain floating tensor for a parameter, including a packed GGUF bias.

    A GGUF linear dequantizes its weight and then adds the bias. Biases on
    these checkpoints are often BF16 stored as bytes, so casting them with
    ``.to()`` leaves the packed storage in place.
    """
    if value is None:
        return None
    if type(value).__name__ == 'GGUFParameter' or _is_gguf_parameter(value):
        value = _gguf_float_tensor(value)
    if dtype is not None and value.dtype != dtype:
        value = value.to(dtype)
    return value


def _gguf_float_tensor(param):
    """
    Decode a GGUFParameter to a plain floating tensor.

    F16 is stored as a normal tensor by Diffusers' loader; BF16 is wrapped as
    ``GGUFParameter`` whose storage width is bytes (2x). ``dequantize_gguf_tensor``
    has no F16/BF16 codec, so reinterpret the packed storage with ``quant_shape``.
    """
    import gguf
    from diffusers.quantizers.gguf import utils as gguf_utils

    dtype_map = {
        gguf.GGMLQuantizationType.F32: torch.float32,
        gguf.GGMLQuantizationType.F16: torch.float16,
        gguf.GGMLQuantizationType.BF16: torch.bfloat16,
    }
    dtype = dtype_map.get(param.quant_type)
    if dtype is not None:
        quant_shape = _shape(param.quant_shape)
        return (
            param.as_tensor().contiguous().view(torch.uint8)
            .view(dtype).reshape(quant_shape)
        )
    if param.quant_type in gguf_utils.UNQUANTIZED_TYPES:
        return _slicable_tensor(param)
    return gguf_utils.dequantize_gguf_tensor(param)


def _materialize_embedding_weight(param, target_shape: tuple[int, ...]):
    """
    Turn a GGUF Embedding weight into a plain float table Diffusers can index.

    ``nn.Embedding`` does not go through ``GGUFLinear``, so a BF16/quant
    ``GGUFParameter`` keeps its packed byte width (e.g. 6144 for BF16
    ``(2, 3072)``). Some files also store the table transposed.
    """
    if _is_gguf_parameter(param):
        tensor = _gguf_float_tensor(param)
    else:
        tensor = param
    if not isinstance(tensor, torch.Tensor):
        return param
    tensor = tensor.detach().contiguous()
    current = _shape(tensor.shape)
    target_shape = _shape(target_shape)
    if current == target_shape:
        return tensor
    if current == tuple(reversed(target_shape)):
        return tensor.transpose(0, 1).contiguous()
    return tensor


def adapt_qwen_layered_checkpoint(checkpoint: dict) -> dict:
    """Strip Comfy prefixes and materialize ``addition_t_embedding`` for indexing."""
    checkpoint = adapt_qwen_checkpoint(checkpoint)
    weight = checkpoint.get(_QWEN_LAYERED_ADDITION_T)
    if weight is None:
        return checkpoint
    # nn.Embedding(2, inner_dim); Qwen-Image-Layered inner_dim is 3072.
    target = (2, 3072)
    quant_shape = getattr(weight, 'quant_shape', None)
    if quant_shape is not None:
        quant_shape = _shape(quant_shape)
        if quant_shape == (3072, 2) or quant_shape == (2, 3072):
            target = (2, 3072)
        elif len(quant_shape) == 2 and quant_shape[0] == 2:
            target = quant_shape
    elif isinstance(weight, torch.Tensor):
        shape = _shape(weight.shape)
        if shape == (3072, 2) or shape == (2, 3072):
            target = (2, 3072)
        elif len(shape) == 2 and shape[0] == 2:
            target = shape
    checkpoint[_QWEN_LAYERED_ADDITION_T] = _materialize_embedding_weight(
        weight, target)
    return checkpoint


_kernels = None
_triton_available = None
_iq4_tables = {}

# Same 16 levels Diffusers uses for IQ4_NL and IQ4_XS.
_IQ4_LEVELS = (
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _has_triton() -> bool:
    global _triton_available
    if _triton_available is None:
        import importlib.util
        _triton_available = importlib.util.find_spec('triton') is not None
    return _triton_available


def _compile_gguf_kernels():
    """One Triton kernel per GGUF type. Scales are prepared in PyTorch."""
    global _kernels
    if _kernels is not None:
        return _kernels

    import triton
    import triton.language as tl

    @triton.jit
    def _nibbles(blocks, stride_b, pid, offset):
        t = tl.arange(0, 32)
        byte_index = tl.where(t < 16, t, t - 16)
        shift = tl.where(t < 16, 0, 4)
        raw = tl.load(blocks + pid * stride_b + offset + byte_index).to(tl.int32) & 0xFF
        return t, (raw >> shift) & 0xF

    @triton.jit
    def q4_0(blocks, scale, out, stride_b, stride_o):
        pid = tl.program_id(0)
        t, nibble = _nibbles(blocks, stride_b, pid, 2)
        d = tl.load(scale + pid)
        tl.store(out + pid * stride_o + t, (nibble - 8).to(tl.float32) * d)

    @triton.jit
    def q4_1(blocks, scale, minimum, out, stride_b, stride_o):
        pid = tl.program_id(0)
        t, nibble = _nibbles(blocks, stride_b, pid, 4)
        d = tl.load(scale + pid)
        m = tl.load(minimum + pid)
        tl.store(out + pid * stride_o + t, nibble.to(tl.float32) * d + m)

    @triton.jit
    def q5_0(blocks, scale, words, out, stride_b, stride_o):
        pid = tl.program_id(0)
        t, nibble = _nibbles(blocks, stride_b, pid, 6)
        bit = (tl.load(words + pid) >> t) & 1
        d = tl.load(scale + pid)
        value = (nibble | (bit << 4)) - 16
        tl.store(out + pid * stride_o + t, value.to(tl.float32) * d)

    @triton.jit
    def q5_1(blocks, scale, minimum, words, out, stride_b, stride_o):
        pid = tl.program_id(0)
        t, nibble = _nibbles(blocks, stride_b, pid, 8)
        bit = (tl.load(words + pid) >> t) & 1
        d = tl.load(scale + pid)
        m = tl.load(minimum + pid)
        value = (nibble | (bit << 4)).to(tl.float32)
        tl.store(out + pid * stride_o + t, value * d + m)

    @triton.jit
    def q8_0(blocks, scale, out, stride_b, stride_o):
        pid = tl.program_id(0)
        t = tl.arange(0, 32)
        raw = tl.load(blocks + pid * stride_b + 2 + t).to(tl.int32) & 0xFF
        value = tl.where(raw >= 128, raw - 256, raw).to(tl.float32)
        tl.store(out + pid * stride_o + t, value * tl.load(scale + pid))

    @triton.jit
    def iq4_nl(blocks, scale, table, out, stride_b, stride_o):
        pid = tl.program_id(0)
        t, nibble = _nibbles(blocks, stride_b, pid, 2)
        level = tl.load(table + nibble)
        tl.store(out + pid * stride_o + t, level * tl.load(scale + pid))

    @triton.jit
    def q4_k(blocks, scale, mins, out, stride_b, stride_s, stride_o):
        # Lows of each 32-byte group, then its highs. Eight groups.
        pid = tl.program_id(0)
        t = tl.arange(0, 256)
        group = t // 32
        column = t % 32
        byte = (group // 2) * 32 + column
        shift = (group % 2) * 4
        raw = tl.load(blocks + pid * stride_b + 16 + byte).to(tl.int32) & 0xFF
        nibble = ((raw >> shift) & 0xF).to(tl.float32)
        base = pid * stride_s + group
        tl.store(
            out + pid * stride_o + t,
            nibble * tl.load(scale + base) - tl.load(mins + base))

    @triton.jit
    def q5_k(blocks, scale, mins, out, stride_b, stride_s, stride_o):
        pid = tl.program_id(0)
        t = tl.arange(0, 256)
        group = t // 32
        column = t % 32
        byte = (group // 2) * 32 + column
        shift = (group % 2) * 4
        raw = tl.load(blocks + pid * stride_b + 48 + byte).to(tl.int32) & 0xFF
        nibble = (raw >> shift) & 0xF
        high = tl.load(blocks + pid * stride_b + 16 + column).to(tl.int32) & 0xFF
        value = (nibble | (((high >> group) & 1) << 4)).to(tl.float32)
        base = pid * stride_s + group
        tl.store(
            out + pid * stride_o + t,
            value * tl.load(scale + base) - tl.load(mins + base))

    @triton.jit
    def q6_k(blocks, scale, out, stride_b, stride_s, stride_o):
        pid = tl.program_id(0)
        t = tl.arange(0, 256)
        group = t // 32
        column = t % 32
        low_index = t // 64
        low_byte = (low_index // 2) * 64 + (t % 64)
        low_shift = (low_index % 2) * 4
        low = tl.load(blocks + pid * stride_b + low_byte).to(tl.int32) & 0xFF
        low = (low >> low_shift) & 0xF
        high_shift = (group % 4) * 2
        high_byte = (group // 4) * 32 + column
        high = tl.load(blocks + pid * stride_b + 128 + high_byte).to(tl.int32) & 0xFF
        high = (high >> high_shift) & 3
        value = (low | (high << 4)) - 32
        sc = tl.load(scale + pid * stride_s + (t // 16))
        tl.store(out + pid * stride_o + t, value.to(tl.float32) * sc)

    @triton.jit
    def q3_k(blocks, scale, out, stride_b, stride_s, stride_o):
        pid = tl.program_id(0)
        t = tl.arange(0, 256)
        column = t % 32
        band = t // 32
        shift = (band % 4) * 2
        byte = (band // 4) * 32 + column
        raw = tl.load(blocks + pid * stride_b + 32 + byte).to(tl.int32) & 0xFF
        low = (raw >> shift) & 3
        mask = tl.load(blocks + pid * stride_b + column).to(tl.int32) & 0xFF
        high = ((mask >> band) & 1) ^ 1
        value = low.to(tl.float32) - (high << 2).to(tl.float32)
        sc = tl.load(scale + pid * stride_s + (t // 16))
        tl.store(out + pid * stride_o + t, value * sc)

    @triton.jit
    def q2_k(blocks, scale, minimum, out, stride_b, stride_s, stride_o):
        pid = tl.program_id(0)
        t = tl.arange(0, 256)
        column = t % 32
        band = t // 32
        shift = (band % 4) * 2
        byte = (band // 4) * 32 + column
        raw = tl.load(blocks + pid * stride_b + 16 + byte).to(tl.int32) & 0xFF
        code = ((raw >> shift) & 3).to(tl.float32)
        base = pid * stride_s + (t // 16)
        tl.store(
            out + pid * stride_o + t,
            code * tl.load(scale + base) - tl.load(minimum + base))

    @triton.jit
    def iq4_xs(blocks, scale, table, out, stride_b, stride_s, stride_o):
        pid = tl.program_id(0)
        t = tl.arange(0, 256)
        group = t // 32
        pos = t % 32
        byte = group * 16 + (pos % 16)
        shift = (pos // 16) * 4
        raw = tl.load(blocks + pid * stride_b + 8 + byte).to(tl.int32) & 0xFF
        level = tl.load(table + ((raw >> shift) & 0xF))
        sc = tl.load(scale + pid * stride_s + group)
        tl.store(out + pid * stride_o + t, level * sc)

    _kernels = {
        'q4_0': q4_0, 'q4_1': q4_1, 'q5_0': q5_0, 'q5_1': q5_1, 'q8_0': q8_0,
        'iq4_nl': iq4_nl, 'q4_k': q4_k, 'q5_k': q5_k, 'q6_k': q6_k,
        'q3_k': q3_k, 'q2_k': q2_k, 'iq4_xs': iq4_xs,
    }
    return _kernels


def _half(blocks, offset):
    """fp16 field at a byte offset, one value per block, as float32."""
    return blocks[:, offset:offset + 2].contiguous().view(torch.float16).reshape(-1).float()


def _word(blocks, offset):
    """Little-endian uint32 at a byte offset, kept as int32 bits."""
    raw = blocks[:, offset:offset + 4].to(torch.int32)
    return raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16) | (raw[:, 3] << 24)


def _scale_min(scales):
    """Q4_K / Q5_K 12-byte scale block -> ``(blocks, 8)`` scale and min codes."""
    low = scales[:, 0:4].to(torch.int32)
    high = scales[:, 4:8].to(torch.int32)
    mixed = scales[:, 8:12].to(torch.int32)
    sc = torch.cat([low & 0x3F, (mixed & 0x0F) | ((low >> 2) & 0x30)], dim=1)
    minimum = torch.cat([high & 0x3F, (mixed >> 4) | ((high >> 2) & 0x30)], dim=1)
    return sc.float(), minimum.float()


def _iq4_table(device):
    table = _iq4_tables.get(str(device))
    if table is None:
        table = torch.tensor(_IQ4_LEVELS, dtype=torch.float32, device=device)
        _iq4_tables[str(device)] = table
    return table


def _launch_block(qtype, blocks, out):
    """Run the kernel for ``qtype``. ``blocks`` is ``(n, type_size)`` uint8."""
    kernels = _compile_gguf_kernels()
    n = blocks.shape[0]
    stride_b = blocks.stride(0)
    stride_o = out.stride(0)
    name = qtype.name.lower()

    if name in ('q4_0', 'q8_0'):
        kernels[name][(n,)](blocks, _half(blocks, 0), out, stride_b, stride_o)
        return
    if name == 'q4_1':
        kernels[name][(n,)](
            blocks, _half(blocks, 0), _half(blocks, 2), out, stride_b, stride_o)
        return
    if name == 'q5_0':
        kernels[name][(n,)](
            blocks, _half(blocks, 0), _word(blocks, 2), out, stride_b, stride_o)
        return
    if name == 'q5_1':
        kernels[name][(n,)](
            blocks, _half(blocks, 0), _half(blocks, 2), _word(blocks, 4),
            out, stride_b, stride_o)
        return
    if name == 'iq4_nl':
        kernels[name][(n,)](
            blocks, _half(blocks, 0), _iq4_table(blocks.device), out, stride_b, stride_o)
        return
    if name in ('q4_k', 'q5_k'):
        scale, minimum = _scale_min(blocks[:, 4:16])
        scale = (_half(blocks, 0).unsqueeze(1) * scale).contiguous()
        minimum = (_half(blocks, 2).unsqueeze(1) * minimum).contiguous()
        kernels[name][(n,)](
            blocks, scale, minimum, out, stride_b, scale.stride(0), stride_o)
        return
    if name == 'q6_k':
        codes = blocks[:, 192:208].contiguous().view(torch.int8).float()
        scale = (_half(blocks, 208).unsqueeze(1) * codes).contiguous()
        kernels[name][(n,)](blocks, scale, out, stride_b, scale.stride(0), stride_o)
        return
    if name == 'q3_k':
        index = torch.arange(16, device=blocks.device)
        low = blocks[:, 96:104].to(torch.int32)
        high = blocks[:, 104:108].to(torch.int32)
        low = (low[:, index % 8] >> ((index // 8) * 4)) & 0xF
        high = (high[:, index % 4] >> ((index // 4) * 2)) & 3
        codes = (low | (high << 4)).to(torch.int8).float() - 32
        scale = (_half(blocks, 108).unsqueeze(1) * codes).contiguous()
        kernels[name][(n,)](blocks, scale, out, stride_b, scale.stride(0), stride_o)
        return
    if name == 'q2_k':
        codes = blocks[:, :16].to(torch.int32)
        scale = (_half(blocks, 80).unsqueeze(1) * (codes & 0xF).float()).contiguous()
        minimum = (_half(blocks, 82).unsqueeze(1) * (codes >> 4).float()).contiguous()
        kernels[name][(n,)](
            blocks, scale, minimum, out, stride_b, scale.stride(0), stride_o)
        return
    if name == 'iq4_xs':
        group = torch.arange(8, device=blocks.device)
        low = blocks[:, 4:8].to(torch.int32)
        high = blocks[:, 2:4].contiguous().view(torch.int16).reshape(-1).to(torch.int32)
        low = (low[:, group // 2] >> ((group % 2) * 4)) & 0xF
        high = (high.unsqueeze(1) >> (group * 2)) & 3
        codes = (low | (high << 4)).float() - 32
        scale = (_half(blocks, 0).unsqueeze(1) * codes).contiguous()
        kernels[name][(n,)](
            blocks, scale, _iq4_table(blocks.device), out,
            stride_b, scale.stride(0), stride_o)
        return
    raise KeyError(name)


def dequantize(param):
    """
    Dense float32 weight for a GGUF parameter, or ``None``.

    ``None`` means Triton is unavailable, the device is not CUDA or XPU, or
    the quant type is not one of the unpacked types. The shape matches
    Diffusers ``dequantize_gguf_tensor``.
    """
    qtype = getattr(param, 'quant_type', None)
    if qtype is None or not _has_triton() or param.device.type not in ('cuda', 'xpu'):
        return None
    import gguf
    try:
        qtype = gguf.GGMLQuantizationType(int(qtype))
    except ValueError:
        return None
    if qtype.name.lower() not in (
            'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0', 'q2_k', 'q3_k', 'q4_k',
            'q5_k', 'q6_k', 'iq4_nl', 'iq4_xs'):
        return None
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    data = param.as_tensor() if hasattr(param, 'as_tensor') else param.detach()
    data = data.view(torch.uint8)
    if data.numel() % type_size:
        return None
    n_blocks = data.numel() // type_size
    blocks = data.reshape(n_blocks, type_size)
    if not blocks.is_contiguous():
        blocks = blocks.contiguous()
    out = torch.empty(n_blocks, block_size, device=blocks.device, dtype=torch.float32)
    if n_blocks:
        _launch_block(qtype, blocks, out)
    shape = (*param.shape[:-1], param.shape[-1] // type_size * block_size)
    return out.reshape(shape)


def linear(module, inputs):
    """
    F.linear on a Triton-unpacked GGUF weight.

    None leaves the caller on the PyTorch dequant path. A Comfy transposed
    weight is transposed after the unpack, same as the PyTorch path.
    """
    try:
        weight = dequantize(module.weight)
    except Exception as error:
        _messages.debug_log(f'GGUF Triton dequant failed, using PyTorch: {error}')
        return None
    if weight is None:
        return None
    if getattr(module.weight, '_dgenerate_comfy_transpose', False):
        weight = weight.transpose(0, 1).contiguous()
    dtype = getattr(module, 'compute_dtype', None) or inputs.dtype
    weight = weight.to(dtype)
    return torch.nn.functional.linear(inputs, weight, as_float_tensor(module.bias, dtype))


def install_gguf_patches() -> None:
    """
    Let a Comfy quantized linear whose packed shape is transposed load
    into a Diffusers linear and transpose after dequantization.
    """
    global _patches_installed
    if _patches_installed:
        return

    import diffusers
    import diffusers.loaders.single_file_model as single_file_model
    import diffusers.quantizers.gguf.gguf_quantizer as gguf_quantizer
    import diffusers.quantizers.gguf.utils as gguf_utils

    original_should_convert = single_file_model._should_convert_state_dict_to_diffusers
    original_check = gguf_quantizer.GGUFQuantizer.check_quantized_param_shape
    original_create = gguf_quantizer.GGUFQuantizer.create_quantized_param
    original_new = gguf_utils.GGUFParameter.__new__
    original_torch_function = gguf_utils.GGUFParameter.__torch_function__

    _singlefilecompat.register_single_file_classes()

    def should_convert(model_state_dict, checkpoint_state_dict):
        if os.environ.get('DGENERATE_GGUF_SKIP_RECONVERT') == '1':
            return False
        return original_should_convert(model_state_dict, checkpoint_state_dict)

    def check_shape(self, param_name, current_param, loaded_param):
        if getattr(loaded_param, '_dgenerate_comfy_transpose', False):
            block_size, type_size = gguf_utils.GGML_QUANT_SIZES[loaded_param.quant_type]
            inferred = gguf_utils._quant_shape_from_byte_shape(
                loaded_param.shape, type_size, block_size)
            expected = tuple(current_param.shape)
            if tuple(inferred) == tuple(reversed(expected)):
                return True
        return original_check(self, param_name, current_param, loaded_param)

    def create_quantized_param(self, model, param_value, param_name, target_device,
                               *args, **kwargs):
        # keep_in_fp32_modules (Wan norm*, time_embedder, motion_synthesis_weight)
        # are dequantized and written straight into _parameters as a Tensor.
        # Sequential offload rebuilds type(param)(value, requires_grad=...),
        # and torch.Tensor rejects requires_grad.
        original_create(
            self, model, param_value, param_name, target_device, *args, **kwargs)
        from diffusers.utils import get_module_from_name
        module, tensor_name = get_module_from_name(model, param_name)
        value = module._parameters.get(tensor_name)
        if type(value) is torch.Tensor:
            module._parameters[tensor_name] = torch.nn.Parameter(
                value, requires_grad=bool(value.requires_grad))

    def forward_native(self, inputs):
        # Triton unpack plus a PyTorch GEMM. Comfy's transposed weights are
        # unpacked and transposed, because the packed axis is flipped.
        output = linear(self, inputs)
        if output is not None:
            return output
        weight = gguf_utils.dequantize_gguf_tensor(self.weight)
        if getattr(self.weight, '_dgenerate_comfy_transpose', False):
            weight = weight.transpose(0, 1).contiguous()
        weight = weight.to(self.compute_dtype)
        bias = as_float_tensor(self.bias, self.compute_dtype)
        return torch.nn.functional.linear(inputs, weight, bias)

    def forward(self, inputs):
        # Diffusers' CUDA path calls ggml_dequantize. Those ops are not built
        # for every platform, so the Triton path runs first and the PyTorch
        # dequant is the fallback.
        return self.forward_native(inputs)

    def parameter_new(cls, data, requires_grad=False, quant_type=None):
        # Accelerate rebuilds the parameter class when offloading to meta and
        # does not pass quant_type. The value just moved still has it.
        if quant_type is None:
            quant_type = getattr(data, 'quant_type', None)
        transpose = getattr(data, '_dgenerate_comfy_transpose', False)
        if quant_type is None:
            blank = data if data is not None else torch.empty(0)
            self = torch.Tensor._make_subclass(cls, blank, requires_grad)
            self.quant_type = None
            self.quant_shape = tuple(self.shape)
            self._dgenerate_comfy_transpose = transpose
            return self
        self = original_new(cls, data, requires_grad=requires_grad, quant_type=quant_type)
        if transpose:
            self._dgenerate_comfy_transpose = True
        return self

    def _source_parameter(args):
        for arg in args:
            if isinstance(arg, gguf_utils.GGUFParameter):
                return arg
            if isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], gguf_utils.GGUFParameter):
                return arg[0]
        return None

    def torch_function(cls, func, types, args=(), kwargs=None):
        result = original_torch_function.__func__(cls, func, types, args, kwargs)
        source = _source_parameter(args)
        if source is not None and getattr(source, '_dgenerate_comfy_transpose', False):
            tensors = result if isinstance(result, (list, tuple)) else (result,)
            for item in tensors:
                if isinstance(item, torch.Tensor):
                    item._dgenerate_comfy_transpose = True
        return result

    single_file_model._should_convert_state_dict_to_diffusers = should_convert
    gguf_quantizer.GGUFQuantizer.check_quantized_param_shape = check_shape
    gguf_quantizer.GGUFQuantizer.create_quantized_param = create_quantized_param
    gguf_utils.GGUFLinear.forward_native = forward_native
    gguf_utils.GGUFLinear.forward = forward
    gguf_utils.GGUFParameter.__new__ = parameter_new
    gguf_utils.GGUFParameter.__torch_function__ = classmethod(torch_function)
    _patches_installed = True


@contextlib.contextmanager
def skip_diffusers_reconvert():
    """The LTX converter already ran. A second pass would rename keys again."""
    install_gguf_patches()
    previous = os.environ.get('DGENERATE_GGUF_SKIP_RECONVERT')
    os.environ['DGENERATE_GGUF_SKIP_RECONVERT'] = '1'
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('DGENERATE_GGUF_SKIP_RECONVERT', None)
        else:
            os.environ['DGENERATE_GGUF_SKIP_RECONVERT'] = previous


def plan_gguf_load(path: str, config: str | None, subfolder: str | None,
                   token=None, local_files_only: bool = False):
    """
    Choose a Diffusers config and, when the file is a Comfy layout Diffusers
    will not convert, return a remapped state dict.

    :return: ``(checkpoint or None, config repo or None, subfolder or None, already_converted)``
        ``checkpoint`` is ``None`` when Diffusers should read the file itself.
    """
    shapes = read_gguf_shapes(path)
    layout = detect_gguf_layout(shapes)
    if layout is None:
        return None, config, subfolder, False

    if config is None:
        config = layout.config_repo
        subfolder = subfolder or layout.subfolder
    elif subfolder is None:
        subfolder = layout.subfolder

    message = (
        f'GGUF transformer detected as {layout.kind}'
        + (' (Comfy layout)' if layout.comfy else '')
        + f'. Using {config}.')
    if config != layout.config_repo:
        message += f' Detected config was {layout.config_repo}.'
    _messages.debug_log(message)

    if not layout.adapt:
        return None, config, subfolder, False

    from diffusers.models.model_loading_utils import load_gguf_checkpoint
    install_gguf_patches()
    checkpoint = load_gguf_checkpoint(path)
    if layout.kind in {'ltx-2', 'ltx-2.5'}:
        checkpoint = adapt_ltx25_checkpoint(
            checkpoint, config, subfolder or 'transformer',
            token=token, local_files_only=local_files_only)
        return checkpoint, config, subfolder, True
    if layout.kind == 'qwen-image':
        return adapt_qwen_checkpoint(checkpoint), config, subfolder, False
    if layout.kind == 'krea2':
        checkpoint = adapt_krea2_checkpoint(
            checkpoint, config, subfolder or 'transformer',
            token=token, local_files_only=local_files_only)
        return checkpoint, config, subfolder, True
    if layout.kind == 'flux3':
        return adapt_flux3_checkpoint(checkpoint), config, subfolder, True
    if layout.kind == 'kandinsky5':
        checkpoint = adapt_kandinsky5_checkpoint(
            checkpoint, config, subfolder or 'transformer',
            token=token, local_files_only=local_files_only)
        return checkpoint, config, subfolder, False
    if layout.kind == 'kandinsky6':
        checkpoint = adapt_kandinsky6_checkpoint(
            checkpoint, config, subfolder or 'transformer',
            token=token, local_files_only=local_files_only)
        return checkpoint, config, subfolder, False
    if layout.kind == 'minimax-h3':
        checkpoint = adapt_minimax_h3_checkpoint(
            checkpoint, config, subfolder or 'transformer',
            token=token, local_files_only=local_files_only)
        return checkpoint, config, subfolder, True
    if layout.kind == 'minimax-h3-image':
        checkpoint = adapt_minimax_h3_image_checkpoint(
            checkpoint, config, subfolder or 'transformer',
            token=token, local_files_only=local_files_only)
        return checkpoint, config, subfolder, True
    if layout.kind == 'qwen-image-layered':
        return adapt_qwen_layered_checkpoint(checkpoint), config, subfolder, False
    if layout.kind == 'hyimage21':
        import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat
        converted = _singlefilecompat.convert_hunyuanimage_transformer_for_single_file(checkpoint)
        return converted, config, subfolder, True
    if layout.kind == 'wan-animate-14B':
        return adapt_wan_animate_checkpoint(checkpoint), config, subfolder, False
    if layout.kind == 'wan-flf2v-14B':
        return adapt_wan_flf_checkpoint(checkpoint), config, subfolder, True
    return None, config, subfolder, False


# llama.cpp text-encoder architectures. The tensor names are llama.cpp names
# (``enc.blk.0.attn_q.weight``, ``blk.0.attn_q.weight``), not Hugging Face names.
_TEXT_GGUF_ARCHITECTURES = frozenset({
    't5', 't5encoder', 'llama', 'qwen2', 'qwen2vl', 'qwen3', 'qwen3vl', 'qwen35', 'qwen35moe',
    'gemma3', 'gemma4',
    'clip', 'clip_l', 'clip_g',
})

# Replacements run in order, the same order ComfyUI-GGUF uses. A longer
# name such as ``attn_q_norm`` has to be replaced before ``attn_q``.
_T5_GGUF_KEYS = (
    ('enc.', 'encoder.'),
    ('.blk.', '.block.'),
    ('token_embd', 'shared'),
    ('output_norm', 'final_layer_norm'),
    ('attn_rel_b', 'layer.0.SelfAttention.relative_attention_bias'),
    ('attn_q', 'layer.0.SelfAttention.q'),
    ('attn_k', 'layer.0.SelfAttention.k'),
    ('attn_v', 'layer.0.SelfAttention.v'),
    ('attn_o', 'layer.0.SelfAttention.o'),
    ('attn_norm', 'layer.0.layer_norm'),
    ('ffn_up', 'layer.1.DenseReluDense.wi_1'),
    ('ffn_down', 'layer.1.DenseReluDense.wo'),
    ('ffn_gate', 'layer.1.DenseReluDense.wi_0'),
    ('ffn_norm', 'layer.1.layer_norm'),
)

_LLAMA_GGUF_KEYS = (
    ('blk.', 'model.layers.'),
    ('attn_q_norm.', 'self_attn.q_norm.'),
    ('attn_k_norm.', 'self_attn.k_norm.'),
    ('attn_v_norm.', 'self_attn.v_norm.'),
    ('attn_q', 'self_attn.q_proj'),
    ('attn_k', 'self_attn.k_proj'),
    ('attn_v', 'self_attn.v_proj'),
    ('attn_output', 'self_attn.o_proj'),
    ('attn_norm', 'input_layernorm'),
    ('ffn_up', 'mlp.up_proj'),
    ('ffn_down', 'mlp.down_proj'),
    ('ffn_gate', 'mlp.gate_proj'),
    ('ffn_norm', 'post_attention_layernorm'),
    ('token_embd', 'model.embed_tokens'),
    ('output_norm', 'model.norm'),
    ('output.weight', 'lm_head.weight'),
)

_GEMMA3_GGUF_KEYS = (
    ('blk.', 'model.layers.'),
    ('attn_q_norm.', 'self_attn.q_norm.'),
    ('attn_k_norm.', 'self_attn.k_norm.'),
    ('attn_v_norm.', 'self_attn.v_norm.'),
    ('attn_q', 'self_attn.q_proj'),
    ('attn_k', 'self_attn.k_proj'),
    ('attn_v', 'self_attn.v_proj'),
    ('attn_output', 'self_attn.o_proj'),
    ('attn_norm', 'input_layernorm'),
    ('ffn_up', 'mlp.up_proj'),
    ('ffn_down', 'mlp.down_proj'),
    ('ffn_gate', 'mlp.gate_proj'),
    ('ffn_norm', 'pre_feedforward_layernorm'),
    ('post_ffw_norm', 'post_feedforward_layernorm'),
    ('post_attention_norm', 'post_attention_layernorm'),
    ('token_embd', 'model.embed_tokens'),
    ('output_norm', 'model.norm'),
    ('output.weight', 'lm_head.weight'),
)

_GEMMA3_NORM_MARKERS = (
    'input_layernorm.weight',
    'post_attention_layernorm.weight',
    'pre_feedforward_layernorm.weight',
    'post_feedforward_layernorm.weight',
    'self_attn.q_norm.weight',
    'self_attn.k_norm.weight',
    'model.norm.weight',
)


def _gguf_architecture(path: str) -> str | None:
    import gguf

    reader = gguf.GGUFReader(path)
    field = reader.get_field('general.architecture')
    if field is None or not field.types:
        return None
    return str(bytes(field.parts[field.data[-1]]), encoding='utf-8')


def _remap_keys(state: dict, key_map: tuple) -> dict:
    remapped = {}
    for key, value in state.items():
        for source, target in key_map:
            key = key.replace(source, target)
        remapped[key] = value
    return remapped


def _clip_text_keys(keys) -> bool:
    """Hugging Face CLIP text tower, the names Comfy clip files and our quantizer use."""
    return any(
        key.endswith('embeddings.token_embedding.weight') and 'vision' not in key
        for key in keys)


def _huggingface_names(keys, architecture: str) -> bool:
    """True when the file already stores Hugging Face parameter names."""
    if architecture in {'t5', 't5encoder'}:
        return any(key.startswith('encoder.block.') or key == 'shared.weight' for key in keys)
    if architecture in {'clip', 'clip_l', 'clip_g'}:
        return _clip_text_keys(keys) or any('.self_attn.' in key for key in keys)
    return any(key.endswith('embed_tokens.weight') or '.self_attn.' in key for key in keys)


def logical_tensor_shape(value) -> tuple[int, ...]:
    """Decoded tensor shape. A packed GGUF weight's ``shape`` is its byte storage."""
    quant_shape = getattr(value, 'quant_shape', None)
    if _is_gguf_parameter(value) and quant_shape is not None:
        return tuple(int(item) for item in quant_shape)
    return tuple(int(item) for item in value.shape)


def permute_llama_attention(state: dict, n_head: int, n_kv: int) -> dict:
    """Undo the query and key interleave llama.cpp applies when it writes a GGUF.

    Quantized rows stay packed: each row is its own sequence of quant blocks,
    so moving rows moves the packed bytes the same way it moves a dense matrix.
    """
    from diffusers.quantizers.gguf.utils import GGUFParameter

    def permute(value, heads):
        rows = value.shape[0]
        if heads <= 0 or rows % (heads * 2) != 0:
            return value
        quant_type = getattr(value, 'quant_type', None) if _is_gguf_parameter(value) else None
        plain = value.as_tensor() if hasattr(value, 'as_tensor') else value.detach()
        plain = plain.contiguous()
        permuted = plain.reshape(
            heads, rows // heads // 2, 2, *plain.shape[1:]).swapaxes(1, 2).reshape(plain.shape).contiguous()
        if quant_type is None:
            return permuted
        return GGUFParameter(permuted, quant_type=quant_type)

    updated = {}
    for key, value in state.items():
        language = 'vision' not in key and 'visual' not in key
        if language and key.endswith(('q_proj.weight', 'q_proj.bias')):
            value = permute(value, n_head)
        elif language and key.endswith(('k_proj.weight', 'k_proj.bias')):
            value = permute(value, n_kv)
        updated[key] = value
    return updated


def correct_gemma3_norms(state: dict) -> dict:
    """Subtract the 1 llama.cpp adds to Gemma norms when it writes the GGUF.

    Transformers multiplies by ``1 + weight``. A file converted for llama.cpp
    stores that sum, so the weight Hugging Face expects is one less.
    """
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

    updated = {}
    for key, value in state.items():
        if any(marker in key for marker in _GEMMA3_NORM_MARKERS):
            if _is_gguf_parameter(value):
                value = dequantize_gguf_tensor(value)
            value = value.float() - 1.0
        updated[key] = value
    return updated


def _gguf_model_tensor(name: str) -> bool:
    """A tensor the text encoder consumes, rather than a packed tokenizer or config."""
    # ``.weight.1`` is the second temporal slice of a Qwen patch embedding.
    return name.endswith(('.weight', '.bias', '.weight.1', 'layer_scalar', 'pos_embedding'))


def _read_text_encoder_gguf(path: str) -> dict:
    """Read model tensors. Skip sidecar tensors Diffusers cannot decode."""
    import gguf
    from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    from gguf import GGUFReader

    reader = GGUFReader(path)
    plain = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}
    state = {}
    skipped = []
    for tensor in reader.tensors:
        quant_type = tensor.tensor_type
        if quant_type not in plain and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            if _gguf_model_tensor(tensor.name):
                supported = ', '.join(str(int(item)) for item in SUPPORTED_GGUF_QUANT_TYPES)
                raise ValueError(
                    f'{tensor.name} has quantization type {int(quant_type)}, which Diffusers '
                    f'cannot decode. Supported types: {supported}.')
            skipped.append(tensor.name)
            continue
        weights = torch.from_numpy(tensor.data.copy())
        state[tensor.name] = (
            weights if quant_type in plain else GGUFParameter(weights, quant_type=quant_type))
    if skipped:
        _messages.debug_log(
            f'GGUF text encoder "{path}" skipped {len(skipped)} sidecar tensors '
            f'(for example {skipped[0]}).')
    return state


# llama.cpp mmproj names for a Qwen2 / Qwen2.5 vision tower, back to Hugging Face.
# Qwen3-VL keeps a fused qkv and names its merger ``mm.0`` / ``mm.2``, so it does
# not use this map.
_QWEN_VISION_MMPROJ = (
    ('mm.', 'visual.merger.mlp.'),
    ('v.post_ln.', 'visual.merger.ln_q.'),
    ('v.patch_embd', 'visual.patch_embed.proj'),
    ('v.blk.', 'visual.blocks.'),
    ('ffn_up', 'mlp.up_proj'),
    ('ffn_down', 'mlp.down_proj'),
    ('ffn_gate', 'mlp.gate_proj'),
    ('attn_out.', 'attn.proj.'),
    ('ln1.', 'norm1.'),
    ('ln2.', 'norm2.'),
)


def _dense_gguf_tensor(value):
    if _is_gguf_parameter(value):
        from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
        return dequantize_gguf_tensor(value)
    return value


def _qwen3_mmproj(state: dict) -> bool:
    """Qwen3-VL keeps ``attn_qkv`` fused and records DeepStack and a position embedding."""
    return any(
        'attn_qkv' in key or key.startswith('v.deepstack.') or 'position_embd' in key
        for key in state)


def _gemma4_patch_perm(length: int):
    """Inverse of the column order llama.cpp writes for a Gemma 4 patch."""
    if length % 3:
        return None
    side = length // 3
    patch = int(side ** 0.5)
    if patch * patch != side:
        return None
    index = torch.arange(length)
    channel = index // side
    row = (index % side) // patch
    col = index % patch
    perm = row * patch * 3 + col * 3 + channel
    inverse = torch.empty_like(perm)
    inverse[perm] = index
    return inverse


def _restore_gemma4_mmproj(state: dict) -> dict:
    """``gemma4uv`` / ``gemma4ua`` tensors as ``model.embed_vision`` / ``model.embed_audio``."""
    restored = {}
    for key, value in state.items():
        value = _dense_gguf_tensor(value)
        if key == 'mm.a.input_projection.weight':
            restored['model.embed_audio.embedding_projection.weight'] = value
            continue
        if key == 'mm.input_projection.weight':
            restored['model.embed_vision.multimodal_embedder.embedding_projection.weight'] = value
            continue
        if key == 'v.position_embd.weight':
            restored['model.embed_vision.pos_embedding'] = value.permute(1, 0, 2).contiguous()
            continue
        if key == 'v.patch_embd.weight':
            inverse = _gemma4_patch_perm(value.shape[-1])
            restored['model.embed_vision.patch_dense.weight'] = (
                value.index_select(-1, inverse) if inverse is not None else value)
            continue
        if key == 'v.patch_embd.bias':
            restored['model.embed_vision.patch_dense.bias'] = value
            continue
        if key.startswith('v.patch_norm.'):
            rest = key[len('v.patch_norm.'):]
            index, suffix = rest.split('.', 1)
            target = {'1': 'patch_ln1', '2': 'patch_ln2', '3': 'pos_norm'}.get(index)
            if target is None:
                restored['model.embed_vision.' + key] = value
                continue
            if index == '1':
                inverse = _gemma4_patch_perm(value.shape[0])
                if inverse is not None:
                    value = value.index_select(0, inverse)
            restored[f'model.embed_vision.{target}.{suffix}'] = value
            continue
        restored['model.embed_vision.' + key] = value
    return restored


def _gemma4_unified_mmproj(state: dict) -> bool:
    """LTX Gemma 4: the encoder-free embedder, named ``v.patch_norm``."""
    return any(key.startswith('v.patch_norm.') for key in state)


# ggml block stem -> (Hugging Face suffix, wrapped in Gemma4ClippableLinear).
_GEMMA4V_VISION = {
    'attn_q': ('self_attn.q_proj', True),
    'attn_k': ('self_attn.k_proj', True),
    'attn_v': ('self_attn.v_proj', True),
    'attn_out': ('self_attn.o_proj', True),
    'attn_q_norm': ('self_attn.q_norm', False),
    'attn_k_norm': ('self_attn.k_norm', False),
    'ln1': ('input_layernorm', False),
    'attn_post_norm': ('post_attention_layernorm', False),
    'ln2': ('pre_feedforward_layernorm', False),
    'ffn_post_norm': ('post_feedforward_layernorm', False),
    'ffn_gate': ('mlp.gate_proj', True),
    'ffn_up': ('mlp.up_proj', True),
    'ffn_down': ('mlp.down_proj', True),
    'out_scale': ('layer_scalar', False),
}

_GEMMA4V_AUDIO = {
    'attn_q': ('self_attn.q_proj', True),
    'attn_k': ('self_attn.k_proj', True),
    'attn_v': ('self_attn.v_proj', True),
    'attn_out': ('self_attn.post', True),
    'attn_k_rel': ('self_attn.relative_k_proj', False),
    'attn_pre_norm': ('norm_pre_attn', False),
    'attn_post_norm': ('norm_post_attn', False),
    'ln2': ('norm_out', False),
    'ffn_norm': ('feed_forward1.pre_layer_norm', False),
    'ffn_post_norm': ('feed_forward1.post_layer_norm', False),
    'ffn_up': ('feed_forward1.ffw_layer_1', True),
    'ffn_down': ('feed_forward1.ffw_layer_2', True),
    'ffn_norm_1': ('feed_forward2.pre_layer_norm', False),
    'ffn_post_norm_1': ('feed_forward2.post_layer_norm', False),
    'ffn_up_1': ('feed_forward2.ffw_layer_1', True),
    'ffn_down_1': ('feed_forward2.ffw_layer_2', True),
    'conv_norm': ('lconv1d.pre_layer_norm', False),
    'norm_conv': ('lconv1d.conv_norm', False),
    'conv_pw1': ('lconv1d.linear_start', True),
    'conv_pw2': ('lconv1d.linear_end', True),
}

_GEMMA4_CLAMP = ('input_min', 'input_max', 'output_min', 'output_max')


def _gemma4v_mmproj(state: dict) -> bool:
    """Full Gemma 4 tower (Unsloth ``mmproj``): ``gemma4v`` / ``gemma4a``."""
    return any(
        key.startswith(('a.', 'mm.a.', 'mm.input_projection'))
        or '.attn_post_norm.' in key or '.ffn_post_norm.' in key or '.attn_q_norm.' in key
        for key in state)


def _gemma4_tail(tail: str):
    for suffix in ('weight', 'bias', *_GEMMA4_CLAMP):
        marker = '.' + suffix
        if tail.endswith(marker):
            return tail[:-len(marker)], suffix
    return tail, None


def _inverse_softplus(value):
    value = value.to(dtype=torch.float32)
    return torch.where(value > 20, value, torch.log(torch.expm1(value.clamp_min(1e-6))))


def _gemma4_place(prefix: str, index: str, table: dict, tail: str, value):
    stem, suffix = _gemma4_tail(tail)
    found = table.get(stem)
    if found is None or suffix is None:
        return None
    path, clippable = found
    target = f'{prefix}{index}.{path}'
    if stem == 'out_scale':
        return target, value.reshape(())
    if clippable and suffix == 'weight':
        return target + '.linear.weight', value
    if clippable and suffix in _GEMMA4_CLAMP:
        return target + '.' + suffix, value.reshape(())
    if clippable and suffix == 'bias':
        return target + '.linear.bias', value
    return target + '.' + suffix, value


def _restore_gemma4v_mmproj(state: dict) -> dict:
    """``gemma4v`` / ``gemma4a`` tensors as the Gemma 4 vision and audio towers."""
    restored = {}
    for key, value in state.items():
        value = _dense_gguf_tensor(value)
        if key == 'v.patch_embd.weight' and value.ndim == 4:
            out, channels, height, width = value.shape
            value = value.permute(0, 2, 3, 1).contiguous().reshape(out, channels * height * width)
            restored['model.vision_tower.patch_embedder.input_proj.weight'] = value
            continue
        if key == 'v.position_embd.weight':
            restored['model.vision_tower.patch_embedder.position_embedding_table'] = value
            continue
        if key in ('v.std_bias', 'v.std_bias.weight'):
            restored['model.vision_tower.std_bias'] = value.reshape(-1)
            continue
        if key in ('v.std_scale', 'v.std_scale.weight'):
            restored['model.vision_tower.std_scale'] = value.reshape(-1)
            continue
        if key == 'mm.input_projection.weight':
            restored['model.embed_vision.embedding_projection.weight'] = value
            continue
        if key == 'mm.a.input_projection.weight':
            restored['model.embed_audio.embedding_projection.weight'] = value
            continue
        if key == 'a.input_projection.weight':
            restored['model.audio_tower.subsample_conv_projection.input_proj_linear.weight'] = value
            continue
        if key in ('a.pre_encode.out.weight', 'a.pre_encode.out.bias'):
            suffix = key.rsplit('.', 1)[1]
            restored['model.audio_tower.output_proj.' + suffix] = value
            continue
        if key.startswith('a.conv1d.') and '.norm.' not in key:
            layer = key.split('.')[2]
            restored[f'model.audio_tower.subsample_conv_projection.layer{layer}.conv.weight'] = value
            continue
        if key.startswith('a.conv1d.') and key.endswith('.norm.weight'):
            layer = key.split('.')[2]
            restored[f'model.audio_tower.subsample_conv_projection.layer{layer}.norm.weight'] = value
            continue
        if key.startswith('v.blk.'):
            index, tail = key[len('v.blk.'):].split('.', 1)
            placed = _gemma4_place('model.vision_tower.encoder.layers.', index, _GEMMA4V_VISION, tail, value)
            if placed is not None:
                restored[placed[0]] = placed[1]
            continue
        if key.startswith('a.blk.'):
            index, tail = key[len('a.blk.'):].split('.', 1)
            if tail == 'conv_dw.weight':
                restored[f'model.audio_tower.layers.{index}.lconv1d.depthwise_conv1d.weight'] = value.unsqueeze(1)
                continue
            if tail == 'per_dim_scale.weight':
                restored[f'model.audio_tower.layers.{index}.self_attn.per_dim_scale'] = _inverse_softplus(value)
                continue
            placed = _gemma4_place('model.audio_tower.layers.', index, _GEMMA4V_AUDIO, tail, value)
            if placed is not None:
                restored[placed[0]] = placed[1]
    return restored


def restore_mmproj(state: dict, deepstack_indexes=None) -> dict:
    """
    llama.cpp mmproj tensors as Hugging Face vision names (``visual.*``).

    A Qwen2 / Qwen2.5 file splits qkv and the temporal patch embedding. A
    Qwen3-VL file keeps qkv fused. Gemma 4 unified vision and audio use the
    ``gemma4uv`` / ``gemma4ua`` names. A full Gemma 4 tower (Unsloth ``mmproj``)
    uses ``gemma4v`` / ``gemma4a``. ``deepstack_indexes`` are the vision-block
    indexes stored in a Qwen3-VL file; the model lists those mergers in that
    order. A file that already uses ``visual.*`` names is returned unchanged.
    """
    if any(key.startswith('visual.') or '.visual.' in key for key in state):
        return dict(state)
    if _gemma4_unified_mmproj(state):
        return _restore_gemma4_mmproj(state)
    if _gemma4v_mmproj(state):
        return _restore_gemma4v_mmproj(state)
    if _qwen3_mmproj(state):
        return _restore_qwen3_mmproj(state, deepstack_indexes)
    return _restore_qwen_vision_mmproj(state)


def _restore_qwen_vision_mmproj(state: dict) -> dict:
    renamed = {}
    for key, value in state.items():
        for source, target in _QWEN_VISION_MMPROJ:
            key = key.replace(source, target)
        renamed[key] = value
    patch = renamed.get('visual.patch_embed.proj.weight')
    patch_next = renamed.get('visual.patch_embed.proj.weight.1')
    if patch is not None and patch_next is not None:
        renamed['visual.patch_embed.proj.weight'] = torch.stack([
            _dense_gguf_tensor(renamed.pop('visual.patch_embed.proj.weight')),
            _dense_gguf_tensor(renamed.pop('visual.patch_embed.proj.weight.1')),
        ], dim=2)
    fused = {}
    consumed = []
    for key in list(renamed):
        marker = '.attn_q.'
        if marker not in key:
            continue
        stem, suffix = key.split(marker, 1)
        key_k = stem + '.attn_k.' + suffix
        key_v = stem + '.attn_v.' + suffix
        if key_k not in renamed or key_v not in renamed:
            continue
        fused[stem + '.attn.qkv.' + suffix] = torch.cat([
            _dense_gguf_tensor(renamed[key]),
            _dense_gguf_tensor(renamed[key_k]),
            _dense_gguf_tensor(renamed[key_v]),
        ], dim=0)
        consumed.extend((key, key_k, key_v))
    for key in consumed:
        renamed.pop(key, None)
    renamed.update(fused)
    return renamed


def _restore_qwen3_mmproj(state: dict, deepstack_indexes) -> dict:
    indexes = list(deepstack_indexes or [])
    restored = {}
    patch = {}
    for key, value in state.items():
        if key in ('v.patch_embd.weight', 'v.patch_embd.weight.1', 'v.patch_embd.bias'):
            patch[key] = value
            continue
        if key.startswith('v.deepstack.'):
            rest = key[len('v.deepstack.'):]
            index, tail = rest.split('.', 1)
            absolute = int(index)
            position = indexes.index(absolute) if absolute in indexes else absolute
            tail = tail.replace('fc1.', 'linear_fc1.').replace('fc2.', 'linear_fc2.')
            restored[f'visual.deepstack_merger_list.{position}.{tail}'] = value
            continue
        if key.startswith('mm.0.'):
            restored['visual.merger.linear_fc1.' + key[len('mm.0.'):]] = value
            continue
        if key.startswith('mm.2.'):
            restored['visual.merger.linear_fc2.' + key[len('mm.2.'):]] = value
            continue
        if key.startswith('v.post_ln.'):
            restored['visual.merger.norm.' + key[len('v.post_ln.'):]] = value
            continue
        if key == 'v.position_embd.weight':
            restored['visual.pos_embed.weight'] = value
            continue
        name = key.replace('v.blk.', 'visual.blocks.')
        name = name.replace('.ffn_up.', '.mlp.linear_fc1.')
        name = name.replace('.ffn_down.', '.mlp.linear_fc2.')
        name = name.replace('.attn_qkv.', '.attn.qkv.')
        name = name.replace('attn_out.', 'attn.proj.')
        name = name.replace('ln1.', 'norm1.')
        name = name.replace('ln2.', 'norm2.')
        restored[name] = value
    if 'v.patch_embd.weight' in patch and 'v.patch_embd.weight.1' in patch:
        restored['visual.patch_embed.proj.weight'] = torch.stack([
            _dense_gguf_tensor(patch['v.patch_embd.weight']),
            _dense_gguf_tensor(patch['v.patch_embd.weight.1']),
        ], dim=2)
    elif 'v.patch_embd.weight' in patch:
        restored['visual.patch_embed.proj.weight'] = patch['v.patch_embd.weight']
    if 'v.patch_embd.bias' in patch:
        restored['visual.patch_embed.proj.bias'] = patch['v.patch_embd.bias']
    return restored


def load_mmproj_gguf(path: str, deepstack_indexes=None) -> dict:
    """
    Read a llama.cpp mmproj GGUF as Hugging Face vision tensors.

    Names start with ``visual.``. The text-encoder loader attaches the
    ``model.`` prefix when the encoder class keeps the tower there.

    :raises ValueError: The file has no vision tensors, or a model weight uses
        a quantization type Diffusers cannot decode.
    """
    state = _read_text_encoder_gguf(path)
    if not state:
        raise ValueError(f'GGUF file "{path}" has no vision tensors.')
    restored = restore_mmproj(state, deepstack_indexes)
    _messages.debug_log(f'GGUF mmproj "{path}": {len(restored)} vision tensors.')
    return restored


def _restore_gemma4_names(state: dict) -> dict:
    """Gemma 4 tensors that are not in the Gemma 3 map, back to Hugging Face names.

    ``token_embd`` is a substring of ``per_layer_token_embd``, so these names
    are restored before that replacement. ``rope_freqs`` is synthesized for
    llama.cpp and is not a checkpoint tensor.
    """
    restored = {}
    for key, value in state.items():
        if key == 'rope_freqs.weight' or key.startswith('rope_freqs.'):
            continue
        if key.startswith('per_layer_token_embd.'):
            name = 'model.embed_tokens_per_layer.' + key[len('per_layer_token_embd.'):]
        elif key.startswith('per_layer_model_proj.'):
            name = 'model.per_layer_model_projection.' + key[len('per_layer_model_proj.'):]
        elif key.startswith('per_layer_proj_norm.'):
            name = 'model.per_layer_projection_norm.' + key[len('per_layer_proj_norm.'):]
        else:
            name = key
            name = name.replace('inp_gate', 'per_layer_input_gate')
            name = name.replace('post_norm', 'post_per_layer_input_norm')
            name = name.replace('.proj.', '.per_layer_projection.')
            if name.endswith('layer_output_scale.weight'):
                name = name[:-len('layer_output_scale.weight')] + 'layer_scalar'
        restored[name] = value
    return restored


def _qwen35_heads(text: dict | None):
    """``(num_k, num_v, head_k, head_v)`` for the gated delta net, or ``None``."""
    if not isinstance(text, dict):
        return None
    values = [text.get(name) for name in (
        'linear_num_key_heads', 'linear_num_value_heads',
        'linear_key_head_dim', 'linear_value_head_dim')]
    if not all(isinstance(item, int) and item > 0 for item in values):
        return None
    num_k, num_v, head_k, head_v = values
    if num_v % num_k:
        return None
    return num_k, num_v, head_k, head_v


def _reorder_v_heads(tensor, dim: int, num_k: int, num_v_per_k: int, head_dim: int):
    """V heads from grouped-by-K order to the tiled order llama.cpp broadcasts."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    view = shape[:dim] + [num_k, num_v_per_k, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*view)
    order = list(range(len(view)))
    order[dim], order[dim + 1] = order[dim + 1], order[dim]
    return tensor.permute(*order).contiguous().reshape(*shape)


def _qwen35_dense(value):
    if getattr(value, 'quant_type', None) is None:
        return value
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
    return dequantize_gguf_tensor(value)


_QWEN35_NAME_PAIRS = (
    ('.linear_attn.in_proj_qkv.', '.attn_qkv.'),
    ('.linear_attn.in_proj_z.', '.attn_gate.'),
    ('.linear_attn.in_proj_a.', '.ssm_alpha.'),
    ('.linear_attn.in_proj_b.', '.ssm_beta.'),
    ('.linear_attn.out_proj.', '.ssm_out.'),
    ('.linear_attn.conv1d.', '.ssm_conv1d.'),
    ('.linear_attn.norm.', '.ssm_norm.'),
    ('.linear_attn.dt_bias', '.ssm_dt.bias'),
    ('.linear_attn.A_log', '.ssm_a'),
)


def _qwen35_forward_value(key: str, value, heads):
    import torch

    num_k, num_v, head_k, head_v = heads
    per = num_v // num_k
    reorder = num_k != num_v
    if key.endswith('.linear_attn.A_log'):
        if reorder and value.ndim == 1:
            value = _reorder_v_heads(value.unsqueeze(-1), 0, num_k, per, 1).squeeze(-1)
        return -torch.exp(value.float())
    if key.endswith('.linear_attn.dt_bias'):
        if reorder and value.ndim == 1:
            value = _reorder_v_heads(value.unsqueeze(-1), 0, num_k, per, 1).squeeze(-1)
        return value
    if key.endswith('.linear_attn.conv1d.weight'):
        if value.ndim == 3 and value.shape[1] == 1:
            value = value.squeeze(1)
        if reorder and value.ndim >= 1:
            qk = head_k * num_k * 2
            kept, rest = value[:qk], value[qk:]
            rest = _reorder_v_heads(rest, 0, num_k, per, head_v)
            value = torch.cat((kept, rest), dim=0)
        return value
    if not reorder:
        return value
    if key.endswith('.linear_attn.in_proj_qkv.weight'):
        width = head_k * num_k
        query, key_rows, value_rows = value[:width], value[width:width * 2], value[width * 2:]
        value_rows = _reorder_v_heads(value_rows, 0, num_k, per, head_v)
        return torch.cat((query, key_rows, value_rows), dim=0)
    if key.endswith('.linear_attn.in_proj_z.weight'):
        return _reorder_v_heads(value, 0, num_k, per, head_v)
    if key.endswith(('.linear_attn.in_proj_a.weight', '.linear_attn.in_proj_b.weight')):
        return _reorder_v_heads(value, 0, num_k, per, 1)
    if key.endswith('.linear_attn.out_proj.weight'):
        return _reorder_v_heads(value, 1, num_k, per, head_v)
    return value


def qwen35_to_llama(state: dict, text: dict | None) -> dict:
    """llama.cpp qwen35 names and the gated-delta layout its loader multiplies."""
    heads = _qwen35_heads(text)
    renamed = {}
    for key, value in state.items():
        if '.linear_attn.' in key:
            if heads is None:
                raise ValueError(
                    'A Qwen3.5 linear attention tensor needs linear_num_key_heads, '
                    'linear_num_value_heads, and the linear head sizes in config.json.')
            value = _qwen35_forward_value(key, value, heads)
        name = key.replace('ffn_norm', 'post_attention_norm')
        for source, target in _QWEN35_NAME_PAIRS:
            if source in name:
                name = name.replace(source, target)
                break
        if '.linear_attn.' in name:
            raise ValueError(f'Unexpected Qwen3.5 linear attention tensor "{key}".')
        renamed[name] = value
    return renamed


def _qwen35_inverse_value(key: str, value, heads):
    import torch

    value = _qwen35_dense(value)
    if heads is None:
        if key.endswith('.ssm_a'):
            return torch.log(-value.float())
        if key.endswith('.ssm_conv1d.weight') and value.ndim == 2:
            return value.unsqueeze(1)
        return value
    num_k, num_v, head_k, head_v = heads
    per = num_v // num_k
    reorder = num_k != num_v
    if key.endswith('.ssm_a'):
        value = torch.log(-value.float())
        if reorder and value.ndim == 1:
            value = _reorder_v_heads(value.unsqueeze(-1), 0, num_k, per, 1).squeeze(-1)
        return value
    if key.endswith('.ssm_dt.bias'):
        if reorder and value.ndim == 1:
            value = _reorder_v_heads(value.unsqueeze(-1), 0, num_k, per, 1).squeeze(-1)
        return value
    if key.endswith('.ssm_conv1d.weight'):
        if reorder and value.ndim >= 1:
            qk = head_k * num_k * 2
            kept, rest = value[:qk], value[qk:]
            rest = _reorder_v_heads(rest, 0, num_k, per, head_v)
            value = torch.cat((kept, rest), dim=0)
        if value.ndim == 2:
            value = value.unsqueeze(1)
        return value
    if not reorder:
        return value
    if key.endswith('.attn_qkv.weight') and 'ssm' not in key:
        # Only the linear stack's fused projection is attn_qkv. Full attention stays split.
        width = head_k * num_k
        if value.shape[0] == width * 2 + head_v * num_v:
            query, key_rows, value_rows = value[:width], value[width:width * 2], value[width * 2:]
            value_rows = _reorder_v_heads(value_rows, 0, num_k, per, head_v)
            return torch.cat((query, key_rows, value_rows), dim=0)
        return value
    if key.endswith('.attn_gate.weight'):
        return _reorder_v_heads(value, 0, num_k, per, head_v)
    if key.endswith(('.ssm_alpha.weight', '.ssm_beta.weight')):
        return _reorder_v_heads(value, 0, num_k, per, 1)
    if key.endswith('.ssm_out.weight'):
        return _reorder_v_heads(value, 1, num_k, per, head_v)
    return value


def _qwen35_heads_from_gguf(path: str, architecture: str):
    import gguf

    reader = gguf.GGUFReader(path)

    def number(suffix):
        field = reader.get_field(f'{architecture}.{suffix}')
        if field is None or not field.types:
            return None
        raw = field.parts[field.data[-1]]
        try:
            return int(raw)
        except TypeError:
            return int(raw.reshape(-1)[0])

    num_k = number('ssm.group_count')
    num_v = number('ssm.time_step_rank')
    head = number('ssm.state_size')
    if not all(isinstance(item, int) and item > 0 for item in (num_k, num_v, head)):
        return None
    if num_v % num_k:
        return None
    return num_k, num_v, head, head


def qwen35_from_llama(state: dict, heads=None) -> dict:
    """Undo :func:`qwen35_to_llama` before the ordinary Qwen name restore.

    RMSNorm weights written for llama.cpp already include the ``+ 1`` the
    runtime multiplies. The linear-attention norm does not.
    """
    restored = {}
    for key, value in state.items():
        if any(marker in key for marker in (
                '.ssm_', '.attn_qkv.', '.attn_gate.', '.ssm_a', '.ssm_dt.')):
            value = _qwen35_inverse_value(key, value, heads)
        elif key.endswith('norm.weight') and '.ssm_norm.' not in key:
            value = _qwen35_dense(value).float() - 1
        name = key.replace('post_attention_norm', 'ffn_norm')
        for source, target in _QWEN35_NAME_PAIRS:
            if target in name:
                name = name.replace(target, source)
                break
        restored[name] = value
    return restored


def load_text_encoder_gguf(path: str) -> tuple[str, dict, bool]:
    """
    Read a text-encoder GGUF and rename llama.cpp tensors to Hugging Face names.

    Packed weights stay :py:class:`diffusers.quantizers.gguf.utils.GGUFParameter`.
    Float embeddings, norms, and biases stay dense tensors. A file that already
    stores Hugging Face or ComfyUI names, including the LTX-2.5 Gemma 4 GGUF,
    is left as-is. Gemma 3 norms written by llama.cpp are converted back to the
    Hugging Face values. Gemma 4 does not add one, so those norms stay as stored.
    Sidecar tensors such as a packed tokenizer are skipped when Diffusers cannot
    decode their type.

    :return: ``(architecture, state dict, llama.cpp names)``. The third value
        is true when tensor names were llama.cpp names. Llama and Mistral
        query and key rows are interleaved only in that layout.
    :raises ValueError: The file is not a text-encoder GGUF, or a model weight
        uses a quantization type Diffusers cannot decode.
    """
    architecture = _gguf_architecture(path)
    state = None
    # CLIP GGUFs written before this reader existed are labeled ``diffusion``
    # and still store the CLIP module names.
    if architecture == 'diffusion':
        state = _read_text_encoder_gguf(path)
        if _clip_text_keys(state):
            architecture = 'clip'
    if architecture in {'clip_l', 'clip_g'}:
        architecture = 'clip'
    if architecture not in _TEXT_GGUF_ARCHITECTURES:
        raise ValueError(
            f'GGUF file "{path}" has architecture {architecture!r}, which is not a text encoder. '
            f'Text encoder files use one of: {", ".join(sorted(_TEXT_GGUF_ARCHITECTURES))}.')

    if state is None:
        state = _read_text_encoder_gguf(path)
    if architecture in {'t5', 't5encoder'}:
        key_map = _T5_GGUF_KEYS
    elif architecture in {'gemma3', 'gemma4'}:
        key_map = _GEMMA3_GGUF_KEYS
    else:
        key_map = _LLAMA_GGUF_KEYS

    converted = not _huggingface_names(state, architecture)
    if converted:
        if architecture == 'gemma4':
            state = _restore_gemma4_names(state)
        if architecture in {'qwen35', 'qwen35moe'}:
            state = qwen35_from_llama(state, _qwen35_heads_from_gguf(path, architecture))
        state = _remap_keys(state, key_map)
    if architecture == 'gemma3' and converted:
        state = correct_gemma3_norms(state)

    _messages.debug_log(
        f'GGUF text encoder "{path}" architecture {architecture}'
        + (' (llama.cpp names)' if converted else ' (Hugging Face names)')
        + f', {len(state)} tensors.')
    return architecture, state, converted
