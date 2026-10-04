# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Detect Flux.2, Qwen-Image, Z-Image, LTX 2.5, and Wan GGUF checkpoints and
adapt ComfyUI layouts so Diffusers can load them without a manual config.
"""

import contextlib
import math
import os

import torch

import dgenerate.messages as _messages

_COMFY_PREFIX = 'model.diffusion_model.'

# img_in / qkv hidden width -> public Diffusers transformer repo.
# 4B and 9B Klein share Flux.2 key names with dev, so the width is the
# only thing that stops Diffusers from building the 32B dev module.
_FLUX2_BY_HIDDEN = {
    3072: 'black-forest-labs/FLUX.2-klein-4B',
    4096: 'black-forest-labs/FLUX.2-klein-base-9B',
    6144: 'black-forest-labs/FLUX.2-dev',
}

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

    if _has(shapes, 'keyframes_abs_pos_embedding', 'audio_patchify_proj.weight',
            _COMFY_PREFIX + 'keyframes_abs_pos_embedding',
            _COMFY_PREFIX + 'audio_patchify_proj.weight'):
        return GGUFLayout(
            'ltx-2.5', 'Lightricks/LTX-2.5-Diffusers', comfy=comfy, adapt=True)

    if _has(shapes, 'cap_embedder.0.weight', 'layers.0.adaLN_modulation.0.weight',
            _COMFY_PREFIX + 'cap_embedder.0.weight',
            _COMFY_PREFIX + 'layers.0.adaLN_modulation.0.weight'):
        return GGUFLayout(
            'z-image-turbo', 'Tongyi-MAI/Z-Image-Turbo', comfy=comfy, adapt=False)

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

    wan = _detect_wan_layout(shapes, comfy)
    if wan is not None:
        return wan

    return None


def _detect_wan_layout(shapes: dict, comfy: bool) -> GGUFLayout | None:
    if not _has(shapes, 'head.modulation', 'patch_embedding.weight',
                _COMFY_PREFIX + 'head.modulation',
                _COMFY_PREFIX + 'patch_embedding.weight'):
        return None
    if _has(shapes, 'vace_blocks.0.after_proj.bias',
            _COMFY_PREFIX + 'vace_blocks.0.after_proj.bias'):
        patch = _lookup_shape(
            shapes, 'patch_embedding.weight',
            _COMFY_PREFIX + 'patch_embedding.weight')
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
    patch = _lookup_shape(
        shapes, 'patch_embedding.weight',
        _COMFY_PREFIX + 'patch_embedding.weight')
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
        return GGUFLayout(
            'wan-i2v-14B', 'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers',
            comfy=comfy, adapt=False)
    if width == 5120:
        return GGUFLayout(
            'wan-t2v-14B', 'Wan-AI/Wan2.1-T2V-14B-Diffusers',
            comfy=comfy, adapt=False)
    return None


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


def _strip_comfy_prefix(checkpoint: dict) -> dict:
    if not any(key.startswith(_COMFY_PREFIX) for key in checkpoint):
        return checkpoint
    renamed = {}
    for key, value in checkpoint.items():
        new_key = key[len(_COMFY_PREFIX):] if key.startswith(_COMFY_PREFIX) else key
        renamed[new_key] = value
    return renamed


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
        if (len(quant_shape) == 2 and len(target_shape) == 2
                and quant_shape == tuple(reversed(target_shape))):
            param._dgenerate_comfy_transpose = True
        return param

    current = _shape(param.shape)
    if current == target_shape:
        return param
    if param.numel() != math.prod(target_shape):
        return param
    if param.ndim == 1 and target_shape == (1, current[0]):
        return param.reshape(target_shape)
    if param.ndim == 2 and target_shape == tuple(reversed(current)):
        return param.transpose(0, 1).contiguous()
    return param


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
    # Comfy keeps the original ``*_adaln_single`` prefix. Diffusers drops
    # ``_single`` for the prompt embeddings and maps the main ones itself.
    renamed = {}
    for key, value in checkpoint.items():
        new_key = key.replace('audio_prompt_adaln_single.', 'audio_prompt_adaln.')
        new_key = new_key.replace('prompt_adaln_single.', 'prompt_adaln.')
        renamed[new_key] = value
    converted = convert_ltx2_transformer_to_diffusers(renamed)
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
            f'LTX 2.5 GGUF is missing {len(missing)} Diffusers tensors, '
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


def adapt_wan_flf_checkpoint(checkpoint: dict) -> dict:
    """Map ``img_emb.emb_pos`` onto the Diffusers first-last image embedding."""
    checkpoint = _strip_comfy_prefix(dict(checkpoint))
    if _wan_checkpoint_needs_convert(checkpoint):
        from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
        checkpoint = convert_wan_transformer_to_diffusers(checkpoint)
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


def adapt_qwen_checkpoint(checkpoint: dict) -> dict:
    """Drop the Comfy ``model.diffusion_model.`` prefix. Names are otherwise Diffusers names."""
    return _strip_comfy_prefix(checkpoint)


_QWEN_LAYERED_ADDITION_T = 'time_text_embed.addition_t_embedding.weight'


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


def install_gguf_patches() -> None:
    """
    Let a Comfy quantized linear whose packed shape is transposed load
    into a Diffusers linear and transpose after dequantization.
    """
    global _patches_installed
    if _patches_installed:
        return

    import diffusers.loaders.single_file_model as single_file_model
    import diffusers.quantizers.gguf.gguf_quantizer as gguf_quantizer
    import diffusers.quantizers.gguf.utils as gguf_utils

    original_should_convert = single_file_model._should_convert_state_dict_to_diffusers
    original_check = gguf_quantizer.GGUFQuantizer.check_quantized_param_shape
    original_forward = gguf_utils.GGUFLinear.forward_native
    original_new = gguf_utils.GGUFParameter.__new__
    original_torch_function = gguf_utils.GGUFParameter.__torch_function__

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

    def forward_native(self, inputs):
        if not getattr(self.weight, '_dgenerate_comfy_transpose', False):
            return original_forward(self, inputs)
        weight = gguf_utils.dequantize_gguf_tensor(self.weight)
        weight = weight.transpose(0, 1).contiguous().to(self.compute_dtype)
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
        return torch.nn.functional.linear(inputs, weight, bias)

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
    gguf_utils.GGUFLinear.forward_native = forward_native
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

    _messages.debug_log(
        f'GGUF transformer detected as {layout.kind}'
        + (' (Comfy layout)' if layout.comfy else '')
        + f'. Using {layout.config_repo}.')

    if config is None:
        config = layout.config_repo
        subfolder = subfolder or layout.subfolder
    elif subfolder is None:
        subfolder = layout.subfolder

    if not layout.adapt:
        return None, config, subfolder, False

    from diffusers.models.model_loading_utils import load_gguf_checkpoint
    install_gguf_patches()
    checkpoint = load_gguf_checkpoint(path)
    if layout.kind == 'ltx-2.5':
        checkpoint = adapt_ltx25_checkpoint(
            checkpoint, config, subfolder or 'transformer',
            token=token, local_files_only=local_files_only)
        return checkpoint, config, subfolder, True
    if layout.kind == 'qwen-image':
        return adapt_qwen_checkpoint(checkpoint), config, subfolder, False
    if layout.kind == 'qwen-image-layered':
        return adapt_qwen_layered_checkpoint(checkpoint), config, subfolder, False
    if layout.kind == 'wan-animate-14B':
        return adapt_wan_animate_checkpoint(checkpoint), config, subfolder, False
    if layout.kind == 'wan-flf2v-14B':
        return adapt_wan_flf_checkpoint(checkpoint), config, subfolder, True
    return None, config, subfolder, False
