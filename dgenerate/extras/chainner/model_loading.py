import safetensors
import torch

import dgenerate.extras.chainner.checkpoint_pickle as checkpoint_pickle
import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import dgenerate.types as _types
from .architecture.DAT import DAT
from .architecture.HAT import HAT
from .architecture.LaMa import LaMa
from .architecture.OmniSR.OmniSR import OmniSR
from .architecture.RRDB import RRDBNet as ESRGAN
from .architecture.SCUNet import SCUNet
from .architecture.SPSR import SPSRNet as SPSR
from .architecture.SRVGG import SRVGGNetCompact as RealESRGANv2
from .architecture.SwiftSRGAN import Generator as SwiftSRGAN
from .architecture.Swin2SR import Swin2SR
from .architecture.SwinIR import SwinIR
from .architecture.face.codeformer import CodeFormer
from .architecture.face.gfpganv1_clean_arch import GFPGANv1Clean
from .architecture.face.restoreformer_arch import RestoreFormer
from .types import PyTorchModel


class UnsupportedModelError(Exception):
    """chaiNNer model is not of a supported type."""
    pass


def _load_state_dict(state_dict) -> PyTorchModel:
    state_dict_keys = list(state_dict.keys())

    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    elif "params-ema" in state_dict_keys:
        state_dict = state_dict["params-ema"]
    elif "params" in state_dict_keys:
        state_dict = state_dict["params"]

    state_dict_keys = list(state_dict.keys())
    # SRVGGNet Real-ESRGAN (v2)
    if "body.0.weight" in state_dict_keys and "body.1.weight" in state_dict_keys:
        model = RealESRGANv2(state_dict)
    # SPSR (ESRGAN with lots of extra layers)
    elif "f_HR_conv1.0.weight" in state_dict:
        model = SPSR(state_dict)
    # Swift-SRGAN
    elif (
            "model" in state_dict_keys
            and "initial.cnn.depthwise.weight" in state_dict["model"].keys()
    ):
        model = SwiftSRGAN(state_dict)
    # SwinIR, Swin2SR, HAT
    elif "layers.0.residual_group.blocks.0.norm1.weight" in state_dict_keys:
        if (
                "layers.0.residual_group.blocks.0.conv_block.cab.0.weight"
                in state_dict_keys
        ):
            model = HAT(state_dict)
        elif "patch_embed.proj.weight" in state_dict_keys:
            model = Swin2SR(state_dict)
        else:
            model = SwinIR(state_dict)
    # GFPGAN
    elif (
            "toRGB.0.weight" in state_dict_keys
            and "stylegan_decoder.style_mlp.1.weight" in state_dict_keys
    ):
        model = GFPGANv1Clean(state_dict)
    # RestoreFormer
    elif (
            "encoder.conv_in.weight" in state_dict_keys
            and "encoder.down.0.block.0.norm1.weight" in state_dict_keys
    ):
        model = RestoreFormer(state_dict)
    elif (
            "encoder.blocks.0.weight" in state_dict_keys
            and "quantize.embedding.weight" in state_dict_keys
    ):
        model = CodeFormer(state_dict)
    # LaMa
    elif (
            "model.model.1.bn_l.running_mean" in state_dict_keys
            or "generator.model.1.bn_l.running_mean" in state_dict_keys
    ):
        model = LaMa(state_dict)
    # Omni-SR
    elif "residual_layer.0.residual_layer.0.layer.0.fn.0.weight" in state_dict_keys:
        model = OmniSR(state_dict)
    # SCUNet
    elif "m_head.0.weight" in state_dict_keys and "m_tail.0.weight" in state_dict_keys:
        model = SCUNet(state_dict)
    # DAT
    elif "layers.0.blocks.2.attn.attn_mask_0" in state_dict_keys:
        model = DAT(state_dict)
    # Regular ESRGAN, "new-arch" ESRGAN, Real-ESRGAN v1
    else:
        try:
            model = ESRGAN(state_dict)
        except:
            # pylint: disable=raise-missing-from
            raise UnsupportedModelError
    return model


def _load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if 'weights_only' not in torch.load.__code__.co_varnames:
                _messages.log(
                    "torch.load doesn't support weights_only on this pytorch version, loading unsafely.",
                    level=_messages.WARNING)
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=checkpoint_pickle)
        if "global_step" in pl_sd:
            _messages.debug_log(f'dgenerate.extras.chainner._load_torch_file(): '
                                f'Global Step: {pl_sd["global_step"]}')
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


def _state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])),
                           filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def load_model(model_path) -> PyTorchModel:
    """
    Load an upscaler model from a file path or URL.

    :param model_path: path
    :return: model
    """
    if _mediainput.is_downloadable_url(model_path):
        # Any mimetype
        _, model_path = _mediainput.create_web_cache_file(
            model_path, mimetype_is_supported=None)

    state_dict = _load_torch_file(model_path, safe_load=True)

    if "module.layers.0.residual_group.blocks.0.norm1.weight" in state_dict:
        state_dict = _state_dict_prefix_replace(state_dict, {"module.": ""})

    model = _load_state_dict(state_dict)

    _messages.debug_log(
        f'{_types.fullname(load_model)}("{model_path}") -> {model.__class__.__name__}')

    return model.eval()
