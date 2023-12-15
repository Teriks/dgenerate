import safetensors
import spandrel
import torch
import dgenerate.messages as _messages
import dgenerate.mediainput as _mediainput
import dgenerate.types as _types
from spandrel import ModelLoader, ImageModelDescriptor
from . import checkpoint_pickle


class UnsupportedModelError(Exception):
    """chaiNNer model is not of a supported type."""
    pass


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


def load_upscaler_model(model_path) -> spandrel.ImageModelDescriptor:
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

    model = ModelLoader().load_from_state_dict(state_dict).eval()

    if not isinstance(model, ImageModelDescriptor):
        raise UnsupportedModelError("Upscale model must be a single-image model.")

    _messages.debug_log(
        f'{_types.fullname(load_upscaler_model)}("{model_path}") -> {model.__class__.__name__}')

    return model
