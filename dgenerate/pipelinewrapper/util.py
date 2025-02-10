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
import re
import dgenerate.types as _types
from contextlib import contextmanager

import diffusers.loaders.single_file as _single_file
import diffusers.pipelines
import huggingface_hub
import huggingface_hub.errors
import torch

import dgenerate.pipelinewrapper.uris


class InvalidDeviceOrdinalException(Exception):
    """
    GPU in device specification (cuda:N) does not exist
    """
    pass


def default_device() -> str:
    """
    Return a string representing the systems default accelerator device.

    Possible Values:

        * ``"cuda"``
        * ``"mps"``
        * ``"cpu"``

    :return: ``"cuda"``, ``"mps"``, etc.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def is_valid_device_string(device, raise_ordinal=True):
    """
    Is a device string valid? including the device ordinal specified?

    Other than cuda, "mps" (MacOS metal performance shaders) is experimentally supported.

    :param device: device string, such as ``cpu``, or ``cuda``, or ``cuda:N``
    :param raise_ordinal: Raise :py:exc:`.InvalidDeviceOrdinalException` if
        a specified CUDA device ordinal is found to not exist?

    :raises InvalidDeviceOrdinalException: If ``raise_ordinal=True`` and a the
        device ordinal specified in a CUDA device string does not exist.

    :return: ``True`` or ``False``
    """

    match = re.match(r'^(?:cpu|cuda(?::([0-9]+))?)$', device)

    if match:
        if match.lastindex:
            ordinal = int(match[1])
            valid_ordinal = ordinal < torch.cuda.device_count()
            if raise_ordinal and not valid_ordinal:
                raise InvalidDeviceOrdinalException(f'CUDA device ordinal {ordinal} is invalid, no such device exists.')
            return valid_ordinal
        return True

    if device == 'mps' and hasattr(torch.backends, 'mps') \
            and torch.backends.mps.is_available():
        return True

    return False


@contextmanager
def _disable_tqdm():
    huggingface_hub.utils.enable_progress_bars()
    try:
        yield
    finally:
        huggingface_hub.utils.enable_progress_bars()


def single_file_load_sub_module(
        path: str,
        class_name: str,
        library_name: str,
        name: str,
        use_auth_token: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        dtype: torch.dtype | None = None
) -> torch.nn.Module:
    """
    Load a submodule (vae, unet, textencoder, etc..) by name out of an in one checkpoint file.

    :param path: checkpoint file path
    :param class_name: submodule class name
    :param library_name: python module where the class exists
    :param name: submodule name, i.e. vae, unet, text_encoder, text_encoder_2, etc.
    :param use_auth_token: Hugging Face auth token for downloading configs?
    :param local_files_only: Do not attempt to download files, only use cache?
    :param revision: Repo revision for detected config repo
    :param dtype: torch dtype for the module

    :return: The module.
    """
    checkpoint = _single_file.load_single_file_checkpoint(
        path,
        token=use_auth_token,
        local_files_only=local_files_only,
        revision=revision,
    )

    config = _single_file.fetch_diffusers_config(checkpoint)
    default_pretrained_model_config_name = config["pretrained_model_name_or_path"]
    allow_patterns = ["**/*.json", "*.json", "*.txt", "**/*.txt", "**/*.model"]
    with _disable_tqdm():
        # the mischief I am doing here does not really integrate well with
        # the diffusers tqdm bars for single file loads, so disable them
        # for this helper function

        try:
            cached_model_config_path = huggingface_hub.snapshot_download(
                default_pretrained_model_config_name,
                revision=revision,
                local_files_only=True,
                token=use_auth_token,
                allow_patterns=allow_patterns
            )
        except huggingface_hub.errors.LocalEntryNotFoundError:
            if local_files_only:
                raise FileNotFoundError(
                    f'offline mode is active, but a config file is needed from Hugging Face '
                    f'hub to utilize the {class_name} in: {path}')
            else:
                try:
                    cached_model_config_path = huggingface_hub.snapshot_download(
                        default_pretrained_model_config_name,
                        revision=revision,
                        local_files_only=False,
                        token=use_auth_token,
                        allow_patterns=allow_patterns
                    )
                except huggingface_hub.errors.LocalEntryNotFoundError:
                    raise FileNotFoundError(
                        f'could not find the config file on Hugging Face hub '
                        f'for {class_name} in: {path}')

    model = _single_file.load_single_file_sub_model(
        library_name=library_name,
        checkpoint=checkpoint,
        class_name=class_name,
        name=name,
        pipelines=diffusers.pipelines,
        cached_model_config_path=cached_model_config_path,
        is_pipeline_module=False,
        torch_dtype=dtype,
        local_files_only=local_files_only)

    return model


def get_quantizer_uri_class(uri, exception=ValueError):
    """
    Get the URI parser class needed for a particular quantizer URI
    :param uri: The URI
    :param exception: Exception type to raise on unsupported quantization backend.
    :return: Class from :py:mod:`dgenerate.pipelinewrapper.uris`
    """
    concept = uri.split(';')[0].strip()
    if concept in {'bnb', 'bitsandbytes'}:
        if not diffusers.utils.is_bitsandbytes_available():
            raise exception(
                f'Cannot load quantization backend bitsandbytes, '
                f'as bitsandbytes is not installed.')
        return dgenerate.pipelinewrapper.uris.BNBQuantizerUri
    elif concept == 'torchao':
        if not diffusers.utils.is_torchao_available():
            raise exception(
                f'Cannot load quantization backend torchao, '
                f'as torchao is not installed.')
        return dgenerate.pipelinewrapper.uris.TorchAOQuantizerUri
    else:
        raise exception(f'Unknown quantization backend: {concept}')


def estimate_memory_usage(module: torch.nn.Module) -> str:
    """
    Estimate the static memory use of a torch module.

    :param module: the module

    :return: static memory use in bytes
    """

    dtype = next(module.parameters()).dtype
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1
    }
    bytes_per_param = dtype_sizes.get(dtype, 4)
    num_params = sum(p.numel() for p in module.parameters())
    return num_params * bytes_per_param


__all__ = _types.module_all()
