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

"""LTX component discovery, availability checks, and the latent upsampler."""

from __future__ import annotations

import json
import os
import re

import torch

import dgenerate.hfhub as _hfhub
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _putil
import dgenerate.pipelinewrapper.videopipelines as _vp


def _ltx_repo_component_files(wrapper, repo_id: str, subfolder: str,
                              revision: str | None,
                              variant: str | None = None) -> set[str]:
    """List component filenames from a local HF cache or Hub metadata only."""
    import huggingface_hub

    local_files_only = bool(wrapper._local_files_only) or _hfhub.is_offline_mode()
    config_name = os.path.join(subfolder, 'config.json').replace('\\', '/')
    cached_config = huggingface_hub.try_to_load_from_cache(
        repo_id, config_name, revision=revision)
    if isinstance(cached_config, str):
        component_dir = os.path.dirname(cached_config)
        cached_names = set(os.listdir(component_dir))
        if local_files_only or _ltx_component_has_weights(cached_names, variant):
            return cached_names

    if local_files_only:
        return set()

    # list_repo_files only queries repository metadata; it does not download
    # model weights. Strip the component prefix to make local and Hub results
    # use the same filename checks below.
    filenames = huggingface_hub.HfApi(token=wrapper._auth_token).list_repo_files(
        repo_id, revision=revision)
    prefix = subfolder.strip('/\\')
    if not prefix:
        return {os.path.basename(name) for name in filenames}
    prefix = prefix.replace('\\', '/') + '/'
    return {
        name[len(prefix):]
        for name in filenames
        if name.replace('\\', '/').startswith(prefix)
    }


def _ltx_component_has_weights(filenames: set[str], variant: str | None) -> bool:
    """Check for Diffusers single-file or complete indexed component weights."""
    base_names = {
        'diffusion_pytorch_model.safetensors',
        'diffusion_pytorch_model.bin',
    }
    if variant:
        base_names = {
            'diffusion_pytorch_model.' + variant + '.safetensors',
            'diffusion_pytorch_model.' + variant + '.bin',
        }

    available = set()
    for filename in filenames:
        name = os.path.basename(filename)
        if name in base_names:
            return True
        if not _putil.variant_match(filename, variant):
            continue
        if name.endswith(('.safetensors', '.bin')) and name.startswith('diffusion_pytorch_model'):
            available.add(name)

    # Sharded Diffusers models have an index plus numbered weight files.
    index_names = {
        'diffusion_pytorch_model.safetensors.index.json',
        'diffusion_pytorch_model.bin.index.json',
    }
    if variant:
        index_names |= {
            'diffusion_pytorch_model.safetensors.index.' + variant + '.json',
            'diffusion_pytorch_model.bin.index.' + variant + '.json',
            'diffusion_pytorch_model.safetensors.' + variant + '.index.json',
            'diffusion_pytorch_model.bin.' + variant + '.index.json',
        }
    indexes = [name for name in filenames if os.path.basename(name) in index_names]
    shards = [name for name in available if '-of-' in name]
    if not indexes or not shards:
        return False

    groups = {}
    for shard in shards:
        match = re.search(r'-(\d+)-of-(\d+)\.(?:safetensors|bin)$', shard)
        if match:
            index, total = map(int, match.groups())
            groups.setdefault((total, os.path.splitext(shard)[1]), set()).add(index)
    return any(len(indices) == total and indices == set(range(1, total + 1))
               for (total, _extension), indices in groups.items())


def _ltx_component_files(wrapper, model_path: str, subfolder: str,
                         revision: str | None, variant: str | None,
                         allow_single_file: bool = False):
    """Return (filenames, is_single_file) without loading model weights."""
    if os.path.isdir(model_path):
        component_dir = os.path.join(model_path, subfolder)
        if not os.path.isdir(component_dir):
            return set(), False
        return set(os.listdir(component_dir)), False

    if os.path.isfile(model_path):
        return ({os.path.basename(model_path)}, True) if allow_single_file else (set(), False)

    blob = _hfhub.HFBlobLink.parse(model_path)
    if blob is not None:
        if not allow_single_file:
            return set(), False
        import huggingface_hub
        filename = os.path.join(blob.subfolder or '', blob.weight_name).replace('\\', '/')
        cached_file = huggingface_hub.try_to_load_from_cache(
            blob.repo_id, filename, revision=blob.revision)
        if isinstance(cached_file, str):
            return {blob.weight_name}, True
        if bool(wrapper._local_files_only) or _hfhub.is_offline_mode():
            return set(), True
        filenames = huggingface_hub.HfApi(token=wrapper._auth_token).list_repo_files(
            blob.repo_id, revision=blob.revision)
        return ({blob.weight_name} if filename in filenames else set()), True

    if model_path.startswith(('http://', 'https://')):
        if not allow_single_file:
            return set(), False
        # Probe a direct URL with HEAD only. Some hosts do not support HEAD;
        # in that case leave the decision to the normal loader.
        if bool(wrapper._local_files_only) or _hfhub.is_offline_mode():
            return set(), True
        import urllib.error
        import urllib.request
        try:
            request = urllib.request.Request(model_path, method='HEAD')
            with urllib.request.urlopen(request, timeout=15) as response:
                return ({os.path.basename(model_path)} if 200 <= response.status < 400 else set()), True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return set(), True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        return None

    if _hfhub.is_single_file_model_load(model_path):
        return (set(), True) if allow_single_file else (set(), False)

    return _ltx_repo_component_files(wrapper, model_path, subfolder, revision, variant), False


def _check_ltx_component_available(wrapper, model_path: str, component: str,
                                   display_name: str, subfolder: str,
                                   revision: str | None = None,
                                   variant: str | None = None,
                                   allow_single_file: bool = False,
                                   config_name: str = 'config.json') -> None:
    files = _ltx_component_files(
        wrapper, model_path, subfolder, revision, variant, allow_single_file)
    if files is None:
        return
    filenames, single_file = files
    available = bool(filenames) if single_file else (
        config_name in filenames and _ltx_component_has_weights(filenames, variant))
    if available:
        return

    if model_path == wrapper.model_path:
        if bool(wrapper._local_files_only) or _hfhub.is_offline_mode():
            message = (f'The {display_name} in "{model_path}/{subfolder}" is not available '
                       'in the local cache.')
        else:
            message = (f'The LTX checkpoint at "{model_path}" does not contain an available '
                       f'{display_name} in "{subfolder}".')
    else:
        message = (f'The {display_name} model "{model_path}" is missing its config or '
                   'Diffusers weight files.')
    if component == 'diffusion_decoder':
        message += ' Use --ltx-video-decoder conv to use the convolutional VAE instead.'
    elif component == 'latent_upsampler':
        message += ' Ensure the selected model is an LTX latent upsampler.'
    raise _pipelines.UnsupportedPipelineConfigError(message)


def _check_ltx_latent_upsampler_available(wrapper, user_args, temporal: bool = False) -> None:
    model_path = wrapper.model_path
    subfolder = 'temporal_latent_upsampler' if temporal else 'latent_upsampler'
    revision = None
    variant = None
    if isinstance(getattr(user_args, 'ltx_latent_upscale', None), str):
        uri = _uris.LTXLatentUpsamplerUri.parse(user_args.ltx_latent_upscale)
        model_path = uri.model
        revision = uri.revision
        variant = uri.variant
        if uri.subfolder:
            subfolder = uri.subfolder
        if _hfhub.is_single_file_model_load(model_path):
            subfolder = ''

    _check_ltx_component_available(
        wrapper, model_path, 'latent_upsampler', 'latent upsampler',
        subfolder, revision, variant, _hfhub.is_single_file_model_load(model_path))

    config = _ltx_latent_upsampler_config(wrapper, model_path, subfolder, revision)
    if config is not None and config.get('spatial_upsample') is not (not temporal):
        actual = 'spatial' if config.get('spatial_upsample') else 'temporal-only'
        raise _pipelines.UnsupportedPipelineConfigError(
            f'The latent upsampler in "{model_path}/{subfolder}" is {actual}. '
            f'--ltx-latent-upscale-mode {"temporal" if temporal else "spatial"} requires a '
            f'component with spatial_upsample={not temporal}.')


def _check_ltx_diffusion_decoder_available(wrapper, user_args) -> None:
    model_path = wrapper.model_path
    revision = None
    subfolder = 'diffusion_decoder'
    if user_args.ltx_diffusion_decoder:
        uri = _uris.LTXDiffusionDecoderUri.parse(user_args.ltx_diffusion_decoder)
        model_path = uri.model
        revision = uri.revision
        if uri.subfolder:
            subfolder = uri.subfolder
    _check_ltx_component_available(
        wrapper, model_path, 'diffusion_decoder', 'diffusion decoder',
        subfolder, revision=revision, config_name='config.json')


def _ltx_latent_upsampler_config(wrapper, model_path: str, subfolder: str,
                                  revision: str | None) -> dict | None:
    """Read a latent upsampler config without loading its weights."""
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, subfolder, 'config.json')
        if not os.path.isfile(config_path):
            return None
    else:
        # Single-file checkpoints may not have a separate Diffusers config.
        if os.path.isfile(model_path) or _hfhub.HFBlobLink.parse(model_path) is not None \
                or _hfhub.is_single_file_model_load(model_path):
            return None

        import huggingface_hub

        filename = os.path.join(subfolder, 'config.json').replace('\\', '/')
        config_path = huggingface_hub.try_to_load_from_cache(
            model_path, filename, revision=revision)
        local_files_only = bool(wrapper._local_files_only) or _hfhub.is_offline_mode()
        if not isinstance(config_path, str):
            if local_files_only:
                return None
            config_path = huggingface_hub.hf_hub_download(
                model_path,
                filename=filename,
                revision=revision,
                token=wrapper._auth_token,
                local_files_only=False)

    with open(config_path, encoding='utf-8') as config_file:
        return json.load(config_file)


def _is_missing_ltx_component(error, model_path: str, component: str, config_name: str) -> bool:
    """Return whether Diffusers could not find the component's config file."""
    if os.path.isdir(model_path):
        # Diffusers falls back to the root config when a local subfolder config
        # is missing. Check the requested component directly to avoid treating
        # the main pipeline config as the component config.
        return not os.path.isfile(os.path.join(model_path, component, config_name))

    if error is None:
        return False

    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        message = str(current).lower()
        missing_config = config_name.lower()
        if (f'does not appear to have a file named {missing_config}' in message
                or f'error no file named {missing_config} found in directory' in message):
            return True

        pending.extend((current.__cause__, current.__context__))
    return False


def _load_ltx_latent_upsampler(wrapper, user_args, use_custom_uri=True):
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    dtype = _enums.get_torch_dtype(wrapper._dtype) or torch.bfloat16
    model_path = wrapper.model_path
    load_kwargs = {
        'subfolder': ('temporal_latent_upsampler'
                      if getattr(user_args, 'ltx_latent_upscale_mode', 'spatial') == 'temporal'
                      else 'latent_upsampler'),
        'torch_dtype': dtype,
        'token': wrapper._auth_token,
        'local_files_only': bool(wrapper._local_files_only),
    }

    using_custom_upsampler = (use_custom_uri and
                              isinstance(getattr(user_args, 'ltx_latent_upscale', None), str))
    if using_custom_upsampler:
        uri = _uris.LTXLatentUpsamplerUri.parse(user_args.ltx_latent_upscale)
        model_path = uri.model
        if uri.revision is not None:
            load_kwargs['revision'] = uri.revision
        if uri.variant is not None:
            load_kwargs['variant'] = uri.variant
        if uri.subfolder:
            load_kwargs['subfolder'] = uri.subfolder
        if _hfhub.is_single_file_model_load(model_path):
            load_kwargs.pop('subfolder', None)

    if (not using_custom_upsampler
            and _is_missing_ltx_component(None, model_path, 'latent_upsampler',
                                          LTX2LatentUpsamplerModel.config_name)):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'The LTX checkpoint at "{wrapper.model_path}" does not contain a latent upsampler. '
            'Ensure you are using an LTX-2 checkpoint that supports latent upscaling.')

    try:
        if _hfhub.is_single_file_model_load(model_path):
            from_single_file = getattr(LTX2LatentUpsamplerModel, 'from_single_file', None)
            if from_single_file is None:
                from diffusers.loaders.single_file_model import FromOriginalModelMixin
                from_single_file = FromOriginalModelMixin.from_single_file.__get__(
                    LTX2LatentUpsamplerModel, LTX2LatentUpsamplerModel)
            return from_single_file(model_path, **load_kwargs)
        return LTX2LatentUpsamplerModel.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        # Preserve the loader's original error for a separately selected model;
        # it may indicate authentication, revision, or weight format problems.
        if (not using_custom_upsampler
                and _is_missing_ltx_component(
                    e, model_path, 'latent_upsampler', LTX2LatentUpsamplerModel.config_name)):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'The LTX checkpoint at "{wrapper.model_path}" does not contain a latent upsampler. '
                'Ensure you are using an LTX-2 checkpoint that supports latent upscaling.') from e
        raise


def _upsample_ltx_latents(wrapper, pipe, video_latent, user_args, use_custom_uri=True):
    from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline

    # Keep model weights out of memory until stage 1 is complete.
    upsampler = _load_ltx_latent_upsampler(wrapper, user_args, use_custom_uri=use_custom_uri)
    dtype = _enums.get_torch_dtype(wrapper._dtype) or torch.bfloat16
    up_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=upsampler)
    # Sequential offload stores this model's weights as meta tensors. The
    # upsample forward then cannot copy them onto the GPU. The upsampler is
    # small next to the transformer, so it stays on the compute device for
    # this call and is moved off again afterward. The shared VAE is left on
    # the main pipeline's offload hooks; this call does not decode with it.
    device = wrapper.device
    upsampler.to(device=device, dtype=dtype)
    if torch.is_tensor(video_latent):
        video_latent = video_latent.to(device=device)
    offload = _vp._offload_requested(wrapper)
    try:
        upscaled = up_pipe(
            latents=video_latent,
            latents_normalized=False,
            output_type='latent',
            return_dict=False)[0]
    finally:
        if offload:
            upsampler.to('cpu')
    return upscaled
