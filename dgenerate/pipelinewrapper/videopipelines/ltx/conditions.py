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

"""LTX conditioning media, IC-LoRA reference clips, and input fitting."""

from __future__ import annotations

import math
import os

import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.videopipelines as _vp


def _ltx_temporal_compression(pipe) -> int:
    return int(getattr(pipe, 'vae_temporal_compression_ratio', 8) or 8)


def _trim_ltx_clip(frames: list, limit: int | None, temporal: int) -> list:
    count = len(frames) if limit is None else min(len(frames), limit)
    count = (count - 1) // temporal * temporal + 1
    if count < 1:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The LTX output is too short to hold the conditioning clips. '
            'Use a longer --video-lengths.')
    return frames[:count]


def _ltx_conditions(pipe, family: str, user_args, num_frames: int | None) -> list:
    """
    Build the condition list for the LTX condition pipeline.

    Leading media starts at frame 0 and trailing media ends on the last frame.
    Clips are cut to ``8k+1`` frames, and the leading clip is shortened so the
    two never overlap.
    """
    temporal = _ltx_temporal_compression(pipe)
    start_image = user_args.images[0] if user_args.images else None
    end_image = user_args.end_images[0] if user_args.end_images else None
    start_video = user_args.video_frames or None
    end_video = user_args.end_video_frames or None
    has_start = start_image is not None or start_video is not None

    if num_frames is None:
        if family == 'ltx':
            num_frames = 161
        elif getattr(pipe, 'duration_head', None) is None:
            num_frames = 121
        elif end_video is not None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'An LTX end clip needs a fixed output length. This checkpoint picks its '
                'length from the prompt, so set --video-lengths.')

    end_clip = None
    end_count = 0
    if end_video is not None:
        end_clip = _trim_ltx_clip(
            end_video, num_frames - (1 if has_start else 0), temporal)
        end_count = len(end_clip)
    elif end_image is not None:
        end_count = 1

    start_clip = None
    if start_video is not None:
        start_clip = _trim_ltx_clip(
            start_video, None if num_frames is None else num_frames - end_count, temporal)

    for name, clip in (('start', start_clip), ('end', end_clip)):
        if clip is not None:
            _messages.debug_log(f'LTX {name} conditioning clip: {len(clip)} frames.')

    start_index = 0 if user_args.ltx_condition_index is None else int(user_args.ltx_condition_index)
    start_strength = _ltx_resolved_strength(user_args.ltx_condition_strength, user_args)

    conditions = []
    if family == 'ltx':
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        if start_clip is not None:
            conditions.append(LTXVideoCondition(
                video=start_clip, frame_index=start_index, strength=start_strength))
        elif start_image is not None:
            conditions.append(LTXVideoCondition(
                image=start_image, frame_index=start_index, strength=start_strength))
        if end_clip is not None:
            conditions.append(LTXVideoCondition(
                video=end_clip, frame_index=num_frames - end_count, strength=1.0))
        elif end_image is not None:
            conditions.append(LTXVideoCondition(
                image=end_image, frame_index=num_frames - 1, strength=1.0))
        _append_extra_ltx_conditions(
            conditions, family, user_args.ltx_extra_conditions, LTXVideoCondition, user_args)
        return conditions

    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

    if start_clip is not None:
        conditions.append(LTX2VideoCondition(
            frames=start_clip, index=start_index, strength=start_strength))
    elif start_image is not None:
        conditions.append(LTX2VideoCondition(
            frames=start_image, index=start_index, strength=start_strength))
    if end_clip is not None:
        # LTX-2 indices are latent frames, and each latent after the first covers ``temporal`` pixels.
        latent_frames = (num_frames - 1) // temporal + 1
        clip_latents = (end_count - 1) // temporal + 1
        conditions.append(LTX2VideoCondition(
            frames=end_clip, index=latent_frames - clip_latents, strength=1.0))
    elif end_image is not None:
        conditions.append(LTX2VideoCondition(frames=end_image, index=-1, strength=1.0))
    _append_extra_ltx_conditions(
        conditions, family, user_args.ltx_extra_conditions, LTX2VideoCondition, user_args)
    return conditions


def _ltx_image_needs_conditions(user_args) -> bool:
    if user_args.ltx_condition_index not in (None, 0):
        return True
    if user_args.ltx_extra_conditions:
        return True
    return _ltx_resolved_strength(user_args.ltx_condition_strength, user_args) != 1.0


def _ltx_resolved_strength(explicit, user_args) -> float:
    """
    Condition weight for one LTX image seed group.

    The image-seed keyword wins. ``--image-seed-strengths`` fills a group
    that omitted it. Otherwise the frame stays at full strength.
    """
    if explicit is not None:
        return float(explicit)
    if user_args.image_seed_strength is not None:
        return float(user_args.image_seed_strength)
    return 1.0


def _append_extra_ltx_conditions(conditions, family, extras, condition_cls, user_args):
    for frames, index, strength in extras or []:
        index = int(index)
        strength = _ltx_resolved_strength(strength, user_args)
        if family == 'ltx':
            if isinstance(frames, list):
                conditions.append(condition_cls(video=frames, frame_index=index, strength=strength))
            else:
                conditions.append(condition_cls(image=frames, frame_index=index, strength=strength))
        else:
            conditions.append(condition_cls(frames=frames, index=index, strength=strength))


def _ltx_reference_kwargs(wrapper, pipe, held, user_args, num_frames: int,
                          width: int | None, height: int | None) -> dict:
    """
    Build the IC-LoRA reference arguments for :py:class:`diffusers.LTX2InContextPipeline`.

    The reference is the image seed control clip. Its downscale factor comes from
    the ``--ltx-ic-lora`` URI, or else from the IC-LoRA file metadata.

    Frames are fit to the output canvas with the still-image ``--output-size``
    rule first. The pipeline then center-crops to the LoRA reference size, and a
    matching aspect means that crop only scales.
    """
    from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition

    factor = held.reference_downscale_factor or 1
    ic_lora = _vp._parsed_ic_lora(wrapper)
    attention = ic_lora.attention if ic_lora is not None else 1.0
    spatial = int(getattr(pipe, 'vae_spatial_compression_ratio', 32) or 32)

    clip = _trim_ltx_clip(
        user_args.reference_video_frames, num_frames, _ltx_temporal_compression(pipe))
    clip = _resize_ltx_control_clip(
        clip, width, height, user_args.aspect_correct, spatial * max(1, int(factor)))
    _messages.debug_log(
        f'LTX IC-LoRA reference clip: {len(clip)} frames, downscale factor {factor}, '
        f'attention {attention}.')
    return {
        'reference_conditions': [LTX2ReferenceCondition(frames=clip, strength=1.0)],
        'reference_downscale_factor': factor,
        'conditioning_attention_strength': attention,
    }


def _ltx_canvas_align(pipe, held, mode: str, user_args) -> int:
    spatial = int(getattr(pipe, 'vae_spatial_compression_ratio', 32) or 32)
    align = spatial
    if mode == 'ltx-control':
        factor = int(getattr(held, 'reference_downscale_factor', None) or 1)
        align = spatial * max(1, factor)
    if user_args.ltx_latent_upscale:
        # stage 1 runs at half size, which must still be spatially aligned
        align = math.lcm(align, spatial * 2)
    return align


def _fit_ltx_inputs(pipe, held, mode, user_args, width, height):
    """
    Fit every LTX conditioning image and clip to one canvas.

    The first available control clip, opening media, or end media chooses
    the canvas. The other slots are resized onto it.
    """
    primary = None
    for attr in ('reference_video_frames', 'video_frames', 'images',
                 'end_video_frames', 'end_images'):
        frames = getattr(user_args, attr, None)
        if frames:
            primary = frames[0].size
            break
    if primary is None:
        return width, height
    align = _ltx_canvas_align(pipe, held, mode, user_args)
    target_w, target_h = _vp.conditioning_canvas(
        primary, width, height, user_args.aspect_correct, align)
    target = (target_w, target_h)
    if primary != target:
        _messages.log(
            f'Resizing LTX conditioning media from {primary[0]}x{primary[1]} '
            f'to {target_w}x{target_h}.')
    for attr in ('reference_video_frames', 'video_frames', 'images',
                 'end_video_frames', 'end_images'):
        frames = getattr(user_args, attr, None)
        if frames:
            setattr(user_args, attr, _vp.resize_media(frames, target))
    extras = []
    for frames, index, strength in user_args.ltx_extra_conditions or []:
        if isinstance(frames, list):
            frames = _vp.resize_media(frames, target)
        else:
            frames = _vp.resize_media([frames], target)[0]
        extras.append((frames, index, strength))
    if extras:
        user_args.ltx_extra_conditions = extras
    user_args.width = target_w
    user_args.height = target_h
    return target_w, target_h


def _resize_ltx_control_clip(frames, width, height, aspect_correct, align: int):
    """Fit an IC-LoRA control clip with the still-image ``--output-size`` rule."""
    source = frames[0].size
    target_w, target_h = _vp.conditioning_canvas(
        source, width, height, aspect_correct, align)
    target = (target_w, target_h)
    if source == target:
        return frames
    _messages.log(
        f'Resizing LTX IC-LoRA control clip from {source[0]}x{source[1]} '
        f'to {target[0]}x{target[1]}.')
    return _vp.resize_media(frames, target)


def _ic_lora_downscale_factor(lora_uri: str, override: int | None,
                              auth_token, local_files_only) -> int:
    """
    Return the ``--ltx-ic-lora`` ``downscale`` value, or else ``reference_downscale_factor``
    from the IC-LoRA safetensors metadata. IC-LoRAs trained on reduced-size references
    store it there.
    """
    if override is not None:
        return override
    try:
        metadata = _lora_file_metadata(lora_uri, auth_token, local_files_only)
    except Exception as e:
        _messages.warning(
            f'Could not read the metadata of IC-LoRA "{lora_uri}", assuming reference '
            f'downscale factor 1. Set weight-name if the repository has several files, '
            f'or set downscale in the --ltx-ic-lora URI. Error: {e}')
        return 1
    value = metadata.get('reference_downscale_factor') if metadata else None
    if value is None:
        return 1
    try:
        factor = max(1, int(float(value)))
    except ValueError:
        _messages.warning(
            f'IC-LoRA "{lora_uri}" has an invalid reference_downscale_factor {value!r}, assuming 1.')
        return 1
    _messages.debug_log(f'IC-LoRA "{lora_uri}" sets reference downscale factor {factor}.')
    return factor


def _lora_file_metadata(uri, auth_token, local_files_only) -> dict | None:
    """
    Return the safetensors header metadata of a ``--loras`` URI, resolving the
    weight file the same way ``load_lora_weights`` does. ``None`` for non-safetensors files.
    """
    import safetensors
    from diffusers.loaders.lora_base import _best_guess_weight_name
    from diffusers.utils import _get_model_file

    lora = uri if isinstance(uri, _uris.LoRAUri) else _uris.LoRAUri.parse(uri)
    path = _hfhub.download_non_hf_slug_model(lora.model, local_files_only=local_files_only, auth_token=auth_token)

    if os.path.isfile(path):
        model_file = path
    else:
        weight_name = lora.weight_name
        if weight_name is None:
            search = path
            if os.path.isdir(path) and lora.subfolder:
                search = os.path.join(path, lora.subfolder)
            weight_name = _best_guess_weight_name(
                search, file_extension='.safetensors', local_files_only=local_files_only)
        if not weight_name:
            return None
        model_file = _get_model_file(
            path,
            weights_name=weight_name,
            subfolder=lora.subfolder,
            revision=lora.revision,
            token=auth_token,
            local_files_only=local_files_only)

    if not str(model_file).endswith('.safetensors'):
        return None
    with safetensors.safe_open(model_file, framework='pt') as f:
        return f.metadata()
