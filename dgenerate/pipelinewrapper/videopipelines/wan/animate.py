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

"""Wan-Animate: animate and replace modes, and driving-clip preprocessing."""

from __future__ import annotations

import os

import PIL.Image

import dgenerate.filelock as _filelock
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.types as _types
import dgenerate.pipelinewrapper.videopipelines as _vp
from .common import (
    _apply_wan_config,
    _fit_output_media,
    _require_image_encoder,
    _spatial_multiple,
    _temporal_factor,
    _trim_wan_clip,
)


def _call_wan_animate(wrapper, user_args):

    if user_args.video_length is not None:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--video-lengths cannot be used with --model-type wan-animate. '
            'The output length follows the pose clip.')
    if not user_args.images:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate needs a character image in --image-seeds.')

    pose = user_args.wan_pose_video_frames
    face = user_args.wan_face_video_frames
    if user_args.wan_animate_preprocess and user_args.wan_driving_video_frames:
        derived_pose, derived_face = preprocess_wan_animate_driving(
            user_args.wan_driving_video_frames,
            user_args.wan_driving_video_path,
            user_args.wan_animate_output_dir,
            motion_encoder_size=_animate_motion_size(wrapper),
            device=wrapper.device,
            write_clips=bool(user_args.wan_animate_preprocess_output))
        pose = pose or derived_pose
        face = face or derived_face
    if not pose or not face:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate needs wan-pose= and wan-face= clips, or wan-driving= with '
            '--wan-animate-preprocess or --wan-pose-image-processors and '
            '--wan-face-image-processors.')
    user_args.wan_pose_video_frames = pose

    pipe, held = _vp._video_pipeline(wrapper, 'wan-animate', user_args.scheduler_uri)
    _require_image_encoder(pipe, 'wan-animate')
    _apply_wan_config(pipe, user_args)
    positive, negative = _vp._prompt_text(user_args)
    width, height = _vp._size(user_args)
    align = _spatial_multiple(pipe)
    width, height = _vp._force_output_size_alignment(
        width, height, align, 'Wan-Animate')
    width, height = _fit_output_media(
        user_args, 'wan_pose_video_frames',
        [('wan_background_video_frames', False), ('mask_video_frames', True)],
        width, height, align, 'Wan-Animate pose clip')
    pose = user_args.wan_pose_video_frames

    fps = float(user_args.video_fps or _constants.DEFAULT_WAN_ANIMATE_FPS)
    mode = user_args.wan_animate_mode or 'animate'
    if mode not in ('animate', 'replace'):
        raise _pipelines.UnsupportedPipelineConfigError(
            '--wan-animate-mode must be animate or replace.')
    if mode == 'replace':
        if not user_args.wan_background_video_frames or not user_args.mask_video_frames:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Wan-Animate replace mode needs wan-background= and mask= clips.')

    guidance = float(
        _types.default(user_args.guidance_scale, _constants.DEFAULT_GUIDANCE_SCALE))
    steps = int(_types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
    if guidance == _constants.DEFAULT_GUIDANCE_SCALE:
        guidance = _constants.DEFAULT_WAN_ANIMATE_GUIDANCE_SCALE
        user_args.guidance_scale = guidance
        _messages.debug_log(
            'Wan-Animate runs unguided. Default guidance scale 5 was replaced with 1.')
    if steps == _constants.DEFAULT_INFERENCE_STEPS:
        steps = _constants.DEFAULT_WAN_ANIMATE_INFERENCE_STEPS
        user_args.inference_steps = steps
        _messages.debug_log(
            'Wan-Animate default inference steps 30 was replaced with 20.')

    temporal = _temporal_factor(pipe)
    pose = _trim_wan_clip(pose, None, temporal)
    face = face[:len(pose)]
    kwargs = {
        'image': user_args.images[0],
        'pose_video': pose,
        'face_video': face,
        'prompt': positive,
        'generator': _vp._generator(wrapper, user_args),
        'output_type': 'pil',
        'mode': mode,
        'guidance_scale': guidance,
        'num_inference_steps': steps,
        'segment_frame_length': int(
            _types.default(user_args.wan_segment_frame_length, _constants.DEFAULT_WAN_ANIMATE_SEGMENT_FRAME_LENGTH)),
        'prev_segment_conditioning_frames': int(
            _types.default(user_args.wan_prev_segment_frames, _constants.DEFAULT_WAN_ANIMATE_PREV_SEGMENT_FRAMES)),
    }
    if negative:
        kwargs['negative_prompt'] = negative
    if width is not None:
        kwargs['width'] = width
        kwargs['height'] = height
    if user_args.wan_motion_encode_batch_size is not None:
        kwargs['motion_encode_batch_size'] = int(user_args.wan_motion_encode_batch_size)
    if user_args.max_sequence_length is not None:
        kwargs['max_sequence_length'] = int(user_args.max_sequence_length)
    if mode == 'replace':
        kwargs['background_video'] = user_args.wan_background_video_frames[:len(pose)]
        kwargs['mask_video'] = user_args.mask_video_frames[:len(pose)]

    _vp._set_video_vae_slicing(pipe, bool(user_args.vae_slicing))
    output = _vp._invoke(wrapper, pipe, kwargs, user_args=user_args)
    frames = _vp.frames_from_output(getattr(output, 'frames', output))
    return frames[:len(pose)], None, None, fps


def _animate_motion_size(wrapper) -> int:
    return 512


def preprocess_wan_animate_driving(frames: list[PIL.Image.Image],
                                   source_path: str | None,
                                   output_dir: str | None,
                                   motion_encoder_size: int = 512,
                                   device: str = 'cpu',
                                   write_clips: bool = False
                                   ) -> tuple[list[PIL.Image.Image], list[PIL.Image.Image]]:
    """
    Derive pose and face clips from a driving video.

    Pose is ``openpose``. Face is a square crop around the first
    ``yolo`` face box. Both processors run on ``device`` for the clip
    and return to CPU when the clip is finished. The returned frames
    are what generation uses. ``write_clips`` also stores them under
    ``output_dir``. A name that already exists receives a
    ``_duplicate_`` suffix.
    """
    pose_processor, face_processor = _animate_processors(motion_encoder_size, device)
    try:
        pose = [pose_processor.process(frame.copy()) for frame in frames]
        face = [face_processor.process(frame.copy()) for frame in frames]
    finally:
        pose_processor.to('cpu')
        if hasattr(face_processor, 'to'):
            face_processor.to('cpu')
    if write_clips:
        _write_derived_clips(pose, face, source_path, output_dir)
    return pose, face


def _derived_clip_paths(source_path: str | None, output_dir: str | None) -> tuple[str, str]:
    # ``output`` matches the ``--output-path`` default when no directory was passed.
    directory = output_dir or 'output'
    stem = _driving_clip_stem(source_path)
    return (os.path.join(directory, f'{stem}.wan-pose.mp4'),
            os.path.join(directory, f'{stem}.wan-face.mp4'))


def _driving_clip_stem(source_path: str | None) -> str:
    if not source_path:
        return 'wan-animate'
    path = str(source_path).split('?', 1)[0].split('#', 1)[0].rstrip('/\\')
    stem = os.path.splitext(os.path.basename(path))[0]
    cleaned = ''.join(
        ch if ch.isalnum() or ch in ('-', '_', '.') else '-' for ch in stem).strip('.-')
    return cleaned or 'wan-animate'


def _write_derived_clips(pose_frames: list[PIL.Image.Image],
                         face_frames: list[PIL.Image.Image],
                         source_path: str | None,
                         output_dir: str | None):
    pose_path, face_path = _derived_clip_paths(source_path, output_dir)
    directory = os.path.dirname(pose_path) or '.'
    try:
        os.makedirs(directory, exist_ok=True)
        pose_path, face_path = _filelock.touch_avoid_duplicate(
            directory,
            path_maker=_filelock.suffix_path_maker(
                [pose_path, face_path], suffix='_duplicate_'))
        _write_clip(pose_frames, pose_path)
        _write_clip(face_frames, face_path)
    except Exception as error:
        _messages.debug_log(
            f'Could not write Wan-Animate pose and face clips: {error}')


def _write_clip(frames: list[PIL.Image.Image], path: str):
    import dgenerate.mediaoutput as _mediaoutput
    with _mediaoutput.VideoWriter(path, fps=16) as writer:
        for frame in frames:
            writer.write(frame)
    _messages.log(f'Wrote Wan-Animate clip "{path}".')


def _animate_processors(motion_encoder_size: int = 512, device: str = 'cpu'):
    import dgenerate.imageprocessors as _imgp
    pose = _imgp.create_image_processor(
        'openpose;include-hand=true', device=device)
    face = _imgp.create_image_processor(
        'yolo;model=Bingsu/adetailer;weight-name=face_yolov8n.pt;'
        f'crops=True;crop-square=True;crop-scale=1.4;'
        f'crop-size={int(motion_encoder_size)}',
        device=device)
    return pose, face
