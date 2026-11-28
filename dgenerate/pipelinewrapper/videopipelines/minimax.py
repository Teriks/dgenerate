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

"""Run one MiniMax-H3 modular workflow and return frames plus audio."""

from __future__ import annotations

import contextlib

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.minimaxh3 as _minimax
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.videopipelines as _vp
import dgenerate.pipelinewrapper.videopipelines.media as _media
import dgenerate.pipelinewrapper.videopipelines.schedules as _schedules


def _overlay_scheduler(scheduler, uri):
    """Apply ``--scheduler`` arguments onto one ``MiniMaxH3Scheduler``."""
    import diffusers

    import dgenerate.pipelinewrapper.schedulers as _schedulers

    name = str(uri).split(';', 1)[0].strip()
    scheduler_cls = diffusers.MiniMaxH3Scheduler
    if not scheduler_cls.__name__.startswith(name):
        raise _schedulers.InvalidSchedulerNameError(
            f'Scheduler named "{name}" is not valid for MiniMax-H3. '
            f'Use {scheduler_cls.__name__}. Arguments overlay that checkpoint scheduler.')
    args = _schedulers.parse_scheduler_uri(scheduler_cls, uri)
    args.pop('eta', None)
    clone_config = args.pop('clone_config', True)
    try:
        if clone_config:
            return scheduler_cls.from_config(scheduler.config, **args)
        return scheduler_cls(**args)
    except Exception as error:
        raise _schedulers.SchedulerArgumentError(
            f'Error constructing scheduler "{scheduler_cls.__name__}" '
            f'with given URI argument values, encountered error: {error}') from error


def _configure_schedulers(pipe, scheduler_uri, audio_scheduler_uri):
    """``--scheduler`` is the video scheduler. ``--second-model-scheduler`` is audio.

    Like ``load_scheduler``, each URI overlays the checkpoint's scheduler, and
    a run without one gets the checkpoint's scheduler back.
    """
    if not hasattr(pipe, '_DGENERATE_ORIGINAL_SCHEDULER'):
        pipe._DGENERATE_ORIGINAL_SCHEDULER = pipe.scheduler
    original = pipe._DGENERATE_ORIGINAL_SCHEDULER
    pipe.scheduler = _overlay_scheduler(original, scheduler_uri) if scheduler_uri else original
    if scheduler_uri:
        _messages.debug_log(
            f'minimax-h3-video scheduler shift={pipe.scheduler.config.shift}.')

    audio = getattr(pipe, 'audio_scheduler', None)
    if audio is None:
        if audio_scheduler_uri:
            raise _pipelines.UnsupportedPipelineConfigError(
                'minimax-h3-video has no audio scheduler for --second-model-scheduler.')
        return
    if not hasattr(pipe, '_DGENERATE_ORIGINAL_AUDIO_SCHEDULER'):
        pipe._DGENERATE_ORIGINAL_AUDIO_SCHEDULER = audio
    original = pipe._DGENERATE_ORIGINAL_AUDIO_SCHEDULER
    pipe.audio_scheduler = (
        _overlay_scheduler(original, audio_scheduler_uri) if audio_scheduler_uri else original)
    if audio_scheduler_uri:
        _messages.debug_log(
            f'minimax-h3-video audio scheduler shift={pipe.audio_scheduler.config.shift}.')


def _user_sigmas(wrapper, pipe, user_args):
    """``--sigmas`` for both schedulers, or ``None``.

    The list ends at 0. ``--inference-steps`` counts that terminal 0.
    """
    if user_args.sigmas is None:
        return None
    sigmas = type(wrapper)._sigmas_eval(
        'minimax-h3-video', pipe, int(user_args.inference_steps), user_args.sigmas)
    user_args.inference_steps = len(sigmas)
    return sigmas


@contextlib.contextmanager
def _forced_sigmas(pipe, sigmas):
    """Give both schedulers ``sigmas`` for the call.

    The list is the grid before the shift. The video scheduler shifts it by
    12 and the audio scheduler by 3, so the two keep one step count. A
    scheduler with ``shift_provided_sigmas=false`` from ``--scheduler`` or
    ``--second-model-scheduler`` takes the list as written.
    """
    schedulers = (pipe.scheduler, pipe.audio_scheduler)
    saved = [scheduler.config.shift_provided_sigmas for scheduler in schedulers]
    try:
        with contextlib.ExitStack() as stack:
            for scheduler, value in zip(schedulers, saved):
                if value is None:
                    scheduler.register_to_config(shift_provided_sigmas=True)
                stack.enter_context(_schedules._forced_scheduler_sigmas(scheduler, sigmas))
            yield
    finally:
        for scheduler, value in zip(schedulers, saved):
            scheduler.register_to_config(shift_provided_sigmas=value)


def call_minimax_h3(wrapper, user_args):
    """Generate one MiniMax-H3 clip and its soundtrack."""
    if user_args.timesteps is not None:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video does not accept --timesteps. '
            'Use --sigmas or --inference-steps. 30 is the published recipe.')
    if getattr(user_args, 'video_frames', None) or getattr(user_args, 'end_video_frames', None):
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video keyframes are still images. Put a video in reference=.')
    positive, negative = _vp._prompt_text(user_args)
    if negative:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video is guidance-distilled and does not accept a negative prompt.')
    guidance = user_args.guidance_scale
    if guidance is not None and float(guidance) != 1.0:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video is guidance-distilled. Omit --guidance-scales, or pass 1.')
    user_args.guidance_scale = 1.0
    if user_args.video_fps is not None and float(user_args.video_fps) != _minimax.FPS:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'minimax-h3-video generates at {_minimax.FPS:g} fps. Omit --video-fps, or pass {_minimax.FPS:g}.')
    user_args.video_fps = _minimax.FPS
    if user_args.inference_steps is None:
        user_args.inference_steps = _minimax.DEFAULT_STEPS
        if user_args.sigmas is None:
            _messages.log(
                f'minimax-h3-video using --inference-steps {_minimax.DEFAULT_STEPS}. '
                'That count includes the terminal sigma, so the model runs one step fewer.')
    # A concrete --sigmas list sets the step count itself. An expression uses this one.
    if int(user_args.inference_steps) < 2 and (
            user_args.sigmas is None or isinstance(user_args.sigmas, str)):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'minimax-h3-video --inference-steps counts the terminal sigma, so it must be '
            f'at least 2 for one model step. Got {int(user_args.inference_steps)}. '
            f'30 is the published recipe.')

    references, reference_canvas = _minimax.build_references(
        list(getattr(user_args, 'reference_media_paths', None) or ()))
    control_uris = list(getattr(wrapper, 'controlnet_uris', None) or ())
    control_frames = getattr(user_args, 'minimax_control_frames', None)
    inpaint_frames = getattr(user_args, 'minimax_inpaint_frames', None)
    mask_frames = getattr(user_args, 'minimax_mask_frames', None)
    if control_uris:
        if len(control_uris) != 1:
            raise _pipelines.UnsupportedPipelineConfigError(
                'minimax-h3-video accepts one --control-nets model.')
        if references or user_args.images or user_args.end_images or user_args.video_frames:
            raise _pipelines.UnsupportedPipelineConfigError(
                'minimax-h3-video --control-nets is text to video. '
                'Keyframes and reference= use a different transformer.')
        if not control_frames and not mask_frames:
            raise _pipelines.UnsupportedPipelineConfigError(
                'minimax-h3-video --control-nets needs a control video in --image-seeds, '
                'or an inpaint source with mask=.')
    workflow = 't2va'
    if references:
        workflow = 'ref2va'
    elif user_args.images or user_args.end_images:
        workflow = 'fl2va'

    width, height = _vp._size(user_args)
    if width is None and (control_frames or inpaint_frames or mask_frames):
        sample = (control_frames or inpaint_frames or mask_frames)[0]
        width, height = sample.size
        width = max(_minimax.CANVAS_MULTIPLE, int(width) - int(width) % _minimax.CANVAS_MULTIPLE)
        height = max(_minimax.CANVAS_MULTIPLE, int(height) - int(height) % _minimax.CANVAS_MULTIPLE)
        _messages.log(
            f'minimax-h3-video control canvas is {width}x{height}, '
            'from the conditioning video. Pass --output-size to choose another.')
    if (width is None) != (height is None):
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video needs both --output-size width and height, or neither. '
            'With neither, the canvas follows the first keyframe, or 16:9.')
    if width is not None and (width % _minimax.CANVAS_MULTIPLE or height % _minimax.CANVAS_MULTIPLE):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'minimax-h3-video height and width must be multiples of {_minimax.CANVAS_MULTIPLE}. '
            f'Got {width}x{height}.')

    seconds = user_args.video_length if user_args.video_length is not None else _minimax.DEFAULT_SECONDS
    num_frames = _minimax.frames_for_seconds(seconds)

    pipe, _held = _vp._video_pipeline(wrapper, workflow)
    _configure_schedulers(
        pipe, user_args.scheduler_uri, getattr(user_args, 'second_model_scheduler_uri', None))
    sigmas = _user_sigmas(wrapper, pipe, user_args)
    import dgenerate.pipelinewrapper.videopipelines.single_image_video as _single_image_video
    _single_image_video._enhance_loaded_video_prompt(
        wrapper, pipe, user_args, references=references or None)
    positive, _negative = _vp._prompt_text(user_args)
    if control_uris:
        import dgenerate.pipelinewrapper.minimaxh3control as _control

        state, source, scale, dtype, uri_quantizer = _control.load_control_file(
            control_uris[0], wrapper.auth_token, wrapper.local_files_only)
        _control.attach_control_branch(
            pipe.transformer, state, source, dtype,
            model_path=wrapper.model_path,
            revision=wrapper.revision,
            auth_token=wrapper.auth_token,
            local_files_only=wrapper.local_files_only)
        del state
        quantizer_uri = uri_quantizer
        if not quantizer_uri:
            import dgenerate.pipelinewrapper.videopipelines.quantization as _quantization
            if _quantization._quantize_component(
                    'controlnet', wrapper.quantizer_uri, wrapper.quantizer_map):
                quantizer_uri = wrapper.quantizer_uri
        if quantizer_uri:
            _control.quantize_control_branch(pipe.transformer, quantizer_uri, wrapper.dtype)
        rows = _control.encode_control_rows(
            pipe, control_frames, inpaint_frames, mask_frames,
            int(height), int(width), num_frames)
        _control.set_control_rows(pipe.transformer, rows, scale)
    kwargs = {
        'prompt': positive,
        'num_frames': num_frames,
        'num_inference_steps': int(user_args.inference_steps),
        'generator': _vp._generator(wrapper, user_args),
        'output': ['videos', 'audio', 'sampling_rate'],
    }
    if width is not None:
        kwargs['width'] = int(width)
        kwargs['height'] = int(height)
    if user_args.images:
        kwargs['image'] = user_args.images[0]
    if user_args.end_images:
        kwargs['last_image'] = user_args.end_images[0]
    if references:
        kwargs['references'] = references
    latents = _first_tensor(getattr(user_args, 'latents', None))
    if latents is not None:
        kwargs['latents'] = latents
    audio_latents = _first_tensor(getattr(user_args, 'audio_latents', None))
    if audio_latents is not None:
        kwargs['audio_latents'] = audio_latents

    _vp._set_video_vae_slicing(pipe, bool(user_args.vae_slicing))
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(_minimax.honor_reference_canvas(pipe, reference_canvas))
            if sigmas is not None:
                stack.enter_context(_forced_sigmas(pipe, sigmas))
            result = _vp._invoke(wrapper, pipe, kwargs, user_args=user_args)
    finally:
        if control_uris and hasattr(pipe, 'transformer'):
            import dgenerate.pipelinewrapper.minimaxh3control as _control
            _control.set_control_rows(pipe.transformer, None, 1.0)
    videos = result.get('videos') if isinstance(result, dict) else result
    audio = result.get('audio') if isinstance(result, dict) else None
    sample_rate = result.get('sampling_rate') if isinstance(result, dict) else None
    if sample_rate is None:
        sample_rate = getattr(pipe, 'audio_sampling_rate', None)
    frames = _media.frames_from_output(videos)
    waveform = _media.audio_to_numpy(audio)
    return frames, waveform, int(sample_rate) if sample_rate else None, _minimax.FPS


def _first_tensor(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, str):
        import dgenerate.mediainput as _mediainput
        return _mediainput.load_tensor_file(value, value)
    return value
