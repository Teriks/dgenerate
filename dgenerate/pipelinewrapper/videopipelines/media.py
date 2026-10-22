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

"""Frame, audio, and canvas helpers shared by the video model families."""

from __future__ import annotations

import PIL.Image
import numpy
import torch

import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.types as _types
import dgenerate.pipelinewrapper.videopipelines as _vp


def load_rgb_frames(path: str,
                    local_files_only: bool = False,
                    resize_resolution: _types.OptionalSize = None,
                    aspect_correct: bool = True,
                    align: int = 1,
                    frame_start: int = 0,
                    frame_end: _types.OptionalInteger = None,
                    max_frames: _types.OptionalInteger = None) -> tuple[list[PIL.Image.Image], float | None]:
    """
    Open a local path or URL as a list of RGB frames.

    A still image gives one frame. Videos and animated images are sliced by
    ``frame_start`` and ``frame_end`` (inclusive), then cut to ``max_frames``.

    :param path: file path or URL
    :param local_files_only: refuse to download
    :param resize_resolution: optional resize
    :param aspect_correct: preserve aspect ratio when resizing
    :param align: pixel alignment, ``1`` disables it
    :param frame_start: first frame index to keep
    :param frame_end: last frame index to keep, ``None`` for the end of the file
    :param max_frames: stop after this many frames, ``None`` for no limit
    :raise UnsupportedPipelineConfigError: if the slice selects no frames
    :return: ``(frames, fps)``. ``fps`` is ``None`` for a still image.
    """
    mime_type, stream = _mediainput.fetch_media_data_stream(
        path, local_files_only=local_files_only)
    try:
        if _mediainput.mimetype_is_static_image(mime_type):
            return [_mediainput.create_image(
                stream,
                file_source=path,
                resize_resolution=resize_resolution,
                aspect_correct=aspect_correct,
                align=align)], None

        if not (_mediainput.mimetype_is_video(mime_type) or
                _mediainput.mimetype_is_animated_image(mime_type)):
            raise _mediainput.UnknownMimetypeError(
                f'Expected an image, animated image, or video for "{path}", '
                f'got mimetype "{mime_type}".')

        frames = []
        try:
            with _mediainput.create_animation_reader(
                    mime_type,
                    file_source=path,
                    file=stream,
                    resize_resolution=resize_resolution,
                    aspect_correct=aspect_correct,
                    align=align) as reader:
                fps = float(reader.fps)
                for index, frame in enumerate(reader):
                    if frame_end is not None and index > frame_end:
                        frame.close()
                        break
                    if index < frame_start:
                        frame.close()
                        continue
                    frames.append(_to_rgb_image(frame))
                    if max_frames is not None and len(frames) >= max_frames:
                        break
        except Exception:
            for frame in frames:
                frame.close()
            raise

        if not frames:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'No frames were read from "{path}" between frame {frame_start} '
                f'and frame {"end" if frame_end is None else frame_end}.')
        return frames, fps
    finally:
        stream.close()


def frames_from_output(value) -> list[PIL.Image.Image]:
    """
    Normalize a pipeline video output to a list of RGB images.

    Accepts a list of images, a batch of those lists, or an array/tensor in
    frame-major channels-last or channels-first layout. A batch uses the first clip.

    :param value: pipeline video output
    :return: RGB frames
    """
    if value is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The video pipeline did not return frames.')

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    if isinstance(value, numpy.ndarray):
        return _frames_from_array(value)

    if isinstance(value, list):
        if not value:
            raise _pipelines.UnsupportedPipelineConfigError(
                'The video pipeline returned an empty frame list.')
        head = value[0]
        if isinstance(head, PIL.Image.Image):
            return [_to_rgb_image(frame) for frame in value]
        if isinstance(head, list):
            return _vp.frames_from_output(head)
        if torch.is_tensor(head) or isinstance(head, numpy.ndarray):
            array = head.detach().cpu().numpy() if torch.is_tensor(head) else numpy.asarray(head)
            if array.ndim >= 4:
                return _vp.frames_from_output(head)
            return [_array_frame_to_image(frame) for frame in value]
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Cannot read video frames from a list of {type(head).__name__}.')

    raise _pipelines.UnsupportedPipelineConfigError(
        f'Cannot read video frames from {type(value).__name__}.')


def audio_to_numpy(audio) -> numpy.ndarray | None:
    """
    Normalize pipeline audio to a float32 array of shape ``(channels, samples)``.

    At most two channels are kept. ``None`` and empty audio return ``None``.

    :param audio: tensor or array, or ``None``
    :return: planar audio, or ``None``
    """
    if audio is None:
        return None

    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu().numpy()

    audio = numpy.asarray(audio, dtype=numpy.float32)
    if audio.size == 0:
        return None
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    if audio.ndim != 2:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Cannot mux audio of shape {audio.shape}.')

    if audio.shape[0] > 2 and audio.shape[1] <= 2:
        audio = numpy.transpose(audio, (1, 0))
    if audio.shape[0] > 2:
        audio = audio[:2]
    return numpy.ascontiguousarray(audio, dtype=numpy.float32)


def _to_rgb_image(image: PIL.Image.Image) -> PIL.Image.Image:
    if image.mode == 'RGB':
        return image
    converted = image.convert('RGB')
    if converted is not image:
        image.close()
    return converted


def _array_frame_to_image(frame) -> PIL.Image.Image:
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = numpy.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = numpy.transpose(frame, (1, 2, 0))
    if frame.dtype != numpy.uint8:
        frame = frame.astype(numpy.float32)
        finite = frame[numpy.isfinite(frame)]
        if finite.size and float(finite.max()) <= 1.0:
            frame = frame * 255.0
        frame = numpy.clip(frame, 0, 255).astype(numpy.uint8)
    if frame.ndim == 2:
        return PIL.Image.fromarray(frame, mode='L').convert('RGB')
    if frame.shape[-1] == 1:
        return PIL.Image.fromarray(frame.squeeze(-1), mode='L').convert('RGB')
    if frame.shape[-1] == 4:
        return PIL.Image.fromarray(frame, mode='RGBA').convert('RGB')
    return PIL.Image.fromarray(frame[..., :3], mode='RGB')


def _frames_from_array(array: numpy.ndarray) -> list[PIL.Image.Image]:
    array = numpy.asarray(array)
    if array.ndim == 5:
        array = array[0]
    if array.ndim == 3:
        return [_array_frame_to_image(array)]
    if array.ndim != 4:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Cannot read video frames from an array of shape {array.shape}.')

    if array.shape[-1] in (1, 3, 4):
        frames = (array[index] for index in range(array.shape[0]))
    elif array.shape[1] in (1, 3, 4):
        frames = (numpy.transpose(array[index], (1, 2, 0)) for index in range(array.shape[0]))
    else:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Cannot read video frames from an array of shape {array.shape}.')
    return [_array_frame_to_image(frame) for frame in frames]


def conditioning_canvas(source_size,
                        width,
                        height,
                        aspect_correct: bool,
                        align: int) -> tuple[int, int]:
    """
    Generation canvas for one conditioning image or clip.

    This matches still-image ``--output-size`` handling. The requested width
    is kept. Unless aspect correction is off, the height follows the source
    and is aligned down. With no requested size, the source itself is aligned.
    """
    import dgenerate.image as _image

    source_size = (int(source_size[0]), int(source_size[1]))
    align = max(1, int(align))
    if width is None and height is None:
        requested = source_size
        aspect = False
    else:
        requested = (
            int(width) if width is not None else source_size[0],
            int(height) if height is not None else source_size[1])
        aspect = bool(aspect_correct)
    target = _image.resize_image_calc(
        old_size=source_size,
        new_size=requested,
        aspect_correct=aspect,
        align=align)
    if target[0] < align or target[1] < align:
        aligned = max(align, target[0]), max(align, target[1])
        _messages.warning(
            f'Conditioning media {source_size[0]}x{source_size[1]} would produce '
            f'a canvas smaller than the required alignment of {align}; '
            f'forcing dimensions to at least {align} pixels: {aligned[0]}x{aligned[1]}.')
        target = aligned
    return int(target[0]), int(target[1])


def resize_media(frames, target, nearest: bool = False):
    """Resize every frame to ``target``. ``nearest`` keeps mask edges hard."""
    import PIL.Image

    import dgenerate.image as _image

    if not frames:
        return frames
    target = (int(target[0]), int(target[1]))
    if all(frame.size == target for frame in frames):
        return frames
    algo = PIL.Image.Resampling.NEAREST if nearest else None
    return [
        _image.resize_image(frame, target, aspect_correct=False, algo=algo)
        for frame in frames]


def _force_output_size_alignment(width: int | None,
                                 height: int | None,
                                 align: int,
                                 model_name: str) -> tuple[int | None, int | None]:
    if width is None or height is None:
        return width, height
    width, height = int(width), int(height)
    aligned = (
        max(align, width - width % align),
        max(align, height - height % align))
    if aligned != (width, height):
        _messages.warning(
            f'{model_name} output size {width}x{height} is not aligned to {align}; '
            f'forcing dimensions to {aligned[0]}x{aligned[1]}.')
    return aligned
