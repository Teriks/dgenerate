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

import collections.abc
import datetime
import functools
import os.path
import pathlib
import tempfile
import time
import typing

import PIL.Image

import dgenerate.filelock as _filelock
import dgenerate.files as _files
import dgenerate.image_process.renderloopconfig as _renderloopconfig
import dgenerate.imageprocessors as _imageprocessors
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.messages as _messages
import dgenerate.types as _types
from dgenerate.events import \
    Event, \
    AnimationFinishedEvent, \
    StartingGenerationStepEvent, \
    AnimationETAEvent, \
    StartingAnimationEvent, \
    StartingAnimationFileEvent


class AnimationFileFinishedEvent(Event):
    """
    Generated in the event stream of :py:meth:`.ImageProcessRenderLoop.events`

    Occurs when an animation (video or animated image) has finished being written to disk.
    """

    path: str
    """
    Path on disk where the video/animated image was saved.
    """

    starting_event: StartingAnimationFileEvent
    """
    Animation :py:class:`.StartingAnimationFileEvent` related to this file finished event.
    """

    def __init__(self, origin: 'ImageProcessRenderLoop', path: str, starting_event: StartingAnimationFileEvent):
        super().__init__(origin)
        self.starting_event = starting_event
        self.path = path


class ImageGeneratedEvent(Event):
    """
    Generated in the event stream of :py:meth:`.ImageProcessRenderLoop.events`

    Occurs when an image is generated (but not saved yet).
    """

    image: PIL.Image.Image
    """
    The generated image.
    """

    generation_step: int
    """
    The current generation step. (zero indexed)
    """

    suggested_directory: str
    """
    A suggested directory path for saving this image in.
    
    A value of ``'.'`` may be present, this indicates the current working directory.
    """

    suggested_filename: str
    """
    A suggested filename for saving this image as. This filename will be unique
    to the render loop run / configuration. This is just the filename, it will
    not contain a directory name.
    """

    is_animation_frame: bool
    """
    Is this image a frame in an animation?
    """

    frame_index: _types.OptionalInteger
    """
    The frame index if this is an animation frame.
    """

    def __init__(self,
                 origin: 'ImageProcessRenderLoop',
                 image: PIL.Image.Image,
                 generation_step: int,
                 suggested_directory: str,
                 suggested_filename: str,
                 is_animation_frame=False,
                 frame_index: _types.OptionalInteger = None):
        super().__init__(origin)

        self.image = image
        self.generation_step = generation_step
        self.suggested_directory = suggested_directory if suggested_directory.strip() else '.'
        self.suggested_filename = suggested_filename
        self.is_animation_frame = is_animation_frame
        self.frame_index = frame_index


class ImageFileSavedEvent(Event):
    """
    Generated in the event stream of :py:meth:`.ImageProcessRenderLoop.events`

    Occurs when an image file is written to disk.
    """

    generated_event: ImageGeneratedEvent
    """
    The :py:class:`.ImageGeneratedEvent` for the image that was saved.
    """

    path: str
    """
    Path to the saved image.
    """

    def __init__(self, origin: 'ImageProcessRenderLoop', generated_event, path):
        super().__init__(origin)
        self.generated_event = generated_event
        self.path = path


RenderLoopEvent = \
    typing.Union[ImageGeneratedEvent,
    StartingAnimationEvent,
    StartingAnimationFileEvent,
    AnimationFileFinishedEvent,
    ImageFileSavedEvent,
    AnimationFinishedEvent,
    StartingGenerationStepEvent,
    AnimationETAEvent]
"""
Possible events from the event stream created by :py:meth:`.ImageProcessRenderLoop.events`
"""

RenderLoopEventStream = typing.Generator[RenderLoopEvent, None, None]
"""
Event stream created by :py:meth:`.ImageProcessRenderLoop.events`
"""


class ImageProcessRenderLoop:
    """
    Implements the behavior of the ``image-process`` sub-command as well as ``\\image_process`` directive.
    """

    image_processor_loader: _imageprocessors.ImageProcessorLoader
    """
    The loader responsible for loading user specified image processors
    """

    message_header: str = 'image-process'
    """
    Used as the header for messages written via :py:mod:`dgenerate.messages`
    """

    disable_writes: bool = False
    """
    Disable or enable all writes to disk, if you intend to only ever use the event
    stream of the render loop when using dgenerate as a library, this is a useful option.
    
    :py:attr:`RenderLoop.last_images` and :py:attr:`last_animations` will not be available
    if writes to disk are disabled.
    """

    config: _renderloopconfig.ImageProcessRenderLoopConfig = None
    """
    Render loop configuration.
    """

    def __init__(self,
                 config: _renderloopconfig.ImageProcessRenderLoopConfig = None,
                 image_processor_loader: _imageprocessors.ImageProcessorLoader | None = None,
                 message_header: str = 'image-process',
                 disable_writes: bool = False):
        """
        :param config: :py:class:`.ImageProcessRenderLoopConfig`. If ``None`` is provided, a
            :py:class:`.ImageProcessRenderLoopConfig` instance will be created and assigned to
            :py:attr:`.ImageProcessRenderLoop.config`.

        :param image_processor_loader: :py:class:`dgenerate.imageprocessors.ImageProcessorLoader`.
            If ``None`` is provided, an instance will be created and assigned to
            :py:attr:`.ImageProcessRenderLoop.image_processor_loader`.

        :param message_header: Used as the header for messages written via :py:mod:`dgenerate.messages`

        :param disable_writes: Disable or enable all writes to disk, if you intend to
            only ever use the event stream of the render loop when using dgenerate as a
            library, this is a useful option. :py:attr:`.ImageProcessRenderLoop.written_images` and
            :py:attr:`.ImageProcessRenderLoop.written_animations` will not be available if
            writes to disk are disabled.
        """

        if config is None:
            self.config = _renderloopconfig.ImageProcessRenderLoopConfig()
        else:
            self.config = config
            
        self._c_config = None

        if image_processor_loader is None:
            self.image_processor_loader = _imageprocessors.ImageProcessorLoader()
        else:
            self.image_processor_loader = image_processor_loader

        self._written_images: _files.GCFile | None = None
        self._written_animations: _files.GCFile | None = None
        self._iterating = False

        self.message_header = message_header
        self.disable_writes = disable_writes

    @property
    def written_images(self) -> collections.abc.Iterable[str]:
        """
        Iterable over image filenames written by the last run
        """

        class Iterable:
            def __init__(self, images):
                self.images = images

            def __iter__(self):
                if self.images is None:
                    return

                self.images.seek(0)
                for line in self.images:
                    yield line.rstrip('\n')

        return Iterable(self._written_images)

    @property
    def written_animations(self) -> collections.abc.Iterable[str]:
        """
        Iterable over animation filenames written by the last run
        """

        class Iterable:
            def __init__(self, animations):
                self.animations = animations

            def __iter__(self):
                if self.animations is None:
                    return

                self.animations.seek(0)
                for line in self.animations:
                    yield line.rstrip('\n')

        return Iterable(self._written_animations)

    def _record_save_image(self, filename):
        self._written_images.write(pathlib.Path(filename).absolute().as_posix() + '\n')

    def _record_save_animation(self, filename):
        self._written_animations.write(pathlib.Path(filename).absolute().as_posix() + '\n')

    def _process_reader(self, file, reader: _mediainput.MediaReader, out_filename, generation_step):
        out_directory = os.path.dirname(out_filename)

        duplicate_output_suffix = '_duplicate_'

        if out_directory and not self.disable_writes:
            pathlib.Path(out_directory).mkdir(
                parents=True, exist_ok=True)

        _messages.log(fr'{self.message_header}: Processing "{file}"',
                      underline=True)

        if reader.total_frames == 1:

            if not self._c_config.output_overwrite and not self.disable_writes:
                out_filename = _filelock.touch_avoid_duplicate(
                    out_directory if out_directory else '.',
                    path_maker=_filelock.suffix_path_maker(out_filename, duplicate_output_suffix))

            # Processing happens here, when the frame is read
            with next(reader) as processed_image:

                generated_event = ImageGeneratedEvent(
                    origin=self,
                    image=processed_image,
                    suggested_directory=os.path.dirname(out_filename),
                    suggested_filename=os.path.basename(out_filename),
                    generation_step=generation_step)

                yield generated_event

                if not self.disable_writes:
                    processed_image.save(out_filename)
                    self._record_save_image(out_filename)
                    yield ImageFileSavedEvent(origin=self,
                                              generated_event=generated_event,
                                              path=out_filename)

                    _messages.log(fr'{self.message_header}: Wrote Image "{out_filename}"',
                                  underline=True)
        else:
            out_filename_base, ext = os.path.splitext(out_filename)

            if not self._c_config.output_overwrite and not self.disable_writes:
                out_anim_name = _filelock.touch_avoid_duplicate(
                    out_directory if out_directory else '.',
                    path_maker=_filelock.suffix_path_maker(out_filename, duplicate_output_suffix))
            else:
                out_anim_name = out_filename

            if not self._c_config.no_animation_file and not self.disable_writes:
                anim_writer = _mediaoutput.create_animation_writer(
                    animation_format=ext.lstrip('.'),
                    out_filename=out_anim_name,
                    fps=reader.fps)
            else:
                # mock
                anim_writer = _mediaoutput.AnimationWriter()

            starting_animation_event = StartingAnimationEvent(
                origin=self,
                total_frames=reader.total_frames,
                fps=reader.fps,
                frame_duration=reader.frame_duration)

            yield starting_animation_event

            starting_animation_file_event = None
            if not self._c_config.no_animation_file and not self.disable_writes:
                starting_animation_file_event = StartingAnimationFileEvent(
                    origin=self,
                    path=out_anim_name,
                    fps=reader.fps,
                    frame_duration=reader.frame_duration,
                    total_frames=reader.total_frames
                )
                yield starting_animation_file_event

            with anim_writer as writer:

                for frame_idx in range(0, reader.total_frames):

                    if self._last_frame_time == 0:
                        eta = None
                    else:
                        self._frame_time_sum += time.time() - self._last_frame_time
                        eta_seconds = (self._frame_time_sum / frame_idx) * (
                                reader.total_frames - frame_idx)
                        eta = datetime.timedelta(seconds=eta_seconds)
                    self._last_frame_time = time.time()

                    eta_str = str(eta) if eta is not None else 'tbd...'

                    _messages.log(
                        fr'{self.message_header}: Processing Frame {frame_idx + 1}/{reader.total_frames}, Completion ETA: {eta_str}')

                    if eta is not None:
                        yield AnimationETAEvent(origin=self,
                                                frame_index=frame_idx,
                                                total_frames=reader.total_frames,
                                                eta=eta)

                    frame_filename = out_filename_base + f'_frame_{frame_idx + 1}.{self._c_config.frame_format}'

                    # Processing happens here, when the frame is read
                    with next(reader) as frame:

                        frame_generated_event = ImageGeneratedEvent(
                            origin=self,
                            image=frame,
                            generation_step=generation_step,
                            suggested_directory=os.path.dirname(out_filename_base),
                            suggested_filename=os.path.basename(frame_filename),
                            is_animation_frame=True,
                            frame_index=frame_idx
                        )
                        yield frame_generated_event

                        if not self._c_config.no_animation_file:
                            writer.write(frame)

                        if not self._c_config.no_frames and not self.disable_writes:

                            # frames do not get the _processed_ suffix in any case

                            if not self._c_config.output_overwrite:
                                frame_filename = _filelock.touch_avoid_duplicate(
                                    out_directory if out_directory else '.',
                                    path_maker=_filelock.suffix_path_maker(frame_filename,
                                                                           duplicate_output_suffix))

                            frame.save(frame_filename)
                            self._record_save_image(frame_filename)

                            yield ImageFileSavedEvent(
                                origin=self,
                                path=frame_filename,
                                generated_event=frame_generated_event)

                            _messages.log(fr'{self.message_header}: Wrote Frame "{frame_filename}"')

                    frame_idx += 1

            yield AnimationFinishedEvent(
                origin=self,
                starting_event=starting_animation_event)

            if not self._c_config.no_animation_file and not self.disable_writes:
                self._record_save_animation(out_filename)

                yield AnimationFileFinishedEvent(
                    origin=self,
                    path=out_filename,
                    starting_event=starting_animation_file_event)

                _messages.log(fr'{self.message_header}: Wrote File "{out_anim_name}"',
                              underline=True)

    def _process_file(self, file, out_filename, generation_step, total_generation_steps, processor):
        with _mediainput.MediaReader(
                path=file,
                image_processor=processor,
                resize_resolution=self._c_config.resize,
                aspect_correct=not self._c_config.no_aspect,
                align=self._c_config.align,
                frame_start=self._c_config.frame_start,
                frame_end=self._c_config.frame_end,
                path_opener=functools.partial(
                    _mediainput.fetch_media_data_stream,
                    local_files_only=self._c_config.offline_mode)) as reader:
            self._last_frame_time = 0
            self._frame_time_sum = 0

            yield StartingGenerationStepEvent(origin=self,
                                              generation_step=generation_step,
                                              total_steps=total_generation_steps)

            yield from self._process_reader(file, reader, out_filename, generation_step)

    def _run(self) -> RenderLoopEventStream:
        self._c_config = self.config.copy()
        self._c_config.check()

        self._written_images = _files.GCFile(
            tempfile.TemporaryFile('w+t'))
        self._written_animations = _files.GCFile(
            tempfile.TemporaryFile('w+t'))

        total_generation_steps = len(self._c_config.input)

        def _is_dir_spec(path):
            return os.path.isdir(path) or path[-1] in '/\\'

        if self._c_config.processors:
            processor = self.image_processor_loader.load(
                self._c_config.processors,
                device=self._c_config.device,
                local_files_only=self._c_config.offline_mode
            )
        else:
            processor = None

        try:
            if self._c_config.output and len(self._c_config.output) == 1 and _is_dir_spec(self._c_config.output[0]):
                for idx, file in enumerate(self._c_config.input):
                    file = _mediainput.url_aware_normpath(file)
                    base, ext = os.path.splitext(_mediainput.url_aware_basename(file))
                    output_file = os.path.normpath(
                        os.path.join(self._c_config.output[0], base + f'_processed_{idx + 1}{ext}'))
                    yield from self._process_file(file, output_file, idx, total_generation_steps, processor)
            else:
                for idx, file in enumerate(self._c_config.input):
                    file = _mediainput.url_aware_normpath(file)
                    output_file = _mediainput.url_aware_normpath(
                        self._c_config.output[idx] if self._c_config.output else file)

                    if file == output_file and not self._c_config.output_overwrite:
                        if not _mediainput.is_downloadable_url(file):
                            base, ext = os.path.splitext(output_file)
                        else:
                            base, ext = os.path.splitext(_mediainput.url_aware_basename(output_file))
                        output_file = base + f'_processed_{idx + 1}{ext}'
                    elif _is_dir_spec(output_file):
                        base, ext = os.path.splitext(_mediainput.url_aware_basename(file))
                        output_file = os.path.join(output_file, base + f'_processed_{idx + 1}{ext}')

                    yield from self._process_file(file, output_file, idx, total_generation_steps, processor)
        finally:
            if processor is not None:
                processor.to('cpu')

    def run(self):
        """
        Run the render loop, this calls :py:meth:`ImageProcessRenderLoopConfig.check`
        on a copy of your config prior to running.

        :raises dgenerate.OutOfMemoryError: if the execution device runs out of memory

        :raises ImageProcessRenderLoopConfigError: on config errors
        """
        for _ in self._run():
            continue

    def events(self) -> RenderLoopEventStream:
        """
        Run the render loop, and iterate over a stream of event objects produced by the render loop.

        This calls :py:meth:`ImageProcessRenderLoopConfig.check` on a copy of your configuration prior to running.

        Event objects are of the union type :py:class:`.RenderLoopEvent`

        The exceptions mentioned here are those you may encounter upon iterating,
        they will not occur upon simple acquisition of the event stream iterator.

        :raises dgenerate.OutOfMemoryError: if the execution device runs out of memory

        :raises ImageProcessRenderLoopConfigError: on config errors

        :return: :py:class:`.RenderLoopEventStream`
        """
        try:
            self._iterating = True
            yield from self._run()
        finally:
            self._iterating = False


__all__ = _types.module_all()
