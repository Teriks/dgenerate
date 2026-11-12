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
import datetime

import dgenerate.types as _types

__doc__ = """
Common render loop events.
"""


class Event:
    """
    Base class for event stream objects.
    """

    def __init__(self, origin):
        self.origin = origin

    def __str__(self):
        return f'{self.__class__.__name__}({_types.get_public_attributes(self)})'


class AnimationETAEvent(Event):
    """
    Common event stream object produced by the ``events()`` event stream of a render loop.

    Occurs when there is an update about the estimated finish time of an animation being generated.
    """
    frame_index: int
    """
    Frame index at which the ETA was calculated.
    """

    total_frames: int
    """
    Total frames needed for the animation to complete.
    """

    eta: datetime.timedelta
    """
    Current estimated time to complete the animation.
    """

    def __init__(self,
                 origin,
                 frame_index: int,
                 total_frames: int,
                 eta: datetime.timedelta):
        super().__init__(origin)
        self.frame_index = frame_index
        self.total_frames = total_frames
        self.eta = eta


class StartingGenerationStepEvent(Event):
    """
    Common event stream object produced by the ``events()`` event stream of a render loop.

    Occurs when a generation step is starting, a generation step may produce multiple images and or an animation.
    """
    generation_step: int
    """
    The generation step number.
    """

    total_steps: int
    """
    The total number of steps that are needed to complete the render loop.
    """

    def __init__(self,
                 origin,
                 generation_step: int, total_steps: int):
        super().__init__(origin)
        self.generation_step = generation_step
        self.total_steps = total_steps


class StartingAnimationEvent(Event):
    """
    Common event stream object produced by the ``events()`` event stream of a render loop.

    Occurs when a sequence of images that belong to an animation are about to start being generated.

    This occurs whether an animation is going to be written to disk or not.
    Assign :py:attr:`file` while handling the event to also receive the animation
    on a binary stream. Writes to disk still happen when they are enabled.

    .. code-block:: python

        for event in render_loop.events():
            if isinstance(event, StartingAnimationEvent):
                event.file = stream
    """

    total_frames: int
    """
    Number of frames written.
    """

    fps: float
    """
    FPS of the generated file.
    """

    frame_duration: float
    """
    Frame duration of the generated file, (the time a frame is visible in milliseconds)
    """

    file = None
    """
    Binary stream that also receives this animation, or a sequence of streams
    when one frame produces several images.

    Set this while handling the event. An mp4 receives each finished frame
    while later frames are still generating, and its soundtrack is written
    as those frames are encoded. GIF and WebP are written when the animation
    finishes. The loop closes the container before
    :py:class:`AnimationFinishedEvent`, and rewinds a seekable stream.

    One stream is one animation file. The next animation yields another
    :py:class:`StartingAnimationEvent`. ``frames`` writes one image per frame
    and does not accept a stream. A video-pipeline clip is written here when
    the clip returns, because its frames arrive together.
    """

    def __init__(self,
                 origin,
                 total_frames: int,
                 fps: float,
                 frame_duration: float):
        super().__init__(origin)
        self.total_frames = total_frames
        self.fps = fps
        self.frame_duration = frame_duration
        self.file = None

    def streams(self) -> list:
        """
        Binary streams assigned to :py:attr:`file`.

        One stream is returned as a one-item list. ``None`` returns an empty list.
        """
        file = self.file
        if file is None:
            return []
        if callable(getattr(file, 'write', None)):
            return [file]
        return list(file)


class StartingAnimationFileEvent(Event):
    """
    Common event stream object produced by the ``events()`` event stream of a render loop.

    Occurs when a sequence of images that belong to an animation are about to start being written to a file.
    """

    path: str
    """
    File path where the animation will reside.
    """

    total_frames: int
    """
    Number of frames written.
    """

    fps: float
    """
    FPS of the generated file.
    """

    frame_duration: float
    """
    Frame duration of the generated file, (the time a frame is visible in milliseconds)
    """

    def __init__(self,
                 origin,
                 path: str,
                 total_frames: int,
                 fps: float,
                 frame_duration: float):
        super().__init__(origin)
        self.path = path
        self.total_frames = total_frames
        self.fps = fps
        self.frame_duration = frame_duration


class AnimationFinishedEvent(Event):
    """
    Common event stream object produced by the ``events()`` event stream of a render loop.

    Occurs when a sequence of images that belong to an animation are done generating.

    This occurs whether an animation was written to disk or not.
    """

    starting_event: StartingAnimationEvent
    """
    Animation :py:class:`.StartingAnimationEvent` related to this file finished event.
    """

    def __init__(self,
                 origin,
                 starting_event: StartingAnimationEvent):
        super().__init__(origin)
        self.starting_event = starting_event


__all__ = _types.module_all()
