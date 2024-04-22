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
import sys
import typing

import torch.cuda

import dgenerate.events as _event
import dgenerate.image_process.arguments as _arguments
import dgenerate.image_process.renderloop as _renderloop
import dgenerate.imageprocessors as _imageprocessors
import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import dgenerate.types as _types


class ImageProcessExitEvent(_event.Event):
    """
    Generated in the event stream created by :py:func:`.invoke_image_process_events`

    Exit with return code event for :py:func:`.invoke_image_process_events`
    """
    return_code: int

    def __init__(self, origin, return_code: int):
        super().__init__(origin)
        self.return_code = return_code


InvokeImageProcessEvent = typing.Union[ImageProcessExitEvent, _renderloop.RenderLoopEvent]
"""
Events yield-able by :py:func:`.invoke_image_process_events`
"""

InvokeImageProcessEventStream = typing.Generator[InvokeImageProcessEvent, None, None]
"""
Event stream produced by :py:func:`.invoke_image_process_events`
"""


def invoke_image_process(
        args: collections.abc.Sequence[str],
        render_loop: typing.Optional[_renderloop.ImageProcessRenderLoop] = None,
        throw: bool = False,
        log_error: bool = True,
        help_raises: bool = False,
        help_name: str = 'image-process',
        help_desc: typing.Optional[str] = None) -> int:
    """
    Invoke image-process using its command line arguments and return a return code.

    image-process is invoked in the current process, this method does not spawn a subprocess.

    :param args: image-process command line arguments in the form of a list, see: shlex module, or sys.argv
    :param render_loop: :py:class:`.ImageProcessRenderLoop` instance,
        if None is provided one will be created.
    :param throw: Whether to throw known exceptions or handle them.
    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?
    :param help_raises: ``--help`` raises :py:exc:`.ImageProcessHelpException` ?
        When ``True``, this will occur even if ``throw=False``
    :param help_name: name used in the ``--help`` output
    :param help_desc: description used in the ``--help`` output, if ``None`` is provided a default value will be used.

    :raises ImageProcessUsageError:
    :raises ImageProcessHelpException:
    :raises dgenerate.ImageProcessorArgumentError:
    :raises dgenerate.ImageProcessorNotFoundError:
    :raises dgenerate.FrameStartOutOfBounds:
    :raises EnvironmentError:

    :return: integer return-code, anything other than 0 is failure
    """

    for event in invoke_image_process_events(**locals()):
        if isinstance(event, ImageProcessExitEvent):
            return event.return_code
    return 0


def invoke_image_process_events(
        args: collections.abc.Sequence[str],
        render_loop: typing.Optional[_renderloop.ImageProcessRenderLoop] = None,
        throw: bool = False,
        log_error: bool = True,
        help_raises: bool = False,
        help_name: str = 'image-process',
        help_desc: typing.Optional[str] = None) -> InvokeImageProcessEventStream:
    """
    Invoke image-process using its command line arguments and return a stream of events.

    image-process is invoked in the current process, this method does not spawn a subprocess.

    The exceptions mentioned here are those you may encounter upon iterating,
    they will not occur upon simple acquisition of the event stream iterator.

    :param args: image-process command line arguments in the form of a list, see: shlex module, or sys.argv
    :param render_loop: :py:class:`.ImageProcessRenderLoop` instance,
        if None is provided one will be created.
    :param throw: Whether to throw known exceptions or handle them.
    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?
    :param help_raises: ``--help`` raises :py:exc:`.ImageProcessHelpException` ?
        When ``True``, this will occur even if ``throw=False``
    :param help_name: name used in the ``--help`` output
    :param help_desc: description used in the ``--help`` output, if ``None`` is provided a default value will be used.

    :raises ImageProcessUsageError:
    :raises ImageProcessHelpException:
    :raises dgenerate.ImageProcessorArgumentError:
    :raises dgenerate.ImageProcessorNotFoundError:
    :raises dgenerate.FrameStartOutOfBounds:
    :raises EnvironmentError:

    :return: :py:data:`.InvokeImageProcessEventStream`
    """

    try:
        try:
            parsed = _arguments.parse_args(args,
                                           help_name=help_name,
                                           help_desc=help_desc,
                                           help_raises=True,
                                           log_error=False)

        except _arguments.ImageProcessHelpException:
            # --help
            if help_raises:
                raise
            yield ImageProcessExitEvent(invoke_image_process_events, 0)
            return

        render_loop = _renderloop.ImageProcessRenderLoop() if render_loop is None else render_loop
        render_loop.config = parsed

        render_loop.image_processor_loader.load_plugin_modules(parsed.plugin_module_paths)

        yield from render_loop.events()
    except (_arguments.ImageProcessUsageError,
            _imageprocessors.ImageProcessorArgumentError,
            _imageprocessors.ImageProcessorNotFoundError,
            _mediainput.FrameStartOutOfBounds,
            torch.cuda.OutOfMemoryError,
            EnvironmentError) as e:
        if log_error:
            _messages.log(f'{help_name}: error: {str(e).strip()}', level=_messages.ERROR)
        if throw:
            raise
        yield ImageProcessExitEvent(invoke_image_process_events, 1)
        return
    yield ImageProcessExitEvent(invoke_image_process_events, 0)


__all__ = _types.module_all()
