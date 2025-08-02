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

import os
import typing

import dgenerate.webcache as _webcache
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.types as _types
import dgenerate.torchutil as _torchutil


class ImageProcessRenderLoopConfigError(Exception):
    """
    Raised by :py:meth:`.ImageProcessRenderLoopConfig.check` on validation errors.
    """
    pass


class ImageProcessRenderLoopConfig(_types.SetFromMixin):
    input: _types.Paths
    """
    Input file paths.
    """

    output: _types.OptionalPaths = None
    """
    Output file paths, corresponds to ``-o/--output``
    """

    processors: _types.OptionalUris = None
    """
    Image processor URIs, corresponds to ``-p/--processors``
    """

    frame_format: str = 'png'
    """
    Animation frame format, corresponds to ``-ff/-frame-format``
    """

    output_overwrite: bool = False
    """
    Should existing files be overwritten? corresponds to ``-ox/--output-overwrite``
    """

    resize: _types.OptionalSize = None
    """
    Naive resizing value, corresponds to ``-r/--resize``
    """

    no_aspect: bool = False
    """
    Disable aspect correction? corresponds to ``-na/--no-aspect``
    """

    align: int = 1
    """
    Forced image alignment, corresponds to ``-al/--align``
    """

    device: _types.Name = _torchutil.default_device()
    """
    Rendering device, corresponds to ``-d/--device``
    """

    frame_start: int = 0
    """
    Zero indexed inclusive frame slice start, corresponds to ``-fs/--frame-start``
    """

    frame_end: _types.OptionalInteger = None
    """
    Optional zero indexed inclusive frame slice end, corresponds to ``-fe/--frame-end``
    """

    no_frames: bool = False
    """
    Disable frame output when rendering an animation? mutually exclusive with ``no_animation``.
    Corresponds to ``-nf/--no-frames``
    """

    no_animation_file: bool = False
    """
    Disable animated file output when rendering an animation? mutually exclusive with ``no_frames``.
    Corresponds to ``-naf/--no-animation-file``
    """

    offline_mode: bool = False
    """
    Setting to true prevents dgenerate from downloading Hugging Face hub models 
    that do not exist in the disk cache or a folder on disk. Referencing a model on 
    Hugging Face hub that has not been cached because it was not previously downloaded will result 
    in a failure when using this option.
    """

    def __init__(self):
        self.input = []

    def copy(self) -> 'ImageProcessRenderLoopConfig':
        """
        Create a deep copy of this :py:class:`ImageProcessRenderLoopConfig` instance.

        :return: :py:class:`ImageProcessRenderLoopConfig` instance that is a deep copy of this instance.
        """
        new_config = ImageProcessRenderLoopConfig()

        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, (list, tuple, dict, set)):
                new_config.__dict__[attr_name] = _types.partial_deep_copy_container(attr_value)
            elif hasattr(attr_value, 'copy') and callable(getattr(attr_value, 'copy')):
                new_config.__dict__[attr_name] = attr_value.copy()
            else:
                new_config.__dict__[attr_name] = attr_value

        return new_config

    def check(self, attribute_namer: typing.Callable[[str], str] = None):
        """
        Performs logical validation on the configuration.

        This may modify the configuration.
        """

        def a_namer(attr_name):
            if attribute_namer:
                return attribute_namer(attr_name)
            return f'{self.__name__}.{attr_name}'

        try:
            _types.type_check_struct(self, attribute_namer)
        except ValueError as e:
            raise ImageProcessRenderLoopConfigError(e) from e

        if self.no_frames and self.no_animation_file:
            raise ImageProcessRenderLoopConfigError(
                f'{a_namer("no_frames")} and {a_namer("no_animation_file")} are mutually exclusive.')

        if self.frame_end is not None and \
                self.frame_start > self.frame_end:
            raise ImageProcessRenderLoopConfigError(
                f'{a_namer("frame_start")} must be less than or equal to {a_namer("frame_end")}')

        if self.output:
            if len(self.input) != len(self.output) and not (len(self.output) == 1 and self.output[0][-1] in '/\\'):
                raise ImageProcessRenderLoopConfigError(
                    'Mismatched number of file inputs and outputs, and output '
                    'is not single a directory (indicated by a trailing slash).')

        for idx, file in enumerate(self.input):
            if not _mediainput.is_downloadable_url(file):

                if not os.path.exists(file):
                    raise ImageProcessRenderLoopConfigError(f'File input "{file}" does not exist.')
                if not os.path.isfile(file):
                    raise ImageProcessRenderLoopConfigError(f'File input "{file}" is not a file.')

                input_mime_type = _mediainput.guess_mimetype(file)
            else:
                try:
                    input_mime_type = _mediainput.request_mimetype(file, local_files_only=self.offline_mode)
                except _webcache.WebFileCacheOfflineModeException:
                    raise ImageProcessRenderLoopConfigError(
                        f'File input "{file}" is not cached and cannot be downloaded in offline mode.')

            if input_mime_type is None:
                raise ImageProcessRenderLoopConfigError(f'File type of "{file}" could not be determined.')

            if not _mediainput.mimetype_is_supported(input_mime_type):
                raise ImageProcessRenderLoopConfigError(
                    f'File input "{file}" is of unsupported mimetype "{input_mime_type}".')

            if self.output and len(self.output) == len(self.input):
                output_name = self.output[idx]

                if os.path.isdir(output_name) or output_name[-1] in '/\\':
                    # directory specification, input dictates the output format
                    continue

                _, output_ext = os.path.splitext(output_name)
                output_ext = output_ext.lstrip('.').lower()

                if not _mediainput.mimetype_is_static_image(input_mime_type):
                    if output_ext not in _mediaoutput.get_supported_animation_writer_formats():
                        raise ImageProcessRenderLoopConfigError(
                            f'Animated file output "{output_name}" specifies '
                            f'unsupported animation format "{output_ext}".')
                else:
                    if output_ext not in _mediaoutput.get_supported_static_image_formats():
                        raise ImageProcessRenderLoopConfigError(
                            f'Image file output "{output_name}" specifies '
                            f'unsupported image format "{output_ext}".')

            else:
                _, output_ext = _mediainput.url_aware_splitext(file)
                output_ext = output_ext.lstrip('.').lower()

                if not _mediainput.mimetype_is_static_image(input_mime_type):
                    if output_ext not in _mediaoutput.get_supported_animation_writer_formats():
                        raise ImageProcessRenderLoopConfigError(
                            f'Animated file input "{file}" specifies unsupported animation output format "{output_ext}".')
                else:
                    if output_ext not in _mediaoutput.get_supported_static_image_formats():
                        raise ImageProcessRenderLoopConfigError(
                            f'Image file input "{file}" specifies unsupported image output format "{output_ext}".')


__all__ = _types.module_all()
