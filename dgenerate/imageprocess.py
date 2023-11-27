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

import argparse
import collections.abc
import datetime
import os.path
import pathlib
import tempfile
import time
import typing

import dgenerate.arguments as _arguments
import dgenerate.filelock as _filelock
import dgenerate.imageprocessors as _imageprocessors
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


def _type_align(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 1:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 1')
    return val


class ImageProcessHelpException(Exception):
    pass


class _ImageProcessUnknownArgumentError(Exception):
    pass


def _create_arg_parser(prog, description):
    if description is None:
        description = 'This command allows you to use dgenerate image processors directly on files of your choosing.'

    parser = argparse.ArgumentParser(
        prog,
        description=description,
        exit_on_error=False,
        allow_abbrev=False)

    def _exit(status=0, message=None):
        if status == 0:
            # help
            raise ImageProcessHelpException('image-process --help used.')
        raise _ImageProcessUnknownArgumentError(message)

    parser.exit = _exit

    actions = []

    actions.append(parser.add_argument(
        'files', nargs='+',
        help='Input file paths, may be a static images or animated files supported by dgenerate. '
             'URLs will be downloaded.'))

    actions.append(parser.add_argument(
        '-p', '--processors', nargs='+',
        help='One or more image processor URIs, specifying multiple will chain them together.'))

    actions.append(
        parser.add_argument('--plugin-modules', action='store', default=[], nargs="+",
                            dest='plugin_module_paths',
                            metavar="PATH",
                            help="""Specify one or more plugin module folder paths (folder containing __init__.py) or 
                            python .py file paths to load as plugins. Plugin modules can implement image processors."""))

    actions.append(parser.add_argument(
        '-o', '--output', nargs='+', default=None,
        help="""Output files, directories will be created for you.
        If you do not specify output files, the output file will be placed next to the input file with the 
        added suffix '_processed_N' unless --output-overwrite is specified, in that case it will be overwritten. 
        If you specify multiple input files and output files, you must specify an output file for every input file, 
        or a directory (indicated with a trailing directory seperator character, for example "my_dir/" or "my_dir\"). 
        Failure to specify an output file with a URL as an input is considered an error. Supported file extensions 
        for image output are equal to those listed under --frame-format."""))

    actions.append(parser.add_argument(
        '-ff', '--frame-format', default='png', type=_arguments._type_image_format,
        help=f'Image format for animation frames. '
             f'Must be one of: {_textprocessing.oxford_comma(_mediaoutput.supported_static_image_formats(), "or")}.'))

    actions.append(parser.add_argument(
        '-ox', '--output-overwrite', action='store_true',
        help='Indicate that it is okay to overwrite files, instead of appending a duplicate suffix.'))

    actions.append(parser.add_argument(
        '-r', '--resize', default=None, type=_arguments._type_size,
        help='Preform naive image resizing (LANCZOS).'))

    actions.append(parser.add_argument(
        '-na', '--no-aspect', action='store_true',
        help='Make --resize ignore aspect ratio.'))

    actions.append(parser.add_argument(
        '-al', '--align', default=8, type=_type_align,
        help="""Align images / videos to this value in pixels, default is 8.
            Specifying 1 will disable resolution alignment."""))

    actions.append(parser.add_argument(
        '-d', '--device', default='cuda', type=_arguments._type_device,
        help='Processing device, for example "cuda", "cuda:1".'))

    actions.append(parser.add_argument('-fs', '--frame-start', default=0, type=_arguments._type_frame_start,
                                       metavar="FRAME_NUMBER",
                                       help="""Starting frame slice point for animated files (zero-indexed), the specified
                         frame will be included.  (default: 0)"""))

    actions.append(parser.add_argument('-fe', '--frame-end', default=None, type=_arguments._type_frame_end,
                                       metavar="FRAME_NUMBER",
                                       help="""Ending frame slice point for animated files (zero-indexed), the specified 
                        frame will be included."""))

    write_types = parser.add_mutually_exclusive_group()

    actions.append(write_types.add_argument(
        '-nf', '--no-frames', action='store_true',
        help='Do not write frames, only an animation file. Cannot be used with --no-animation-file.'))

    actions.append(write_types.add_argument(
        '-naf', '--no-animation-file', action='store_true',
        help='Do not write an animation file, only frames. Cannot be used with --no-frames.'))

    return parser, actions


class ImageProcessConfigError(Exception):
    """
    Raised by :py:meth:`.ImageProcessConfig.check` on validation errors.
    """
    pass


class ImageProcessConfig(_types.SetFromMixin):
    files: _types.Paths
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
    Naive resizing value (LANCZOS), corresponds to ``-r/--resize``
    """

    no_aspect: bool = False
    """
    Disable aspect correction? corresponds to ``-na/--no-aspect``
    """

    align: int = 8
    """
    Forced image alignment, corresponds to ``-al/--align``
    """

    device: str = 'cuda'
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

    def __init__(self):
        self.files = []

    def check(self, attribute_namer: typing.Callable[[str], str] = None):
        """
        Preforms logical validation on the configuration.
        """

        def a_namer(attr_name):
            if attribute_namer:
                return attribute_namer(attr_name)
            return f'{self.__name__}.{attr_name}'

        try:
            _types.type_check_struct(self, attribute_namer)
        except ValueError as e:
            raise ImageProcessConfigError(e)

        if self.no_frames and self.no_animation_file:
            raise ImageProcessConfigError(
                f'{a_namer("no_frames")} and {a_namer("no_animation_file")} are mutually exclusive.')

        if self.frame_end is not None and \
                self.frame_start > self.frame_end:
            raise ImageProcessConfigError(
                f'{a_namer("frame_start")} must be less than or equal to {a_namer("frame_end")}')

        if self.output:
            if len(self.files) != len(self.output) and not (len(self.output) == 1 and self.output[0][-1] in '/\\'):
                raise ImageProcessConfigError(
                    'Mismatched number of file inputs and outputs, and output '
                    'is not single a directory (indicated by a trailing slash).')

        for idx, file in enumerate(self.files):
            if not os.path.exists(file):
                raise ImageProcessConfigError(f'File input "{file}" does not exist.')
            if not os.path.isfile(file):
                raise ImageProcessConfigError(f'File input "{file}" is not a file.')

            input_mime_type = _mediainput.guess_mimetype(file)

            if input_mime_type is None:
                raise ImageProcessConfigError(f'File type of "{file}" could not be determined.')

            if not _mediainput.mimetype_is_supported(input_mime_type):
                raise ImageProcessConfigError(f'File input "{file}" is of unsupported mimetype "{input_mime_type}".')

            if self.output and len(self.output) == len(self.files):
                output_name = self.output[idx]

                if output_name[-1] in '/\\':
                    # directory specification, input dictates the output format
                    continue

                _, output_ext = os.path.splitext(output_name)
                output_ext = output_ext.lstrip('.')

                if not _mediainput.mimetype_is_static_image(input_mime_type):
                    if output_ext not in _mediaoutput.supported_animation_writer_formats():
                        raise ImageProcessConfigError(
                            f'Animated file output "{output_name}" specifies '
                            f'unsupported animation format "{output_ext}".')
                else:
                    if output_ext not in _mediaoutput.supported_static_image_formats():
                        raise ImageProcessConfigError(
                            f'Image file output "{output_name}" specifies '
                            f'unsupported image format "{output_ext}".')

            else:
                _, output_ext = os.path.splitext(file)
                output_ext = output_ext.lstrip('.')
                if not _mediainput.mimetype_is_static_image(input_mime_type):
                    if output_ext not in _mediaoutput.supported_animation_writer_formats():
                        raise ImageProcessConfigError(
                            f'Animated file input "{file}" specifies unsupported animation output format "{output_ext}".')
                else:
                    if output_ext not in _mediaoutput.supported_static_image_formats():
                        raise ImageProcessConfigError(
                            f'Image file input "{file}" specifies unsupported image output format "{output_ext}".')


class ImageProcessArgs(ImageProcessConfig):
    """
    Configuration object for :py:class:`.ImageProcessRenderLoop`
    """

    plugin_module_paths: _types.Paths

    def __init__(self):
        super().__init__()
        self.plugin_module_paths = []


class ImageProcessUsageError(Exception):
    """
    Thrown by :py:func:`.parse_args` on usage errors.
    """
    pass


def config_attribute_name_to_option(name):
    """
    Convert an attribute name of :py:class:`.ImageProcessArgs` into its command line option name.

    :param name: the attribute name
    :return: the command line argument name as a string
    """
    _, actions = _create_arg_parser('image-process', None)

    return {a.dest: a.option_strings[-1] if a.option_strings else a.dest for a in actions}[name]


def parse_args(args: typing.Optional[collections.abc.Sequence[str]] = None,
               help_name='image-process',
               help_desc=None,
               throw: bool = True,
               log_error: bool = True,
               help_raises: bool = False) -> typing.Optional[ImageProcessArgs]:
    """
    Parse and validate the arguments used for ``image-process``, which is a dgenerate
    sub-command as well as config directive.

    :param args: command line arguments
    :param help_name: program name displayed in ``--help`` output.
    :param help_desc: program description displayed in ``--help`` output.
    :param throw: throw :py:exc:`.ImageProcessUsageError` on error? defaults to True
    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?
    :param help_raises: ``--help`` raises :py:exc:`.ImageProcessHelpException` ?

    :raises ImageProcessUsageError:
    :raises ImageProcessHelpException:

    :return: parsed arguments object
    """

    parser = _create_arg_parser(help_name, help_desc)[0]
    parsed = None
    try:
        parsed = typing.cast(ImageProcessArgs, parser.parse_args(args, namespace=ImageProcessArgs()))
        parsed.check()
    except ImageProcessHelpException as e:
        if help_raises:
            raise e
    except (ImageProcessConfigError,
            argparse.ArgumentTypeError,
            argparse.ArgumentError,
            _ImageProcessUnknownArgumentError) as e:
        if log_error:
            _messages.log(f'{help_name}: error: {str(e).strip()}', level=_messages.ERROR)
        if throw:
            raise ImageProcessUsageError(e)
        return None

    if parsed is not None:
        try:
            parsed.check(config_attribute_name_to_option)
        except ImageProcessConfigError as e:
            raise ImageProcessUsageError(e)
    return parsed


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

    def __init__(self, config: ImageProcessConfig = None,
                 image_processor_loader: typing.Optional[_imageprocessors.ImageProcessorLoader] = None):

        if config is None:
            self.config = ImageProcessConfig()
        else:
            self.config = config

        if image_processor_loader is None:
            self.image_processor_loader = _imageprocessors.ImageProcessorLoader()
        else:
            self.image_processor_loader = image_processor_loader

        self._written_animations = None
        self._written_images = None

    @property
    def written_images(self) -> collections.abc.Iterable[str]:
        """
        Iterable over image filenames written by the last run
        """
        loop = self

        class Iterable:
            def __iter__(self):
                if loop._written_images is None:
                    return

                loop._written_images.seek(0)
                for line in loop._written_images:
                    yield line.rstrip('\n')

        return Iterable()

    @property
    def written_animations(self) -> collections.abc.Iterable[str]:
        """
        Iterable over animation filenames written by the last run
        """
        loop = self

        class Iterable:
            def __iter__(self):
                if loop._written_animations is None:
                    return

                loop._written_animations.seek(0)
                for line in loop._written_animations:
                    yield line.rstrip('\n')

        return Iterable()

    def _record_save_image(self, filename):
        self._written_images.write(os.path.abspath(filename) + '\n')

    def _record_save_animation(self, filename):
        self._written_animations.write(os.path.abspath(filename) + '\n')

    def _process_reader(self, file, reader: _mediainput.MultiAnimationReader, out_filename):
        out_directory = os.path.dirname(out_filename)

        duplicate_output_suffix = '_duplicate_'

        if out_directory:
            pathlib.Path(out_directory).mkdir(
                parents=True, exist_ok=True)

        _messages.log(fr'{self.message_header}: Processing "{file}"',
                      underline=True)

        if reader.total_frames == 1:

            if not self.config.output_overwrite:
                out_filename = _filelock.touch_avoid_duplicate(
                    out_directory if out_directory else '.',
                    path_maker=_filelock.suffix_path_maker(out_filename, duplicate_output_suffix))

            next(reader)[0].save(out_filename)
            self._record_save_image(out_filename)

            _messages.log(fr'{self.message_header}: Wrote Image "{out_filename}"',
                          underline=True)
        else:
            out_filename_base, ext = os.path.splitext(out_filename)

            if not self.config.output_overwrite:
                out_anim_name = _filelock.touch_avoid_duplicate(
                    out_directory if out_directory else '.',
                    path_maker=_filelock.suffix_path_maker(out_filename, duplicate_output_suffix))
            else:
                out_anim_name = out_filename

            if not self.config.no_animation_file:
                anim_writer = _mediaoutput.create_animation_writer(
                    animation_format=ext.lstrip('.'),
                    out_filename=out_anim_name,
                    fps=reader.anim_fps)
            else:
                # mock
                anim_writer = _mediaoutput.AnimationWriter()

            with anim_writer as writer:

                for frame_idx in range(0, reader.total_frames):

                    if self._last_frame_time == 0:
                        eta = 'tbd...'
                    else:
                        self._frame_time_sum += time.time() - self._last_frame_time
                        eta_seconds = (self._frame_time_sum / frame_idx) * (
                                reader.total_frames - frame_idx)
                        eta = str(datetime.timedelta(seconds=eta_seconds))
                    self._last_frame_time = time.time()

                    _messages.log(
                        fr'{self.message_header}: Processing Frame {frame_idx + 1}/{reader.total_frames}, Completion ETA: {eta}')

                    # Processing happens here
                    frame = next(reader)[0]

                    if not self.config.no_animation_file:
                        writer.write(frame)

                    if not self.config.no_frames:
                        frame_name = out_filename_base + f'_frame_{frame_idx + 1}.{self.config.frame_format}'

                        # frames do not get the _processed_ suffix in any case

                        if not self.config.output_overwrite:
                            frame_name = _filelock.touch_avoid_duplicate(
                                out_directory if out_directory else '.',
                                path_maker=_filelock.suffix_path_maker(frame_name, duplicate_output_suffix))

                        frame.save(frame_name)
                        self._record_save_image(frame_name)

                        _messages.log(fr'{self.message_header}: Wrote Frame "{frame_name}"')

                    frame_idx += 1

                self._record_save_animation(out_filename)
                _messages.log(fr'{self.message_header}: Wrote File "{out_anim_name}"',
                              underline=True)

    def _process_file(self, file, out_filename):
        if self.config.processors:
            processor = self.image_processor_loader.load(self.config.processors, device=self.config.device)
        else:
            processor = None

        with _mediainput.MultiAnimationReader([
            _mediainput.AnimationReaderSpec(path=file,
                                            image_processor=processor,
                                            resize_resolution=self.config.resize,
                                            aspect_correct=not self.config.no_aspect,
                                            align=self.config.align)],
                frame_start=self.config.frame_start,
                frame_end=self.config.frame_end) as reader:

            self._last_frame_time = 0
            self._frame_time_sum = 0

            self._process_reader(file, reader, out_filename)

    def run(self):
        """
        Run the render loop, this calls :py:meth:`ImageProcessConfig.check` to validate the configuration.
        """
        self.config.check()

        if self._written_images is not None:
            self._written_images.close()

        if self._written_animations is not None:
            self._written_animations.close()

        self._written_images = tempfile.TemporaryFile('w+t')
        self._written_animations = tempfile.TemporaryFile('w+t')

        if self.config.output and len(self.config.output) == 1 and self.config.output[0][-1] in '/\\':
            for idx, file in enumerate(self.config.files):
                base, ext = os.path.splitext(os.path.basename(file))
                output_file = os.path.join(self.config.output[0], base + f'_processed_{idx + 1}{ext}')
                self._process_file(file, output_file)
        else:
            for idx, file in enumerate(self.config.files):
                output_file = self.config.output[idx] if self.config.output else file

                if file == output_file and not self.config.output_overwrite:
                    base, ext = os.path.splitext(output_file)
                    output_file = base + f'_processed_{idx + 1}{ext}'
                elif output_file[-1] in '/\\':
                    base, ext = os.path.splitext(os.path.basename(file))
                    output_file = os.path.join(output_file, base + f'_processed_{idx + 1}{ext}')
                self._process_file(file, output_file)


def invoke_image_process(
        args: collections.abc.Sequence[str],
        render_loop: typing.Optional[ImageProcessRenderLoop] = None,
        throw: bool = False,
        log_error: bool = True,
        help_raises: bool = False,
        help_name: str = 'image-process',
        help_desc: typing.Optional[str] = None):
    """
    Invoke dgenerate using its command line arguments and return a return code.

    dgenerate is invoked in the current process, this method does not spawn a subprocess.

    :param args: image-process command line arguments in the form of a list, see: shlex module, or sys.argv
    :param render_loop: :py:class:`.ImageProcessRenderLoop` instance,
        if None is provided one will be created.
    :param throw: Whether to throw known exceptions or handle them.
    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?
    :param help_raises: ``--help`` raises :py:exc:`.ImageProcessHelpException` ?
    :param help_name: name used in the ``--help`` output
    :param help_desc: description used in the ``--help`` output, if ``None`` is provided a default value will be used.

    :raises ImageProcessUsageError:
    :raises ImageProcessHelpException:
    :raises dgenerate.imageprocessors.ImageProcessorArgumentError:
    :raises dgenerate.imageprocessors.ImageProcessorNotFoundError:
    :raises dgenerate.mediainput.FrameStartOutOfBounds:
    :raises NotImplementedError:
    :raises EnvironmentError:

    :return: integer return-code, anything other than 0 is failure
    """

    try:
        try:
            parsed = parse_args(args,
                                help_name=help_name,
                                help_desc=help_desc,
                                help_raises=True,
                                log_error=False)

        except ImageProcessHelpException:
            # --help
            if help_raises:
                raise
            return 0

        render_loop = ImageProcessRenderLoop() if render_loop is None else render_loop
        render_loop.config = parsed

        render_loop.image_processor_loader.load_plugin_modules(parsed.plugin_module_paths)

        render_loop.run()
    except (ImageProcessUsageError,
            _imageprocessors.ImageProcessorArgumentError,
            _imageprocessors.ImageProcessorNotFoundError,
            _mediainput.FrameStartOutOfBounds,
            NotImplementedError,
            EnvironmentError) as e:
        if log_error:
            _messages.log(f'{help_name}: error: {str(e).strip()}', level=_messages.ERROR)
        if throw:
            raise
        return 1
    return 0


__all__ = _types.module_all()
