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
import datetime
import os.path
import pathlib
import tempfile
import time
import typing

import dgenerate.batchprocess.configrunnerplugin as _configrunnerplugin
import dgenerate.filelock as _filelock
import dgenerate.imageprocessors
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.messages as _messages


def _type_align(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 1:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 1')
    return val


def _create_arg_parser(prog, description):
    if description is None:
        description = 'This directive allows you to use dgenerate image processors directly on files of your choosing.'

    parser = argparse.ArgumentParser(
        prog,
        description=description,
        exit_on_error=False,
        allow_abbrev=False)

    parser.add_argument(
        'files', nargs='+',
        help='Input file paths, may be a static images or animated files supported by dgenerate. '
             'URLs will be downloaded.')

    parser.add_argument(
        '-p', '--processors', nargs='+',
        help='One or more image processor URIs.')

    parser.add_argument(
        '-o', '--output', nargs='+', default=None,
        help="""Output files, directories will be created for you.
        If you do not specify output files, the output file will be placed next to the input file with the added 
        suffix '_processed_N', unless --output-overwrite is specified, in which case it will be overwritten. If you 
        specify multiple input files and output files, you must specify an output file for every input file. 
        Failure to specify an output file with a URL as an input is considered an error.""")

    parser.add_argument(
        '-if', '--frame-format', default='png',
        help='Image format for animation frames.')

    parser.add_argument(
        '-ox', '--output-overwrite', action='store_true',
        help='Indicate that it is okay to overwrite files, instead of appending a duplicate suffix.')

    parser.add_argument(
        '-r', '--resize', default=None, type=dgenerate.arguments._type_size,
        help='Preform naive image resizing (LANCZOS).')

    parser.add_argument(
        '-a', '--no-aspect', action='store_true',
        help='Make --resize ignore aspect ratio.')

    parser.add_argument(
        '-al', '--align', default=8, type=_type_align,
        help="""Align images / videos to this value in pixels, default is 8.
            Specifying 1 will disable resolution alignment.""")

    parser.add_argument(
        '-d', '--device', default='cuda', type=dgenerate.arguments._type_device,
        help='Processing device, for example "cuda", "cuda:1".')

    write_types = parser.add_mutually_exclusive_group()

    write_types.add_argument(
        '-nf', '--no-frames', action='store_true',
        help='Do not write frames, only an animation file. Cannot be used with --no-animation.')

    write_types.add_argument(
        '-na', '--no-animation', action='store_true',
        help='Do not write an animation file, only frames. Cannot be used with --no-frames.')

    return parser


class ImageProcessDirective(_configrunnerplugin.ConfigRunnerPlugin):
    NAMES = ['image_process']

    def __init__(self,
                 allow_exit: bool = False,
                 message_header: str = '\\image_process:',
                 help_name: str = '\\image_process',
                 help_desc: typing.Optional[str] = None,
                 **kwargs):
        """
        :param allow_exit: Parsing arguments can result in an ``exit()`` call?
        :param message_header: Header string for informational output.
        :param help_name: Name in ``--help`` output.
        :param help_desc: Override argument parser ``description`` value.

        :param kwargs: plugin base class arguments
        """

        super().__init__(**kwargs)

        self._allow_exit = allow_exit
        self._message_header = message_header

        self._arg_parser = _create_arg_parser(help_name, help_desc)
        self._parsed_args = None

        self._written_animations = None
        self._written_images = None

        self.register_directive('image_process',
                                lambda args: self.image_process(args))

    @property
    def written_images(self) -> typing.Iterator[str]:
        """
        Iterator over image filenames written by the last run
        """
        if self._written_images is None:
            return

        pos = self._written_images.tell()
        self._written_images.seek(0)
        for line in self._written_images:
            yield line.rstrip('\n')
        self._written_images.seek(pos)

    @property
    def written_animations(self) -> typing.Iterator[str]:
        """
        Iterator over animation filenames written by the last run
        """
        if self._written_animations is None:
            return

        pos = self._written_animations.tell()
        self._written_animations.seek(0)
        for line in self._written_animations:
            yield line.rstrip('\n')
        self._written_animations.seek(pos)

    def _record_save_image(self, filename):
        self._written_images.write(os.path.abspath(filename) + '\n')

    def _record_save_animation(self, filename):
        self._written_animations.write(os.path.abspath(filename) + '\n')

    def _process_reader(self, file, reader: _mediainput.AnimationReader, out_filename):

        out_directory = os.path.dirname(out_filename)

        duplicate_output_suffix = '_duplicate_' if file != out_filename else '_processed_'

        if out_directory:
            pathlib.Path(out_directory).mkdir(
                parents=True, exist_ok=True)

        _messages.log(fr'{self._message_header} Processing "{file}"',
                      underline=True)

        if reader.total_frames == 1:

            if not self._parsed_args.output_overwrite:
                out_filename = _filelock.touch_avoid_duplicate(
                    out_directory if out_directory else '.',
                    path_maker=_filelock.suffix_path_maker(out_filename, duplicate_output_suffix))

            next(reader).save(out_filename)
            self._record_save_image(out_filename)

            _messages.log(fr'{self._message_header} Wrote Image "{out_filename}"',
                          underline=True)
        else:
            out_filename_base, ext = os.path.splitext(out_filename)

            if not self._parsed_args.output_overwrite:
                out_anim_name = _filelock.touch_avoid_duplicate(
                    out_directory if out_directory else '.',
                    path_maker=_filelock.suffix_path_maker(out_filename, duplicate_output_suffix))
            else:
                out_anim_name = out_filename

            if not self._parsed_args.no_animation:
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
                        fr'{self._message_header} Processing Frame {frame_idx + 1}/{reader.total_frames}, Completion ETA: {eta}')

                    # Processing happens here
                    frame = next(reader)

                    if not self._parsed_args.no_animation:
                        writer.write(frame)

                    if not self._parsed_args.no_frames:
                        frame_name = out_filename_base + f'_frame_{frame_idx + 1}.{self._parsed_args.frame_format}'

                        # frames do not get the _processed_ suffix in any case

                        if not self._parsed_args.output_overwrite:
                            frame_name = _filelock.touch_avoid_duplicate(
                                out_directory if out_directory else '.',
                                path_maker=_filelock.suffix_path_maker(frame_name, '_duplicate_'))

                        frame.save(frame_name)
                        self._record_save_image(frame_name)

                        _messages.log(fr'{self._message_header} Wrote Frame "{frame_name}"')

                    frame_idx += 1

                self._record_save_animation(out_filename)
                _messages.log(fr'{self._message_header} Wrote File "{out_filename}"',
                              underline=True)

    def _process_file(self, file, out_filename):
        if out_filename.startswith('http') or out_filename.startswith('https'):
            self.argument_error('--output cannot be a URL, please specify --output manually.')

        loader = dgenerate.imageprocessors.ImageProcessorLoader()

        loader.load_plugin_modules(self.plugin_module_paths)

        if self._parsed_args.processors:
            processor = loader.load(self._parsed_args.processors, device=self._parsed_args.device)
        else:
            processor = None

        stream_def = _mediainput.fetch_media_data_stream(file)

        with stream_def[1], _mediainput.create_animation_reader(
                mimetype=stream_def[0],
                file=stream_def[1],
                file_source=file,
                resize_resolution=self._parsed_args.resize,
                aspect_correct=not self._parsed_args.no_aspect,
                align=self._parsed_args.align,
                image_processor=processor) as reader:

            self._last_frame_time = 0
            self._frame_time_sum = 0

            try:
                self._process_reader(file, reader, out_filename)
            except dgenerate.mediaoutput.UnknownAnimationFormatError as e:
                self.argument_error(fr'{self._message_header} error: {e}')

    def image_process(self, args: typing.List[str]):
        """
        Runs the ``\\image_process`` directive.

        :param args: command line arguments
        """

        try:
            self._parsed_args = self._arg_parser.parse_args(args)
        except SystemExit as e:
            _messages.log()  # newline
            if self._allow_exit:
                raise e
            return

        if self._written_images is not None:
            self._written_images.close()

        if self._written_animations is not None:
            self._written_animations.close()

        self._written_images = tempfile.TemporaryFile('w+t')
        self._written_animations = tempfile.TemporaryFile('w+t')

        if self._parsed_args.output:
            if len(self._parsed_args.files) != len(self._parsed_args.output):
                self.argument_error('Mismatched number of file inputs and outputs.')

        try:
            for idx, file in enumerate(self._parsed_args.files):
                self._process_file(file,
                                   self._parsed_args.output[idx] if self._parsed_args.output else file)
        except FileNotFoundError as e:
            self.argument_error(str(e))

        self.set_template_variable('last_images', self.written_images)
        self.set_template_variable('last_animations', self.written_animations)
