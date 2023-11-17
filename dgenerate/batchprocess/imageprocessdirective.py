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
import time
import typing

import dgenerate.batchprocess.batchprocessplugin as _batchprocessplugin
import dgenerate.filelock as _filelock
import dgenerate.imageprocessors
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.messages as _messages

_parser = argparse.ArgumentParser(
    r'\image_process',
    description='This directive allows you to use dgenerate image processors directly on files of your choosing.',
    exit_on_error=False,
    allow_abbrev=False)


def _type_align(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 1:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 1')
    return val


_parser.add_argument('file',
                     help='Input file path, may be a static image or animated file supported by dgenerate. '
                          'URLs will be downloaded.')

_parser.add_argument('-p', '--processors', nargs='+',
                     help='One or more image processor URIs.')

_parser.add_argument('-o', '--output', default=None,
                     help='Output file, directories will be created for you. '
                          'If you do not specify an output file, the input file will be modified if it exists on disk. '
                          'Failure to specify an output file with a URL as input is an error.')

_parser.add_argument('-if', '--frame-format', default='png',
                     help='Image format for animation frames.')

_parser.add_argument('-ox', '--output-overwrite', action='store_true',
                     help='Indicate that it is okay to overwrite files, instead of appending a duplicate suffix.')

_parser.add_argument('-r', '--resize', default=None, type=dgenerate.arguments._type_size,
                     help='Preform naive image resizing (LANCZOS).')

_parser.add_argument('-a', '--no-aspect', action='store_true',
                     help='Make --resize ignore aspect ratio.')

_parser.add_argument('-al', '--align', default=8, type=_type_align,
                     help='Align images / videos to this value in pixels, default is 8. '
                          'Specifying 1 will disable resolution alignment.')

_parser.add_argument('-d', '--device', default='cuda', type=dgenerate.arguments._type_device,
                     help='Processing device, for example "cuda", "cuda:1".')

write_types = _parser.add_mutually_exclusive_group()

write_types.add_argument('-nf', '--no-frames', action='store_true',
                         help='Do not write frames, only an animation file. Cannot be used with --no-animation.')

write_types.add_argument('-na', '--no-animation', action='store_true',
                         help='Do not write an animation file, only frames. Cannot be used with --no-frames.')


class ImageProcessDirective(_batchprocessplugin.BatchProcessPlugin):
    NAMES = ['image_process']

    def __init__(self, allow_exit=False, **kwargs):
        super().__init__(**kwargs)
        self._allow_exit = self.get_bool_arg('allow_exit', allow_exit)

    def _process_reader(self, reader: _mediainput.AnimationReader, out_filename):

        out_directory = os.path.dirname(out_filename)

        if out_directory:
            pathlib.Path(out_directory).mkdir(
                parents=True, exist_ok=True)

        _messages.log(fr'\image_process Processing "{self._parsed_args.file}"',
                      underline=True)

        if reader.total_frames == 1:

            if not self._parsed_args.output_overwrite:
                out_filename = _filelock.touch_avoid_duplicate(
                    out_directory if out_directory else '.',
                    path_maker=_filelock.suffix_path_maker(out_filename, '_duplicate_'))

            next(reader).save(out_filename)

            _messages.log(fr'\image_process Wrote File "{out_filename}"',
                          underline=True)
        else:
            out_filename_base, ext = os.path.splitext(out_filename)

            if not self._parsed_args.output_overwrite:
                out_anim_name = _filelock.touch_avoid_duplicate(
                    out_directory if out_directory else '.',
                    path_maker=_filelock.suffix_path_maker(out_filename, '_duplicate_'))
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
                        fr'\image_process Processing Frame {frame_idx + 1}/{reader.total_frames}, Completion ETA: {eta}')

                    # Processing happens here
                    frame = next(reader)

                    if not self._parsed_args.no_animation:
                        writer.write(frame)

                    if not self._parsed_args.no_frames:
                        frame_name = out_filename_base + f'_frame_{frame_idx + 1}.{self._parsed_args.frame_format}'

                        if not self._parsed_args.output_overwrite:
                            frame_name = _filelock.touch_avoid_duplicate(
                                out_directory if out_directory else '.',
                                path_maker=_filelock.suffix_path_maker(frame_name, '_duplicate_'))

                        frame.save(frame_name)

                        _messages.log(fr'\image_process Wrote Frame "{frame_name}"')

                    frame_idx += 1

                _messages.log(fr'\image_process Wrote File "{out_filename}"',
                              underline=True)

    def directive_lookup(self, name) -> typing.Optional[typing.Callable[[typing.List[str]], None]]:
        if name == 'image_process':
            return lambda args: self._image_process(args)
        return None

    def _image_process(self, args: typing.List[str]):
        try:
            self._parsed_args = _parser.parse_args(args)
        except SystemExit as e:
            _messages.log()  # newline
            if self._allow_exit:
                raise e
            return

        out_filename = self._parsed_args.output if self._parsed_args.output else self._parsed_args.file

        if out_filename.startswith('http') or out_filename.startswith('https'):
            self.argument_error('--output cannot be a URL, please specify --output manually.')

        loader = dgenerate.imageprocessors.ImageProcessorLoader()

        loader.load_plugin_modules(self.plugin_module_paths)

        if self._parsed_args.processors:
            processor = loader.load(self._parsed_args.processors, device=self._parsed_args.device)
        else:
            processor = None

        stream_def = _mediainput.fetch_media_data_stream(self._parsed_args.file)

        with stream_def[1], _mediainput.create_animation_reader(
                mimetype=stream_def[0],
                file=stream_def[1],
                file_source=self._parsed_args.file,
                resize_resolution=self._parsed_args.resize,
                aspect_correct=not self._parsed_args.no_aspect,
                align=self._parsed_args.align,
                image_processor=processor) as reader:
            self._last_frame_time = 0
            self._frame_time_sum = 0

            try:
                self._process_reader(reader, out_filename)
            except dgenerate.mediaoutput.UnknownAnimationFormatError as e:
                self.argument_error(fr'\image_process error: {e}')
