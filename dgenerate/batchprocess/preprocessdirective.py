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
import typing

import PIL.Image

import dgenerate.batchprocess.batchprocessordirective as _batchprocessordirective
import dgenerate.preprocessors

_parser = argparse.ArgumentParser(r'\preprocess', exit_on_error=False)

_parser.add_argument('file')
_parser.add_argument('-pp', '--preprocessors', nargs='+')
_parser.add_argument('-o', '--output-file', default=None)
_parser.add_argument('-r', '--resize', default=None, type=dgenerate.arguments._type_size)
_parser.add_argument('-a', '--no-aspect', action='store_true')


class PreprocessDirective(_batchprocessordirective.BatchProcessorDirective):
    NAMES = ['preprocess']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, args: typing.List[str]):
        parsed = _parser.parse_args(args)
        loader = dgenerate.preprocessors.Loader()

        loader.load_plugin_modules(self.injected_plugin_modules)

        processor = dgenerate.preprocessors.ImagePreprocessorMixin(loader.load(parsed.preprocessors))

        with PIL.Image.open(parsed.file) as img:
            img = processor.preprocess_image(
                img,
                resize_to=parsed.resize,
                aspect_correct=not parsed.no_aspect)
            if parsed.output_file:
                img.save(parsed.output_file)
            else:
                img.save(parsed.file)
