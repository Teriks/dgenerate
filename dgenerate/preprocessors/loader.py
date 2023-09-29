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
import inspect
import sys
import typing

from .preprocessor import ImagePreprocessor
from .preprocessorchain import ImagePreprocessorChain
from ..textprocessing import ConceptPathParser, ConceptPathParseError
from .exceptions import ImagePreprocessorArgumentError

def _load(path):
    name = path.split(';', 1)[0].strip()

    classes = [cls for cls in list_available() if name in names_from_class(cls)]

    if len(classes) > 1:
        raise RuntimeError(f'Found more than one ImagePreprocessor with the name: {name}')

    if classes:
        class_to_create = classes[0]

        parser_accepted_args = [arg for arg in
                                inspect.getfullargspec(class_to_create.__init__).args if arg != 'self']

        for arg in parser_accepted_args.copy():
            parser_accepted_args.append(arg.replace('_', '-'))

        arg_parser = ConceptPathParser("Image Preprocessor", parser_accepted_args)

        try:
            parsed_args = arg_parser.parse_concept_path(path).args
        except ConceptPathParseError as e:
            raise ImagePreprocessorArgumentError(e)

        args_dict = {}

        for k, v in parsed_args.items():
            fixed_key = k.replace('-', '_')

            if fixed_key in args_dict:
                raise ImagePreprocessorArgumentError(
                    f'Duplicate argument "{fixed_key}" given to image preprocessor "{name}".')
            else:
                args_dict[fixed_key] = v

        try:
            return classes[0](**args_dict)
        except ImagePreprocessorArgumentError as e:
            raise ImagePreprocessorArgumentError(f'Invalid argument given to image preprocessor "{name}": {e}')

    raise RuntimeError(f'Found no ImagePreprocessor derived classes with the name: {name}')


def list_available():
    mod = sys.modules['dgenerate.preprocessors']

    def _excluded(cls):
        if not inspect.isclass(cls):
            return True
        if not issubclass(cls, ImagePreprocessor):
            return True

        if hasattr(cls, 'HIDDEN'):
            return cls.HIDDEN
        else:
            return False

    classes = [cls for cls in mod.__dict__.values() if not _excluded(cls)]

    return classes


def names_from_class(cls):
    if not issubclass(cls, ImagePreprocessor):
        raise ValueError(
            'provided class is not a subclass of dgenerate.preprocessors.ImagePreprocessor')

    if hasattr(cls, 'NAMES'):
        if isinstance(cls.NAMES, str):
            return {cls.NAMES}
        else:
            return cls.NAMES
    else:
        return {cls.__name__}


def load(path: typing.Union[str, list, tuple, None]):
    if path is None:
        return None

    if isinstance(path, str):
        return _load(path)

    if len(path) == 1:
        return _load(path[0])

    chain = ImagePreprocessorChain()
    for i in path:
        chain.add_processor(_load(i))

    return chain
