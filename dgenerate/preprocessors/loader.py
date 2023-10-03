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
import itertools
import sys
import typing

from .exceptions import ImagePreprocessorArgumentError, ImagePreprocessorNotFoundError
from .preprocessor import ImagePreprocessor
from .preprocessorchain import ImagePreprocessorChain
from ..textprocessing import ConceptPathParser, ConceptPathParseError, dashdown

SEARCH_MODULES = []


def _load(path, device):
    call_by_name = path.split(';', 1)[0].strip()

    preprocessor_class = get_class_by_name(call_by_name)

    inherited_args = ['output-file', 'output-dir', 'device']

    parser_accepted_args = preprocessor_class.get_accepted_args(call_by_name)

    if 'called-by-name' in parser_accepted_args:
        raise RuntimeError(f'called-by-name is a reserved ImagePreprocessor module argument, '
                           'chose another argument name for your module.')

    for inherited_arg in inherited_args:
        if inherited_arg in parser_accepted_args:
            raise RuntimeError(f'{inherited_arg} is a reserved ImagePreprocessor module argument, '
                               'chose another argument name for your module.')

        parser_accepted_args.append(inherited_arg)

    arg_parser = ConceptPathParser("Image Preprocessor", parser_accepted_args)

    try:
        parsed_args = arg_parser.parse_concept_path(path).args
    except ConceptPathParseError as e:
        raise ImagePreprocessorArgumentError(str(e))

    args_dict = {}

    for arg in preprocessor_class.get_default_args(call_by_name):
        args_dict[dashdown(arg[0])] = arg[1]

    for k, v in parsed_args.items():
        args_dict[dashdown(k)] = v

    args_dict['output_dir'] = parsed_args.get('output-dir')
    args_dict['output_file'] = parsed_args.get('output-file')
    args_dict['device'] = parsed_args.get('device', device)
    args_dict['called_by_name'] = call_by_name

    for arg in preprocessor_class.get_required_args(call_by_name):
        if dashdown(arg) not in args_dict:
            raise ImagePreprocessorArgumentError(
                f'Missing required argument "{arg}" for image preprocessor "{call_by_name}".')

    try:
        return preprocessor_class(**args_dict)
    except ImagePreprocessorArgumentError as e:
        raise ImagePreprocessorArgumentError(
            f'Invalid argument given to image preprocessor "{call_by_name}": {e}')


def get_available_classes():
    found_classes = []
    for mod in itertools.chain([sys.modules['dgenerate.preprocessors']], SEARCH_MODULES):
        def _excluded(cls):
            if not inspect.isclass(cls):
                return True

            if cls is ImagePreprocessor:
                return True

            if not issubclass(cls, ImagePreprocessor):
                return True

            if hasattr(cls, 'HIDDEN'):
                return cls.HIDDEN
            else:
                return False

        found_classes += [cls for cls in mod.__dict__.values() if not _excluded(cls)]

    return found_classes


def get_class_by_name(preprocessor_name):
    classes = [cls for cls in get_available_classes() if
               preprocessor_name in cls.get_names()]

    if len(classes) > 1:
        raise RuntimeError(
            f'Found more than one ImagePreprocessor with the name: {preprocessor_name}')

    if not classes:
        raise ImagePreprocessorNotFoundError(
            f'Found no image preprocessor with the name: {preprocessor_name}')

    return classes[0]


def get_all_names():
    names = []
    for cls in get_available_classes():
        names += cls.get_names()
    return names


def get_help(preprocessor_name: str):
    return get_class_by_name(preprocessor_name).get_help(preprocessor_name)


def load(path: typing.Union[str, list, tuple, None], device='cpu'):
    if path is None:
        return None

    if isinstance(path, str):
        return _load(path, device)

    if len(path) == 1:
        return _load(path[0], device)

    chain = ImagePreprocessorChain()
    for i in path:
        chain.add_processor(_load(i, device))

    return chain
