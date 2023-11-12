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

import dgenerate.plugin as _plugin
import dgenerate.postprocessors.exceptions as _exceptions
import dgenerate.postprocessors.postprocessor as _postprocessor
import dgenerate.postprocessors.postprocessorchain as _postprocessorchain
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class Loader:
    search_modules: typing.Set
    """Additional module objects for this loader to search, aside from the postprocessors sub module."""

    extra_classes: typing.Set
    """
    Additional directly defined implementation classes. This is empty by default and is for allowing
    library users to quickly add a class implementing :py:class:`dgenerate.postprocessors.postprocessor.ImagePostprocessor`
    if desired without creating a new file for it.
    """

    def __init__(self):
        self.search_modules = set()
        self.extra_classes = set()

    def add_class(self, cls: typing.Type[_postprocessor.ImagePostprocessor]):
        """
        Add an image postprocessor implementation by its class, this is unused by dgenerate
        and is provided for the convenience of library users.

        :param cls: class the implements :py:class:`dgenerate.postprocessors.postprocessor.ImagePostprocessor`,
            (not an instance, use the type itself)
        """

        self.extra_classes.add(cls)

    def load_plugin_modules(self, paths: _types.Paths):
        """
        Add a list of python module directories or python files to :py:attr:`.Loader.search_modules`

        They will be newly imported or fetched if already previously imported.

        :param paths: python module directories, python file paths
        """
        self.search_modules.update(_plugin.load_modules(paths))

    def _load(self, path, device):
        call_by_name = path.split(';', 1)[0].strip()

        postprocessor_class = self.get_class_by_name(call_by_name)

        inherited_args = ['device']

        parser_accepted_args = postprocessor_class.get_accepted_args(call_by_name)

        if 'called-by-name' in parser_accepted_args:
            raise RuntimeError(f'called-by-name is a reserved ImagePostprocessor module argument, '
                               'chose another argument name for your module.')

        for inherited_arg in inherited_args:
            if inherited_arg in parser_accepted_args:
                raise RuntimeError(f'{inherited_arg} is a reserved ImagePostprocessor module argument, '
                                   'chose another argument name for your module.')

            parser_accepted_args.append(inherited_arg)

        arg_parser = _textprocessing.ConceptUriParser("Image Postprocessor", parser_accepted_args)

        try:
            parsed_args = arg_parser.parse_concept_uri(path).args
        except _textprocessing.ConceptPathParseError as e:
            raise _exceptions.ImagePostprocessorArgumentError(str(e))

        args_dict = {}

        for arg in postprocessor_class.get_default_args(call_by_name):
            args_dict[_textprocessing.dashdown(arg[0])] = arg[1]

        for k, v in parsed_args.items():
            args_dict[_textprocessing.dashdown(k)] = v

        args_dict['device'] = parsed_args.get('device', device)
        args_dict['called_by_name'] = call_by_name

        for arg in postprocessor_class.get_required_args(call_by_name):
            if _textprocessing.dashdown(arg) not in args_dict:
                raise _exceptions.ImagePostprocessorNotFoundError(
                    f'Missing required argument "{arg}" for image postprocessor "{call_by_name}".')

        try:
            return postprocessor_class(**args_dict)
        except _exceptions.ImagePostprocessorArgumentError as e:
            raise _exceptions.ImagePostprocessorArgumentError(
                f'Invalid argument given to image postprocessor "{call_by_name}": {e}')

    def get_available_classes(self) -> typing.List[typing.Type[_postprocessor.ImagePostprocessor]]:
        """
        Return a list of all :py:class:`dgenerate.postprocessors.ImagePostprocessor` implementations
        visible to this loader.

        :return: list of :py:class:`dgenerate.postprocessors.ImagePostprocessor`
        """

        found_classes = []
        for mod in itertools.chain([sys.modules['dgenerate.postprocessors']], self.search_modules):
            def _excluded(cls):
                if not inspect.isclass(cls):
                    return True

                if cls is _postprocessor.ImagePostprocessor:
                    return True

                if not issubclass(cls, _postprocessor.ImagePostprocessor):
                    return True

                if hasattr(cls, 'HIDDEN'):
                    return cls.HIDDEN
                else:
                    return False

            found_classes += [value for value in
                              itertools.chain(_types.get_public_members(mod).values(), self.extra_classes)
                              if not _excluded(value)]

        return found_classes

    def get_class_by_name(self, postprocessor_name) -> typing.Type[_postprocessor.ImagePostprocessor]:
        """
        Get a :py:class:`dgenerate.postprocessors.ImagePostprocessor` implementation from the loader
        using one of the implementations defined :py:attr:`dgenerate.postprocessors.ImagePostprocessor.NAMES`

        :raises RuntimeError: if more than one class was found using the provided name.
        :raises ImagePostprocessorNotFoundError: if the name could not be found.

        :param postprocessor_name: the name to search for
        :return: :py:class:`dgenerate.postprocessors.ImagePostprocessor`
        """

        classes = [cls for cls in self.get_available_classes() if
                   postprocessor_name in cls.get_names()]

        if len(classes) > 1:
            raise RuntimeError(
                f'Found more than one ImagePostprocessor with the name: {postprocessor_name}')

        if not classes:
            raise _exceptions.ImagePostprocessorNotFoundError(
                f'Found no image postprocessor with the name: {postprocessor_name}')

        return classes[0]

    def get_all_names(self) -> _types.Names:
        """
        Get all :py:attr:`dgenerate.postprocessors.ImagePostprocessor.NAMES` values visible to this loader.

        :return: list of names (strings)
        """

        names = []
        for cls in self.get_available_classes():
            names += cls.get_names()
        return names

    def get_help(self, postprocessor_name: _types.Name) -> str:
        """
        Get the formatted help string for a specific postprocessor by name.

        :raises RuntimeError: if more than one class was found using the provided name.
        :raises ImagePostprocessorNotFoundError: if the name could not be found.

        :param postprocessor_name: the postprocessor name to search for
        :return: formatted help string
        """

        return self.get_class_by_name(postprocessor_name).get_help(postprocessor_name)

    def load(self, uri: typing.Union[_types.Uri, typing.Iterable[_types.Uri]], device: str = 'cpu') -> \
            typing.Union[_postprocessor.ImagePostprocessor, _postprocessorchain.ImagePostprocessorChain, None]:
        """
        Load an image postprocessor or multiple image postprocessors. They are loaded by URI, which
        is their name and any module arguments, for example: ``upscaler;model=esrgan_x4.pth;tile=512``

        Specifying multiple postprocessors with a list will create an image postprocessor chain object.

        :raises RuntimeError: if more than one class was found using the provided name mentioned in the URI.
        :raises ImagePostprocessorNotFoundError: if the name mentioned in the URI could not be found.
        :raises ImagePostprocessorArgumentError: if the URI contained invalid arguments.


        :param uri: Postprocessor URI or list of URIs
        :param device: Request a specific rendering device, default is CPU
        :return: :py:class:`dgenerate.postprocessors.ImagePostprocessor` or
            :py:class:`dgenerate.postprocessors.ImagePostprocessorChain`
        """

        if uri is None:
            raise ValueError('uri must not be None')

        if isinstance(uri, str):
            return self._load(uri, device)

        paths = list(uri)

        if not paths:
            return None

        if len(paths) == 1:
            return self._load(paths[0], device)

        return _postprocessorchain.ImagePostprocessorChain(self._load(i, device) for i in paths)


__all__ = _types.module_all()
