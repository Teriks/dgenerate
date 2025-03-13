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

import dgenerate.pipelinewrapper.schedulers as _schedulers
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import diffusers
import typing


def get_scheduler_help(pipeline_cls, help_args: bool = False, indent: int = 0):
    """
    Generate a help string containing info about a pipline classes compatible schedulers.

    :param pipeline_cls: The pipeline class
    :param help_args: Show individual scheduler arguments that can be specified via URI?
    :param indent: Indent all text output by this amount of spaces.
    :return: help string
    """
    compatibles = _schedulers.get_compatible_schedulers(pipeline_cls)

    if not help_args:
        return '\n'.join((indent * " ") + _textprocessing.quote(i.__name__) for i in compatibles)
    else:
        schemas = _schedulers.get_scheduler_uri_schema(compatibles)

        help_string = ''
        for s_idx, (scheduler_name, arg_schema) in enumerate(schemas.items()):
            help_string += scheduler_name + ":\n"
            is_end = s_idx == len(schemas) - 1
            for d_idx, (arg, details) in enumerate(arg_schema.items()):
                is_args_end = d_idx == len(arg_schema) - 1
                type_anno = details.get('types')

                if type_anno:
                    if details.get('optional'):
                        type_anno.append('None')

                    if len(type_anno) > 1:
                        type_anno = ' | '.join(type_anno)
                    else:
                        type_anno = type_anno[0]

                    type_anno = f": {type_anno}"
                else:
                    type_anno = ''

                help_string += (indent * " ") + (" " * 4) + arg + type_anno + " = " + str(details['default']) + (
                    '' if is_end and is_args_end else '\n')

        return help_string


def scheduler_is_help(name: str | None):
    """
    This scheduler URI is simply a request for help?, IE: ``"help"`` or ``"helpargs"``?

    :param name: string to test
    :return: ``True`` or ``False``
    """
    if name is None:
        return False
    lname = name.strip().lower()

    return lname == 'help' or lname == 'helpargs'


def scheduler_is_help_args(name: str | None):
    """
    This scheduler URI is explicitly requesting argument help, IE: ``"helpargs"``

    :param name: string to test
    :return: ``True`` or ``False``
    """
    if name is None:
        return False
    lname = name.strip().lower()

    return lname == 'helpargs'


def text_encoder_help(pipeline_class: type[diffusers.DiffusionPipeline], indent: int = 0) -> str:
    """
    Describe compatible text encoders for a pipeline in terms of ``--text-encoders`` argument position and type.

    :param pipeline_class: Diffusers pipeline class
    :param indent: Text indent level
    :return: help string
    """
    return (' ' * indent + (('\n' + ' ' * 4).join(
        str(idx) + ' = ' + n for idx, n in
        enumerate(v[1].__name__ for v in
                  typing.get_type_hints(pipeline_class.__init__).items()
                  if v[0].startswith('text_encoder')))))


def text_encoder_is_help(text_encoder_uris: _types.OptionalUris):
    """
    Text encoder uris specification is simply a request for help?, IE: ``"help"``?

    :param text_encoder_uris: list of text encoder URIs to test
    :return: ``True`` or ``False``
    """
    if text_encoder_uris is None:
        return False
    return any(t == 'help' for t in text_encoder_uris)
