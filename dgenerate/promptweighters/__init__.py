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


import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.types as _types
from .compelpromptweighter import CompelPromptWeighter
from .exceptions import \
    PromptWeighterArgumentError, \
    PromptWeighterNotFoundError, \
    PromptWeightingUnsupported
from .promptweighter import PromptWeighter
from .promptweighterloader import PromptWeighterLoader


class PromptWeighterHelpUsageError(Exception):
    """
    Raised on argument parse errors in :py:func:`.prompt_weighters_help`
    """
    pass


def prompt_weighter_help(names: _types.Names,
                         throw=False,
                         log_error=True):
    """
    Implements ``--prompt-weighter-help`` command line option

    :param names: arguments (prompt weighter names, or empty list)
    :param throw: throw on error? or simply print to stderr and return a return code.
    :param log_error: log errors to stderr?

    :raises PromptWeighterHelpUsageError:

    :return: return-code, anything other than 0 is failure
    """

    try:
        return PromptWeighterLoader().loader_help(
            names=names,
            title='prompt weighter',
            title_plural='prompt weighters',
            throw=True,
            log_error=log_error)
    except PromptWeighterNotFoundError as e:
        if throw:
            raise PromptWeighterHelpUsageError(str(e).strip())
        return 1


def prompt_weighter_names():
    """
    Implementation names.
    :return: a list of prompt weighter implementation names.
    """

    return list(PromptWeighterLoader().get_all_names())


def prompt_weighter_name_from_uri(uri):
    return uri.split(';')[0].strip()


def is_valid_prompt_weighter_uri(uri):
    return prompt_weighter_name_from_uri(uri) in prompt_weighter_names()


def create_prompt_weighter(uri, model_type: _enums.ModelType, pipeline_type: _enums.PipelineType) -> PromptWeighter:
    return PromptWeighterLoader().load(uri, model_type=model_type, pipeline_type=pipeline_type)


__all__ = _types.module_all()