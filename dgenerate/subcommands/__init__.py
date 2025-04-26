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

import dgenerate.plugin as _plugin
import dgenerate.types as _types
from .exceptions import SubCommandError, SubCommandArgumentError, SubCommandNotFoundError
from .image_process import ImageProcessSubCommand
from .civitai_links import CivitAILinksSubCommand
from .subcommand import SubCommand
from .subcommandloader import SubCommandLoader
from .to_diffusers import ToDiffusersSubCommand
from .prompt_upscale import PromptUpscaleSubCommand
from .auto1111_metadata import Auto1111MetadataSubCommand

__doc__ = """
Sub-Commands implemented by the dgenerate command line tool.
"""


def sub_command_help(names: _types.Names,
                     plugin_module_paths: _types.OptionalPaths = None,
                     throw=False,
                     log_error=True):
    """
    Implements ``--sub-command-help`` command line option


    :param names: arguments (sub-command names, or empty list)
    :param plugin_module_paths: extra plugin module paths to search
    :param throw: throw on error? or simply print to stderr and return a return code.
    :param log_error: log errors to stderr?

    :raises SubCommandNotFoundError:
    :raises dgenerate.ModuleFileNotFoundError:

    :return: return-code, anything other than 0 is failure
    """
    try:
        return SubCommandLoader().loader_help(
            names=names,
            plugin_module_paths=plugin_module_paths,
            title='sub-command',
            title_plural='sub-commands',
            throw=True,
            log_error=log_error)
    except (SubCommandNotFoundError, _plugin.ModuleFileNotFoundError) as e:
        if throw:
            raise e
        return 1


__all__ = _types.module_all()
