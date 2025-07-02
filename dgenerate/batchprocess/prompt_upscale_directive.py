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
import collections.abc

import dgenerate.batchprocess.configrunnerplugin as _configrunnerplugin
import dgenerate.batchprocess.util as _util
import dgenerate.subcommands.subcommandloader as _subcommandloader


class PromptUpscaleDirective(_configrunnerplugin.ConfigRunnerPlugin):
    def __init__(self, **kwargs):
        """
        :param kwargs: plugin base class arguments
        """

        super().__init__(**kwargs)

        self.register_directive('prompt_upscale', self._directive)

        self._parser = _util.DirectiveArgumentParser(prog='\\prompt_upscale', add_help=False)

        self._parser.add_argument('--setp', metavar='VARIABLE_NAME', type=str, required=False)

    def _directive(self, args: collections.abc.Sequence[str]) -> int:
        """
        Alias for: --sub-command prompt-upscale, with the additional option
        to set a config template variable to a python list of upscaled prompt strings.

        This allows upscaling prompts without performing diffusion.

        You can use the additional option --setp to set a config template variable
        with the output of the subcommand, this will set the template variable to a python
        list of upscaled prompt strings, similar to the behavior of \\setp.

        This additional option disables all --sub-command prompt-upscale
        options except "prompts", --help, and --upscaler.

        See: \\prompt_upscale --help

        This does not cause the config to exit.
        """

        parsed_args, args = self._parser.parse_known_args(args)
        if self._parser.return_code is not None:
            return self._parser.return_code

        if parsed_args.setp:
            prompts = []
            subcommand = _subcommandloader.SubCommandLoader().load(
                'prompt-upscale',
                args=args,
                program_name='\\prompt_upscale',
                output_list=prompts,
                local_files_only=self.local_files_only
            )
            return_code = subcommand()
            self.set_template_variable(parsed_args.setp, prompts)
            return return_code
        else:
            subcommand = _subcommandloader.SubCommandLoader().load(
                'prompt-upscale',
                args=args,
                program_name='\\prompt_upscale',
                local_files_only=self.local_files_only
            )
            return subcommand()
