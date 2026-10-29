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

import dgenerate.subcommands.subcommand as _subcommand


class AssistantSubCommand(_subcommand.SubCommand):
    """
    Write a dgenerate config from a plain language request.

    A local Qwen model running through xllamacpp writes the config, guided by
    the closest example configs and documentation, retrieved from an index of
    the dgenerate examples and docs that ships with dgenerate. The config is
    then checked with dgenerate's own config runner and argument parser without
    loading any diffusion models, and the model is asked to fix anything
    dgenerate rejects.

    Requires the xllamacpp extra (pip install dgenerate[xllamacpp]).

    The chat and embedding models are downloaded from Hugging Face on first use.
    ``--model`` and ``--embed-model`` must be one of the supported Qwen GGUFs.
    Each embedding model has an index packaged with dgenerate.

    File paths in the request are treated as input files, relative paths are
    relative to the current directory and are rewritten relative to --output.

    Examples:

    dgenerate --sub-command assistant "use ltx 2 with the canny ic lora to turn clip.gif into a dancing fox"

    dgenerate --sub-command assistant -o fox.dgen "use ltx 2 with the canny ic lora to turn clip.gif into a dancing fox"

    See: dgenerate --sub-command assistant --help
    """

    NAMES = ['assistant']

    def __init__(self, program_name='assistant', **kwargs):
        super().__init__(**kwargs)
        self._program_name = program_name

    def __call__(self) -> int:
        import dgenerate.assistant.cli as _cli
        return _cli.main(self.args, prog=self._program_name, local_files_only=self.local_files_only)
