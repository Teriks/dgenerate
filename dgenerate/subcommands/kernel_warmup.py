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

import dgenerate.batchprocess.util as _b_util
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.quantkernels as _quantkernels
import dgenerate.subcommands.subcommand as _subcommand


class KernelWarmupSubCommand(_subcommand.SubCommand):
    """
    Download the quant kernels for this computer.

    GGUF and ComfyUI quantized checkpoints fetch a platform-specific kernel
    from the Hugging Face Hub the first time they run. This command fetches
    the kernels for the current accelerator, and no others, so a later run
    can stay offline.

    CUDA and Intel XPU ComfyUI matmuls use kernels that ship with PyTorch, so
    there is nothing to download for those. Apple Metal downloads the MLX
    quant kernel. AMD ROCm downloads the AITER kernel. CUDA downloads the
    GGUF kernel when a build exists for this operating system. A machine with
    no published build skips that kernel.

    The network installer runs this after a new install. An older dgenerate
    that does not provide the subcommand is left unchanged.

    If the Hub cannot be reached:

    Cannot download platform specific quant kernels, you are not online.

    The command then exits with status 1.

    Examples:

    dgenerate --sub-command kernel-warmup

    See: dgenerate --sub-command kernel-warmup --help
    """

    NAMES = ['kernel-warmup']

    def __init__(self, program_name='kernel-warmup', **kwargs):
        super().__init__(**kwargs)
        self._parser = _b_util.DirectiveArgumentParser(
            prog=program_name,
            description=(
                'Download the GGUF and ComfyUI quant kernels for this '
                'computer so a later run can stay offline.'))

    def __call__(self) -> int:
        self._parser.parse_args(self.args)
        if self._parser.return_code is not None:
            return self._parser.return_code

        specs = _quantkernels.current_quant_kernel_specs()
        if self.local_files_only and specs:
            _messages.error(_quantkernels.OFFLINE_MESSAGE)
            return 1
        return _quantkernels.download_quant_kernels(specs)
