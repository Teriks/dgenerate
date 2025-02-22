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

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.types as _types
import torch

__doc__ = """
High level utilitys for interacting with objects cached by dgenerate in GPU side memory.

These objects may be cached in small quantity by various dgenerate sub-modules.
"""


def clear_device_cache(device: torch.device | str):
    """
    Clear every object cached by dgenerate in its GPU side cache for a specific device.

    :param device: The device the objects are cached on.
    """

    device = torch.device(device)

    active_pipe = _pipelines.get_last_called_pipeline()

    if active_pipe is None:
        return

    current_device_index = torch.cuda.current_device()

    pipe_device_index = _types.default(
        _pipelines.get_torch_device(active_pipe).index, current_device_index)

    device_index = _types.default(
        device.index, current_device_index)

    if pipe_device_index == device_index:
        # get rid of this reference immediately
        # noinspection PyUnusedLocal
        active_pipe = None

        _messages.debug_log(
            f'{_types.fullname(clear_device_cache)} is attempting to evacuate any previously '
            f'called diffusion pipeline in the VRAM of device: {device}.')

        # potentially free up VRAM on the GPU we are
        # about to move to
        _pipelines.destroy_last_called_pipeline()


__all__ = _types.module_all()
