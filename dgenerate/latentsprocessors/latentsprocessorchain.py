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
import typing

import torch

import dgenerate.latentsprocessors.latentsprocessor as _latentsprocessor
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.types as _types


class LatentsProcessorChain(_latentsprocessor.LatentsProcessor):
    """
    Implements chainable latents processors.

    Chains processing steps together in a sequence.
    """

    HIDDEN = True

    def __init__(self,
                 latents_processors: typing.Optional[collections.abc.Iterable[_latentsprocessor.LatentsProcessor]] = None):
        """
        :param latents_processors: optional initial latents processors to fill the chain, accepts an iterable
        """
        if latents_processors is None:
            self._latents_processors = []
        else:
            self._latents_processors = list(latents_processors)
        
        # Use the model_type from the first processor if available, otherwise default to SD
        model_type = _enums.ModelType.SD
        if self._latents_processors:
            model_type = self._latents_processors[0].model_type
        
        super().__init__(loaded_by_name='chain', model_type=model_type)

    def _latentsprocessor_names(self):
        for latentsprocessor in self._latents_processors:
            yield str(latentsprocessor)

    def __str__(self):
        if not self._latents_processors:
            return f'{self.__class__.__name__}([])'
        else:
            return f'{self.__class__.__name__}([{", ".join(self._latentsprocessor_names())}])'

    def __repr__(self):
        return str(self)

    def add_processor(self, latents_processor: _latentsprocessor.LatentsProcessor):
        """
        Add a latents processor implementation to the chain.

        :param latents_processor: :py:class:`dgenerate.latentsprocessors.latentsprocessor.LatentsProcessor`
        """
        self._latents_processors.append(latents_processor)

    def impl_process(self,
                     pipeline,
                     latents: torch.Tensor) -> torch.Tensor:
        """
        Process latents through all latents processors in this chain in turn.

        Every subsequent invocation receives the last processed latents tensor as its argument.

        :param pipeline: The pipeline object
        :param latents: Input latents tensor with shape [B, C, H, W]
        :return: the processed latents tensor, possibly affected by every latents processor in the chain
        """

        if self._latents_processors:
            p_latents = latents
            for latentsprocessor in self._latents_processors:
                p_latents = latentsprocessor.process(pipeline, p_latents)
            return p_latents
        else:
            return latents

    def to(self, device: torch.device | str) -> "LatentsProcessorChain":
        """
        Move all :py:class:`torch.nn.Module` modules registered
        to this latents processor to a specific device.

        :raise dgenerate.OutOfMemoryError: if there is not enough memory on the specified device

        :param device: The device string, or torch device object
        :return: the latents processor itself
        """
        # Move all processors in the chain to the device
        for p in self._latents_processors:
            p.to(device)

        return self


__all__ = _types.module_all()
