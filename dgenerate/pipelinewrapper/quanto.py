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
import optimum.quanto
import torch


def quantize_freeze(
        model: torch.nn.Module,
        weights: str | optimum.quanto.qtype | None = None,
        activations: str | optimum.quanto.qtype | None = None,
        optimizer: optimum.quanto.Optimizer | None = None,
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None):
    """
    Quantize and freeze a ``nn.Module`` with ``optimum.quanto.quantize``.

    Add an internal tag so it can be determined that this has happened.

    Args:
        model (`torch.nn.Module`): the model whose submodules will be quantized.
        weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
        activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
        include (`Optional[Union[str, List[str]]]`):
            Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
        exclude (`Optional[Union[str, List[str]]]`):
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.

    :param model: The model
    :param weights: the ``qtype`` for weights quantization.
    :param activations: the ``qtype`` for activations quantization.
    :param optimizer:
    :param include: Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
    :param exclude: optimizer
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.
    :return: The model
    """
    optimum.quanto.quantize(model,
                            weights=weights,
                            activations=activations,
                            optimizer=optimizer,
                            include=include,
                            exclude=exclude)
    optimum.quanto.freeze(model)
    model._DGENERATE_QUANTO_FREEZE = True
    return model


def is_quantized_and_frozen(model):
    """
    Return true if a model object has had :py:class:`dgenerate.pipelinewrapper.quanto.quantize_freeze`
    called on it.

    :param model: The model to test
    :return: ``True`` or ``False``
    """
    return hasattr(model, '_DGENERATE_QUANTO_FREEZE')
