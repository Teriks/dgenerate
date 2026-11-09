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


"""Resolve omitted guidance and step counts from diffusers pipeline defaults.

Used by :mod:`dgenerate.pipelinewrapper.videopipelines` and
:func:`~dgenerate.pipelinewrapper.wrapper._resolve_omitted_sampling`.
"""

from __future__ import annotations

import inspect
import typing

import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.types as _types

if typing.TYPE_CHECKING:
    import dgenerate.pipelinewrapper.arguments as _arguments


def pipeline_call_default(pipeline_or_class, name: str):
    """Default of a pipeline ``__call__`` parameter, or ``None`` when it has none."""
    if pipeline_or_class is None:
        return None
    try:
        signature = inspect.signature(pipeline_or_class.__call__)
    except (TypeError, ValueError):
        return None
    parameter = signature.parameters.get(name)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    if isinstance(parameter.default, (bool, str)):
        return None
    return parameter.default


def resolved_guidance_scale(
        pipeline,
        user_args: _arguments.GenerationArguments,
        *,
        fallback: float | None = None) -> float:
    """Return guidance, filling ``user_args`` from the pipeline when omitted."""
    if user_args.guidance_scale is not None:
        return float(user_args.guidance_scale)
    default = pipeline_call_default(pipeline, 'guidance_scale')
    if default is None:
        default = (
            fallback if fallback is not None
            else _constants.DEFAULT_FALLBACK_GUIDANCE_SCALE)
    value = float(default)
    user_args.guidance_scale = value
    return value


def resolved_inference_steps(
        pipeline,
        user_args: _arguments.GenerationArguments,
        *,
        fallback: int | None = None) -> int:
    """Return step count, filling ``user_args`` from the pipeline when omitted."""
    if user_args.inference_steps is not None:
        return int(user_args.inference_steps)
    default = pipeline_call_default(pipeline, 'num_inference_steps')
    if default is None:
        default = (
            fallback if fallback is not None
            else _constants.DEFAULT_FALLBACK_INFERENCE_STEPS)
    value = int(default)
    user_args.inference_steps = value
    return value


def default_guidance_scale_value(
        pipeline,
        guidance_scale: float | None,
        *,
        fallback: float | None = None) -> float:
    """Like :func:`resolved_guidance_scale` without a ``user_args`` object."""
    if guidance_scale is not None:
        return float(guidance_scale)
    default = pipeline_call_default(pipeline, 'guidance_scale')
    return float(_types.default(
        default,
        fallback if fallback is not None else _constants.DEFAULT_FALLBACK_GUIDANCE_SCALE))


def default_inference_steps_value(
        pipeline,
        inference_steps: int | None,
        *,
        fallback: int | None = None) -> int:
    """Like :func:`resolved_inference_steps` without a ``user_args`` object."""
    if inference_steps is not None:
        return int(inference_steps)
    default = pipeline_call_default(pipeline, 'num_inference_steps')
    return int(_types.default(
        default,
        fallback if fallback is not None else _constants.DEFAULT_FALLBACK_INFERENCE_STEPS))
