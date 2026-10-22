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

"""Timestep and sigma schedule helpers shared by the video model families."""

from __future__ import annotations

import collections.abc
import contextlib
import inspect
import types

import numpy

import dgenerate.eval as _eval
import dgenerate.pipelinewrapper.pipelines as _pipelines


@contextlib.contextmanager
def _preserved_scheduler_state(scheduler):
    """Restore ``timesteps``, ``sigmas``, and ``num_inference_steps`` on exit.

    Used when ``set_timesteps`` is called only to read a default schedule, so
    the pipeline call that follows starts from the scheduler's prior state.
    """
    old_state = {
        name: getattr(scheduler, name)
        for name in ('timesteps', 'sigmas', 'num_inference_steps')
        if hasattr(scheduler, name)
    }
    try:
        yield scheduler
    finally:
        for name, value in old_state.items():
            setattr(scheduler, name, value)


def _integer_timesteps(scheduled) -> list[int]:
    """Round a scheduler timestep tensor or array to a list of ints."""
    if scheduled is None:
        raise ValueError('scheduler did not produce a timestep schedule')
    if hasattr(scheduled, 'detach'):
        scheduled = scheduled.detach().cpu().numpy()
    return numpy.rint(numpy.asarray(scheduled, dtype=float)).astype(numpy.int64).tolist()


def _default_video_timesteps(pipe, family, steps) -> list[int]:
    """Read the scheduler's default timestep array for a ``--timesteps`` expression.

    Scheduler state is restored afterward.
    """
    scheduler = pipe.scheduler
    call_kwargs = {}
    if 'device' in inspect.signature(scheduler.set_timesteps).parameters:
        call_kwargs['device'] = 'cpu'
    try:
        with _preserved_scheduler_state(scheduler):
            scheduler.set_timesteps(steps, **call_kwargs)
            return _integer_timesteps(getattr(scheduler, 'timesteps', None))
    except Exception as e:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Could not read {family} scheduler timesteps for a --timesteps expression. '
            f'Pass a CSV list with --timesteps instead.') from e


@contextlib.contextmanager
def _forced_scheduler_timesteps(scheduler, timesteps: list[int]):
    """Substitute a custom schedule inside ``scheduler.set_timesteps``.

    Motif and LTX-2 pipelines always pass their own ``sigmas`` to
    ``retrieve_timesteps``, which rejects a caller ``timesteps`` argument
    alongside them, so the custom schedule is applied at the scheduler
    instead. ``mu`` is still forwarded.

    The substitute is bound with :class:`types.MethodType` so that a
    ``copy.deepcopy`` of the scheduler (LTX-2 copies it for the audio latents)
    rebinds it to the copy instead of updating the original scheduler.
    """
    shadowed = scheduler.__dict__.get('set_timesteps')
    accepted = inspect.signature(scheduler.set_timesteps).parameters

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, **kwargs):
        kwargs = {name: value for name, value in kwargs.items()
                  if name in accepted and name != 'timesteps'}
        if shadowed is not None:
            return shadowed(device=device, timesteps=list(timesteps), **kwargs)
        return type(self).set_timesteps(self, device=device, timesteps=list(timesteps), **kwargs)

    scheduler.set_timesteps = types.MethodType(set_timesteps, scheduler)
    try:
        yield
    finally:
        if shadowed is None:
            del scheduler.set_timesteps
        else:
            scheduler.set_timesteps = shadowed


def _eval_video_timesteps_expression(expression: str, base_timesteps) -> list[int]:
    expression = str(expression).strip().removeprefix('expr:').strip()
    if not expression:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Video timesteps expression cannot be empty.')
    interpreter = _eval.standard_interpreter(symtable=_eval.safe_builtins())
    interpreter.symtable['np'] = numpy
    interpreter.symtable['timesteps'] = numpy.asarray(base_timesteps, dtype=float)
    try:
        value = interpreter.eval(expression, show_errors=False, raise_errors=True)
    except Exception as e:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Error interpreting video timesteps expression "{expression}":\n{e}') from e
    if not isinstance(value, collections.abc.Iterable) or isinstance(value, (str, bytes)):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Video timesteps expression did not evaluate to an array, got: {value}')
    result = []
    for item in value:
        try:
            numeric = float(item)
        except (TypeError, ValueError) as e:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Video timesteps expression must produce integer values.') from e
        if not numpy.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Video timesteps expression must produce non-negative integer values.')
        result.append(int(numeric))
    if not result:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Video timesteps expression produced an empty schedule.')
    return result


def _eval_sigma_expression(expression: str, base_sigmas):
    interpreter = _eval.standard_interpreter(
        symtable=_eval.safe_builtins()
    )
    interpreter.symtable['np'] = numpy
    interpreter.symtable['sigmas'] = numpy.array(base_sigmas, dtype=float)
    try:
        value = interpreter.eval(expression, show_errors=False, raise_errors=True)
    except Exception as e:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Error interpreting sigmas expression "{expression}":\n{e}'
        ) from e
    if not isinstance(value, collections.abc.Iterable) or isinstance(value, (str, bytes)):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Sigmas expression did not evaluate to an array, got: {value}'
        )
    return [float(item) for item in value]
