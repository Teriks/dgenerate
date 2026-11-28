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
import torch

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


def _timestep_limit(scheduler) -> float | None:
    """Largest timestep ``scheduler`` can place, or ``None`` when unknown.

    Noise schedulers index ``num_train_timesteps`` training steps from 0.
    FlowMatch divides by ``num_train_timesteps`` to get a sigma, so its
    top timestep is the count itself and sigma 1.
    """
    config = getattr(scheduler, 'config', None)
    count = getattr(config, 'num_train_timesteps', None)
    if count is None:
        return None
    if scheduler.__class__.__name__.startswith('FlowMatch'):
        return float(count)
    return float(count) - 1


def _check_timesteps(timesteps, scheduler, label: str) -> list:
    """Reject a timestep list the scheduler would sample incorrectly.

    The list has to be strictly descending: schedulers look the current
    timestep up in the list, and a repeated first value makes them skip a
    step. Values above the training range are clamped by most schedulers
    and turn into a sigma above 1 under FlowMatch.
    """
    values = list(timesteps)
    if not values:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{label} timesteps cannot be empty.')
    for left, right in zip(values, values[1:]):
        if right >= left:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'{label} timesteps must strictly descend, got {left:g} followed by {right:g}.')
    limit = _timestep_limit(scheduler)
    if limit is not None and float(values[0]) > limit:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{label} timesteps start at {values[0]:g}, but '
            f'{scheduler.__class__.__name__} was trained on timesteps up to {limit:g}.')
    return values


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
    config = getattr(scheduler, 'config', None)
    if (getattr(config, 'use_dynamic_shifting', False)
            and 'mu' in inspect.signature(scheduler.set_timesteps).parameters):
        # The pipeline computes mu from the frame size. This value leaves
        # the schedule unshifted.
        linear = getattr(config, 'time_shift_type', 'exponential') == 'linear'
        call_kwargs['mu'] = 1.0 if linear else 0.0
    try:
        with _preserved_scheduler_state(scheduler):
            scheduler.set_timesteps(steps, **call_kwargs)
            return _integer_timesteps(getattr(scheduler, 'timesteps', None))
    except Exception as e:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Could not read {family} scheduler timesteps for a --timesteps expression. '
            f'Pass a CSV list with --timesteps instead.') from e


def _pin_flow_timesteps(scheduler, timesteps) -> None:
    """Set FlowMatch sigmas to ``timestep / num_train_timesteps``.

    Given custom timesteps, FlowMatch derives sigmas from them and then
    shifts those sigmas, but keeps the timesteps unshifted. With any shift
    but 1 the model would be told one noise level while the latents sit at
    another. The custom timesteps are what the model sees, so the sigmas
    are made to agree with them. The terminal sigma is kept.
    """
    if scheduler.__class__.__name__ != 'FlowMatchEulerDiscreteScheduler':
        return
    config = scheduler.config
    sigmas = getattr(scheduler, 'sigmas', None)
    if getattr(config, 'invert_sigmas', False) or sigmas is None:
        return
    values = [float(value) for value in timesteps]
    if len(sigmas) != len(values) + 1:
        return
    count = float(config.num_train_timesteps)
    target = torch.tensor([value / count for value in values],
                          dtype=sigmas.dtype, device=sigmas.device)
    scheduler.sigmas = torch.cat([target, sigmas[len(values):]])


@contextlib.contextmanager
def _flow_timesteps_pinned(scheduler):
    """Apply :func:`_pin_flow_timesteps` whenever the pipeline passes timesteps."""
    if scheduler is None or scheduler.__class__.__name__ != 'FlowMatchEulerDiscreteScheduler':
        yield
        return
    shadowed = scheduler.__dict__.get('set_timesteps')

    def set_timesteps(self, *args, **kwargs):
        if shadowed is not None:
            result = shadowed(*args, **kwargs)
        else:
            result = type(self).set_timesteps(self, *args, **kwargs)
        timesteps = kwargs.get('timesteps')
        if timesteps is not None:
            _pin_flow_timesteps(self, timesteps)
        return result

    set_timesteps.__signature__ = inspect.signature(scheduler.set_timesteps)
    scheduler.set_timesteps = types.MethodType(set_timesteps, scheduler)
    try:
        yield
    finally:
        if shadowed is None:
            del scheduler.set_timesteps
        else:
            scheduler.set_timesteps = shadowed


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
    forced = list(timesteps)
    with _substituted_set_timesteps(
            scheduler, lambda: {'timesteps': list(forced)},
            after=lambda self: _pin_flow_timesteps(self, forced)):
        yield


@contextlib.contextmanager
def _substituted_set_timesteps(scheduler, schedule, after=None):
    """Make ``scheduler.set_timesteps`` use ``schedule()`` in place of the caller's schedule.

    The step count, ``timesteps``, and ``sigmas`` the caller passes are
    dropped. Other arguments such as ``mu`` are forwarded. The substitute
    keeps the signature of the method it replaces, plus ``timesteps`` and
    ``sigmas``: ``retrieve_timesteps`` and the denoise range inspect it.
    """
    shadowed = scheduler.__dict__.get('set_timesteps')
    signature = inspect.signature(scheduler.set_timesteps)
    parameters = signature.parameters
    open_ended = any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                     for parameter in parameters.values())

    def set_timesteps(self, num_inference_steps=None, device=None,
                      timesteps=None, sigmas=None, **kwargs):
        if not open_ended:
            kwargs = {name: value for name, value in kwargs.items() if name in parameters}
        if shadowed is not None:
            result = shadowed(num_inference_steps=None, device=device, **schedule(), **kwargs)
        else:
            result = type(self).set_timesteps(
                self, num_inference_steps=None, device=device, **schedule(), **kwargs)
        if after is not None:
            after(self)
        return result

    keyword = inspect.Parameter.KEYWORD_ONLY
    leading = [parameter for parameter in parameters.values()
               if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                     inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    named = [parameter.replace(kind=keyword) for parameter in parameters.values()
             if parameter.kind is keyword]
    named += [inspect.Parameter(name, keyword, default=None)
              for name in ('timesteps', 'sigmas') if name not in parameters]
    set_timesteps.__signature__ = signature.replace(parameters=[
        inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        *leading, *named,
        inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD)])

    scheduler.set_timesteps = types.MethodType(set_timesteps, scheduler)
    try:
        yield
    finally:
        if shadowed is None:
            del scheduler.set_timesteps
        else:
            scheduler.set_timesteps = shadowed


def _sigma_step_count(scheduler, sigmas) -> int:
    """Steps ``scheduler`` runs for ``set_timesteps(sigmas=sigmas)``.

    Euler takes a terminal sigma that is not a step, other schedulers
    append their own.
    """
    with _preserved_scheduler_state(scheduler):
        accepted = inspect.signature(scheduler.set_timesteps).parameters
        kwargs = {'device': 'cpu'} if 'device' in accepted else {}
        config = getattr(scheduler, 'config', None)
        if getattr(config, 'use_dynamic_shifting', False) and 'mu' in accepted:
            kwargs['mu'] = 1.0 if getattr(config, 'time_shift_type', 'exponential') == 'linear' else 0.0
        scheduler.set_timesteps(sigmas=sigmas, **kwargs)
        steps = getattr(scheduler, 'num_inference_steps', None)
        return int(steps) if steps else len(scheduler.timesteps)


@contextlib.contextmanager
def _forced_scheduler_sigmas(scheduler, sigmas):
    """Substitute a custom sigma list inside ``scheduler.set_timesteps``.

    For pipelines whose call has no ``sigmas`` argument. ``sigmas`` is the
    value ``set_timesteps(sigmas=)`` takes for this scheduler. A list the
    pipeline passes itself is replaced. ``mu`` is still forwarded.
    """
    with _substituted_set_timesteps(scheduler, lambda: {'sigmas': sigmas}):
        yield


def _forced_user_schedule(wrapper, pipe, family: str, steps: int, user_args, shift_sigmas=None):
    """``--timesteps`` or ``--sigmas`` for a pipeline that calls ``set_timesteps(steps)``.

    Returns the step count and a context manager that substitutes the
    schedule at the scheduler. The step count is also written to
    ``user_args.inference_steps``.

    ``shift_sigmas`` maps the user's sigmas to the list the scheduler is
    given, for a scheduler that does not apply the checkpoint's shift itself.
    """
    import dgenerate.pipelinewrapper.schedulers as _schedulers
    if user_args.timesteps is None and user_args.sigmas is None:
        return steps, contextlib.nullcontext()
    scheduler = pipe.scheduler
    if user_args.timesteps is not None:
        if not _schedulers.scheduler_accepts_custom_timesteps(scheduler):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'The loaded {family} scheduler, {type(scheduler).__name__}, '
                f'does not support custom timesteps.')
        if isinstance(user_args.timesteps, str):
            timesteps = _eval_video_timesteps_expression(
                user_args.timesteps, _default_video_timesteps(pipe, family, steps))
        else:
            timesteps = [int(value) for value in user_args.timesteps]
        _check_timesteps(timesteps, scheduler, family)
        user_args.inference_steps = len(timesteps)
        return len(timesteps), _forced_scheduler_timesteps(scheduler, timesteps)
    if user_args.sigmas is not None:
        if not _schedulers.scheduler_accepts_custom_sigmas(type(scheduler)):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'The loaded {family} scheduler, {type(scheduler).__name__}, '
                f'does not support custom sigmas.')
        sigmas = type(wrapper)._sigmas_eval(family, pipe, steps, user_args.sigmas)
        if shift_sigmas is not None:
            sigmas = shift_sigmas(sigmas)
        steps = _sigma_step_count(scheduler, sigmas)
        user_args.inference_steps = steps
        return steps, _forced_scheduler_sigmas(scheduler, sigmas)
    return steps, contextlib.nullcontext()


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
