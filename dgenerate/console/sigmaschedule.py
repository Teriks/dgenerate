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

"""Shape a sigma list without leaving the schedule a model starts from.

The numbers here are the schedule ``--sigmas`` receives, before a pipeline
applies its own shift. Every edit, including a pure scale, is written as the
numbers themselves, so the run uses the curve the editor showed.
"""


import math
import re


class SigmaScheduleError(ValueError):
    pass


def parse_base(values) -> list[float]:
    """A strictly descending schedule of at least two positive numbers."""
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise SigmaScheduleError('A sigma schedule needs a list of numbers.')
    if len(values) < 2:
        raise SigmaScheduleError('A sigma schedule needs at least two values.')
    try:
        schedule = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise SigmaScheduleError('A sigma schedule must be numbers.') from error
    if any(value <= 0 for value in schedule):
        raise SigmaScheduleError('Sigma values must be greater than 0.')
    if any(left <= right for left, right in zip(schedule, schedule[1:])):
        raise SigmaScheduleError('Sigma values must descend.')
    return schedule


def resample(schedule: list[float], steps: int) -> list[float]:
    """Sample ``schedule`` at ``steps`` points, keeping its shape."""
    if steps < 2:
        raise SigmaScheduleError('A sigma schedule needs at least two steps.')
    if steps == len(schedule):
        return list(schedule)
    last = len(schedule) - 1
    sampled = []
    for index in range(steps):
        position = index * last / (steps - 1)
        lower = int(position)
        upper = min(lower + 1, last)
        weight = position - lower
        sampled.append(schedule[lower] * (1 - weight) + schedule[upper] * weight)
    return sampled


def is_noise_range(values) -> bool:
    """Values start above 1.5 and stay positive, as noise and EDM schedules do.

    Flow schedules stay in (0, 1].
    """
    values = list(values)
    return bool(values) and max(values) > 1.5 and min(values) > 0


def resample_sigmas(schedule: list[float], steps: int) -> list[float]:
    """``resample``, interpolating a noise schedule in log space.

    A noise schedule falls by orders of magnitude. Linear interpolation
    between its points lifts the low tail where most steps are spent.
    """
    if steps == len(schedule) or not is_noise_range(schedule):
        return resample(schedule, steps)
    return [math.exp(value) for value in resample([math.log(value) for value in schedule], steps)]


def split_values(text: str) -> list[str]:
    """Comma or whitespace separated parts of a typed list."""
    return [part for part in re.split(r'[\s,]+', text.strip()) if part]


def parse_sigmas(text: str) -> list[float]:
    """A typed sigma list that stays distinct once the pipeline casts it to float32."""
    schedule = parse_base(split_values(text))
    if any(left - right < _float_gap(left) for left, right in zip(schedule, schedule[1:])):
        raise SigmaScheduleError('Two sigma values are too close together to stay distinct.')
    return schedule


def _float_gap(higher: float) -> float:
    """One float32 step at this magnitude, so two sigmas cannot collapse."""
    return max(1e-6, abs(higher) * 1.2e-6)


def clamp_point(schedule: list[float], index: int, proposed: float) -> float:
    """Keep one point strictly between its neighbors."""
    if index > 0:
        proposed = min(proposed, schedule[index - 1] - _float_gap(schedule[index - 1]))
    if index + 1 < len(schedule):
        proposed = max(proposed, schedule[index + 1] + _float_gap(schedule[index + 1]))
    else:
        proposed = max(proposed, 1e-6)
    return proposed


def move_point(schedule: list[float], index: int, proposed: float,
               ceiling: float | None = None) -> list[float]:
    """Place one point, pushing neighbors so the list stays strictly descending.

    A point may travel past where its neighbors were. Those neighbors move
    with it instead of stopping the drag. With ``ceiling``, the point stops
    low enough that the neighbors it pushes stay at or below the ceiling.
    """
    values = [float(value) for value in schedule]
    if ceiling is not None:
        cap = float(ceiling)
        for _ in range(index):
            cap -= _float_gap(cap)
        proposed = min(float(proposed), cap)
    values[index] = float(proposed)
    for cursor in range(index, 0, -1):
        needed = values[cursor] + _float_gap(values[cursor])
        if values[cursor - 1] < needed:
            values[cursor - 1] = needed
    for cursor in range(index, len(values) - 1):
        ceiling = values[cursor] - _float_gap(values[cursor])
        if values[cursor + 1] > ceiling:
            values[cursor + 1] = ceiling
    if values[-1] < 1e-6:
        values[-1] = 1e-6
    for cursor in range(len(values) - 2, -1, -1):
        needed = values[cursor + 1] + _float_gap(values[cursor + 1])
        if values[cursor] < needed:
            values[cursor] = needed
    return values


def move_timestep(schedule: list[int], index: int, proposed: float,
                  ceiling: int | None = None) -> list[int]:
    """Place one timestep, pushing neighbors so the list stays strictly descending.

    With ``ceiling``, the point stops low enough that the neighbors it
    pushes stay at or below the ceiling.
    """
    values = [int(value) for value in schedule]
    if ceiling is not None:
        proposed = min(proposed, int(ceiling) - index)
    values[index] = int(round(proposed))
    for cursor in range(index, 0, -1):
        if values[cursor - 1] <= values[cursor]:
            values[cursor - 1] = values[cursor] + 1
    for cursor in range(index, len(values) - 1):
        if values[cursor + 1] >= values[cursor]:
            values[cursor + 1] = values[cursor] - 1
    if values[-1] < 0:
        values[-1] = 0
    for cursor in range(len(values) - 2, -1, -1):
        if values[cursor] <= values[cursor + 1]:
            values[cursor] = values[cursor + 1] + 1
    return values


def format_number(value: float) -> str:
    return f'{value:.6f}'.rstrip('0').rstrip('.') or '0'


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def _schema():
    global _SCHEMA
    if _SCHEMA is None:
        import importlib.resources
        import json
        with importlib.resources.open_text('dgenerate.console.schemas', 'karrasschedulers.json') as file:
            _SCHEMA = json.load(file)
    return _SCHEMA


def _spec_default(spec: dict, key: str, fallback=None):
    item = spec.get(key)
    if isinstance(item, dict) and 'default' in item:
        return item['default']
    return fallback


def _betas(count: int, start: float, end: float, schedule: str) -> list[float]:
    if schedule == 'linear':
        return _linspace(start, end, count)
    if schedule == 'scaled_linear':
        return [value * value for value in _linspace(start ** 0.5, end ** 0.5, count)]
    if schedule == 'squaredcos_cap_v2':
        betas = []
        for index in range(count):
            t1 = index / count
            t2 = (index + 1) / count

            def alpha_bar(t: float) -> float:
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return betas
    raise SigmaScheduleError(f'Beta schedule {schedule} is not built into the sigma editor.')


def _timestep_indices(spec: dict, steps: int, plus_one: bool) -> list[float]:
    count = int(_spec_default(spec, 'num-train-timesteps', 1000))
    spacing = str(_spec_default(spec, 'timestep-spacing', 'linspace'))
    offset = int(_spec_default(spec, 'steps-offset', 0) or 0)
    if spacing == 'linspace':
        if plus_one:
            grid = [round(value) for value in _linspace(0, count - 1, steps + 1)]
            return list(reversed(grid))[:-1]
        return list(reversed(_linspace(0, count - 1, steps)))
    if spacing == 'leading':
        denom = steps + 1 if plus_one else steps
        ratio = count // denom
        width = steps + 1 if plus_one else steps
        indices = list(reversed([round(i * ratio) for i in range(width)]))
        if plus_one:
            indices = indices[:-1]
        return [value + offset for value in indices]
    if spacing == 'trailing':
        ratio = count / steps
        raw = []
        value = float(count)
        while value > 0 and len(raw) < steps + 2:
            raw.append(value)
            value -= ratio
        indices = [round(item) - 1 for item in raw]
        if plus_one:
            return indices[:steps]
        return indices
    raise SigmaScheduleError(f'Timestep spacing {spacing} is not built into the sigma editor.')


def _train_sigmas(spec: dict) -> list[float]:
    count = int(_spec_default(spec, 'num-train-timesteps', 1000))
    betas = _betas(
        count,
        float(_spec_default(spec, 'beta-start', 0.0001)),
        float(_spec_default(spec, 'beta-end', 0.02)),
        str(_spec_default(spec, 'beta-schedule', 'linear')))
    product = 1.0
    train = []
    for beta in betas:
        product *= 1.0 - beta
        train.append(math.sqrt((1.0 - product) / product))
    return train


def _noise_sigmas(spec: dict, steps: int, plus_one: bool) -> list[float]:
    train = _train_sigmas(spec)
    indices = _timestep_indices(spec, steps, plus_one)
    last = len(train) - 1
    sampled = []
    for index in indices:
        if index <= 0:
            sampled.append(train[0])
        elif index >= last:
            sampled.append(train[last])
        else:
            lower = int(index)
            weight = index - lower
            sampled.append(train[lower] * (1 - weight) + train[lower + 1] * weight)
    return sampled


def _shift(sigmas: list[float], shift: float) -> list[float]:
    if shift == 1:
        return list(sigmas)
    return [shift * value / (1 + (shift - 1) * value) for value in sigmas]


def _flow_match(spec: dict, steps: int, shifted: bool = True) -> list[float]:
    """FlowMatch's default schedule, or with ``shifted=False`` the grid before its shift.

    A sigma list passed to FlowMatch is shifted again, so the grid is the
    list that reproduces the default. Mirrors ``_expression_base_sigmas``.
    """
    count = int(_spec_default(spec, 'num-train-timesteps', 1000))
    shift = float(_spec_default(spec, 'shift', 1.0) or 1.0)
    sigma_max = _shift([1.0], shift)[0]
    sigma_min = _shift([1.0 / count], shift)[0]
    grid = [value / count for value in _linspace(sigma_max * count, sigma_min * count, steps)]
    return _shift(grid, shift) if shifted else grid


def _flow_linspace(spec: dict, steps: int, unipc: bool) -> list[float]:
    count = int(_spec_default(spec, 'num-train-timesteps', 1000))
    shift = float(_spec_default(spec, 'flow-shift', 1.0) or 1.0)
    if unipc:
        raw = _linspace(1.0, 1.0 / count, steps + 1)[:-1]
        sigmas = _shift(raw, shift)
        if abs(sigmas[0] - 1.0) < 1e-6:
            sigmas[0] -= 1e-6
        return sigmas
    alphas = _linspace(1.0, 1.0 / count, steps + 1)
    raw = list(reversed(_shift([1.0 - value for value in alphas], shift)))[:-1]
    return raw


def _minimax(spec: dict, steps: int) -> list[float]:
    # --sigmas is the grid before the shift. The video and audio schedulers
    # each apply their own shift. The terminal 0 is not a point to edit.
    return _linspace(1.0, 0.0, steps + 1)


def _piflow(spec: dict, steps: int) -> list[float]:
    final_scale = float(_spec_default(spec, 'final-step-size-scale', 0.5))
    epsilon = float(_spec_default(spec, 'eps', 1e-6))
    shift = float(_spec_default(spec, 'shift', 5.0) or 5.0)
    segment = 1.0 / (steps - (1.0 - final_scale))
    value = 1.0
    raw = []
    for index in range(steps):
        raw.append(value)
        size = segment if index < steps - 1 else segment * final_scale
        value = max(value - size, epsilon)
    return _shift(raw, shift)


def _edm(spec: dict, steps: int) -> list[float]:
    sigma_min = float(_spec_default(spec, 'sigma-min', 0.002))
    sigma_max = float(_spec_default(spec, 'sigma-max', 80.0))
    rho = 7.0
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    return [(max_inv + t * (min_inv - max_inv)) ** rho for t in _linspace(0.0, 1.0, steps)]


def _clean(values: list[float]) -> list[float]:
    schedule = []
    for value in values:
        number = round(float(value), 6)
        if number <= 0:
            break
        if schedule and number > schedule[-1]:
            break
        if schedule and number == schedule[-1]:
            continue
        schedule.append(number)
    if len(schedule) < 2:
        raise SigmaScheduleError('That scheduler did not produce a descending sigma list.')
    return schedule


_SCHEMA = None
_COMPATIBLES = None
_PLUS_ONE = ('DPMSolver', 'UniPC', 'DEIS', 'SASolver')
_FLOW_MATCH = ('FlowMatch', 'FlowMap')
_NOISE = (
    'EulerDiscrete', 'EulerAncestral', 'LMSDiscrete', 'HeunDiscrete', 'KDPM2',
    'DEIS', 'DPMSolver', 'UniPC')


def scheduler_names() -> list[str]:
    return list(_schema())


def _compatibles():
    global _COMPATIBLES
    if _COMPATIBLES is None:
        import importlib.resources
        import json
        with importlib.resources.open_text(
                'dgenerate.console.schemas', 'compatibleschedulers.json') as file:
            _COMPATIBLES = json.load(file)
    return _COMPATIBLES


def _model_schedulers(model_type: str) -> dict | None:
    entry = _compatibles().get(model_type)
    if entry is None:
        return None
    if isinstance(entry, list):
        return {'default': None, 'schedulers': entry}
    return entry


def compatible_scheduler_names(model_type: str) -> list[str] | None:
    """Scheduler class names the model type can load, or ``None`` when it is unknown."""
    entry = _model_schedulers(model_type)
    if entry is None:
        return None
    return list(entry.get('schedulers') or [])


def default_scheduler_name(model_type: str | None) -> str | None:
    """Scheduler used when ``--scheduler`` is omitted, from the compatible-scheduler schema."""
    if not model_type:
        return None
    entry = _model_schedulers(model_type)
    if entry is None:
        return None
    name = entry.get('default')
    return name or None


def _sigma_capable_names() -> set[str]:
    """Scheduler classes that accept a custom sigma list, from the schema."""
    names = set()
    for entry in _compatibles().values():
        if isinstance(entry, dict):
            names.update(entry.get('accepts-sigmas') or [])
    return names


def scheduler_accepts_sigmas(name: str) -> bool:
    """``set_timesteps`` accepts a sigma list for this scheduler class."""
    return name in _sigma_capable_names()


def default_scheduler_accepts_sigmas(model_type: str | None) -> bool:
    """The model type's unnamed scheduler can take a sigma list.

    Unknown model types are treated as accepting one, so the form does not
    demand a scheduler it cannot identify.
    """
    name = default_scheduler_name(model_type)
    if not name:
        return True
    entry = _model_schedulers(model_type) if model_type else None
    if entry is None or 'accepts-sigmas' not in entry:
        return scheduler_accepts_sigmas(name)
    return name in set(entry.get('accepts-sigmas') or [])


def _for_model(names: list[str], model_type: str | None) -> list[str]:
    if not model_type:
        return names
    allowed = compatible_scheduler_names(model_type)
    if allowed is None:
        return names
    allowed = set(allowed)
    return [name for name in names if name in allowed]


def sigma_scheduler_names(model_type: str | None = None) -> list[str]:
    entry = _model_schedulers(model_type) if model_type else None
    if entry is not None and 'accepts-sigmas' in entry:
        accepted = set(entry.get('accepts-sigmas') or [])
    else:
        accepted = _sigma_capable_names()
    names = [name for name in _schema() if name in accepted]
    return _for_model(names, model_type)


def timestep_scheduler_names(model_type: str | None = None) -> list[str]:
    names = [name for name in _schema() if name in _TIMESTEP_SCHEDULERS]
    return _for_model(names, model_type)


def scheduler_supports_flow(name: str) -> bool:
    return 'use-flow-sigmas' in _schema().get(name, {})


def scheduler_requires_flow_sigmas(name: str) -> bool:
    """Custom sigmas are rejected unless ``use-flow-sigmas`` is on."""
    return name == 'UniPCMultistepScheduler'


def _noise_config(model_type: str | None) -> dict | None:
    entry = _model_schedulers(model_type) if model_type else None
    if not isinstance(entry, dict):
        return None
    return entry.get('noise-config') or None


def _flow_shift(model_type: str | None) -> float | None:
    """FlowMatch ``shift`` of the model type's reference checkpoint, when it is not 1."""
    entry = _model_schedulers(model_type) if model_type else None
    if not isinstance(entry, dict) or entry.get('flow-shift') is None:
        return None
    return float(entry['flow-shift'])


def pipeline_builds_sigmas(model_type: str | None) -> bool:
    """The model type's pipeline passes ``linspace(1, 1/steps, steps)`` when ``--sigmas`` is omitted.

    That list is the default for every scheduler the model loads, so it is the
    curve to start from. Qwen-Image 2.1 can carry a ``sample_sigmas`` list in
    its checkpoint instead, which the console cannot read.
    """
    entry = _model_schedulers(model_type) if model_type else None
    return isinstance(entry, dict) and entry.get('pipeline-sigmas') == 'linspace'


# Stays in sync with the MiniMax-H3 video pipeline, which appends the terminal 0.
_TERMINAL_STEP_MODEL_TYPES = frozenset({'minimax-h3-video'})


def terminal_steps(model_type: str | None) -> int:
    """Steps ``--inference-steps`` counts beyond the ``--sigmas`` values.

    MiniMax-H3 Video counts the terminal 0 as a step, so 29 values run as 30 steps.
    """
    return 1 if model_type in _TERMINAL_STEP_MODEL_TYPES else 0


def _model_spec(name: str, model_type: str | None) -> dict:
    """The scheduler's schema entry, with the model's beta schedule in place.

    A scheduler loads on top of the checkpoint's scheduler config, so its
    betas come from the checkpoint, not from the class defaults.
    """
    spec = _schema().get(name)
    if spec is None:
        raise SigmaScheduleError(f'Unknown scheduler {name}.')
    overrides = dict(_noise_config(model_type) or {})
    flow_shift = _flow_shift(model_type)
    if flow_shift is not None and any(name.startswith(prefix) for prefix in _FLOW_MATCH):
        overrides['shift'] = flow_shift
    if not overrides:
        return spec
    spec = dict(spec)
    for key, value in overrides.items():
        spec[key] = {'default': value}
    return spec


def is_flow_curve(name: str, flow: bool) -> bool:
    """The scheduler's sigmas are flow-matching noise fractions in (0, 1]."""
    if flow and scheduler_supports_flow(name):
        return True
    return (any(name.startswith(prefix) for prefix in _FLOW_MATCH)
            or name in ('MiniMaxH3Scheduler', 'PiflowScheduler'))


def _is_noise_curve(name: str, flow: bool) -> bool:
    if flow and scheduler_supports_flow(name):
        return False
    return any(name.startswith(prefix) for prefix in _NOISE)


def sigma_ceiling(model_type: str | None, name: str | None = None,
                  use_flow_sigmas: bool | None = None) -> float | None:
    """Largest sigma the scheduler can take, or ``None`` when there is no fixed limit.

    A flow sigma is the share of noise in the latent, so it stops at 1. A
    noise scheduler maps each sigma to a training timestep. A sigma above
    the last training step has no timestep, so the pipeline rejects it.
    ``None`` for EDM, and for a noise schedule whose model type has no
    reference beta schedule.
    """
    if name:
        spec = _schema().get(name, {})
        flow = bool(use_flow_sigmas) if use_flow_sigmas is not None else bool(
            _spec_default(spec, 'use-flow-sigmas', False))
        if is_flow_curve(name, flow):
            return 1.0
        if not _is_noise_curve(name, flow):
            return None
    noise = _noise_config(model_type)
    if not noise:
        return None
    return round(_train_sigmas(_model_spec('EulerDiscreteScheduler', model_type))[-1], 6)


def scheduler_curve(name: str, steps: int, use_flow_sigmas: bool | None = None,
                    model_type: str | None = None) -> list[float]:
    """Sigma list for ``name`` at ``steps``.

    Noise schedulers use the beta schedule of ``model_type``'s reference
    checkpoint when the schema has one, and the scheduler's defaults otherwise.
    """
    spec = _model_spec(name, model_type)
    steps = int(steps)
    if steps < 2:
        raise SigmaScheduleError('A sigma schedule needs at least two steps.')
    if pipeline_builds_sigmas(model_type):
        # The pipeline passes this list to whichever scheduler is loaded.
        return _clean(_linspace(1.0, 1.0 / steps, steps))
    flow = bool(use_flow_sigmas) if use_flow_sigmas is not None else bool(
        _spec_default(spec, 'use-flow-sigmas', False))
    if flow and scheduler_supports_flow(name):
        return _clean(_flow_linspace(spec, steps, name.startswith('UniPC')))
    if name == 'MiniMaxH3Scheduler':
        return _clean(_minimax(spec, steps))
    if name == 'PiflowScheduler':
        return _clean(_piflow(spec, steps))
    if name == 'EDMEulerScheduler':
        return _clean(_edm(spec, steps))
    if any(name.startswith(prefix) for prefix in _FLOW_MATCH):
        return _clean(_flow_match(spec, steps, shifted=False))
    if (name == 'EulerDiscreteScheduler'
            and _spec_default(spec, 'interpolation-type', 'linear') == 'log_linear'):
        return _clean(_log_linear_sigmas(spec, steps))
    if any(name.startswith(prefix) for prefix in _NOISE):
        plus_one = any(name.startswith(prefix) for prefix in _PLUS_ONE)
        return _clean(_noise_sigmas(spec, steps, plus_one))
    raise SigmaScheduleError(f'{name} does not publish a sigma list.')


def _log_linear_sigmas(spec: dict, steps: int) -> list[float]:
    """Euler's ``log_linear`` schedule, geometric from the highest to the lowest training sigma.

    Euler builds ``steps + 1`` points and samples toward the last one. A
    passed list ends at zero instead, so the last point is dropped.
    """
    train = _train_sigmas(spec)
    logs = _linspace(math.log(train[-1]), math.log(train[0]), steps + 1)
    return [math.exp(value) for value in logs[:steps]]


# Schedulers whose set_timesteps accepts an integer schedule, and whose
# default schedule is the spacing formula above. LCM and TCD use a
# distilled index list instead, so they are left out.
_TIMESTEP_SCHEDULERS = (
    'DDPMScheduler',
    'DPMSolverMultistepScheduler',
    'DPMSolverSinglestepScheduler',
    'EulerDiscreteScheduler',
    'FlowMatchEulerDiscreteScheduler',
    'HeunDiscreteScheduler',
)


def _integers(indices) -> list[int]:
    schedule = []
    for value in indices:
        number = int(round(float(value)))
        if number < 0:
            break
        if schedule and number >= schedule[-1]:
            continue
        schedule.append(number)
    if len(schedule) < 2:
        raise SigmaScheduleError('That scheduler did not produce a descending timestep list.')
    return schedule


def timestep_ceiling(name: str | None, model_type: str | None = None) -> int | None:
    """Largest timestep the scheduler can place on this model, or ``None``.

    Noise schedulers index training steps from 0. FlowMatch divides by the
    training length, so its top timestep is the length itself. Mirrors
    ``videopipelines.schedules._timestep_limit``. A blank ``name`` means
    the model type's default scheduler.
    """
    name = name or default_scheduler_name(model_type)
    if not name or name not in _schema():
        return None
    count = int(_spec_default(_model_spec(name, model_type), 'num-train-timesteps', 1000))
    if name.startswith('FlowMatch'):
        return count
    return count - 1


def scheduler_timesteps(name: str, steps: int, use_flow_sigmas: bool | None = None,
                        model_type: str | None = None) -> list[int]:
    """Integer timesteps for ``name`` at ``steps``.

    The training length comes from ``model_type``'s reference checkpoint
    when the schema has one, and from the scheduler's defaults otherwise.
    """
    spec = _model_spec(name, model_type)
    if name not in _TIMESTEP_SCHEDULERS:
        raise SigmaScheduleError(f'{name} does not accept a timestep list.')
    steps = int(steps)
    if steps < 2:
        raise SigmaScheduleError('A timestep schedule needs at least two steps.')
    count = int(_spec_default(spec, 'num-train-timesteps', 1000))
    if pipeline_builds_sigmas(model_type):
        # dgenerate pins FlowMatch sigmas to timestep / count, so this list
        # runs the pipeline's own grid without its resolution shift.
        return _integers(value * count for value in _linspace(1.0, 1.0 / steps, steps))
    flow = bool(use_flow_sigmas) if use_flow_sigmas is not None else bool(
        _spec_default(spec, 'use-flow-sigmas', False))
    if flow and scheduler_supports_flow(name):
        return _integers(value * count for value in _flow_linspace(spec, steps, False))
    if name.startswith('FlowMatch'):
        return _integers(value * count for value in _flow_match(spec, steps))
    plus_one = name.startswith('DPMSolver')
    return _integers(_timestep_indices(spec, steps, plus_one))


def parse_timesteps(text: str) -> list[int]:
    """A strictly descending list of at least two non-negative integers."""
    parts = split_values(text)
    if len(parts) < 2:
        raise SigmaScheduleError('A timestep schedule needs at least two values.')
    try:
        schedule = [int(part) for part in parts]
    except ValueError as error:
        raise SigmaScheduleError('Timesteps must be integers.') from error
    if any(value < 0 for value in schedule):
        raise SigmaScheduleError('Timesteps must be non-negative.')
    if any(left <= right for left, right in zip(schedule, schedule[1:])):
        raise SigmaScheduleError('Timesteps must descend.')
    return schedule


def clamp_timestep(schedule: list[int], index: int, proposed: float) -> int:
    """Keep one integer strictly between its neighbors. The tail may be 0."""
    value = int(round(proposed))
    if index > 0:
        value = min(value, schedule[index - 1] - 1)
    if index + 1 < len(schedule):
        value = max(value, schedule[index + 1] + 1)
    else:
        value = max(value, 0)
    return value


def argument(values: list[float], base: list[float], scale: float, edited: bool) -> str | None:
    """
    Text for ``--sigmas``, or ``None`` when the starting schedule is unchanged.

    A changed schedule is always written as numbers. ``expr: sigmas * scale``
    would scale the loaded scheduler's own schedule, which depends on the
    checkpoint's spacing and can differ from the curve the editor showed.
    """
    if not edited and len(values) == len(base) and abs(scale - 1.0) < 1e-9:
        return None
    return ','.join(format_number(value) for value in values)
