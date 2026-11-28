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
import ast
import inspect
import itertools

import diffusers
import diffusers.schedulers
import typing
import dgenerate.types as _types
import numpy

import dgenerate.textprocessing as _textprocessing
import diffusers.loaders
import dgenerate.messages as _messages
import collections.abc


class SchedulerLoadError(Exception):
    """
    Base class for scheduler loading exceptions.
    """


class SchedulerArgumentError(SchedulerLoadError):
    """
    Scheduler URI argument error.
    """
    pass


class InvalidSchedulerNameError(SchedulerLoadError):
    """
    Unknown scheduler name used.
    """
    pass


def _resolve_karras_schedulers():
    """Resolves `KarrasDiffusionSchedulers` enum values to actual scheduler class names in `diffusers.schedulers`."""
    scheduler_names = {}

    for scheduler_name in diffusers.schedulers.KarrasDiffusionSchedulers.__members__.keys():
        if hasattr(diffusers.schedulers, scheduler_name):
            scheduler_names[diffusers.schedulers.KarrasDiffusionSchedulers[scheduler_name]] = getattr(
                diffusers.schedulers, scheduler_name)

    return scheduler_names


_KARRAS_SCHEDULERS_MAP = _resolve_karras_schedulers()


def _expand_with_compatibles(scheduler_names):
    expanded_schedulers = set(scheduler_names)

    for scheduler_cls in list(expanded_schedulers):

        compatibles = getattr(scheduler_cls, "_compatibles", None)
        if not compatibles:
            continue

        for compatible in compatibles:
            if isinstance(compatible, diffusers.schedulers.KarrasDiffusionSchedulers):
                expanded_schedulers.add(_KARRAS_SCHEDULERS_MAP.get(compatible, None))
            elif inspect.isclass(compatible) and issubclass(compatible, diffusers.schedulers.SchedulerMixin):
                expanded_schedulers.add(compatible)

    return list(expanded_schedulers)


def get_compatible_schedulers(pipeline_cls: type[diffusers.DiffusionPipeline]) -> list[type[diffusers.SchedulerMixin]]:
    """
    Finds all compatible scheduler classes for a given diffusers pipeline class without instantiating it.

    :param pipeline_cls: The pipeline class, for example :py:class:`diffusers.StableDiffusionPipeline`
    :return A list of compatible scheduler class types
    """
    if pipeline_cls is diffusers.StableDiffusionLatentUpscalePipeline:
        # Seems to only work with this scheduler
        return [diffusers.EulerDiscreteScheduler]

    if any(pipeline_cls is x for x in (diffusers.IFPipeline,
                                       diffusers.IFInpaintingPipeline,
                                       diffusers.IFImg2ImgPipeline,
                                       diffusers.IFSuperResolutionPipeline,
                                       diffusers.IFInpaintingSuperResolutionPipeline,
                                       diffusers.IFImg2ImgSuperResolutionPipeline)):
        # same here
        return [diffusers.DDPMScheduler]

    compatible_schedulers = set()

    # Get constructor signature. String annotations from postponed
    # evaluation are resolved so a hint like FlowMatchEulerDiscreteScheduler
    # is the class, not the name.
    init_sig = inspect.signature(pipeline_cls.__init__)
    try:
        hints = typing.get_type_hints(pipeline_cls.__init__)
    except Exception:
        hints = {}

    for name, param in init_sig.parameters.items():
        param_type = hints.get(name, param.annotation)

        # Case 1: Direct scheduler class type hint (e.g., `param: DDIMScheduler`)
        if (inspect.isclass(param_type)
                and param_type is not diffusers.schedulers.SchedulerMixin
                and issubclass(param_type, diffusers.schedulers.SchedulerMixin)):
            compatible_schedulers.add(param_type)

        # Case 2: Union type hint (e.g., `Union[DDIMScheduler, EulerScheduler]`)
        elif _types.is_union(param_type):
            for sub_type in typing.get_args(param_type):
                if inspect.isclass(sub_type) and issubclass(sub_type, diffusers.schedulers.SchedulerMixin):
                    compatible_schedulers.add(sub_type)

        # Case 3: Enum-based schedulers (KarrasDiffusionSchedulers)
        elif param_type is diffusers.schedulers.KarrasDiffusionSchedulers:
            compatible_schedulers.update(_KARRAS_SCHEDULERS_MAP.values())

    # The modular pipeline constructor does not annotate its scheduler.
    modular = _modular_pipeline_scheduler(pipeline_cls)
    if modular is not None:
        return [getattr(diffusers, modular)]

    video = _video_pipeline_scheduler_names(pipeline_cls)
    if video is not None:
        return [getattr(diffusers, name) for name in video if hasattr(diffusers, name)]

    # Published Wan 2.1 and 2.2 checkpoints use UniPCMultistepScheduler.
    # The pipeline annotation only names FlowMatchEulerDiscreteScheduler, so
    # flow-shift on that checkpoint scheduler would otherwise be rejected.
    if (pipeline_cls.__name__.startswith('Wan')
            and not pipeline_cls.__name__.startswith('WanAnimate')):
        compatible_schedulers.add(diffusers.UniPCMultistepScheduler)

    # Expand using _compatibles
    compatibles = _expand_with_compatibles(compatible_schedulers)

    if issubclass(pipeline_cls, (diffusers.loaders.StableDiffusionLoraLoaderMixin,
                                 diffusers.loaders.StableDiffusionXLLoraLoaderMixin)):
        compatibles.append(diffusers.LCMScheduler)
        # Hyper-SD 1-step LoRAs are trained for TCDScheduler, which is not
        # part of the Karras scheduler enum those pipelines declare.
        compatibles.append(diffusers.TCDScheduler)

    return compatibles


# Schedulers the video loaders accept. These pipelines do not publish a
# usable constructor annotation: Motif names the scheduler base class, and
# Wan-Animate-2 and MiniMax-H3 video are modular pipelines with no scheduler
# parameter. The image models below are read from the pipeline class.
_VIDEO_SCHEDULER_NAMES = {
    'ltx2': ('FlowMatchEulerDiscreteScheduler',),
    'ltx-video': ('FlowMatchEulerDiscreteScheduler',),
    'wan': ('FlowMatchEulerDiscreteScheduler', 'UniPCMultistepScheduler'),
    'wan-animate': ('UniPCMultistepScheduler',),
    'wan-animate-2': (
        'DPMSolverMultistepScheduler',
        'EulerDiscreteScheduler',
        'FlowMatchEulerDiscreteScheduler',
        'UniPCMultistepScheduler',
    ),
    'hyvideo-15': ('FlowMatchEulerDiscreteScheduler',),
    'motif-video': (
        'DPMSolverMultistepScheduler',
        'FlowMatchEulerDiscreteScheduler',
    ),
    'kandinsky5-video': ('FlowMatchEulerDiscreteScheduler',),
    'kandinsky6-video': ('FlowMatchEulerDiscreteScheduler', 'PiflowScheduler'),
    'minimax-h3-video': ('MiniMaxH3Scheduler',),
}

# Video pipeline classes with no usable scheduler annotation, keyed to their
# entry above. Subclasses, such as the distilled Wan-Animate-2 pipeline, match.
_VIDEO_PIPELINE_MODEL_TYPES = {
    'MotifVideoPipeline': 'motif-video',
    'MotifVideoImage2VideoPipeline': 'motif-video',
    'WanAnimate2ModularPipeline': 'wan-animate-2',
    'MiniMaxH3ModularPipeline': 'minimax-h3-video',
}


def _video_pipeline_scheduler_names(pipeline_cls) -> tuple[str, ...] | None:
    for base in getattr(pipeline_cls, '__mro__', (pipeline_cls,)):
        model_type = _VIDEO_PIPELINE_MODEL_TYPES.get(getattr(base, '__name__', ''))
        if model_type is not None:
            return _VIDEO_SCHEDULER_NAMES[model_type]
    return None


def _model_type_string(model_type) -> str:
    import dgenerate.pipelinewrapper.enums as _enums

    if isinstance(model_type, str):
        return _enums.get_model_type_string(_enums.get_model_type_enum(model_type))
    return _enums.get_model_type_string(model_type)


def compatible_scheduler_names(model_type) -> list[str]:
    """Scheduler class names ``--model-type`` can load.

    Video entries are the schedulers the video loader accepts. Image entries
    are :func:`get_compatible_schedulers` on the pipeline class that runs.
    """
    name = _model_type_string(model_type)
    video = _VIDEO_SCHEDULER_NAMES.get(name)
    if video is not None:
        return list(video)
    return _image_scheduler_names(name)


# The checkpoint fixes the sampling grid. MiniMax-H3 Image has one output
# head per step of that grid, so --sigmas and --timesteps are rejected.
_FIXED_SCHEDULE_MODEL_TYPES = frozenset({'minimax-h3-image'})


def model_type_accepts_custom_sigmas(model_type) -> bool:
    """``--sigmas`` can be used with ``--model-type`` when its scheduler accepts them."""
    return _model_type_string(model_type) not in _FIXED_SCHEDULE_MODEL_TYPES


# The constructor annotation for these is the whole Karras enum. The name
# here is the scheduler the reference checkpoint loads when ``--scheduler``
# is omitted. SD 2.x checkpoints that ship Euler still have to name it once
# a sigma list is passed, because the model type's usual default is PNDM.
_KARRAS_MODEL_DEFAULTS = {
    'sd': 'PNDMScheduler',
    'pix2pix': 'PNDMScheduler',
    'upscaler-x4': 'DDIMScheduler',
    'sdxl': 'EulerDiscreteScheduler',
    'sdxl-pix2pix': 'EulerDiscreteScheduler',
    'kolors': 'EulerDiscreteScheduler',
}

# Beta schedule of the reference checkpoint for each noise-prediction model
# type, read from its scheduler_config.json. A noise scheduler cloned from
# that config cannot reach a sigma above the one at its last training step.
_REFERENCE_NOISE_CONFIG = {
    'sd': {'beta-start': 0.00085, 'beta-end': 0.012,
           'beta-schedule': 'scaled_linear', 'num-train-timesteps': 1000},
    'pix2pix': {'beta-start': 0.00085, 'beta-end': 0.012,
                'beta-schedule': 'scaled_linear', 'num-train-timesteps': 1000},
    'sdxl': {'beta-start': 0.00085, 'beta-end': 0.012,
             'beta-schedule': 'scaled_linear', 'num-train-timesteps': 1000},
    'sdxl-pix2pix': {'beta-start': 0.00085, 'beta-end': 0.012,
                     'beta-schedule': 'scaled_linear', 'num-train-timesteps': 1000},
    'kolors': {'beta-start': 0.00085, 'beta-end': 0.014,
               'beta-schedule': 'scaled_linear', 'num-train-timesteps': 1100},
    'upscaler-x2': {'beta-start': 0.00085, 'beta-end': 0.012,
                    'beta-schedule': 'scaled_linear', 'num-train-timesteps': 1000,
                    'interpolation-type': 'log_linear'},
    'upscaler-x4': {'beta-start': 0.0001, 'beta-end': 0.02,
                    'beta-schedule': 'scaled_linear', 'num-train-timesteps': 1000},
}


def reference_noise_config(model_type) -> dict | None:
    """Beta schedule of the model type's reference checkpoint, or ``None``.

    Keys are URI argument names: ``beta-start``, ``beta-end``,
    ``beta-schedule``, ``num-train-timesteps``, and, when the checkpoint
    sets it, ``interpolation-type``.
    """
    config = _REFERENCE_NOISE_CONFIG.get(_model_type_string(model_type))
    return dict(config) if config is not None else None


# FlowMatch ``shift`` of the reference checkpoint, for model types whose
# pipeline leaves the default schedule to the scheduler and whose shift is not
# the class default of 1. A passed sigma list is shifted again by this value.
_REFERENCE_FLOW_SHIFT = {
    'sd3': 3.0,
    'kandinsky5-image': 5.0,
    'kandinsky5-video': 5.0,
    'kandinsky6-video': 5.0,
}


def reference_flow_shift(model_type) -> float | None:
    """FlowMatch ``shift`` of the model type's reference checkpoint, or ``None``."""
    return _REFERENCE_FLOW_SHIFT.get(_model_type_string(model_type))


# Wan's constructor names FlowMatch, but the published checkpoints load UniPC.
_VIDEO_DEFAULT_SCHEDULER = {
    'wan': 'UniPCMultistepScheduler',
}


def default_scheduler_name(model_type) -> str | None:
    """Scheduler class used when ``--scheduler`` is left unset.

    Video entries are the loader's usual checkpoint scheduler. Image entries
    are the pipeline constructor's scheduler class, or the reference
    checkpoint when that annotation is the Karras enum.
    """
    name = _model_type_string(model_type)
    video = _VIDEO_SCHEDULER_NAMES.get(name)
    if video is not None:
        return _VIDEO_DEFAULT_SCHEDULER.get(name, video[0])
    if name in _KARRAS_MODEL_DEFAULTS:
        return _KARRAS_MODEL_DEFAULTS[name]
    classes = _model_pipeline_classes(name)
    if not classes:
        return _modular_default_scheduler(name)
    return _declared_scheduler_name(classes[0])


# set_timesteps lists ``sigmas`` and then raises if one is passed.
_REJECTS_CUSTOM_SIGMAS = frozenset({'PiflowScheduler'})


def scheduler_accepts_custom_sigmas(scheduler) -> bool:
    """``set_timesteps`` accepts a custom sigma list for this scheduler.

    A declared ``sigmas`` parameter is not enough. PiFlow documents the
    argument and then rejects anything but its own distilled schedule.
    """
    if isinstance(scheduler, str):
        scheduler_cls = getattr(diffusers, scheduler, None)
        name = scheduler
    else:
        scheduler_cls = scheduler
        name = getattr(scheduler_cls, '__name__', '').split('.')[-1]
    if scheduler_cls is None or name in _REJECTS_CUSTOM_SIGMAS:
        return False
    try:
        parameters = inspect.signature(scheduler_cls.set_timesteps).parameters
    except (TypeError, ValueError):
        return False
    return 'sigmas' in parameters


def scheduler_accepts_custom_timesteps(scheduler) -> bool:
    """``set_timesteps`` accepts a custom timestep list for this scheduler.

    PiFlow documents the argument and then rejects it, as it does sigmas.
    """
    if isinstance(scheduler, str):
        scheduler_cls = getattr(diffusers, scheduler, None)
        name = scheduler
    else:
        scheduler_cls = scheduler if inspect.isclass(scheduler) else type(scheduler)
        name = scheduler_cls.__name__
    if scheduler_cls is None or name in _REJECTS_CUSTOM_SIGMAS:
        return False
    try:
        parameters = inspect.signature(scheduler_cls.set_timesteps).parameters
    except (TypeError, ValueError):
        return False
    return 'timesteps' in parameters


# Modular pipelines take their components from a blocks spec, so the
# constructor does not annotate a scheduler class.
_MODULAR_PIPELINE_SCHEDULERS = {
    'Flux3ModularPipeline': 'FlowMatchEulerDiscreteScheduler',
    'MiniMaxH3ImageModularPipeline': 'MiniMaxH3Scheduler',
}

_MODULAR_MODEL_SCHEDULERS = {
    'flux3': 'FlowMatchEulerDiscreteScheduler',
    'minimax-h3-image': 'MiniMaxH3Scheduler',
    'minimax-h3-video': 'MiniMaxH3Scheduler',
}


def _modular_default_scheduler(model_type: str) -> str | None:
    return _MODULAR_MODEL_SCHEDULERS.get(model_type)


def _modular_pipeline_scheduler(pipeline_cls) -> str | None:
    """Scheduler of a modular pipeline class or of a class it extends.

    A checkpoint's ``modular_model_index.json`` can name a subclass, such as
    ``Flux3DistilledModularPipeline`` for FLUX.3 Image Klein.
    """
    for base in getattr(pipeline_cls, '__mro__', (pipeline_cls,)):
        name = _MODULAR_PIPELINE_SCHEDULERS.get(getattr(base, '__name__', ''))
        if name is not None:
            return name
    return None


def _declared_scheduler_name(pipeline_cls) -> str | None:
    """Scheduler class named by the pipeline constructor, before compatibles expand."""
    modular = _modular_pipeline_scheduler(pipeline_cls)
    if modular is not None:
        return modular

    init_sig = inspect.signature(pipeline_cls.__init__)
    try:
        hints = typing.get_type_hints(pipeline_cls.__init__)
    except Exception:
        hints = {}
    param = init_sig.parameters.get('scheduler')
    if param is None:
        return None
    param_type = hints.get('scheduler', param.annotation)
    if (inspect.isclass(param_type)
            and param_type is not diffusers.schedulers.SchedulerMixin
            and issubclass(param_type, diffusers.schedulers.SchedulerMixin)):
        return param_type.__name__.split('.')[-1]
    if _types.is_union(param_type):
        for sub_type in typing.get_args(param_type):
            if (inspect.isclass(sub_type)
                    and sub_type is not diffusers.schedulers.SchedulerMixin
                    and issubclass(sub_type, diffusers.schedulers.SchedulerMixin)):
                return sub_type.__name__.split('.')[-1]
    return None


def _model_pipeline_classes(model_type: str) -> list:
    import dgenerate.pipelinewrapper.enums as _enums
    import dgenerate.pipelinewrapper.pipelines as _pipelines
    import dgenerate.pipelinewrapper.textencoderschema as _text_encoder_schema

    classes = list(_text_encoder_schema._pipeline_classes().get(model_type) or [])
    if classes:
        return classes
    preferred = {
        'flux-fill': _enums.PipelineType.INPAINT,
        'flux-kontext': _enums.PipelineType.IMG2IMG,
    }
    kinds = []
    if model_type in preferred:
        kinds.append(preferred[model_type])
    for kind in (
            _enums.PipelineType.TXT2IMG,
            _enums.PipelineType.IMG2IMG,
            _enums.PipelineType.INPAINT):
        if kind not in kinds:
            kinds.append(kind)
    for kind in kinds:
        try:
            found = _pipelines.get_pipeline_class(
                _enums.get_model_type_enum(model_type), kind, help_mode=True)
        except Exception:
            continue
        if found is diffusers.StableDiffusionPipeline and model_type != 'sd':
            continue
        return [found]
    return []


def _image_scheduler_names(model_type: str) -> list[str]:
    classes = _model_pipeline_classes(model_type)
    names = set()
    for pipeline_cls in classes:
        for scheduler_cls in get_compatible_schedulers(pipeline_cls):
            if scheduler_cls is None:
                continue
            scheduler_name = scheduler_cls.__name__.split('.')[-1]
            if scheduler_name == 'SchedulerMixin':
                continue
            names.add(scheduler_name)
    return sorted(names)


_scheduler_option_args = {
    diffusers.DDIMScheduler: {
        "beta_schedule": ["linear", "scaled_linear", "squaredcos_cap_v2"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.DDPMScheduler: {
        "beta_schedule": ["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"],
        'variance_type': ["fixed_small", "fixed_small_log",
                          "fixed_large", "fixed_large_log",
                          "learned", "learned_range"]
    },

    diffusers.DDPMWuerstchenScheduler: {
        # Note: This scheduler doesn't use the standard beta_schedule, prediction_type, or timestep_spacing
        # It uses custom parameters: scaler and s
    },

    diffusers.DEISMultistepScheduler: {
        "beta_schedule": ["linear", "scaled_linear", "squaredcos_cap_v2"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.DPMSolverMultistepScheduler: {
        "beta_schedule": ["linear", "scaled_linear", "squaredcos_cap_v2"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"],
        "algorithm_type": ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"],
        "solver_type": ["midpoint", "heun"],
        "final_sigmas_type": ["zero", "sigma_min"],
        "variance_type": ["learned", "learned_range"]
    },

    diffusers.DPMSolverSDEScheduler: {
        "beta_schedule": ["linear", "scaled_linear"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.DPMSolverSinglestepScheduler: {
        "beta_schedule": ["linear", "scaled_linear", "squaredcos_cap_v2"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "algorithm_type": ["dpmsolver", "dpmsolver++", "sde-dpmsolver++"],
        "solver_type": ["midpoint", "heun"],
        "final_sigmas_type": ["zero", "sigma_min"],
        "variance_type": ["learned", "learned_range"]

    },

    diffusers.EDMEulerScheduler: {
        "sigma_schedule": ["karras", "exponential"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "final_sigmas_type": ["zero", "sigma_min"]
    },

    diffusers.EulerAncestralDiscreteScheduler: {
        "beta_schedule": ["linear", "scaled_linear"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.EulerDiscreteScheduler: {
        "beta_schedule": ["linear", "scaled_linear", "squaredcos_cap_v2"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"],
        "timestep_type": ["discrete", "continuous"],
        "interpolation_type": ["linear", "log_linear"],
        "final_sigmas_type": ["zero", "sigma_min"]
    },

    diffusers.FlowMatchEulerDiscreteScheduler: {
        "time_shift_type": ["exponential", "linear"]
    },

    diffusers.MiniMaxH3Scheduler: {
        "euler_update_form": ["blend", "velocity"]
    },

    diffusers.PiflowScheduler: {},

    diffusers.HeunDiscreteScheduler: {
        "beta_schedule": ["linear", "scaled_linear"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.KDPM2AncestralDiscreteScheduler: {
        "beta_schedule": ["linear", "scaled_linear"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.KDPM2DiscreteScheduler: {
        "beta_schedule": ["linear", "scaled_linear"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.LCMScheduler: {
        "beta_schedule": ["linear", "scaled_linear", "squaredcos_cap_v2"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.TCDScheduler: {
        "beta_schedule": ["linear", "scaled_linear", "squaredcos_cap_v2"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.LMSDiscreteScheduler: {
        "beta_schedule": ["linear", "scaled_linear"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.PNDMScheduler: {
        "beta_schedule": ["linear", "scaled_linear", "squaredcos_cap_v2"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"]
    },

    diffusers.UniPCMultistepScheduler: {
        "beta_schedule": ["linear", "scaled_linear", "squaredcos_cap_v2"],
        "prediction_type": ["epsilon", "sample", "v_prediction"],
        "timestep_spacing": ["leading", "trailing", "linspace"],
        "solver_type": ["bh1", "bh2"],
        "final_sigmas_type": ["zero", "sigma_min"]
    }
}


def get_scheduler_uri_schema(scheduler: type[diffusers.SchedulerMixin] | list[type[diffusers.SchedulerMixin]]):
    """
    Return a schema describing initialization arguments from a ``diffusers`` scheduler type, or list of scheduler types.

    This returns a set of schemas keyed by scheduler name, which are identical to the schema format returned by
    :py:meth:`dgenerate.plugin.Plugin.get_accepted_args_schema`.

    Arguments which cannot be passed through a URI such as class references are omitted.

    :param scheduler: ``diffusers`` scheduler type, or list of them.

    :return: ``dict`` schema.
    """
    if not isinstance(scheduler, list):
        scheduler = [scheduler]

    schema = dict()

    for class_type in scheduler:
        # first argument is the config cloning behavior
        parameter_schema = {
            'clone-config': {
                'optional': False,
                'default': True,
                'types': ['bool']
            }
        }

        schema[class_type.__name__] = parameter_schema

        def _type_name(t):
            return (str(t) if t.__module__ != 'builtins' else t.__name__).strip()

        def _resolve_union(t):
            t_name = _type_name(t)
            if _types.is_union(t):
                return set(itertools.chain.from_iterable(
                    [_resolve_union(t) for t in parameter.annotation.__args__]))
            return [t_name]

        def _filter_types(typs):
            o = set()
            for t in typs:
                if t.startswith('typing.List'):
                    o.add('list')
                elif t == "<class 'numpy.ndarray'>":
                    o.add('list')
                elif t.startswith("<class"):
                    pass
                else:
                    o.add(t)
            return list(o)

        option_args = _scheduler_option_args.get(class_type, dict())

        for parameter_name, parameter in inspect.signature(class_type.__init__).parameters.items():
            if parameter_name == 'self':
                continue

            parameter_details = dict()

            type_name = _type_name(parameter.annotation)

            if _types.is_union(parameter.annotation):
                union_args = _resolve_union(parameter.annotation)
                if 'NoneType' in union_args:
                    parameter_details['optional'] = True
                    union_args.remove('NoneType')

                filtered_types = _filter_types(list(sorted(union_args)))
                if not filtered_types:
                    continue
                parameter_details['types'] = filtered_types

            else:
                filtered_types = _filter_types([type_name])
                if not filtered_types:
                    continue
                parameter_details['optional'] = False
                parameter_details['types'] = filtered_types

            if isinstance(parameter.default, list):
                if not all(isinstance(i, typing.SupportsIndex) or
                           not isinstance(i, collections.abc.Iterable) for i in parameter.default):
                    # cannot support multiple dimensions
                    continue
            if isinstance(parameter.default, numpy.ndarray):
                if parameter.default.ndim != 1:
                    # cannot support multiple dimensions
                    continue

            if parameter.default is not inspect.Parameter.empty:
                parameter_details['default'] = parameter.default
                if parameter.default is None:
                    # Diffusers often annotates these as `float = None`, not Optional[float]
                    parameter_details['optional'] = True

            if parameter.name in option_args:
                parameter_details['options'] = option_args[parameter.name]

            parameter_schema[_textprocessing.dashup(parameter_name)] = parameter_details

        if class_type is diffusers.TCDScheduler:
            # eta is a pipeline call argument, not a TCDScheduler constructor
            # argument. Hyper-SD 1-step uses eta=1.0; the scheduler default is 0.3.
            parameter_schema['eta'] = {
                'optional': True,
                'default': 0.3,
                'types': ['float']
            }
    return schema


def parse_scheduler_uri(scheduler_type: type[diffusers.SchedulerMixin], scheduler_uri: _types.Uri) -> dict:
    """
    Parse the arguments of a scheduler URI into scheduler constructor keyword arguments.

    :raises SchedulerArgumentError: If invalid arguments are supplied to the scheduler via the URI.

    :param scheduler_type: The scheduler class the URI names.
    :param scheduler_uri: Scheduler URI.
    :return: Keyword arguments, including ``clone_config`` if it was given.
    """

    def _get_uri_arg_value(
            scheduler: str,
            value: typing.Any,
            arg_name: str,
            optional: bool,
            types: list):

        if isinstance(value, list):
            return value
        elif optional and value.lower() == 'none':
            return None
        elif any(t == 'list' for t in types):
            try:
                val = ast.literal_eval(value)
                if not isinstance(val, (list, tuple, set)):
                    return [val]
                else:
                    return val
            except (ValueError, SyntaxError) as e:
                raise SchedulerArgumentError(
                    f'{scheduler} argument "{arg_name}" '
                    f'must be a singular literal, list, '
                    f'tuple, or set value in python syntax.'
                ) from e
        elif any(t == 'float' for t in types):
            try:
                return float(value)
            except ValueError as e:
                raise SchedulerArgumentError(
                    f'{scheduler} argument "{arg_name}" '
                    f'must be a floating point value.'
                ) from e
        elif any(t == 'int' for t in types):
            try:
                return int(value)
            except ValueError as e:
                raise SchedulerArgumentError(
                    f'{scheduler} argument "{arg_name}" '
                    f'must be an integer value.'
                ) from e
        elif any(t == 'bool' for t in types):
            try:
                return _types.parse_bool(value)
            except ValueError as e:
                raise SchedulerArgumentError(
                    f'{scheduler} argument "{arg_name}" '
                    f'must be a boolean value.'
                ) from e

        try:
            # string literal?
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # token (string)
            return value

    schema = get_scheduler_uri_schema(scheduler_type)[scheduler_type.__name__]

    parser = _textprocessing.ConceptUriParser(
        'Scheduler',
        known_args=list(schema.keys()),
        args_raw=[k for k, v in schema.items()
                  if any(t == 'list' for t in v['types'])])

    try:
        result = parser.parse(scheduler_uri)
    except _textprocessing.ConceptUriParseError as e:
        raise SchedulerArgumentError(e) from e

    args = {_textprocessing.dashdown(k): _get_uri_arg_value(
        scheduler_type.__name__, v, k, schema[k]['optional'], schema[k]['types'])
        for k, v in result.args.items()}

    option_args = _scheduler_option_args.get(scheduler_type, dict())
    for arg, value in args.items():
        if arg in option_args and value not in option_args[arg]:
            raise SchedulerArgumentError(
                f'Invalid value "{value}" for argument "{_textprocessing.dashup(arg)}" '
                f'of scheduler "{scheduler_type.__name__}", '
                f'valid options are: '
                f'{_textprocessing.oxford_comma(option_args[arg], "or")}')

    return args


def _find_scheduler(scheduler_uri: _types.Uri, schedulers: list[type[diffusers.SchedulerMixin]]):
    name = scheduler_uri.split(';')[0].strip()
    return next((s for s in schedulers if s.__name__.startswith(name)), None)


def check_scheduler_uri(scheduler_uri: _types.Uri):
    """
    Validate the scheduler name and URI arguments without a pipeline. Whether the
    scheduler is compatible with a particular pipeline is not checked.

    :raises InvalidSchedulerNameError: If the scheduler name is unknown.
    :raises SchedulerArgumentError: If invalid arguments are supplied to the scheduler via the URI.

    :param scheduler_uri: Scheduler URI.
    """
    schedulers = list(_scheduler_option_args)
    scheduler_type = _find_scheduler(scheduler_uri, schedulers)
    if scheduler_type is None:
        raise InvalidSchedulerNameError(
            f'Scheduler named "{scheduler_uri}" is not a valid scheduler, options are:\n\n' +
            '\n'.join(sorted(' ' * 4 + _textprocessing.quote(i.__name__) for i in schedulers)))
    parse_scheduler_uri(scheduler_type, scheduler_uri)


def load_scheduler(pipeline: diffusers.DiffusionPipeline, scheduler_uri: _types.Uri | None):
    """
    Load a specific compatible scheduler class name onto a huggingface diffusers pipeline object.

    Passing ``None`` to the URI reloads the original scheduler that the pipeline was loaded
    with, if no new scheduler has been set since then, this is a no-op.

    :raises InvalidSchedulerNameError: If an invalid scheduler name is specified specifically.
    :raises SchedulerArgumentError: If invalid arguments are supplied to the scheduler via the URI.

    :param pipeline: pipeline object
    :param scheduler_uri: Compatible scheduler URI.
    """

    if scheduler_uri is None:
        if hasattr(pipeline, '_DGENERATE_ORIGINAL_SCHEDULER'):
            pipeline.scheduler = pipeline._DGENERATE_ORIGINAL_SCHEDULER
        return

    compatibles = get_compatible_schedulers(pipeline.__class__)

    scheduler_type = _find_scheduler(scheduler_uri, compatibles)
    if scheduler_type is None:
        raise InvalidSchedulerNameError(
            f'Scheduler named "{scheduler_uri}" is not a valid compatible scheduler, '
            'options are:\n\n' + '\n'.join(
                sorted(' ' * 4 + _textprocessing.quote(i.__name__.split('.')[-1]) for i in compatibles)))

    args = parse_scheduler_uri(scheduler_type, scheduler_uri)
    # eta is forwarded to the pipeline call. See _get_pipeline_defaults.
    call_eta = args.pop('eta', None)

    _messages.debug_log(
        f'Constructing Scheduler: "{scheduler_type.__name__}", URI Args: {args}')

    try:
        if not hasattr(pipeline, '_DGENERATE_ORIGINAL_SCHEDULER'):
            # first time
            pipeline._DGENERATE_ORIGINAL_SCHEDULER = pipeline.scheduler

        clone_config = args.pop('clone_config', True)

        if clone_config:
            # init from original scheduler config
            # apply any user overrides over top
            pipeline.scheduler = scheduler_type.from_config(
                pipeline._DGENERATE_ORIGINAL_SCHEDULER.config, **args
            )
        else:
            # raw init with possible overrides to defaults
            pipeline.scheduler = scheduler_type(**args)

    except Exception as e:
        raise SchedulerArgumentError(
            f'Error constructing scheduler "{scheduler_type.__name__}" '
            f'with given URI argument values, encountered error: {e}') from e

    if call_eta is not None:
        pipeline.scheduler._dgenerate_call_eta = call_eta

    _messages.debug_log(
        f'Scheduler: "{scheduler_type.__name__}", '
        f'Successfully added to pipeline: {pipeline.__class__.__name__}')
