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

    # Get constructor signature
    init_sig = inspect.signature(pipeline_cls.__init__)

    for param in init_sig.parameters.values():
        param_type = param.annotation

        # Case 1: Direct scheduler class type hint (e.g., `param: DDIMScheduler`)
        if inspect.isclass(param_type) and issubclass(param_type, diffusers.schedulers.SchedulerMixin):
            compatible_schedulers.add(param_type)

        # Case 2: Union type hint (e.g., `Union[DDIMScheduler, EulerScheduler]`)
        elif typing.get_origin(param_type) is typing.Union:
            for sub_type in typing.get_args(param_type):
                if inspect.isclass(sub_type) and issubclass(sub_type, diffusers.schedulers.SchedulerMixin):
                    compatible_schedulers.add(sub_type)

        # Case 3: Enum-based schedulers (KarrasDiffusionSchedulers)
        elif param_type is diffusers.schedulers.KarrasDiffusionSchedulers:
            compatible_schedulers.update(_KARRAS_SCHEDULERS_MAP.values())

    # Expand using _compatibles
    compatibles = _expand_with_compatibles(compatible_schedulers)

    if any(pipeline_cls is x for x in (diffusers.loaders.StableDiffusionLoraLoaderMixin,
                                       diffusers.loaders.StableDiffusionXLLoraLoaderMixin)):
        compatibles.append(diffusers.LCMScheduler)

    return compatibles


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
        parameter_schema = dict()
        schema[class_type.__name__] = parameter_schema

        def _type_name(t):
            return (str(t) if t.__module__ != 'builtins' else t.__name__).strip()

        def _resolve_union(t):
            t_name = _type_name(t)
            if t_name.startswith('typing.Union') or \
                    t_name.startswith('typing.Optional'):
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

        for parameter_name, parameter in inspect.signature(class_type.__init__).parameters.items():
            if parameter_name == 'self':
                continue

            parameter_details = dict()

            type_name = _type_name(parameter.annotation)

            if type_name.startswith('typing.Union') or \
                    type_name.startswith('typing.Optional'):
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
                           not isinstance(i, typing.Iterable) for i in parameter.default):
                    # cannot support multiple dimensions
                    continue
            if isinstance(parameter.default, numpy.ndarray):
                if parameter.default.ndim != 1:
                    # cannot support multiple dimensions
                    continue

            if parameter.default is not inspect.Parameter.empty:
                parameter_details['default'] = parameter.default

            parameter_schema[_textprocessing.dashup(parameter_name)] = parameter_details
    return schema


def load_scheduler(pipeline: diffusers.DiffusionPipeline, scheduler_uri: _types.Uri):
    """
    Load a specific compatible scheduler class name onto a huggingface diffusers pipeline object.

    :raises InvalidSchedulerNameError: If an invalid scheduler name is specified specifically.
    :raises SchedulerArgumentError: If invalid arguments are supplied to the scheduler via the URI.

    :param pipeline: pipeline object
    :param scheduler_uri: Compatible scheduler URI.
    """

    compatibles = get_compatible_schedulers(pipeline.__class__)

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
            except (ValueError, SyntaxError):
                raise SchedulerArgumentError(
                    f'{scheduler} argument "{arg_name}" '
                    f'must be a singular literal, list, '
                    f'tuple, or set value in python syntax.'
                )
        elif any(t == 'float' for t in types):
            try:
                return float(value)
            except ValueError:
                raise SchedulerArgumentError(
                    f'{scheduler} argument "{arg_name}" '
                    f'must be a floating point value.'
                )
        elif any(t == 'int' for t in types):
            try:
                return int(value)
            except ValueError:
                raise SchedulerArgumentError(
                    f'{scheduler} argument "{arg_name}" '
                    f'must be an integer value.'
                )
        elif any(t == 'bool' for t in types):
            try:
                return _types.parse_bool(value)
            except ValueError:
                raise SchedulerArgumentError(
                    f'{scheduler} argument "{arg_name}" '
                    f'must be a boolean value.'
                )

        try:
            # string literal?
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # token (string)
            return value

    for scheduler_type in compatibles:
        if scheduler_type.__name__.startswith(scheduler_uri.split(';')[0].strip()):
            schema = get_scheduler_uri_schema(scheduler_type)[scheduler_type.__name__]

            parser = _textprocessing.ConceptUriParser(
                'Scheduler',
                known_args=list(schema.keys()),
                args_raw=[k for k, v in schema.items()
                          if any(t == 'list' for t in v['types'])])

            try:
                result = parser.parse(scheduler_uri)
            except _textprocessing.ConceptUriParseError as e:
                raise SchedulerArgumentError(e)

            args = {_textprocessing.dashdown(k): _get_uri_arg_value(
                scheduler_type.__name__, v, k, schema[k]['optional'], schema[k]['types'])
                for k, v in result.args.items()}

            _messages.debug_log(
                f'Constructing Scheduler: "{scheduler_type.__name__}", URI Args: {args}')

            try:
                pipeline.scheduler = scheduler_type.from_config(pipeline.scheduler.config, **args)
            except Exception as e:
                raise SchedulerArgumentError(
                    f'Error constructing scheduler "{scheduler_type.__name__}" '
                    f'with given URI argument values, encountered error: {e}')

            _messages.debug_log(
                f'Scheduler: "{scheduler_type.__name__}", '
                f'Successfully added to pipeline: {pipeline.__class__.__name__}')

            # found a matching scheduler, return
            return

    raise InvalidSchedulerNameError(
        f'Scheduler named "{scheduler_uri}" is not a valid compatible scheduler, '
        'options are:\n\n' + '\n'.join(
            sorted(' ' * 4 + _textprocessing.quote(i.__name__.split('.')[-1]) for i in compatibles)))
