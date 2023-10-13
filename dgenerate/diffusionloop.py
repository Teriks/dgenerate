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

import datetime
import inspect
import itertools
import os
import pathlib
import random
import textwrap
import time
import typing

import PIL.PngImagePlugin

import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.preprocessors as _preprocessors
import dgenerate.prompt as _prompt
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


def iterate_attribute_combinations(
        attribute_defs: typing.List[typing.Tuple[str, typing.List]],
        my_class: typing.Type):
    """
    Iterate over every combination of attributes in a given class using a list of tuples maping
    attribute names to a list of possible values.

    :param attribute_defs: list of tuple (attribute_name, [list of values])
    :param my_class: Construct this class and assign attribute values to it
    :return: A generator over instances of the type mentioned in the my_class argument
    """

    def assign(ctx, dir_attr, name, val):
        if val is not None:
            if name in dir_attr:
                setattr(ctx, name, val)
            else:
                raise RuntimeError(f'{ctx.__class__.__name__} missing attribute "{name}"')

    for combination in itertools.product(*[d[1] for d in attribute_defs]):
        ctx_out = my_class()
        dir_attributes = set(_types.get_public_attributes(ctx_out).keys())
        for idx, d in enumerate(attribute_defs):
            attr = d[0]
            if len(d) == 2:
                assign(ctx_out, dir_attributes, attr, combination[idx])
            else:
                assign(ctx_out, dir_attributes, attr, d[2](ctx_out, attr, combination[idx]))
        yield ctx_out


def iterate_diffusion_args(prompt: _types.OptionalPrompts,
                           sdxl_second_prompt: _types.OptionalPrompts,
                           sdxl_refiner_prompt: _types.OptionalPrompts,
                           sdxl_refiner_second_prompt: _types.OptionalPrompts,
                           seed: _types.OptionalIntegers,
                           image_seed_strength: _types.OptionalFloats,
                           upscaler_noise_level: _types.OptionalIntegers,
                           sdxl_high_noise_fraction: _types.OptionalFloats,
                           sdxl_refiner_inference_steps: _types.OptionalIntegers,
                           sdxl_refiner_guidance_scale: _types.OptionalFloats,
                           sdxl_refiner_guidance_rescale: _types.OptionalFloats,
                           sdxl_aesthetic_score: _types.OptionalFloats,
                           sdxl_original_size: _types.OptionalSizes,
                           sdxl_target_size: _types.OptionalSizes,
                           sdxl_crops_coords_top_left: _types.OptionalCoordinateList,
                           sdxl_negative_aesthetic_score: _types.OptionalFloats,
                           sdxl_negative_original_size: _types.OptionalSizes,
                           sdxl_negative_target_size: _types.OptionalSizes,
                           sdxl_negative_crops_coords_top_left: _types.OptionalCoordinateList,
                           sdxl_refiner_aesthetic_score: _types.OptionalFloats,
                           sdxl_refiner_original_size: _types.OptionalSizes,
                           sdxl_refiner_target_size: _types.OptionalSizes,
                           sdxl_refiner_crops_coords_top_left: _types.OptionalCoordinateList,
                           sdxl_refiner_negative_aesthetic_score: _types.OptionalFloats,
                           sdxl_refiner_negative_original_size: _types.OptionalSizes,
                           sdxl_refiner_negative_target_size: _types.OptionalSizes,
                           sdxl_refiner_negative_crops_coords_top_left: _types.OptionalCoordinateList,
                           guidance_scale: _types.OptionalFloats,
                           image_guidance_scale: _types.OptionalFloats,
                           guidance_rescale: _types.OptionalFloats,
                           inference_steps: _types.OptionalIntegers) -> typing.Generator[
    _pipelinewrapper.DiffusionArguments, None, None]:
    """
    Iterate over every combination of possible attribute values of :py:class:`dgenerate.pipelinewrapper.DiffusionArguments` given a list of
    values for each attribute.

    :param prompt: list of :py:class:`dgenerate.prompt.Prompt` or None
    :param sdxl_second_prompt: : list of :py:class:`dgenerate.prompt.Prompt` or None
    :param sdxl_refiner_prompt: : list of :py:class:`dgenerate.prompt.Prompt` or None
    :param sdxl_refiner_second_prompt: : list of :py:class:`dgenerate.prompt.Prompt` or None
    :param seed: list of integers or None
    :param image_seed_strength: list of floats or None
    :param upscaler_noise_level: list of integers or None
    :param sdxl_high_noise_fraction: list of floats or None
    :param sdxl_refiner_inference_steps: list of integers or None
    :param sdxl_refiner_guidance_scale: list of floats or None
    :param sdxl_refiner_guidance_rescale: list of floats or None
    :param sdxl_aesthetic_score: list of floats or None
    :param sdxl_original_size: list of tuple(x, y) or None
    :param sdxl_target_size: list of tuple(x, y) or None
    :param sdxl_crops_coords_top_left: list of tuple(x, y) or None
    :param sdxl_negative_aesthetic_score: list of floats or None
    :param sdxl_negative_original_size: list of tuple(x, y) or None
    :param sdxl_negative_target_size: list of tuple(x, y) or None
    :param sdxl_negative_crops_coords_top_left: list of tuple(x, y) or None
    :param sdxl_refiner_aesthetic_score: list of floats or None
    :param sdxl_refiner_original_size: list of tuple(x, y) or None
    :param sdxl_refiner_target_size: list of tuple(x, y) or None
    :param sdxl_refiner_crops_coords_top_left: list of tuple(x, y) or None
    :param sdxl_refiner_negative_aesthetic_score: list of floats or None
    :param sdxl_refiner_negative_original_size: list of tuple(x, y) or None
    :param sdxl_refiner_negative_target_size: list of tuple(x, y) or None
    :param sdxl_refiner_negative_crops_coords_top_left: list of tuple(x, y) or None
    :param guidance_scale: list of floats or None
    :param image_guidance_scale: list of floats or None
    :param guidance_rescale: list of floats or None
    :param inference_steps: list of integers or None
    :return: a generator over :py:class:`dgenerate.pipelinewrapper.DiffusionArguments` objects
    """

    def _list_or_list_of_none(val):
        return val if val else [None]

    args = locals()
    defs = []
    for arg_name in inspect.getfullargspec(iterate_diffusion_args).args:
        defs.append((arg_name, _list_or_list_of_none(args[arg_name])))

    yield from iterate_attribute_combinations(defs, _pipelinewrapper.DiffusionArguments)


def _safe_len(lst):
    if lst is None:
        return 0
    return len(lst)


def gen_seeds(n):
    """
    Generate a list of N random seed integers

    :param n: number of seeds to generate
    :return: list of integer seeds
    """
    return [random.randint(0, 99999999999999) for _ in range(0, n)]


def _last_or_none(ls):
    if ls:
        val = ls[-1]
        if isinstance(val, str):
            val = _textprocessing.quote(val)
        return val
    return None


def _quote_string_lists(ls):
    if not ls:
        return ls
    if ls and isinstance(ls[0], str):
        return [_textprocessing.quote(i) for i in ls]
    return ls


class DiffusionRenderLoopConfigError(Exception):
    pass


class DiffusionRenderLoopConfig:
    """
    This object represents configuration for :py:class:`DiffusionRenderLoop`.

    It nearly directly maps to dgenerates command line arguments.

    See subclass :py:class:`dgenerate.arguments.DgenerateArguments`
    """

    model_path: _types.OptionalPath = None
    model_subfolder: _types.OptionalPath = None
    sdxl_refiner_path: _types.OptionalPath = None

    batch_size: _types.OptionalInteger = None
    batch_grid_size: _types.OptionalSize = None

    prompts: _types.Prompts = [_prompt.Prompt()]
    sdxl_second_prompts: _types.OptionalPrompts = None
    sdxl_refiner_prompts: _types.OptionalPrompts = None
    sdxl_refiner_second_prompts: _types.OptionalPrompts = None

    seeds: _types.Integers = gen_seeds(1)
    guidance_scales: _types.Floats = [_pipelinewrapper.DEFAULT_GUIDANCE_SCALE]
    inference_steps_values: _types.Integers = [_pipelinewrapper.DEFAULT_INFERENCE_STEPS]

    image_seeds: _types.OptionalPaths = None
    image_seed_strengths: _types.OptionalFloats = None
    upscaler_noise_levels: _types.OptionalIntegers = None
    guidance_rescales: _types.OptionalFloats = None
    image_guidance_scales: _types.OptionalFloats = None

    sdxl_high_noise_fractions: _types.OptionalFloats = None
    sdxl_refiner_inference_steps: _types.OptionalIntegers = None
    sdxl_refiner_guidance_scales: _types.OptionalFloats = None
    sdxl_refiner_guidance_rescales: _types.OptionalFloats = None

    sdxl_aesthetic_scores: _types.OptionalFloats = None
    sdxl_original_sizes: _types.OptionalSizes = None
    sdxl_target_sizes: _types.OptionalSizes = None
    sdxl_crops_coords_top_left: _types.OptionalCoordinateList = None
    sdxl_negative_aesthetic_scores: _types.OptionalFloats = None
    sdxl_negative_original_sizes: _types.OptionalSizes = None
    sdxl_negative_target_sizes: _types.OptionalSizes = None
    sdxl_negative_crops_coords_top_left: _types.OptionalCoordinateList = None

    sdxl_refiner_aesthetic_scores: _types.OptionalFloats = None
    sdxl_refiner_original_sizes: _types.OptionalSizes = None
    sdxl_refiner_target_sizes: _types.OptionalSizes = None
    sdxl_refiner_crops_coords_top_left: _types.OptionalCoordinateList = None
    sdxl_refiner_negative_aesthetic_scores: _types.OptionalFloats = None
    sdxl_refiner_negative_original_sizes: _types.OptionalSizes = None
    sdxl_refiner_negative_target_sizes: _types.OptionalSizes = None
    sdxl_refiner_negative_crops_coords_top_left: _types.OptionalCoordinateList = None

    vae_path: _types.OptionalPath = None
    vae_tiling: bool = False
    vae_slicing: bool = False

    lora_paths: _types.OptionalPaths = None
    textual_inversion_paths: _types.OptionalPaths = None
    control_net_paths: _types.OptionalPaths = None

    scheduler: _types.OptionalName = None
    sdxl_refiner_scheduler: _types.OptionalName = None
    safety_checker: bool = False
    model_type: _pipelinewrapper.ModelTypes = _pipelinewrapper.ModelTypes.TORCH
    device: _types.Name = 'cuda'
    dtype: typing.Literal['float16', 'float32', 'auto'] = 'auto'
    revision: _types.Name = 'main'
    variant: _types.OptionalName = None
    output_size: _types.OptionalSize = None
    output_path: _types.Path = os.path.join(os.getcwd(), 'output')
    output_prefix: typing.Optional[str] = None
    output_overwrite: bool = False
    output_configs: bool = False
    output_metadata: bool = False

    animation_format: _types.Name = 'mp4'
    frame_start: _types.Integer = 0
    frame_end: _types.OptionalInteger = None

    auth_token: typing.Optional[str] = None

    seed_image_preprocessors: _types.OptionalPaths = None
    mask_image_preprocessors: _types.OptionalPaths = None
    control_image_preprocessors: _types.OptionalPaths = None

    def __init__(self):
        pass

    def generate_template_variables_with_types(self, variable_prefix: typing.Optional[str] = None) \
            -> typing.Dict[str, typing.Tuple[typing.Type, typing.Any]]:
        """
        Generate a dictionary from this configuration object that maps attribute names to a tuple
        containing (type_hint_type, value)

        :param variable_prefix: Prefix every variable name with this prefix if specified
        :return: a dictionary of attribute names to tuple(type_hint_type, value)
        """

        template_variables = {}

        if variable_prefix is None:
            variable_prefix = ''

        for attr, hint in typing.get_type_hints(self.__class__).items():
            value = getattr(self, attr)
            if variable_prefix:
                prefix = variable_prefix if not attr.startswith(variable_prefix) else ''
            else:
                prefix = ''
            gen_name = prefix + attr
            if gen_name not in template_variables:
                if _types.is_type_or_optional(hint, list):
                    t_val = value if value is not None else []
                    template_variables[gen_name] = (hint, _quote_string_lists(t_val))
                else:
                    template_variables[gen_name] = (hint, value if value is not None else None)

        return template_variables

    def generate_template_variables(self,
                                    variable_prefix:
                                    typing.Optional[str] = None) -> typing.Dict[str, typing.Any]:
        """
        Generate a dictionary from this configuration object that is suitable 
        for using as Jinja2 environmental variables.
        
        :param variable_prefix: Prefix every variable name with this prefix if specified
        :return: a dictionary of attribute names to values
        """
        return {k: v[1] for k, v in
                self.generate_template_variables_with_types(variable_prefix=variable_prefix).items()}

    def set_from(self,
                 obj: typing.Union[typing.Any, dict],
                 missing_value_throws: bool = True):
        """
        Set the attributes in this configuration object from a dictionary or another object
        possessing keys / attributes of the same name.

        :param obj: The object, or dictionary
        :param missing_value_throws: whether to throw :py:class:`ValueError` if obj is missing
            an attribute that exist in this object
        :return: self
        """

        if isinstance(obj, dict):
            source = obj
        else:
            source = _types.get_public_attributes(obj)

        for k, v in _types.get_public_attributes(self):
            if not callable(v):
                if missing_value_throws and k not in source:
                    raise ValueError(f'Source object does not define: "{k}"')
                setattr(self, k, source.get(k))
        return self

    def check(self, attribute_namer: typing.Callable[[str], str] = None):
        """
        Check the configuration for type and logical usage errors, set
        defaults for certain values when needed and not specified.

        :raises: :py:class:`.DiffusionRenderLoopConfigError` on errors

        :param attribute_namer: Callable for naming attributes mentioned in exception messages
        """

        def a_namer(attr):
            if attribute_namer:
                return attribute_namer(attr)
            return f'DiffusionRenderLoopConfig.{attr}'

        def _has_len(name, value):
            try:
                len(value)
                return True
            except TypeError:
                raise DiffusionRenderLoopConfigError(
                    f'{a_namer(name)} must be able to be used with len(), value was: {value}')

        def _is_optional_two_tuple(name, value):
            if value is not None and not (isinstance(value, tuple) and len(value) == 2):
                raise DiffusionRenderLoopConfigError(
                    f'{a_namer(name)} must be None or a tuple of length 2, value was: {value}')

        def _is_optional(type, name, value):
            if value is not None and not isinstance(value, type):
                raise DiffusionRenderLoopConfigError(
                    f'{a_namer(name)} must be None or type {type.__name__}, value was: {value}')

        def _is(type, name, value):
            if not isinstance(value, type):
                raise DiffusionRenderLoopConfigError(
                    f'{a_namer(name)} must be type {type.__name__}, value was: {value}')

        # Detect incorrect types
        for attr, hint in typing.get_type_hints(self).items():
            v = getattr(self, attr)
            if _types.is_optional(hint):
                if _types.is_type_or_optional(hint, typing.Tuple[int, int]):
                    _is_optional_two_tuple(attr, v)
                else:
                    _is_optional(_types.get_type_of_optional(hint), attr, v)
            else:
                if _types.is_type(hint, list):
                    _has_len(attr, v)
                if not _types.is_type(hint, typing.Literal):
                    # Cant handle literals, like dtype
                    _is(_types.get_type(hint), attr, v)

        # Detect logically incorrect config and set certain defaults

        if self.dtype not in {'float32', 'float16', 'auto'}:
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("dtype")} must be float32, float16 or auto')
        if self.batch_size is not None and self.batch_size < 1:
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("batch_size")} must be greater than or equal to 1.')
        if self.model_type not in _pipelinewrapper.supported_model_type_enums():
            supported_model_types = _textprocessing.oxford_comma(_pipelinewrapper.supported_model_type_strings(), "or")
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("model_type")} must be one of: {supported_model_types}')
        if not _pipelinewrapper.is_valid_device_string(self.device):
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("device")} must be "cuda" (optionally with a device ordinal "cuda:N") or "cpu"')

        def attr_that_start_with(s):
            return (a for a in dir(self) if a.startswith(s) and getattr(self, a))

        def attr_that_end_with(s):
            return (a for a in dir(self) if a.endswith(s) and getattr(self, a))

        if self.model_path is None:
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("model_path")} must be specified')

        if self.frame_end is not None and self.frame_start > self.frame_end:
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("frame_start")} must be less than or equal to {a_namer("frame_end")}')

        if self.batch_size is not None:
            if _pipelinewrapper.model_type_is_flax(self.model_type):
                raise DiffusionRenderLoopConfigError(
                    f'you cannot specify {a_namer("batch_size")} when using flax, '
                    'use the environmental variable: CUDA_VISIBLE_DEVICES')
        else:
            self.batch_size = 1

        if self.output_size is None and not self.image_seeds:
            self.output_size = (512, 512) if not \
                _pipelinewrapper.model_type_is_sdxl(self.model_type) else (1024, 1024)

        if not self.image_seeds and self.image_seed_strengths:
            raise DiffusionRenderLoopConfigError(
                f'you cannot specify {a_namer("image_seed_strengths")} without {a_namer("image_seeds")}.')

        if not _pipelinewrapper.model_type_is_upscaler(self.model_type):
            if self.upscaler_noise_levels:
                raise DiffusionRenderLoopConfigError(
                    f'you cannot specify {a_namer("upscaler_noise_levels")} for a '
                    f'non upscaler model type, see: {a_namer("model_type")}.')
        elif self.control_net_paths:
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("control_net_paths")} is not compatible '
                f'with upscaler models, see: {a_namer("model_type")}.')
        elif self.upscaler_noise_levels is None:
            self.upscaler_noise_levels = [_pipelinewrapper.DEFAULT_X4_UPSCALER_NOISE_LEVEL]

        if not _pipelinewrapper.model_type_is_pix2pix(self.model_type):
            if self.image_guidance_scales:
                raise DiffusionRenderLoopConfigError(
                    f'argument {a_namer("image_guidance_scales")} only valid with '
                    f'pix2pix models, see: {a_namer("model_type")}.')
        elif self.control_net_paths:
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("control_net_paths")} is not compatible with '
                f'pix2pix models, see: {a_namer("model_type")}.')
        elif not self.image_guidance_scales:
            self.image_guidance_scales = [_pipelinewrapper.DEFAULT_IMAGE_GUIDANCE_SCALE]

        if self.control_image_preprocessors:
            if not self.image_seeds:
                raise DiffusionRenderLoopConfigError(
                    f'you cannot specify {a_namer("control_image_preprocessors")} '
                    f'without {a_namer("image_seeds")}.')

        if not self.image_seeds:
            invalid_self = []
            for preprocessor_self in attr_that_end_with('preprocessors'):
                invalid_self.append(
                    f'you cannot specify {a_namer(preprocessor_self)} '
                    f'without {a_namer("image_seeds")}.')
            if invalid_self:
                raise DiffusionRenderLoopConfigError('\n'.join(invalid_self))

        if not _pipelinewrapper.model_type_is_sdxl(self.model_type):
            invalid_self = []
            for sdxl_self in attr_that_start_with('sdxl'):
                invalid_self.append(f'you cannot specify {a_namer(sdxl_self)} '
                                    f'for a non SDXL model type, see: {a_namer("model_type")}.')
            if invalid_self:
                raise DiffusionRenderLoopConfigError('\n'.join(invalid_self))

            self.sdxl_high_noise_fractions = []
        else:
            if not self.sdxl_refiner_path:
                invalid_self = []
                for sdxl_self in attr_that_start_with('sdxl_refiner'):
                    invalid_self.append(f'you cannot specify {a_namer(sdxl_self)} '
                                        f'without {a_namer("sdxl_refiner_path")}.')
                if invalid_self:
                    raise DiffusionRenderLoopConfigError('\n'.join(invalid_self))
            else:
                if self.sdxl_high_noise_fractions is None:
                    # Default value
                    self.sdxl_high_noise_fractions = [_pipelinewrapper.DEFAULT_SDXL_HIGH_NOISE_FRACTION]

        if not _pipelinewrapper.model_type_is_torch(self.model_type):
            if self.vae_tiling or self.vae_slicing:
                raise DiffusionRenderLoopConfigError(
                    f'{a_namer("vae_tiling")}/{a_namer("vae_slicing")} not supported for '
                    f'non torch model type, see: {a_namer("model_type")}.')

        if self.scheduler == 'help' and self.sdxl_refiner_scheduler == 'help':
            raise DiffusionRenderLoopConfigError(
                'cannot list compatible schedulers for the main model and the SDXL refiner at '
                f'the same time. Do not use the scheduler "help" option for {a_namer("scheduler")} '
                f'and {a_namer("sdxl_refiner_scheduler")} simultaneously.')
        if self.image_seeds:
            if self.image_seed_strengths is None:
                # Default value
                self.image_seed_strengths = [_pipelinewrapper.DEFAULT_IMAGE_SEED_STRENGTH]
        else:
            self.image_seed_strengths = []

    def calculate_generation_steps(self):
        """
        Calculate the number of generation steps that this configuration results in.

        :return: int
        """
        optional_factors = [
            self.sdxl_second_prompts,
            self.sdxl_refiner_prompts,
            self.sdxl_refiner_second_prompts,
            self.image_guidance_scales,
            self.textual_inversion_paths,
            self.control_net_paths,
            self.image_seeds,
            self.image_seed_strengths,
            self.upscaler_noise_levels,
            self.guidance_rescales,
            self.sdxl_high_noise_fractions,
            self.sdxl_aesthetic_scores,
            self.sdxl_original_sizes,
            self.sdxl_target_sizes,
            self.sdxl_crops_coords_top_left,
            self.sdxl_negative_aesthetic_scores,
            self.sdxl_negative_original_sizes,
            self.sdxl_negative_target_sizes,
            self.sdxl_negative_crops_coords_top_left,
            self.sdxl_refiner_aesthetic_scores,
            self.sdxl_refiner_original_sizes,
            self.sdxl_refiner_target_sizes,
            self.sdxl_refiner_crops_coords_top_left,
            self.sdxl_refiner_negative_aesthetic_scores,
            self.sdxl_refiner_negative_original_sizes,
            self.sdxl_refiner_negative_target_sizes,
            self.sdxl_refiner_negative_crops_coords_top_left,
        ]

        product = 1
        for i in optional_factors:
            product *= max(_safe_len(i), 1)

        return (product *
                len(self.prompts) *
                len(self.seeds) *
                len(self.guidance_scales) *
                len(self.inference_steps_values))

    def iterate_diffusion_args(self, **overrides) -> typing.Generator[_pipelinewrapper.DiffusionArguments, None, None]:
        """
        Iterate over :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper` argument objects using
        every combination of argument values in this configuration.

        :param overrides: use key word arguments to override specific attributes of this object with a new list value.
        :return: a generator over :py:class:`dgenerate.pipelinewrapper.DiffusionArguments`
        """

        def ov(n, v):
            if not _pipelinewrapper.model_type_is_sdxl(self.model_type):
                if n.startswith('sdxl'):
                    return None
            else:
                if n.startswith('sdxl_refiner') and not self.sdxl_refiner_path:
                    return None

            if n in overrides:
                return overrides[n]
            return v

        yield from iterate_diffusion_args(
            prompt=ov('prompt', self.prompts),
            sdxl_second_prompt=ov('sdxl_second_prompt',
                                  self.sdxl_second_prompts),
            sdxl_refiner_prompt=ov('sdxl_refiner_prompt',
                                   self.sdxl_refiner_prompts),
            sdxl_refiner_second_prompt=ov('sdxl_refiner_second_prompt',
                                          self.sdxl_refiner_second_prompts),
            seed=ov('seed', self.seeds),
            image_seed_strength=ov('image_seed_strength', self.image_seed_strengths),
            guidance_scale=ov('guidance_scale', self.guidance_scales),
            image_guidance_scale=ov('image_guidance_scale', self.image_guidance_scales),
            guidance_rescale=ov('guidance_rescale', self.guidance_rescales),
            inference_steps=ov('inference_steps', self.inference_steps_values),
            sdxl_high_noise_fraction=ov('sdxl_high_noise_fraction', self.sdxl_high_noise_fractions),
            sdxl_refiner_inference_steps=ov('sdxl_refiner_inference_steps', self.sdxl_refiner_inference_steps),
            sdxl_refiner_guidance_scale=ov('sdxl_refiner_guidance_scale', self.sdxl_refiner_guidance_scales),
            sdxl_refiner_guidance_rescale=ov('sdxl_refiner_guidance_rescale',
                                             self.sdxl_refiner_guidance_rescales),
            upscaler_noise_level=ov('upscaler_noise_level', self.upscaler_noise_levels),
            sdxl_aesthetic_score=ov('sdxl_aesthetic_score', self.sdxl_aesthetic_scores),
            sdxl_original_size=ov('sdxl_original_size', self.sdxl_original_sizes),
            sdxl_target_size=ov('sdxl_target_size', self.sdxl_target_sizes),
            sdxl_crops_coords_top_left=ov('sdxl_crops_coords_top_left', self.sdxl_crops_coords_top_left),
            sdxl_negative_aesthetic_score=ov('sdxl_negative_aesthetic_score',
                                             self.sdxl_negative_aesthetic_scores),
            sdxl_negative_original_size=ov('sdxl_negative_original_size', self.sdxl_negative_original_sizes),
            sdxl_negative_target_size=ov('sdxl_negative_target_size', self.sdxl_negative_target_sizes),
            sdxl_negative_crops_coords_top_left=ov('sdxl_negative_crops_coords_top_left',
                                                   self.sdxl_negative_crops_coords_top_left),
            sdxl_refiner_aesthetic_score=ov('sdxl_refiner_aesthetic_score', self.sdxl_refiner_aesthetic_scores),
            sdxl_refiner_original_size=ov('sdxl_refiner_original_size', self.sdxl_refiner_original_sizes),
            sdxl_refiner_target_size=ov('sdxl_refiner_target_size', self.sdxl_refiner_target_sizes),
            sdxl_refiner_crops_coords_top_left=ov('sdxl_refiner_crops_coords_top_left',
                                                  self.sdxl_refiner_crops_coords_top_left),
            sdxl_refiner_negative_aesthetic_score=ov('sdxl_refiner_negative_aesthetic_score',
                                                     self.sdxl_refiner_negative_aesthetic_scores),
            sdxl_refiner_negative_original_size=ov('sdxl_refiner_negative_original_size',
                                                   self.sdxl_refiner_negative_original_sizes),
            sdxl_refiner_negative_target_size=ov('sdxl_refiner_negative_target_size',
                                                 self.sdxl_refiner_negative_target_sizes),
            sdxl_refiner_negative_crops_coords_top_left=ov('sdxl_refiner_negative_crops_coords_top_left',
                                                           self.sdxl_refiner_negative_crops_coords_top_left))


class DiffusionRenderLoop:
    """
    Render loop which implements the bulk of dgenerates rendering capability.

    This object handles the scatter gun iteration over requested diffusion parameters,
    the generation of animations, and writing images and media to disk or providing
    those to library users through callbacks.
    """

    def __init__(self, config=None, preprocessor_loader=None):
        """
        Constructor.

        :param config: :py:class:`.DiffusionRenderLoopConfig` or :py:class:`dgenerate.arguments.DgenerateArguments`
        :param preprocessor_loader: :py:class:`dgenerate.preprocessors.loader.Loader`
        """

        self._generation_step = -1
        self._frame_time_sum = 0
        self._last_frame_time = 0
        self._written_images = []
        self._written_animations = []

        self.config = \
            DiffusionRenderLoopConfig() if config is None else config

        self.preprocessor_loader = \
            _preprocessors.Loader() if preprocessor_loader is None else preprocessor_loader

    @property
    def written_images(self):
        """
        List of image filenames written by the last run
        """
        return self._written_images

    @property
    def written_animations(self):
        """
        List of animation filenames written by the last run
        """
        return self._written_animations

    def generate_template_variables_with_types(self):
        """
        Generate a dictionary from the render loop that describes its current / last used configuration with type hints.

        :return: a dictionary of attribute names to tuple(type_hint_type, value)
        """

        template_variables = self.config.generate_template_variables_with_types(
            variable_prefix='last_')

        template_variables.update({
            'last_images': (_types.Paths, _quote_string_lists(self.written_images)),
            'last_animations': (_types.Paths, _quote_string_lists(self.written_animations)),
        })

        return template_variables

    def generate_template_variables(self):
        """
        Generate a dictionary from the render loop that describes its current / last used configuration.

        This is consumed by the :py:class:`dgenerate.batchprocess.BatchProcessor`
        that is created by :py:meth:`dgenerate.batchprocess.create_config_runner` for
        use in Jinja2 templating.

        :return: a dictionary of attribute names to values
        """
        return {k: v[1] for k, v in self.generate_template_variables_with_types().items()}

    def generate_template_variables_help(self):
        """
        Generate a help string describing available template variables, their types, and values
        for use in batch processing.

        This is used to implement --templates-help in :py:meth:`dgenerate.invoker.invoke_dgenerate`

        :return: A human-readable description of all template variables
        """

        help_string = _textprocessing.underline(
            'Available post invocation template variables are:') + '\n\n'

        def wrap(val):
            return textwrap.fill(str(val),
                                 width=_textprocessing.long_text_wrap_width(),
                                 break_long_words=False,
                                 break_on_hyphens=False,
                                 subsequent_indent=' ' * 17)

        return help_string + '\n'.join(
            ' ' * 4 + f'Name: {_textprocessing.quote(i[0])}\n{" " * 8}'
                      f'Type: {i[1][0]}\n{" " * 8}Value: {wrap(i[1][1])}' for i in
            self.generate_template_variables_with_types().items())

    @property
    def generation_step(self):
        """
        Returns the current generation step
        """
        return self._generation_step

    def _gen_filename(self, *args, ext):
        def _make_path(dup_number=None):
            return os.path.join(self.config.output_path,
                                f'{self.config.output_prefix + "_" if self.config.output_prefix is not None else ""}' + '_'.
                                join(str(s).replace('.', '-') for s in args) + (
                                    '' if dup_number is None else f'_duplicate_{dup_number}') + '.' + ext)

        path = _make_path()

        if self.config.output_overwrite:
            return path

        if not os.path.exists(path):
            return path

        duplicate_number = 1
        while os.path.exists(path):
            path = _make_path(duplicate_number)
            duplicate_number += 1

        return path

    @staticmethod
    def _gen_filename_base(args_ctx: _pipelinewrapper.DiffusionArguments):
        args = ['s', args_ctx.seed]

        if args_ctx.upscaler_noise_level is not None:
            args += ['unl', args_ctx.upscaler_noise_level]
        elif args_ctx.image_seed_strength is not None:
            args += ['st', args_ctx.image_seed_strength]

        args += ['g', args_ctx.guidance_scale]

        if args_ctx.guidance_rescale is not None:
            args += ['gr', args_ctx.guidance_rescale]

        if args_ctx.image_guidance_scale is not None:
            args += ['igs', args_ctx.image_guidance_scale]

        args += ['i', args_ctx.inference_steps]

        if args_ctx.sdxl_high_noise_fraction is not None:
            args += ['hnf', args_ctx.sdxl_high_noise_fraction]

        if args_ctx.sdxl_refiner_guidance_scale is not None:
            args += ['rg', args_ctx.sdxl_refiner_guidance_scale]

        if args_ctx.sdxl_refiner_guidance_rescale is not None:
            args += ['rgr', args_ctx.sdxl_refiner_guidance_rescale]

        if args_ctx.sdxl_refiner_inference_steps is not None:
            args += ['ri', args_ctx.sdxl_refiner_inference_steps]

        return args

    def _gen_animation_filename(self,
                                args_ctx: _pipelinewrapper.DiffusionArguments,
                                generation_step,
                                animation_format):
        args = ['ANIM', *self._gen_filename_base(args_ctx)]

        return self._gen_filename(*args, 'step', generation_step + 1, ext=animation_format)

    def _write_image(self, filename, image, batch_index, generation_result):

        extra_args = []

        if self.config.batch_size > 1 and not \
                _pipelinewrapper.model_type_is_flax(self.config.model_type):
            # Batch size is controlled by CUDA_VISIBLE_DEVICES for flax
            extra_args.append(('--batch-size', self.config.batch_size))

        if self.config.batch_grid_size is not None:
            extra_args.append(('--batch-grid-size', self.config.batch_grid_size))

        config_txt = generation_result.gen_dgenerate_config(extra_args=extra_args)

        is_last_image = batch_index == generation_result.image_count - 1

        if self.config.output_metadata:
            metadata = PIL.PngImagePlugin.PngInfo()
            metadata.add_text("DgenerateConfig", config_txt)
            image.save(filename, pnginfo=metadata)
        else:
            image.save(filename)
        if self.config.output_configs:
            config_file_name = os.path.splitext(filename)[0] + '.txt'
            with open(config_file_name, "w") as config_file:
                config_file.write(config_txt)
            _messages.log(
                f'Wrote Image File: "{filename}"\n'
                f'Wrote Config File: "{config_file_name}"',
                underline=is_last_image)
        else:
            _messages.log(f'Wrote Image File: "{filename}"',
                          underline=is_last_image)

        self._written_images.append(os.path.abspath(filename))

    def _write_generation_result(self,
                                 filename,
                                 generation_result: _pipelinewrapper.PipelineWrapperResult):

        if self.config.batch_grid_size is None:

            for batch_idx, image in enumerate(generation_result.images):
                file_name, ext = os.path.splitext(filename)

                file_name = file_name + f'_image_{batch_idx}' + ext

                self._write_image(file_name, image, batch_idx, generation_result)
        else:
            if generation_result.image_count > 1:
                image = generation_result.image_grid(self.config.batch_grid_size)
            else:
                image = generation_result.image

            self._write_image(filename, image, 0, generation_result)

    def _write_animation_frame(self,
                               args_ctx: _pipelinewrapper.DiffusionArguments,
                               image_seed_obj: _mediainput.ImageSeed,
                               generation_result: _pipelinewrapper.PipelineWrapperResult):
        args = self._gen_filename_base(args_ctx)

        filename = self._gen_filename(*args,
                                      'frame',
                                      image_seed_obj.frame_index + 1,
                                      'step',
                                      self._generation_step + 1,
                                      ext='png')

        generation_result.add_dgenerate_opt('--frame-start', image_seed_obj.frame_index)
        generation_result.add_dgenerate_opt('--frame-end', image_seed_obj.frame_index)
        self._write_generation_result(filename, generation_result)

    def _write_image_seed_gen_image(self, args_ctx: _pipelinewrapper.DiffusionArguments,
                                    generation_result: _pipelinewrapper.PipelineWrapperResult):
        args = self._gen_filename_base(args_ctx)
        filename = self._gen_filename(*args, 'step', self._generation_step + 1, ext='png')
        self._write_generation_result(filename, generation_result)

    def _write_prompt_only_image(self, args_ctx: _pipelinewrapper.DiffusionArguments,
                                 generation_result: _pipelinewrapper.PipelineWrapperResult):
        args = self._gen_filename_base(args_ctx)
        filename = self._gen_filename(*args, 'step', self._generation_step + 1, ext='png')
        self._write_generation_result(filename, generation_result)

    def _pre_generation_step(self, args_ctx: _pipelinewrapper.DiffusionArguments):
        self._last_frame_time = 0
        self._frame_time_sum = 0
        self._generation_step += 1

        desc = args_ctx.describe_pipeline_wrapper_args()

        _messages.log(
            f'Generation step {self._generation_step + 1} / {self.config.calculate_generation_steps()}\n'
            + desc, underline=True)

    def _pre_generation(self, args_ctx):
        pass

    def _animation_frame_pre_generation(self, args_ctx: _pipelinewrapper.DiffusionArguments, image_seed_obj):
        if self._last_frame_time == 0:
            eta = 'tbd...'
        else:
            self._frame_time_sum += time.time() - self._last_frame_time
            eta_seconds = (self._frame_time_sum / image_seed_obj.frame_index) * (
                    image_seed_obj.total_frames - image_seed_obj.frame_index)
            eta = str(datetime.timedelta(seconds=eta_seconds))

        self._last_frame_time = time.time()
        _messages.log(
            f'Generating frame {image_seed_obj.frame_index + 1} / {image_seed_obj.total_frames}, Completion ETA: {eta}',
            underline=True)

    def _with_image_seed_pre_generation(self, args_ctx: _pipelinewrapper.DiffusionArguments, image_seed_obj):
        pass

    def _load_preprocessors(self, preprocessors):
        return self.preprocessor_loader.load(preprocessors, self.config.device)

    def run(self):
        """
        Run the diffusion loop, this calls :py:meth:`.DiffusionRenderLoopConfig.check` prior to running.
        """
        try:
            self._run()
        except _pipelinewrapper.SchedulerHelpException:
            pass

    def _run(self):
        self.config.check()

        pathlib.Path(self.config.output_path).mkdir(parents=True, exist_ok=True)

        self._written_images = []
        self._written_animations = []
        self._generation_step = -1
        self._frame_time_sum = 0
        self._last_frame_time = 0

        generation_steps = self.config.calculate_generation_steps()

        if generation_steps == 0:
            _messages.log(f'Options resulted in no generation steps, nothing to do.', underline=True)
            return

        _messages.log(f'Beginning {generation_steps} generation steps...', underline=True)

        if self.config.image_seeds:
            self._render_with_image_seeds()
        else:
            diffusion_model = _pipelinewrapper.DiffusionPipelineWrapper(self.config.model_path,
                                                                        model_subfolder=self.config.model_subfolder,
                                                                        dtype=self.config.dtype,
                                                                        device=self.config.device,
                                                                        model_type=self.config.model_type,
                                                                        revision=self.config.revision,
                                                                        variant=self.config.variant,
                                                                        vae_path=self.config.vae_path,
                                                                        lora_paths=self.config.lora_paths,
                                                                        textual_inversion_paths=self.config.textual_inversion_paths,
                                                                        scheduler=self.config.scheduler,
                                                                        sdxl_refiner_scheduler=self.config.sdxl_refiner_scheduler,
                                                                        safety_checker=self.config.safety_checker,
                                                                        sdxl_refiner_path=self.config.sdxl_refiner_path,
                                                                        auth_token=self.config.auth_token)

            sdxl_high_noise_fractions = self.config.sdxl_high_noise_fractions if self.config.sdxl_refiner_path is not None else None

            for args_ctx in self.config.iterate_diffusion_args(sdxl_high_noise_fraction=sdxl_high_noise_fractions,
                                                               image_seed_strength=None,
                                                               upscaler_noise_level=None):
                self._pre_generation_step(args_ctx)
                self._pre_generation(args_ctx)

                with diffusion_model(**args_ctx.get_pipeline_wrapper_args(),
                                     width=self.config.output_size[0],
                                     height=self.config.output_size[1],
                                     batch_size=self.config.batch_size) as generation_result:
                    self._write_prompt_only_image(args_ctx, generation_result)

    def _render_with_image_seeds(self):
        diffusion_model = _pipelinewrapper.DiffusionPipelineWrapper(self.config.model_path,
                                                                    model_subfolder=self.config.model_subfolder,
                                                                    dtype=self.config.dtype,
                                                                    device=self.config.device,
                                                                    model_type=self.config.model_type,
                                                                    revision=self.config.revision,
                                                                    variant=self.config.variant,
                                                                    vae_path=self.config.vae_path,
                                                                    vae_tiling=self.config.vae_tiling,
                                                                    vae_slicing=self.config.vae_slicing,
                                                                    lora_paths=self.config.lora_paths,
                                                                    textual_inversion_paths=self.config.textual_inversion_paths,
                                                                    control_net_paths=self.config.control_net_paths,
                                                                    scheduler=self.config.scheduler,
                                                                    safety_checker=self.config.safety_checker,
                                                                    sdxl_refiner_path=self.config.sdxl_refiner_path,
                                                                    auth_token=self.config.auth_token)

        sdxl_high_noise_fractions = self.config.sdxl_high_noise_fractions if self.config.sdxl_refiner_path is not None else None

        image_seed_strengths = self.config.image_seed_strengths if \
            not (_pipelinewrapper.model_type_is_upscaler(self.config.model_type) or
                 _pipelinewrapper.model_type_is_pix2pix(self.config.model_type)) else None

        upscaler_noise_levels = self.config.upscaler_noise_levels if \
            self.config.model_type == _pipelinewrapper.ModelTypes.TORCH_UPSCALER_X4 else None

        def validate_image_seeds():
            for img_seed in self.config.image_seeds:
                parsed = _mediainput.parse_image_seed_uri(img_seed)

                if self.config.control_net_paths and not parsed.is_single_image() and parsed.control_uri is None:
                    raise NotImplementedError(
                        f'You must specify a control image with the control argument '
                        f'IE: --image-seeds "my-seed.png;control=my-control.png" in your '
                        f'--image-seeds "{img_seed}" when using --control-nets in order '
                        f'to use inpainting. If you want to use the control image alone '
                        f'without a mask, use --image-seeds "{parsed.uri}".')

                yield img_seed, parsed

        for image_seed, parsed_image_seed in list(validate_image_seeds()):

            is_single_control_image = self.config.control_net_paths and parsed_image_seed.is_single_image()
            image_seed_strengths = image_seed_strengths if not is_single_control_image else None
            upscaler_noise_levels = upscaler_noise_levels if not is_single_control_image else None

            if is_single_control_image:
                _messages.log(f'Processing Control Image: "{image_seed}"', underline=True)
            else:
                _messages.log(f'Processing Image Seed: "{image_seed}"', underline=True)

            arg_iterator = self.config.iterate_diffusion_args(
                sdxl_high_noise_fraction=sdxl_high_noise_fractions,
                image_seed_strength=image_seed_strengths,
                upscaler_noise_level=upscaler_noise_levels
            )

            if is_single_control_image:
                seed_info = _mediainput.get_image_seed_info(
                    parsed_image_seed, self.config.frame_start, self.config.frame_end)
            else:
                seed_info = _mediainput.get_control_image_info(
                    parsed_image_seed, self.config.frame_start, self.config.frame_end)

            if is_single_control_image:
                def seed_iterator_func():
                    yield from _mediainput.iterate_control_image(
                        parsed_image_seed,
                        self.config.frame_start,
                        self.config.frame_end,
                        self.config.output_size,
                        preprocessor=self._load_preprocessors(self.config.control_image_preprocessors))

            else:
                def seed_iterator_func():
                    yield from _mediainput.iterate_image_seed(
                        parsed_image_seed,
                        self.config.frame_start,
                        self.config.frame_end,
                        self.config.output_size,
                        seed_image_preprocessor=self._load_preprocessors(self.config.seed_image_preprocessors),
                        mask_image_preprocessor=self._load_preprocessors(self.config.mask_image_preprocessors),
                        control_image_preprocessor=self._load_preprocessors(self.config.control_image_preprocessors))

            if seed_info.is_animation:

                if is_single_control_image:
                    def get_extra_args(ci_obj: _mediainput.ImageSeed):
                        return {'control_image': ci_obj.image}
                else:
                    def get_extra_args(ims_obj: _mediainput.ImageSeed):
                        extra_args = {'image': ims_obj.image}
                        if ims_obj.mask_image is not None:
                            extra_args['mask_image'] = ims_obj.mask_image
                        if ims_obj.control_image is not None:
                            extra_args['control_image'] = ims_obj.control_image
                        return extra_args

                self._render_animation(diffusion_model,
                                       arg_iterator,
                                       seed_info.fps,
                                       seed_iterator_func=seed_iterator_func,
                                       get_extra_args=get_extra_args)
                break

            for args_ctx in arg_iterator:
                self._pre_generation_step(args_ctx)
                with next(seed_iterator_func()) as image_obj:
                    with image_obj as image_seed_obj:
                        self._with_image_seed_pre_generation(args_ctx, image_seed_obj)

                        pipeline_args = args_ctx.get_pipeline_wrapper_args()

                        if not is_single_control_image:
                            pipeline_args['image'] = image_seed_obj.image

                        if image_seed_obj.mask_image is not None:
                            pipeline_args['mask_image'] = image_seed_obj.mask_image
                        else:
                            pipeline_args['control_image'] = (image_seed_obj.image if is_single_control_image
                                                              else image_seed_obj.control_image)

                        with _mediainput.MultiContextManager(
                                [image_seed_obj.mask_image, image_seed_obj.control_image]), \
                                diffusion_model(**pipeline_args,
                                                batch_size=self.config.batch_size) as generation_result:

                            self._write_image_seed_gen_image(args_ctx, generation_result)

    def _render_animation(self, diffusion_model, arg_iterator, fps, seed_iterator_func, get_extra_args):

        animation_format_lower = self.config.animation_format.lower()
        first_args_ctx = next(arg_iterator)

        out_filename = self._gen_animation_filename(first_args_ctx, self._generation_step + 1,
                                                    animation_format_lower)
        next_frame_terminates_anim = False

        with _mediaoutput.create_animation_writer(animation_format_lower, out_filename, fps) as video_writer:

            for args_ctx in itertools.chain([first_args_ctx], arg_iterator):
                self._pre_generation_step(args_ctx)

                if next_frame_terminates_anim:
                    next_frame_terminates_anim = False
                    video_writer.end(
                        new_file=self._gen_animation_filename(args_ctx, self._generation_step,
                                                              animation_format_lower))

                if self.config.output_configs:
                    anim_config_file_name = os.path.splitext(video_writer.filename)[0] + '.txt'
                    _messages.log(
                        f'Writing Animation: "{video_writer.filename}"\nWriting Config File: "{anim_config_file_name}"',
                        underline=True)
                else:
                    _messages.log(f'Writing Animation: "{video_writer.filename}"', underline=True)

                self._written_animations.append(os.path.abspath(video_writer.filename))

                for image_obj in seed_iterator_func():

                    with image_obj as image_seed_obj:
                        self._animation_frame_pre_generation(args_ctx, image_seed_obj)

                        extra_args = get_extra_args(image_seed_obj)

                        with diffusion_model(**args_ctx.get_pipeline_wrapper_args(),
                                             **extra_args,
                                             batch_size=self.config.batch_size) as generation_result:

                            if generation_result.image_count > 1 and self.config.batch_grid_size is not None:
                                video_writer.write(
                                    generation_result.image_grid(self.config.batch_grid_size))
                            else:
                                video_writer.write(generation_result.image)

                            if self.config.output_configs:
                                if not os.path.exists(anim_config_file_name):

                                    if self.config.frame_start is not None:
                                        generation_result.add_dgenerate_opt('--frame-start',
                                                                            self.config.frame_start)

                                    if self.config.frame_end is not None:
                                        generation_result.add_dgenerate_opt('--frame-end',
                                                                            self.config.frame_end)

                                    if self.config.animation_format is not None:
                                        generation_result.add_dgenerate_opt('--animation-format',
                                                                            self.config.animation_format)

                                    config_text = \
                                        generation_result.gen_dgenerate_config()

                                    with open(anim_config_file_name, "w") as config_file:
                                        config_file.write(config_text)

                            self._write_animation_frame(args_ctx, image_seed_obj, generation_result)

                        next_frame_terminates_anim = image_seed_obj.frame_index == (image_seed_obj.total_frames - 1)

                video_writer.end()
