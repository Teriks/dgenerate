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
import time
import typing

import PIL.Image
import PIL.PngImagePlugin

import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.preprocessors as _preprocessors
import dgenerate.prompt as _prompt
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.util as _util


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


def gen_seeds(n):
    """
    Generate a list of N random seed integers

    :param n: number of seeds to generate
    :return: list of integer seeds
    """
    return [random.randint(0, 99999999999999) for _ in range(0, n)]


class DiffusionRenderLoopConfigError(Exception):
    """
    Raised by :py:meth:`.DiffusionRenderLoopConfig.check` on configuration errors.
    """
    pass


class DiffusionRenderLoopConfig(_types.SetFromMixin):
    """
    This object represents configuration for :py:class:`DiffusionRenderLoop`.

    It nearly directly maps to dgenerates command line arguments.

    See subclass :py:class:`dgenerate.arguments.DgenerateArguments`
    """

    model_path: _types.OptionalPath = None
    model_subfolder: _types.OptionalPath = None
    sdxl_refiner_uri: _types.OptionalUri = None

    batch_size: _types.OptionalInteger = None
    batch_grid_size: _types.OptionalSize = None

    prompts: _types.Prompts = [_prompt.Prompt()]
    sdxl_second_prompts: _types.OptionalPrompts = None
    sdxl_refiner_prompts: _types.OptionalPrompts = None
    sdxl_refiner_second_prompts: _types.OptionalPrompts = None

    seeds: _types.Integers = gen_seeds(1)
    guidance_scales: _types.Floats = [_pipelinewrapper.DEFAULT_GUIDANCE_SCALE]
    inference_steps_values: _types.Integers = [_pipelinewrapper.DEFAULT_INFERENCE_STEPS]

    image_seeds: _types.OptionalUris = None
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

    vae_uri: _types.OptionalUri = None
    vae_tiling: bool = False
    vae_slicing: bool = False

    lora_uris: _types.OptionalUris = None
    textual_inversion_uris: _types.OptionalUris = None
    control_net_uris: _types.OptionalUris = None

    scheduler: _types.OptionalName = None
    sdxl_refiner_scheduler: _types.OptionalName = None
    safety_checker: bool = False
    model_type: _pipelinewrapper.ModelTypes = _pipelinewrapper.ModelTypes.TORCH
    device: _types.Name = 'cuda'
    dtype: _pipelinewrapper.DataTypes = _pipelinewrapper.DataTypes.AUTO
    revision: _types.Name = 'main'
    variant: _types.OptionalName = None
    output_size: _types.OptionalSize = None
    output_path: _types.Path = 'output'
    output_prefix: typing.Optional[str] = None
    output_overwrite: bool = False
    output_configs: bool = False
    output_metadata: bool = False

    animation_format: _types.Name = 'mp4'
    frame_start: _types.Integer = 0
    frame_end: _types.OptionalInteger = None

    auth_token: typing.Optional[str] = None

    seed_image_preprocessors: _types.OptionalUris = None
    mask_image_preprocessors: _types.OptionalUris = None
    control_image_preprocessors: _types.OptionalUris = None

    offline_mode: bool = False

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
                    template_variables[gen_name] = (hint, t_val)
                else:
                    template_variables[gen_name] = (hint, value)

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

    def check(self, attribute_namer: typing.Callable[[str], str] = None):
        """
        Check the configuration for type and logical usage errors, set
        defaults for certain values when needed and not specified.

        :raises: :py:class:`.DiffusionRenderLoopConfigError` on errors

        :param attribute_namer: Callable for naming attributes mentioned in exception messages
        """

        def a_namer(attr_name):
            if attribute_namer:
                return attribute_namer(attr_name)
            return f'DiffusionRenderLoopConfig.{attr_name}'

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
        supported_dtypes = _pipelinewrapper.supported_data_type_strings()
        if self.dtype not in _pipelinewrapper.supported_data_type_enums():
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("dtype")} must be {_textprocessing.oxford_comma(supported_dtypes, "or")}')
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

        if not self.image_seeds and self.control_net_uris:
            raise DiffusionRenderLoopConfigError(
                f'you cannot specify {a_namer("control_net_uris")} without {a_namer("image_seeds")}.')

        if not _pipelinewrapper.model_type_is_upscaler(self.model_type):
            if self.upscaler_noise_levels:
                raise DiffusionRenderLoopConfigError(
                    f'you cannot specify {a_namer("upscaler_noise_levels")} for a '
                    f'non upscaler model type, see: {a_namer("model_type")}.')
        elif self.control_net_uris:
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("control_net_uris")} is not compatible '
                f'with upscaler models, see: {a_namer("model_type")}.')
        elif self.upscaler_noise_levels is None:
            self.upscaler_noise_levels = [_pipelinewrapper.DEFAULT_X4_UPSCALER_NOISE_LEVEL]

        if not _pipelinewrapper.model_type_is_pix2pix(self.model_type):
            if self.image_guidance_scales:
                raise DiffusionRenderLoopConfigError(
                    f'argument {a_namer("image_guidance_scales")} only valid with '
                    f'pix2pix models, see: {a_namer("model_type")}.')
        elif self.control_net_uris:
            raise DiffusionRenderLoopConfigError(
                f'{a_namer("control_net_uris")} is not compatible with '
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
            if not self.sdxl_refiner_uri:
                invalid_self = []
                for sdxl_self in attr_that_start_with('sdxl_refiner'):
                    invalid_self.append(f'you cannot specify {a_namer(sdxl_self)} '
                                        f'without {a_namer("sdxl_refiner_uri")}.')
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
            self.textual_inversion_uris,
            self.control_net_uris,
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
        for lst in optional_factors:
            product *= max(0 if lst is None else len(lst), 1)

        return (product *
                len(self.prompts) *
                len(self.seeds) *
                len(self.guidance_scales) *
                len(self.inference_steps_values))

    def iterate_diffusion_args(self, **overrides) -> typing.Generator[_pipelinewrapper.DiffusionArguments, None, None]:
        """
        Iterate over :py:class:`dgenerate.pipelinewrapper.DiffusionArguments` argument objects using
        every combination of argument values provided for that object by this configuration.

        :param overrides: use key word arguments to override specific attributes of this object with a new list value.
        :return: a generator over :py:class:`dgenerate.pipelinewrapper.DiffusionArguments`
        """

        def ov(n, v):
            if not _pipelinewrapper.model_type_is_sdxl(self.model_type):
                if n.startswith('sdxl'):
                    return None
            else:
                if n.startswith('sdxl_refiner') and not self.sdxl_refiner_uri:
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


class ImageGeneratedCallbackArgument(_types.SetFromMixin):
    """
    This argument object gets passed to callbacks registered to
    :py:class:`.DiffusionRenderLoop.image_generated_callbacks`.
    """

    image: PIL.Image.Image = None
    """
    The generated image.
    """

    generation_step: int = 0
    """
    The current generation step. (zero indexed)
    """

    batch_index: int = 0
    """
    The index in the image batch for this image. Will only every be greater than zero if 
    :py:attr:`.DiffusionRenderLoopConfig.batch_size` > 1 and :py:attr:`.DiffusionRenderLoopConfig.batch_grid_size` is None.
    """

    suggested_filename: str = None
    """
    A suggested filename for saving this image as. This filename will be unique
    to the render loop run and configuration.
    """

    args: _pipelinewrapper.DiffusionArguments = None
    """
    Diffusion argument object, contains :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper` 
    arguments used to produce this image.
    """

    command: str = None
    """
    Reproduction of a command line that can be used to reproduce this image.
    """

    config: str = None
    """
    Reproduction of a dgenerate config file that can be used to reproduce this image.
    """

    is_frame: bool = False
    """
    Is this image a frame in an animation?
    """

    frame_index: int = 0
    """
    The frame index if this is an animation frame.
    Also available through image_seed.frame_index, 
    though here for convenience.
    """

    image_seed: typing.Optional[_mediainput.ImageSeed] = None
    """
    If an --image-seed specification was used in the generation of this image,
    this object represents that image seed and contains the images that contributed
    to the generation of this image.
    """


ImageGeneratedCallbacks = typing.List[
    typing.Callable[[ImageGeneratedCallbackArgument], None]]


class DiffusionRenderLoop:
    """
    Render loop which implements the bulk of dgenerates rendering capability.

    This object handles the scatter gun iteration over requested diffusion parameters,
    the generation of animations, and writing images and media to disk or providing
    those to library users through callbacks.
    """

    disable_writes: bool = False
    """
    Disable or enable all writes to disk, if you intend to only ever use the callbacks of the
    render loop when using dgenerate as a library, this is a useful option.
    
    last_images and last_animations will not be available as template variables in
    batch processing scripts with this enabled, they will be empty lists
    """

    image_generated_callbacks: ImageGeneratedCallbacks = []
    """
    Optional callbacks for handling individual images that have been generated.
    
    The callback has a single argument: :py:class:`.ImageGeneratedCallbackArgument`
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
            'last_images': (_types.Paths, self.written_images),
            'last_animations': (_types.Paths, self.written_animations),
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

        :return: a human-readable description of all template variables
        """

        help_string = _textprocessing.underline(
            'Available post invocation template variables are:') + '\n\n'

        def wrap(val):
            return _textprocessing.wrap(
                str(val),
                width=_textprocessing.long_text_wrap_width(),
                subsequent_indent=' ' * 17)

        return help_string + '\n'.join(
            ' ' * 4 + f'Name: {_textprocessing.quote(i[0])}\n{" " * 8}'
                      f'Type: {i[1][0]}\n{" " * 8}Value: {wrap(i[1][1])}' for i in
            self.generate_template_variables_with_types().items())

    @property
    def generation_step(self):
        """
        Returns the current generation step, (zero indexed)
        """
        return self._generation_step

    def _join_output_filename(self, components, ext):

        prefix = self.config.output_prefix + '_' if \
            self.config.output_prefix is not None else ''

        components = (str(s).replace('.', '-') for s in components)

        return os.path.join(self.config.output_path,
                            f'{prefix}' + '_'.join(components)) + '.' + ext.lstrip('.')

    @staticmethod
    def _gen_filename_components_base(args_ctx: _pipelinewrapper.DiffusionArguments):
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

    def _write_image(self,
                     filename_components: typing.List[str],
                     image: PIL.Image.Image,
                     batch_index: int,
                     generation_result: _pipelinewrapper.PipelineWrapperResult,
                     callback_argument_extras: dict):

        extra_args = []
        extra_comments = []

        config_txt = None

        # Generate a reconstruction of dgenerates arguments
        # For this image if necessary

        if self.image_generated_callbacks or self.config.output_configs or self.config.output_metadata:
            if generation_result.image_count > 1:
                if not _pipelinewrapper.model_type_is_flax(self.config.model_type):
                    # Batch size is controlled by CUDA_VISIBLE_DEVICES for flax
                    extra_args.append(('--batch-size', self.config.batch_size))

                if self.config.batch_grid_size is not None:
                    extra_args.append(('--batch-grid-size',
                                       _textprocessing.format_size(self.config.batch_grid_size)))
                else:
                    extra_comments.append(
                        f'Image {batch_index + 1} from a batch of {generation_result.image_count}')

            config_txt = generation_result.gen_dgenerate_config(
                extra_args=extra_args,
                extra_comments=extra_comments)

        if self.image_generated_callbacks:
            argument = ImageGeneratedCallbackArgument()

            argument.set_from(callback_argument_extras,
                              missing_value_throws=False)

            argument.generation_step = self.generation_step

            argument.image = image
            argument.batch_index = batch_index
            argument.suggested_filename = self._join_output_filename(filename_components, ext='png')
            argument.config = config_txt
            argument.command = generation_result.gen_dgenerate_command(extra_args=extra_args)

            for callback in self.image_generated_callbacks:
                callback(argument)

        if self.disable_writes:
            return

        config_filename = None

        # Generate and touch filenames avoiding duplicates in a way
        # that is multiprocess safe between instances of dgenerate
        if self.config.output_configs:
            image_filename, config_filename = \
                _util.touch_avoid_duplicate(
                    self.config.output_path,
                    pathmaker=_util.suffix_path_maker(
                        [self._join_output_filename(filename_components, ext='png'),
                         self._join_output_filename(filename_components, ext='txt')],
                        suffix='_duplicate_'))
        else:
            image_filename = _util.touch_avoid_duplicate(
                self.config.output_path,
                pathmaker=_util.suffix_path_maker(
                    self._join_output_filename(filename_components, ext='png'),
                    suffix='_duplicate_'))

        # Write out to the empty files

        if self.config.output_metadata:
            metadata = PIL.PngImagePlugin.PngInfo()
            metadata.add_text("DgenerateConfig", config_txt)
            image.save(image_filename, pnginfo=metadata)
        else:
            image.save(image_filename)

        is_last_image = batch_index == generation_result.image_count - 1
        # Only underline the last image write message in a batch of rendered
        # images when --batch-size > 1

        if self.config.output_configs:
            with open(config_filename, "w") as config_file:
                config_file.write(config_txt)
            _messages.log(
                f'Wrote Image File: "{image_filename}"\n'
                f'Wrote Config File: "{config_filename}"',
                underline=is_last_image)
        else:
            _messages.log(f'Wrote Image File: "{image_filename}"',
                          underline=is_last_image)

        # Append to written images for the current run
        self._written_images.append(os.path.abspath(image_filename))

    def _write_generation_result(self,
                                 filename_components: typing.List[str],
                                 generation_result: _pipelinewrapper.PipelineWrapperResult,
                                 callback_argument_extras: dict):

        if self.config.batch_grid_size is None:

            for batch_idx, image in enumerate(generation_result.images):
                name_components = filename_components.copy()
                if generation_result.image_count > 1:
                    name_components += ['image', batch_idx + 1]

                self._write_image(name_components,
                                  image,
                                  batch_idx,
                                  generation_result,
                                  callback_argument_extras)
        else:
            if generation_result.image_count > 1:
                image = generation_result.image_grid(self.config.batch_grid_size)
            else:
                image = generation_result.image

            self._write_image(filename_components,
                              image, 0,
                              generation_result,
                              callback_argument_extras)

    def _write_animation_frame(self,
                               args_ctx: _pipelinewrapper.DiffusionArguments,
                               image_seed_obj: _mediainput.ImageSeed,
                               generation_result: _pipelinewrapper.PipelineWrapperResult):
        filename_components = [*self._gen_filename_components_base(args_ctx),
                               'frame',
                               image_seed_obj.frame_index + 1,
                               'step',
                               self._generation_step + 1]

        generation_result.add_dgenerate_opt('--frame-start', image_seed_obj.frame_index)
        generation_result.add_dgenerate_opt('--frame-end', image_seed_obj.frame_index)

        self._write_generation_result(filename_components, generation_result,
                                      {'args': args_ctx,
                                       'is_frame': True,
                                       'frame_index': image_seed_obj.frame_index,
                                       'image_seed': image_seed_obj})

    def _write_image_seed_gen_image(self, args_ctx: _pipelinewrapper.DiffusionArguments,
                                    image_seed_obj: _mediainput.ImageSeed,
                                    generation_result: _pipelinewrapper.PipelineWrapperResult):
        filename_components = [*self._gen_filename_components_base(args_ctx),
                               'step',
                               self._generation_step + 1]

        self._write_generation_result(filename_components, generation_result,
                                      {'args': args_ctx,
                                       'image_seed': image_seed_obj})

    def _write_prompt_only_image(self, args_ctx: _pipelinewrapper.DiffusionArguments,
                                 generation_result: _pipelinewrapper.PipelineWrapperResult):
        filename_components = [*self._gen_filename_components_base(args_ctx),
                               'step',
                               self._generation_step + 1]

        self._write_generation_result(filename_components, generation_result,
                                      {'args': args_ctx})

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

    def _animation_frame_pre_generation(self,
                                        args_ctx: _pipelinewrapper.DiffusionArguments,
                                        image_seed: _mediainput.ImageSeed):
        if self._last_frame_time == 0:
            eta = 'tbd...'
        else:
            self._frame_time_sum += time.time() - self._last_frame_time
            eta_seconds = (self._frame_time_sum / image_seed.frame_index) * (
                    image_seed.total_frames - image_seed.frame_index)
            eta = str(datetime.timedelta(seconds=eta_seconds))

        self._last_frame_time = time.time()
        _messages.log(
            f'Generating frame {image_seed.frame_index + 1} / {image_seed.total_frames}, Completion ETA: {eta}',
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

    def _create_pipeline_wrapper(self):
        return _pipelinewrapper.DiffusionPipelineWrapper(
            self.config.model_path,
            dtype=self.config.dtype,
            device=self.config.device,
            model_type=self.config.model_type,
            revision=self.config.revision,
            variant=self.config.variant,
            model_subfolder=self.config.model_subfolder,
            vae_uri=self.config.vae_uri,
            vae_tiling=self.config.vae_tiling,
            vae_slicing=self.config.vae_slicing,
            lora_uris=self.config.lora_uris,
            textual_inversion_uris=self.config.textual_inversion_uris,
            control_net_uris=
            self.config.control_net_uris if self.config.image_seeds else [],
            sdxl_refiner_uri=self.config.sdxl_refiner_uri,
            scheduler=self.config.scheduler,
            sdxl_refiner_scheduler=
            self.config.sdxl_refiner_scheduler if self.config.sdxl_refiner_uri else None,
            safety_checker=self.config.safety_checker,
            auth_token=self.config.auth_token,
            local_files_only=self.config.offline_mode)

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
            pipeline_wrapper = self._create_pipeline_wrapper()

            sdxl_high_noise_fractions = \
                self.config.sdxl_high_noise_fractions if \
                    self.config.sdxl_refiner_uri is not None else None

            for args_ctx in self.config.iterate_diffusion_args(
                    sdxl_high_noise_fraction=sdxl_high_noise_fractions,
                    image_seed_strength=None,
                    upscaler_noise_level=None):
                self._pre_generation_step(args_ctx)
                self._pre_generation(args_ctx)

                with pipeline_wrapper(**args_ctx.get_pipeline_wrapper_args(),
                                      width=self.config.output_size[0],
                                      height=self.config.output_size[1],
                                      batch_size=self.config.batch_size) as generation_result:
                    self._write_prompt_only_image(args_ctx, generation_result)

    def _render_with_image_seeds(self):
        pipeline_wrapper = self._create_pipeline_wrapper()

        sdxl_high_noise_fractions = \
            self.config.sdxl_high_noise_fractions if \
                self.config.sdxl_refiner_uri is not None else None

        image_seed_strengths = self.config.image_seed_strengths if \
            not (_pipelinewrapper.model_type_is_upscaler(self.config.model_type) or
                 _pipelinewrapper.model_type_is_pix2pix(self.config.model_type)) else None

        upscaler_noise_levels = self.config.upscaler_noise_levels if \
            self.config.model_type == _pipelinewrapper.ModelTypes.TORCH_UPSCALER_X4 else None

        def validate_image_seeds():
            for uri in self.config.image_seeds:
                parsed = _mediainput.parse_image_seed_uri(uri)

                if self.config.control_net_uris and not parsed.is_single_image() and parsed.control_path is None:
                    raise NotImplementedError(
                        f'You must specify a control image with the control argument '
                        f'IE: --image-seeds "my-seed.png;control=my-control.png" in your '
                        f'--image-seeds "{uri}" when using --control-nets in order '
                        f'to use inpainting. If you want to use the control image alone '
                        f'without a mask, use --image-seeds "{parsed.seed_path}".')

                yield uri, parsed

        for image_seed_uri, parsed_image_seed in list(validate_image_seeds()):

            is_single_control_image = self.config.control_net_uris and parsed_image_seed.is_single_image()
            image_seed_strengths = image_seed_strengths if not is_single_control_image else None
            upscaler_noise_levels = upscaler_noise_levels if not is_single_control_image else None

            if is_single_control_image:
                _messages.log(f'Processing Control Image: "{image_seed_uri}"', underline=True)
            else:
                _messages.log(f'Processing Image Seed: "{image_seed_uri}"', underline=True)

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
                def image_seed_iterator():
                    yield from _mediainput.iterate_control_image(
                        parsed_image_seed,
                        self.config.frame_start,
                        self.config.frame_end,
                        self.config.output_size,
                        preprocessor=self._load_preprocessors(self.config.control_image_preprocessors))

            else:
                def image_seed_iterator():
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

                self._render_animation(pipeline_wrapper=pipeline_wrapper,
                                       get_extra_wrapper_args=get_extra_args,
                                       arg_iterator=arg_iterator,
                                       image_seed_iterator=image_seed_iterator(),
                                       fps=seed_info.fps)
                break

            for diffusion_arguments in arg_iterator:
                self._pre_generation_step(diffusion_arguments)
                with next(image_seed_iterator()) as image_seed:
                    with image_seed:
                        self._with_image_seed_pre_generation(diffusion_arguments, image_seed)

                        pipeline_args = diffusion_arguments.get_pipeline_wrapper_args()

                        if not is_single_control_image:
                            pipeline_args['image'] = image_seed.image

                        if image_seed.mask_image is not None:
                            pipeline_args['mask_image'] = image_seed.mask_image
                        else:
                            pipeline_args['control_image'] = (image_seed.image if is_single_control_image
                                                              else image_seed.control_image)

                        with _mediainput.MultiContextManager(
                                [image_seed.mask_image, image_seed.control_image]), \
                                pipeline_wrapper(**pipeline_args,
                                                 batch_size=self.config.batch_size) as generation_result:

                            self._write_image_seed_gen_image(diffusion_arguments, image_seed, generation_result)

    def _gen_animation_filename(self,
                                args_ctx: _pipelinewrapper.DiffusionArguments,
                                generation_step,
                                ext):

        components = ['ANIM', *self._gen_filename_components_base(args_ctx), 'step', generation_step + 1]

        return self._join_output_filename(components, ext=ext)

    def _render_animation(self,
                          pipeline_wrapper: _pipelinewrapper.DiffusionPipelineWrapper,
                          get_extra_wrapper_args:
                          typing.Callable[[_mediainput.ImageSeed], typing.Dict[str, typing.Any]],
                          arg_iterator:
                          typing.Generator[_pipelinewrapper.DiffusionArguments, None, None],
                          image_seed_iterator:
                          typing.Generator[_mediainput.ImageSeed, None, None],
                          fps: typing.Union[int, float]):

        animation_format_lower = self.config.animation_format.lower()
        first_args_ctx = next(arg_iterator)

        base_filename = \
            self._gen_animation_filename(
                first_args_ctx, self._generation_step + 1,
                ext=animation_format_lower)

        next_frame_terminates_anim = False

        if self.disable_writes:
            # The interface can be used as a mock object
            anim_writer = _mediaoutput.AnimationWriter()
        else:
            anim_writer = _mediaoutput.MultiAnimationWriter(
                animation_format=animation_format_lower,
                filename=base_filename,
                fps=fps, allow_overwrites=self.config.output_overwrite)

        with anim_writer:

            for args_ctx in itertools.chain([first_args_ctx], arg_iterator):
                self._pre_generation_step(args_ctx)

                if next_frame_terminates_anim:
                    next_frame_terminates_anim = False

                    anim_writer.end(
                        new_file=self._gen_animation_filename(
                            args_ctx, self._generation_step,
                            ext=animation_format_lower))

                for image_seed in image_seed_iterator:
                    with image_seed:

                        self._animation_frame_pre_generation(args_ctx, image_seed)

                        extra_args = get_extra_wrapper_args(image_seed)

                        with pipeline_wrapper(**args_ctx.get_pipeline_wrapper_args(),
                                              **extra_args,
                                              batch_size=self.config.batch_size) as generation_result:

                            if generation_result.image_count > 1 and self.config.batch_grid_size is not None:
                                anim_writer.write(
                                    generation_result.image_grid(self.config.batch_grid_size))
                            else:
                                anim_writer.write(generation_result.images)

                            if image_seed.frame_index == 0:
                                # Preform on first frame write

                                if not self.disable_writes:

                                    animation_filenames_message = \
                                        '\n'.join(f'Beginning Writes To Animation: "{f}"'
                                                  for f in anim_writer.filenames)

                                    if self.config.output_configs:

                                        _messages.log(animation_filenames_message)

                                        for idx, filename in enumerate(anim_writer.filenames):
                                            self._write_animation_config_file(
                                                filename=os.path.splitext(filename)[0] + '.txt',
                                                batch_idx=idx,
                                                generation_result=generation_result)
                                    else:
                                        _messages.log(animation_filenames_message, underline=True)

                                    for filename in anim_writer.filenames:
                                        self._written_animations.append(os.path.abspath(filename))

                            self._write_animation_frame(args_ctx, image_seed, generation_result)

                        next_frame_terminates_anim = image_seed.frame_index == (image_seed.total_frames - 1)

                anim_writer.end()

    def _write_animation_config_file(self, filename, batch_idx, generation_result):
        extra_opts = []

        if self.config.frame_start is not None and \
                self.config.frame_start != 0:
            extra_opts.append(('--frame-start',
                               self.config.frame_start))

        if self.config.frame_end is not None:
            extra_opts.append(('--frame-end',
                               self.config.frame_end))

        if self.config.animation_format is not None:
            extra_opts.append(('--animation-format',
                               self.config.animation_format))

        extra_comments = []

        if generation_result.image_count > 1 and \
                self.config.batch_grid_size is None:
            extra_comments.append(
                f'Animation {batch_idx + 1} from a batch of {generation_result.image_count}')

        config_text = \
            generation_result.gen_dgenerate_config(
                extra_args=extra_opts,
                extra_comments=extra_comments)

        if not self.config.output_overwrite:
            filename = \
                _util.touch_avoid_duplicate(
                    self.config.output_path,
                    pathmaker=_util.suffix_path_maker(filename,
                                                      '_duplicate_'))

        with open(filename, "w") as config_file:
            config_file.write(config_text)

        _messages.log(f'Wrote Animation Config File: "{filename}"',
                      underline=batch_idx == generation_result.image_count - 1)
