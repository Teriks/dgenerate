import inspect
import itertools
import random
import typing

import dgenerate.mediainput as _mediainput
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.prompt as _prompt
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


def iterate_attribute_combinations(
        attribute_defs: typing.List[typing.Tuple[str, typing.List]],
        my_class: typing.Type):
    """
    Iterate over every combination of attributes in a given class using a list of tuples mapping
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


class RenderLoopConfigError(Exception):
    """
    Raised by :py:meth:`.RenderLoopConfig.check` on configuration errors.
    """
    pass


class RenderLoopConfig(_types.SetFromMixin):
    """
    This object represents configuration for :py:class:`RenderLoop`.

    It nearly directly maps to dgenerates command line arguments.

    See subclass :py:class:`dgenerate.arguments.DgenerateArguments`
    """

    model_path: _types.OptionalPath = None
    """
    Primary diffusion model path, ``model_path`` argument of dgenerate command line tool.
    """

    subfolder: _types.OptionalPath = None
    """
    Primary model subfolder argument, ``--subfolder`` argument of dgenerate command line tool.
    """

    sdxl_refiner_uri: _types.OptionalUri = None
    """
    SDXL Refiner model URI, ``--sdxl-refiner`` argument of dgenerate command line tool.
    """

    batch_size: _types.OptionalInteger = None
    """
    Image generation batch size, ``--batch-size`` argument of dgenerate command line tool.
    """

    batch_grid_size: _types.OptionalSize = None
    """
    Optional image grid size specification for when **batch_size** is greater than one.
    This is the ``--batch-grid-size`` argument of the dgenerate command line tool.
    """

    prompts: _types.Prompts
    """
    List of prompt objects, this corresponds to the ``--prompts`` argument of the dgenerate
    command line tool.
    """

    sdxl_second_prompts: _types.OptionalPrompts = None
    """
    Optional list of SDXL secondary prompts, this corresponds to the ``--sdxl-secondary-prompts`` argument
    of the dgenerate command line tool.
    """

    sdxl_refiner_prompts: _types.OptionalPrompts = None
    """
    Optional list of SDXL refiner prompt overrides, this corresponds to the ``--sdxl-refiner-prompts`` argument
    of the dgenerate command line tool.
    """

    sdxl_refiner_second_prompts: _types.OptionalPrompts = None
    """
    Optional list of SDXL refiner secondary prompt overrides, this corresponds 
    to the ``--sdxl-refiner-second-prompts`` argument of the dgenerate command line tool.
    """

    seeds: _types.Integers
    """
    List of integer seeds, this corresponds to the ``--seeds`` argument of 
    the dgenerate command line tool.
    """

    seeds_to_images: bool = False
    """
    Should :py:attr:`RenderLoopConfig.seeds` be interpreted as seeds for each
    image input instead of combinatorial? this includes control images.
    """

    guidance_scales: _types.Floats
    """
    List of floating point guidance scales, this corresponds to the ``--guidance-scales`` argument 
    of the dgenerate command line tool.
    """

    inference_steps: _types.Integers
    """
    List of inference steps values, this corresponds to the ``--inference-steps`` argument of the
    dgenerate command line tool.
    """

    image_seeds: _types.OptionalUris = None
    """
    List of ``--image-seeds`` URI strings.
    """

    parsed_image_seeds: typing.Optional[typing.List[_mediainput.ImageSeedParseResult]] = None
    """
    The results of parsing URIs mentioned in :py:attr:`.RenderLoopConfig.image_seeds`, 
    will only be available if :py:meth:`.RenderLoopConfig.check` has been called.
    """

    image_seed_strengths: _types.OptionalFloats = None
    """
    Optional list of floating point image seed strengths, this corresponds to the ``--image-seed-strengths`` argument
    of the dgenerate command line tool.
    """

    upscaler_noise_levels: _types.OptionalIntegers = None
    """
    Optional list of integer upscaler noise levels, this corresponds to the ``--upscaler-noise-levels`` argument
    of the dgenerate command line tool that is used with upscaler model types.
    """

    guidance_rescales: _types.OptionalFloats = None
    """
    List of floating point guidance rescale values which are supported by some pipelines, (there will be an
    error if it is unsupported upon running), this corresponds to the ``--guidance-rescales`` argument of 
    the dgenerate command line tool.
    """

    image_guidance_scales: _types.OptionalFloats = None
    """
    Optional list of floating point image guidance scales, used for pix2pix model types, this corresponds
    to the ``--image-guidance-scales`` argument of the dgenerate command line tool.
    """

    sdxl_high_noise_fractions: _types.OptionalFloats = None
    """
    Optional list of SDXL refiner high noise fractions (floats), this value is the fraction of inference steps
    that the base model handles, the inverse proportion of the provided fraction is handled by the refiner model.
    This corresponds to the ``--sdxl-high-noise-fractions`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_inference_steps: _types.OptionalIntegers = None
    """
    Optional list of inference steps value overrides for the SDXL refiner, this corresponds 
    to the ``--sdxl-refiner-inference-steps`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_guidance_scales: _types.OptionalFloats = None
    """
    Optional list of guidance scale value overrides for the SDXL refiner, this corresponds 
    to the ``--sdxl-refiner-guidance-scales`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_guidance_rescales: _types.OptionalFloats = None
    """
    Optional list of guidance rescale value overrides for the SDXL refiner, this corresponds 
    to the ``--sdxl-refiner-guidance-rescales`` argument of the dgenerate command line tool.
    """

    sdxl_aesthetic_scores: _types.OptionalFloats = None
    """
    Optional list of SDXL aesthetic-score conditioning values, this corresponds 
    to the ``--sdxl-aesthetic-scores`` argument of the dgenerate command line tool.
    """

    sdxl_original_sizes: _types.OptionalSizes = None
    """
    Optional list of SDXL original-size micro-conditioning parameters, this corresponds 
    to the ``--sdxl-original-sizes`` argument of the dgenerate command line tool.
    """

    sdxl_target_sizes: _types.OptionalSizes = None
    """
    Optional list of SDXL target-size micro-conditioning parameters, this corresponds 
    to the ``--sdxl-target-sizes`` argument of the dgenerate command line tool.
    """

    sdxl_crops_coords_top_left: _types.OptionalCoordinateList = None
    """
    Optional list of SDXL top-left-crop-coords micro-conditioning parameters, this corresponds 
    to the ``--sdxl-crops-coords-top-left`` argument of the dgenerate command line tool.
    """

    sdxl_negative_aesthetic_scores: _types.OptionalFloats = None
    """
    Optional list of negative influence SDXL aesthetic-score conditioning values, 
    this corresponds to the ``--sdxl-negative-aesthetic-scores`` argument of the dgenerate
    command line tool.
    """

    sdxl_negative_original_sizes: _types.OptionalSizes = None
    """
    Optional list of negative influence SDXL original-size micro-conditioning parameters, 
    this corresponds to the ``--sdxl-negative-original-sizes`` argument of the dgenerate command 
    line tool.
    """

    sdxl_negative_target_sizes: _types.OptionalSizes = None
    """
    Optional list of negative influence SDXL target-size micro-conditioning parameters, 
    this corresponds to the ``--sdxl-negative-target-sizes`` argument of the dgenerate 
    command line tool.
    """

    sdxl_negative_crops_coords_top_left: _types.OptionalCoordinateList = None
    """
    Optional list of negative influence SDXL top-left crop coords micro-conditioning parameters, 
    this corresponds to the ``--sdxl-negative-crops-coords-top-left`` argument of the dgenerate 
    command line tool.
    """

    sdxl_refiner_aesthetic_scores: _types.OptionalFloats = None
    """
    Optional list of SDXL-refiner override aesthetic-score conditioning values, this 
    corresponds to the ``--sdxl-refiner-aesthetic-scores`` argument of the dgenerate command 
    line tool.
    """

    sdxl_refiner_original_sizes: _types.OptionalSizes = None
    """
    Optional list of SDXL-refiner override original-size micro-conditioning parameters, 
    this corresponds to the ``--sdxl-refiner-original-sizes`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_target_sizes: _types.OptionalSizes = None
    """
    Optional list of SDXL-refiner override target-size micro-conditioning parameters, this 
    corresponds to the ``--sdxl-refiner-target-sizes`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_crops_coords_top_left: _types.OptionalCoordinateList = None
    """
    Optional list of SDXL-refiner override top-left-crop-coords micro-conditioning parameters, this 
    corresponds to the ``--sdxl-refiner-crops-coords-top-left`` argument of the dgenerate command line tool.
    """

    sdxl_refiner_negative_aesthetic_scores: _types.OptionalFloats = None
    """
    Optional list of negative influence SDXL-refiner override aesthetic-score conditioning values, 
    this corresponds to the ``--sdxl-refiner-negative-aesthetic-scores`` argument of the dgenerate
    command line tool.
    """

    sdxl_refiner_negative_original_sizes: _types.OptionalSizes = None
    """
    Optional list of negative influence SDXL-refiner override original-size micro-conditioning 
    parameters, this corresponds to the ``--sdxl-refiner-negative-original-sizes`` argument of 
    the dgenerate command line tool.
    """

    sdxl_refiner_negative_target_sizes: _types.OptionalSizes = None
    """
    Optional list of negative influence SDXL-refiner override target-size micro-conditioning 
    parameters, this corresponds to the ``--sdxl-refiner-negative-target-sizes`` argument of 
    the dgenerate command line tool.
    """

    sdxl_refiner_negative_crops_coords_top_left: _types.OptionalCoordinateList = None
    """
    Optional list of negative influence SDXL-refiner top-left crop coords micro-conditioning parameters, 
    this corresponds to the ``--sdxl-refiner-negative-crops-coords-top-left`` argument of the dgenerate 
    command line tool.
    """

    vae_uri: _types.OptionalUri = None
    """
    Optional user specified VAE URI, this corresponds to the ``--vae`` argument of the dgenerate command line tool.
    """

    vae_tiling: bool = False
    """
    Enable VAE tiling? ``--vae-tiling``
    """

    vae_slicing: bool = False
    """
    Enable VAE slicing? ``--vae-slicing``
    """

    lora_uris: _types.OptionalUris = None
    """
    Optional user specified LoRA URIs, this corresponds to the ``--lora/--loras`` argument of the dgenerate 
    command line tool. Currently only one lora is supported, providing more than one will cause an error.
    """

    textual_inversion_uris: _types.OptionalUris = None
    """
    Optional user specified Textual Inversion URIs, this corresponds to the ``--textual-inversions``
    argument of the dgenerate command line tool.
    """

    control_net_uris: _types.OptionalUris = None
    """
    Optional user specified ControlNet URIs, this corresponds to the ``--control-nets`` argument
    of the dgenerate command line tool.
    """

    scheduler: _types.OptionalName = None
    """
    Optional primary model scheduler/sampler class name specification, this corresponds to the ``--scheduler``
    argument of the dgenerate command line tool. Setting this to 'help' will yield a help message to stdout
    describing scheduler names compatible with the current configuration upon running.
    """

    sdxl_refiner_scheduler: _types.OptionalName = None
    """
    Optional SDXL refiner model scheduler/sampler class name specification, this corresponds to the 
    ``--sdxl-refiner-scheduler`` argument of the dgenerate command line tool. Setting this to 'help' 
    will yield a help message to stdout describing scheduler names compatible with the current 
    configuration upon running.
    """

    safety_checker: bool = False
    """
    Enable safety checker? ``--safety-checker``
    """

    model_type: _pipelinewrapper.ModelTypes = _pipelinewrapper.ModelTypes.TORCH
    """
    Corresponds to the ``--model-type`` argument of the dgenerate command line tool.
    """

    device: _types.Name = 'cuda'
    """
    Processing device specification, for example "cuda" or "cuda:N" where N is an 
    alternate GPU id as reported by nvidia-smi if you want to specify a specific GPU.
    This corresponds to the ``--device`` argument of the dgenerate command line tool.
    """

    dtype: _pipelinewrapper.DataTypes = _pipelinewrapper.DataTypes.AUTO
    """
    Primary model data type specification, IE: integer precision. Default is auto selection.
    Lower precision datatypes result in less GPU memory usage.
    This corresponds to the ``--dtype`` argument of the dgenerate command line tool.
    """

    revision: _types.Name = 'main'
    """
    Repo revision selector for the main model when loading from a huggingface repository.
    This corresponds to the ``--revision`` argument of the dgenerate command line tool.
    """

    variant: _types.OptionalName = None
    """
    Primary model weights variant string.
    This corresponds to the ``--variant`` argument of the dgenerate command line tool.
    """

    output_size: _types.OptionalSize = None
    """
    Desired output size, sizes not aligned by 8 pixels will result in an error message.
    This corresponds to the ``--output-size`` argument of the dgenerate command line tool.
    """

    output_path: _types.Path = 'output'
    """
    Render loop write folder, where images and animations will be written.
    This corresponds to the ``--output-path`` argument of the dgenerate command line tool.
    """

    output_prefix: typing.Optional[str] = None
    """
    Output filename prefix, add an optional prefix string to all written files.
    This corresponds to the ``--output-prefix`` argument of the dgenerate command line tool.
    """

    output_overwrite: bool = False
    """
    Allow overwrites of files? or avoid this with a file suffix in a multiprocess safe manner?
    This corresponds to the ``--output-overwrite`` argument of the dgenerate command line tool.
    """

    output_configs: bool = False
    """
    Output a config text file next to every generated image or animation? this file will contain configuration
    that is pipeable to dgenerate stdin, which will reproduce said file.
    This corresponds to the ``--output-configs`` argument of the dgenerate command line tool.
    """

    output_metadata: bool = False
    """
    Write config text to the metadata of all written images? this data is not written to animated files, only PNGs.
    This corresponds to the ``--output-metadata`` argument of the dgenerate command line tool.
    """

    animation_format: _types.Name = 'mp4'
    """
    Format for any rendered animations, see: :py:meth:`dgenerate.mediaoutput.supported_animation_writer_formats()`
    This corresponds to the ``--animation-format`` argument of the dgenerate command line tool.
    """

    frame_start: _types.Integer = 0
    """
    Start frame inclusive frame slice for any rendered animations.
    This corresponds to the ``--frame-start`` argument of the dgenerate command line tool.
    """

    frame_end: _types.OptionalInteger = None
    """
    Optional end frame inclusive frame slice for any rendered animations.
    This corresponds to the ``--frame-end`` argument of the dgenerate command line tool.
    """

    auth_token: typing.Optional[str] = None
    """
    Optional huggingface API token which will allow the download of restricted repositories 
    that your huggingface account has been granted access to.
    This corresponds to the ``--auth-token`` argument of the dgenerate command line tool.
    """

    seed_image_preprocessors: _types.OptionalUris = None
    """
    Corresponds to the ``--seed-image-preprocessors`` argument of the dgenerate command line tool verbatim.
    """

    mask_image_preprocessors: _types.OptionalUris = None
    """
    Corresponds to the ``--mask-image-preprocessors`` argument of the dgenerate command line tool verbatim.
    """

    control_image_preprocessors: _types.OptionalUris = None
    """
    Corresponds to the ``--control-image-preprocessors`` argument of the dgenerate command line tool verbatim,
    including the grouping syntax using the "+" symbol, the plus symbol should be used as its own list element,
    IE: it is a token.
    """

    offline_mode: bool = False
    """
    Avoid ever connecting to the huggingface API? this can be used if all your models are cached or
    if you are only ever using models that exist on disk. This is currently broken for LoRA 
    specifications due to a bug in the huggingface diffusers library. Corresponds to the
    ``--offline-mode`` argument of the dgenerate command line tool.
    """

    def __init__(self):
        self.guidance_scales = [_pipelinewrapper.DEFAULT_GUIDANCE_SCALE]
        self.inference_steps = [_pipelinewrapper.DEFAULT_INFERENCE_STEPS]
        self.prompts = [_prompt.Prompt()]
        self.seeds = gen_seeds(1)

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

        :raises: :py:class:`.RenderLoopConfigError` on errors

        :param attribute_namer: Callable for naming attributes mentioned in exception messages
        """

        def a_namer(attr_name):
            if attribute_namer:
                return attribute_namer(attr_name)
            return f'RenderLoopConfig.{attr_name}'

        def _has_len(name, value):
            try:
                len(value)
                return True
            except TypeError:
                raise RenderLoopConfigError(
                    f'{a_namer(name)} must be able to be used with len(), value was: {value}')

        def _is_optional_two_tuple(name, value):
            if value is not None and not (isinstance(value, tuple) and len(value) == 2):
                raise RenderLoopConfigError(
                    f'{a_namer(name)} must be None or a tuple of length 2, value was: {value}')

        def _is_optional(value_type, name, value):
            if value is not None and not isinstance(value, value_type):
                raise RenderLoopConfigError(
                    f'{a_namer(name)} must be None or type {value_type.__name__}, value was: {value}')

        def _is(value_type, name, value):
            if not isinstance(value, value_type):
                raise RenderLoopConfigError(
                    f'{a_namer(name)} must be type {value_type.__name__}, value was: {value}')

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
            raise RenderLoopConfigError(
                f'{a_namer("dtype")} must be {_textprocessing.oxford_comma(supported_dtypes, "or")}')
        if self.batch_size is not None and self.batch_size < 1:
            raise RenderLoopConfigError(
                f'{a_namer("batch_size")} must be greater than or equal to 1.')
        if self.model_type not in _pipelinewrapper.supported_model_type_enums():
            supported_model_types = _textprocessing.oxford_comma(_pipelinewrapper.supported_model_type_strings(), "or")
            raise RenderLoopConfigError(
                f'{a_namer("model_type")} must be one of: {supported_model_types}')
        if not _pipelinewrapper.is_valid_device_string(self.device):
            raise RenderLoopConfigError(
                f'{a_namer("device")} must be "cuda" (optionally with a device ordinal "cuda:N") or "cpu"')

        def attr_that_start_with(s):
            return (a for a in dir(self) if a.startswith(s) and getattr(self, a))

        def attr_that_end_with(s):
            return (a for a in dir(self) if a.endswith(s) and getattr(self, a))

        if self.model_path is None:
            raise RenderLoopConfigError(
                f'{a_namer("model_path")} must be specified')

        if self.frame_end is not None and self.frame_start > self.frame_end:
            raise RenderLoopConfigError(
                f'{a_namer("frame_start")} must be less than or equal to {a_namer("frame_end")}')

        if self.batch_size is not None:
            if _pipelinewrapper.model_type_is_flax(self.model_type):
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("batch_size")} when using flax, '
                    'use the environmental variable: CUDA_VISIBLE_DEVICES')
        elif not _pipelinewrapper.model_type_is_flax(self.model_type):
            self.batch_size = 1

        if self.output_size is None and not self.image_seeds:
            if _pipelinewrapper.model_type_is_sdxl(self.model_type):
                self.output_size = (1024, 1024)
            elif _pipelinewrapper.model_type_is_floyd_if(self.model_type):
                self.output_size = (64, 64)
            else:
                self.output_size = (512, 512)

        if not self.image_seeds:
            if _pipelinewrapper.model_type_is_floyd_ifs(self.model_type):
                raise RenderLoopConfigError(
                    f'you cannot specify Deep Floyd IF super-resolution '
                    f'({a_namer("model_type")} "{self.model_type})" without {a_namer("image_seeds")}.'
                )

            if _pipelinewrapper.model_type_is_upscaler(self.model_type):
                raise RenderLoopConfigError(
                    f'you cannot specify an upscaler model '
                    f'({a_namer("model_type")} "{self.model_type})" without {a_namer("image_seeds")}.'
                )

            if _pipelinewrapper.model_type_is_pix2pix(self.model_type):
                raise RenderLoopConfigError(
                    f'you cannot specify a pix2pix model '
                    f'({a_namer("model_type")} "{self.model_type})" without {a_namer("image_seeds")}.'
                )

            if self.image_seed_strengths:
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("image_seed_strengths")} without {a_namer("image_seeds")}.')

            if self.seeds_to_images:
                raise RenderLoopConfigError(
                    f'{a_namer("seeds_to_images")} can not be specified without {a_namer("image_seeds")}.')

            if self.control_net_uris:
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("control_net_uris")} without {a_namer("image_seeds")}.')

        upscaler_noise_levels_default_set = False
        if not _pipelinewrapper.model_type_is_upscaler(self.model_type):
            if self.upscaler_noise_levels:
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("upscaler_noise_levels")} for a '
                    f'non upscaler model type, see: {a_namer("model_type")}.')
        elif self.control_net_uris:
            raise RenderLoopConfigError(
                f'{a_namer("control_net_uris")} is not compatible '
                f'with upscaler models, see: {a_namer("model_type")}.')
        elif self.upscaler_noise_levels is None:
            if self.model_type == _pipelinewrapper.ModelTypes.TORCH_UPSCALER_X4:
                upscaler_noise_levels_default_set = True
                self.upscaler_noise_levels = [_pipelinewrapper.DEFAULT_X4_UPSCALER_NOISE_LEVEL]
        elif self.model_type != _pipelinewrapper.ModelTypes.TORCH_UPSCALER_X4:
            raise RenderLoopConfigError(
                f'you cannot specify {a_namer("upscaler_noise_levels")} for an upscaler '
                f'model type that is not "torch-upscaler-x4", see: {a_namer("model_type")}.')

        if not _pipelinewrapper.model_type_is_pix2pix(self.model_type):
            if self.image_guidance_scales:
                raise RenderLoopConfigError(
                    f'argument {a_namer("image_guidance_scales")} only valid with '
                    f'pix2pix models, see: {a_namer("model_type")}.')
        elif self.control_net_uris:
            raise RenderLoopConfigError(
                f'{a_namer("control_net_uris")} is not compatible with '
                f'pix2pix models, see: {a_namer("model_type")}.')
        elif not self.image_guidance_scales:
            self.image_guidance_scales = [_pipelinewrapper.DEFAULT_IMAGE_GUIDANCE_SCALE]

        if self.control_image_preprocessors:
            if not self.image_seeds:
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("control_image_preprocessors")} '
                    f'without {a_namer("image_seeds")}.')

        if not self.image_seeds:
            invalid_self = []
            for preprocessor_self in attr_that_end_with('preprocessors'):
                invalid_self.append(
                    f'you cannot specify {a_namer(preprocessor_self)} '
                    f'without {a_namer("image_seeds")}.')
            if invalid_self:
                raise RenderLoopConfigError('\n'.join(invalid_self))

        if not _pipelinewrapper.model_type_is_sdxl(self.model_type):
            invalid_self = []
            for sdxl_self in attr_that_start_with('sdxl'):
                invalid_self.append(f'you cannot specify {a_namer(sdxl_self)} '
                                    f'for a non SDXL model type, see: {a_namer("model_type")}.')
            if invalid_self:
                raise RenderLoopConfigError('\n'.join(invalid_self))

            self.sdxl_high_noise_fractions = None
        else:
            if not self.sdxl_refiner_uri:
                invalid_self = []
                for sdxl_self in attr_that_start_with('sdxl_refiner'):
                    invalid_self.append(f'you cannot specify {a_namer(sdxl_self)} '
                                        f'without {a_namer("sdxl_refiner_uri")}.')
                if invalid_self:
                    raise RenderLoopConfigError('\n'.join(invalid_self))
            else:
                if self.sdxl_high_noise_fractions is None:
                    # Default value
                    self.sdxl_high_noise_fractions = [_pipelinewrapper.DEFAULT_SDXL_HIGH_NOISE_FRACTION]

        if not _pipelinewrapper.model_type_is_torch(self.model_type):
            if self.vae_tiling or self.vae_slicing:
                raise RenderLoopConfigError(
                    f'{a_namer("vae_tiling")}/{a_namer("vae_slicing")} not supported for '
                    f'non torch model type, see: {a_namer("model_type")}.')

        if self.scheduler == 'help' and self.sdxl_refiner_scheduler == 'help':
            raise RenderLoopConfigError(
                'cannot list compatible schedulers for the main model and the SDXL refiner at '
                f'the same time. Do not use the scheduler "help" option for {a_namer("scheduler")} '
                f'and {a_namer("sdxl_refiner_scheduler")} simultaneously.')

        if self.image_seeds:
            no_seed_strength = (_pipelinewrapper.model_type_is_upscaler(self.model_type) or
                                _pipelinewrapper.model_type_is_pix2pix(self.model_type))

            image_seed_strengths_default_set = False
            if self.image_seed_strengths is None:
                if not no_seed_strength:
                    image_seed_strengths_default_set = True
                    # Default value
                    self.image_seed_strengths = [_pipelinewrapper.DEFAULT_IMAGE_SEED_STRENGTH]
            else:
                if no_seed_strength:
                    raise RenderLoopConfigError(
                        f'{a_namer("image_seed_strengths")} cannot be used with pix2pix or upscaler models.')

            self.parsed_image_seeds = []

            for uri in self.image_seeds:
                parsed = _mediainput.parse_image_seed_uri(uri)
                self.parsed_image_seeds.append(parsed)

                is_control_guidance_spec = self.control_net_uris and parsed.is_single_spec()

                if is_control_guidance_spec and self.image_seed_strengths:
                    if image_seed_strengths_default_set:
                        # check() set this default that isn't valid
                        # upon further parsing
                        self.image_seed_strengths = None
                    else:
                        # user set this
                        raise RenderLoopConfigError(
                            f'Cannot use {a_namer("image_seed_strengths")} with a control guidance image '
                            f'specification "{uri}". IE: when {a_namer("control_net_uris")} is specified and '
                            f'your {a_namer("image_seeds")} specification has a single source or comma '
                            f'seperated list of sources.')

                if is_control_guidance_spec and self.upscaler_noise_levels:
                    # upscaler noise level should already be handled but handle it again just incase
                    if upscaler_noise_levels_default_set:
                        # check() set this default that isn't valid
                        # upon further parsing
                        self.upscaler_noise_levels = None
                    else:
                        # user set this
                        raise RenderLoopConfigError(
                            f'Cannot use {a_namer("upscaler_noise_levels")} with a control guidance image '
                            f'specification "{uri}". IE: when {a_namer("control_net_uris")} is specified and '
                            f'your {a_namer("image_seeds")} specification has a single source or comma '
                            f'seperated list of sources.')

                if self.control_net_uris and not parsed.is_single_spec() and parsed.control_path is None:
                    raise RenderLoopConfigError(
                        f'You must specify a control image with the control argument '
                        f'IE: "my-seed.png;control=my-control.png" in your '
                        f'{a_namer("image_seeds")} "{uri}" when using {a_namer("control_net_uris")} '
                        f'in order to use inpainting. If you want to use the control image alone '
                        f'without a mask, use {a_namer("image_seeds")} "{parsed.seed_path}".')

                if self.model_type == _pipelinewrapper.ModelTypes.TORCH_IFS_IMG2IMG or \
                        (parsed.mask_path and _pipelinewrapper.model_type_is_floyd_ifs(self.model_type)):

                    if not parsed.floyd_path:
                        mask_part = 'mask=my-mask.png;' if parsed.mask_path else ''

                        raise RenderLoopConfigError(
                            f'You must specify a floyd image with the floyd argument '
                            f'IE: "my-seed.png;{mask_part}floyd=previous-stage-image.png" '
                            f'in your {a_namer("image_seeds")} "{uri}" to disambiguate this '
                            f'usage of Deep Floyd IF super-resolution.')

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
                (len(self.seeds) if not self.seeds_to_images else 1) *
                len(self.guidance_scales) *
                len(self.inference_steps))

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
            inference_steps=ov('inference_steps', self.inference_steps),
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
