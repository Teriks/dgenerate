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
import collections.abc
import random
import typing

import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.prompt as _prompt
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.image as _image

CONTROL_IMAGE_PROCESSOR_SEP = '+'
"""
The character that is used to separate control image processor chains
when specifying processors for multiple control guidance images
"""


def _iterate_diffusion_args(**kwargs) -> collections.abc.Iterator[_pipelinewrapper.DiffusionArguments]:
    def _list_or_list_of_none(val):
        return val if val else [None]

    yield from _types.iterate_attribute_combinations(
        [(arg_name, _list_or_list_of_none(value)) for arg_name, value in kwargs.items()],
        _pipelinewrapper.DiffusionArguments)


def gen_seeds(n) -> list[int]:
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

    sdxl_refiner_edit: _types.OptionalBoolean = None
    """
    Force the SDXL refiner to operate in edit mode instead of cooperative denoising mode.
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

    clip_skips: _types.OptionalIntegers = None
    """
    List of clip skip values. Clip skip is the number of layers to be skipped from CLIP while computing the 
    prompt embeddings. A value of 1 means that the output of the pre-final layer will be used for computing 
    the prompt embeddings. Only supported for ``model_type`` values ``torch`` and ``torch-sdxl``, including with 
    ``control_net_uris`` defined.
    """

    sdxl_refiner_clip_skips: _types.OptionalIntegers = None
    """
    Clip skip override values for the SDXL refiner, which normally defaults to the clip skip 
    value for the main model when it is defined.
    """

    image_seeds: _types.OptionalUris = None
    """
    List of ``--image-seeds`` URI strings.
    """

    parsed_image_seeds: typing.Optional[collections.abc.Sequence[_mediainput.ImageSeedParseResult]] = None
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
    of the dgenerate command line tool that is used for the :py:attr:`dgenerate.pipelinewrapper.ModelType.TORCH_UPSCALER_X4`
    model type only.
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

    s_cascade_decoder_uri: _types.OptionalUri = None
    """
    Stable Cascade model URI, ``--s-cascade-decoder-uri`` argument of dgenerate command line tool.
    """

    s_cascade_decoder_prompts: _types.OptionalPrompts = None
    """
    Optional list of Stable Cascade decoder prompt overrides, this corresponds to the ``--s-cascade-decoder-prompts`` 
    argument of the dgenerate command line tool.
    """

    s_cascade_decoder_inference_steps: _types.OptionalIntegers = None
    """
    List of inference steps values for the Stable Cascade decoder, this corresponds 
    to the ``--s-cascade-decoder-inference-steps`` argument of the dgenerate command line tool.
    """

    s_cascade_decoder_guidance_scales: _types.OptionalFloats = None
    """
    List of guidance scale values for the Stable Cascade refiner, this 
    corresponds to the ``--s-cascade-decoder-guidance-scales`` argument of the dgenerate
    command line tool.
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

    sdxl_crops_coords_top_left: _types.OptionalCoordinates = None
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

    sdxl_negative_crops_coords_top_left: _types.OptionalCoordinates = None
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

    sdxl_refiner_crops_coords_top_left: _types.OptionalCoordinates = None
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

    sdxl_refiner_negative_crops_coords_top_left: _types.OptionalCoordinates = None
    """
    Optional list of negative influence SDXL-refiner top-left crop coords micro-conditioning parameters, 
    this corresponds to the ``--sdxl-refiner-negative-crops-coords-top-left`` argument of the dgenerate 
    command line tool.
    """

    unet_uri: _types.OptionalUri = None
    """
    Optional user specified UNet URI, this corresponds to the ``--unet`` argument of the dgenerate command line tool.
    """

    second_unet_uri: _types.OptionalUri = None
    """
    Optional user specified second UNet URI, this corresponds to the ``--unet2`` argument of the dgenerate command line tool.
    This UNet uri will be used for the SDXL refiner or Stable Cascade decoder model.
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
    Optional user specified LoRA URIs, this corresponds to the ``--loras`` argument 
    of the dgenerate command line tool.
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

    s_cascade_decoder_scheduler: _types.OptionalName = None
    """
    Optional Stable Cascade decoder model scheduler/sampler class name specification, this corresponds to the 
    ``--s-cascade-decoder-scheduler`` argument of the dgenerate command line tool. Setting this to 'help' 
    will yield a help message to stdout describing scheduler names compatible with the current 
    configuration upon running.
    """

    safety_checker: bool = False
    """
    Enable safety checker? ``--safety-checker``
    """

    model_type: _pipelinewrapper.ModelType = _pipelinewrapper.ModelType.TORCH
    """
    Corresponds to the ``--model-type`` argument of the dgenerate command line tool.
    """

    device: _types.Name = 'cuda'
    """
    Processing device specification, for example "cuda" or "cuda:N" where N is an 
    alternate GPU id as reported by nvidia-smi if you want to specify a specific GPU.
    This corresponds to the ``--device`` argument of the dgenerate command line tool.
    """

    dtype: _pipelinewrapper.DataType = _pipelinewrapper.DataType.AUTO
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

    no_aspect: bool = False
    """
    Should Seed, Mask, and Control guidance images specified in :py:attr:`.RenderLoopConfig.image_seeds`
    definitions (``--image-seeds``) have their aspect ratio ignored when being resized to 
    :py:attr:`.RenderLoopConfig.output_size` (``--output-size``) ?
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
    Format for any rendered animations, see: :py:func:`dgenerate.mediaoutput.supported_animation_writer_formats()`.
    This value may also be set to 'frames' to indicate that only individual frames should be output and no animation file coalesced.
    This corresponds to the ``--animation-format`` argument of the dgenerate command line tool.
    """

    image_format: _types.Name = 'png'
    """
    Format for any images that are written including animation frames.
    
    Anything other than "png" is not compatible with ``output_metadata=True`` and a
    :py:exc:`.RenderLoopConfigError` will be raised upon running the render loop if 
    ``output_metadata=True`` and this value is not "png"
    """

    no_frames: bool = False
    """
    Should individual frames not be output when rendering an animation? defaults to False.
    This corresponds to the ``--no-frames`` argument of the dgenerate command line tool.
    Using this option when :py:attr:`RenderLoopConfig.animation_format` is equal to ``"frames"`` will
    cause a :py:exc:`RenderLoopConfigError` be raised.
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

    seed_image_processors: _types.OptionalUris = None
    """
    Corresponds to the ``--seed-image-processors`` argument of the dgenerate command line tool verbatim.
    """

    mask_image_processors: _types.OptionalUris = None
    """
    Corresponds to the ``--mask-image-processors`` argument of the dgenerate command line tool verbatim.
    """

    control_image_processors: _types.OptionalUris = None
    """
    Corresponds to the ``--control-image-processors`` argument of the dgenerate command line tool verbatim,
    including the grouping syntax using the "+" symbol, the plus symbol should be used as its own list element,
    IE: it is a token.
    """

    post_processors: _types.OptionalUris = None
    """
    Corresponds to the ``--post-processors`` argument of the dgenerate command line tool verbatim.
    """

    offline_mode: bool = False
    """
    Avoid ever connecting to the huggingface API? this can be used if all your models are cached or
    if you are only ever using models that exist on disk. Corresponds to the
    ``--offline-mode`` argument of the dgenerate command line tool.
    """

    model_cpu_offload: bool = False
    """
    Force model cpu offloading for the main pipeline, this may reduce memory consumption
    and allow large models to run when they would otherwise not fit in your GPUs VRAM. 
    Inference will be slower. Mutually exclusive with :py:attr:`RenderLoopConfig.model_sequential_offload`
    """

    model_sequential_offload: bool = False
    """
    Force sequential model offloading for the main pipeline, this may drastically reduce memory consumption
    and allow large models to run when they would otherwise not fit in your GPUs VRAM. 
    Inference will be much slower. Mutually exclusive with :py:attr:`RenderLoopConfig.model_cpu_offload`
    """

    sdxl_refiner_cpu_offload: _types.OptionalBoolean = None
    """
    Force model cpu offloading for the SDXL refiner pipeline, this may reduce memory consumption
    and allow large models to run when they would otherwise not fit in your GPUs VRAM. 
    Inference will be slower. Mutually exclusive with :py:attr:`RenderLoopConfig.sdxl_refiner_sequential_offload`
    """

    sdxl_refiner_sequential_offload: _types.OptionalBoolean = None
    """
    Force sequential model offloading for the SDXL refiner pipeline, this may drastically
    reduce memory consumption and allow large models to run when they would otherwise not fit in 
    your GPUs VRAM. Inference will be much slower. Mutually exclusive with :py:attr:`RenderLoopConfig.sdxl_refiner_cpu_offload`
    """

    s_cascade_decoder_cpu_offload: _types.OptionalBoolean = None
    """
    Force model cpu offloading for the Stable Cascade decoder pipeline, this may reduce memory consumption
    and allow large models to run when they would otherwise not fit in your GPUs VRAM. 
    Inference will be slower. Mutually exclusive with 
    :py:attr:`RenderLoopConfig.s_cascade_decoder_sequential_offload`
    """

    s_cascade_decoder_sequential_offload: _types.OptionalBoolean = None
    """
    Force sequential model offloading for the Stable Cascade decoder pipeline, this may drastically
    reduce memory consumption and allow large models to run when they would otherwise not fit in 
    your GPUs VRAM. Inference will be much slower. Mutually exclusive with 
    :py:attr:`RenderLoopConfig.s_cascade_decoder_cpu_offload`
    """

    def __init__(self):
        self.guidance_scales = [_pipelinewrapper.DEFAULT_GUIDANCE_SCALE]
        self.inference_steps = [_pipelinewrapper.DEFAULT_INFERENCE_STEPS]
        self.prompts = [_prompt.Prompt()]
        self.seeds = gen_seeds(1)

    def check(self, attribute_namer: typing.Callable[[str], str] = None):
        """
        Check the configuration for type and logical usage errors, set
        defaults for certain values when needed and not specified.

        :raises RenderLoopConfigError: on errors

        :param attribute_namer: Callable for naming attributes mentioned in exception messages
        """

        def a_namer(attr_name):
            if attribute_namer:
                return attribute_namer(attr_name)
            return f'{self.__class__.__name__}.{attr_name}'

        try:
            _types.type_check_struct(self, attribute_namer)
        except ValueError as e:
            raise RenderLoopConfigError(e)

        # Detect logically incorrect config and set certain defaults
        def non_null_attr_that_start_with(s):
            return (a for a in dir(self) if a.startswith(s) and getattr(self, a) is not None)

        def non_null_attr_that_end_with(s):
            return (a for a in dir(self) if a.endswith(s) and getattr(self, a) is not None)

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

        if self.model_cpu_offload and self.model_sequential_offload:
            raise RenderLoopConfigError(
                f'{a_namer("model_cpu_offload")} and {a_namer("model_sequential_offload")} '
                f'may not be enabled simultaneously.')

        if self.sdxl_refiner_cpu_offload and self.sdxl_refiner_sequential_offload:
            raise RenderLoopConfigError(
                f'{a_namer("refiner_cpu_offload")} and {a_namer("refiner_sequential_offload")} '
                f'may not be enabled simultaneously.')

        if self.s_cascade_decoder_cpu_offload and self.s_cascade_decoder_sequential_offload:
            raise RenderLoopConfigError(
                f'{a_namer("s_cascade_decoder_cpu_offload")} and {a_namer("s_cascade_decoder_sequential_offload")} '
                f'may not be enabled simultaneously.')

        if self.model_type == _pipelinewrapper.ModelType.TORCH_S_CASCADE_DECODER:
            raise RenderLoopConfigError(
                f'Stable Cascade decoder {a_namer("model_type")} may not be used as the primary model.')

        if self.model_type == _pipelinewrapper.ModelType.TORCH_S_CASCADE:
            if not self.s_cascade_decoder_uri:
                raise RenderLoopConfigError(
                    f'You must specify a Stable Cascade decoder '
                    f'model when {a_namer("model_type")} is "torch-s-cascade"')

            if self.vae_uri is not None:
                raise RenderLoopConfigError(
                    f'Stable Cascade can not use a {a_namer("vae_uri")} value.')

            if not self.s_cascade_decoder_guidance_scales:
                self.s_cascade_decoder_guidance_scales = [
                    _pipelinewrapper.DEFAULT_S_CASCADE_DECODER_GUIDANCE_SCALE]

            if not self.s_cascade_decoder_inference_steps:
                self.s_cascade_decoder_inference_steps = [
                    _pipelinewrapper.DEFAULT_S_CASCADE_DECODER_INFERENCE_STEPS]

            if self.control_net_uris is not None:
                raise RenderLoopConfigError(
                    f'Stable Cascade does not currently support the use of {a_namer("control_net_uris")}.')

            if self.output_size is not None and not _image.is_power_of_two(self.output_size):
                raise RenderLoopConfigError(
                    f'Stable Cascade requires {a_namer("output_size")} to be a power of 2.')

        elif self.s_cascade_decoder_uri:
            raise RenderLoopConfigError(
                f'{a_namer("s_cascade_decoder_uri")} may only be used with "torch-s-cascade"')
        elif self.s_cascade_decoder_inference_steps is not None:
            raise RenderLoopConfigError(
                f'{a_namer("s_cascade_decoder_inference_steps")} may only be used with "torch-s-cascade"')
        elif self.s_cascade_decoder_guidance_scales is not None:
            raise RenderLoopConfigError(
                f'{a_namer("s_cascade_decoder_guidance_scales")} may only be used with "torch-s-cascade"')

        if _pipelinewrapper.model_type_is_flax(self.model_type):
            if not _pipelinewrapper.have_jax_flax():
                raise RenderLoopConfigError(
                    f'Cannot use {a_namer("model_type")} flax because '
                    f'flax and jax are not installed.')

            if self.lora_uris:
                raise RenderLoopConfigError(
                    f'{a_namer("loras")} is not supported for '
                    f'flax, see: {a_namer("model_type")}.')

            if self.guidance_rescales:
                raise RenderLoopConfigError(
                    f'{a_namer("guidance_rescales")} is not supported for '
                    f'flax, see: {a_namer("model_type")}.')

            if self.textual_inversion_uris:
                raise RenderLoopConfigError(
                    f'{a_namer("textual_inversion_uris")} is not supported for '
                    f'flax, see: {a_namer("model_type")}.')

            if self.vae_tiling or self.vae_slicing:
                raise RenderLoopConfigError(
                    f'{a_namer("vae_tiling")} and {a_namer("vae_slicing")} are not '
                    f'supported for flax, see: {a_namer("model_type")}.')

        if self.model_path is None:
            raise RenderLoopConfigError(
                f'{a_namer("model_path")} must be specified')

        if self.frame_end is not None and self.frame_start > self.frame_end:
            raise RenderLoopConfigError(
                f'{a_namer("frame_start")} must be less than or equal to {a_namer("frame_end")}')

        self.animation_format = self.animation_format.strip().lower()
        self.image_format = self.image_format.strip().lower()

        if self.animation_format not in _mediaoutput.supported_animation_writer_formats() + ['frames']:
            raise RenderLoopConfigError(
                f'Unsupported {a_namer("animation_format")} value "{self.image_format}". Must be one of '
                f'{_textprocessing.oxford_comma(_mediaoutput.supported_animation_writer_formats(), "or")}')

        if self.image_format != "png" and self.output_metadata:
            raise RenderLoopConfigError(
                f'{a_namer("image_format")} value "{self.image_format}" is '
                f'unsupported when {a_namer("output_metadata")} is enabled.')

        if self.animation_format == 'frames' and self.no_frames:
            raise RenderLoopConfigError(
                f'Cannot specify {a_namer("no_frames")} when {a_namer("animation_format")} is set to "frames"')

        if self.batch_size is not None:
            if _pipelinewrapper.model_type_is_flax(self.model_type):
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("batch_size")} when using flax, '
                    'use the environmental variable: CUDA_VISIBLE_DEVICES')
        elif not _pipelinewrapper.model_type_is_flax(self.model_type):
            self.batch_size = 1

        if self.clip_skips and not (self.model_type == _pipelinewrapper.ModelType.TORCH or
                                    self.model_type == _pipelinewrapper.ModelType.TORCH_SDXL):
            raise RenderLoopConfigError(f'you cannot specify {a_namer("clip_skips")} for '
                                        f'{a_namer("model_type")} values other than "torch" or "torch-sdxl"')

        if self.output_size is None and not self.image_seeds:
            if _pipelinewrapper.model_type_is_sdxl(self.model_type):
                self.output_size = (_pipelinewrapper.DEFAULT_SDXL_OUTPUT_WIDTH,
                                    _pipelinewrapper.DEFAULT_SDXL_OUTPUT_HEIGHT)
            elif _pipelinewrapper.model_type_is_floyd_if(self.model_type):
                self.output_size = (_pipelinewrapper.DEFAULT_FLOYD_IF_OUTPUT_WIDTH,
                                    _pipelinewrapper.DEFAULT_FLOYD_IF_OUTPUT_HEIGHT)
            elif self.model_type == _pipelinewrapper.ModelType.TORCH_S_CASCADE:
                self.output_size = (_pipelinewrapper.DEFAULT_S_CASCADE_OUTPUT_WIDTH,
                                    _pipelinewrapper.DEFAULT_S_CASCADE_OUTPUT_HEIGHT)
            else:
                self.output_size = (_pipelinewrapper.DEFAULT_OUTPUT_WIDTH,
                                    _pipelinewrapper.DEFAULT_OUTPUT_HEIGHT)

        if not self.image_seeds:
            if _pipelinewrapper.model_type_is_floyd_ifs(self.model_type):
                raise RenderLoopConfigError(
                    f'you cannot specify Deep Floyd IF super-resolution '
                    f'({a_namer("model_type")} "{self.model_type})" without {a_namer("image_seeds")}'
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
            if self.model_type == _pipelinewrapper.ModelType.TORCH_UPSCALER_X4:
                upscaler_noise_levels_default_set = True
                self.upscaler_noise_levels = [_pipelinewrapper.DEFAULT_X4_UPSCALER_NOISE_LEVEL]
        elif self.model_type != _pipelinewrapper.ModelType.TORCH_UPSCALER_X4:
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

        if self.control_image_processors:
            if not self.image_seeds:
                raise RenderLoopConfigError(
                    f'you cannot specify {a_namer("control_image_processors")} '
                    f'without {a_namer("image_seeds")}.')

        if not self.image_seeds:
            invalid_self = []
            for processor_self in non_null_attr_that_end_with('image_processors'):
                invalid_self.append(
                    f'you cannot specify {a_namer(processor_self)} '
                    f'without {a_namer("image_seeds")}.')
            if invalid_self:
                raise RenderLoopConfigError('\n'.join(invalid_self))

        if not _pipelinewrapper.model_type_is_sdxl(self.model_type):
            invalid_self = []
            for sdxl_self in non_null_attr_that_start_with('sdxl'):
                invalid_self.append(f'you cannot specify {a_namer(sdxl_self)} '
                                    f'for a non SDXL model type, see: {a_namer("model_type")}.')
            if invalid_self:
                raise RenderLoopConfigError('\n'.join(invalid_self))

            self.sdxl_high_noise_fractions = None
        else:
            if self.textual_inversion_uris:
                raise RenderLoopConfigError(
                    f'{a_namer("textual_inversion_uris")} is not supported for '
                    f'SDXL, see: {a_namer("model_type")}.')

            if not self.sdxl_refiner_uri:
                invalid_self = []
                for sdxl_self in non_null_attr_that_start_with('sdxl_refiner'):
                    invalid_self.append(f'you cannot specify {a_namer(sdxl_self)} '
                                        f'without {a_namer("sdxl_refiner_uri")}.')
                if invalid_self:
                    raise RenderLoopConfigError('\n'.join(invalid_self))
            elif self.sdxl_high_noise_fractions is None:
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

            parsed_image_seeds = []

            for uri in self.image_seeds:
                parsed_image_seeds.append(
                    self._check_image_seed_uri(
                        uri=uri,
                        attribute_namer=attribute_namer,
                        image_seed_strengths_default_set=image_seed_strengths_default_set,
                        upscaler_noise_levels_default_set=upscaler_noise_levels_default_set))

            self.parsed_image_seeds = parsed_image_seeds

    def _check_image_seed_uri(self,
                              uri,
                              attribute_namer,
                              image_seed_strengths_default_set,
                              upscaler_noise_levels_default_set) -> _mediainput.ImageSeedParseResult:
        """
        :param uri: The URI
        :param attribute_namer: attribute namer
        :param image_seed_strengths_default_set: whether check() has set an image_seed_strengths default value
        :param upscaler_noise_levels_default_set: whether check() has set an upscaler_noise_levels default value
        """

        a_namer = attribute_namer

        try:
            parsed = _mediainput.parse_image_seed_uri(uri)
        except _mediainput.ImageSeedError as e:
            raise RenderLoopConfigError(e)

        mask_part = 'mask=my-mask.png;' if parsed.mask_path else ''
        # ^ Used for nice messages about image seed keyword argument misuse

        if self.control_net_uris:
            control_image_paths = parsed.get_control_image_paths()
            num_control_images = len(control_image_paths) if control_image_paths is not None else 0

            if _pipelinewrapper.model_type_is_flax(self.model_type) and not parsed.is_single_spec:
                raise RenderLoopConfigError(
                    f'img2img and inpainting are not supported for flax when '
                    f'{a_namer("control_net_uris")} is specified.')

            if not parsed.is_single_spec and parsed.control_path is None:
                raise RenderLoopConfigError(
                    f'You must specify a control image with the control argument '
                    f'IE: "my-seed.png;control=my-control.png" in your '
                    f'{a_namer("image_seeds")} "{uri}" when using {a_namer("control_net_uris")} '
                    f'in order to use inpainting. If you want to use the control image alone '
                    f'without a mask, use {a_namer("image_seeds")} "{parsed.seed_path}".')

            if control_image_paths is None:
                raise RenderLoopConfigError(
                    f'You must specify control net guidance images in your {a_namer("image_seeds")} '
                    f'specification "{uri}" (for example: "img2img;{mask_part}control=control1.png, control2.png") '
                    f'when using {a_namer("control_net_uris")}'
                )

            if num_control_images != len(self.control_net_uris):
                raise RenderLoopConfigError(
                    f'Your {a_namer("image_seeds")} specification "{uri}" defines {num_control_images} '
                    f'control guidance image sources, and you have specified {len(self.control_net_uris)} '
                    f'{a_namer("control_net_uris")} URIs. The amount of guidance image sources and the '
                    f'amount of ControlNet models must be equal.'
                )

            if self.control_image_processors:
                control_processor_chain_count = \
                    (sum(1 for p in self.control_image_processors if p == CONTROL_IMAGE_PROCESSOR_SEP) + 1)

                if control_processor_chain_count > num_control_images:
                    raise RenderLoopConfigError(
                        f'Your {a_namer("image_seeds")} specification "{uri}" defines {num_control_images} '
                        f'control guidance image sources, and you have specified {control_processor_chain_count} '
                        f'{a_namer("control_image_processors")} actions / action chains. The amount of processors '
                        f'must not exceed the amount of control guidance images.'
                    )

        is_control_guidance_spec = self.control_net_uris and parsed.is_single_spec

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
                    f'separated list of sources.')

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
                    f'separated list of sources.')

        if self.model_type == _pipelinewrapper.ModelType.TORCH_IFS_IMG2IMG or \
                (parsed.mask_path and _pipelinewrapper.model_type_is_floyd_ifs(self.model_type)):

            if not parsed.floyd_path:
                raise RenderLoopConfigError(
                    f'You must specify a floyd image with the floyd argument '
                    f'IE: "my-seed.png;{mask_part}floyd=previous-stage-image.png" '
                    f'in your {a_namer("image_seeds")} "{uri}" to disambiguate this '
                    f'usage of Deep Floyd IF super-resolution.')

        return parsed

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
            self.s_cascade_decoder_inference_steps,
            self.s_cascade_decoder_guidance_scales,
            self.s_cascade_decoder_prompts
        ]

        product = 1
        for lst in optional_factors:
            product *= max(0 if lst is None else len(lst), 1)

        return (product *
                len(self.prompts) *
                (len(self.seeds) if not self.seeds_to_images else 1) *
                len(self.guidance_scales) *
                len(self.inference_steps))

    def iterate_diffusion_args(self, **overrides) -> collections.abc.Iterator[_pipelinewrapper.DiffusionArguments]:
        """
        Iterate over :py:class:`dgenerate.pipelinewrapper.DiffusionArguments` argument objects using
        every combination of argument values provided for that object by this configuration.

        :param overrides: use key word arguments to override specific attributes of this object with a new list value.
        :return: an iterator over :py:class:`dgenerate.pipelinewrapper.DiffusionArguments`
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

        yield from _iterate_diffusion_args(
            prompt=ov('prompt', self.prompts),
            sdxl_second_prompt=ov('sdxl_second_prompt',
                                  self.sdxl_second_prompts),
            sdxl_refiner_prompt=ov('sdxl_refiner_prompt',
                                   self.sdxl_refiner_prompts),
            sdxl_refiner_second_prompt=ov('sdxl_refiner_second_prompt',
                                          self.sdxl_refiner_second_prompts),
            clip_skip=ov('clip_skip', self.clip_skips),
            sdxl_refiner_clip_skip=ov('sdxl_refiner_clip_skip', self.sdxl_refiner_clip_skips),
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

            s_cascade_decoder_inference_steps=ov('s_cascade_decoder_inference_steps',
                                                 self.s_cascade_decoder_inference_steps),
            s_cascade_decoder_guidance_scale=ov('s_cascade_decoder_guidance_scale',
                                                self.s_cascade_decoder_guidance_scales),
            s_cascade_decoder_prompt=ov('s_cascade_decoder_prompt',
                                        self.s_cascade_decoder_prompts),
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
