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
import decimal
import inspect
import shlex
import typing

import PIL.Image
import diffusers
import torch

import dgenerate.image as _image
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.cache as _cache
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.prompt as _prompt
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

try:
    import jax
    import jaxlib
    import jax.numpy as jnp
    from flax.jax_utils import replicate as _flax_replicate
    from flax.training.common_utils import shard as _flax_shard
except ImportError:
    jaxlib = None
    jnp = None
    _flax_replicate = None
    _flax_shard = None
    jax = None
    flax = None


class OutOfMemoryError(Exception):
    """
    Raised when a GPU or processing device runs out of memory.
    """

    def __init__(self, message):
        super().__init__(f'Device Out Of Memory: {message}')


class PipelineWrapperResult:
    """
    The result of calling :py:class:`.DiffusionPipelineWrapper`
    """
    images: typing.Optional[typing.List[PIL.Image.Image]]

    @property
    def image_count(self):
        """
        The number of images produced.

        :return: int
        """
        return len(self.images)

    @property
    def image(self):
        """
        The first image in the batch of requested batch size.

        :return: :py:class:`PIL.Image.Image`
        """
        return self.images[0] if self.images else None

    def image_grid(self, cols_rows: _types.Size):
        """
        Render an image grid from the images in this result.

        :param cols_rows: columns and rows (WxH) desired as a tuple
        :return: :py:class:`PIL.Image.Image`
        """
        if not self.images:
            raise ValueError('No images present.')

        if len(self.images) == 1:
            return self.images[0]

        cols, rows = cols_rows

        w, h = self.images[0].size
        grid = PIL.Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(self.images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def __init__(self, images: typing.Optional[typing.List[PIL.Image.Image]]):
        self.images = images
        self.dgenerate_opts = list()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.images is not None:
            for i in self.images:
                if i is not None:
                    i.close()
                    self.images = None


class DiffusionArguments:
    """
    Represents all possible arguments for a :py:class:`.DiffusionPipelineWrapper` call.
    """
    prompt: _types.OptionalPrompt = None
    image: typing.Optional[PIL.Image.Image] = None
    mask_image: typing.Optional[PIL.Image.Image] = None
    control_images: typing.Optional[typing.List[PIL.Image.Image]] = None
    width: _types.OptionalSize = None
    height: _types.OptionalSize = None
    batch_size: _types.OptionalInteger = None
    sdxl_second_prompt: _types.OptionalPrompt = None
    sdxl_refiner_prompt: _types.OptionalPrompt = None
    sdxl_refiner_second_prompt: _types.OptionalPrompt = None
    seed: _types.OptionalInteger = None
    image_seed_strength: _types.OptionalFloat = None
    upscaler_noise_level: _types.OptionalInteger = None
    sdxl_high_noise_fraction: _types.OptionalFloat = None
    sdxl_refiner_inference_steps: _types.OptionalInteger = None
    sdxl_refiner_guidance_scale: _types.OptionalFloat = None
    sdxl_refiner_guidance_rescale: _types.OptionalFloat = None
    sdxl_aesthetic_score: _types.OptionalFloat = None
    sdxl_original_size: _types.OptionalSize = None
    sdxl_target_size: _types.OptionalSize = None
    sdxl_crops_coords_top_left: _types.OptionalCoordinate = None
    sdxl_negative_aesthetic_score: _types.OptionalFloat = None
    sdxl_negative_original_size: _types.OptionalSize = None
    sdxl_negative_target_size: _types.OptionalSize = None
    sdxl_negative_crops_coords_top_left: _types.OptionalCoordinate = None
    sdxl_refiner_aesthetic_score: _types.OptionalFloat = None
    sdxl_refiner_original_size: _types.OptionalSize = None
    sdxl_refiner_target_size: _types.OptionalSize = None
    sdxl_refiner_crops_coords_top_left: _types.OptionalCoordinate = None
    sdxl_refiner_negative_aesthetic_score: _types.OptionalFloat = None
    sdxl_refiner_negative_original_size: _types.OptionalSize = None
    sdxl_refiner_negative_target_size: _types.OptionalSize = None
    sdxl_refiner_negative_crops_coords_top_left: _types.OptionalCoordinate = None
    guidance_scale: _types.OptionalFloat = None
    image_guidance_scale: _types.OptionalFloat = None
    guidance_rescale: _types.OptionalFloat = None
    inference_steps: _types.OptionalInteger = None

    def get_pipeline_wrapper_kwargs(self):
        """
        Get the arguments dictionary needed to call :py:class:`.DiffusionPipelineWrapper`

        :return: dictionary of argument names with values
        """
        pipeline_args = {}
        for attr, hint in typing.get_type_hints(self).items():
            val = getattr(self, attr)
            if not attr.startswith('_') and not (callable(val) or val is None):
                pipeline_args[attr] = val
        return pipeline_args

    @staticmethod
    def _describe_prompt(prompt_format, prompt: _prompt.Prompt, pos_title, neg_title):
        if prompt is None:
            return

        prompt_wrap_width = _textprocessing.long_text_wrap_width()
        prompt_val = prompt.positive
        if prompt_val:
            header = f'{pos_title}: '
            prompt_val = \
                _textprocessing.wrap(
                    prompt_val,
                    width=prompt_wrap_width - len(header),
                    subsequent_indent=' ' * len(header))
            prompt_format.append(f'{header}"{prompt_val}"')

        prompt_val = prompt.negative
        if prompt_val:
            header = f'{neg_title}: '
            prompt_val = \
                _textprocessing.wrap(
                    prompt_val,
                    width=prompt_wrap_width - len(header),
                    subsequent_indent=' ' * len(header))
            prompt_format.append(f'{header}"{prompt_val}"')

    def describe_pipeline_wrapper_args(self) -> str:
        """
        Describe the pipeline wrapper arguments in a pretty, human-readable way, with word wrapping
        depending on console size or a maximum length depending on what stdout currently is.

        :return: description string.
        """
        prompt_format = []
        DiffusionArguments._describe_prompt(
            prompt_format, self.prompt,
            "Prompt",
            "Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sdxl_second_prompt,
            "SDXL Second Prompt",
            "SDXL Second Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sdxl_refiner_prompt,
            "SDXL Refiner Prompt",
            "SDXL Refiner Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sdxl_refiner_second_prompt,
            "SDXL Refiner Second Prompt",
            "SDXL Refiner Second Negative Prompt")

        prompt_format = '\n'.join(prompt_format)
        if prompt_format:
            prompt_format = '\n' + prompt_format

        inputs = [f'Seed: {self.seed}']

        descriptions = [
            (self.image_seed_strength, "Image Seed Strength:"),
            (self.upscaler_noise_level, "Upscaler Noise Level:"),
            (self.sdxl_high_noise_fraction, "SDXL High Noise Fraction:"),
            (self.sdxl_refiner_inference_steps, "SDXL Refiner Inference Steps:"),
            (self.sdxl_refiner_guidance_scale, "SDXL Refiner Guidance Scale:"),
            (self.sdxl_refiner_guidance_rescale, "SDXL Refiner Guidance Rescale:"),
            (self.sdxl_aesthetic_score, "SDXL Aesthetic Score:"),
            (self.sdxl_original_size, "SDXL Original Size:"),
            (self.sdxl_target_size, "SDXL Target Size:"),
            (self.sdxl_crops_coords_top_left, "SDXL Top Left Crop Coords:"),
            (self.sdxl_negative_aesthetic_score, "SDXL Negative Aesthetic Score:"),
            (self.sdxl_negative_original_size, "SDXL Negative Original Size:"),
            (self.sdxl_negative_target_size, "SDXL Negative Target Size:"),
            (self.sdxl_negative_crops_coords_top_left, "SDXL Negative Top Left Crop Coords:"),
            (self.sdxl_refiner_aesthetic_score, "SDXL Refiner Aesthetic Score:"),
            (self.sdxl_refiner_original_size, "SDXL Refiner Original Size:"),
            (self.sdxl_refiner_target_size, "SDXL Refiner Target Size:"),
            (self.sdxl_refiner_crops_coords_top_left, "SDXL Refiner Top Left Crop Coords:"),
            (self.sdxl_refiner_negative_aesthetic_score, "SDXL Refiner Negative Aesthetic Score:"),
            (self.sdxl_refiner_negative_original_size, "SDXL Refiner Negative Original Size:"),
            (self.sdxl_refiner_negative_target_size, "SDXL Refiner Negative Target Size:"),
            (self.sdxl_refiner_negative_crops_coords_top_left, "SDXL Refiner Negative Top Left Crop Coords:"),
            (self.guidance_scale, "Guidance Scale:"),
            (self.image_guidance_scale, "Image Guidance Scale:"),
            (self.guidance_rescale, "Guidance Rescale:"),
            (self.inference_steps, "Inference Steps:")
        ]

        for prompt_val, desc in descriptions:
            if prompt_val is not None:
                inputs.append(desc + ' ' + str(prompt_val))

        inputs = '\n'.join(inputs)

        return inputs + prompt_format


class DiffusionPipelineWrapper:
    """
    Monolithic diffusion pipelines wrapper.
    """

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def __init__(self,
                 model_path: _types.Path,
                 dtype: typing.Union[_enums.DataTypes, str] = _enums.DataTypes.AUTO,
                 device: str = 'cuda',
                 model_type: typing.Union[_enums.ModelTypes, str] = _enums.ModelTypes.TORCH,
                 revision: _types.OptionalName = None,
                 variant: _types.OptionalName = None,
                 model_subfolder: _types.OptionalName = None,
                 vae_uri: _types.OptionalUri = None,
                 vae_tiling: bool = False,
                 vae_slicing: bool = False,
                 lora_uris: typing.Union[str, _types.OptionalUris] = None,
                 textual_inversion_uris: typing.Union[str, _types.OptionalUris] = None,
                 control_net_uris: typing.Union[str, _types.OptionalUris] = None,
                 sdxl_refiner_uri: _types.OptionalUri = None,
                 scheduler: _types.OptionalName = None,
                 sdxl_refiner_scheduler: _types.OptionalName = None,
                 safety_checker: bool = False,
                 auth_token: _types.OptionalString = None,
                 local_files_only: bool = False):

        self._model_subfolder = model_subfolder
        self._device = device
        self._model_type = _enums.get_model_type_enum(model_type)
        self._model_path = model_path
        self._pipeline = None
        self._flax_params = None
        self._revision = revision
        self._variant = variant
        self._dtype = _enums.get_data_type_enum(dtype)
        self._device = device
        self._vae_uri = vae_uri
        self._vae_tiling = vae_tiling
        self._vae_slicing = vae_slicing
        self._safety_checker = safety_checker
        self._scheduler = scheduler
        self._sdxl_refiner_scheduler = sdxl_refiner_scheduler
        self._lora_uris = lora_uris
        self._lora_scale = None
        self._textual_inversion_uris = textual_inversion_uris
        self._control_net_uris = control_net_uris
        self._parsed_control_net_uris = []
        self._sdxl_refiner_pipeline = None
        self._auth_token = auth_token
        self._pipeline_type = None
        self._local_files_only = local_files_only

        self._parsed_sdxl_refiner_uri = None
        self._sdxl_refiner_uri = sdxl_refiner_uri
        if sdxl_refiner_uri is not None:
            # up front validation of this URI is optimal
            self._parsed_sdxl_refiner_uri = _uris.parse_sdxl_refiner_uri(sdxl_refiner_uri)

        if lora_uris:
            if model_type == 'flax':
                raise NotImplementedError('LoRA loading is not implemented for flax.')

            if not isinstance(lora_uris, str):
                raise NotImplementedError('Using multiple LoRA models is currently not supported.')

            self._lora_scale = _uris.parse_lora_uri(lora_uris).scale

    @staticmethod
    def _pipeline_to(pipeline, device):
        if hasattr(pipeline, 'to'):
            if not pipeline.DGENERATE_CPU_OFFLOAD and \
                    not pipeline.DGENERATE_SEQUENTIAL_OFFLOAD:

                if device == 'cpu':
                    _cache.pipeline_to_cpu_update_cache_info(pipeline)
                else:
                    _cache.pipeline_off_cpu_update_cache_info(pipeline)
                try:
                    return pipeline.to(device)
                except RuntimeError as e:
                    if 'memory' in str(e).lower():
                        raise OutOfMemoryError(e)
                    raise e
            else:
                return pipeline
        return pipeline

    _LAST_CALLED_PIPE = None

    @staticmethod
    def _call_pipeline(pipeline, device, **kwargs):
        _messages.debug_log(f'Calling Pipeline: "{pipeline.__class__.__name__}",',
                            f'Device: "{device}",',
                            'Args:',
                            lambda: _textprocessing.debug_format_args(kwargs,
                                                                      value_transformer=lambda key, value:
                                                                      f'torch.Generator(seed={value.initial_seed()})'
                                                                      if isinstance(value, torch.Generator) else value))

        if pipeline is DiffusionPipelineWrapper._LAST_CALLED_PIPE:
            return pipeline(**kwargs)
        else:
            DiffusionPipelineWrapper._pipeline_to(
                DiffusionPipelineWrapper._LAST_CALLED_PIPE, 'cpu')

        DiffusionPipelineWrapper._pipeline_to(pipeline, device)
        r = pipeline(**kwargs)

        DiffusionPipelineWrapper._LAST_CALLED_PIPE = pipeline
        return r

    @property
    def local_files_only(self) -> bool:
        """
        Currently set value for **local_files_only**

        :return:
        """
        return self._local_files_only

    @property
    def revision(self) -> _types.OptionalName:
        """
        Currently set revision for the main model or None
        """
        return self._revision

    @property
    def safety_checker(self) -> bool:
        """
        Safety checker enabled status
        """
        return self._safety_checker

    @property
    def variant(self) -> _types.OptionalName:
        """
        Currently set variant for the main model or None
        """
        return self._variant

    @property
    def dtype(self) -> _enums.DataTypes:
        """
        Currently set dtype for the main model
        """
        return self._dtype

    @property
    def textual_inversion_uris(self) -> _types.OptionalUris:
        """
        List of supplied --textual-inversions uri strings or None
        """
        return [self._textual_inversion_uris] if \
            isinstance(self._textual_inversion_uris, str) else self._textual_inversion_uris

    @property
    def control_net_uris(self) -> _types.OptionalUris:
        """
        List of supplied --control-nets uri strings or None
        """
        return [self._control_net_uris] if \
            isinstance(self._control_net_uris, str) else self._control_net_uris

    @property
    def device(self) -> _types.Name:
        """
        Currently set --device string
        """
        return self._device

    @property
    def model_path(self) -> _types.Path:
        """
        Model path for the main model
        """
        return self._model_path

    @property
    def scheduler(self) -> _types.OptionalName:
        """
        Selected scheduler name for the main model or None
        """
        return self._scheduler

    @property
    def sdxl_refiner_scheduler(self) -> _types.OptionalName:
        """
        Selected scheduler name for the SDXL refiner or None
        """
        return self._sdxl_refiner_scheduler

    @property
    def sdxl_refiner_uri(self) -> _types.OptionalUri:
        """
        Model path for the SDXL refiner or None
        """
        return self._sdxl_refiner_uri

    @property
    def model_type_enum(self) -> _enums.ModelTypes:
        """
        Currently set ``--model-type`` enum value
        """
        return self._model_type

    @property
    def model_type_string(self) -> str:
        """
        Currently set ``--model-type`` string value
        """
        return _enums.get_model_type_string(self._model_type)

    @property
    def dtype_enum(self) -> _enums.DataTypes:
        """
        Currently set --dtype enum value
        """
        return self._dtype

    @property
    def dtype_string(self) -> str:
        """
        Currently set --dtype string value
        """
        return _enums.get_data_type_string(self._dtype)

    @property
    def model_subfolder(self) -> _types.OptionalName:
        """
        Selected model subfolder for the main model, (remote repo subfolder or local) or None
        """
        return self._model_subfolder

    @property
    def vae_uri(self) -> _types.OptionalUri:
        """
        Selected --vae uri for the main model or None
        """
        return self._vae_uri

    @property
    def vae_tiling(self) -> bool:
        """
        Current --vae-tiling status
        """
        return self._vae_tiling

    @property
    def vae_slicing(self) -> bool:
        """
        Current --vae-slicing status
        """
        return self._vae_slicing

    @property
    def lora_uris(self) -> _types.OptionalUris:
        """
        List of supplied --lora uri strings or None
        """
        return [self._lora_uris] if \
            isinstance(self._lora_uris, str) else self._lora_uris

    @property
    def auth_token(self) -> _types.OptionalString:
        """
        Current --auth-token value or None
        """
        return self._auth_token

    def reconstruct_dgenerate_opts(self,
                                   args: typing.Optional[DiffusionArguments] = None,
                                   extra_opts: typing.Optional[
                                       typing.List[
                                           typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]] = None,
                                   shell_quote=True,
                                   **kwargs) -> \
            typing.List[typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]:
        """
        Reconstruct dgenerates command line arguments from a particular set of pipeline wrapper call arguments.

        :param args: :py:class:`.DiffusionArguments` object to take values from

        :param extra_opts: Extra option pairs to be added to the end of reconstructed options,
            this should be a list of tuples of length 1 (switch only) or length 2 (switch with args)

        :param shell_quote: Shell quote and format the argument values? or return them raw.

        :param kwargs: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`

        :return: List of tuples of length 1 or 2 representing the option
        """

        if args is not None:
            kwargs = args.get_pipeline_wrapper_kwargs() | (kwargs if kwargs else dict())

        def _format_size(val):
            if val is None:
                return None

            return f'{val[0]}x{val[1]}'

        batch_size: int = kwargs.get('batch_size', None)
        prompt: _prompt.Prompt = kwargs.get('prompt', None)
        sdxl_second_prompt: _prompt.Prompt = kwargs.get('sdxl_second_prompt', None)
        sdxl_refiner_prompt: _prompt.Prompt = kwargs.get('sdxl_refiner_prompt', None)
        sdxl_refiner_second_prompt: _prompt.Prompt = kwargs.get('sdxl_refiner_second_prompt', None)

        image = kwargs.get('image', None)
        control_images = kwargs.get('control_images', None)
        image_seed_strength = kwargs.get('image_seed_strength', None)
        upscaler_noise_level = kwargs.get('upscaler_noise_level', None)
        mask_image = kwargs.get('mask_image', None)
        seed = kwargs.get('seed')
        width = kwargs.get('width', None)
        height = kwargs.get('height', None)
        inference_steps = kwargs.get('inference_steps')
        guidance_scale = kwargs.get('guidance_scale')
        guidance_rescale = kwargs.get('guidance_rescale')
        image_guidance_scale = kwargs.get('image_guidance_scale')

        sdxl_refiner_inference_steps = kwargs.get('sdxl_refiner_inference_steps')
        sdxl_refiner_guidance_scale = kwargs.get('sdxl_refiner_guidance_scale')
        sdxl_refiner_guidance_rescale = kwargs.get('sdxl_refiner_guidance_rescale')

        sdxl_high_noise_fraction = kwargs.get('sdxl_high_noise_fraction', None)
        sdxl_aesthetic_score = kwargs.get('sdxl_aesthetic_score', None)

        sdxl_original_size = \
            _format_size(kwargs.get('sdxl_original_size', None))
        sdxl_target_size = \
            _format_size(kwargs.get('sdxl_target_size', None))
        sdxl_crops_coords_top_left = \
            _format_size(kwargs.get('sdxl_crops_coords_top_left', None))

        sdxl_negative_aesthetic_score = kwargs.get('sdxl_negative_aesthetic_score', None)

        sdxl_negative_original_size = \
            _format_size(kwargs.get('sdxl_negative_original_size', None))
        sdxl_negative_target_size = \
            _format_size(kwargs.get('sdxl_negative_target_size', None))
        sdxl_negative_crops_coords_top_left = \
            _format_size(kwargs.get('sdxl_negative_crops_coords_top_left', None))

        sdxl_refiner_aesthetic_score = kwargs.get('sdxl_refiner_aesthetic_score', None)

        sdxl_refiner_original_size = \
            _format_size(kwargs.get('sdxl_refiner_original_size', None))
        sdxl_refiner_target_size = \
            _format_size(kwargs.get('sdxl_refiner_target_size', None))
        sdxl_refiner_crops_coords_top_left = \
            _format_size(kwargs.get('sdxl_refiner_crops_coords_top_left', None))

        sdxl_refiner_negative_aesthetic_score = kwargs.get('sdxl_refiner_negative_aesthetic_score', None)

        sdxl_refiner_negative_original_size = \
            _format_size(kwargs.get('sdxl_refiner_negative_original_size', None))
        sdxl_refiner_negative_target_size = \
            _format_size(kwargs.get('sdxl_refiner_negative_target_size', None))
        sdxl_refiner_negative_crops_coords_top_left = \
            _format_size(kwargs.get('sdxl_refiner_negative_crops_coords_top_left', None))

        opts = [(self.model_path,),
                ('--model-type', self.model_type_string),
                ('--dtype', self.dtype_string),
                ('--device', self._device),
                ('--inference-steps', inference_steps),
                ('--guidance-scales', guidance_scale),
                ('--seeds', seed)]

        if batch_size is not None and batch_size > 1:
            opts.append(('--batch-size', batch_size))

        if guidance_rescale is not None:
            opts.append(('--guidance-rescales', guidance_rescale))

        if image_guidance_scale is not None:
            opts.append(('--image-guidance-scales', image_guidance_scale))

        if prompt is not None:
            opts.append(('--prompts', prompt))

        if sdxl_second_prompt is not None:
            opts.append(('--sdxl-second-prompt', sdxl_second_prompt))

        if sdxl_refiner_prompt is not None:
            opts.append(('--sdxl-refiner-prompt', sdxl_refiner_prompt))

        if sdxl_refiner_second_prompt is not None:
            opts.append(('--sdxl-refiner-second-prompt', sdxl_refiner_second_prompt))

        if self._revision is not None:
            opts.append(('--revision', self._revision))

        if self._variant is not None:
            opts.append(('--variant', self._variant))

        if self._model_subfolder is not None:
            opts.append(('--subfolder', self._model_subfolder))

        if self._vae_uri is not None:
            opts.append(('--vae', self._vae_uri))

        if self._vae_tiling:
            opts.append(('--vae-tiling',))

        if self._vae_slicing:
            opts.append(('--vae-slicing',))

        if self._sdxl_refiner_uri is not None:
            opts.append(('--sdxl-refiner', self._sdxl_refiner_uri))

        if self._lora_uris:
            opts.append(('--lora', self._lora_uris))

        if self._textual_inversion_uris:
            opts.append(('--textual-inversions', self._textual_inversion_uris))

        if self._control_net_uris:
            opts.append(('--control-nets', self._control_net_uris))

        if self._scheduler is not None:
            opts.append(('--scheduler', self._scheduler))

        if self._sdxl_refiner_scheduler is not None:
            if self._sdxl_refiner_scheduler != self._scheduler:
                opts.append(('--sdxl-refiner-scheduler', self._sdxl_refiner_scheduler))

        if sdxl_high_noise_fraction is not None:
            opts.append(('--sdxl-high-noise-fractions', sdxl_high_noise_fraction))

        if sdxl_refiner_inference_steps is not None:
            opts.append(('--sdxl-refiner-inference-steps', sdxl_refiner_inference_steps))

        if sdxl_refiner_guidance_scale is not None:
            opts.append(('--sdxl-refiner-guidance-scales', sdxl_refiner_guidance_scale))

        if sdxl_refiner_guidance_rescale is not None:
            opts.append(('--sdxl-refiner-guidance-rescales', sdxl_refiner_guidance_rescale))

        if sdxl_aesthetic_score is not None:
            opts.append(('--sdxl-aesthetic-scores', sdxl_aesthetic_score))

        if sdxl_original_size is not None:
            opts.append(('--sdxl-original-size', sdxl_original_size))

        if sdxl_target_size is not None:
            opts.append(('--sdxl-target-size', sdxl_target_size))

        if sdxl_crops_coords_top_left is not None:
            opts.append(('--sdxl-crops-coords-top-left', sdxl_crops_coords_top_left))

        if sdxl_negative_aesthetic_score is not None:
            opts.append(('--sdxl-negative-aesthetic-scores', sdxl_negative_aesthetic_score))

        if sdxl_negative_original_size is not None:
            opts.append(('--sdxl-negative-original-sizes', sdxl_negative_original_size))

        if sdxl_negative_target_size is not None:
            opts.append(('--sdxl-negative-target-sizes', sdxl_negative_target_size))

        if sdxl_negative_crops_coords_top_left is not None:
            opts.append(('--sdxl-negative-crops-coords-top-left', sdxl_negative_crops_coords_top_left))

        if sdxl_refiner_aesthetic_score is not None:
            opts.append(('--sdxl-refiner-aesthetic-scores', sdxl_refiner_aesthetic_score))

        if sdxl_refiner_original_size is not None:
            opts.append(('--sdxl-refiner-original-sizes', sdxl_refiner_original_size))

        if sdxl_refiner_target_size is not None:
            opts.append(('--sdxl-refiner-target-sizes', sdxl_refiner_target_size))

        if sdxl_refiner_crops_coords_top_left is not None:
            opts.append(('--sdxl-refiner-crops-coords-top-left', sdxl_refiner_crops_coords_top_left))

        if sdxl_refiner_negative_aesthetic_score is not None:
            opts.append(('--sdxl-refiner-negative-aesthetic-scores', sdxl_refiner_negative_aesthetic_score))

        if sdxl_refiner_negative_original_size is not None:
            opts.append(('--sdxl-refiner-negative-original-sizes', sdxl_refiner_negative_original_size))

        if sdxl_refiner_negative_target_size is not None:
            opts.append(('--sdxl-refiner-negative-target-sizes', sdxl_refiner_negative_target_size))

        if sdxl_refiner_negative_crops_coords_top_left is not None:
            opts.append(('--sdxl-refiner-negative-crops-coords-top-left', sdxl_refiner_negative_crops_coords_top_left))

        if width is not None and height is not None:
            opts.append(('--output-size', f'{width}x{height}'))
        elif width is not None:
            opts.append(('--output-size', f'{width}'))

        if image is not None:
            seed_args = []

            if mask_image is not None:
                seed_args.append(f'mask={_image.get_filename(mask_image)}')
            if control_images:
                seed_args.append(f'control={", ".join(_image.get_filename(c) for c in control_images)}')

            if isinstance(image, list):
                opts.append(('--image-seeds',
                             ','.join(_image.get_filename(i) for i in image)))
            elif image:
                if not seed_args:
                    opts.append(('--image-seeds',
                                 _image.get_filename(image)))
                else:
                    opts.append(('--image-seeds',
                                 _image.get_filename(image) + ';' + ';'.join(seed_args)))

            if upscaler_noise_level is not None:
                opts.append(('--upscaler-noise-levels', upscaler_noise_level))

            if image_seed_strength is not None:
                opts.append(('--image-seed-strengths', image_seed_strength))

        elif control_images:
            opts.append(('--image-seeds',
                         ', '.join(_image.get_filename(c) for c in control_images)))

        if extra_opts:
            opts += extra_opts

        if shell_quote:
            for idx, option in enumerate(opts):
                if len(option) > 1:
                    name, value = option
                    if isinstance(value, (str, _prompt.Prompt)):
                        opts[idx] = (name, shlex.quote(str(value)))
                    elif isinstance(value, tuple):
                        opts[idx] = (name, _textprocessing.format_size(value))
                    else:
                        opts[idx] = (name, str(value))
                else:
                    solo_val = str(option[0])
                    if not solo_val.startswith('-'):
                        # not a solo switch option, some value
                        opts[idx] = (shlex.quote(solo_val),)

        return opts

    @staticmethod
    def _set_opt_value_syntax(val):
        if isinstance(val, tuple):
            return _textprocessing.format_size(val)
        if isinstance(val, list):
            return ' '.join(DiffusionPipelineWrapper._set_opt_value_syntax(v) for v in val)
        return shlex.quote(str(val))

    @staticmethod
    def _format_option_pair(val):
        if len(val) > 1:
            opt_name, opt_value = val

            if isinstance(opt_value, _prompt.Prompt):
                header_len = len(opt_name) + 2
                prompt_text = \
                    _textprocessing.wrap(
                        shlex.quote(str(opt_value)),
                        subsequent_indent=' ' * header_len,
                        width=75)

                prompt_text = ' \\\n'.join(prompt_text.split('\n'))
                return f'{opt_name} {prompt_text}'

            return f'{opt_name} {DiffusionPipelineWrapper._set_opt_value_syntax(opt_value)}'

        solo_val = str(val[0])

        if solo_val.startswith('-'):
            return solo_val

        # Not a switch option, some value
        return shlex.quote(solo_val)

    def gen_dgenerate_config(self,
                             args: typing.Optional[DiffusionArguments] = None,
                             extra_opts: typing.Optional[
                                 typing.List[typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]] = None,
                             extra_comments: typing.Optional[typing.Sequence[str]] = None,
                             **kwargs):
        """
        Generate a valid dgenerate config file with a single invocation that reproduces this result.

        :param args: :py:class:`.DiffusionArguments` object to take values from
        :param extra_comments: Extra strings to use as comments after the initial
            version check directive
        :param extra_opts: Extra option pairs to be added to the end of reconstructed options
            of the dgenerate invocation, this should be a list of tuples of length 1 (switch only)
            or length 2 (switch with args)
        :param kwargs: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
        :return: The configuration as a string
        """

        from dgenerate import __version__

        config = f'#! dgenerate {__version__}\n\n'

        if extra_comments:
            wrote_comments = False
            for comment in extra_comments:
                wrote_comments = True
                for part in comment.split('\n'):
                    config += '# ' + part.rstrip()

            if wrote_comments:
                config += '\n\n'

        opts = \
            self.reconstruct_dgenerate_opts(args, **kwargs, shell_quote=False) + \
            (extra_opts if extra_opts else [])

        for opt in opts[:-1]:
            config += f'{self._format_option_pair(opt)} \\\n'

        last = opts[-1]

        if len(last) == 2:
            config += self._format_option_pair(last)

        return config

    def gen_dgenerate_command(self,
                              args: typing.Optional[DiffusionArguments] = None,
                              extra_opts: typing.Optional[
                                  typing.List[typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]] = None,
                              **kwargs):
        """
        Generate a valid dgenerate command line invocation that reproduces this result.

        :param args: :py:class:`.DiffusionArguments` object to take values from
        :param extra_opts: Extra option pairs to be added to the end of reconstructed options
            of the dgenerate invocation, this should be a list of tuples of length 1 (switch only)
            or length 2 (switch with args)
        :param kwargs: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
        :return: A string containing the dgenerate command line needed to reproduce this result.
        """

        opt_string = \
            ' '.join(f"{self._format_option_pair(opt)}"
                     for opt in self.reconstruct_dgenerate_opts(args, **kwargs,
                                                                shell_quote=False) + extra_opts)

        return f'dgenerate {opt_string}'

    def _pipeline_defaults(self, user_args):
        args = dict()
        args['guidance_scale'] = float(user_args.get('guidance_scale', _constants.DEFAULT_GUIDANCE_SCALE))
        args['num_inference_steps'] = int(user_args.get('inference_steps', _constants.DEFAULT_INFERENCE_STEPS))

        def set_strength():
            strength = float(user_args.get('image_seed_strength', _constants.DEFAULT_IMAGE_SEED_STRENGTH))
            inference_steps = args.get('num_inference_steps')

            if (strength * inference_steps) < 1.0:
                strength = 1.0 / inference_steps
                _messages.log(
                    f'image-seed-strength * inference-steps '
                    f'was calculated at < 1, image-seed-strength defaulting to (1.0 / inference-steps): {strength}',
                    level=_messages.WARNING)

            args['strength'] = strength

        if self._control_net_uris:
            control_images = user_args.get('control_images')

            if not control_images:
                raise ValueError(
                    'Must provide control_images argument when using ControlNet models.')

            control_images_cnt = len(control_images)
            control_net_uris_cnt = len(self._control_net_uris)

            if control_images_cnt < control_net_uris_cnt:
                # Pad it out so that the last image mentioned is used
                # for the rest of the controlnets specified

                for i in range(0, control_net_uris_cnt - control_images_cnt):
                    control_images.append(control_images[-1])

            elif control_images_cnt > control_net_uris_cnt:
                # User provided too many control_images, behavior is undefined.

                raise ValueError(
                    f'You specified {control_images_cnt} control image sources and '
                    f'only {control_net_uris_cnt} ControlNet URIs. The amount of '
                    f'control images must be less than or equal to the amount of ControlNet URIs.')

            # They should always be of equal dimension, anything
            # else results in an error down the line.
            args['width'] = user_args.get('width', control_images[0].width)
            args['height'] = user_args.get('height', control_images[0].height)

            if self._pipeline_type == _enums.PipelineTypes.TXT2IMG:
                args['image'] = control_images
            elif self._pipeline_type == _enums.PipelineTypes.IMG2IMG or \
                    self._pipeline_type == _enums.PipelineTypes.INPAINT:
                args['image'] = user_args['image']
                args['control_image'] = control_images
                set_strength()

            mask_image = user_args.get('mask_image')
            if mask_image is not None:
                args['mask_image'] = mask_image

        elif 'image' in user_args:
            image = user_args['image']
            args['image'] = image

            if _enums.model_type_is_upscaler(self._model_type):
                if self._model_type == _enums.ModelTypes.TORCH_UPSCALER_X4:
                    args['noise_level'] = int(
                        user_args.get('upscaler_noise_level', _constants.DEFAULT_X4_UPSCALER_NOISE_LEVEL))
            elif not _enums.model_type_is_pix2pix(self._model_type):
                set_strength()

            mask_image = user_args.get('mask_image')
            if mask_image is not None:
                args['mask_image'] = mask_image
                args['width'] = image.size[0]
                args['height'] = image.size[1]

            if self._model_type == _enums.ModelTypes.TORCH_SDXL_PIX2PIX:
                # Required
                args['width'] = image.size[0]
                args['height'] = image.size[1]
        else:
            args['height'] = user_args.get('height', _constants.DEFAULT_OUTPUT_HEIGHT)
            args['width'] = user_args.get('width', _constants.DEFAULT_OUTPUT_WIDTH)

        if self._lora_scale is not None:
            args['cross_attention_kwargs'] = {'scale': self._lora_scale}

        return args

    def _get_control_net_conditioning_scale(self):
        if not self._parsed_control_net_uris:
            return 1.0
        return [p.scale for p in self._parsed_control_net_uris] if \
            len(self._parsed_control_net_uris) > 1 else self._parsed_control_net_uris[0].scale

    def _get_control_net_guidance_start(self):
        if not self._parsed_control_net_uris:
            return 0.0
        return [p.start for p in self._parsed_control_net_uris] if \
            len(self._parsed_control_net_uris) > 1 else self._parsed_control_net_uris[0].start

    def _get_control_net_guidance_end(self):
        if not self._parsed_control_net_uris:
            return 1.0
        return [p.end for p in self._parsed_control_net_uris] if \
            len(self._parsed_control_net_uris) > 1 else self._parsed_control_net_uris[0].end

    def _call_flax_control_net(self, positive_prompt, negative_prompt, default_args, user_args):
        # Only works with txt2image

        device_count = jax.device_count()

        pipe: diffusers.FlaxStableDiffusionControlNetPipeline = self._pipeline

        default_args['prng_seed'] = jax.random.split(jax.random.PRNGKey(user_args.get('seed', 0)),
                                                     device_count)
        prompt_ids = pipe.prepare_text_inputs([positive_prompt] * device_count)

        if negative_prompt is not None:
            negative_prompt_ids = pipe.prepare_text_inputs([negative_prompt] * device_count)
        else:
            negative_prompt_ids = None

        control_net_image = default_args.get('image')
        if isinstance(control_net_image, list):
            control_net_image = control_net_image[0]

        processed_image = pipe.prepare_image_inputs([control_net_image] * device_count)
        default_args.pop('image')

        p_params = _flax_replicate(self._flax_params)
        prompt_ids = _flax_shard(prompt_ids)
        negative_prompt_ids = _flax_shard(negative_prompt_ids)
        processed_image = _flax_shard(processed_image)

        default_args.pop('width', None)
        default_args.pop('height', None)

        images = DiffusionPipelineWrapper._call_pipeline(
            pipeline=self._pipeline,
            device=self.device,
            prompt_ids=prompt_ids,
            image=processed_image,
            params=p_params,
            neg_prompt_ids=negative_prompt_ids,
            controlnet_conditioning_scale=self._get_control_net_conditioning_scale(),
            jit=True, **default_args)[0]

        return PipelineWrapperResult(
            self._pipeline.numpy_to_pil(images.reshape((images.shape[0],) + images.shape[-3:])))

    def _flax_prepare_text_input(self, text):
        tokenizer = self._pipeline.tokenizer
        text_input = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids

    def _call_flax(self, default_args, user_args):
        for arg, val in user_args.items():
            if arg.startswith('sdxl') and val is not None:
                raise NotImplementedError(
                    f'{arg.replace("_", "-")}s may only be used with SDXL models.')

        if user_args.get('guidance_rescale') is not None:
            raise NotImplementedError('--guidance-rescales is not supported when using --model-type flax.')

        prompt: _prompt.Prompt() = user_args.get('prompt', _prompt.Prompt())
        positive_prompt = prompt.positive if prompt.positive else ''
        negative_prompt = prompt.negative

        if hasattr(self._pipeline, 'controlnet'):
            return self._call_flax_control_net(positive_prompt, negative_prompt,
                                               default_args, user_args)

        device_count = jax.device_count()

        default_args['prng_seed'] = jax.random.split(jax.random.PRNGKey(user_args.get('seed', 0)),
                                                     device_count)

        if negative_prompt is not None:
            negative_prompt_ids = _flax_shard(
                self._flax_prepare_text_input([negative_prompt] * device_count))
        else:
            negative_prompt_ids = None

        if 'image' in default_args:
            if 'mask_image' in default_args:

                prompt_ids, processed_images, processed_masks = \
                    self._pipeline.prepare_inputs(prompt=[positive_prompt] * device_count,
                                                  image=[default_args['image']] * device_count,
                                                  mask=[default_args['mask_image']] * device_count)

                default_args['masked_image'] = _flax_shard(processed_images)
                default_args['mask'] = _flax_shard(processed_masks)

                # inpainting pipeline does not have a strength argument, simply ignore it
                default_args.pop('strength')

                default_args.pop('image')
                default_args.pop('mask_image')
            else:
                prompt_ids, processed_images = self._pipeline.prepare_inputs(
                    prompt=[positive_prompt] * device_count,
                    image=[default_args['image']] * device_count)
                default_args['image'] = _flax_shard(processed_images)

            default_args['width'] = processed_images[0].shape[2]
            default_args['height'] = processed_images[0].shape[1]
        else:
            prompt_ids = self._pipeline.prepare_inputs([positive_prompt] * device_count)

        images = DiffusionPipelineWrapper._call_pipeline(
            pipeline=self._pipeline,
            device=self._device,
            prompt_ids=_flax_shard(prompt_ids),
            neg_prompt_ids=negative_prompt_ids,
            params=_flax_replicate(self._flax_params),
            **default_args, jit=True)[0]

        return PipelineWrapperResult(self._pipeline.numpy_to_pil(
            images.reshape((images.shape[0],) + images.shape[-3:])))

    def _get_non_universal_pipeline_arg(self,
                                        pipeline,
                                        default_args,
                                        user_args,
                                        pipeline_arg_name,
                                        user_arg_name,
                                        option_name,
                                        default,
                                        transform=None):
        if pipeline.__call__.__wrapped__ is not None:
            # torch.no_grad()
            func = pipeline.__call__.__wrapped__
        else:
            func = pipeline.__call__

        if pipeline_arg_name in inspect.getfullargspec(func).args:
            if user_arg_name in user_args:
                # Only provide a default if the user
                # provided the option, and it's value was None
                val = user_args.get(user_arg_name, default)
                val = val if not transform else transform(val)
                default_args[pipeline_arg_name] = val
                return val
            return None
        else:
            val = user_args.get(user_arg_name, None)
            if val is not None:
                raise NotImplementedError(
                    f'{option_name} cannot be used with --model-type "{self.model_type_string}" in '
                    f'{_enums.get_pipeline_type_string(self._pipeline_type)} mode with the current '
                    f'combination of arguments and model.')
            return None

    def _get_sdxl_conditioning_args(self, pipeline, default_args, user_args, user_prefix=None):
        if user_prefix:
            user_prefix += '_'
            option_prefix = _textprocessing.dashup(user_prefix)
        else:
            user_prefix = ''
            option_prefix = ''

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'aesthetic_score', f'sdxl_{user_prefix}aesthetic_score',
                                             f'--sdxl-{option_prefix}aesthetic-scores', None)
        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'original_size', f'sdxl_{user_prefix}original_size',
                                             f'--sdxl-{option_prefix}original-sizes', None)
        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'target_size', f'sdxl_{user_prefix}target_size',
                                             f'--sdxl-{option_prefix}target-sizes', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'crops_coords_top_left',
                                             f'sdxl_{user_prefix}crops_coords_top_left',
                                             f'--sdxl-{option_prefix}crops-coords-top-left', (0, 0))

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_aesthetic_score',
                                             f'sdxl_{user_prefix}negative_aesthetic_score',
                                             f'--sdxl-{option_prefix}negative-aesthetic-scores', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_original_size',
                                             f'sdxl_{user_prefix}negative_original_size',
                                             f'--sdxl-{option_prefix}negative-original-sizes', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_target_size',
                                             f'sdxl_{user_prefix}negative_target_size',
                                             f'--sdxl-{option_prefix}negative-target-sizes', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_crops_coords_top_left',
                                             f'sdxl_{user_prefix}negative_crops_coords_top_left',
                                             f'--sdxl-{option_prefix}negative-crops-coords-top-left', (0, 0))

    @staticmethod
    def _pop_sdxl_conditioning_args(default_args):
        default_args.pop('aesthetic_score', None)
        default_args.pop('target_size', None)
        default_args.pop('original_size', None)
        default_args.pop('crops_coords_top_left', None)
        default_args.pop('negative_aesthetic_score', None)
        default_args.pop('negative_target_size', None)
        default_args.pop('negative_original_size', None)
        default_args.pop('negative_crops_coords_top_left', None)

    def _call_torch(self, default_args, user_args):

        prompt: _prompt.Prompt() = user_args.get('prompt', _prompt.Prompt())
        default_args['prompt'] = prompt.positive if prompt.positive else ''
        default_args['negative_prompt'] = prompt.negative

        self._get_sdxl_conditioning_args(self._pipeline, default_args, user_args)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'prompt_2', 'sdxl_second_prompt',
                                             '--sdxl-second-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.positive)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'negative_prompt_2', 'sdxl_second_prompt',
                                             '--sdxl-second-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.negative)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'guidance_rescale', 'guidance_rescale',
                                             '--guidance-rescales', 0.0)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'image_guidance_scale', 'image_guidance_scale',
                                             '--image-guidance-scales', 1.5)

        batch_size = user_args.get('batch_size', 1)
        mock_batching = False

        if self._model_type != _enums.ModelTypes.TORCH_UPSCALER_X2:
            # Upscaler does not take this argument, can only produce one image
            default_args['num_images_per_prompt'] = batch_size
        else:
            mock_batching = batch_size > 1

        def generate_images(*args, **kwargs):
            if mock_batching:
                images = []
                for i in range(0, batch_size):
                    images.append(DiffusionPipelineWrapper._call_pipeline(
                        *args, **kwargs).images[0])
                return images
            else:
                return DiffusionPipelineWrapper._call_pipeline(
                    *args, **kwargs).images

        default_args['generator'] = torch.Generator(device=self._device).manual_seed(user_args.get('seed', 0))

        if isinstance(self._pipeline, diffusers.StableDiffusionInpaintPipelineLegacy):
            # Not necessary, will cause an error
            default_args.pop('width')
            default_args.pop('height')

        has_control_net = hasattr(self._pipeline, 'controlnet')
        sd_edit = has_control_net or isinstance(self._pipeline,
                                                diffusers.StableDiffusionXLInpaintPipeline)

        if has_control_net:
            default_args['controlnet_conditioning_scale'] = \
                self._get_control_net_conditioning_scale()

            default_args['control_guidance_start'] = \
                self._get_control_net_guidance_start()

            default_args['control_guidance_end'] = \
                self._get_control_net_guidance_end()

        if self._sdxl_refiner_pipeline is None:
            return PipelineWrapperResult(generate_images(
                pipeline=self._pipeline,
                device=self._device, **default_args))

        high_noise_fraction = user_args.get('sdxl_high_noise_fraction',
                                            _constants.DEFAULT_SDXL_HIGH_NOISE_FRACTION)

        if sd_edit:
            i_start = dict()
            i_end = dict()
        else:
            i_start = {'denoising_start': high_noise_fraction}
            i_end = {'denoising_end': high_noise_fraction}

        image = DiffusionPipelineWrapper._call_pipeline(pipeline=self._pipeline,
                                                        device=self._device,
                                                        **default_args,
                                                        **i_end,
                                                        output_type='latent').images

        default_args['image'] = image

        if not isinstance(self._sdxl_refiner_pipeline, diffusers.StableDiffusionXLInpaintPipeline):
            # Width / Height not necessary for any other refiner
            if not (isinstance(self._pipeline, diffusers.StableDiffusionXLImg2ImgPipeline) and
                    isinstance(self._sdxl_refiner_pipeline, diffusers.StableDiffusionXLImg2ImgPipeline)):
                # Width / Height does not get passed to img2img
                default_args.pop('width')
                default_args.pop('height')

        # refiner does not use LoRA
        default_args.pop('cross_attention_kwargs', None)

        # Or any of these
        self._pop_sdxl_conditioning_args(default_args)
        default_args.pop('guidance_rescale', None)
        default_args.pop('controlnet_conditioning_scale', None)
        default_args.pop('control_guidance_start', None)
        default_args.pop('control_guidance_end', None)
        default_args.pop('image_guidance_scale', None)
        default_args.pop('control_image', None)

        # we will handle the strength parameter if it is necessary below
        default_args.pop('strength', None)

        # We do not want to override the refiner secondary prompt
        # with that of --sdxl-second-prompts by default
        default_args.pop('prompt_2', None)
        default_args.pop('negative_prompt_2', None)

        self._get_sdxl_conditioning_args(self._sdxl_refiner_pipeline,
                                         default_args, user_args,
                                         user_prefix='refiner')

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'prompt', 'sdxl_refiner_prompt',
                                             '--sdxl-refiner-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.positive)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'negative_prompt', 'sdxl_refiner_prompt',
                                             '--sdxl-refiner-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.negative)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'prompt_2', 'sdxl_refiner_second_prompt',
                                             '--sdxl-refiner-second-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.positive)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'negative_prompt_2', 'sdxl_refiner_second_prompt',
                                             '--sdxl-refiner-second-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.negative)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'guidance_rescale', 'sdxl_refiner_guidance_rescale',
                                             '--sdxl-refiner-guidance-rescales', 0.0)

        sdxl_refiner_inference_steps = user_args.get('sdxl_refiner_inference_steps')
        if sdxl_refiner_inference_steps is not None:
            default_args['num_inference_steps'] = sdxl_refiner_inference_steps

        sdxl_refiner_guidance_scale = user_args.get('sdxl_refiner_guidance_scale')
        if sdxl_refiner_guidance_scale is not None:
            default_args['guidance_scale'] = sdxl_refiner_guidance_scale

        sdxl_refiner_guidance_rescale = user_args.get('sdxl_refiner_guidance_rescale')
        if sdxl_refiner_guidance_rescale is not None:
            default_args['guidance_rescale'] = sdxl_refiner_guidance_rescale

        if sd_edit:
            strength = float(decimal.Decimal('1.0') - decimal.Decimal(str(high_noise_fraction)))

            if strength <= 0.0:
                strength = 0.2
                _messages.log(f'Refiner edit mode image seed strength (1.0 - high-noise-fraction) '
                              f'was calculated at <= 0.0, defaulting to {strength}',
                              level=_messages.WARNING)
            else:
                _messages.log(f'Running refiner in edit mode with '
                              f'refiner image seed strength = {strength}, IE: (1.0 - high-noise-fraction)')

            inference_steps = default_args.get('num_inference_steps')

            if (strength * inference_steps) < 1.0:
                strength = 1.0 / inference_steps
                _messages.log(
                    f'Refiner edit mode image seed strength (1.0 - high-noise-fraction) * inference-steps '
                    f'was calculated at < 1, defaulting to (1.0 / inference-steps): {strength}',
                    level=_messages.WARNING)

            default_args['strength'] = strength

        return PipelineWrapperResult(
            DiffusionPipelineWrapper._call_pipeline(
                pipeline=self._sdxl_refiner_pipeline,
                device=self._device,
                **default_args, **i_start).images)

    def _lazy_init_pipeline(self, pipeline_type):
        if self._pipeline is not None:
            if self._pipeline_type == pipeline_type:
                return False

        self._pipeline_type = pipeline_type

        if _enums.model_type_is_sdxl(self._model_type) and self._textual_inversion_uris:
            raise NotImplementedError('Textual inversion not supported for SDXL.')

        if self._model_type == _enums.ModelTypes.FLAX:
            if not _enums.have_jax_flax():
                raise NotImplementedError('flax and jax are not installed.')

            if self._textual_inversion_uris:
                raise NotImplementedError('Textual inversion not supported for flax.')

            if self._pipeline_type != _enums.PipelineTypes.TXT2IMG and self._control_net_uris:
                raise NotImplementedError('Inpaint and Img2Img not supported for flax with ControlNet.')

            if self._vae_tiling or self._vae_slicing:
                raise NotImplementedError('--vae-tiling/--vae-slicing not supported for flax.')

            creation_result = \
                _pipelines.create_flax_diffusion_pipeline(pipeline_type=pipeline_type,
                                                          model_type=self._model_path,
                                                          revision=self._revision,
                                                          dtype=self._dtype,
                                                          vae_uri=self._vae_uri,
                                                          control_net_uris=self._control_net_uris,
                                                          scheduler=self._scheduler,
                                                          safety_checker=self._safety_checker,
                                                          auth_token=self._auth_token,
                                                          local_files_only=self._local_files_only)

            self._pipeline = creation_result.pipeline
            self._flax_params = creation_result.flax_params
            self._parsed_control_net_uris = creation_result.parsed_control_net_uris

        elif self._sdxl_refiner_uri is not None:
            if not _enums.model_type_is_sdxl(self._model_type):
                raise NotImplementedError('Only Stable Diffusion XL models support refiners, '
                                          'please use --model-type torch-sdxl if you are trying to load an sdxl model.')

            if not _pipelines.scheduler_is_help(self._sdxl_refiner_scheduler):
                # Don't load this up if were just going to be getting
                # information about compatible schedulers for the refiner
                creation_result = \
                    _pipelines.create_torch_diffusion_pipeline(pipeline_type=pipeline_type,
                                                               model_path=self._model_path,
                                                               model_type=self._model_type,
                                                               subfolder=self._model_subfolder,
                                                               revision=self._revision,
                                                               variant=self._variant,
                                                               dtype=self._dtype,
                                                               vae_uri=self._vae_uri,
                                                               lora_uris=self._lora_uris,
                                                               control_net_uris=self._control_net_uris,
                                                               scheduler=self._scheduler,
                                                               safety_checker=self._safety_checker,
                                                               auth_token=self._auth_token,
                                                               device=self._device,
                                                               local_files_only=self._local_files_only)
                self._pipeline = creation_result.pipeline
                self._parsed_control_net_uris = creation_result.parsed_control_net_uris

            refiner_pipeline_type = _enums.PipelineTypes.IMG2IMG if pipeline_type is _enums.PipelineTypes.TXT2IMG else pipeline_type

            if self._pipeline is not None:
                refiner_extra_args = {'vae': self._pipeline.vae,
                                      'text_encoder_2': self._pipeline.text_encoder_2}
            else:
                refiner_extra_args = None

            creation_result = \
                _pipelines.create_torch_diffusion_pipeline(pipeline_type=refiner_pipeline_type,
                                                           model_path=self._parsed_sdxl_refiner_uri.model,
                                                           model_type=_enums.ModelTypes.TORCH_SDXL,
                                                           subfolder=self._parsed_sdxl_refiner_uri.subfolder,
                                                           revision=self._parsed_sdxl_refiner_uri.revision,

                                                           variant=self._parsed_sdxl_refiner_uri.variant if
                                                           self._parsed_sdxl_refiner_uri.variant is not None else self._variant,

                                                           dtype=self._parsed_sdxl_refiner_uri.dtype if
                                                           self._parsed_sdxl_refiner_uri.dtype is not None else self._dtype,

                                                           scheduler=self._scheduler if
                                                           self._sdxl_refiner_scheduler is None else self._sdxl_refiner_scheduler,

                                                           safety_checker=self._safety_checker,
                                                           auth_token=self._auth_token,
                                                           extra_args=refiner_extra_args,
                                                           local_files_only=self._local_files_only)
            self._sdxl_refiner_pipeline = creation_result.pipeline
        else:
            offload = self._control_net_uris and self._model_type == _enums.ModelTypes.TORCH_SDXL

            creation_result = \
                _pipelines.create_torch_diffusion_pipeline(pipeline_type=pipeline_type,
                                                           model_path=self._model_path,
                                                           model_type=self._model_type,
                                                           subfolder=self._model_subfolder,
                                                           revision=self._revision,
                                                           variant=self._variant,
                                                           dtype=self._dtype,
                                                           vae_uri=self._vae_uri,
                                                           lora_uris=self._lora_uris,
                                                           textual_inversion_uris=self._textual_inversion_uris,
                                                           control_net_uris=self._control_net_uris,
                                                           scheduler=self._scheduler,
                                                           safety_checker=self._safety_checker,
                                                           auth_token=self._auth_token,
                                                           device=self._device,
                                                           sequential_cpu_offload=offload,
                                                           local_files_only=self._local_files_only)
            self._pipeline = creation_result.pipeline
            self._parsed_control_net_uris = creation_result.parsed_control_net_uris

        _pipelines.set_vae_slicing_tiling(pipeline=self._pipeline,
                                          vae_tiling=self._vae_tiling,
                                          vae_slicing=self._vae_slicing)

        if self._sdxl_refiner_pipeline is not None:
            _pipelines.set_vae_slicing_tiling(pipeline=self._sdxl_refiner_pipeline,
                                              vae_tiling=self._vae_tiling,
                                              vae_slicing=self._vae_slicing)

        return True

    @staticmethod
    def _determine_pipeline_type(kwargs):
        if 'image' in kwargs and 'mask_image' in kwargs:
            # Inpainting is handled by INPAINT type
            return _enums.PipelineTypes.INPAINT

        if 'image' in kwargs:
            # Image only is handled by IMG2IMG type
            return _enums.PipelineTypes.IMG2IMG

        # All other situations handled by BASIC type
        return _enums.PipelineTypes.TXT2IMG

    def __call__(self, args: typing.Optional[DiffusionArguments] = None, **kwargs) -> PipelineWrapperResult:
        """
        Call the pipeline and generate a result.

        :param args: Optional :py:class:`.DiffusionArguments`

        :param kwargs: See :py:meth:`.DiffusionArguments.get_pipeline_wrapper_kwargs`,
            any keyword arguments given here will override values derived from the
            :py:class:`.DiffusionArguments` object given to the *args* parameter.

        :raises: :py:class:`.InvalidModelUriError`
            :py:class:`.InvalidSchedulerName`
            :py:class:`.OutOfMemoryError`
            :py:class:`NotImplementedError`

        :return: :py:class:`.PipelineWrapperResult`
        """

        if args is not None:
            kwargs = args.get_pipeline_wrapper_kwargs() | (kwargs if kwargs else dict())

        _messages.debug_log(f'Calling Pipeline Wrapper: "{self}"')
        _messages.debug_log(f'Pipeline Wrapper Args: ',
                            lambda: _textprocessing.debug_format_args(kwargs))

        _cache.enforce_cache_constraints()

        loaded_new = self._lazy_init_pipeline(
            DiffusionPipelineWrapper._determine_pipeline_type(kwargs))

        if loaded_new:
            _cache.enforce_cache_constraints()

        default_args = self._pipeline_defaults(kwargs)

        if self._model_type == _enums.ModelTypes.FLAX:
            try:
                result = self._call_flax(default_args, kwargs)
            except jaxlib.xla_extension.XlaRuntimeError as e:
                raise OutOfMemoryError(e)
        else:
            try:
                result = self._call_torch(default_args, kwargs)
            except torch.cuda.OutOfMemoryError as e:
                raise OutOfMemoryError(e)

        return result
