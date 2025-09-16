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
import datetime
import functools
import itertools
import os
import pathlib
import tempfile
import time
import typing

import PIL.Image
import PIL.PngImagePlugin
import torch

import dgenerate.filelock as _filelock
import dgenerate.files as _files
import dgenerate.image as _image
import dgenerate.imageprocessors as _imageprocessors
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.promptweighters as _promptweighters
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
from dgenerate.events import \
    Event as _Event, \
    AnimationFinishedEvent, \
    StartingGenerationStepEvent, \
    AnimationETAEvent, \
    StartingAnimationEvent, \
    StartingAnimationFileEvent
# noinspection PyUnresolvedReferences
from dgenerate.renderloopconfig import \
    RenderLoopConfig, \
    RenderLoopConfigError, \
    IMAGE_PROCESSOR_SEP, \
    gen_seeds

__doc__ = """
The main dgenerate render loop, which implements the primary functionality of dgenerate.
"""


class AnimationFileFinishedEvent(_Event):
    """
    Generated in the event stream of :py:meth:`.RenderLoop.events`

    Occurs when an animation (video or animated image) has finished being written to disk.
    """

    path: str
    """
    Path on disk where the video/animated image was saved.
    """

    config_filename: str | None
    """
    Path to a dgenerate config file if ``output_configs`` is enabled.
    """

    starting_event: StartingAnimationFileEvent
    """
    Animation :py:class:`.StartingAnimationFileEvent` related to this file finished event.
    """

    def __init__(self,
                 origin: 'RenderLoop',
                 path: str,
                 config_filename: str,
                 starting_event: StartingAnimationFileEvent):
        super().__init__(origin)
        self.config_filename = config_filename
        self.path = path
        self.starting_event = starting_event


class ImageGeneratedEvent(_Event):
    """
    Generated in the event stream of :py:meth:`.RenderLoop.events`

    Occurs when an image is generated (but not saved yet).
    """

    image: PIL.Image.Image | None
    """
    The generated image. Will be None if latent output is being used.
    """

    latents: torch.Tensor | None
    """
    The generated latents tensor. Will be None if image output is being used.
    """

    generation_step: int
    """
    The current generation step. (zero indexed)
    """

    batch_index: int
    """
    The index in the image batch for this image. Will only every be greater than zero if 
    :py:attr:`.RenderLoopConfig.batch_size` > 1 and :py:attr:`.RenderLoopConfig.batch_grid_size` is ``None``.
    """

    suggested_directory: str
    """
    A suggested directory path for saving this image in.
    
    A value of ``'.'`` may be present, this indicates the current working directory.
    """

    suggested_filename: str
    """
    A suggested filename for saving this image as. This filename will be unique
    to the render loop run / configuration. This filename will not contain
    :py:attr:`.RenderLoopConfig.output_path`, it is the suggested filename by itself.
    """

    diffusion_args: _pipelinewrapper.DiffusionArguments
    """
    Diffusion argument object, contains :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper` 
    arguments used to produce this image.
    """

    image_seed: _mediainput.ImageSeed | None
    """
    If an ``--image-seeds`` specification was used in the generation of this image,
    this object represents that image seed and contains the images that contributed
    to the generation of this image.
    """

    command_string: str
    """
    Reproduction of a command line that can be used to reproduce this image.
    
    This does not include the ``--device`` argument.
    """

    config_string: str
    """
    Reproduction of a dgenerate config file that can be used to reproduce this image.
    
    This does not include the ``--device`` argument.
    """

    @property
    def is_animation_frame(self) -> bool:
        """
        Is this image a frame in an animation?
        """
        if self.image_seed is not None:
            return self.image_seed.is_animation_frame
        return False

    @property
    def frame_index(self) -> _types.OptionalInteger:
        """
        The frame index if this is an animation frame.
        Also available through *image_seed.frame_index*,
        though here for convenience.
        """
        if self.image_seed is not None:
            return self.image_seed.frame_index
        return None

    @property
    def is_latents(self) -> bool:
        """
        Is this event representing latents tensor output?
        """
        return self.latents is not None

    @property
    def is_image_output(self) -> bool:
        """
        Is this event representing image output?
        """
        return self.image is not None

    def __init__(self, origin: 'RenderLoop',
                 image: PIL.Image.Image | None,
                 latents: torch.Tensor | None,
                 generation_step: int,
                 batch_index: int,
                 suggested_directory: str,
                 suggested_filename: str,
                 diffusion_args: _pipelinewrapper.DiffusionArguments,
                 image_seed: _mediainput.ImageSeed,
                 command_string: str,
                 config_string: str):
        super().__init__(origin)

        self.image = image
        self.latents = latents
        self.generation_step = generation_step
        self.batch_index = batch_index
        self.suggested_directory = suggested_directory if suggested_directory.strip() else '.'
        self.suggested_filename = suggested_filename
        self.diffusion_args = diffusion_args
        self.image_seed = image_seed
        self.command_string = command_string
        self.config_string = config_string


class ImageFileSavedEvent(_Event):
    """
    Generated in the event stream of :py:meth:`.RenderLoop.events`

    Occurs when an image file is written to disk.
    """

    generated_event: ImageGeneratedEvent
    """
    The :py:class:`.ImageGeneratedEvent` for the image that was saved.
    """

    path: str
    """
    Path to the saved image.
    """

    config_filename: str | None = None
    """
    Path to a dgenerate config file if ``output_configs`` is enabled.
    """

    def __init__(self,
                 origin: 'RenderLoop',
                 generated_event: ImageGeneratedEvent,
                 path: str,
                 config_filename: str | None = None):
        super().__init__(origin)
        self.generated_event = generated_event
        self.path = path
        self.config_filename = config_filename


RenderLoopEvent = typing.Union[
    ImageGeneratedEvent,
    StartingAnimationEvent,
    StartingAnimationFileEvent,
    AnimationFileFinishedEvent,
    ImageFileSavedEvent,
    AnimationFinishedEvent,
    StartingGenerationStepEvent,
    AnimationETAEvent]
"""
Possible events from the event stream created by :py:meth:`.RenderLoop.events`
"""

RenderLoopEventStream = typing.Generator[RenderLoopEvent, None, None]
"""
Event stream created by :py:meth:`.RenderLoop.events`
"""


class RenderLoop:
    """
    Render loop which implements the bulk of dgenerate's rendering capability.

    This object handles the scatter gun iteration over requested diffusion parameters,
    the generation of animations, and writing images and media to disk or providing
    those to library users through callbacks.
    """

    disable_writes: bool = False
    """
    Disable or enable all writes to disk, if you intend to only ever use the event
    stream of the render loop when using dgenerate as a library, this is a useful option.
    
    :py:attr:`.RenderLoop.written_images` and :py:attr:`.RenderLoop.written_animations` will not be available
    if writes to disk are disabled.
    """

    model_extra_modules: dict[str, typing.Any] = None
    """
    Extra raw diffusers modules to use in the creation of the main model pipeline.
    """

    second_model_extra_modules: dict[str, typing.Any] = None
    """
    Extra raw diffusers modules to use in the creation of any refiner or stable cascade decoder model pipeline.
    """

    image_processor_loader: _imageprocessors.ImageProcessorLoader
    """
    Responsible for loading any image processors referenced in the render loop configuration.
    """

    prompt_weighter_loader: _promptweighters.PromptWeighterLoader
    """
    Responsible for loading any prompt weighters referenced in the render loop configuration.
    """

    config: RenderLoopConfig
    """
    Render loops generation related configuration.
    """

    @property
    def pipeline_wrapper(self) -> _pipelinewrapper.DiffusionPipelineWrapper:
        """
        Get the last used :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper` instance.

        Will be ``None`` if :py:meth:`.RenderLoop.run` has never been called.

        :return: :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper` or ``None``
        """

        return self._pipeline_wrapper

    def __init__(self,
                 config: RenderLoopConfig | None = None,
                 image_processor_loader: _imageprocessors.ImageProcessorLoader | None = None,
                 prompt_weighter_loader: _promptweighters.PromptWeighterLoader | None = None,
                 model_extra_modules: dict[str, typing.Any] = None,
                 second_model_extra_modules: dict[str, typing.Any] = None,
                 disable_writes: bool = False):
        """
        :param config: :py:class:`.RenderLoopConfig` or :py:class:`dgenerate.arguments.DgenerateArguments`.
            If ``None`` is provided, a :py:class:`.RenderLoopConfig` instance will be created and
            assigned to :py:attr:`.RenderLoop.config`.

        :param image_processor_loader: :py:class:`dgenerate.imageprocessors.ImageProcessorLoader`.
            If ``None`` is provided, an instance will be created and assigned to
            :py:attr:`.RenderLoop.image_processor_loader`.

        :param prompt_weighter_loader: :py:class:`dgenerate.promptweighters.PromptWeighterLoader`.
            If ``None`` is provided, an instance will be created and assigned to
            :py:attr:`.RenderLoop.prompt_weighter_loader`.

        :param model_extra_modules: Extra raw diffusers modules to use in the creation
            of the main model pipeline.

        :param second_model_extra_modules: Extra raw diffusers modules to use in the creation of
            any refiner or stable cascade decoder model pipeline.

        :param disable_writes: Disable or enable all writes to disk, if you intend to only
            ever use the event stream of the render loop when using dgenerate as a
            library, this is a useful option. :py:attr:`.RenderLoop.written_images` and
            :py:attr:`.RenderLoop.written_animations` will not be available if writes to
            disk are disabled.
        """

        self._generation_step = -1
        self._frame_time_sum = 0
        self._last_frame_time = 0
        self._written_images: _files.GCFile | None = None
        self._written_animations: _files.GCFile | None = None
        self._pipeline_wrapper = None

        self.config = \
            RenderLoopConfig() if config is None else config

        # checked / validated config
        self._c_config = None

        self.image_processor_loader = \
            _imageprocessors.ImageProcessorLoader() if \
                image_processor_loader is None else image_processor_loader

        self.prompt_weighter_loader = \
            _promptweighters.PromptWeighterLoader() if \
                prompt_weighter_loader is None else prompt_weighter_loader

        self.model_extra_modules = model_extra_modules
        self.second_model_extra_modules = second_model_extra_modules
        self.disable_writes = disable_writes

        self._post_processor = None

    @property
    def written_images(self) -> collections.abc.Iterable[str]:
        """
        Iterable over image filenames written by the last run
        """

        class Iterable:
            def __init__(self, images):
                self.images = images

            def __iter__(self):
                if self.images is None:
                    return

                self.images.seek(0)
                for line in self.images:
                    yield line.rstrip('\n')

        return Iterable(self._written_images)

    @property
    def written_animations(self) -> collections.abc.Iterable[str]:
        """
        Iterable over animation filenames written by the last run
        """

        class Iterable:
            def __init__(self, animations):
                self.animations = animations

            def __iter__(self):
                if self.animations is None:
                    return

                self.animations.seek(0)
                for line in self.animations:
                    yield line.rstrip('\n')

        return Iterable(self._written_animations)

    @property
    def generation_step(self):
        """
        Returns the current generation step, (zero indexed)
        """
        return self._generation_step

    def _join_output_filename(self, components, ext, with_output_path=True):

        prefix = self._c_config.output_prefix + '_' if \
            self._c_config.output_prefix is not None else ''

        components = (str(s).replace('.', '-') for s in components)

        name = f'{prefix}' + '_'.join(components) + '.' + ext.lstrip('.')
        if with_output_path:
            return os.path.normpath(os.path.join(self._c_config.output_path, name))
        return name

    def _gen_filename_components_base(self, diffusion_args: _pipelinewrapper.DiffusionArguments):

        scheduler_components = []

        if diffusion_args.scheduler_uri:
            scheduler_components.append(diffusion_args.scheduler_uri.split(';')[0].strip())

        if diffusion_args.second_model_scheduler_uri:
            scheduler_components.append(diffusion_args.second_model_scheduler_uri.split(';')[0].strip())

        args = scheduler_components + ['s', diffusion_args.seed]

        if diffusion_args.upscaler_noise_level is not None:
            args += ['unl', diffusion_args.upscaler_noise_level]
        elif diffusion_args.image_seed_strength is not None:
            args += ['st', diffusion_args.image_seed_strength]

        args += ['g', diffusion_args.guidance_scale]

        if diffusion_args.second_model_guidance_scale is not None:
            args += ['g2', diffusion_args.second_model_guidance_scale]

        if diffusion_args.guidance_rescale is not None:
            args += ['gr', diffusion_args.guidance_rescale]

        if diffusion_args.image_guidance_scale is not None:
            args += ['igs', diffusion_args.image_guidance_scale]

        if diffusion_args.pag_scale is not None:
            args += ['ps', diffusion_args.pag_scale]

        if diffusion_args.pag_adaptive_scale is not None:
            args += ['pas', diffusion_args.pag_adaptive_scale]

        args += ['i', diffusion_args.inference_steps]

        if diffusion_args.second_model_inference_steps is not None:
            args += ['i2', diffusion_args.second_model_inference_steps]

        if diffusion_args.clip_skip is not None:
            args += ['cs', diffusion_args.clip_skip]

        if diffusion_args.adetailer_mask_shape is not None:
            args += ['ads', diffusion_args.adetailer_mask_shape]

        if diffusion_args.adetailer_detector_padding is not None:
            args += ['addp', _textprocessing.format_size(diffusion_args.adetailer_detector_padding)]

        if diffusion_args.adetailer_mask_padding is not None:
            args += ['admp', _textprocessing.format_size(diffusion_args.adetailer_mask_padding)]

        if diffusion_args.adetailer_mask_blur is not None:
            args += ['adb', diffusion_args.adetailer_mask_blur]

        if diffusion_args.adetailer_mask_dilation is not None:
            args += ['add', diffusion_args.adetailer_mask_dilation]

        if diffusion_args.adetailer_size is not None:
            args += ['adsz', diffusion_args.adetailer_size]

        if diffusion_args.sdxl_refiner_clip_skip is not None:
            args += ['rcs', diffusion_args.sdxl_refiner_clip_skip]

        if diffusion_args.sdxl_high_noise_fraction is not None:
            args += ['hnf', diffusion_args.sdxl_high_noise_fraction]

        if diffusion_args.sdxl_refiner_pag_scale is not None:
            args += ['rps', diffusion_args.sdxl_refiner_pag_scale]

        if diffusion_args.sdxl_refiner_pag_adaptive_scale is not None:
            args += ['rpas', diffusion_args.sdxl_refiner_pag_adaptive_scale]

        if diffusion_args.sdxl_refiner_guidance_rescale is not None:
            args += ['rgr', diffusion_args.sdxl_refiner_guidance_rescale]

        if diffusion_args.tea_cache_rel_l1_threshold is not None:
            args += ['tct', diffusion_args.tea_cache_rel_l1_threshold]

        if diffusion_args.ras_high_ratio is not None:
            args += ['rshr', diffusion_args.ras_high_ratio]

        if diffusion_args.ras_sample_ratio is not None:
            args += ['rssr', diffusion_args.ras_sample_ratio]

        if diffusion_args.ras_starvation_scale is not None:
            args += ['rsss', diffusion_args.ras_starvation_scale]

        if diffusion_args.ras_error_reset_steps is not None:
            args += ['rsrs', '-'.join(map(str, diffusion_args.ras_error_reset_steps))]

        if diffusion_args.ras_metric is not None:
            args += ['rsm', diffusion_args.ras_metric]

        if diffusion_args.ras_start_step is not None:
            args += ['rssp', diffusion_args.ras_start_step]

        if diffusion_args.ras_end_step is not None:
            args += ['rsep', diffusion_args.ras_end_step]

        if diffusion_args.ras_skip_num_step is not None:
            args += ['rssn', diffusion_args.ras_skip_num_step]

        if diffusion_args.ras_skip_num_step_length is not None:
            args += ['rssl', diffusion_args.ras_skip_num_step_length]

        return args

    def _get_base_extra_config_opts(self, args: _pipelinewrapper.DiffusionArguments):
        render_loop_opts = []

        if self._c_config.seed_image_processors:
            render_loop_opts.append(('--seed-image-processors',
                                     self._c_config.seed_image_processors))

        if self._c_config.mask_image_processors:
            render_loop_opts.append(('--mask-image-processors',
                                     self._c_config.mask_image_processors))

        if self._c_config.control_image_processors:
            render_loop_opts.append(('--control-image-processors',
                                     self._c_config.control_image_processors))

        if self._c_config.post_processors:
            render_loop_opts.append(('--post-processors',
                                     self._c_config.post_processors))

        if self.image_processor_loader.plugin_module_paths:
            render_loop_opts.append(('--plugin-modules',
                                     self.image_processor_loader.plugin_module_paths))

        if self._c_config.seeds_to_images:
            render_loop_opts.append(('--seeds-to-images',))

        if self._c_config.output_prefix:
            render_loop_opts.append(('--output-prefix', self._c_config.output_prefix))

        if self._c_config.output_size is not None and args.width is None:
            # sometimes, output size can be specified with effects (such as resizing input images)
            # even when it does not get passed as a parameter to the diffusion
            # pipeline wrapper, without this statement, the command line will not be accurately
            # reproduced in entirety for those cases
            render_loop_opts.append(('--output-size',
                                     _textprocessing.format_size(self._c_config.output_size)))

        return render_loop_opts

    def _get_gen_config_overrides(self):
        # We already define --seed-image-processors in extra opts,
        # and this value maps to that as well, so override it with None
        # so the option is not generated twice

        return {
            'decoded_latents_image_processor_uris': None
        }

    def _setup_batch_size_config_opts(self,
                                      file_title: str,
                                      extra_opts_out: list[tuple[str, typing.Any] | tuple[str]],
                                      extra_comments_out: list[str],
                                      batch_index: int,
                                      generation_result: _pipelinewrapper.PipelineWrapperResult):

        if generation_result.image_count > 1:
            if self._c_config.batch_grid_size is not None:
                extra_opts_out.append(('--batch-grid-size',
                                       _textprocessing.format_size(self._c_config.batch_grid_size)))
            else:
                extra_comments_out.append(
                    f'{file_title} {batch_index + 1} from a batch of {generation_result.image_count}')

    def _gen_dgenerate_config(self,
                              args: _pipelinewrapper.DiffusionArguments | None = None,
                              extra_opts:
                              collections.abc.Sequence[tuple[str] | tuple[str, typing.Any]] | None = None,
                              extra_comments: collections.abc.Iterable[str] | None = None) -> str:

        return self._pipeline_wrapper.gen_dgenerate_config(
            args,
            extra_opts=self._get_base_extra_config_opts(args) + (
                extra_opts if extra_opts else []),
            extra_comments=extra_comments,
            omit_device=True,
            overrides=self._get_gen_config_overrides()
        )

    def _gen_dgenerate_command(self,
                               args: _pipelinewrapper.DiffusionArguments | None = None,
                               extra_opts: collections.abc.Sequence[
                                               tuple[str] | tuple[str, typing.Any]] | None = None) -> str:

        return self._pipeline_wrapper.gen_dgenerate_command(
            args,
            extra_opts=self._get_base_extra_config_opts(args) + (
                extra_opts if extra_opts else []),
            omit_device=True,
            overrides=self._get_gen_config_overrides()
        )

    def _write_image(self,
                     filename_components: list[str],
                     image: PIL.Image.Image | None,
                     latents: torch.Tensor | None,
                     batch_index: int,
                     diffusion_args: _pipelinewrapper.DiffusionArguments,
                     generation_result: _pipelinewrapper.PipelineWrapperResult,
                     image_seed: _mediainput.ImageSeed | None = None) -> RenderLoopEventStream:

        self._ensure_output_path()

        extra_opts = []
        extra_comments = []

        # Determine if we're outputting tensors or images
        is_output_latents = self._c_config.is_output_latents()

        file_title = "Tensor" if is_output_latents else "Image"
        self._setup_batch_size_config_opts(file_title=file_title,
                                           extra_opts_out=extra_opts,
                                           extra_comments_out=extra_comments,
                                           batch_index=batch_index,
                                           generation_result=generation_result)

        if image_seed is not None:
            if image_seed.is_animation_frame:
                extra_opts.append(('--frame-start', image_seed.frame_index))
                extra_opts.append(('--frame-end', image_seed.frame_index))
            extra_opts.append(('--image-seeds', image_seed.uri))

        config_txt = \
            self._gen_dgenerate_config(
                diffusion_args,
                extra_opts=extra_opts,
                extra_comments=extra_comments)

        generated_image_event = ImageGeneratedEvent(
            origin=self,
            image=image,
            latents=latents,
            generation_step=self.generation_step,
            batch_index=batch_index,
            suggested_directory=self._c_config.output_path,
            suggested_filename=self._join_output_filename(
                filename_components,
                ext=self._c_config.image_format,
                with_output_path=False),
            diffusion_args=diffusion_args,
            image_seed=image_seed,
            command_string=self._gen_dgenerate_command(
                diffusion_args,
                extra_opts=extra_opts),
            config_string=config_txt
        )

        yield generated_image_event

        if self.disable_writes or (self._c_config.no_frames and image_seed and image_seed.is_animation_frame):
            return

        config_filename = None

        # Generate and touch filenames avoiding duplicates in a way
        # that is multiprocess safe between instances of dgenerate
        # when --output-overwrite is not specified, otherwise just
        # overwrite the files if they exist.

        if self._c_config.output_configs:
            if not self._c_config.output_overwrite:
                output_filename, config_filename = \
                    _filelock.touch_avoid_duplicate(
                        self._c_config.output_path,
                        path_maker=_filelock.suffix_path_maker(
                            [self._join_output_filename(filename_components, ext=self._c_config.image_format),
                             self._join_output_filename(filename_components, ext='dgen')],
                            suffix='_duplicate_'))
            else:
                output_filename = self._join_output_filename(
                    filename_components, ext=self._c_config.image_format
                )

                config_filename = self._join_output_filename(
                    filename_components, ext='dgen'
                )
        else:
            if not self._c_config.output_overwrite:
                output_filename = _filelock.touch_avoid_duplicate(
                    self._c_config.output_path,
                    path_maker=_filelock.suffix_path_maker(
                        self._join_output_filename(filename_components,
                                                   ext=self._c_config.image_format),
                        suffix='_duplicate_'))
            else:
                output_filename = self._join_output_filename(
                    filename_components, ext=self._c_config.image_format
                )

        # Write out to the empty files
        if is_output_latents:
            # Save tensor output
            assert latents is not None

            _mediaoutput.save_tensor_file(
                tensor=latents,
                path_or_file=output_filename,
                file_format=self._c_config.image_format
            )

            output_type_name = "Tensor"
        else:
            # Save image output
            assert image is not None

            if self._c_config.output_metadata:
                if output_filename.lower().endswith(('.jpg', '.jpeg')):
                    image.save(
                        output_filename,
                        exif=_image.create_jpeg_exif_with_user_comment(config_txt)
                    )
                else:
                    metadata = PIL.PngImagePlugin.PngInfo()
                    metadata.add_text("DgenerateConfig", config_txt)
                    image.save(output_filename, pnginfo=metadata)
            else:
                image.save(output_filename)

                # prevent circular import
                import dgenerate.auto1111_metadata as _auto1111_metadata

                if self._c_config.output_auto1111_metadata:
                    _auto1111_metadata.convert_and_insert_metadata(
                        output_filename, dgenerate_config=config_txt
                    )

            output_type_name = "Image"

        is_last_output = batch_index == generation_result.output_count - 1
        # Only underline the last output write message in a batch of rendered
        # outputs when --batch-size > 1

        if self._c_config.output_configs:
            with open(config_filename, "w", encoding='utf-8') as config_file:
                config_file.write(config_txt)

            yield ImageFileSavedEvent(origin=self,
                                      generated_event=generated_image_event,
                                      path=output_filename,
                                      config_filename=config_filename)

            _messages.log(
                f'Wrote {output_type_name} File: "{output_filename}"\n'
                f'Wrote Config File: "{config_filename}"',
                underline=is_last_output)
        else:
            yield ImageFileSavedEvent(origin=self,
                                      generated_event=generated_image_event,
                                      path=output_filename)

            _messages.log(f'Wrote {output_type_name} File: "{output_filename}"',
                          underline=is_last_output)

        # Append to written images for the current run
        self._written_images.write(pathlib.Path(output_filename).absolute().as_posix() + '\n')

    def _write_generation_result(self,
                                 filename_components: list[str],
                                 diffusion_args: _pipelinewrapper.DiffusionArguments,
                                 generation_result: _pipelinewrapper.PipelineWrapperResult,
                                 image_seed: _mediainput.ImageSeed | None = None) -> RenderLoopEventStream:

        # Determine if we're working with images or latents
        has_images = generation_result.has_images
        has_latents = generation_result.has_latents
        is_output_latents = self._c_config.is_output_latents()

        if self._c_config.batch_grid_size is None:
            # Handle individual outputs (no grid)
            if has_images and not is_output_latents:
                # Standard image output
                for batch_idx, image in enumerate(generation_result.images):
                    name_components = filename_components.copy()
                    if generation_result.image_count > 1:
                        name_components += ['image', batch_idx + 1]

                    yield from self._write_image(name_components,
                                                 image,
                                                 None,  # latents
                                                 batch_idx,
                                                 diffusion_args,
                                                 generation_result,
                                                 image_seed)
            elif has_latents and is_output_latents:
                # Tensor output
                for batch_idx, latents in enumerate(generation_result.latents):
                    name_components = filename_components.copy()
                    if generation_result.latents_count > 1:
                        name_components += ['latent', batch_idx + 1]

                    yield from self._write_image(name_components,
                                                 None,  # image
                                                 latents,
                                                 batch_idx,
                                                 diffusion_args,
                                                 generation_result,
                                                 image_seed)
            elif has_images and is_output_latents:
                assert False, (
                    f"Mismatch between output format and generation result: "
                    f"tensor_output={is_output_latents}, has_images={has_images}, has_latents={has_latents}"
                )
        else:
            # Handle grid output
            if has_images and not is_output_latents:
                if generation_result.image_count > 1:
                    image = generation_result.image_grid(self._c_config.batch_grid_size)
                else:
                    image = generation_result.image

                yield from self._write_image(filename_components,
                                             image,
                                             None,  # latents
                                             0,
                                             diffusion_args,
                                             generation_result,
                                             image_seed)
            elif has_latents and is_output_latents:
                raise RenderLoopConfigError(
                    "RenderLoopConfig option batch_grid_size is unsupported when using a latent output."
                )
            else:
                assert False, (
                    f"Mismatch between output format and generation result: "
                    f"tensor_output={is_output_latents}, has_images={has_images}, has_latents={has_latents}"
                )

    def _write_animation_frame(self,
                               diffusion_args: _pipelinewrapper.DiffusionArguments,
                               image_seed_obj: _mediainput.ImageSeed,
                               generation_result: _pipelinewrapper.PipelineWrapperResult) -> RenderLoopEventStream:

        filename_components = [*self._gen_filename_components_base(diffusion_args),
                               'frame',
                               image_seed_obj.frame_index + 1,
                               'step',
                               self._generation_step + 1]

        yield from self._write_generation_result(filename_components,
                                                 diffusion_args,
                                                 generation_result,
                                                 image_seed_obj)

    def _write_image_seed_gen_image(self,
                                    diffusion_args: _pipelinewrapper.DiffusionArguments,
                                    image_seed_obj: _mediainput.ImageSeed,
                                    generation_result: _pipelinewrapper.PipelineWrapperResult) -> RenderLoopEventStream:
        filename_components = [*self._gen_filename_components_base(diffusion_args),
                               'step',
                               self._generation_step + 1]

        yield from self._write_generation_result(filename_components,
                                                 diffusion_args,
                                                 generation_result,
                                                 image_seed_obj)

    def _write_prompt_only_image(self,
                                 diffusion_args: _pipelinewrapper.DiffusionArguments,
                                 generation_result: _pipelinewrapper.PipelineWrapperResult) \
            -> RenderLoopEventStream:
        filename_components = [*self._gen_filename_components_base(diffusion_args),
                               'step',
                               self._generation_step + 1]

        yield from self._write_generation_result(filename_components,
                                                 diffusion_args,
                                                 generation_result)

    def _pre_generation_step(self,
                             diffusion_args: _pipelinewrapper.DiffusionArguments):

        self._last_frame_time = 0
        self._frame_time_sum = 0
        self._generation_step += 1

        desc = diffusion_args.describe_pipeline_wrapper_args()

        total_steps = self._c_config.calculate_generation_steps()

        _messages.log(
            f'Generation Step: {self._generation_step + 1} / {total_steps}\n'
            + desc, underline=True)

        yield StartingGenerationStepEvent(
            origin=self,
            generation_step=self._generation_step,
            total_steps=total_steps
        )

    def _animation_frame_pre_generation(self, image_seed: _mediainput.ImageSeed):
        if self._last_frame_time == 0:
            eta = None
        else:
            self._frame_time_sum += time.time() - self._last_frame_time
            eta_seconds = (self._frame_time_sum / image_seed.frame_index) * (
                    image_seed.total_frames - image_seed.frame_index)
            eta = datetime.timedelta(seconds=eta_seconds)

        self._last_frame_time = time.time()

        eta_str = str(eta) if eta is not None else 'tbd...'

        _messages.log(
            f'Generating frame {image_seed.frame_index + 1} / {image_seed.total_frames}, Completion ETA: {eta_str}',
            underline=True)

        if eta is not None:
            yield AnimationETAEvent(origin=self,
                                    frame_index=image_seed.frame_index,
                                    total_frames=image_seed.total_frames, eta=eta)

    def run(self):
        """
        Run the diffusion loop, this calls :py:meth:`RenderLoopConfig.check`
        on a copy of your configuration prior to running.

        :raises RenderLoopConfigError:
        :raises dgenerate.ModelNotFoundError:
        :raises dgenerate.OutOfMemoryError:

        """
        try:
            for _ in self._run():
                continue
        except _pipelinewrapper.DiffusionArgumentsHelpException as e:
            _messages.log(e)
            pass

    def events(self) -> RenderLoopEventStream:
        """
        Run the render loop, and iterate over a stream of event objects produced by the render loop.

        This calls :py:meth:`RenderLoopConfig.check` on a copy of your configuration prior to running.

        Event objects are of the union type :py:data:`.RenderLoopEvent`

        The exceptions mentioned here are those you may encounter upon iterating,
        they will not occur upon simple acquisition of the event stream iterator.

        :raises RenderLoopConfigError:
        :raises dgenerate.ModelNotFoundError:
        :raises dgenerate.OutOfMemoryError:

        :return: :py:data:`.RenderLoopEventStream`
        """
        try:
            yield from self._run()
        except _pipelinewrapper.DiffusionArgumentsHelpException as e:
            _messages.log(e)
            pass

    def _create_pipeline_wrapper(self):
        self._pipeline_wrapper = _pipelinewrapper.DiffusionPipelineWrapper(
            self._c_config.model_path,
            dtype=self._c_config.dtype,
            device=self._c_config.device,
            model_type=self._c_config.model_type,
            revision=self._c_config.revision,
            variant=self._c_config.variant,
            subfolder=self._c_config.subfolder,
            original_config=self._c_config.original_config,
            second_model_original_config=self._c_config.second_model_original_config,
            unet_uri=self._c_config.unet_uri,
            second_model_unet_uri=self._c_config.second_model_unet_uri,
            transformer_uri=self._c_config.transformer_uri,
            vae_uri=self._c_config.vae_uri,
            lora_uris=self._c_config.lora_uris,
            lora_fuse_scale=self._c_config.lora_fuse_scale,
            image_encoder_uri=self._c_config.image_encoder_uri,
            ip_adapter_uris=self._c_config.ip_adapter_uris,
            textual_inversion_uris=self._c_config.textual_inversion_uris,
            text_encoder_uris=self._c_config.text_encoder_uris,
            second_model_text_encoder_uris=self._c_config.second_model_text_encoder_uris,
            controlnet_uris=
            self._c_config.controlnet_uris if self._c_config.image_seeds else [],
            t2i_adapter_uris=
            self._c_config.t2i_adapter_uris if self._c_config.image_seeds else [],
            sdxl_refiner_uri=self._c_config.sdxl_refiner_uri,
            s_cascade_decoder_uri=self._c_config.s_cascade_decoder_uri,
            second_model_cpu_offload=bool(self._c_config.second_model_cpu_offload),
            second_model_sequential_offload=bool(self._c_config.second_model_sequential_offload),
            safety_checker=self._c_config.safety_checker,
            auth_token=self._c_config.auth_token,
            local_files_only=self._c_config.offline_mode,
            model_extra_modules=self.model_extra_modules,
            second_model_extra_modules=self.second_model_extra_modules,
            model_cpu_offload=self._c_config.model_cpu_offload,
            model_sequential_offload=self._c_config.model_sequential_offload,
            prompt_weighter_loader=self.prompt_weighter_loader,
            adetailer_detector_uris=self._c_config.adetailer_detector_uris,
            adetailer_crop_control_image=bool(self._c_config.adetailer_crop_control_image),
            quantizer_uri=self._c_config.quantizer_uri,
            quantizer_map=self._c_config.quantizer_map,
            second_model_quantizer_uri=self._c_config.second_model_quantizer_uri,
            second_model_quantizer_map=self._c_config.second_model_quantizer_map
        )
        return self._pipeline_wrapper

    def _ensure_output_path(self):
        """
        Create the output path mentioned in the configuration and its parent directory's if necessary
        """

        if not self.disable_writes:
            pathlib.Path(self._c_config.output_path).mkdir(parents=True, exist_ok=True)

    def _run(self) -> RenderLoopEventStream:
        self._c_config = self.config.copy()
        self._c_config.check()

        self._ensure_output_path()

        self._written_images = _files.GCFile(
            tempfile.TemporaryFile('w+t'))
        self._written_animations = _files.GCFile(
            tempfile.TemporaryFile('w+t'))

        self._generation_step = -1
        self._frame_time_sum = 0
        self._last_frame_time = 0

        generation_steps = self._c_config.calculate_generation_steps()

        if generation_steps == 0:
            _messages.log(f'Options resulted in no generation steps, nothing to do.', underline=True)
            return

        _messages.log(f'Beginning {generation_steps} generation steps...', underline=True)

        try:
            self._init_post_processor()

            if self._c_config.image_seeds:
                yield from self._render_with_image_seeds()
            else:
                pipeline_wrapper = self._create_pipeline_wrapper()

                sdxl_high_noise_fractions = \
                    self._c_config.sdxl_high_noise_fractions if \
                        self._c_config.sdxl_refiner_uri is not None else None

                for diffusion_arguments in self._c_config.iterate_diffusion_args(
                        sdxl_high_noise_fraction=sdxl_high_noise_fractions,
                        image_seed_strength=None,
                        upscaler_noise_level=None):
                    diffusion_arguments.batch_size = self._c_config.batch_size
                    diffusion_arguments.sdxl_refiner_edit = self._c_config.sdxl_refiner_edit

                    yield from self._pre_generation_step(diffusion_arguments)

                    with pipeline_wrapper(diffusion_arguments) as generation_result:
                        self._run_postprocess(generation_result)
                        yield from self._write_prompt_only_image(
                            diffusion_arguments,
                            generation_result)
        finally:
            self._destroy_post_processor()

    def _init_post_processor(self):
        if self._c_config.post_processors is None:
            self._post_processor = None
        else:
            self._post_processor = self.image_processor_loader.load(
                self._c_config.post_processors,
                device=self._c_config.device,
                local_files_only=self._c_config.offline_mode
            )
            _messages.debug_log('Loaded Post Processor:', self._post_processor)

    def _destroy_post_processor(self):
        if self._post_processor is None:
            return

        self._post_processor.to('cpu')
        del self._post_processor
        self._post_processor = None

    def _run_postprocess(self, generation_result: _pipelinewrapper.PipelineWrapperResult):
        if self._post_processor is not None:
            if generation_result.has_images:
                for idx, image in enumerate(generation_result.images):
                    img = self._post_processor.process(image)
                    generation_result.images[idx] = img
            else:
                _messages.warning(
                    f'Post processor step ({str(self._post_processor)}) is being '
                    f'skipped due to output being latents.')

    def _load_image_processors(self, processors):
        if not processors:
            return None

        processor_chain = [[]]

        for processor in processors:
            if processor != IMAGE_PROCESSOR_SEP:
                processor_chain[-1].append(processor)
            else:
                processor_chain.append([])

        if len(processor_chain) == 1:
            r = self.image_processor_loader.load(
                processor_chain[0],
                device=self._c_config.device,
                local_files_only=self._c_config.offline_mode)
        else:
            r = [self.image_processor_loader.load(p,
                                                  device=self._c_config.device,
                                                  local_files_only=self._c_config.offline_mode)
                 for p in processor_chain]

        return r

    def _load_seed_image_processors(self):
        if not self._c_config.seed_image_processors:
            return None

        r = self._load_image_processors(self._c_config.seed_image_processors)

        _messages.debug_log('Loaded Seed Image Processor(s):', r)

        return r

    def _load_mask_image_processors(self):
        if not self._c_config.mask_image_processors:
            return None

        r = self._load_image_processors(self._c_config.mask_image_processors)

        _messages.debug_log('Loaded Mask Image Processor(s):', r)

        return r

    def _load_control_image_processors(self):
        if not self._c_config.control_image_processors:
            return None

        r = self._load_image_processors(self._c_config.control_image_processors)

        _messages.debug_log('Loaded Control Image Processor(s):', r)

        return r

    def _render_with_image_seeds(self):
        # unintuitive, but these should be long-lived and then
        # garbage collected, if they are not specified by the user
        # these will return None
        seed_image_processor = self._load_seed_image_processors()
        mask_image_processor = self._load_mask_image_processors()
        control_image_processor = self._load_control_image_processors()
        try:
            yield from self._render_with_image_seeds_unmanaged(
                seed_image_processor,
                mask_image_processor,
                control_image_processor)
        finally:
            if seed_image_processor is not None:
                if isinstance(seed_image_processor, list):
                    for p in seed_image_processor:
                        if p is not None:
                            p.to('cpu')
                else:
                    seed_image_processor.to('cpu')
            if mask_image_processor is not None:
                if isinstance(mask_image_processor, list):
                    for p in mask_image_processor:
                        if p is not None:
                            p.to('cpu')
                else:
                    mask_image_processor.to('cpu')
            if control_image_processor is not None:
                if isinstance(control_image_processor, list):
                    for p in control_image_processor:
                        if p is not None:
                            p.to('cpu')
                else:
                    control_image_processor.to('cpu')

    def _get_media_reader_path_opener(self):
        """
        Returns a function that can be used to open media files for reading.
        This is used by the image seed processor to read images and videos.
        """
        return functools.partial(
            _mediainput.fetch_media_data_stream,
            local_files_only=self._c_config.offline_mode)

    def _should_skip_external_resize(self) -> bool:
        """
        Determines if we should skip external resizing and let the DiffusionPipelineWrapper handle it.
        This is needed for:

        - Models using image encoders (like Stable Cascade) where input size doesn't matter
        
        :return: True if resizing should be skipped, False otherwise
        """
        return _pipelinewrapper.model_type_is_s_cascade(self._c_config.model_type)

    def _render_with_image_seeds_unmanaged(
            self,
            seed_image_processor: _mediainput.ImageProcessorSpec,
            mask_image_processor: _mediainput.ImageProcessorSpec,
            control_image_processor: _mediainput.ImageProcessorSpec):

        pipeline_wrapper = self._create_pipeline_wrapper()

        def iterate_image_seeds():
            # image seeds have already had logical and syntax validation performed
            for idx, uri_to_parsed in enumerate(zip(self._c_config.image_seeds, self._c_config.parsed_image_seeds)):
                yield uri_to_parsed[0], uri_to_parsed[1], self._c_config.seeds[idx % len(self._c_config.seeds)]

        for image_seed_uri, parsed_image_seed, seed_to_image in list(iterate_image_seeds()):

            is_control_guidance_spec = (self._c_config.controlnet_uris or self._c_config.t2i_adapter_uris) \
                                       and parsed_image_seed.is_single_spec

            if is_control_guidance_spec:
                _messages.log(f'Processing Control Image: "{image_seed_uri}"', underline=True)
            else:
                _messages.log(f'Processing Image Seed: "{image_seed_uri}"', underline=True)

            overrides = {}
            if self._c_config.seeds_to_images:
                overrides['seed'] = [seed_to_image]

            arg_iterator = self._c_config.iterate_diffusion_args(**overrides)

            if is_control_guidance_spec:
                seed_info = _mediainput.get_control_image_info(
                    parsed_image_seed,
                    self._c_config.frame_start,
                    self._c_config.frame_end,
                    path_opener=self._get_media_reader_path_opener()
                )
            else:
                seed_info = _mediainput.get_image_seed_info(
                    parsed_image_seed,
                    self._c_config.frame_start,
                    self._c_config.frame_end,
                    path_opener=self._get_media_reader_path_opener()
                )

            if is_control_guidance_spec:
                def image_seed_iterator():
                    yield from _mediainput.iterate_control_image(
                        uri=parsed_image_seed,
                        frame_start=self._c_config.frame_start,
                        frame_end=self._c_config.frame_end,
                        resize_resolution=None if self._should_skip_external_resize() else self._c_config.output_size,
                        aspect_correct=not self._c_config.no_aspect,
                        image_processor=control_image_processor,
                        path_opener=self._get_media_reader_path_opener()
                    )

            else:
                def image_seed_iterator():
                    # Skip automatic resizing when inpaint crop is enabled - let the pipeline wrapper handle it
                    if self._c_config.inpaint_crop:
                        resize_resolution = None
                    else:
                        if not self._should_skip_external_resize():
                            resize_resolution = self._c_config.output_size
                        else:
                            resize_resolution = None

                    yield from _mediainput.iterate_image_seed(
                        uri=parsed_image_seed,
                        frame_start=self._c_config.frame_start,
                        frame_end=self._c_config.frame_end,
                        resize_resolution=resize_resolution,
                        aspect_correct=not self._c_config.no_aspect,
                        seed_image_processor=seed_image_processor,
                        mask_image_processor=mask_image_processor,
                        control_image_processor=control_image_processor,
                        path_opener=self._get_media_reader_path_opener(),
                        check_dimensions_match=
                        not _pipelinewrapper.model_type_is_s_cascade(self._c_config.model_type))

            if seed_info.is_animation:

                if is_control_guidance_spec:
                    def set_extra_args(args: _pipelinewrapper.DiffusionArguments,
                                       ci_obj: _mediainput.ImageSeed):
                        args.control_images = ci_obj.control_images
                else:
                    def set_extra_args(args: _pipelinewrapper.DiffusionArguments,
                                       ims_obj: _mediainput.ImageSeed):
                        if ims_obj.images is not None:
                            args.images = ims_obj.images
                            if self._c_config.seed_image_processors and \
                                    any(_torchutil.is_tensor(img) for img in ims_obj.images):
                                args.decoded_latents_image_processor_uris = \
                                    self._c_config.seed_image_processors
                        if ims_obj.latents is not None:
                            args.latents = ims_obj.latents
                        if ims_obj.mask_images is not None:
                            args.mask_images = ims_obj.mask_images
                        if ims_obj.control_images is not None:
                            args.control_images = ims_obj.control_images
                        if ims_obj.adapter_images is not None:
                            args.ip_adapter_images = ims_obj.adapter_images
                        if ims_obj.floyd_image is not None:
                            args.floyd_image = ims_obj.floyd_image

                yield from self._render_animation(pipeline_wrapper=pipeline_wrapper,
                                                  set_wrapper_args_per_image_seed=set_extra_args,
                                                  arg_iterator=arg_iterator,
                                                  image_seed_iterator=image_seed_iterator,
                                                  fps=seed_info.fps)
                continue

            for diffusion_arguments in arg_iterator:
                diffusion_arguments.batch_size = self._c_config.batch_size
                diffusion_arguments.sdxl_refiner_edit = self._c_config.sdxl_refiner_edit

                yield from self._pre_generation_step(diffusion_arguments)

                with next(image_seed_iterator()) as image_seed:
                    if not is_control_guidance_spec and image_seed.images is not None:
                        diffusion_arguments.images = image_seed.images
                        if self._c_config.seed_image_processors and \
                                any(_torchutil.is_tensor(img) for img in image_seed.images):
                            diffusion_arguments.decoded_latents_image_processor_uris = \
                                self._c_config.seed_image_processors

                    if image_seed.latents is not None:
                        diffusion_arguments.latents = image_seed.latents

                    if image_seed.mask_images is not None:
                        diffusion_arguments.mask_images = image_seed.mask_images

                    if image_seed.control_images is not None:
                        diffusion_arguments.control_images = image_seed.control_images

                    if image_seed.adapter_images is not None:
                        diffusion_arguments.ip_adapter_images = image_seed.adapter_images

                    if image_seed.floyd_image is not None:
                        diffusion_arguments.floyd_image = image_seed.floyd_image

                    with pipeline_wrapper(diffusion_arguments) as generation_result:
                        self._run_postprocess(generation_result)
                        yield from self._write_image_seed_gen_image(
                            diffusion_arguments,
                            image_seed,
                            generation_result)

    def _gen_animation_filename(self,
                                diffusion_args: _pipelinewrapper.DiffusionArguments,
                                generation_step,
                                ext):

        components = ['ANIM',
                      *self._gen_filename_components_base(diffusion_args),
                      'step', generation_step + 1]

        return self._join_output_filename(components, ext=ext)

    def _render_animation(self,
                          pipeline_wrapper: _pipelinewrapper.DiffusionPipelineWrapper,
                          set_wrapper_args_per_image_seed:
                          typing.Callable[[_pipelinewrapper.DiffusionArguments, _mediainput.ImageSeed], None],
                          arg_iterator:
                          collections.abc.Iterator[_pipelinewrapper.DiffusionArguments],
                          image_seed_iterator:
                          typing.Callable[[], collections.abc.Iterator[_mediainput.ImageSeed]],
                          fps: float) \
            -> RenderLoopEventStream:

        first_diffusion_args = next(arg_iterator)

        base_filename = \
            self._gen_animation_filename(
                first_diffusion_args,
                self._generation_step + 1,
                ext=self._c_config.animation_format)

        next_args_terminates_anim = False

        not_writing_animation_file = \
            self.disable_writes or self._c_config.animation_format == 'frames'

        if not_writing_animation_file:
            # The interface can be used as a mock object
            anim_writer = _mediaoutput.AnimationWriter()
        else:
            anim_writer = _mediaoutput.MultiAnimationWriter(
                animation_format=self._c_config.animation_format,
                filename=base_filename,
                fps=fps, allow_overwrites=self._c_config.output_overwrite)

        with anim_writer:

            for diffusion_args in itertools.chain([first_diffusion_args], arg_iterator):
                diffusion_args.batch_size = self._c_config.batch_size
                diffusion_args.sdxl_refiner_edit = self._c_config.sdxl_refiner_edit

                yield from self._pre_generation_step(diffusion_args)

                if next_args_terminates_anim:
                    next_args_terminates_anim = False

                    # this just starts a new file, the last file
                    # has already been ended, and an event generated
                    # for the finished animation file
                    anim_writer.end(
                        new_file=self._gen_animation_filename(
                            diffusion_args,
                            self._generation_step,
                            ext=self._c_config.animation_format))

                for image_seed_frame in image_seed_iterator():
                    frame_duration = image_seed_frame.frame_duration
                    fps = image_seed_frame.fps
                    total_frames = image_seed_frame.total_frames

                    if image_seed_frame.frame_index == 0:
                        starting_animation_event = StartingAnimationEvent(
                            origin=self,
                            total_frames=total_frames,
                            fps=fps,
                            frame_duration=frame_duration)
                        yield starting_animation_event

                    with image_seed_frame:

                        yield from self._animation_frame_pre_generation(image_seed_frame)

                        set_wrapper_args_per_image_seed(diffusion_args, image_seed_frame)

                        with pipeline_wrapper(diffusion_args) as generation_result:
                            self._run_postprocess(generation_result)
                            self._ensure_output_path()

                            if generation_result.has_images:
                                if generation_result.image_count > 1 and self._c_config.batch_grid_size is not None:
                                    anim_writer.write(
                                        generation_result.image_grid(self._c_config.batch_grid_size))
                                else:
                                    anim_writer.write(generation_result.images)
                            elif generation_result.has_latents:
                                # For latent output in animations, we need to decode to images for the animation writer
                                # This is a limitation - animations require decoded images
                                raise RenderLoopConfigError(
                                    'Animated file output with latent output is not supported. '
                                    'Latent output can only be used for individual frame output, '
                                    'i.e. --animation-format "frames".'
                                )

                            if image_seed_frame.frame_index == 0:
                                # Perform on first frame write

                                if not not_writing_animation_file:

                                    animation_filenames_message = \
                                        '\n'.join(f'Beginning Writes To Animation: "{f}"'
                                                  for f in anim_writer.filenames)

                                    if self._c_config.output_configs:

                                        _messages.log(animation_filenames_message)
                                        config_filenames = []
                                        for idx, filename in enumerate(anim_writer.filenames):
                                            config_filenames.append(
                                                self._write_animation_config_file(
                                                    filename=os.path.splitext(filename)[0] + '.dgen',
                                                    image_seed_uri=image_seed_frame.uri,
                                                    batch_index=idx,
                                                    diffusion_args=diffusion_args,
                                                    generation_result=generation_result))
                                    else:
                                        _messages.log(animation_filenames_message, underline=True)

                                    starting_animation_file_events = []
                                    for f in anim_writer.filenames:
                                        starting_animation_file_event = \
                                            StartingAnimationFileEvent(
                                                origin=self,
                                                path=f,
                                                total_frames=total_frames,
                                                fps=fps,
                                                frame_duration=frame_duration)
                                        starting_animation_file_events.append(
                                            starting_animation_file_event)
                                        yield starting_animation_file_event

                            yield from self._write_animation_frame(
                                diffusion_args,
                                image_seed_frame,
                                generation_result)

                        next_args_terminates_anim = image_seed_frame.frame_index == (image_seed_frame.total_frames - 1)

                yield AnimationFinishedEvent(
                    origin=self,
                    starting_event=starting_animation_event)

                written_filenames = anim_writer.filenames.copy() if not not_writing_animation_file else []
                anim_writer.end()
                for idx, file in enumerate(written_filenames):
                    self._written_animations.write(pathlib.Path(file).absolute().as_posix() + '\n')
                    yield AnimationFileFinishedEvent(
                        origin=self,
                        path=file,
                        config_filename=config_filenames[idx] if self._c_config.output_configs else None,
                        starting_event=starting_animation_file_events[idx])

    def _write_animation_config_file(self,
                                     filename: str,
                                     image_seed_uri: str,
                                     batch_index: int,
                                     diffusion_args: _pipelinewrapper.DiffusionArguments,
                                     generation_result: _pipelinewrapper.PipelineWrapperResult):
        self._ensure_output_path()

        extra_opts = []

        if self._c_config.frame_start is not None and \
                self._c_config.frame_start != 0:
            extra_opts.append(('--frame-start',
                               self._c_config.frame_start))

        if self._c_config.frame_end is not None:
            extra_opts.append(('--frame-end',
                               self._c_config.frame_end))

        if self._c_config.animation_format is not None:
            extra_opts.append(('--animation-format',
                               self._c_config.animation_format))

        extra_opts.append(('--image-seeds', image_seed_uri))

        extra_comments = []

        self._setup_batch_size_config_opts(file_title="Animation",
                                           extra_opts_out=extra_opts,
                                           extra_comments_out=extra_comments,
                                           batch_index=batch_index,
                                           generation_result=generation_result)

        config_text = \
            self._gen_dgenerate_config(
                diffusion_args,
                extra_opts=extra_opts,
                extra_comments=extra_comments)

        if not self._c_config.output_overwrite:
            filename = \
                _filelock.touch_avoid_duplicate(
                    self._c_config.output_path,
                    path_maker=_filelock.suffix_path_maker(filename,
                                                           '_duplicate_'))

        with open(filename, "w", encoding='utf-8') as config_file:
            config_file.write(config_text)

        _messages.log(f'Wrote Animation Config File: "{filename}"',
                      underline=batch_index == generation_result.image_count - 1)

        return filename


__all__ = _types.module_all()
