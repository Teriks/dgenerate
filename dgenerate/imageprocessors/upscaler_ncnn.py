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
import os.path

import PIL.Image
import numpy
import psutil
import tqdm.auto

import dgenerate
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
try:
    import dgenerate.imageprocessors.ncnn_model as _ncnn_model
except ImportError:
    _ncnn_model = None
import dgenerate.imageprocessors.upscale_tiler as _upscale_tiler
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.types as _types
import dgenerate.webcache as _webcache
import dgenerate.imageprocessors.exceptions as _exceptions
import dgenerate.hfhub as _hfhub


class _UnsupportedModelError(Exception):
    """model is not of a supported type."""
    pass


def _stack_images(images: list) -> numpy.ndarray:
    out = []
    for img in images:
        normalized = numpy.array(img, dtype=numpy.float32) / 255.0
        # Convert to (c, h, w) format
        out.append(normalized.transpose((2, 0, 1)))
    return numpy.stack(out)


def _output_to_pil(output):
    return PIL.Image.fromarray(
        (output.transpose(1, 2, 0) * 255).astype(numpy.uint8))


class UpscalerNCNNProcessor(_imageprocessor.ImageProcessor):
    """
    Implements tiled upscaling with NCNN upscaler models.

    The "model" argument should be a path or URL to a NCNN compatible upscaler model.
    This may also be a Hugging Face blob link.

    The "param" argument should be a path or URL to the NCNN param file for the model.
    This may also be a Hugging Face blob link.

    Downloaded model / param files are cached in the dgenerate web cache on disk
    until the cache expiry time for the file is met.

    When using this processor as a pre-processor or post-processor for diffusion,
    GPU memory will be fenced, any cached models related to diffusion on the GPU
    will be evacuated entirely before this processor runs if they exist on the same GPU
    as the processor, this is to prevent catastrophic interaction between the Vulkan
    and Torch cuda allocators.

    Once a Vulkan allocator exists on a specific GPU it cannot be destroyed except
    via the process exiting due to issues with the ncnn python binding. If you
    create this processor on a GPU you intend to perform diffusion on, you are
    going to run into memory errors after the first image generation and
    there on out until the process exits.

    When the process exits it is very likely to exit with a non-zero return
    code after using this processor even if the upscale operations were successful,
    this is due to problems with the ncnn python binding creating a segfault at exit.
    If you are using dgenerate interactively in shell mode or from the Console UI,
    this will occur without consequence when the interpreter process exits.

    Note that if any other process runs diffusion / inference via torch on
    the same GPU as this image processor while ncnn is performing inference,
    you will likely encounter a segfault in either of the processes and
    a very hard crash.

    You can safely run this processor in parallel with diffusion, or other torch
    based image processors with GPU acceleration, by placing it on a separate gpu
    using the "gpu-index" argument.

    For these reasons, this processor runs on the CPU by default, you can enable
    GPU usage with GPU related arguments mentioned in this documentation below.

    -----

    The "use-gpu" argument determines if the gpu is used, defaults to False.

    The "gpu-index" argument determines which gpu is used, it is 0 indexed,
    and defaults to 0 which is most likely your main GPU.

    The "threads" argument determines the number of cpu threads used, the
    default value is "auto" which uses the maximum amount. You may also pass
    "half" to use half the cpus logical thread count.

    The "blocktime" argument determines the blocktime in milliseconds
    for OpenMP parallelization when running on the cpu, this value
    should be between 0 and 400 milliseconds, if you do not specify
    the NCNN default for your platform is used. You cannot use this
    argument when running on the GPU, an argument error will be thrown.

    The "winograd" argument determines if the winograd convolution
    optimization is used when running on the CPU. If you do not specify
    it, the NCNN default for you platform is used. You cannot use this
    argument when running on the GPU, an argument error will be thrown.

    The "sgemm" argument determines if the sgemm convolution
    optimization is used when running on the CPU. If you do not specify
    it, the NCNN default for you platform is used. You cannot use this
    argument when running on the GPU, an argument error will be thrown.

    The "tile" argument can be used to specify the tile size for tiled upscaling, it
    must be divisible by 2 and be less than or equal to 400, the default is 400.
    Tile size is limited to a max of 400 due to memory allocator issues in ncnn.
    You may disable tiling by setting "tile=0" however, you can only do this for
    images under 400 pixels in both dimensions, if the image is over 400 pixels
    in any dimension, and you disabled tiling, a usage error will be thrown.

    The "overlap" argument can be used to specify the overlap amount of each
    tile in pixels, it must be greater than or equal to 0, and defaults to 8.

    The "scale" argument can be used to specify the output scale of the image regardless of the models
    scale, this equates to an image resize on each tile output of the model as necessary, with auto selected
    resizing algorithm for the best quality. This is effectively equivalent to basic image resizing of
    the upscaled output post upscale, just with somewhat reduced memory overhead as it occurs during tiling.
    When this argument is not specified, the scale of the model architecture is used and no resizing occurs.
    This argument must be greater than or equal to 1.

    The "pre-resize" argument is a boolean value determining if the processing
    should take place before or after the image is resized by dgenerate.

    x4.bin: https://github.com/nihui/realsr-ncnn-vulkan/blob/master/models/models-DF2K/x4.bin
    x4.param: https://github.com/nihui/realsr-ncnn-vulkan/blob/master/models/models-DF2K/x4.param

    Example: "upscaler-ncnn;model=x4.bin;param=x4.param;tile=256;overlap=16;use-gpu=True"
    """

    NAMES = ['upscaler-ncnn']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    FILE_ARGS = {
        'model': {'mode': 'in', 'filetypes': [('Models', ['*.bin'])]},
        'param': {'mode': 'in', 'filetypes': [('Params', ['*.param'])]},
    }

    def __init__(self,
                 model: str,
                 param: str,
                 use_gpu: bool = False,
                 gpu_index: int = 0,
                 threads: int | str = "auto",
                 blocktime: int | None = None,
                 winograd: bool | None = None,
                 sgemm: bool | None = None,
                 tile: int = 400,
                 overlap: int = 8,
                 scale: int | None = None,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param model: NCNN compatible upscaler model on disk, or at a URL
        :param param: NCNN model param file on disk, or at a URL
        :param factor: upscale factor
        :param use_gpu: use a GPU?
        :param gpu_index: which GPU to use, zero indexed.
        :param threads: Number of CPU threads, or "auto", or "half"
        :param tile: specifies the tile size for tiled upscaling, it must be divisible by 2, and defaults to 400.
            Specifying values over 400 is not supported.
        :param overlap: the overlap amount of each tile in pixels, it must be greater than or equal to 0, and defaults to 8.
        :param scale: the scale factor to use for the upscaler, it must be greater than 1, and defaults to None (meaning use the model's scale).
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """

        super().__init__(**kwargs)

        if _ncnn_model is None:
            raise _exceptions.ImageProcessorError(
                'Cannot use ncnn-upscaler without ncnn installed.')

        if isinstance(tile, str):
            tile = tile.lower()
            if tile != 'auto':
                raise self.argument_error(
                    f'Argument "tile" passed unrecognized string: "{tile}", expected "auto".')
        else:
            if tile != 0 and tile < 2:
                raise self.argument_error('Argument "tile" must be greater than 2, or exactly 0.')

            if tile % 2 != 0:
                raise self.argument_error('Argument "tile" must be divisible by 2.')

        if tile > 400:
            raise self.argument_error('Argument "tile" may not be greater than 400.')

        if overlap < 0:
            raise self.argument_error('Argument "overlap" must be greater than or equal to 0.')

        if tile != 0 and tile < overlap:
            raise self.argument_error('Argument "tile" must be greater than "overlap".')

        if gpu_index < 0:
            raise self.argument_error('Argument "gpu-index" must be greater than or equal to 0.')

        if scale is not None and scale < 1:
            raise self.argument_error('Argument "scale" must be greater than or equal to 1.')

        if isinstance(threads, int) and threads < 1:
            raise self.argument_error(
                'Argument "cpu-threads" must be greater than or equal to 1, or "auto" or "half".')

        if use_gpu:
            if winograd is not None:
                raise self.argument_error(
                    'Argument "winograd" cannot be specified when running on the gpu.'
                )
            if sgemm is not None:
                raise self.argument_error(
                    'Argument "sgemm" cannot be specified when running on the gpu.'
                )
            if blocktime is not None:
                raise self.argument_error(
                    'Argument "blocktime" cannot be specified when running on the gpu.'
                )

        if blocktime is not None:
            if blocktime < 0:
                raise self.argument_error(
                    'Argument "blocktime" cannot be less than 0.'
                )

            if blocktime > 400:
                raise self.argument_error(
                    'Argument "blocktime" cannot be greater than 400.'
                )

        if _webcache.is_downloadable_url(model):
            try:
                self._model_path = _hfhub.webcache_or_hf_blob_download(
                    model, local_files_only=self.local_files_only
                )
            except Exception as e:
                raise self.argument_error(
                    f'Could not download argument "model": "{model}", error: {e}') from e
        else:
            self._model_path = model

        if _webcache.is_downloadable_url(param):
            try:
                self._param_path = _hfhub.webcache_or_hf_blob_download(
                    param, local_files_only=self.local_files_only
                )
            except Exception as e:
                raise self.argument_error(
                    f'Could not download argument "param": "{param}", error: {e}') from e
        else:
            self._param_path = param

        if not os.path.exists(self._model_path):
            raise self.argument_error(f'Model file does not exist: {self._model_path}')

        if not os.path.exists(self._param_path):
            raise self.argument_error(f'Param file does not exist: {self._param_path}')

        self._tile = tile
        self._overlap = overlap
        self._scale = scale
        self._pre_resize = pre_resize

        self._threads = threads if isinstance(threads, int) else psutil.cpu_count(logical=True)

        if threads == "half":
            self._threads = self._threads // 2

        self._gpu_index = gpu_index
        self._use_gpu = use_gpu
        self._winograd = winograd
        self._sgemm = sgemm
        self._blocktime = blocktime

        def broadcast_check(data: _ncnn_model.NCNNBroadcastData):
            if data.input_channels != 3 or data.output_channels != 3:
                raise _ncnn_model.NCNNModelLoadError(
                    f'No support for input channels or output channels not equal to 3. '
                    f'model input channels: {data.input_channels}, '
                    f'model output channels: {data.output_channels}')

        try:
            # destroy last diffusers pipeline in VRAM
            self._pipeline_fence()

            self._model = _ncnn_model.NCNNUpscaleModel(
                self._param_path,
                self._model_path,
                use_gpu=use_gpu,
                gpu_index=gpu_index,
                threads=self._threads,
                openmp_blocktime=self._blocktime,
                use_winograd_convolution=self._winograd,
                use_sgemm_convolution=self._sgemm,
                broadcast_data_check=broadcast_check)
        except _ncnn_model.NCNNGPUIndexError as e:
            raise self.argument_error(str(e)) from e
        except _ncnn_model.NCNNNoGPUError as e:
            raise self.argument_error(str(e)) from e
        except _ncnn_model.NCNNModelLoadError as e:
            raise self.argument_error(f'Unsupported NCNN model: {e}') from e

    def _pipeline_fence(self):
        if self._use_gpu:
            active_pipe = _pipelines.get_last_called_pipeline()
            if active_pipe is not None \
                    and _pipelines.get_torch_device(active_pipe).index == self._gpu_index:
                # get rid of this reference immediately
                # noinspection PyUnusedLocal
                active_pipe = None

                # fence, a vulkan context cannot exist
                # with a diffusers torch diffusion pipeline
                # in VRAM without massive issues.
                _pipelines.destroy_last_called_pipeline()

    def _process_upscale(self, image, model):

        tile = self._tile

        in_img = _stack_images([image])

        if tile == 0:
            if image.width > 400 or image.height > 400:
                raise self.argument_error(
                    'upscaler-ncnn cannot handle argument value "tile=0" '
                    'with images larger than 400 pixels.')

            return _output_to_pil(self._model(in_img)[0])

        oom = True

        while oom:
            try:
                steps = \
                    in_img.shape[0] * _upscale_tiler.get_tiled_scale_steps(
                        in_img.shape[3],
                        in_img.shape[2],
                        tile_x=tile,
                        tile_y=tile,
                        overlap=self._overlap)

                pbar = tqdm.auto.tqdm(total=steps)

                output = _upscale_tiler.tiled_scale(
                    in_img,
                    model,
                    tile_x=tile,
                    tile_y=tile,
                    overlap=self._overlap,
                    upscale_amount=_types.default(self._scale, model.broadcast_data.scale),
                    pbar=pbar.update)

                oom = False
            except _ncnn_model.NCNNExtractionFailure as e:
                pbar.close()
                tile //= 2
                _messages.warning(
                    f'Reducing tile size to {tile} and retrying due to running out of memory.'
                )
                if tile < 128:
                    raise e

        return _output_to_pil(output[0])

    def _process(self, image):
        # destroy last diffusers pipeline in VRAM
        self._pipeline_fence()

        try:
            return self._process_upscale(image, self._model)
        except _ncnn_model.NCNNExtractionFailure as e:
            raise dgenerate.OutOfMemoryError(e) from e

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize) -> PIL.Image.Image:
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "UpscalerNCNNProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


__all__ = _types.module_all()
