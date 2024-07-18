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
import gc
import os.path
import typing

import PIL.Image
import PIL.Image
import numpy
import psutil
import tqdm.auto

import dgenerate
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.imageprocessors.ncnn_model as _ncnn_model
import dgenerate.types as _types
import dgenerate.webcache as _webcache
import dgenerate.imageprocessors.upscale_tiler as _upscale_tiler
import dgenerate.messages as _messages


class _UnsupportedModelError(Exception):
    """model is not of a supported type."""
    pass


def _stack_images(images: list) -> numpy.ndarray:
    out = []
    for img in images:
        img = img.convert('RGB')
        img = (numpy.array(img).transpose((2, 0, 1)) / 255.0).astype(numpy.float32)  # Convert to (c, h, w) format
        out.append(img)
    return numpy.stack(out)


def _output_to_pil(output):
    return PIL.Image.fromarray(
        (output.transpose(1, 2, 0) * 255).astype(numpy.uint8))


class UpscalerNCNNProcessor(_imageprocessor.ImageProcessor):
    """
    Implements tiled upscaling with NCNN upscaler models.

    It is not recommended to use this as a pre-processor or post-processor
    on the same gpu that diffusion is being preformed on. The vulkan allocator
    does not play nice with the torch cuda allocator and it will likely hard
    crash your entire system. If you want to use it that way, set "gpu-index"
    to a gpu that diffusion is not going to be running on.

    Downloaded model / param files are cached in the dgenerate web cache on disk
    until the cache expiry time for the file is met.

    The "model" argument should be a path or URL to a NCNN compatible upscaler model.

    The "param" argument should be a path or URL to the NCNN param file for the model.

    The "use-gpu" argument determines if the gpu is used, defaults to False.

    The "gpu-index" argument determines which gpu is used, it is 0 indexed.

    The "cpu-threads" argument determines the number of cpu threads used, the
    default value is "auto" which uses the maximum amount. You may also pass
    "half" to use half the cpus logical thread count.

    The "tile" argument can be used to specify the tile size for tiled upscaling, it
    must be divisible by 2 and less than or equal to 400. The default is 400. Tile size
    is limited to a max of 400 due to memory allocator issues in ncnn.

    The "overlap" argument can be used to specify the overlap amount of each
    tile in pixels, it must be greater than or equal to 0, and defaults to 8.

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

    def __init__(self,
                 model: str,
                 param: str,
                 use_gpu: bool = False,
                 gpu_index: int = 0,
                 cpu_threads: typing.Union[int, str] = "auto",
                 tile: int = 400,
                 overlap: int = 8,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param model: NCNN compatible upscaler model on disk, or at a URL
        :param param: NCNN model param file on disk, or at a URL
        :param factor: upscale factor
        :param use_gpu: use a GPU?
        :param gpu_index: which GPU to use, zero indexed.
        :param cpu_threads: Number of CPU threads, or "auto", or "half"
        :param tile: specifies the tile size for tiled upscaling, it must be divisible by 2, and defaults to 400.
            Specifying values over 400 is not supported.
        :param overlap: the overlap amount of each tile in pixels, it must be greater than or equal to 0, and defaults to 8.
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """

        super().__init__(**kwargs)

        if type(tile) is str:
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

        if tile < overlap:
            raise self.argument_error('Argument "tile" must be greater than "overlap".')

        if gpu_index < 0:
            raise self.argument_error('Argument "gpu-index" must be greater than or equal to 0.')

        if isinstance(cpu_threads, int) and cpu_threads < 1:
            raise self.argument_error(
                'Argument "cpu-threads" must be greater than or equal to 1, or "auto" or "half".')

        if _webcache.is_downloadable_url(model):
            self._model_path = _webcache.create_web_cache_file(model)[1]
        else:
            self._model_path = model

        if _webcache.is_downloadable_url(param):
            self._param_path = _webcache.create_web_cache_file(param)[1]
        else:
            self._param_path = param

        if not os.path.exists(self._model_path):
            raise self.argument_error(f'Model file does not exist: {self._model_path}')

        if not os.path.exists(self._param_path):
            raise self.argument_error(f'Param file does not exist: {self._param_path}')

        self._tile = tile
        self._overlap = overlap
        self._pre_resize = pre_resize
        self._use_gpu = use_gpu
        self._gpu_index = gpu_index
        self._threads = cpu_threads if isinstance(cpu_threads, int) else psutil.cpu_count(logical=True)

        if cpu_threads == "half":
            self._threads = self._threads // 2

    def _process_upscale(self, image, model):

        if model.broadcast_data.input_channels != 3 or model.broadcast_data.output_channels != 3:
            raise ValueError(
                f'No support for input channels or output channels not equal to 3. '
                f'model input channels: {model.broadcast_data.input_channels}, '
                f'model output channels: {model.broadcast_data.output_channels}')

        tile = self._tile

        in_img = _stack_images([image])

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
                    upscale_amount=model.broadcast_data.scale,
                    pbar=pbar.update)

                oom = False
            except _ncnn_model.NCNNExtractionFailure as e:
                pbar.close()
                tile //= 2
                _messages.log(
                    f'Reducing tile size to {tile} and retrying due to running out of memory.',
                    level=_messages.WARNING)
                if tile < 128:
                    raise e

        return _output_to_pil(output[0])

    def _process(self, image):

        try:
            model = _ncnn_model.NCNNUpscaleModel(
                self._param_path,
                self._model_path,
                use_gpu=self._use_gpu,
                gpu_index=self._gpu_index,
                threads=self._threads)
        except _ncnn_model.NCNNGPUIndexError as e:
            raise self.argument_error(str(e))
        except _ncnn_model.NCNNNoGPUError as e:
            raise self.argument_error(str(e))
        except ValueError as e:
            raise self.argument_error(f'Unsupported NCNN model: {e}')

        try:
            return self._process_upscale(image, model)
        except _ncnn_model.NCNNExtractionFailure as e:
            raise dgenerate.OutOfMemoryError(e)
        except _ncnn_model.NCNNIncorrectScaleFactor as e:
            raise self.argument_error(f'argument "factor" is incorrect: {e}')
        finally:
            del model
            gc.collect()

    def __str__(self):
        args = [
            ('model', self._model_path),
            ('param', self._param_path),
            ('use-gpu', self._use_gpu),
            ('gpu-index', self._gpu_index),
            ('cpu-threads', self._threads),
            ('tile', self._tile),
            ('overlap', self._overlap),
            ('pre_resize', self._pre_resize)
        ]

        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize) -> PIL.Image.Image:
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        if not self._pre_resize:
            return self._process(image)
        return image


__all__ = _types.module_all()
