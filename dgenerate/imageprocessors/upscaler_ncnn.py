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
import math
import typing

import PIL.Image
import PIL.Image
import ncnn
import numpy
import psutil
import tqdm.auto

import dgenerate
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.imageprocessors.ncnn_model as _ncnn_model
import dgenerate.types as _types
import dgenerate.webcache as _webcache


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


def _get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


def _tiled_scale(samples,
                 upscale_model,
                 tile_x=64,
                 tile_y=64,
                 overlap=8,
                 upscale_amount=4,
                 out_channels=3,
                 pbar=None):
    # Pre-calculate the scaled dimensions
    scaled_height = round(samples.shape[2] * upscale_amount)
    scaled_width = round(samples.shape[3] * upscale_amount)

    # Initialize the output tensor on CPU
    output = numpy.empty((samples.shape[0], out_channels, scaled_height, scaled_width))

    tile_blending = overlap > 0

    # Pre-calculate the feathering factor
    if tile_blending:
        feather = round(overlap * upscale_amount)
        feather_factor = 1.0 / feather

    # Iterate over each sample
    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        out = numpy.zeros((s.shape[0], out_channels, scaled_height, scaled_width))
        out_div = numpy.zeros_like(out)

        # Iterate over the tiles
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y:y + tile_y, x:x + tile_x]
                ps = upscale_model(s_in)

                if tile_blending:
                    mask = numpy.ones_like(ps)

                    # Apply feathering to the mask
                    for t in range(feather):
                        mask[:, :, t:1 + t, :] *= (feather_factor * (t + 1))
                        mask[:, :, -1 - t: -t, :] *= (feather_factor * (t + 1))
                        mask[:, :, :, t:1 + t] *= (feather_factor * (t + 1))
                        mask[:, :, :, -1 - t: -t] *= (feather_factor * (t + 1))

                # Calculate the indices for the output tensor
                y_start = round(y * upscale_amount)
                y_end = y_start + round(tile_y * upscale_amount)
                x_start = round(x * upscale_amount)
                x_end = x_start + round(tile_x * upscale_amount)

                # Update the output tensor
                if tile_blending:
                    out[:, :, y_start:y_end, x_start:x_end] += ps * mask
                    out_div[:, :, y_start:y_end, x_start:x_end] += mask
                else:
                    out[:, :, y_start:y_end, x_start:x_end] += ps

                # Update progress bar if provided
                if pbar is not None:
                    pbar.update(1)

        # Divide the accumulated output by the mask to get the final result
        if tile_blending:
            output[b:b + 1] = out / out_div
        else:
            output[b:b + 1] = out

    return output


class UpscalerNCNNProcessor(_imageprocessor.ImageProcessor):
    """
    Implements tiled upscaling with NCNN upscaler models.

    It is not recommended to use this as a preprocessor or postprocessor
    on the same gpu that diffusion is being preformed on. The vulkan allocator
    does not play nice with the torch cuda allocator and it will likely hard
    crash your entire system. If you want to use it that way, set "gpu-index"
    to a gpu that diffusion is not going to be running on.

    Downloaded model / param files are cached in the dgenerate web cache on disk
    until the cache expiry time for the file is met.

    The "model" argument should be a path or URL to a NCNN compatible upscaler model.

    The "params" argument should be a path or URL to the NCNN params file for the model.

    The "use-gpu" argument determines if the gpu is used, defaults to False, see note below.

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

    Example: "upscaler-ncnn;model=x4.bin;params=x4.params;factor=4;tile=256;overlap=16;use-gpu=True"
    """

    NAMES = ['upscaler-ncnn']

    def __init__(self,
                 model: str,
                 params: str,
                 use_gpu: bool = False,
                 gpu_index: int = 0,
                 cpu_threads: int | str = "auto",
                 tile: typing.Union[int, str] = 400,
                 overlap: int = 8,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param model: NCNN compatible upscaler model on disk, or at a URL
        :param params: NCNN model params file on disk, or at a URL
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

        if _webcache.is_downloadable_url(params):
            self._params_path = _webcache.create_web_cache_file(params)[1]
        else:
            self._params_path = params

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

        steps = _get_tiled_scale_steps(
            in_img.shape[3],
            in_img.shape[2],
            tile_x=tile,
            tile_y=tile,
            overlap=self._overlap)

        pbar = tqdm.auto.tqdm(total=steps)

        output = _tiled_scale(
            samples=_stack_images([image]),
            tile_y=self._tile,
            tile_x=self._tile,
            upscale_amount=model.broadcast_data.scale,
            out_channels=3,
            overlap=self._overlap,
            upscale_model=model,
            pbar=pbar)[0]

        return _output_to_pil(output)

    def _process_scope(self, image):

        try:
            model = _ncnn_model.NCNNUpscaleModel(
                self._params_path,
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
            result = self._process_upscale(image, model)
            del model
            return result
        except _ncnn_model.NCNNExtractionFailure as e:
            raise dgenerate.OutOfMemoryError(e)
        except _ncnn_model.NCNNIncorrectScaleFactor as e:
            raise self.argument_error(f'argument "factor" is incorrect: {e}')

    def _process(self, image):
        try:
            return self._process_scope(image)
        finally:
            gc.collect()
            if self._use_gpu:
                ncnn.destroy_gpu_instance()

    def __str__(self):
        args = [
            ('model', self._model_path),
            ('params', self._params_path),
            ('cpu-threads', self._threads),
            ('use-gpu', self._use_gpu),
            ('gpu-index', self._gpu_index),
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
