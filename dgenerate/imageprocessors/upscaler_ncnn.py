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
import typing

import PIL.Image
import PIL.Image
import ncnn
import numpy
import tqdm.auto

import dgenerate
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.imageprocessors.upscaler as _t_upscaler
import dgenerate.messages as _messages
import dgenerate.types as _types
import dgenerate.webcache as _webcache


class _UnsupportedModelError(Exception):
    """model is not of a supported type."""
    pass


class _NCNNExtractionFailure(Exception):
    pass


class _NCNNIncorrectScaleFactor(Exception):
    pass


class _NCNNGPUIndexError(Exception):
    pass


class _NCNNNoGPUError(Exception):
    pass


class NCNNUpscaleModel:
    def __init__(self,
                 param_file: str,
                 bin_file: str,
                 use_gpu: bool = False,
                 gpu_index: int = 0,
                 threads: int = 4):
        self.net = ncnn.Net()

        if use_gpu:
            gpu_count = ncnn.get_gpu_count()

            if gpu_count > 0:
                if gpu_index > gpu_count:
                    raise _NCNNGPUIndexError('specified gpu index is not visible to NCNN.')
                self.net.opt.use_vulkan_compute = True
            else:
                raise _NCNNNoGPUError('NCNN cannot find any gpus.')

        self.net.load_param(param_file)
        self.net.load_model(bin_file)
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index
        self.threads = threads
        self.input, self.output = self._get_input_output(param_file)

        if self.use_gpu:
            self.net.set_vulkan_device(self.gpu_index)
        else:
            ncnn.set_omp_num_threads(threads)

    @staticmethod
    def _parse_param_layer(layer_str: str) -> tuple:
        param_list = layer_str.strip().split()
        op_type, name = param_list[:2]
        if op_type == "MemoryData":
            raise ValueError("This NCNN param file contains invalid layers")
        num_inputs = int(param_list[2])
        num_outputs = int(param_list[3])
        input_end = 4 + num_inputs
        output_end = input_end + num_outputs
        inputs = list(param_list[4:input_end])
        outputs = list(param_list[input_end:output_end])
        return op_type, name, num_inputs, num_outputs, inputs, outputs

    def _get_input_output(self, param_file: str) -> tuple:
        first = None
        last = None
        with open(param_file, 'rt') as file:
            next(file)
            next(file)
            first = self._parse_param_layer(next(file))
            for line in file:
                last = self._parse_param_layer(line)
        return first[-1][0], last[-1][0]

    def __call__(self, images: numpy.ndarray) -> numpy.ndarray:
        """
        Upscale with NCNN

        :param images: (batch, c, h, w) float32 pixels 0.0 to 1.0 normalized

        :return: same as input
        """

        ex = self.net.create_extractor()

        results = []

        for img in images:
            # expect normalized values 0.0 to 1.0

            # Convert into h, w, c from c, h, w and 0.0 - 0.1 -> 0 - 255
            img = (img.transpose((1, 2, 0)) * 255.0).astype(numpy.uint8)

            # Convert image to ncnn Mat
            mat_in = ncnn.Mat.from_pixels(
                img,
                ncnn.Mat.PixelType.PIXEL_BGR,
                img.shape[1],
                img.shape[0]
            )

            # Normalize image (required)
            mean_vals = [0, 0, 0]
            norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
            mat_in.substract_mean_normalize(mean_vals, norm_vals)

            # Make sure the input and output names match the param file

            ex.input(self.input, mat_in)
            ret, mat_out = ex.extract(self.output)

            if ret != 0:
                raise _NCNNExtractionFailure("Extraction failed, GPU OOM?")

            # return c, h, w
            results.append(numpy.array(mat_out))

        return numpy.array(results)


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
                try:
                    if tile_blending:
                        out[:, :, y_start:y_end, x_start:x_end] += ps * mask
                        out_div[:, :, y_start:y_end, x_start:x_end] += mask
                    else:
                        out[:, :, y_start:y_end, x_start:x_end] += ps
                except ValueError:
                    raise _NCNNIncorrectScaleFactor(
                        f'scaling factor value {upscale_amount} is not correct for this NCNN model.')

                # Update progress bar if provided
                if pbar is not None:
                    pbar.update(1)

        # Divide the accumulated output by the mask to get the final result
        if tile_blending:
            output[b:b + 1] = out / out_div
        else:
            output[b:b + 1] = out

    return output


def _stack_images(images: list) -> numpy.ndarray:
    out = []
    for img in images:
        img = img.convert('RGB')
        img = (numpy.array(img).transpose((2, 0, 1)) / 255.0).astype(numpy.float32)  # Convert to (c, h, w) format
        out.append(img)
    return numpy.stack(out)


class UpscalerNCNNProcessor(_imageprocessor.ImageProcessor):
    """
    Implements tiled upscaling with NCNN upscaler models.

    The "model" argument should be a path to a NCNN compatible upscaler model.

    The "params" argument should be a path to the NCNN params file for the model.

    Downloaded model / param files are cached in the dgenerate web cache on disk until the
    cache expiry time for the file is met.

    The "factor" argument should be set to the upscale factor, such as 4. This
    value must be correct for the supplied model to ensure correct operation.

    The "cpu-threads" argument determines the number of cpu threads used.

    The "use-gpu" argument determines if the gpu is used, this defaults to
    False for reasons outlined below.

    Using the GPU is not recommended if you are trying to use this processor
    as a preprocessor or postprocessor for diffusion, as the vulkan allocator
    really does not play nice with the torch cuda allocator, and it is almost
    guaranteed to fail with an out of memory error.  It is fine to use if you are
    doing a one shot upscale from the command line, or if you are only
    ever using the "upscaler-ncnn" image processor in your config or from
    the console UI. Initializing the vulkan allocator on the GPU will cause
    other torch operations to fail, or NCNN operations to fail if a torch cuda
    allocator exists in memory, and process crashes / segfaults are likely.

    The "gpu-index" argument determines which gpu is used, it is 0 indexed.

    The "tile" argument can be used to specify the tile size for tiled upscaling, it must be divisible by 2, and
    defaults to 512. Specifying 0 disables tiling entirely.

    The "overlap" argument can be used to specify the overlap amount of each tile in pixels, it must be
    greater than or equal to 0, and defaults to 32.

    The "pre-resize" argument is a boolean value determining if the processing should take place before or
    after the image is resized by dgenerate.

    Example: "upscaler-ncnn;model=x4.bin;params=x4.params;factor=4;tile=256;overlap=16"
    """

    NAMES = ['upscaler-ncnn']

    def __init__(self,
                 model: str,
                 params: str,
                 factor: int,
                 cpu_threads: int = 4,
                 use_gpu: bool = False,
                 gpu_index: int = 0,
                 tile: typing.Union[int, str] = 512,
                 overlap: int = 32,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param model: NCNN compatible upscaler model on disk, or at a URL
        :param params: NCNN model params file on disk, or at a URL
        :param factor: upscale factor
        :param use_gpu: use a GPU?
        :param gpu_index: which GPU to use, zero indexed.
        :param channels: number of image channels, 1 through 4, default is 3
        :param tile: specifies the tile size for tiled upscaling, it must be divisible by 2, and defaults to 512.
            Specifying 0 disable tiling entirely.
        :param overlap: the overlap amount of each tile in pixels, it must be greater than or equal to 0, and defaults to 32.
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

        if tile != 0 and tile < overlap:
            raise self.argument_error('Argument "tile" must be greater than "overlap".')

        if overlap < 0:
            raise self.argument_error('Argument "overlap" must be greater than or equal to 0.')

        if factor < 1:
            raise self.argument_error('Argument "factor" must be greater than or equal to 1.')

        if gpu_index < 0:
            raise self.argument_error('Argument "gpu-index" must be greater than or equal to 0.')

        if cpu_threads < 1:
            raise self.argument_error('Argument "cpu-threads" must be greater than or equal to 1.')

        if _webcache.is_downloadable_url(model):
            self._model_path = _webcache.create_web_cache_file(model)
        else:
            self._model_path = model

        if _webcache.is_downloadable_url(params):
            self._params_path = _webcache.create_web_cache_file(params)
        else:
            self._params_path = params

        self._tile = tile
        self._overlap = overlap
        self._pre_resize = pre_resize
        self._use_gpu = use_gpu
        self._gpu_index = gpu_index
        self._factor = factor
        self._threads = cpu_threads

    def _process_upscale(self, image, model):

        oom = True

        tile = self._tile

        in_img = _stack_images([image])

        if tile == 0:
            output = model(in_img)[0]
            return PIL.Image.fromarray(
                (output.transpose(1, 2, 0) * 255).astype(numpy.uint8))

        while oom:
            try:
                steps = _t_upscaler._get_tiled_scale_steps(
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
                    upscale_amount=self._factor,
                    out_channels=3,
                    overlap=self._overlap,
                    upscale_model=model,
                    pbar=pbar)[0]

                oom = False
            except _NCNNExtractionFailure as e:
                pbar.close()
                tile //= 2
                _messages.log(
                    f'Reducing tile size to {tile} and retrying due to running out of memory.',
                    level=_messages.WARNING)
                if tile < 128:
                    raise e

        return PIL.Image.fromarray(
            (output.transpose(1, 2, 0) * 255).astype(numpy.uint8))

    def _process_scope(self, image):

        try:
            model = NCNNUpscaleModel(self._params_path,
                                     self._model_path,
                                     use_gpu=self._use_gpu,
                                     gpu_index=self._gpu_index,
                                     threads=self._threads)
        except _NCNNGPUIndexError as e:
            raise self.argument_error(str(e))
        except _NCNNNoGPUError as e:
            raise self.argument_error(str(e))
        except Exception as e:
            raise self.argument_error(f'Unsupported model file format: {e}')

        try:
            return self._process_upscale(image, model)
        except _NCNNExtractionFailure as e:
            raise dgenerate.OutOfMemoryError(e)
        except _NCNNIncorrectScaleFactor as e:
            raise self.argument_error(f'argument "factor" is incorrect: {e}')

    def _process(self, image):
        try:
            return self._process_scope(image)
        finally:
            gc.collect()

    def __str__(self):
        args = [
            ('model', self._model_path),
            ('params', self._params_path),
            ('factor', self._factor),
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
