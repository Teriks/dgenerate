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

import ncnn
import numpy

import dgenerate.types as _types


class NCNNException(Exception):
    pass


class NCNNModelLoadError(NCNNException):
    pass


class NCNNExtractionFailure(NCNNException):
    pass


class NCNNGPUIndexError(NCNNException):
    pass


class NCNNNoGPUError(NCNNException):
    pass


class NCNNBroadcastData:
    def __init__(self, scale, input_channels, output_channels, number_of_filters, float_precision):
        self.scale = scale
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.number_of_filters = number_of_filters
        self.float_precision = float_precision


class _ParamLayer:
    def __init__(self, op_type, name, num_inputs, num_outputs, inputs, outputs, param_dict):
        self.op_type = op_type
        self.name = name
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.inputs = inputs
        self.outputs = outputs
        self.param_dict = param_dict


class NCNNUpscaleModel:
    def __init__(self,
                 param_file: str,
                 bin_file: str,
                 use_gpu: bool = False,
                 gpu_index: int = 0,
                 threads: int = 4,
                 openmp_blocktime: int | None = None,
                 use_winograd_convolution: bool | None = False,
                 use_sgemm_convolution: bool | None = False,
                 broadcast_data_check: typing.Callable[[NCNNBroadcastData], None] | None = None):
        """
        Initialize the NCNN upscale model.
        """
        self.net = ncnn.Net()
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index
        self.threads = threads

        if use_winograd_convolution is not None:
            self.use_winograd_convolution = use_winograd_convolution
        else:
            self.use_winograd_convolution = self.net.opt.use_winograd_convolution

        if use_sgemm_convolution is not None:
            self.use_sgemm_convolution = use_sgemm_convolution
        else:
            self.use_sgemm_convolution = self.net.opt.use_sgemm_convolution

        if openmp_blocktime is not None:
            if openmp_blocktime < 0:
                raise ValueError('openmp_blocktime may not be less than 0.')
            if openmp_blocktime > 400:
                raise ValueError('openmp_blocktime may not be greater than 400.')
            self.openmp_blocktime = openmp_blocktime
        else:
            self.openmp_blocktime = self.net.opt.openmp_blocktime

        self._load_model(param_file, bin_file, broadcast_data_check)

    def _load_model(self,
                    param_file: str,
                    bin_file: str,
                    broadcast_data_check: typing.Callable[[NCNNBroadcastData], None] | None = None):

        with open(param_file, 'rt') as file:
            lines = [line for line in file if line.strip()]
            self.broadcast_data = self._get_broadcast_data(lines)

            if broadcast_data_check:
                broadcast_data_check(self.broadcast_data)

            self.input_layer_name, self.output_layer_name = self._get_input_output(lines)

        self._configure_net()
        self.net.load_param(param_file)
        self.net.load_model(bin_file)

    def _configure_net(self):
        if self.broadcast_data.float_precision == "fp16":
            self.net.opt.use_fp16_packed = True
            self.net.opt.use_fp16_storage = True
            self.net.opt.use_fp16_arithmetic = True
        else:
            self.net.opt.use_fp16_packed = False
            self.net.opt.use_fp16_storage = False
            self.net.opt.use_fp16_arithmetic = False

        if self.use_gpu:
            gpu_count = ncnn.get_gpu_count()
            if gpu_count == 0:
                raise NCNNNoGPUError(
                    'NCNN could not find any GPUs.')
            if self.gpu_index > gpu_count:
                raise NCNNGPUIndexError(
                    f'GPU index {self.gpu_index} does not exist.')

            self.net.opt.use_vulkan_compute = True
            self.net.set_vulkan_device(self.gpu_index)
        else:
            self.net.opt.num_threads = self.threads
            self.net.opt.use_winograd_convolution = self.use_winograd_convolution
            self.net.opt.use_sgemm_convolution = self.use_sgemm_convolution
            self.net.opt.openmp_blocktime = self.openmp_blocktime

    @staticmethod
    def _parse_param_layer(layer_str: str) -> _ParamLayer:
        param_list = layer_str.strip().split()
        if len(param_list) < 4:
            raise NCNNModelLoadError(
                f"Layer string does not have enough elements: {layer_str}")

        op_type, name = param_list[:2]
        num_inputs = int(param_list[2])
        num_outputs = int(param_list[3])
        input_end = 4 + num_inputs
        output_end = input_end + num_outputs
        inputs = list(param_list[4:input_end])
        outputs = list(param_list[input_end:output_end])

        params = param_list[output_end:]
        param_dict = {}
        for param_str in params:
            ks, vs = param_str.split("=")
            k = int(ks)
            if k < 0:
                v = [float(vi) if "." in vi or "e" in vi else int(vi) for vi in vs.split(",")]
                k = abs(k + 23300)
            else:
                v = float(vs) if "." in vs or "e" in vs else int(vs)
            param_dict[k] = v

        return _ParamLayer(op_type, name, num_inputs, num_outputs, inputs, outputs, param_dict)

    @staticmethod
    def _get_nf_and_in_nc(param_dict: dict) -> tuple[int, int]:
        if 0 not in param_dict or 1 not in param_dict or 6 not in param_dict:
            raise NCNNModelLoadError(
                "Required parameters missing in the layer string.")

        nf = param_dict[0]
        kernel_w = param_dict[1]
        kernel_h = param_dict.get(11, kernel_w)
        weight_data_size = param_dict[6]

        if not all(isinstance(x, int) for x in [nf, kernel_w, kernel_h, weight_data_size]):
            raise NCNNModelLoadError(
                "Out nc, kernel width and height, and weight data size must all be ints.")

        in_nc = weight_data_size // nf // kernel_w // kernel_h

        return nf, in_nc

    @staticmethod
    def _get_broadcast_data(param_lines: list[str]) -> NCNNBroadcastData:
        scale = 1.0
        pixel_shuffle = 1
        found_first_conv = False
        in_nc = 0
        nf = 0
        fp = "fp32"
        current_conv = None

        for i, line in enumerate(param_lines[2:]):
            param_layer = NCNNUpscaleModel._parse_param_layer(line)

            if param_layer.op_type == "Interp":
                try:
                    if param_lines[i + 3].strip().split()[0] != "BinaryOp" and param_layer.param_dict[0] != 0:
                        scale *= float(param_layer.param_dict[2])
                except IndexError:
                    scale *= float(param_layer.param_dict[2])
            elif param_layer.op_type == "PixelShuffle":
                shuffle_factor = int(param_layer.param_dict[0])
                scale *= shuffle_factor
                pixel_shuffle *= shuffle_factor
            elif param_layer.op_type in ("Convolution", "Convolution1D", "ConvolutionDepthWise"):
                if not found_first_conv:
                    nf, in_nc = NCNNUpscaleModel._get_nf_and_in_nc(param_layer.param_dict)
                    found_first_conv = True
                stride = int(param_layer.param_dict.get(3, 1))
                scale /= stride
                current_conv = param_layer.param_dict
            elif param_layer.op_type in ("Deconvolution", "DeconvolutionDepthWise"):
                if not found_first_conv:
                    nf, in_nc = NCNNUpscaleModel._get_nf_and_in_nc(param_layer.param_dict)
                    found_first_conv = True
                stride = int(param_layer.param_dict.get(3, 1))
                scale *= stride
                current_conv = param_layer.param_dict

        if current_conv is None:
            raise NCNNModelLoadError(
                "Cannot broadcast, model has no Convolution layers.")

        out_nc = int(current_conv[0]) // (pixel_shuffle ** 2)

        if scale < 1 or scale % 1 != 0:
            raise NCNNModelLoadError(
                f"Model not supported, scale {scale} is not an integer or is less than 1.")

        return NCNNBroadcastData(int(scale), in_nc, out_nc, nf, fp)

    def _get_input_output(self, param_lines: list[str]) -> tuple[str, str]:
        first = self._parse_param_layer(param_lines[2])
        last = self._parse_param_layer(param_lines[-1])
        return first.outputs[0], last.outputs[0]

    @staticmethod
    def _convert_to_uint8(image: numpy.ndarray):
        # Ensure the values are in the correct range
        return (numpy.clip(image, 0.0, 1.0) * 255).round().astype(numpy.uint8)

    def __call__(self, images: numpy.ndarray) -> numpy.ndarray:
        """
        Upscale with NCNN

        :param images: (batch, c, h, w) float32 pixels 0.0 to 1.0 normalized
        :return: same as input
        """
        results = []
        for img in images:
            ex = self.net.create_extractor()

            img = self._convert_to_uint8(img.transpose((1, 2, 0)))

            mat_in = ncnn.Mat.from_pixels(
                img,
                ncnn.Mat.PixelType.PIXEL_RGB,
                img.shape[1],
                img.shape[0]
            )

            mean_vals = []
            norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
            mat_in.substract_mean_normalize(mean_vals, norm_vals)

            ex.input(self.input_layer_name, mat_in)
            ret, mat_out = ex.extract(self.output_layer_name)

            if ret != 0:
                raise NCNNExtractionFailure(
                    "NCNN extraction failed, GPU out of memory.")

            results.append(numpy.clip(mat_out, 0.0, 1.0, dtype=numpy.float32))

            del ex, mat_in, mat_out
            gc.collect()

        return numpy.array(results)

    def __del__(self):
        if hasattr(self, 'net'):
            del self.net


__all__ = _types.module_all()
