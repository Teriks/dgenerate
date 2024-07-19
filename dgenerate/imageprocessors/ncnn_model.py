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

import ncnn
import numpy

import dgenerate.types as _types


class NCNNExtractionFailure(Exception):
    pass


class NCNNIncorrectScaleFactor(Exception):
    pass


class NCNNGPUIndexError(Exception):
    pass


class NCNNNoGPUError(Exception):
    pass


class NCNNBroadcastData:
    scale: int
    input_channels: int
    output_channels: int
    number_of_filters: int
    float_precision: str

    def __init__(self, scale, input_channels, output_channels, number_of_filters, float_precision):
        self.scale = scale
        self.input_channels = input_channels
        self.number_of_filters = number_of_filters
        self.float_precision = float_precision
        self.output_channels = output_channels


class NCNNUpscaleModel:
    def __init__(self,
                 param_file: str,
                 bin_file: str,
                 use_gpu: bool = False,
                 gpu_index: int = 0,
                 threads: int = 4):
        """
        Initialize the NCNN upscale model.
        """
        with open(param_file, 'rt') as file:
            lines = [line for line in file if line.strip()]
            self.input_layer_name, self.output_layer_name = self._get_input_output(lines)
            self.broadcast_data = self._get_broadcast_data(lines)

        net = ncnn.Net()

        if self.broadcast_data.float_precision == "fp16":
            net.opt.use_fp16_packed = True
            net.opt.use_fp16_storage = True
            net.opt.use_fp16_arithmetic = True
        else:
            net.opt.use_fp16_packed = False
            net.opt.use_fp16_storage = False
            net.opt.use_fp16_arithmetic = False

        if use_gpu:
            gpu_count = ncnn.get_gpu_count()

            if gpu_count == 0:
                raise NCNNNoGPUError('NCNN could not find any gpus.')
            if gpu_index > gpu_count:
                raise NCNNGPUIndexError(f'GPU index {gpu_index} does not exist.')

            net.opt.use_vulkan_compute = True
            net.set_vulkan_device(gpu_index)
        else:
            net.opt.num_threads = threads

        self.net = net
        self.net.load_param(param_file)
        self.net.load_model(bin_file)

        self.use_gpu = use_gpu
        self.gpu_index = gpu_index
        self.threads = threads

    @staticmethod
    def _parse_param_layer(layer_str: str) -> tuple:
        param_list = layer_str.strip().split()
        if len(param_list) < 4:
            raise ValueError(f"Layer string does not have enough elements: {layer_str}")

        op_type, name = param_list[:2]
        if op_type == "MemoryData":
            raise ValueError("This NCNN param file contains invalid layers.")

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
                v = []
                for vi in vs.split(","):
                    vi = float(vi) if "." in vi or "e" in vi else int(vi)
                    v.append(vi)
                k = abs(k + 23300)
                ks = str(k)
            elif "." in vs or "e" in vs:
                v = float(vs)
            else:
                v = int(vs)

            param_dict[k] = v

        return op_type, name, num_inputs, num_outputs, inputs, outputs, param_dict

    @staticmethod
    def _get_nf_and_in_nc(param_dict: dict) -> tuple[int, int]:
        if 0 not in param_dict or 1 not in param_dict or 6 not in param_dict:
            raise ValueError("Required parameters missing in the layer string.")

        nf = param_dict[0]
        kernel_w = param_dict[1]
        kernel_h = param_dict.get(11, kernel_w)
        weight_data_size = param_dict[6]

        if not all(isinstance(x, int) for x in [nf, kernel_w, kernel_h, weight_data_size]):
            raise TypeError("Out nc, kernel width and height, and weight data size must all be ints.")

        in_nc = weight_data_size // nf // kernel_w // kernel_h

        return nf, in_nc

    @staticmethod
    def _get_broadcast_data(param_lines: list[str]) -> NCNNBroadcastData:
        scale = 1.0
        pixel_shuffle = 1
        found_first_conv = False
        in_nc = 0
        out_nc = 0
        nf = 0
        fp = "fp32"
        current_conv = None

        for i, line in enumerate(param_lines[2:]):
            op_type, \
                name, \
                num_inputs, \
                num_outputs, \
                inputs, \
                outputs, \
                param_dict = NCNNUpscaleModel._parse_param_layer(line)

            if op_type == "Interp":
                try:
                    if param_lines[i + 3].strip().split()[0] != "BinaryOp" and param_dict[0] != 0:
                        scale *= float(param_dict[2])
                except IndexError:
                    scale *= float(param_dict[2])
            elif op_type == "PixelShuffle":
                shuffle_factor = int(param_dict[0])
                scale *= shuffle_factor
                pixel_shuffle *= shuffle_factor
            elif op_type in ("Convolution", "Convolution1D", "ConvolutionDepthWise"):
                if not found_first_conv:
                    nf, in_nc = NCNNUpscaleModel._get_nf_and_in_nc(param_dict)
                    found_first_conv = True
                stride = int(param_dict.get(3, 1))  # Use a default value of 1 if key 3 is not present
                scale /= stride
                current_conv = param_dict
            elif op_type in ("Deconvolution", "DeconvolutionDepthWise"):
                if not found_first_conv:
                    nf, in_nc = NCNNUpscaleModel._get_nf_and_in_nc(param_dict)
                    found_first_conv = True
                stride = int(param_dict.get(3, 1))  # Use a default value of 1 if key 3 is not present
                scale *= stride
                current_conv = param_dict

        if current_conv is None:
            raise ValueError("Cannot broadcast; model has no Convolution layers.")

        out_nc = int(current_conv[0]) // (pixel_shuffle ** 2)

        if scale < 1 or scale % 1 != 0:
            raise ValueError(f"Model not supported, scale {scale} is not an integer or is less than 1.")

        return NCNNBroadcastData(int(scale), in_nc, out_nc, nf, fp)

    def _get_input_output(self, param_lines: list[str]) -> tuple[str, str]:
        first = self._parse_param_layer(param_lines[2])
        last = self._parse_param_layer(param_lines[-1])
        return first[-2][0], last[-2][0]

    def __del__(self):
        if hasattr(self, 'net'):
            del self.net

    def __call__(self, images: numpy.ndarray) -> numpy.ndarray:
        """
        Upscale with NCNN

        :param images: (batch, c, h, w) float32 pixels 0.0 to 1.0 normalized
        :return: same as input
        """
        results = []
        for img in images:
            ex = self.net.create_extractor()

            # Convert into h, w, c from c, h, w and 0.0 - 1.0 -> 0 - 255
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
            ex.input(self.input_layer_name, mat_in)
            ret, mat_out = ex.extract(self.output_layer_name)

            if ret != 0:
                raise NCNNExtractionFailure("NCNN extraction failed, GPU out of memory.")

            # Return c, h, w
            results.append(numpy.array(mat_out).astype(numpy.float32))

            del ex, mat_in, mat_out
            gc.collect()

        return numpy.array(results)


__all__ = _types.module_all()
