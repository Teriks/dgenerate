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

import math
import typing

import numpy
import torch

import dgenerate.types as _types

T = typing.TypeVar('T', numpy.ndarray, torch.Tensor)


def get_tiled_scale_steps(width: int, height: int, tile_x: int, tile_y: int, overlap: int):
    """
    Determine the number of progress tile steps required.

    :param width: image width
    :param height: image width
    :param tile_x: tile size x
    :param tile_y: tile size y
    :param overlap: tile overlap
    :return: step count
    """
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


def tiled_scale(
        samples: T,
        upscale_model: typing.Callable[[T], T],
        tile_x: int = 64,
        tile_y: int = 64,
        overlap: int = 8,
        upscale_amount: int = 4,
        out_channels: int = 3,
        pbar: typing.Optional[typing.Callable[[int], None]] | None = None
) -> T:
    """
    Tiled upscale with a model which produces scaled tiles.

    :param samples: input images (b, c, h, w) 0.0 - 1.0
    :param upscale_model: model which takes a sample input in the
        format (b, c, h, w) 0.0 - 1.0 and returns it upscaled in the same format.
    :param tile_x: tile width
    :param tile_y: tile height
    :param overlap: overlap amount
    :param upscale_amount: upscale factor of the model
    :param out_channels: output color channels
    :param pbar: optional progress bar callback
    :return: upscaled images in the form (b, c, h, w) 0.0 - 1.0
    """
    is_torch = isinstance(samples, torch.Tensor)

    if is_torch:
        zeros_like = torch.zeros_like
        ones_like = torch.ones_like
        empty = torch.empty
        zeros = torch.zeros
        round_func = lambda x: torch.round(torch.tensor(x)).item()  # to match round() behavior
        to_cpu = lambda x: x.cpu()
    else:
        zeros_like = numpy.zeros_like
        ones_like = numpy.ones_like
        empty = numpy.empty
        zeros = numpy.zeros
        round_func = round
        to_cpu = lambda x: x

    # Pre-calculate the scaled dimensions
    scaled_height = round_func(samples.shape[2] * upscale_amount)
    scaled_width = round_func(samples.shape[3] * upscale_amount)

    # Initialize the output tensor
    output = empty((samples.shape[0], out_channels, scaled_height, scaled_width))

    tile_blending = overlap > 0

    # Pre-calculate the feathering factor
    if tile_blending:
        feather = round_func(overlap * upscale_amount)
        feather_factor = 1.0 / feather

    # Iterate over each sample
    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        out = zeros((s.shape[0], out_channels, scaled_height, scaled_width))
        out_div = zeros_like(out)

        # Iterate over the tiles
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y:y + tile_y, x:x + tile_x]
                ps = to_cpu(upscale_model(s_in))

                if tile_blending:
                    mask = ones_like(ps)

                    # Apply feathering to the mask
                    for t in range(feather):
                        mask[:, :, t:1 + t, :] *= (feather_factor * (t + 1))
                        mask[:, :, -1 - t: -t, :] *= (feather_factor * (t + 1))
                        mask[:, :, :, t:1 + t] *= (feather_factor * (t + 1))
                        mask[:, :, :, -1 - t: -t] *= (feather_factor * (t + 1))

                # Calculate the indices for the output tensor
                y_start = round_func(y * upscale_amount)
                y_end = y_start + round_func(tile_y * upscale_amount)
                x_start = round_func(x * upscale_amount)
                x_end = x_start + round_func(tile_x * upscale_amount)

                # Update the output tensor
                if tile_blending:
                    out[:, :, y_start:y_end, x_start:x_end] += ps * mask
                    out_div[:, :, y_start:y_end, x_start:x_end] += mask
                else:
                    out[:, :, y_start:y_end, x_start:x_end] += ps

                # Update progress bar if provided
                if pbar is not None:
                    pbar(1)

        # Divide the accumulated output by the mask to get the final result
        if tile_blending:
            output[b:b + 1] = out / out_div
        else:
            output[b:b + 1] = out

    return output


__all__ = _types.module_all()
