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
import cv2

import dgenerate.types as _types

T = typing.TypeVar('T', numpy.ndarray, torch.Tensor)


def _bounds_check(width: int, height: int, tile_x: int, tile_y: int, overlap: int):
    # the tile can be larger than the image

    if overlap < 0:
        raise ValueError('overlap must be greater than or equal to 0.')

    if tile_x < 2:
        raise ValueError('tile_x may not be less than 2.')

    if tile_y < 2:
        raise ValueError('tile_y may not be less than 2.')

    if tile_x < overlap:
        raise ValueError('tile_x may not be less than overlap.')

    if tile_y < overlap:
        raise ValueError('tile_y may not be less than overlap.')

    if tile_x % 2 != 0:
        raise ValueError('tile_x must be divisible by 2.')

    if tile_y % 2 != 0:
        raise ValueError('tile_y must be divisible by 2.')


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

    _bounds_check(width, height, tile_x, tile_y, overlap)

    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


def _resize_tensor(tensor: T, target_height: int, target_width: int, scale_factor: float) -> T:
    """
    Resize an image tensor using the most appropriate algorithm based on the scale factor.

    Utilizing CPU based OpenCV resizing in all cases.
    
    :param tensor: Input tensor in (b, c, h, w) format
    :param target_height: Target height
    :param target_width: Target width
    :param scale_factor: Scale factor to help choose algorithm
    :return: Resized tensor in the same format
    """
    
    # Choose the appropriate algorithm based on scale factor
    if scale_factor >= 4.0:
        # Lanczos for large upscaling
        interpolation = cv2.INTER_LANCZOS4
    elif scale_factor >= 1.5:
        # Bicubic for moderate upscaling
        interpolation = cv2.INTER_CUBIC
    elif scale_factor >= 0.5:
        # Bilinear for small upscaling or moderate downscaling
        interpolation = cv2.INTER_LINEAR
    else:
        # Nearest neighbor for significant downscaling
        interpolation = cv2.INTER_NEAREST
    
    is_torch = isinstance(tensor, torch.Tensor)
    
    if is_torch:
        # Save original device and dtype
        original_device = tensor.device
        original_dtype = tensor.dtype
        
        # Convert to numpy for cv2 processing
        # Move to CPU if on different device
        cpu_tensor = tensor.cpu()
        np_tensor = cpu_tensor.numpy()
    else:
        # Already numpy
        np_tensor = tensor
    
    # Create output array
    resized = numpy.empty(
        (np_tensor.shape[0], np_tensor.shape[1], target_height, target_width),
         dtype=np_tensor.dtype
    )
    
    # Process each batch
    for b in range(np_tensor.shape[0]):
        # Get the current batch
        img = np_tensor[b]
        
        # Determine if we're dealing with a multi-channel or single-channel image
        n_channels = img.shape[0]
        
        # Check if normalization needed
        needs_normalization = img.min() >= 0 and img.max() <= 1.0
        
        if n_channels > 1:
            # Multi-channel image (like RGB)
            # Transpose from (C, H, W) to (H, W, C) for cv2
            img = numpy.transpose(img, (1, 2, 0))
            
            if needs_normalization:
                # Scale to 0-255 for OpenCV
                img = (img * 255.0).astype(numpy.uint8)
            
            # Resize the entire multi-channel image at once
            resized_img = cv2.resize(img, (target_width, target_height), 
                                   interpolation=interpolation)
            
            if needs_normalization:
                # Convert back to original range
                resized_img = resized_img.astype(numpy.float32) / 255.0
            
            # Transpose back from (H, W, C) to (C, H, W)
            resized_img = numpy.transpose(resized_img, (2, 0, 1))
            
            # Store the result
            resized[b] = resized_img
            
        else:
            # Single channel image
            img = img[0]  # Take the single channel
            
            if needs_normalization:
                # Scale to 0-255 for OpenCV
                img = (img * 255.0).astype(numpy.uint8)
            
            # Resize the image with cv2
            resized_img = cv2.resize(
                img,
                (target_width, target_height),
                interpolation=interpolation
            )
            
            if needs_normalization:
                # Convert back to original range
                resized_img = resized_img.astype(numpy.float32) / 255.0
            
            # Store the result
            resized[b, 0] = resized_img
    
    # Convert back to torch if original was torch
    if is_torch:
        resized_torch = torch.from_numpy(resized)
        # Convert to original dtype and move to original device
        resized_torch = resized_torch.to(dtype=original_dtype, device=original_device)
        return resized_torch
    else:
        return resized


@torch.inference_mode()
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
    :param upscale_amount: upscale factor of the model, if this does not match
        the scale of the model architecture, automatic resizing of the model output
        will occur to match the user specified scale.
    :param out_channels: output color channels
    :param pbar: optional progress bar callback
    :return: upscaled images in the form (b, c, h, w) 0.0 - 1.0
    """

    _bounds_check(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)

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
        
    # Variables to store model's actual upscale factor and whether resizing is needed
    model_upscale_factor = None
    need_resize = False
    
    # Iterate over each sample
    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        out = zeros((s.shape[0], out_channels, scaled_height, scaled_width))

        if tile_blending:
            out_div = zeros_like(out)

        # Iterate over the tiles
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y:y + tile_y, x:x + tile_x]
                ps = to_cpu(upscale_model(s_in))
                
                # Detect model's actual upscale factor (only once)
                if model_upscale_factor is None:
                    # Calculate the actual upscale factor using height
                    model_upscale_factor = ps.shape[2] / s_in.shape[2]
                    need_resize = abs(model_upscale_factor - upscale_amount) > 1e-5
                
                # Resize if needed
                if need_resize:
                    out_h = round_func(s_in.shape[2] * upscale_amount)
                    out_w = round_func(s_in.shape[3] * upscale_amount)
                    resize_factor = upscale_amount / model_upscale_factor
                    ps = _resize_tensor(ps, out_h, out_w, resize_factor)

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
                y_end = y_start + ps.shape[2]  # Use actual size of ps after potential resize
                x_start = round_func(x * upscale_amount)
                x_end = x_start + ps.shape[3]  # Use actual size of ps after potential resize

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
