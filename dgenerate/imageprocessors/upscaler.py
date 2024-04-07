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
import typing

import PIL.Image
import numpy
import tqdm.auto

import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.types as _types
import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import math
import PIL.Image
import torch
import spandrel
import spandrel_extra_arches

spandrel.MAIN_REGISTRY.add(*spandrel_extra_arches.EXTRA_REGISTRY)


class _UnsupportedModelError(Exception):
    """chaiNNer model is not of a supported type."""
    pass


def _load_upscaler_model(model_path) -> spandrel.ImageModelDescriptor:
    """
    Load an upscaler model from a file path or URL.

    :param model_path: path
    :return: model
    """
    if _mediainput.is_downloadable_url(model_path):
        # Any mimetype
        _, model_path = _mediainput.create_web_cache_file(
            model_path, mimetype_is_supported=None)

    try:
        model = spandrel.ModelLoader().load_from_file(model_path).eval()
    except ValueError as e:
        raise _UnsupportedModelError(e)

    if not isinstance(model, spandrel.ImageModelDescriptor):
        raise _UnsupportedModelError("Upscale model must be a single-image model.")

    _messages.debug_log(
        f'{_types.fullname(_load_upscaler_model)}("{model_path}") -> {model.__class__.__name__}')

    return model


def _get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


def _tiled_scale(samples: torch.Tensor, upscale_model: spandrel.ImageModelDescriptor, tile_x=64, tile_y=64, overlap=8,
                 upscale_amount=4,
                 out_channels=3, pbar=None):
    # Pre-calculate the scaled dimensions
    scaled_height = round(samples.shape[2] * upscale_amount)
    scaled_width = round(samples.shape[3] * upscale_amount)

    # Initialize the output tensor on CPU
    output = torch.empty((samples.shape[0], out_channels, scaled_height, scaled_width), device="cpu")

    # Pre-calculate the feathering factor
    feather = round(overlap * upscale_amount)
    feather_factor = 1.0 / feather

    # Iterate over each sample
    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros((s.shape[0], out_channels, scaled_height, scaled_width), device="cpu")
        out_div = torch.zeros_like(out)

        # Iterate over the tiles
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y:y + tile_y, x:x + tile_x]
                ps = upscale_model(s_in).cpu()
                mask = torch.ones_like(ps)

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
                out[:, :, y_start:y_end, x_start:x_end] += ps * mask
                out_div[:, :, y_start:y_end, x_start:x_end] += mask

                # Update progress bar if provided
                if pbar is not None:
                    pbar.update(1)

        # Divide the accumulated output by the mask to get the final result
        output[b:b + 1] = out / out_div

    return output


def _model_output_to_pil(tensor):
    tensor = torch.clamp(tensor.movedim(-3, -1), min=0, max=1.0)
    return PIL.Image.fromarray(numpy.clip(255. * tensor[0].cpu().numpy(), 0, 255).astype(numpy.uint8))


class UpscalerProcessor(_imageprocessor.ImageProcessor):
    """
    Implements tiled upscaling with chaiNNer compatible upscaler models.

    The "model" argument should be a path to a chaiNNer compatible upscaler model on disk,
    such as a model downloaded from https://openmodeldb.info/, or an HTTP/HTTPS URL
    that points to a raw model file.

    For example: "upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"

    Downloaded models are only cached during the life of the process and will always be downloaded even
    when --offline-mode is specified, dgenerate processes that are alive simultaneously will share the
    cached model file and will not need to re-download it.

    Fresh executions of dgenerate will always download the model, it is recommended that you download your upscaler
    models to some place on disk and provide a full path to them, and to use URLs only for examples and testing.

    The "tile" argument can be used to specify the tile size for tiled upscaling, it must be divisible by 2, and
    defaults to 512. Specifying 'auto' indicates that this value should be calculated based off available GPU
    memory if applicable. Specifying 0 disables tiling entirely.

    The "overlap" argument can be used to specify the overlap amount of each tile in pixels, it must be
    greater than or equal to 0, and defaults to 32.

    The "force-tiling" argument can be used to force external image tiling for upscaler model architectures which
    discourage the use of external tiling (SCUNEt and MixDehazeNet currently), this may mean that the model needs
    information about the whole image to achieve a good result. External tiling breaks up the image into tiles
    before feeding it to the model and reassembles the images output by the model, this is not the default behavior
    when a model specifies that tiling is discouraged, tiling is only on by default for models where external tiling is
    fully supported. Only use this if you run into memory issues with models that discourage external tiling, in the
    case that the model discourages its use, using it may result in substandard image output.

    The "dtype" argument can be used to specify the datatype to use to for the model in memory, it can be
    either "float32" or "float16". using float16 will result in a smaller memory footprint if supported.

    The "pre-resize" argument is a boolean value determining if the processing should take place before or
    after the image is resized by dgenerate.

    Example: "upscaler;model=my-model.pth;tile=256;overlap=16"
    """

    NAMES = ['upscaler']

    def __init__(self,
                 model: str,
                 tile: typing.Union[int, str] = 512,
                 overlap: int = 32,
                 force_tiling: bool = False,
                 dtype: str = 'float32',
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param model: chaiNNer compatible upscaler model on disk, or at a URL
        :param tile: specifies the tile size for tiled upscaling, it must be divisible by 2, and defaults to 512.
            Specifying 'auto' indicates that this value should be calculated based off available GPU memory
            if applicable. Specifying 0 disable tiling entirely.
        :param overlap: the overlap amount of each tile in pixels, it must be greater than or equal to 0, and defaults to 32.
        :param dtype: the datatype to use to for the model in memory, it may be either "float32" or "float16".
            using float16 will result in a smaller memory footprint if supported.
        :param force_tiling: Force external image tiling for model architectures that discourage it.
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """

        super().__init__(**kwargs)

        self._model_path = model

        try:
            self._model = _load_upscaler_model(model)
        except _UnsupportedModelError as e:
            raise self.argument_error(f'Unsupported model file format: {e}')

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

        if overlap < 0:
            raise self.argument_error('Argument "overlap" must be greater than or equal to 0.')

        dtype = dtype.lower()
        if dtype not in ['float32', 'float16']:
            raise self.argument_error('Argument "dtype" must be either float32 or float16.')

        self._dtype = torch.float32 if dtype == 'float32' else torch.float16
        self._tile = tile
        self._overlap = overlap
        self._pre_resize = pre_resize
        self._force_tiling = force_tiling

        # hack
        class UpscalerModel:
            def __init__(self, plugin, m, d):
                self._model = m
                self._dtype = d
                self._plugin = plugin

            def to(self, device):
                try:
                    self._model.to(device=device, dtype=self._dtype)
                except spandrel.UnsupportedDtypeError:
                    raise self._plugin.argument_error(
                        f'Argument "dtype" value "{dtype}" not supported for '
                        f'upscaler model architecture "{_types.fullname(self._model.architecture)}" '
                        f'used with model file: "{model}"')

        self.register_module(UpscalerModel(self, self._model, self._dtype))

    def _auto_tile_size(self, img: PIL.Image.Image) -> int:
        if self.modules_device.type == 'cpu':
            # default
            return 512

        h, w, c = (img.height, img.width, 3)

        dtype_size = 4 if self._dtype is torch.float32 else 2

        model_size = sum(
            p.numel() * 4 for p in self._model.model.parameters()
        )

        # 80 percent of total free memory
        budget = int(torch.cuda.mem_get_info(self.modules_device.index)[0] * .8)

        img_bytes = h * w * c * dtype_size
        mem_required_estimation = (model_size / (1024 * 52)) * img_bytes

        tile_pixels = w * h * budget / mem_required_estimation

        # the largest power-of-2 tile_size such that tile_size**2 < tile_pixels
        tile_size = 2 ** (int(tile_pixels ** 0.5).bit_length() - 1)

        return tile_size

    @torch.inference_mode()
    def _process(self, image):
        in_img = (torch.from_numpy(
            numpy.array(image).astype(numpy.float32) / 255.0)[None,]).movedim(-1, -3).to(self.modules_device)

        if self._model.input_channels == 4:
            # Fill 4th channel with 1.0s, this is definitely incorrect.
            # Not all models expect this to be an alpha channel
            _messages.log(
                f'Appending 1.0 (opaque) alpha channel RGB -> RGBA for model '
                f'architecture type "{_types.fullname(self._model.architecture)}" used by model "{self._model_path}" '
                'which requires 4 input channels. This is not guaranteed to be correct input data for '
                'this model architecture! If you know how this model is supposed to work, please submit an issue.',
                level=_messages.WARNING)

            in_img = torch.cat(
                (in_img, torch.ones(
                    1, 1, in_img.shape[2], in_img.shape[3]).to(self.modules_device)),
                dim=1)
        elif self._model.input_channels == 2:
            raise self.argument_error(
                f'Specified model "{self._model_path}" requires a 2 channel image (non RGB or RGBA input required), '
                'conversion to this format internally is not supported currently for any model. '
                'If you know how this model is supposed to work, please submit an issue.')
        elif self._model.input_channels == 1:
            # noinspection PyTypeChecker
            # operator overloading, a tensor full of bool is returned
            if torch.all(in_img[:, 0, :, :] == in_img[:, 1, :, :]) \
                    and torch.all(in_img[:, 0, :, :] == in_img[:, 2, :, :]):
                # already grayscale, only keep the red value
                in_img = in_img[:, :1, :, :]
            else:
                # make it grayscale [1,1,W,H]
                in_img = 0.299 * in_img[:, 0:1, :, :] + 0.587 * in_img[:, 1:2, :, :] + 0.114 * in_img[:, 2:3, :, :]

        if self._tile == 'auto':
            tile = self._auto_tile_size(image)
            _messages.log(f'Automatically calculated optimal tile size: {tile}')
        else:
            tile = self._tile

        if tile == 0 or self._model.tiling == spandrel.ModelTiling.INTERNAL or \
                (self._model.tiling == spandrel.ModelTiling.DISCOURAGED and not self._force_tiling):
            # do not externally tile, there is either internal tiling, or the model
            # discourages doing so and the user has not forced it to occur
            return _model_output_to_pil(self._model(in_img))

        oom = True

        while oom:
            try:
                steps = in_img.shape[0] * _get_tiled_scale_steps(
                    in_img.shape[3],
                    in_img.shape[2],
                    tile_x=tile,
                    tile_y=tile,
                    overlap=self._overlap)

                pbar = tqdm.auto.tqdm(total=steps)

                s = _tiled_scale(in_img,
                                 self._model,
                                 tile_x=tile,
                                 tile_y=tile,
                                 overlap=self._overlap,
                                 upscale_amount=self._model.scale,
                                 pbar=pbar)

                oom = False
            except torch.cuda.OutOfMemoryError as e:
                tile //= 2
                if tile < 128:
                    raise e

        return _model_output_to_pil(s)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize) -> PIL.Image.Image:
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        if not self._pre_resize:
            return self._process(image)
        return image


__all__ = _types.module_all()
