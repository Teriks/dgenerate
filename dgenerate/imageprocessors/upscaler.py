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
from tqdm.auto import tqdm
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


@torch.inference_mode()
def _tiled_scale(samples: torch.Tensor, upscale_model: spandrel.ImageModelDescriptor, tile_x=64, tile_y=64, overlap=8,
                 upscale_amount=4,
                 out_channels=3, pbar=None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount),
                          round(samples.shape[3] * upscale_amount)), device="cpu")
    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros(
            (s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)),
            device="cpu")
        out_div = torch.zeros(
            (s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)),
            device="cpu")
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y:y + tile_y, x:x + tile_x]

                ps = upscale_model(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    mask[:, :, t:1 + t, :] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, mask.shape[2] - 1 - t: mask.shape[2] - t, :] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, :, t:1 + t] *= ((1.0 / feather) * (t + 1))
                    mask[:, :, :, mask.shape[3] - 1 - t: mask.shape[3] - t] *= ((1.0 / feather) * (t + 1))
                out[:, :, round(y * upscale_amount):round((y + tile_y) * upscale_amount),
                round(x * upscale_amount):round((x + tile_x) * upscale_amount)] += ps * mask
                out_div[:, :, round(y * upscale_amount):round((y + tile_y) * upscale_amount),
                round(x * upscale_amount):round((x + tile_x) * upscale_amount)] += mask
                if pbar is not None:
                    pbar.update(1)
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

    The "tile" argument can be used to specify the tile size for tiled upscaling, it must be divisible by 2, and defaults to 512.

    The "overlap" argument can be used to specify the overlap amount of each tile in pixels, it must be
    greater than or equal to 0, and defaults to 32.

    The "force-tiling" argument can be used to force external image tiling for upscaler model architectures which
    discourage the use of external tiling (SCUNEt and MixDehazeNet currently), this may mean that the model needs
    information about the whole image to achieve a good result. External tiling breaks up the image into tiles
    before feeding it to the model and reassembles the images output by the model, this is not the default behavior
    when a model specifies that tiling is discouraged, tiling is only on by default for models where external tiling is
    fully supported. Only use this if you run into memory issues with models that discourage external tiling, in the
    case that the model discourages its use, using it may result in substandard image output.

    The "pre-resize" argument is a boolean value determining if the processing should take place before or
    after the image is resized by dgenerate.

    Example: "upscaler;model=my-model.pth;tile=256;overlap=16"
    """

    NAMES = ['upscaler']

    def __init__(self,
                 model: str,
                 tile: int = 512,
                 overlap: int = 32,
                 force_tiling: bool = False,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param model: chaiNNer compatible upscaler model on disk, or at a URL
        :param tile: specifies the tile size for tiled upscaling, it must be divisible by 2, and defaults to 512.
        :param overlap: the overlap amount of each tile in pixels, it must be greater than or equal to 0, and defaults to 32.
        :param force_tiling: Force external image tiling for model architectures that discourage it.
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """

        super().__init__(**kwargs)

        try:
            self._model = _load_upscaler_model(model)
        except _UnsupportedModelError as e:
            raise self.argument_error(f'Unsupported model file format: {e}')

        if tile < 2:
            raise self.argument_error('Argument "tile" must be greater than 2.')

        if tile % 2 != 0:
            raise self.argument_error('Argument "tile" must be divisible by 2.')

        if overlap < 0:
            raise self.argument_error('Argument "overlap" must be greater than or equal to 0.')

        self._tile = tile
        self._overlap = overlap
        self._pre_resize = pre_resize
        self._force_tiling = force_tiling

        self.register_module(self._model)

    def _process(self, image):
        image = torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0)[None,]

        in_img = image.movedim(-1, -3)

        if self._model.input_channels == 1:
            # noinspection PyTypeChecker
            # operator overloading, a tensor full of bool is returned
            if torch.all(in_img[:, 0, :, :] == in_img[:, 1, :, :]) \
                    and torch.all(in_img[:, 0, :, :] == in_img[:, 2, :, :]):
                # already grayscale, only keep the red value
                in_img = in_img[:, :1, :, :]
            else:
                # make it grayscale [1,1,W,H]
                in_img = 0.299 * in_img[:, 0:1, :, :] + 0.587 * in_img[:, 1:2, :, :] + 0.114 * in_img[:, 2:3, :, :]

        in_img = in_img.to(self.modules_device)

        if self._model.tiling == spandrel.ModelTiling.INTERNAL or \
                (self._model.tiling == spandrel.ModelTiling.DISCOURAGED and not self._force_tiling):
            # do not externally tile, there is either internal tiling, or the model
            # discourages doing so and the user has not forced it to occur
            return _model_output_to_pil(self._model(in_img))

        oom = True

        tile = self._tile

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
