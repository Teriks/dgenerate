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
import torch
import tqdm.auto

import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.types as _types
from dgenerate.extras import chainner
import spandrel


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
            self._model = chainner.load_upscaler_model(model)
        except chainner.UnsupportedModelError as e:
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
                steps = in_img.shape[0] * chainner.get_tiled_scale_steps(
                    in_img.shape[3],
                    in_img.shape[2],
                    tile_x=tile,
                    tile_y=tile,
                    overlap=self._overlap)

                pbar = tqdm.auto.tqdm(total=steps)

                s = chainner.tiled_scale(in_img,
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
