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
import os

import PIL.Image
import spandrel
import torch
import torchvision
import tqdm.auto

import dgenerate.exceptions as _d_exceptions
import dgenerate.hfhub as _hfhub
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.imageprocessors.upscale_tiler as _upscale_tiler
import dgenerate.messages as _messages
import dgenerate.types as _types
import dgenerate.webcache as _webcache


class UpscalerProcessor(_imageprocessor.ImageProcessor):
    """
    Implements tiled upscaling with chaiNNer compatible upscaler models.

    The "model" argument should be a path to a chaiNNer compatible upscaler model on disk,
    such as a model downloaded from https://openmodeldb.info/, or an HTTP/HTTPS URL
    that points to a raw model file. This may also be a Hugging Face blob link.

    For example: "upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"

    Downloaded models are cached in the dgenerate web cache on disk until the
    cache expiry time for the file is met.

    The "tile" argument can be used to specify the tile size for tiled upscaling, it must be divisible by 2, and
    defaults to 512. Specifying 'auto' indicates that this value should be calculated based off available GPU
    memory if applicable. Specifying 0 disables tiling entirely.

    The "overlap" argument can be used to specify the overlap amount of each tile in pixels, it must be
    greater than or equal to 0, and defaults to 32.

    The "scale" argument can be used to specify the output scale of the image regardless of the models
    scale, this equates to an image resize on each tile output of the model as necessary, with auto selected
    resizing algorithm for the best quality. This is effectively equivalent to basic image resizing of
    the upscaled output post upscale, just with somewhat reduced memory overhead as it occurs during tiling.
    When this argument is not specified, the scale of the model architecture is used and no resizing occurs.
    This argument must be greater than or equal to 1.

    The "force-tiling" argument can be used to force external image tiling for upscaler model architectures which
    discourage the use of external tiling (SCUNEt and MixDehazeNet currently), this may mean that the model needs
    information about the whole image to achieve a good result. External tiling breaks up the image into tiles
    before feeding it to the model and reassembles the images output by the model, this is not the default behavior
    when a model specifies that tiling is discouraged, tiling is only on by default for models where external tiling is
    fully supported. Only use this if you run into memory issues with models that discourage external tiling, in the
    case that the model discourages its use, using it may result in substandard image output.

    The "dtype" argument can be used to specify the datatype to use to for the model in memory, it can be
    either "float32" or "float16". Using "float16" will result in a smaller memory footprint if supported.

    The "pre-resize" argument is a boolean value determining if the processing should take place before or
    after the image is resized by dgenerate.

    Example: "upscaler;model=my-model.pth;tile=256;overlap=16"
    """

    NAMES = ['upscaler']

    OPTION_ARGS = {
        'dtype': ['float32', 'float16']
    }

    FILE_ARGS = {
        'model': {'mode': 'in', 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]},
    }

    def __init__(self,
                 model: str,
                 tile: int | str = 512,
                 overlap: int = 32,
                 scale: int | None = None,
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
        :param scale: the scale factor to use for the upscaler, it must be greater than 1, and defaults to None (meaning use the model's scale).
        :param dtype: the datatype to use to for the model in memory, it may be either "float32" or "float16".
            using float16 will result in a smaller memory footprint if supported.
        :param force_tiling: Force external image tiling for model architectures that discourage it.
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """

        super().__init__(**kwargs)

        self._model_path = model

        dtype = dtype.lower()
        if dtype not in {'float32', 'float16'}:
            raise self.argument_error('Argument "dtype" must be either float32 or float16.')

        self._dtype = torch.float32 if dtype == 'float32' else torch.float16

        self._model = self._load_upscaler_model(model, self._dtype)

        if isinstance(tile, str):
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

        if scale is not None and scale < 1:
            raise self.argument_error('Argument "scale" must be greater than or equal to 1.')

        self._tile = tile
        self._overlap = overlap
        self._pre_resize = pre_resize
        self._force_tiling = force_tiling
        self._scale = scale

        self.register_module(self._model)

    def _load_upscaler_model(self, model_path, dtype) -> spandrel.ImageModelDescriptor:
        """
        Load an upscaler model from a file path or URL.

        :param model_path: path
        :return: model
        """
        og_path = model_path

        if _webcache.is_downloadable_url(model_path):
            # Any mimetype
            try:
                model_path = _hfhub.webcache_or_hf_blob_download(
                    model_path, local_files_only=self.local_files_only
                )
            except Exception as e:
                raise self.argument_error(
                    f'Could not download argument "model": "{model_path}", error: {e}') from e

        # use the model file size as a heuristic
        self.set_size_estimate(os.path.getsize(model_path))

        def load_method():
            try:
                upscaler_model = spandrel.ModelLoader().load_from_file(model_path).eval()
                upscaler_model.to(dtype)
            except spandrel.UnsupportedDtypeError as e:
                raise self.argument_error(
                    f'Argument "dtype" value "{dtype}" not supported for '
                    f'upscaler model architecture "{_types.fullname(self._model.architecture)}" '
                    f'used with model file: "{og_path}"') from e
            except (ValueError,
                    RuntimeError,
                    TypeError,
                    AttributeError) as e:
                raise self.argument_error(
                    f'Unsupported model file format: {e}') from e

            if not isinstance(upscaler_model, spandrel.ImageModelDescriptor):
                raise self.argument_error(
                    f'Unsupported model file format: Upscale model must be a single-image model.')
            return upscaler_model

        # make sure dtype is part of the cache tag
        model = self.load_object_cached(
            tag=model_path + str(dtype).split('.')[1],
            estimated_size=self.size_estimate,
            method=load_method
        )

        _messages.debug_log(
            f'{_types.fullname(UpscalerProcessor._load_upscaler_model)}("{model_path}") -> {model.__class__.__name__}')

        return model

    def _auto_tile_size(self, img: PIL.Image.Image) -> int:
        if self.modules_device.type == 'cpu':
            # default
            return 512

        h, w, c = (img.height, img.width, 3)

        dtype_size = 4 if self._dtype is torch.float32 else 2

        model_size = sum(
            p.numel() * dtype_size for p in self._model.model.parameters()
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
        in_img = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(self.modules_device, dtype=self._dtype)

        if self._model.input_channels == 4:
            # Fill 4th channel with 1.0s, this is definitely incorrect.
            # Not all models expect this to be an alpha channel
            _messages.warning(
                f'Appending 1.0 (opaque) alpha channel RGB -> RGBA for model '
                f'architecture type "{_types.fullname(self._model.architecture)}" used by model "{self._model_path}" '
                'which requires 4 input channels. This is not guaranteed to be correct input data for '
                'this model architecture! If you know how this model is supposed to work, please submit an issue.'
            )

            in_img = torch.cat(
                (in_img, torch.ones(
                    1, 1, in_img.shape[2], in_img.shape[3]).to(self.modules_device)),
                dim=1)

        elif self._model.input_channels == 2:

            raise self.argument_error(
                f'Specified model "{self._model_path}" using architecture type '
                f'"{_types.fullname(self._model.architecture)}" requires a 2 channel image '
                '(non RGB or RGBA input required), conversion to this format internally is not '
                'supported currently for any model. If you know how this model is supposed to '
                'work, please submit an issue.')

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
            return torchvision.transforms.ToPILImage()(self._model(in_img).squeeze(0))

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
                    self._model,
                    tile_x=tile,
                    tile_y=tile,
                    overlap=self._overlap,
                    upscale_amount=_types.default(self._scale, self._model.scale),
                    pbar=pbar.update)

                oom = False
            except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
                _d_exceptions.raise_if_not_cuda_oom(e)

                pbar.close()
                tile //= 2
                _messages.warning(
                    f'Reducing tile size to {tile} and retrying due to GPU running out of memory.'
                )
                if tile < 128:
                    raise e

        return torchvision.transforms.ToPILImage()(output.squeeze(0))

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize) -> PIL.Image.Image:
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        if not self._pre_resize:
            return self._process(image)
        return image


__all__ = _types.module_all()
