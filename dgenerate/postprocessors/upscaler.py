import PIL.Image

import dgenerate.chainner
import dgenerate.postprocessors.postprocessor as _postprocessor


class Upscaler(_postprocessor.ImagePostprocessor):
    """
    Implements tiled upscaling with chaiNNer compatible upscaler models.

    The "model" argument should be a path to a chaiNNer compatible upscaler model,
    such as a model downloaded from https://openmodeldb.info/.  You must download the
    model yourself, it must be a file on disk.

    The "tile" argument indicates the tile size for tiled upscaling, and defaults to 512.

    The "overlap" argument indicates the overlap amount of each tile in pixels, and defaults to 32.
    """

    NAMES = ['upscaler']

    def __init__(self, model, tile=512, overlap=32, **kwargs):
        super().__init__(**kwargs)

        self._model = dgenerate.chainner.load_model(model)
        self._tile = self.get_int_arg('tile', tile)
        self._overlap = self.get_int_arg('overlap', overlap)

    def process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return dgenerate.chainner.upscale(self._model, image, self._tile, self._overlap, self.device)

    def __str__(self):
        return f'{self.__class__.__name__}(function="{self.called_by_name}")'
