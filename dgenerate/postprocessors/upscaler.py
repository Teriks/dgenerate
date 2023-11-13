import PIL.Image

import dgenerate.mediainput
import dgenerate.postprocessors.postprocessor as _postprocessor
from dgenerate.extras import chainner


class Upscaler(_postprocessor.ImagePostprocessor):
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

    The "tile" argument can be used to specify the tile size for tiled upscaling, and defaults to 512.

    The "overlap" argument can be used to specify the overlap amount of each tile in pixels, and defaults to 32.

    Example: "upscaler;model=my-model.pth;tile=256;overlap=16"
    """

    NAMES = ['upscaler']

    def __init__(self, model, tile=512, overlap=32, **kwargs):
        super().__init__(**kwargs)

        try:
            if model.startswith('http') or model.startswith('https'):
                self._model = chainner.load_model(
                    dgenerate.mediainput.create_web_cache_file(model, mimetype_is_supported=None)[1])
            else:
                self._model = chainner.load_model(model)
        except chainner.UnsupportedModelError:
            self.argument_error('Unsupported model file format.')

        self._tile = self.get_int_arg('tile', tile)
        self._overlap = self.get_int_arg('overlap', overlap)

    def impl_process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return chainner.upscale(self._model, image, self._tile, self._overlap, self.device)

