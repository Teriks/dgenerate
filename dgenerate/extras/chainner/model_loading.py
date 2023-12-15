import dgenerate.messages as _messages
import dgenerate.mediainput as _mediainput
import dgenerate.types as _types
import spandrel


class UnsupportedModelError(Exception):
    """chaiNNer model is not of a supported type."""
    pass


def load_upscaler_model(model_path) -> spandrel.ImageModelDescriptor:
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
        raise UnsupportedModelError(e)

    if not isinstance(model, spandrel.ImageModelDescriptor):
        raise UnsupportedModelError("Upscale model must be a single-image model.")

    _messages.debug_log(
        f'{_types.fullname(load_upscaler_model)}("{model_path}") -> {model.__class__.__name__}')

    return model
