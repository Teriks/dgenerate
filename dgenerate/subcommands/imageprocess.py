import typing

import dgenerate.batchprocess
import dgenerate.subcommands.subcommand as _subcommand


class ImageProcessSubCommand(_subcommand.SubCommand):
    """
    Allows for using the \\image_process config directive from the command line, any therefore
    and image preprocessor implemented by dgenerate or a plugin directly from the command line.

    Examples:

    dgenerate --sub-command image-process "my-photo.jpg" --output my-photo-openpose.jpg --processors openpose

    dgenerate --sub-command image-process "my-photo.jpg" --output my-photo-canny.jpg --processors canny

    See: dgenerate --sub-command image-process --help
    """

    NAMES = ['image-process']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._directive = \
            dgenerate.batchprocess.BatchProcessPluginLoader().load(
                'image_process',
                render_loop=None,
                batch_processor=None,
                plugin_module_paths=None,
                allow_exit=True)

    def __call__(self, argv: typing.List[str]) -> int:
        try:
            self._directive.directive_lookup('image_process')(argv)
        except SystemExit as e:
            return e.code
        return 0
