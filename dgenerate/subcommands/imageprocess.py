import dgenerate.batchprocess
import dgenerate.subcommands.subcommand as _subcommand
import dgenerate.subcommands.exceptions as _exceptions

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
            dgenerate.batchprocess.ImageProcessDirective(
                loaded_by_name='image_process',
                message_header='image-process:',
                help_name='image-process',
                help_desc='This sub-command allows you to use dgenerate image processors directly on files of your choosing.',
                render_loop=None,
                batch_processor=None,
                plugin_module_paths=None,
                allow_exit=True)

    def __call__(self) -> int:
        try:
            self._directive.image_process(self.args)
        except dgenerate.batchprocess.ConfigRunnerPluginArgumentError as e:
            raise _exceptions.SubCommandArgumentError(e)
        except SystemExit as e:
            return e.code
        return 0
