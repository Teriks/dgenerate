import typing

import dgenerate.batchprocess
import dgenerate.subcommands.subcommand as _subcommand


class ImageProcessSubCommand(_subcommand.SubCommand):
    NAMES = ['image-process']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._directive = dgenerate.batchprocess.DirectiveLoader().load('image_process',
                                                                        render_loop=None,
                                                                        batch_processor=None,
                                                                        injected_plugin_modules=None,
                                                                        allow_exit=True)

    def __call__(self, argv: typing.List[str]) -> int:
        try:
            self._directive(argv)
        except SystemExit as e:
            return e.code
        return 0
