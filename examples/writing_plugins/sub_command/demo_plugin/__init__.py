import dgenerate.subcommands.subcommand as _subcommand


class MySubCommand(_subcommand.SubCommand):
    """
    Demo sub command extensibility
    """

    NAMES = ['my-subcommand']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self) -> int:
        # print the command line arguments passed to the sub command
        print(self.args)

        # empty because we did not use the --plugin-modules argument
        print(self.plugin_module_paths)

        # return a return code
        return 0
