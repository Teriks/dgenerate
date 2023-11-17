import typing

import dgenerate.batchprocess.batchprocessplugin as _batchprocessplugin


class MyDirective(_batchprocessplugin.BatchProcessPlugin):
    NAMES = ['my_directive']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _directive(self, args):
        # Access to the render loop object containing information about
        # previous invocations of dgenerate, this will always be assigned
        # even if an invocation of dgenerate in the configuration has not
        # happened yet.
        print(self.render_loop)

        # access to the BatchProcessor object running the config, you could add
        # template variables / functions etc if desired. Or preform templating
        # operations on strings / files, and many other things.
        print(self.batch_processor)

        # print the arguments to the directive, which are parsed using shlex
        # and are similar to arguments from sys.argv, you could even use
        # argparse on them if you wanted, you are basically implementing
        # a shell command here
        print(args)

    def directive_lookup(self, name) -> typing.Optional[typing.Callable[[typing.List[str]], None]]:
        if name == 'my_directive':
            return lambda args: self._directive(args)
        return None


class MyMultiDirective(_batchprocessplugin.BatchProcessPlugin):
    NAMES = ['my_multi_directive']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _my_directive_1(self, args):
        print('my_directive_1:', args)

    def _my_directive_2(self, args):
        print('my_directive_2:', args)

    def directive_lookup(self, name) -> typing.Optional[typing.Callable[[typing.List[str]], None]]:
        # A single plugin class can handle
        # multiple directive names in this manner
        if name == 'my_directive_1':
            return lambda args: self._my_directive_1(args)
        if name == 'my_directive_2':
            return lambda args: self._my_directive_2(args)
        return None
