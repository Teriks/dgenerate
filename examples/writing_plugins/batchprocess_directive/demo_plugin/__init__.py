import typing
import dgenerate.batchprocess.batchprocessordirective as _batchprocessordirective


class MyDirective(_batchprocessordirective.BatchProcessorDirective):
    NAMES = ['my_directive']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, args: typing.List[str]):
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


class MyMultiDirective(_batchprocessordirective.BatchProcessorDirective):
    NAMES = ['my_directive_1', 'my_directive_2']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, args: typing.List[str]):
        # A single implementation class can handle multiple
        # directive names in this manner

        if self.called_by_name == 'my_directive_1':
            print('my_directive_1:', args)
        if self.called_by_name == 'my_directive_2':
            print('my_directive_2:', args)
