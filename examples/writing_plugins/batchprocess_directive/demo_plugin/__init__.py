import dgenerate.batchprocess.configrunnerplugin as _configrunnerplugin


class MyDirective(_configrunnerplugin.ConfigRunnerPlugin):
    NAMES = ['my_directive']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.register_directive('my_directive', lambda args: self.directive(args))

    def directive(self, args):
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


class MyMultiDirective(_configrunnerplugin.ConfigRunnerPlugin):
    NAMES = ['my_multi_directive']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.register_directive('my_directive_1', lambda args: self.my_directive_1(args))
        self.register_directive('my_directive_2', lambda args: self.my_directive_2(args))

    def my_directive_1(self, args):
        print('my_directive_1:', args)

    def my_directive_2(self, args):
        print('my_directive_2:', args)
