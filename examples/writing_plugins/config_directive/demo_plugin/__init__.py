import collections.abc

import dgenerate.batchprocess.configrunnerplugin as _configrunnerplugin


class MyDirective(_configrunnerplugin.ConfigRunnerPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.register_directive('my_directive', self.directive)

    def directive(self, args: collections.abc.Sequence[str]):
        """
        This documentation string can be displayed with:

        dgenerate --directives-help my_directive

        or

        dgenerate --directives-help \\my_directive


        To list all directives:

        dgenerate --directives-help
        """

        # Access to the render loop object containing information about
        # previous invocations of dgenerate, this will always be assigned
        # even if an invocation of dgenerate in the configuration has not
        # happened yet.
        print(self.render_loop)

        # access to the ConfigRunner object running the config, you could add
        # template variables / functions etc if desired. Or perform templating
        # operations on strings / files, and many other things.
        print(self.config_runner)

        # print the arguments to the directive, which are parsed using shlex
        # and are similar to arguments from sys.argv, you could even use
        # argparse on them if you wanted, you are basically implementing
        # a shell command here
        print(args)

        # you can raise custom argument errors with self.argument_error

        # raise self.argument_error('My argument error message')

        return 0

        # Return code 0 indicates that the directive executed successfully
        # anything other than 0 halts execution with a message reporting
        # the return code, any error logging should be handled by you

        # Unhandled unknown non SystemExit exceptions will also halt execution and be
        # rethrown as a BatchProcessError from the config runner without a stack trace
        # unless -v/--verbose is enable. The exception message will be printed
        # to the console alone, unless -v/--verbose is specified
        # (injected into the config from the command line).


class MyMultiDirective(_configrunnerplugin.ConfigRunnerPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.register_directive('my_directive_1', self.my_directive_1)
        self.register_directive('my_directive_2', self.my_directive_2)

    def my_directive_1(self, args: collections.abc.Sequence[str]) -> int:
        print('my_directive_1:', args)
        return 0

    def my_directive_2(self, args: collections.abc.Sequence[str]):
        print('my_directive_2:', args)
        return 0
