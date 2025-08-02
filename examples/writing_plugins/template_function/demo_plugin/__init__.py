import dgenerate.batchprocess.configrunnerplugin as _configrunnerplugin


class MyTemplateFunction(_configrunnerplugin.ConfigRunnerPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # you could potentially register multiple template functions if you wanted
        self.register_template_function('my_template_function', self.my_template_function)

    def my_template_function(self, input_string):
        """
        This documentation string can be displayed with:

        dgenerate --functions-help my_template_function

        To list all template functions:

        dgenerate --functions-help
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

        # you can raise custom argument errors with self.argument_error

        # raise self.argument_error('My argument error message')

        # this template function simply changes an input string to uppercase
        return input_string.upper()
