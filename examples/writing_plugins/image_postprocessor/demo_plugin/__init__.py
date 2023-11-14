import PIL.Image

import dgenerate.postprocessors


class FooBarPostprocessor(dgenerate.postprocessors.ImagePostprocessor):
    """My --postprocessor-help documentation, arguments are described automatically"""

    # This static property defines what names this module can be invoked by
    NAMES = ['foobar']

    # All argument names will have _ replaced with - on the command line.
    # Argument signature correctness (missing arguments, unknown arguments) etc.
    # is verified by dgenerate, you are responsible for parsing and making sure your
    # arguments are in the right format IE, int, float, bool, etc. they are actually
    # passed into "__init__" as a string
    def __init__(self,
                 my_argument,
                 my_argument_2=False,
                 my_argument_3=1.0,
                 **kwargs):
        super().__init__(**kwargs)

        # These helper functions can help you validate arguments of int, float, and bool types
        # they will produce a "dgenerate.postprocessors.ImagePostprocessorArgumentError"
        # if the string is not in the correct format for the requested type.
        # You can throw that exception yourself or call "self.argument_error('my message')" anywhere
        # in "__init__" if you are doing custom argument validation, it will be handle
        # correctly and printed for the user on the command line

        self._my_argument = self.get_int_arg('my_argument', my_argument)
        self._my_argument_2 = self.get_bool_arg('my_argument_2', my_argument_2)
        self._my_argument_3 = self.get_float_arg('my_argument_3', my_argument_3)

    def impl_process(self, image: PIL.Image.Image):
        # This is where you will preform the postprocess on a generated image

        # If you do modify the image, it is acceptable to modify it in place, or
        # to return a copy of it

        # Make a white square in the upper left corner
        for x in range(0, 100):
            for y in range(0, 100):
                image.putpixel((x, y), (255, 255, 255))

        print("FOOBAR:", self._my_argument, self._my_argument_2, self._my_argument_3)
        return image


# Below is an example of a postprocessor module that can be invoked
# by multiple names and handle multiple postprocessing tasks within
# the same class

class BooFabPostprocessor(dgenerate.postprocessors.ImagePostprocessor):
    # This static property defines what names this module can be invoked by
    NAMES = ['boo', 'fab']

    # All argument names will have _ replaced with - on the command line
    # This static property can be used to define the argument signature
    # for each of the invokable names above
    ARGS = {
        # required arguments are specified as a string,
        # arguments with default values are specified as a tuple
        'boo': ['my_argument', ('my_argument_2', False), ('my_argument_3', 1.0)],
        'fab': ['my_argument', ('my_argument_2', 2.0), ('my_argument_3', False)]
    }

    # Defining the static method "help" allows you to provide a help string
    # for each invokable name, for --postprocessor-help "boo" and --postprocessor-help "foo"
    # the name this option is used with is passed to "called_by_name"
    @staticmethod
    def help(called_by_name):
        if called_by_name == 'boo':
            return 'My --postprocessor-help "boo" documentation'
        if called_by_name == 'fab':
            return 'My --postprocessor-help "fab" documentation'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # "self.called_by_name" contains the name string
        # that this module is being invoked by, we can do something
        # different in the module depending on what name is used
        # to invoke it

        if self.called_by_name == 'boo':
            self._my_argument = self.get_int_arg('my_argument', kwargs['my_argument'])
            self._my_argument_2 = self.get_bool_arg('my_argument_2', kwargs)
            self._my_argument_3 = self.get_float_arg('my_argument_3', kwargs)
        if self.called_by_name == 'fab':
            self._my_argument = self.get_int_arg('my_argument', kwargs['my_argument'])
            self._my_argument_2 = self.get_float_arg('my_argument_2', kwargs)
            self._my_argument_3 = self.get_bool_arg('my_argument_3', kwargs)

    def impl_process(self, image: PIL.Image.Image):
        print(f'{self.called_by_name}:', self._my_argument, self._my_argument_2, self._my_argument_3)
        return image
