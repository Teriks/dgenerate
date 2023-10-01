import typing

import PIL.Image

import dgenerate.preprocessors


class FooBarPreprocessor(dgenerate.preprocessors.ImagePreprocessor):
    """My --image-preprocessor-help documentation, arguments are described automatically"""

    NAMES = ['foobar']

    # All argument names will have _ replaced with - on the command line
    def __init__(self,
                 my_argument,
                 my_argument_2=False,
                 my_argument_3=1.0,
                 **kwargs):
        super().__init__(**kwargs)

        self._my_argument = self.get_int_arg('my_argument', my_argument)
        self._my_argument_2 = self.get_bool_arg('my_argument_2', my_argument_2)
        self._my_argument_3 = self.get_float_arg('my_argument_3', my_argument_3)

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        print("FOO:", self._my_argument, self._my_argument_2, self._my_argument_3)
        return image

    def post_resize(self, image: PIL.Image):
        print("BAR:", self._my_argument, self._my_argument_2, self._my_argument_3)
        return image


class BooFabPreprocessor(dgenerate.preprocessors.ImagePreprocessor):
    NAMES = ['boo', 'fab']

    # All argument names will have _ replaced with - on the command line
    ARGS = {
        'boo': ['my_argument', ('my_argument_2', False), ('my_argument_3', 1.0)],
        'fab': ['my_argument', ('my_argument_2', 2.0), ('my_argument_3', False)]
    }

    @staticmethod
    def help(called_by_name):
        if called_by_name == 'boo':
            return 'My --image-preprocessor-help "boo" documentation'
        if called_by_name == 'fab':
            return 'My --image-preprocessor-help "fab" documentation'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # args in **kwargs will use underscores
        if self.called_by_name == 'boo':
            self._my_argument = self.get_int_arg('my_argument', kwargs['my_argument'])
            self._my_argument_2 = self.get_bool_arg('my_argument_2', kwargs)
            self._my_argument_3 = self.get_float_arg('my_argument_3', kwargs)
        if self.called_by_name == 'fab':
            self._my_argument = self.get_int_arg('my_argument', kwargs['my_argument'])
            self._my_argument_2 = self.get_float_arg('my_argument_2', kwargs)
            self._my_argument_3 = self.get_bool_arg('my_argument_3', kwargs)

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        print(f'{self.called_by_name}:', self._my_argument, self._my_argument_2, self._my_argument_3)
        return image

    def post_resize(self, image: PIL.Image):
        print(f'{self.called_by_name}:', self._my_argument, self._my_argument_2, self._my_argument_3)
        return image
