import typing

import PIL.Image

import dgenerate.imageprocessors
from dgenerate.plugin import PluginArg as _Pa


class FooBarImageProcessor(dgenerate.imageprocessors.ImageProcessor):
    """My --image-processor-help documentation, arguments are described automatically"""

    # This static property defines what names this module can be invoked by
    NAMES = ['foobar']

    # you can hide base class arguments or any argument from URI usage
    # if your processor does not support the argument
    HIDE_ARGS = ['device']

    # You can force a specific image alignment to be obeyed by overriding get_alignment
    # the input image is forcefully aligned to the returned value, the user will be warned
    # about this if forced alignment takes place

    # def get_alignment(self):
    #    return 64

    # All argument names will have _ replaced with - on the command line.
    # Argument signature correctness (missing arguments, unknown arguments) etc.
    # is verified by dgenerate, adding type hints will cause the argument values
    # to be parsed into that type automatically with validation when parsed from a URI,
    # arguments without type hints will be parsed as python literals unless they cannot
    # be parsed into a python literal, in which case they will be passed straight through
    # as a raw string.

    # The type hint typing.Optional[...] is supported to indicate that an argument is optional,
    # when a value is passed to an optional argument that value will be validated against the
    # specified optional type, if you do not provide a default None value in the constructor
    # (which you should), dgenerate will pass None for you. The type hints (list, dict, set) are
    # are supported as type hints, and they will be parsed into their respective types when
    # defined in the URI with their python literal syntax, this is also supported for optionals

    # The type hint typing.Union[...] may also be used with multiple types present

    # Python type hint syntax using the | operator to declare type hints is not supported
    def __init__(self,
                 my_argument: str,
                 my_argument_2: bool = False,
                 my_argument_3: float = 1.0,
                 my_argument_4: typing.Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self._my_argument = my_argument
        self._my_argument_2 = my_argument_2
        self._my_argument_3 = my_argument_3
        self._my_argument_4 = my_argument_4

        # you can raise custom argument errors with self.argument_error

        # raise self.argument_error('My argument error message')

        # if we are using a large model of some sort
        # we can set a size estimate before proceeding to
        # load it into memory, dgenerate will try to
        # free up space for it if any image processor
        # cpu side memory constraints have been met
        #
        # it will also use this estimate to try
        # to free up cached objects in VRAM when
        # the processor is moving modules to a device

        # self.set_size_estimate(1024)

        # register anything that needs to
        # move onto the device requested by the user
        # anything registered should have a .to(device)
        # argument accepting at least one argument,
        # a torch.device reference or device string

        # self.register_module(have_to)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: tuple | None):

        # This step runs before dgenerate resizes an image to the value of --output-size
        # "resize_resolution" is the resolution that the image is going to be resized to
        # or None if no resizing is going to occur

        # If you do not do any processing in this step, return the image as is

        # If you do modify the image, it is acceptable to modify it in place, or
        # to return a copy of it

        # Make a white square in the upper left corner which will proc
        # a debug image write as requested in the config using the 'output-file` argument
        for x in range(0, 100):
            for y in range(0, 100):
                image.putpixel((x, y), (255, 255, 255))

        print("FOO:", self._my_argument, self._my_argument_2, self._my_argument_3)
        return image

    def impl_post_resize(self, image: PIL.Image):
        # This step runs after dgenerate is done resizing the image to its final size,
        # after which it will be passed into a diffusion pipeline

        # If have an image processing step that produced an image that
        # should not be upscaled or downscaled for quality reasons (probably most of the time),
        # this is the place you should do your processing

        # If you do not do any processing in this step, return the image as is

        # If you do modify the image, it is acceptable to modify it in place, or
        # to return a copy of it

        print("BAR:", self._my_argument, self._my_argument_2, self._my_argument_3)
        return image


# Below is an example of a processor module that can be invoked
# by multiple names and handle multiple processing tasks within
# the same class

class BooFabImageProcessor(dgenerate.imageprocessors.ImageProcessor):
    # This static property defines what names this module can be invoked by
    NAMES = ['boo', 'fab']

    # All argument names will have _ replaced with - on the command line
    # This static property can be used to define the argument signature
    # for each of the invokable names above
    ARGS = {
        'boo': [_Pa('my_argument'),
                _Pa('my_argument_2', type=bool, default=False),
                _Pa('my_argument_3', type=float, default=1.0)],

        'fab': [_Pa('my_argument'),
                _Pa('my_argument_2', type=float, default=2.0),
                _Pa('my_argument_3', type=bool, default=False)]
    }

    # you can hide base class arguments or any argument from URI usage
    # if your processor does not support the argument, this indicates
    # calling this processor by the name boo disables the device argument
    HIDE_ARGS = {'boo': 'device'}

    # Defining the static method "help" allows you to provide a help string
    # for each invokable name, for --image-processor-help "boo" and --image-processor-help "foo"
    # the name this option is used with is passed to "loaded_by_name"
    @staticmethod
    def help(loaded_by_name):
        if loaded_by_name == 'boo':
            return 'My --image-processor-help "boo" documentation'
        if loaded_by_name == 'fab':
            return 'My --image-processor-help "fab" documentation'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # "self.loaded_by_name" contains the name string
        # that this module is being invoked by, we can do something
        # different in the module depending on what name is used
        # to invoke it

        if self.loaded_by_name == 'boo':
            self._my_argument = kwargs['my_argument']
            self._my_argument_2 = kwargs['my_argument_2']
            self._my_argument_3 = kwargs['my_argument_3']

        if self.loaded_by_name == 'fab':
            self._my_argument = kwargs['my_argument']
            self._my_argument_2 = kwargs['my_argument_2']
            self._my_argument_3 = kwargs['my_argument_3']

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: tuple | None):
        print(f'{self.loaded_by_name}:', self._my_argument, self._my_argument_2, self._my_argument_3)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        print(f'{self.loaded_by_name}:', self._my_argument, self._my_argument_2, self._my_argument_3)
        return image
