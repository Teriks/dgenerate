import PIL.Image

from dgenerate.preprocessors import Loader, ImagePreprocessorMixin

# Any image preprocessor plugin implemented by dgenerate can be reused easily

loader = Loader()

# we can print out the canny edge detect preprocessor plugin help for reference
print(loader.get_help('canny'))

# Given that preprocessors have two steps, pre_resize and post_resize
# it is easiest to use the mixin class which implements the resizing
# operation if necessary and calls into the plugin appropriately

canny = ImagePreprocessorMixin(loader.load('canny;lower=50;upper=100', device='cuda'))

with PIL.Image.open('../../media/man-fighting-pose.jpg') as img:
    # The default value of resize_to is None but it
    # is specified for clarity here

    # please note that img will most likely be closed by this
    # call here, because it is not going to be modified in place
    # by this preprocessor implementation and dgenerate will make
    # a copy and dispose the original.
    out_image = canny.preprocess_image(img, resize_to=None)

    # if you want to retain a copy, always pass a copy, management
    # of image object lifetime is aggressive. in this instance it
    # does not matter as we do not reuse the input image.

    out_image.save('canny-man-fighting-pose.png')


# You can create rigging images with openpose for example

# print out the openpose preprocessor plugin help for reference
print(loader.get_help('openpose'))

openpose = ImagePreprocessorMixin(loader.load('openpose', device='cuda'))

with PIL.Image.open('../../media/man-fighting-pose.jpg') as img:
    out_image = openpose.preprocess_image(img, resize_to=None)

    out_image.save('openpose-man-fighting-pose.png')
