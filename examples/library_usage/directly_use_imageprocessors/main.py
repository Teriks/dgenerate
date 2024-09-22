import PIL.Image

import dgenerate.arguments
from dgenerate.imageprocessors import ImageProcessorLoader

# We can use this to parse and validate any --device argument that gets passed
device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()


# Any image processor plugin implemented by dgenerate can be reused easily

loader = ImageProcessorLoader()

# we can print out the canny edge detect processor plugin help for reference
print(loader.get_help('canny'))

canny = loader.load('canny;lower=50;upper=100')

with PIL.Image.open('../../media/man-fighting-pose.jpg') as img:
    # please note that img will most likely be closed by this
    # call here, because it is not going to be modified in place
    # by this processor implementation and dgenerate will make
    # a copy and dispose the original.
    out_image = canny.process(img)

    # if you want to retain a copy, always pass a copy, management
    # of image object lifetime is aggressive. in this instance it
    # does not matter as we do not reuse the input image.

    out_image.save('canny-man-fighting-pose.png')

# You can create rigging images with openpose for example

# print out the openpose processor plugin help for reference
print(loader.get_help('openpose'))

openpose = loader.load('openpose', device=device)

with PIL.Image.open('../../media/man-fighting-pose.jpg') as img:
    out_image = openpose.process(img)

    out_image.save('openpose-man-fighting-pose.png')

# You can upscale using chaiNNer compatible models using
# the upscaler implementation


# print out the upscaler processor plugin help for reference
print(loader.get_help('upscaler'))

upscaler = loader.load(
    'upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
    device=device)

with PIL.Image.open('../../media/earth.jpg') as img:
    out_image = upscaler.process(img)

    out_image.save('upscaled-earth.png')
