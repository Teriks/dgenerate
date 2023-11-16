import argparse

import PIL.Image

from dgenerate.imageprocessors import Loader

arg_parser = argparse.ArgumentParser(exit_on_error=False)
arg_parser.add_argument('-d', '--device', default='cuda')
args, _ = arg_parser.parse_known_args()


# Any image processor plugin implemented by dgenerate can be reused easily

loader = Loader()

# we can print out the canny edge detect processor plugin help for reference
print(loader.get_help('canny'))

canny = loader.load('canny;lower=50;upper=100', device=args.device)

with PIL.Image.open('../../media/man-fighting-pose.jpg') as img:
    # The default value of resize_to is None but it
    # is specified for clarity here

    # please note that img will most likely be closed by this
    # call here, because it is not going to be modified in place
    # by this processor implementation and dgenerate will make
    # a copy and dispose the original.
    out_image = canny.process(img, resize_to=None)

    # if you want to retain a copy, always pass a copy, management
    # of image object lifetime is aggressive. in this instance it
    # does not matter as we do not reuse the input image.

    out_image.save('canny-man-fighting-pose.png')

# You can create rigging images with openpose for example

# print out the openpose processor plugin help for reference
print(loader.get_help('openpose'))

openpose = loader.load('openpose', device=args.device)

with PIL.Image.open('../../media/man-fighting-pose.jpg') as img:
    out_image = openpose.process(img, resize_to=None)

    out_image.save('openpose-man-fighting-pose.png')
