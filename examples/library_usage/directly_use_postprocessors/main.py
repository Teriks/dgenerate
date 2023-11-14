import PIL.Image

import dgenerate.postprocessors

# With the upscaler post processor, you can directly make use of chaiNNer compatible upscaler models
# IE: models from https://openmodeldb.info/ such as ESRGANs etc, using tiled upscaling

# which is extremely cool and good :)

loader = dgenerate.postprocessors.Loader()

# print out usage help for the upscaler plugin

print(loader.get_help('upscaler'))


# Use the original Real-ESRGAN x4 upscaler model from the creators Repo

upscaler = loader.load(
    'upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth', device='cuda')


with PIL.Image.open('../../media/earth.jpg') as img:

    # please note that img will most likely be closed by this
    # call here, because it is not going to be modified in place
    # by this postprocessor implementation and dgenerate will make
    # a copy and dispose the original.
    out_image = upscaler.process(img)

    # if you want to retain a copy, always pass a copy, management
    # of image object lifetime is aggressive. in this instance it
    # does not matter as we do not reuse the input image.

    out_image.save('upscaled-earth.png')
