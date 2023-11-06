import typing

import PIL.Image
import PIL.ImageOps

from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    ImageGeneratedCallbackArgument, \
    Prompt
from dgenerate.preprocessors import ImagePreprocessor


# You can add image preprocessors either explicitly
# through a class implementation you register or by
# loading them out of a module or python file

# This shows how to register a class manually
# and touches a bit on loading from a module or
# file in comments


class MyPreprocessor(ImagePreprocessor):
    # A more indepth example for how to implement these
    # can be found in the examples/plugins/image_preprocessor folder
    # This is a very minimal implementation that just inverts the image
    # before it gets resized

    NAMES = ['foo']

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def pre_resize(self, image: PIL.Image.Image, resize_resolution: typing.Union[None, tuple]):
        return PIL.ImageOps.invert(image)

    def post_resize(self, image: PIL.Image.Image):
        return image


config = RenderLoopConfig()

config.model_path = 'stabilityai/stable-diffusion-2'
config.inference_steps = [40]
config.guidance_scales = [5]
config.prompts = [Prompt.parse('a strange alien planet, view from orbit')]
config.output_path = ''
config.device = 'cuda'

# We need an image seed definition to use a preprocessor
# This uses the exact same syntax as --image-seeds from the
# command line usage.
config.image_seeds = ['../../media/earth.jpg']

# Request to use the preprocessor above by name on our singular image seed,
# (added to the render loop below) ability to access preprocessors implemented
# by dgenerate is automatic, see: dgenerate --image-preprocessor-help
config.seed_image_preprocessors = ['foo']

# below is the default value for --image-seed-strengths

# config.image_seed_strengths = [0.8]

# One seed will be randomly generated for you if none are specified

# config.seeds = [123456789]


render_loop = RenderLoop(config=config)

# Add our preprocessor class
render_loop.preprocessor_loader.add_class(MyPreprocessor)


# Find preprocessor classes in a module and add them, this module
# enters dgenerates plugin cache

# render_loop.preprocessor_loader.load_plugin_modules(['my_python_module'])

# Find preprocessor classes in a file and add them, this file enters
# dgenerates plugin cache

# render_loop.preprocessor_loader.load_plugin_modules(['my_python_file.py'])

def render_callback(arg: ImageGeneratedCallbackArgument):
    print('Filename:', arg.suggested_filename)
    print('Seed:', arg.diffusion_args.seed)
    print('Prompt:', arg.diffusion_args.prompt)
    print('Reproduce With Command:', arg.command_string)
    print(f'Reproduce With Config:\n{arg.config_string}')
    arg.image.save(arg.suggested_filename)


render_loop.disable_writes = True

render_loop.image_generated_callbacks.append(render_callback)

render_loop.run()
