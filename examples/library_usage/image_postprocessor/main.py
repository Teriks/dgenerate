import PIL.Image
import PIL.ImageOps

from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    ImageGeneratedCallbackArgument, \
    Prompt
from dgenerate.postprocessors import ImagePostprocessor


# You can add postprocessors either explicitly
# through a class implementation you register or by
# loading them out of a module or python file

# This shows how to register a class manually
# and touches a bit on loading from a module or
# file in comments


class MyPostprocessor(ImagePostprocessor):
    # A more indepth example for how to implement these
    # can be found in the examples/writing_plugins/image_postprocessor folder
    # This is a very minimal implementation that just inverts the image
    # after it gets generated

    NAMES = ['foo']

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def impl_process(self, image: PIL.Image.Image):
        return PIL.ImageOps.invert(image)


config = RenderLoopConfig()

config.model_path = 'stabilityai/stable-diffusion-2'
config.inference_steps = [40]
config.guidance_scales = [5]
config.prompts = [Prompt.parse('a right side profile view of a male marble statue head')]
config.device = 'cuda'

# Request to use the postprocessor above by name on our singular image seed,
# (added to the render loop below) ability to access builtin postprocessors implemented
# by dgenerate is automatic, see: dgenerate --postprocessors-help
config.postprocessors = ['foo']


# One seed will be randomly generated for you if none are specified

# config.seeds = [123456789]

render_loop = RenderLoop(config=config)


# Add our postprocessor class
render_loop.postprocessor_loader.add_class(MyPostprocessor)


# Find postprocessor classes in a module and add them, this module
# enters dgenerates plugin cache

# render_loop.postprocessor_loader.load_plugin_modules(['my_python_module'])

# Find postprocessor classes in a file and add them, this file enters
# dgenerates plugin cache

# render_loop.postprocessor_loader.load_plugin_modules(['my_python_file.py'])

def render_callback(arg: ImageGeneratedCallbackArgument):
    print('Filename:', arg.suggested_filename)
    print('Seed:', arg.diffusion_args.seed)
    print('Prompt:', arg.diffusion_args.prompt)
    print('Reproduce With Command:', arg.command_string)
    print(f'Reproduce With Config:\n{arg.config_string}')
    arg.image.save(arg.suggested_filename)

    # if you wish to work with this image after the completion
    # of render_loop.run you should copy it out with arg.image.copy()
    # management of PIL.Image lifetime is very aggressive and the
    # image object given in this callback will be disposed of
    # when the callback is finished


render_loop.disable_writes = True

render_loop.image_generated_callbacks.append(render_callback)

render_loop.run()
