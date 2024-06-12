import typing

import PIL.Image
import PIL.ImageOps

import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    Prompt
from dgenerate.imageprocessors import ImageProcessor

# We can use this to parse and validate any --device argument that gets passed
device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = 'cuda'


# You can add image processors either explicitly
# through a class implementation you register or by
# loading them out of a module or python file

# This shows how to register a class manually
# and touches a bit on loading from a module or
# file in comments


class MyProcessor(ImageProcessor):
    # A more indepth example for how to implement these
    # can be found in the examples/writing_plugins/image_processor folder
    # This is a very minimal implementation that just inverts the image
    # before it gets resized

    NAMES = ['foo']

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: tuple[int, int] | None):
        return PIL.ImageOps.invert(image)

    def impl_post_resize(self, image: PIL.Image.Image):
        return image


config = RenderLoopConfig()

config.model_path = 'stabilityai/stable-diffusion-2'
config.inference_steps = [40]
config.guidance_scales = [5]
config.prompts = [Prompt.parse('a strange alien planet, view from orbit')]

config.device = device

# We need an image seed definition to use a processor
# This uses the exact same syntax as --image-seeds from the
# command line usage.
config.image_seeds = ['../../media/earth.jpg']

# Request to use the processor above by name on our singular image seed,
# (added to the render loop below) ability to access processors implemented
# by dgenerate is automatic, see: dgenerate --image-processor-help
config.seed_image_processors = ['foo']

# below is the default value for --image-seed-strengths

# config.image_seed_strengths = [0.8]

# One seed will be randomly generated for you if none are specified

# config.seeds = [123456789]


render_loop = RenderLoop(config=config)

# Add our processor class
render_loop.image_processor_loader.add_class(MyProcessor)

# Find processor classes in a module and add them, this module
# enters dgenerates plugin cache

# render_loop.image_processor_loader.load_plugin_modules(['my_python_module'])

# Find processor classes in a file and add them, this file enters
# dgenerates plugin cache

# render_loop.image_processor_loader.load_plugin_modules(['my_python_file.py'])

# disables all writes to disk
render_loop.disable_writes = True

# run the render loop and handle events,
# you could also use render_loop.run() if you did not care
# about events and just wanted to write to disk

for event in render_loop.events():
    if isinstance(event, dgenerate.ImageGeneratedEvent):
        print('Filename:', event.suggested_filename)
        print('Seed:', event.diffusion_args.seed)
        print('Prompt:', event.diffusion_args.prompt)
        print('Reproduce With Command:', event.command_string)
        print(f'Reproduce With Config:\n{event.config_string}')
        event.image.save(event.suggested_filename)

        # if you wish to work with any image offered by an event object
        # in the event stream outside of the event handler you have written,
        # you should copy it out with arg.image.copy(). management of PIL.Image
        # lifetime is very aggressive and the image objects in events will be disposed
        # of when no longer needed for the event, IE. have .close() called on them
        # making them unusable
