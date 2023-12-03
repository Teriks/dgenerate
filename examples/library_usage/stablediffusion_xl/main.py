import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    Prompt, \
    ModelType, \
    DataType

# We can use this to parse and validate any --device argument that gets passed
device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = 'cuda'

config = RenderLoopConfig()

config.model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

# The fp16 model variant is smaller and makes this easier to run on average hardware
config.variant = 'fp16'

config.inference_steps = [40]
config.guidance_scales = [5]

# This accepts a URI, IE the syntax supported by --sdxl-refiner itself
# which includes things like variant specification, which defaults to the
# main models variant specification above in this case.
# the syntax supported by _uri config options aligns one to one with
# dgenerates command line usage for the corresponding options

config.sdxl_refiner_uri = 'stabilityai/stable-diffusion-xl-refiner-1.0'

# This is the default value for this option,
# there are many options available for sdxl
# they all start with sdxl_, same as the command
# line usage of dgenerate

# config.sdxl_high_noise_fractions = [0.8]


config.prompts = [Prompt.parse('an astronaut walking on the moon; fake')]

config.device = device

config.model_type = ModelType.TORCH_SDXL

# Lower GPU memory consumption with this data type
config.dtype = DataType.FLOAT16

# One seed will be randomly generated for you if none are specified

# config.seeds = [123456789]

render_loop = RenderLoop(config=config)

# Output size for SDXL defaults to 1024


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
