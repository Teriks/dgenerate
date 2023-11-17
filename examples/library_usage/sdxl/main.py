from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    ImageGeneratedCallbackArgument, \
    Prompt, \
    ModelTypes, \
    DataTypes


import dgenerate.arguments

# We can use this to parse and validate any --device argument that gets passed
device = dgenerate.arguments.parse_device()


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

config.model_type = ModelTypes.TORCH_SDXL

# Lower GPU memory consumption with this data type
config.dtype = DataTypes.FLOAT16

# One seed will be randomly generated for you if none are specified

# config.seeds = [123456789]


render_loop = RenderLoop(config=config)


# Output size for SDXL defaults to 1024


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
