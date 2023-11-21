from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    ImageGeneratedCallbackArgument, \
    Prompt

import dgenerate.arguments

args = dgenerate.arguments.parse_known_args()

config = RenderLoopConfig()

config.model_path = 'stabilityai/stable-diffusion-2'
config.inference_steps = [40]
config.guidance_scales = [5]
config.prompts = [Prompt.parse('an astronaut walking on the moon; fake')]
config.device = args.device

# One seed will be randomly generated for you if none are specified

# config.seeds = [123456789]


render_loop = RenderLoop(config=config)


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
