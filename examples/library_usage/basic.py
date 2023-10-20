from dgenerate import \
    DiffusionRenderLoop, \
    DiffusionRenderLoopConfig, \
    ImageGeneratedCallbackArgument, \
    Prompt \


config = DiffusionRenderLoopConfig()

config.model_path = 'stabilityai/stable-diffusion-2'
config.inference_steps_values = [40]
config.guidance_scales = [5]
config.prompts = [Prompt().parse('an astronaut walking on the moon; fake')]
config.output_path = ''
config.device = 'cuda:1'

render_loop = DiffusionRenderLoop(config=config)


def render_callback(arg: ImageGeneratedCallbackArgument):
    print('Filename:', arg.suggested_filename)
    print('Seed:', arg.args.seed)
    print('Prompt:', arg.args.prompt)
    print('Reproduce With Command:', arg.command)
    print(f'Reproduce With Config:\n{arg.config}')
    arg.image.save(arg.suggested_filename)


render_loop.disable_writes = True

render_loop.image_generated_callbacks.append(render_callback)

render_loop.run()
