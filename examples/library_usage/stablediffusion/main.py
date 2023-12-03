import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    ImageGeneratedEvent, \
    Prompt

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = 'cuda'

config = RenderLoopConfig()

config.model_path = 'stabilityai/stable-diffusion-2'
config.inference_steps = [40]
config.guidance_scales = [5]
config.prompts = [Prompt.parse('an astronaut walking on the moon; fake')]
config.device = device

# One seed will be randomly generated for you if none are specified

# config.seeds = [123456789]


render_loop = RenderLoop(config=config)

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

        # if you wish to work with this image after the completion
        # of your event stream handler you should copy it out with arg.image.copy()
        # management of PIL.Image lifetime is very aggressive and the
        # image object given in this event will be disposed of
        # when handling of the event is finished
