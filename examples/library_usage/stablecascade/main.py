import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    ImageGeneratedEvent, \
    Prompt, \
    DataType, \
    ModelType

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = 'cuda'

config = RenderLoopConfig()

config.model_path = 'stabilityai/stable-cascade-prior'
config.model_type = ModelType.TORCH_S_CASCADE
config.dtype = DataType.BFLOAT16
config.s_cascade_decoder_uri = 'stabilityai/stable-cascade;dtype=float16'
config.inference_steps = [20]
config.guidance_scales = [4]
config.s_cascade_decoder_inference_steps = [10]
config.s_cascade_decoder_guidance_scales = [0]
config.prompts = [Prompt.parse('an image of a shiba inu, donning a spacesuit and helmet')]
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
    if isinstance(event, ImageGeneratedEvent):
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
