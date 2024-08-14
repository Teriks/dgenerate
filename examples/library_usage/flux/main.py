import sys

import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    ImageGeneratedEvent, \
    Prompt, \
    ModelType, \
    DataType

import os

if not os.environ.get('HF_TOKEN'):
    print('Set HF_TOKEN environmental variable to run this example.')
    sys.exit(0)

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = 'cuda:1'

config = RenderLoopConfig()

# Flux schnell for minimal inference steps

config.model_path = 'black-forest-labs/FLUX.1-schnell'
config.model_type = ModelType.TORCH_FLUX

# reduce memory consumption with brain float
config.dtype = DataType.BFLOAT16

config.inference_steps = [4]
config.guidance_scales = [0]

# Flux does not support negative prompting currently
config.prompts = [Prompt.parse('an astronaut walking on the moon')]

config.device = device

# It is likely you will need this enabled to run Flux on
# average consumer hardware, even with bfloat8 quantization
# of the text encoder and transformer (which is not used here)
config.model_sequential_offload = True

# further reduce resource usage / memory consumption
config.flux_max_sequence_length = 256

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
