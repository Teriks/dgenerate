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
    device = dgenerate.default_device()

config = RenderLoopConfig()

# Distilled FLUX.3 Image Klein. 16 steps, guidance 1, no negative prompt.
# Leave scheduler_uri unset so the checkpoint scheduler is kept.

config.model_path = 'black-forest-labs/flux-3-image-klein'
config.model_type = ModelType.FLUX3

config.dtype = DataType.BFLOAT16
config.inference_steps = [16]
config.guidance_scales = [1]
config.output_size = (1024, 1024)

config.prompts = [Prompt.parse('A red fox in a snow-covered forest clearing at golden hour')]

config.device = device

config.model_sequential_offload = True


render_loop = RenderLoop(
    config=config,
    disable_writes=True  # disables all writes to disk
)

for event in render_loop.events():
    if isinstance(event, ImageGeneratedEvent):
        print('Filename:', event.suggested_filename)
        print('Seed:', event.diffusion_args.seed)
        print('Prompt:', event.diffusion_args.prompt)
        print('Reproduce With Command:', event.command_string)
        print(f'Reproduce With Config:\n{event.config_string}')
        event.image.save(event.suggested_filename)
