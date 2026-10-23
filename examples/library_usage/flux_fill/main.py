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

config.model_path = 'black-forest-labs/FLUX.1-Fill-dev'
config.model_type = ModelType.FLUX_FILL

config.dtype = DataType.BFLOAT16
config.inference_steps = [50]
config.guidance_scales = [30]
config.output_size = (1024, 1024)

image = 'https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png'
mask = 'https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png'

config.image_seeds = [f'{image};{mask}']
config.prompts = [Prompt.parse('a white paper cup')]

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
