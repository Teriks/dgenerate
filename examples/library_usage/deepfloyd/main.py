import os
import sys

import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    ImageGeneratedEvent, \
    Prompt, \
    ModelType, \
    DataType

if not os.environ.get('HF_TOKEN'):
    print('Set HF_TOKEN environmental variable to run this example.')
    sys.exit(0)

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'

# Stage 1: DeepFloyd IF, produces a 64x64 image

config = RenderLoopConfig()

config.model_path = 'DeepFloyd/IF-I-M-v1.0'
config.model_type = ModelType.IF
config.variant = 'fp16'
config.dtype = DataType.FLOAT16
config.inference_steps = [60]
config.guidance_scales = [7]
config.output_size = (64, 64)
config.safety_checker = True
config.prompts = [Prompt.parse(prompt)]
config.output_prefix = 'stage1'
config.device = device
config.model_sequential_offload = True

stage1_images = []
stage1_seeds = []

render_loop = RenderLoop(config=config, disable_writes=True)

for event in render_loop.events():
    if isinstance(event, ImageGeneratedEvent):
        print('Stage 1 image:', event.suggested_filename)
        event.image.save(event.suggested_filename)
        stage1_images.append(event.suggested_filename)
        stage1_seeds.append(event.diffusion_args.seed)

# Stage 2: DeepFloyd IF super-resolution, from stage 1 images

config = RenderLoopConfig()

config.model_path = 'DeepFloyd/IF-II-M-v1.0'
config.model_type = ModelType.IFS
config.variant = 'fp16'
config.dtype = DataType.FLOAT16
config.inference_steps = [30]
config.guidance_scales = [4]
config.seeds = stage1_seeds
config.seeds_to_images = True
config.image_seeds = stage1_images
config.prompts = [Prompt.parse(prompt)]
config.output_prefix = 'stage2'
config.device = device
config.model_sequential_offload = True

stage2_images = []
stage2_seeds = []

render_loop = RenderLoop(config=config, disable_writes=True)

for event in render_loop.events():
    if isinstance(event, ImageGeneratedEvent):
        print('Stage 2 image:', event.suggested_filename)
        event.image.save(event.suggested_filename)
        stage2_images.append(event.suggested_filename)
        stage2_seeds.append(event.diffusion_args.seed)

# Stage 3: SD x4 upscaler, from stage 2 images

config = RenderLoopConfig()

config.model_path = 'stabilityai/stable-diffusion-x4-upscaler'
config.model_type = ModelType.UPSCALER_X4
config.variant = 'fp16'
config.dtype = DataType.FLOAT16
config.inference_steps = [30]
config.guidance_scales = [9.0]
config.upscaler_noise_levels = [20]
config.seeds = stage2_seeds
config.seeds_to_images = True
config.image_seeds = stage2_images
config.prompts = [Prompt.parse(prompt)]
config.output_prefix = 'stage3'
config.device = device

render_loop = RenderLoop(config=config, disable_writes=True)

for event in render_loop.events():
    if isinstance(event, ImageGeneratedEvent):
        print('Stage 3 image:', event.suggested_filename)
        print('Seed:', event.diffusion_args.seed)
        print('Prompt:', event.diffusion_args.prompt)
        print('Reproduce With Command:', event.command_string)
        print(f'Reproduce With Config:\n{event.config_string}')
        event.image.save(event.suggested_filename)
