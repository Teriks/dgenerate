import os

import dgenerate.arguments
import dgenerate.messages
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    StartingGenerationStepEvent, \
    ImageGeneratedEvent, \
    ImageFileSavedEvent, \
    Prompt, \
    ModelType

# Static image variations.
#
# One still prompt and two seeds. Each seed is its own generation step:
# StartingGenerationStepEvent, then ImageGeneratedEvent, then
# ImageFileSavedEvent when writes are enabled. There is no animation event.
# An animated image seed is frame_by_frame_animation_main.py. A video
# pipeline is video_pipeline_main.py.
#
# The first run lets the loop write each still. The second run keeps the
# image in memory and saves it from ImageGeneratedEvent. event.image is
# closed when the loop moves on. Keep a copy with event.image.copy().

dgenerate.messages.messages_to_null()
dgenerate.messages.errors_to_null()

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

config = RenderLoopConfig()

config.model_path = 'sd2-community/stable-diffusion-2'
config.model_type = ModelType.SD
config.inference_steps = [30]
config.guidance_scales = [5]
config.seeds = [1, 2]
config.prompts = [Prompt.parse('an astronaut walking on the moon; artwork')]
config.output_configs = True
config.device = device

def handle(event, save_image):
    if isinstance(event, StartingGenerationStepEvent):
        print('Generation Step:', event.generation_step, 'of', event.total_steps)
    if isinstance(event, ImageGeneratedEvent):
        print('Image Generated (Suggested Filename):', event.suggested_filename)
        print('Image Generated (Suggested Directory):', event.suggested_directory)
        print('Seed:', event.diffusion_args.seed)
        print('Guidance Scale:', event.diffusion_args.guidance_scale)
        # The loop writes the still when writes are enabled.
        if save_image:
            path = os.path.join(event.suggested_directory, event.suggested_filename)
            os.makedirs(event.suggested_directory, exist_ok=True)
            event.image.save(path)
            print('Image Saved:', path)
    if isinstance(event, ImageFileSavedEvent):
        print('Image Saved:', event.path)
        print('Config Saved:', event.config_filename)


print('Writes enabled')

render_loop = RenderLoop(config=config, disable_writes=False)

for event in render_loop.events():
    handle(event, save_image=False)

print('Writes disabled')

render_loop = RenderLoop(config=config, disable_writes=True)

for event in render_loop.events():
    handle(event, save_image=True)
