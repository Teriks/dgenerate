import sys

import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    AnimationFileFinishedEvent, \
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

config.model_path = 'Lightricks/LTX-2.5-Diffusers'
config.model_type = ModelType.LTX2

config.dtype = DataType.BFLOAT16
config.guidance_scales = [1]
config.video_lengths = [2]
config.video_fps = [24]
config.animation_format = 'mp4'
config.output_configs = True  # write a .dgen config next to each clip
config.output_size = (640, 384)

config.prompts = [Prompt.parse('A woman in a red coat walks along a dirt path through a green field, then stops and looks back at the camera. The camera stays fixed.')]

config.device = device

config.model_sequential_offload = True


# Video models write a finished clip to config.output_path, they do
# not emit ImageGeneratedEvent, so writes must stay enabled

render_loop = RenderLoop(config=config)

for event in render_loop.events():
    if isinstance(event, AnimationFileFinishedEvent):
        print('Video:', event.path)
        print('Reproduce With Config:', event.config_filename)
