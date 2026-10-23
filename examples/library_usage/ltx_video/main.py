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

config.model_path = 'Lightricks/LTX-Video'
config.model_type = ModelType.LTX_VIDEO

config.dtype = DataType.BFLOAT16
config.model_cpu_offload = True
config.transformer_uri = 'https://huggingface.co/city96/LTX-Video-gguf/blob/main/ltx-video-2b-v0.9-Q3_K_S.gguf'
config.inference_steps = [50]
config.guidance_scales = [3]
config.video_lengths = [4.84]
config.video_fps = [25]
config.animation_format = 'mp4'
config.output_configs = True  # write a .dgen config next to each clip
config.output_size = (768, 512)

config.prompts = [Prompt.parse('A woman in a red coat walks along a dirt path through a green field, then stops and looks back at the camera. The camera stays fixed.; worst quality, inconsistent motion, blurry, jittery, distorted')]

config.device = device


# Video models write a finished clip to config.output_path, they do
# not emit ImageGeneratedEvent, so writes must stay enabled

render_loop = RenderLoop(config=config)

for event in render_loop.events():
    if isinstance(event, AnimationFileFinishedEvent):
        print('Video:', event.path)
        print('Reproduce With Config:', event.config_filename)
