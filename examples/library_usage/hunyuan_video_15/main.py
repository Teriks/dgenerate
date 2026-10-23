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

config.model_path = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v'
config.model_type = ModelType.HUNYUAN_VIDEO_15

config.dtype = DataType.BFLOAT16
config.transformer_uri = 'https://huggingface.co/jayn7/HunyuanVideo-1.5_T2V_480p-GGUF/blob/main/480p/hunyuanvideo1.5_480p_t2v-Q4_K_S.gguf'

config.inference_steps = [50]
config.guidance_scales = [6]
config.video_lengths = [4]
config.video_fps = [15]
config.animation_format = 'mp4'
config.output_configs = True  # write a .dgen config next to each clip
config.output_size = (848, 480)

config.prompts = [Prompt.parse('A red kite glides above a windswept coastal meadow at golden hour; smooth cinematic camera movement')]

config.device = device

config.model_sequential_offload = True


# Video models write a finished clip to config.output_path, they do
# not emit ImageGeneratedEvent, so writes must stay enabled

render_loop = RenderLoop(config=config)

for event in render_loop.events():
    if isinstance(event, AnimationFileFinishedEvent):
        print('Video:', event.path)
        print('Reproduce With Config:', event.config_filename)
