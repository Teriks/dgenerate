import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    AnimationFileFinishedEvent, \
    Prompt, \
    ModelType, \
    DataType

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

config = RenderLoopConfig()

config.model_path = 'Wan-AI/Wan2.2-Animate-14B-Diffusers'
config.model_type = ModelType.WAN_ANIMATE

config.dtype = DataType.BFLOAT16
config.inference_steps = [20]
config.guidance_scales = [1]
# extract pose and face clips from the driving video
config.wan_animate_preprocess = True
config.video_fps = [30]
config.animation_format = 'mp4'
config.output_configs = True  # write a .dgen config next to each clip
config.output_size = (832, 480)

image = 'https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/wan_animate/animate/image.jpeg'
driving = 'https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/wan_animate/animate/video.mp4'

config.image_seeds = [f'{image};wan-driving={driving};frame-end=48']
config.prompts = [Prompt.parse('A fantasy woman with pale lavender skin, glowing yellow eyes, pointed ears, and long dark hair, wearing dark ornate armor with gold accents. She performs the motion from the driving video. Cinematic lighting, ultrarealistic detail.')]

config.device = device

config.model_sequential_offload = True


# Video models write a finished clip to config.output_path, they do
# not emit ImageGeneratedEvent, so writes must stay enabled

render_loop = RenderLoop(config=config)

for event in render_loop.events():
    if isinstance(event, AnimationFileFinishedEvent):
        print('Video:', event.path)
        print('Reproduce With Config:', event.config_filename)
