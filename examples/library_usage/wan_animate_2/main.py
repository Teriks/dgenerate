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

config.model_path = 'Wan-AI/Wan2.2-Animate-2-14B-Diffusers'
config.model_type = ModelType.WAN_ANIMATE_2

config.dtype = DataType.BFLOAT16
config.inference_steps = [40]
config.video_fps = [24]
config.animation_format = 'mp4'
config.output_configs = True  # write a .dgen config next to each clip
config.output_size = (640, 800)
config.max_sequence_length = 512
config.wan_segment_frame_lengths = [81]
config.wan_prev_segment_frames = [1]

image = 'https://raw.githubusercontent.com/Wan-Video/Wan-Animate-2/main/examples/demo1/reference.png'
driving = 'https://raw.githubusercontent.com/Wan-Video/Wan-Animate-2/main/examples/demo1/template.mp4'

config.image_seeds = [f'{image};wan-driving={driving};frame-end=48']
config.prompts = [Prompt.parse('Character appearance: a silver-gray tabby kitten with a round face, upright ears, and large circular eyes. It wears a dark blue uniform with a gold-buttoned blazer and a pleated skirt, a white shirt underneath, and a red bow at the collar with white cuffs at the sleeves. Background: plain white, even bright lighting, no other props or decorations.')]

config.device = device

config.model_sequential_offload = True


# Video models write a finished clip to config.output_path, they do
# not emit ImageGeneratedEvent, so writes must stay enabled

render_loop = RenderLoop(config=config)

for event in render_loop.events():
    if isinstance(event, AnimationFileFinishedEvent):
        print('Video:', event.path)
        print('Reproduce With Config:', event.config_filename)
