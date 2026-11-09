import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    VideoGeneratedEvent, \
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


render_loop = RenderLoop(
    config=config,
    disable_writes=True  # the clip stays in memory; write() muxes it
)

for event in render_loop.events():
    if isinstance(event, VideoGeneratedEvent):
        print('Filename:', event.suggested_filename)
        print('Seed:', event.diffusion_args.seed)
        print('Prompt:', event.diffusion_args.prompt)
        print('Reproduce With Command:', event.command_string)
        print(f'Reproduce With Config:\n{event.config_string}')
        # write() accepts a path or a binary stream, such as io.BytesIO().
        # Call it here; the frames are closed when the loop moves on.
        event.write(event.suggested_filename)
