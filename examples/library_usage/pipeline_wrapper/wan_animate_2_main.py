import dgenerate
from dgenerate import DataType, ModelType, Prompt
from dgenerate.mediainput import load_rgb_frames

# Wan-Animate-2 through PipelineWrapper.
#
# There is no pose or face preprocess. The driving frames are the motion
# source. The wrapper resamples them from wan_driving_video_fps to video_fps
# inside the call. A still image leaves that source rate unset, and every
# loaded frame is kept. The output length follows the resampled clip.
#
# The render-loop version, which loads the seed for you, is
# examples/library_usage/wan_animate_2/main.py.

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

wrapper = dgenerate.PipelineWrapper(
    model_path='Wan-AI/Wan2.2-Animate-2-14B-Diffusers',
    model_type=ModelType.WAN_ANIMATE_2,
    dtype=DataType.BFLOAT16,
    device=device,
    model_sequential_offload=True
)

image = 'https://raw.githubusercontent.com/Wan-Video/Wan-Animate-2/main/examples/demo1/reference.png'
driving = 'https://raw.githubusercontent.com/Wan-Video/Wan-Animate-2/main/examples/demo1/template.mp4'

character, _ = load_rgb_frames(image)
driving_frames, source_fps = load_rgb_frames(driving, frame_end=48)

args = dgenerate.GenerationArguments()
args.prompt = Prompt.parse(
    'Character appearance: a silver-gray tabby kitten with a round face, upright ears, '
    'and large circular eyes. It wears a dark blue uniform with a gold-buttoned blazer '
    'and a pleated skirt, a white shirt underneath, and a red bow at the collar with '
    'white cuffs at the sleeves. Background: plain white, even bright lighting, no '
    'other props or decorations.')
args.images = character
args.wan_driving_video_frames = driving_frames
args.wan_driving_video_fps = source_fps
args.inference_steps = 40
args.video_fps = 24
args.width = 640
args.height = 800
args.max_sequence_length = 512
args.wan_segment_frame_length = 81
args.wan_prev_segment_frames = 1

result = wrapper(args)
result.write('wan-animate-2.mp4')
