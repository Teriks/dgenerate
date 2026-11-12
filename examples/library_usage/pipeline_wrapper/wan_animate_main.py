import dgenerate
from dgenerate import DataType, ModelType, Prompt
from dgenerate.mediainput import load_rgb_frames

# Wan-Animate through PipelineWrapper.
#
# The wrapper does not open wan-driving=. This example loads the character
# image and the driving clip, then sets wan_animate_preprocess. The call
# derives a pose frame and a face frame for each driving frame. A pose or
# face clip already set on the arguments would be kept, and the derived
# frames would fill only the missing one.
#
# wan_animate_preprocess_output writes those derived clips. The name is a
# filename stem, so motion.mp4 becomes motion.wan-pose.mp4 and
# motion.wan-face.mp4 in the output directory. Those files are 16 fps.
# The generated character clip is result.write(), at video_fps.
#
# The render-loop version, which loads the seed for you, is
# examples/library_usage/wan_animate/main.py.

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

wrapper = dgenerate.PipelineWrapper(
    model_path='Wan-AI/Wan2.2-Animate-14B-Diffusers',
    model_type=ModelType.WAN_ANIMATE,
    dtype=DataType.BFLOAT16,
    device=device,
    model_sequential_offload=True
)

image = 'https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/wan_animate/animate/image.jpeg'
driving = 'https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/wan_animate/animate/video.mp4'

character, _ = load_rgb_frames(image)
driving_frames, _ = load_rgb_frames(driving, frame_end=48)

args = dgenerate.GenerationArguments()
args.prompt = Prompt.parse(
    'A fantasy woman with pale lavender skin, glowing yellow eyes, pointed ears, '
    'and long dark hair, wearing dark ornate armor with gold accents. She performs '
    'the motion from the driving video. Cinematic lighting, ultrarealistic detail.')
args.images = character
args.wan_driving_video_frames = driving_frames
args.wan_animate_preprocess = True
args.wan_animate_preprocess_output = True
args.wan_animate_preprocess_output_dir = 'output'
args.wan_animate_preprocess_output_name = 'motion'
args.inference_steps = 20
args.guidance_scale = 1
args.video_fps = 30
args.width = 832
args.height = 480

result = wrapper(args)
result.write('wan-animate.mp4')
