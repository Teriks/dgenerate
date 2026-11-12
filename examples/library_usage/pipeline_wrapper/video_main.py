import dgenerate
from dgenerate import DataType, ModelType, Prompt

# One finished clip.
#
# A video model returns every frame from a single PipelineWrapper call.
# GenerationResult.images holds the frames, fps is the clip rate, and
# write() muxes the clip to a path or a binary stream.
#
# The render loop's event stream for the same kind of call is
# examples/library_usage/event_stream_handlers/video_pipeline_main.py.

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

wrapper = dgenerate.PipelineWrapper(
    model_path='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
    model_type=ModelType.WAN,
    dtype=DataType.BFLOAT16,
    device=device,
    model_sequential_offload=True
)

neg = (
    'Bright tones, overexposed, static, blurred details, subtitles, style, works, '
    'paintings, images, static, overall gray, worst quality, low quality, '
    'JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, '
    'poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, '
    'still picture, messy background, three legs, many people in the background, '
    'walking backwards')

args = dgenerate.GenerationArguments()
args.prompt = Prompt.parse('A cat and a dog baking a cake together in a kitchen.;' + neg)
args.inference_steps = 30
args.guidance_scale = 5
args.video_length = 5
args.video_fps = 16
args.width = 832
args.height = 480

result = wrapper(args)
result.write('wan-cat-dog.mp4')
