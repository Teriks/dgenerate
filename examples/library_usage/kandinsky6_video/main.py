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

# Kandinsky 6 Lite distilled. 10 steps, guidance 1, 5 seconds at 24 fps.
# Leave scheduler_uri unset so the checkpoint scheduler is kept.

config.model_path = 'kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers'
config.model_type = ModelType.KANDINSKY6_VIDEO

config.dtype = DataType.BFLOAT16
config.inference_steps = [10]
config.guidance_scales = [1]
config.video_lengths = [5]
config.video_fps = [24]
config.animation_format = 'mp4'
config.output_size = (864, 480)

config.prompts = [Prompt.parse('A red kite glides above a windswept coastal meadow at golden hour. Audio: wind, distant surf, the kite\'s fabric snapping.')]

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
