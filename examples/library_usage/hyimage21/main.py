import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    ImageGeneratedEvent, \
    Prompt, \
    ModelType, \
    DataType

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

config = RenderLoopConfig()

config.model_path = 'hunyuanvideo-community/HunyuanImage-2.1-Diffusers'
config.model_type = ModelType.HYIMAGE_21

config.dtype = DataType.BFLOAT16
config.inference_steps = [50]
config.guidance_scales = [3.5]
config.output_size = (2048, 2048)

config.prompts = [Prompt.parse(
    'A glass greenhouse in a rainstorm, reflections on the panes, cinematic photography')]

config.device = device
config.model_sequential_offload = True

render_loop = RenderLoop(
    config=config,
    disable_writes=True
)

for event in render_loop.events():
    if isinstance(event, ImageGeneratedEvent):
        print('Filename:', event.suggested_filename)
        print('Seed:', event.diffusion_args.seed)
        print('Reproduce With Command:', event.command_string)
        event.image.save(event.suggested_filename)
