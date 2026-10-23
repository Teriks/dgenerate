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

config.model_path = 'Tongyi-MAI/Z-Image-Turbo'
config.model_type = ModelType.Z_IMAGE

config.dtype = DataType.BFLOAT16
config.scheduler_uri = 'FlowMatchEulerDiscreteScheduler'
config.inference_steps = [9]
config.guidance_scales = [0]
config.output_size = (1024, 1024)

config.prompts = [Prompt.parse('Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern.')]

config.device = device

config.model_sequential_offload = True


render_loop = RenderLoop(
    config=config,
    disable_writes=True  # disables all writes to disk
)

for event in render_loop.events():
    if isinstance(event, ImageGeneratedEvent):
        print('Filename:', event.suggested_filename)
        print('Seed:', event.diffusion_args.seed)
        print('Prompt:', event.diffusion_args.prompt)
        print('Reproduce With Command:', event.command_string)
        print(f'Reproduce With Config:\n{event.config_string}')
        event.image.save(event.suggested_filename)
