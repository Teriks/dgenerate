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

config.model_path = 'Kwai-Kolors/Kolors-diffusers'
config.model_type = ModelType.KOLORS

config.variant = 'fp16'
config.dtype = DataType.FLOAT16
config.sdxl_refiner_uri = 'stabilityai/stable-diffusion-xl-refiner-1.0'
config.sdxl_high_noise_fractions = [0.8]
config.inference_steps = [40]
config.output_size = (1024, 1024)

config.prompts = [Prompt.parse('Photo of a horse standing near the open door of a red barn, high resolution; artwork')]

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
