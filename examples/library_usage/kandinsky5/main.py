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

config.model_path = 'kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers'
config.model_type = ModelType.KANDINSKY5_IMAGE

config.dtype = DataType.BFLOAT16
config.inference_steps = [50]
config.guidance_scales = [3.5]
config.output_size = (1024, 1024)

config.prompts = [Prompt.parse('A tiny bookshop inside a hollow tree, warm lamps glowing through the windows')]

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
