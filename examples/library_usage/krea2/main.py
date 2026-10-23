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

config.model_path = 'krea/Krea-2-Raw'
config.model_type = ModelType.KREA2

config.dtype = DataType.BFLOAT16
config.transformer_uri = 'https://huggingface.co/realrebelai/KREA-2_GGUFs/blob/main/BASE/Krea-2-Base-Q4_K_M.gguf'
config.inference_steps = [30]
config.guidance_scales = [4]
config.output_size = (1024, 1024)

config.prompts = [Prompt.parse('A small red fox reading a map beside a mossy forest stream, soft morning light')]

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
