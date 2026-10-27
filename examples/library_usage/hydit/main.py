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

config.model_path = 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers'
config.model_type = ModelType.HYDIT

config.dtype = DataType.FLOAT16
config.inference_steps = [50]
config.guidance_scales = [5]
config.output_size = (1024, 1024)

config.prompts = [Prompt.parse(
    'A porcelain vase with blue floral patterns on a wooden table, soft studio lighting')]

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
