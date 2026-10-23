import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    AnimationFileFinishedEvent, \
    Prompt, \
    ModelType, \
    DataType

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

config = RenderLoopConfig()

config.model_path = 'Motif-Technologies/Motif-Video-2B'
config.model_type = ModelType.MOTIF_VIDEO

config.dtype = DataType.BFLOAT16
config.transformer_uri = 'https://huggingface.co/Motif-Technologies/Motif-Video-2B-GGUF/blob/main/motifv-2b-dev-Q4_K_M.gguf'
config.inference_steps = [30]
config.guidance_scales = [5]
config.video_lengths = [4]
config.video_fps = [24]
config.animation_format = 'mp4'
config.output_configs = True  # write a .dgen config next to each clip
config.output_size = (832, 480)

config.prompts = [Prompt.parse('A red kite glides above a windswept coastal meadow at golden hour; smooth cinematic camera movement')]

config.device = device

config.model_sequential_offload = True


# Video models write a finished clip to config.output_path, they do
# not emit ImageGeneratedEvent, so writes must stay enabled

render_loop = RenderLoop(config=config)

for event in render_loop.events():
    if isinstance(event, AnimationFileFinishedEvent):
        print('Video:', event.path)
        print('Reproduce With Config:', event.config_filename)
