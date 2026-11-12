import os
import sys

import dgenerate.arguments
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    VideoGeneratedEvent, \
    Prompt, \
    ModelType, \
    DataType

# LTX-2.5 with an opening clip and a closing clip.
#
# The image seed path is the opening gif. last-frame= is the closing gif.
# An end clip needs a fixed length, so video_lengths is set.
#
# PipelineWrapper does not parse this syntax. The equivalent, which loads
# the frames itself, is examples/library_usage/pipeline_wrapper/ltx2_main.py.
# The config is examples/ltx2/video_conditioning/first-last-clip-config.dgen.

if not os.environ.get('HF_TOKEN'):
    print('Set HF_TOKEN environmental variable to run this example.')
    sys.exit(0)

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

config = RenderLoopConfig()

config.model_path = 'Lightricks/LTX-2.5-Diffusers'
config.model_type = ModelType.LTX2

config.dtype = DataType.BFLOAT16
config.guidance_scales = [1]
config.seeds = [1]
config.video_lengths = [4]
config.video_fps = [25]
config.animation_format = 'mp4'
config.output_size = (512, 512)
config.image_seeds = ['../../media/rickroll-roll.gif;last-frame=../../media/kitten.gif']

config.prompts = [Prompt.parse(
    'A singer in a dark blazer and striped shirt sways at a vintage microphone, '
    'then the shot cuts to a small kitten looking around. The camera does not move.')]

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
