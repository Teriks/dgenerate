import os
import sys

import dgenerate
from dgenerate import DataType, ModelType, Prompt
from dgenerate.mediainput import load_rgb_frames

# LTX-2.5 through PipelineWrapper, with an opening clip and a closing clip.
#
# The wrapper does not read image-seed syntax. This example loads both gifs
# and assigns the frames. video_frames is the start of the clip.
# end_video_frames is the end. An end clip needs video_length, because
# LTX-2.5 otherwise picks the length from the prompt.
#
# A still uses images or end_images instead of the clip fields. One image
# in images is the first frame. One image in end_images is the last frame.
#
# The render-loop version, which parses the image seed for you, is
# examples/library_usage/ltx2/first_last_main.py.
# The config is examples/ltx2/video_conditioning/first-last-clip-config.dgen.

if not os.environ.get('HF_TOKEN'):
    print('Set HF_TOKEN environmental variable to run this example.')
    sys.exit(0)

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

wrapper = dgenerate.PipelineWrapper(
    model_path='Lightricks/LTX-2.5-Diffusers',
    model_type=ModelType.LTX2,
    dtype=DataType.BFLOAT16,
    device=device,
    model_sequential_offload=True
)

opening, _ = load_rgb_frames('../../media/rickroll-roll.gif')
closing, _ = load_rgb_frames('../../media/kitten.gif')

args = dgenerate.GenerationArguments()
args.prompt = Prompt.parse(
    'A singer in a dark blazer and striped shirt sways at a vintage microphone, '
    'then the shot cuts to a small kitten looking around. The camera does not move.')
args.video_frames = opening
args.end_video_frames = closing
args.guidance_scale = 1
args.seed = 1
args.video_length = 4
args.video_fps = 25
args.width = 512
args.height = 512

result = wrapper(args)
result.write('ltx-first-last-clip.mp4')
