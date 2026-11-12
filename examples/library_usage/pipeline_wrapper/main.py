import dgenerate

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = 'cuda'

# PipelineWrapper use cases in this folder:
#
# main.py                  one image
# video_main.py            one finished video clip
# ltx2_main.py             LTX-2.5, an opening clip and a closing clip
# wan_animate_main.py      Wan-Animate, pose and face derived inside the call
# wan_animate_2_main.py    Wan-Animate-2, driving clip resampled inside the call
#
# This file is the image case, using SDXL.

# It is useful for generating singular images
# and utilizes dgenerate's pipeline memory
# management and caching features

# this is designed in a way where it is friendly to
# browse arguments and documentation inside your IDE

# where every a _uri argument is mentioned, it corresponds
# directly to the URI syntax for said argument from the
# command line

wrapper = dgenerate.PipelineWrapper(
    model_path='stabilityai/stable-diffusion-xl-base-1.0',
    model_type=dgenerate.ModelType.SDXL,
    dtype=dgenerate.DataType.FLOAT16,
    variant='fp16',
    device=device
)

args = dgenerate.GenerationArguments()
args.prompt_weighter_uri = 'sd-embed'
args.inference_steps = 30
args.guidance_scale = 4
args.prompt = dgenerate.Prompt.parse('a man standing by a river, (photo:1.3); (painting)')


# Pipeline always gets moved back onto the desired device when called
# and previous pipelines are moved off the device auto-magically

# If the pipeline has never been called, it is loaded onto the CPU here
# first and into the CPU side pipeline cache

# The CPU side cache is garbage collected if there is not enough space for
# this pipeline to exist on the CPU before moving into VRAM

# This cache fencing happens automatically as modules move to and
# back from VRAM during memory management

result = wrapper(args)


result.image.save('sdxl-output.png')

