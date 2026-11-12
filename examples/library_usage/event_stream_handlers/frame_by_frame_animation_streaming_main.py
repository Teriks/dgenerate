import io

import dgenerate.arguments
import dgenerate.messages
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    StartingAnimationEvent, \
    ImageGeneratedEvent, \
    AnimationFinishedEvent, \
    Prompt, \
    ModelType

# Frame-by-frame animation written to a binary stream.
#
# An image model walks an animated image seed one diffusion call per frame.
# StartingAnimationEvent is yielded before the first frame. Set event.file
# to a binary stream while handling it and that animation is encoded there.
# An mp4 receives each finished frame while later frames are still generating.
# GIF and WebP are written when the animation finishes. The container is
# closed, and a seekable stream is rewound, before AnimationFinishedEvent.
#
# One stream is one animation file. Two guidance scales yield two
# StartingAnimationEvent values. Each one is given a new stream. The
# next animation rewinds a seekable stream and writes over it, so the
# earlier video has to be a separate buffer. A batch that produces
# several images on one frame needs one stream per image, passed as a
# sequence.
#
# The seed is the hiker clip used by the Wan video-to-video configs.
# frame-end keeps the first eight frames. The clip is downloaded when
# the render loop opens the image seed.
#
# The first run writes each mp4 to disk and keeps a copy of each in
# its own stream. The second run disables writes. The streams are the
# only containers. event.file is read by the loop after this handler
# returns. AnimationFinishedEvent.starting_event is that same event,
# so its file is the finished video.

dgenerate.messages.messages_to_null()
dgenerate.messages.errors_to_null()

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

config = RenderLoopConfig()

config.model_path = 'sd2-community/stable-diffusion-2'
config.model_type = ModelType.SD
config.inference_steps = [30]
config.guidance_scales = [5, 8]
config.image_seed_strengths = [0.3]
config.prompts = [Prompt.parse('a hiker walking on a mountain trail')]
config.image_seeds = [
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4;frame-end=7']
config.animation_format = 'mp4'
config.device = device


def handle(event, videos):
    if isinstance(event, StartingAnimationEvent):
        event.file = io.BytesIO()
        videos.append(event.file)
        print('Starting Animation', len(videos), ':', event.total_frames, 'frames at', event.fps, 'fps')
    if isinstance(event, ImageGeneratedEvent) and event.is_animation_frame:
        print('Frame:', event.frame_index + 1, 'of', event.image_seed.total_frames)
    if isinstance(event, AnimationFinishedEvent):
        stream = event.starting_event.file
        print('Finished Animation', videos.index(stream) + 1, ':', len(stream.getvalue()), 'bytes')


def run(disable_writes):
    videos = []
    render_loop = RenderLoop(config=config, disable_writes=disable_writes)
    for event in render_loop.events():
        handle(event, videos)
    for index, stream in enumerate(videos, start=1):
        print('Kept Video', index, ':', len(stream.getvalue()), 'bytes')


print('Writes enabled')
run(disable_writes=False)

print('Writes disabled')
run(disable_writes=True)
