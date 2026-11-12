import os

import dgenerate.arguments
import dgenerate.messages
from dgenerate import \
    RenderLoop, \
    RenderLoopConfig, \
    StartingGenerationStepEvent, \
    StartingAnimationEvent, \
    StartingAnimationFileEvent, \
    VideoGeneratedEvent, \
    AnimationFinishedEvent, \
    AnimationFileFinishedEvent, \
    Prompt, \
    ModelType, \
    DataType

# Finished video clip.
#
# A video pipeline generates the whole clip in one step. When the length is
# requested with video_lengths, StartingAnimationEvent is yielded before the
# pipeline runs. The remaining events are VideoGeneratedEvent and
# AnimationFinishedEvent. With writes enabled, StartingAnimationFileEvent
# comes before the clip and AnimationFileFinishedEvent comes after
# AnimationFinishedEvent.
# VideoGeneratedEvent holds the finished frames, frame rate, and soundtrack.
# write() muxes that clip to a path or a binary stream. Call it while
# handling the event; the frames are closed when the loop moves on.
#
# AnimationETAEvent and ImageGeneratedEvent are produced while an image
# model diffuses an animated image seed one frame at a time. That usage
# is frame_by_frame_animation_main.py.
#
# The first run lets the loop write the mp4. The second run keeps the clip
# in memory and muxes it from VideoGeneratedEvent.

# Disable built in informational output for this example
dgenerate.messages.messages_to_null()
dgenerate.messages.errors_to_null()

device, _ = dgenerate.arguments.parse_device()

if device is None:
    device = dgenerate.default_device()

config = RenderLoopConfig()

config.model_path = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
config.model_type = ModelType.WAN

config.dtype = DataType.BFLOAT16
config.inference_steps = [30]
config.guidance_scales = [5]
config.video_lengths = [5]
config.video_fps = [16]
config.animation_format = 'mp4'
config.output_size = (832, 480)
config.output_configs = True

neg = (
    'Bright tones, overexposed, static, blurred details, subtitles, style, works, '
    'paintings, images, static, overall gray, worst quality, low quality, '
    'JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, '
    'poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, '
    'still picture, messy background, three legs, many people in the background, '
    'walking backwards')

config.prompts = [Prompt.parse('A cat and a dog baking a cake together in a kitchen.;' + neg)]

config.device = device
config.model_sequential_offload = True


def handle(event, mux_clip):
    if isinstance(event, StartingGenerationStepEvent):
        print('Generation Step:', event.generation_step)
    if isinstance(event, StartingAnimationEvent):
        print('Starting Animation:', event.total_frames)
        print('Animation FPS:', event.fps)
    if isinstance(event, StartingAnimationFileEvent):
        print('Starting Writes To Animation File:', event.path)
    if isinstance(event, VideoGeneratedEvent):
        print('Video Generated (Suggested Filename):', event.suggested_filename)
        print('Video FPS:', event.fps)
        print('Frames:', len(event.frames))
        print('Seed:', event.diffusion_args.seed)
        print('Prompt:', event.diffusion_args.prompt)
        # The loop writes the container when writes are enabled.
        # write() is for the in-memory clip, when writes are disabled.
        # dgenerate does not create the output directory when writes are disabled.
        if mux_clip:
            path = os.path.join(event.suggested_directory, event.suggested_filename)
            os.makedirs(event.suggested_directory, exist_ok=True)
            event.write(path)
    if isinstance(event, AnimationFinishedEvent):
        print('Finished Animation:', event.starting_event.total_frames)
    if isinstance(event, AnimationFileFinishedEvent):
        print('Finished Animation File:', event.path)
        print('Animation Config Saved:', event.config_filename)


print('Writes enabled')

render_loop = RenderLoop(config=config, disable_writes=False)

for event in render_loop.events():
    handle(event, mux_clip=False)

print('Writes disabled')

render_loop = RenderLoop(config=config, disable_writes=True)

for event in render_loop.events():
    handle(event, mux_clip=True)

# Keep a frame past this handler with frame.copy(). The render loop closes
# every frame on the event when it moves on.
