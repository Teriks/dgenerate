import dgenerate.image_process
import dgenerate.messages

# Disable built in informational output for this example
dgenerate.messages.messages_to_null()
dgenerate.messages.errors_to_null()

# configure a dgenerate render loop
rl = dgenerate.RenderLoop()

rl.config.model_path = 'stabilityai/stable-diffusion-2'
rl.config.model_type = dgenerate.ModelType.TORCH
rl.config.inference_steps = [30]
rl.config.guidance_scales = [5, 8]
rl.config.image_seed_strengths = [0.3]
rl.config.prompts = [dgenerate.Prompt.parse('cat')]
rl.config.image_seeds = ['../../media/kitten.gif']
rl.config.animation_format = 'GIF'
rl.config.output_configs = True

# simple event handler for every possible event in the event stream

for event in rl.events():
    if isinstance(event, dgenerate.StartingGenerationStepEvent):
        print('Generation Step:', event.generation_step)
    if isinstance(event, dgenerate.StartingAnimationEvent):
        print('Starting Animation:', event.total_frames)
    if isinstance(event, dgenerate.AnimationETAEvent):
        print('Animation ETA:', event.eta)
    if isinstance(event, dgenerate.StartingAnimationFileEvent):
        print('Starting Writes To Animation File:', event.path)
    if isinstance(event, dgenerate.ImageGeneratedEvent):
        print('Image Generated (Suggested Filename):', event.suggested_filename)
        print('Image Generated (Suggested Directory):', event.suggested_directory)
    if isinstance(event, dgenerate.ImageFileSavedEvent):
        print('Image Saved:', event.path)
        print('Config Saved:', event.config_filename)
    if isinstance(event, dgenerate.AnimationFinishedEvent):
        print('Finished Animation:', event.starting_event.total_frames)
    if isinstance(event, dgenerate.AnimationFileFinishedEvent):
        print('Finished Animation File:', event.path)
        print('Animation Config Saved:', event.config_filename)

# The image-process sub command guts can be used in nearly an identical manner.

# configure a dgenerate --sub-command image-process render loop

rl = dgenerate.image_process.ImageProcessRenderLoop()

rl.config.input = ['../../media/kitten.gif']
rl.config.output = ['kitten/']
rl.config.align = 1
rl.config.no_aspect = True
rl.config.resize = (512, 512)

# simple event handler for every possible event in the event stream
# the events and information available is very similar to the
# dgenerate render loop
for event in rl.events():
    if isinstance(event, dgenerate.image_process.StartingGenerationStepEvent):
        print('Generation Step:', event.generation_step)
    if isinstance(event, dgenerate.image_process.StartingAnimationEvent):
        print('Starting Animation:', event.total_frames)
    if isinstance(event, dgenerate.image_process.AnimationETAEvent):
        print('Animation ETA:', event.eta)
    if isinstance(event, dgenerate.image_process.StartingAnimationFileEvent):
        print('Starting Writes To Animation File:', event.path)
    if isinstance(event, dgenerate.image_process.ImageGeneratedEvent):
        print('Image Generated (Suggested Filename):', event.suggested_filename)
        print('Image Generated (Suggested Directory):', event.suggested_directory)
    if isinstance(event, dgenerate.image_process.ImageFileSavedEvent):
        print('Image Saved:', event.path)
    if isinstance(event, dgenerate.image_process.AnimationFinishedEvent):
        print('Finished Animation:', event.starting_event.total_frames)
    if isinstance(event, dgenerate.image_process.AnimationFileFinishedEvent):
        print('Finished Animation File:', event.path)


# if you wish to work with any image offered by an event object
# in the event stream outside of the event handler you have written,
# you should copy it out with arg.image.copy(). management of PIL.Image
# lifetime is very aggressive and the image objects in events will be disposed
# of when no longer needed for the event, IE. have .close() called on them
# making them unusable
