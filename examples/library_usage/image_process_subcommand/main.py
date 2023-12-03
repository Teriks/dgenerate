import dgenerate.image_process

# the functionality of --sub-command image-process and the \image_process config
# directive is reusable in a similar fashion to the dgenerate render loop

config = dgenerate.image_process.ImageProcessRenderLoopConfig()

config.input = ['../../media/earth.jpg']
config.output = ['earth-upscaled.png']

# any dgenerate implemented image processor can be specified by its URI here
config.processors = [
    'upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth']

render_loop = dgenerate.image_process.ImageProcessRenderLoop(config=config)

render_loop.run()

# we can load a custom processor plugin just for fun, this time from disk

# Load the plugin example folder from the current directory

render_loop.image_processor_loader.load_plugin_modules(['plugin_example'])

# custom processor name

config.processors = ['foo']

config.output = ['foo-processed.png']

# do not write anything to disk for us
render_loop.disable_writes = True

# run again, this time observe user handleable events

for event in render_loop.events():
    if isinstance(event, dgenerate.image_process.ImageGeneratedEvent):
        print('Filename:', event.suggested_filename)
        event.image.save(event.suggested_filename)

        # if you wish to work with any image offered by an event object
        # in the event stream outside of the event handler you have written,
        # you should copy it out with arg.image.copy(). management of PIL.Image
        # lifetime is very aggressive and the image objects in events will be disposed
        # of when no longer needed for the event, IE. have .close() called on them
        # making them unusable

