import dgenerate.imageprocess

# the functionality of --sub-command image-process and the \image_process config
# directive is reusable in a similar fashion to the dgenerate render loop

config = dgenerate.imageprocess.ImageProcessConfig()

config.files = ['../../media/earth.jpg']
config.output = ['earth-upscaled.png']

# any dgenerate implemented image processor can be specified by its URI here
config.processors = [
    'upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth']

render_loop = dgenerate.imageprocess.ImageProcessRenderLoop(config=config)

render_loop.run()

# we can load a custom processor plugin just for fun, this time from disk

# Load the plugin example folder from the current directory

render_loop.image_processor_loader.load_plugin_modules(['plugin_example'])

# custom processor name

config.processors = ['foo']

config.output = ['foo-processed.png']

# run again

render_loop.run()
