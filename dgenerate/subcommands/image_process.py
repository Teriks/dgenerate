# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import dgenerate.image_process as _image_process
import dgenerate.subcommands.subcommand as _subcommand


class ImageProcessSubCommand(_subcommand.SubCommand):
    """
    Allows for using the \\image_process config directive from the command line, any therefore
    and image preprocessor implemented by dgenerate or a plugin directly from the command line.

    Examples:

    dgenerate --sub-command image-process "my-photo.jpg" --output my-photo-openpose.jpg --processors openpose

    dgenerate --sub-command image-process "my-photo.jpg" --output my-photo-canny.jpg --processors canny

    See: dgenerate --sub-command image-process --help
    """

    NAMES = ['image-process']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self) -> int:
        render_loop = _image_process.ImageProcessRenderLoop()

        config_overrides = None
        if self.local_files_only:
            config_overrides = {'offline_mode': True}

        render_loop.image_processor_loader.load_plugin_modules(self.plugin_module_paths)
        return _image_process.invoke_image_process(
            self.args,
            render_loop=render_loop,
            config_overrides=config_overrides
        )
