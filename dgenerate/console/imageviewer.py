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

import os

# Try to import OpenGL dependencies
HAS_OPENGL = False
try:
    import pyopengltk
    import OpenGL.GL
    import OpenGL.arrays.vbo
    HAS_OPENGL = os.environ.get('DGENERATE_CONSOLE_UI_OPENGL', '1') == '1'
except ImportError:
    pass

def _use_vulkan() -> bool:
    """Use the Vulkan preview when that extra is installed.

    This is the default on Windows, Linux, and macOS.
    ``DGENERATE_CONSOLE_UI_VULKAN=0`` keeps the OpenGL viewer, or the Tk canvas
    when OpenGL is not installed.
    """
    if os.environ.get('DGENERATE_CONSOLE_UI_VULKAN') == '0':
        return False
    try:
        import vulkan  # noqa: F401
    except ImportError:
        return False
    return True


# Vulkan when that extra is installed, otherwise OpenGL, otherwise the Tk canvas.
if _use_vulkan():
    from dgenerate.console.imageviewer_vk import ImageViewerVulkan as ImageViewer
    HAS_VULKAN = True
elif HAS_OPENGL:
    from dgenerate.console.imageviewer_gl import ImageViewerGL as ImageViewer
    HAS_VULKAN = False
else:
    from dgenerate.console.imageviewer_canvas import ImageViewerCanvas as ImageViewer
    HAS_VULKAN = False

__all__ = ['ImageViewer', 'HAS_OPENGL', 'HAS_VULKAN']
