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

import ctypes
import os
import platform
import sys

# pyopengltk is GLX-only. On Wayland, PyOpenGL may select EGL first; then
# ``from OpenGL import GLX`` dies with EGLPlatform has no attribute GLX.
# Force GLX (works via XWayland) unless the user already chose a platform.
if platform.system() == 'Linux' and 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'glx'

# pyopengltk exports OpenGLFrame on Windows and Linux only. macOS uses Metal.
HAS_OPENGL = False
try:
    import pyopengltk
    import OpenGL.GL
    import OpenGL.arrays.vbo

    HAS_OPENGL = (
            hasattr(pyopengltk, 'OpenGLFrame')
            and os.environ.get('DGENERATE_CONSOLE_UI_OPENGL', '1') == '1'
    )
except (ImportError, AttributeError):
    pass


def _use_metal() -> bool:
    """macOS draws the preview with Metal. ``DGENERATE_CONSOLE_UI_METAL=0`` turns that off."""
    if platform.system() != 'Darwin':
        return False
    if os.environ.get('DGENERATE_CONSOLE_UI_METAL') == '0':
        return False
    try:
        metal = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Metal.framework/Metal')
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        metal.MTLCreateSystemDefaultDevice.argtypes = []
        if not metal.MTLCreateSystemDefaultDevice():
            print('Metal preview is not available: this Mac has no Metal device.', file=sys.stderr)
            return False
        return True
    except Exception as error:
        print(f'Metal preview is not available: {error}', file=sys.stderr)
        return False


def _use_vulkan() -> bool:
    """Use the Vulkan preview when that extra is installed.

    Default on Windows and Linux. macOS uses Metal first.
    ``DGENERATE_CONSOLE_UI_METAL=0`` and ``DGENERATE_CONSOLE_UI_VULKAN=1``
    select Vulkan there. ``DGENERATE_CONSOLE_UI_VULKAN=0`` forces OpenGL
    or the canvas on Windows and Linux.
    """
    if os.environ.get('DGENERATE_CONSOLE_UI_VULKAN') == '0':
        return False
    if (platform.system() == 'Darwin'
            and os.environ.get('DGENERATE_CONSOLE_UI_VULKAN') != '1'):
        return False
    try:
        import vulkan  # noqa: F401
    except ImportError:
        return False
    return True


# Metal on macOS, otherwise Vulkan when that extra is installed (and allowed),
# otherwise OpenGL, otherwise the Tk canvas.
HAS_METAL = False
HAS_VULKAN = False
if _use_metal():
    from dgenerate.console.imageviewer_metal import ImageViewerMetal as ImageViewer

    HAS_METAL = True
elif _use_vulkan():
    from dgenerate.console.imageviewer_vk import ImageViewerVulkan as ImageViewer

    HAS_VULKAN = True
elif HAS_OPENGL:
    from dgenerate.console.imageviewer_gl import ImageViewerGL as ImageViewer
else:
    from dgenerate.console.imageviewer_canvas import ImageViewerCanvas as ImageViewer

# Video and the Load Media command need one of the hardware previews.
PLAYS_MEDIA = HAS_METAL or HAS_VULKAN or HAS_OPENGL

__all__ = ['ImageViewer', 'HAS_OPENGL', 'HAS_VULKAN', 'HAS_METAL', 'PLAYS_MEDIA']
