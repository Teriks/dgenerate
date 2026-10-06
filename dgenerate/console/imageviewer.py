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
import platform

# pyopengltk is GLX-only. On Wayland, PyOpenGL may select EGL first; then
# ``from OpenGL import GLX`` dies with EGLPlatform has no attribute GLX.
# Force GLX (works via XWayland) unless the user already chose a platform.
if platform.system() == 'Linux' and 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'glx'

# Try to import OpenGL dependencies. pyopengltk only defines OpenGLFrame on
# Windows and Linux today; require that symbol so macOS falls through cleanly.
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

def _use_vulkan() -> bool:
    """Use the Vulkan preview when that extra is installed.

    Default on Windows and Linux. macOS stays on OpenGL (or the Tk canvas)
    unless ``DGENERATE_CONSOLE_UI_VULKAN=1``, because MoltenVK / the Vulkan
    loader are not present on a stock Mac. ``DGENERATE_CONSOLE_UI_VULKAN=0``
    forces OpenGL or the canvas on every platform.
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


# Vulkan when that extra is installed (and allowed), otherwise OpenGL,
# otherwise the Tk canvas.
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
