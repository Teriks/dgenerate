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

"""
macOS OpenGL frame for the console preview.

pyopengltk's ``darwin.py`` is a stub, and the package never exports
``OpenGLFrame`` on Darwin. macOS still ships OpenGL 4.1. This frame creates
that context on the Tk view so the same preview used on Windows and Linux
can run here.
"""

import ctypes
import platform

# NSOpenGLPixelFormatAttribute values from NSOpenGL.h.
NSOpenGLPFADoubleBuffer = 5
NSOpenGLPFAColorSize = 8
NSOpenGLPFAAlphaSize = 11
NSOpenGLPFAAccelerated = 73
NSOpenGLPFAOpenGLProfile = 99
NSOpenGLProfileVersion4_1Core = 0x4100


def glsl_source(source: str, darwin: bool | None = None) -> str:
    """
    Apple OpenGL 4.1 compiles GLSL 4.10, not 3.30.

    ``darwin`` overrides the platform check so tests can run elsewhere.
    """
    if darwin is None:
        darwin = platform.system() == 'Darwin'
    if not darwin:
        return source
    return source.replace('#version 330 core', '#version 410 core', 1)


def pixel_format_attributes() -> list[int]:
    """Core 4.1, double buffered, hardware renderer. A trailing 0 ends the list."""
    return [
        NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion4_1Core,
        NSOpenGLPFADoubleBuffer,
        NSOpenGLPFAAccelerated,
        NSOpenGLPFAColorSize, 24,
        NSOpenGLPFAAlphaSize, 8,
        0,
    ]


class _MacDrawable(ctypes.Structure):
    """Leading fields of Tk's ``MacDrawable``. ``winfo_id()`` is this pointer."""
    _fields_ = [
        ('winPtr', ctypes.c_void_p),
        ('view', ctypes.c_void_p),
    ]


def aqua_nsview(drawable_id: int) -> int:
    """Return the ``NSView *`` stored in a Tk Aqua ``winfo_id()`` value."""
    address = int(drawable_id or 0)
    if address == 0:
        raise RuntimeError('The preview window is not on screen yet.')
    drawable = _MacDrawable.from_address(address)
    view = int(drawable.view or 0)
    if view == 0:
        raise RuntimeError('Tk has no NSView for this pane yet.')
    return view


def _objc():
    lib = getattr(_objc, 'lib', None)
    if lib is None:
        lib = ctypes.cdll.LoadLibrary('/usr/lib/libobjc.A.dylib')
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]
        lib.objc_getClass.restype = ctypes.c_void_p
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        _objc.lib = lib
    return lib


def _objc_msg(obj, selector, *args, restype=ctypes.c_void_p, argtypes=()):
    objc = _objc()
    send = objc.objc_msgSend
    send.restype = restype
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    return send(obj, objc.sel_registerName(selector), *args)


def _objc_class(name: bytes):
    cls = _objc().objc_getClass(name)
    if not cls:
        raise RuntimeError(f'{name.decode()} is not available.')
    return cls


def _load_appkit():
    if getattr(_load_appkit, 'done', False):
        return
    ctypes.cdll.LoadLibrary('/System/Library/Frameworks/AppKit.framework/AppKit')
    ctypes.cdll.LoadLibrary('/System/Library/Frameworks/OpenGL.framework/OpenGL')
    _load_appkit.done = True


try:
    from pyopengltk.base import BaseOpenGLFrame as _BaseOpenGLFrame
except ImportError:
    _BaseOpenGLFrame = None

OpenGLFrame = None

if _BaseOpenGLFrame is not None:
    class OpenGLFrame(_BaseOpenGLFrame):
        """pyopengltk frame that binds an OpenGL 4.1 context to the Tk NSView."""

        def __init__(self, *args, **kw):
            self._context = None
            self._pixel_format = None
            self._attrs = None
            self._view = 0
            self._gl_size = (0, 0)
            super().__init__(*args, **kw)
            self.bind('<Destroy>', self._release_gl, add='+')

        def tkCreateContext(self):
            _load_appkit()
            values = pixel_format_attributes()
            self._attrs = (ctypes.c_uint32 * len(values))(*values)
            pixel_format = _objc_msg(
                _objc_msg(_objc_class(b'NSOpenGLPixelFormat'), b'alloc'),
                b'initWithAttributes:',
                ctypes.cast(self._attrs, ctypes.c_void_p),
                argtypes=(ctypes.c_void_p,))
            if not pixel_format:
                raise RuntimeError('macOS could not create an OpenGL 4.1 pixel format.')
            self._pixel_format = pixel_format
            context = _objc_msg(
                _objc_msg(_objc_class(b'NSOpenGLContext'), b'alloc'),
                b'initWithFormat:shareContext:',
                pixel_format, None,
                argtypes=(ctypes.c_void_p, ctypes.c_void_p))
            if not context:
                self._release_gl()
                raise RuntimeError('macOS could not create an OpenGL 4.1 context.')
            self._context = context
            self._attach()

        def tkMakeCurrent(self):
            if self._context is not None and self.winfo_ismapped():
                self._attach()

        def tkSwapBuffers(self):
            if self._context is not None and self.winfo_ismapped():
                _objc_msg(self._context, b'flushBuffer', restype=None)

        def _attach(self):
            view = aqua_nsview(int(getattr(self, '_wid', 0) or 0))
            if view != self._view:
                try:
                    _objc_msg(
                        view, b'setWantsBestResolutionOpenGLSurface:', False,
                        restype=None, argtypes=(ctypes.c_bool,))
                except Exception:
                    pass
                _objc_msg(
                    self._context, b'setView:', view,
                    restype=None, argtypes=(ctypes.c_void_p,))
                self._view = view
                self._gl_size = (0, 0)
            # The framebuffer stays in Tk points. The viewer sizes the viewport
            # from winfo_width/height, so a Retina backing store would draw into
            # the corner of the pane.
            size = (self.winfo_width(), self.winfo_height())
            if size != self._gl_size:
                _objc_msg(self._context, b'update', restype=None)
                self._gl_size = size
            _objc_msg(self._context, b'makeCurrentContext', restype=None)

        def _release_gl(self, _event=None):
            context = self._context
            pixel_format = self._pixel_format
            self._context = None
            self._pixel_format = None
            self._view = 0
            if context is None and pixel_format is None:
                return
            try:
                _objc_msg(_objc_class(b'NSOpenGLContext'), b'clearCurrentContext', restype=None)
                if context is not None:
                    _objc_msg(context, b'clearDrawable', restype=None)
                    _objc_msg(context, b'release', restype=None)
                if pixel_format is not None:
                    _objc_msg(pixel_format, b'release', restype=None)
            except Exception:
                return


def install_frame(module, system: str | None = None) -> bool:
    """
    Export ``OpenGLFrame`` on a pyopengltk module that does not have one.

    Windows and Linux builds already export it. macOS does not.
    """
    if hasattr(module, 'OpenGLFrame'):
        return True
    if (system or platform.system()) != 'Darwin':
        return False
    if OpenGLFrame is None:
        return False
    module.OpenGLFrame = OpenGLFrame
    return True
