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

"""Metal preview pane for the macOS console.

pyopengltk does not provide an OpenGL view on Darwin. Tk gives an ``NSView``
to the toplevel window only, so a child pane cannot host ``CAMetalLayer``.
The preview adds its own overlay view on that toplevel and draws into the
layer there. The overlay does not accept hits, so the pane still receives
the pointer.
"""

import ctypes
import platform
import sys
import time
import tkinter as tk

import numpy as np
import PIL.Image

import dgenerate.console.animationplayback as _animationplayback
import dgenerate.console.helpdialog as _helpdialog
import dgenerate.console.mousewheelbind as _mousewheelbind

# The pane builds Vulkan-style NDC, where y grows downward. Metal's clip
# y grows upward, so the vertex shader flips it.
METAL_SHADER = """
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float2 uv;
};

vertex Varyings vertex_main(uint vid [[vertex_id]],
                            const device float4 *vertices [[buffer(0)]]) {
    float4 v = vertices[vid];
    Varyings out;
    out.position = float4(v.x, -v.y, 0.0, 1.0);
    out.uv = v.zw;
    return out;
}

fragment float4 fragment_texture(Varyings in [[stage_in]],
                                 texture2d<float> tex [[texture(0)]]) {
    constexpr sampler samp(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    return tex.sample(samp, in.uv);
}

fragment float4 fragment_color(Varyings in [[stage_in]],
                               constant float4 &color [[buffer(0)]]) {
    return color;
}
"""

_MTL_PIXEL_BGRA8 = 80
_MTL_PIXEL_RGBA8 = 70
_MTL_LOAD_CLEAR = 2
_MTL_STORE = 1
_MTL_BLEND_SRC_ALPHA = 4
_MTL_BLEND_ONE_MINUS_SRC_ALPHA = 5
_MTL_TRIANGLE = 3
_MTL_STORAGE_SHARED = 0


class _MacDrawable(ctypes.Structure):
    """Leading fields of Tk's ``MacDrawable``. ``winfo_id()`` is this pointer."""
    _fields_ = [
        ('winPtr', ctypes.c_void_p),
        ('view', ctypes.c_void_p),
    ]


class _CGSize(ctypes.Structure):
    _fields_ = [
        ('width', ctypes.c_double),
        ('height', ctypes.c_double),
    ]


class _CGPoint(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_double),
        ('y', ctypes.c_double),
    ]


class _CGRect(ctypes.Structure):
    _fields_ = [
        ('origin', _CGPoint),
        ('size', _CGSize),
    ]


class _NSRange(ctypes.Structure):
    _fields_ = [
        ('location', ctypes.c_uint64),
        ('length', ctypes.c_uint64),
    ]


class _MTLOrigin(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_uint64),
        ('y', ctypes.c_uint64),
        ('z', ctypes.c_uint64),
    ]


class _MTLSize(ctypes.Structure):
    _fields_ = [
        ('width', ctypes.c_uint64),
        ('height', ctypes.c_uint64),
        ('depth', ctypes.c_uint64),
    ]


class _MTLRegion(ctypes.Structure):
    _fields_ = [
        ('origin', _MTLOrigin),
        ('size', _MTLSize),
    ]


def _objc():
    lib = getattr(_objc, 'lib', None)
    if lib is None:
        lib = ctypes.cdll.LoadLibrary('/usr/lib/libobjc.A.dylib')
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]
        lib.objc_getClass.restype = ctypes.c_void_p
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        lib.objc_allocateClassPair.restype = ctypes.c_void_p
        lib.objc_allocateClassPair.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.objc_registerClassPair.argtypes = [ctypes.c_void_p]
        lib.class_addMethod.restype = ctypes.c_bool
        lib.class_addMethod.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        _objc.lib = lib
    return lib


def _msg(obj, selector, *args, restype=ctypes.c_void_p, argtypes=()):
    objc = _objc()
    send = objc.objc_msgSend
    send.restype = restype
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    return send(obj, objc.sel_registerName(selector), *args)


def _class(name: bytes):
    found = _objc().objc_getClass(name)
    if not found:
        raise RuntimeError(f'{name.decode()} is not available.')
    return found


def _frameworks():
    if getattr(_frameworks, 'metal', None) is not None:
        return _frameworks.metal
    ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Foundation.framework/Foundation')
    ctypes.cdll.LoadLibrary('/System/Library/Frameworks/AppKit.framework/AppKit')
    ctypes.cdll.LoadLibrary('/System/Library/Frameworks/QuartzCore.framework/QuartzCore')
    metal = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Metal.framework/Metal')
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    metal.MTLCreateSystemDefaultDevice.argtypes = []
    _frameworks.metal = metal
    return metal


def _nsstring(text: str):
    encoded = text.encode('utf-8')
    return _msg(
        _msg(_class(b'NSString'), b'alloc'),
        b'initWithUTF8String:', encoded,
        argtypes=(ctypes.c_char_p,))


def _error_text(pointer) -> str:
    if not pointer:
        return 'Metal did not report a reason.'
    description = _msg(pointer, b'localizedDescription')
    raw = _msg(description, b'UTF8String', restype=ctypes.c_char_p)
    if not raw:
        return 'Metal did not report a reason.'
    return raw.decode('utf-8', errors='replace')


def _content_view(widget) -> int:
    """``NSView`` of the toplevel. Child windows store a null view."""
    address = int(widget.winfo_toplevel().winfo_id() or 0)
    if address == 0:
        raise RuntimeError('The preview window is not on screen yet.')
    view = int(_MacDrawable.from_address(address).view or 0)
    if view == 0:
        raise RuntimeError('Tk has no NSView for this window yet.')
    return view


_OVERLAY_HIT = None


def _overlay_view_class():
    """An ``NSView`` that ignores hits, so Tk keeps the pane's pointer events."""
    global _OVERLAY_HIT
    objc = _objc()
    name = b'DGenerateMetalView'
    found = objc.objc_getClass(name)
    if found:
        return found
    pair = objc.objc_allocateClassPair(_class(b'NSView'), name, 0)
    if not pair:
        raise RuntimeError('Could not create the Metal overlay view.')

    def _ignore_hit(_self, _cmd, _point):
        return None

    prototype = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, _CGPoint)
    _OVERLAY_HIT = prototype(_ignore_hit)
    added = objc.class_addMethod(
        pair, objc.sel_registerName(b'hitTest:'),
        ctypes.cast(_OVERLAY_HIT, ctypes.c_void_p), b'@@:{CGPoint=dd}')
    if not added:
        raise RuntimeError('Could not install Metal overlay hit testing.')
    objc.objc_registerClassPair(pair)
    return pair


def _pool():
    return _msg(_msg(_class(b'NSAutoreleasePool'), b'alloc'), b'init')


def _pack_vertices(picture, overlays):
    chunks = []
    draws = []
    cursor = 0
    if picture:
        array = np.asarray(picture, dtype=np.float32)
        chunks.append(array)
        draws.append(('texture', cursor, len(array), None))
        cursor += len(array)
    for corners, color in overlays:
        array = np.asarray(corners, dtype=np.float32)
        chunks.append(array)
        draws.append(('color', cursor, len(array), color))
        cursor += len(array)
    if not chunks:
        return [], b''
    return draws, np.concatenate(chunks).tobytes()


class _MetalDevice:
    """A textured quad and colored triangles on a ``CAMetalLayer``."""

    def __init__(self, widget, drawable_id: int, width: int, height: int, scale: float):
        self._widget = widget
        self._drawable_id = int(drawable_id)
        self._scale = max(1.0, float(scale or 1.0))
        self._host = None
        self._points = (0, 0)
        self._layer = None
        self._device = None
        self._queue = None
        self._library = None
        self._texture_pipeline = None
        self._color_pipeline = None
        self._texture = None
        self._texture_size = None
        self._source_image = None
        self._rgba_scratch = None
        self._vbo = None
        self._vbo_cap = 0
        self._owned = []
        self._defer_present = False
        self._placed = None
        self._held_drawable = None
        self._held_command = None
        try:
            _frameworks()
            device = _frameworks().MTLCreateSystemDefaultDevice()
            if not device:
                raise RuntimeError('This Mac has no Metal device.')
            self._device = _msg(device, b'retain')
            self._queue = _msg(self._device, b'newCommandQueue')
            if not self._queue:
                raise RuntimeError('Metal could not create a command queue.')
            self._library = self._library_from_source()
            self._texture_pipeline = self._pipeline(b'fragment_texture')
            self._color_pipeline = self._pipeline(b'fragment_color')
            self._layer = self._make_layer()
            self._place_overlay(width, height)
        except Exception:
            self.destroy()
            raise

    def _own(self, obj):
        if obj:
            self._owned.append(obj)
        return obj

    def _library_from_source(self):
        source = self._own(_nsstring(METAL_SHADER))
        error = ctypes.c_void_p()
        library = _msg(
            self._device, b'newLibraryWithSource:options:error:',
            source, None, ctypes.byref(error),
            argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))
        if not library:
            raise RuntimeError(f'Metal shader failed to compile: {_error_text(error.value)}')
        return library

    def _function(self, name: bytes):
        label = self._own(_nsstring(name.decode()))
        function = _msg(self._library, b'newFunctionWithName:', label, argtypes=(ctypes.c_void_p,))
        if not function:
            raise RuntimeError(f'Metal shader is missing {name.decode()}.')
        return function

    def _pipeline(self, fragment_name: bytes):
        desc = _msg(_msg(_class(b'MTLRenderPipelineDescriptor'), b'alloc'), b'init')
        _msg(desc, b'setVertexFunction:', self._function(b'vertex_main'),
             argtypes=(ctypes.c_void_p,), restype=None)
        _msg(desc, b'setFragmentFunction:', self._function(fragment_name),
             argtypes=(ctypes.c_void_p,), restype=None)
        attachments = _msg(desc, b'colorAttachments')
        attachment = _msg(attachments, b'objectAtIndexedSubscript:', 0, argtypes=(ctypes.c_uint64,))
        _msg(attachment, b'setPixelFormat:', _MTL_PIXEL_BGRA8,
             argtypes=(ctypes.c_uint64,), restype=None)
        _msg(attachment, b'setBlendingEnabled:', True,
             argtypes=(ctypes.c_bool,), restype=None)
        for selector in (
                b'setSourceRGBBlendFactor:', b'setSourceAlphaBlendFactor:'):
            _msg(attachment, selector, _MTL_BLEND_SRC_ALPHA,
                 argtypes=(ctypes.c_uint64,), restype=None)
        for selector in (
                b'setDestinationRGBBlendFactor:', b'setDestinationAlphaBlendFactor:'):
            _msg(attachment, selector, _MTL_BLEND_ONE_MINUS_SRC_ALPHA,
                 argtypes=(ctypes.c_uint64,), restype=None)
        error = ctypes.c_void_p()
        pipeline = _msg(
            self._device, b'newRenderPipelineStateWithDescriptor:error:',
            desc, ctypes.byref(error),
            argtypes=(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))
        _msg(desc, b'release', restype=None)
        if not pipeline:
            raise RuntimeError(f'Metal pipeline failed: {_error_text(error.value)}')
        return pipeline

    def _make_layer(self):
        layer = _msg(_class(b'CAMetalLayer'), b'layer')
        if not layer:
            raise RuntimeError('CAMetalLayer is not available.')
        layer = _msg(layer, b'retain')
        _msg(layer, b'setDevice:', self._device, argtypes=(ctypes.c_void_p,), restype=None)
        _msg(layer, b'setPixelFormat:', _MTL_PIXEL_BGRA8,
             argtypes=(ctypes.c_uint64,), restype=None)
        _msg(layer, b'setFramebufferOnly:', True, argtypes=(ctypes.c_bool,), restype=None)
        _msg(layer, b'setOpaque:', True, argtypes=(ctypes.c_bool,), restype=None)
        # A transaction present has to be waited out on this thread, and that
        # wait holds the console until Metal gives up (about a second per
        # drawable, three of them when a new picture is shown). The overlay
        # view is in the window, so a normal present composites on its own.
        _msg(layer, b'setPresentsWithTransaction:', False, argtypes=(ctypes.c_bool,), restype=None)
        _msg(layer, b'setContentsScale:', self._scale,
             argtypes=(ctypes.c_double,), restype=None)
        self._noted_missing_drawable = False
        self._drawable_pixels = None
        return layer

    def _place_overlay(self, width, height):
        """Keep an overlay ``NSView`` aligned with this pane.

        The view's layer tracks ``setFrame:``. ``CAMetalLayer`` has no
        ``setFrameSize:``; sending that aborts the process.
        """
        content = _content_view(self._widget)
        bounds = _msg(content, b'bounds', restype=_CGRect)
        top = self._widget.winfo_toplevel()
        origin_x = float(self._widget.winfo_rootx() - top.winfo_rootx())
        origin_y = float(self._widget.winfo_rooty() - top.winfo_rooty())
        points = (max(1, int(width)), max(1, int(height)))
        self._points = points
        rect = _CGRect(
            _CGPoint(origin_x, bounds.size.height - origin_y - points[1]),
            _CGSize(float(points[0]), float(points[1])))
        window = _msg(content, b'window')
        scale = float(_msg(window, b'backingScaleFactor', restype=ctypes.c_double) or 0.0)
        self._scale = scale if scale >= 1.0 else self._scale
        placed = (round(rect.origin.x, 1), round(rect.origin.y, 1), points[0], points[1], round(self._scale, 2))
        inserted = False
        if not self._host:
            self._host = _msg(
                _msg(_overlay_view_class(), b'alloc'), b'initWithFrame:', rect,
                argtypes=(_CGRect,))
            if not self._host:
                raise RuntimeError('Could not create the Metal overlay view.')
            _msg(self._host, b'setLayer:', self._layer, argtypes=(ctypes.c_void_p,), restype=None)
            _msg(self._host, b'setWantsLayer:', True, argtypes=(ctypes.c_bool,), restype=None)
            _msg(content, b'addSubview:', self._host, argtypes=(ctypes.c_void_p,), restype=None)
            inserted = True
        else:
            parent = int(_msg(self._host, b'superview') or 0)
            if parent != int(content):
                if parent:
                    _msg(self._host, b'removeFromSuperview', restype=None)
                _msg(content, b'addSubview:', self._host, argtypes=(ctypes.c_void_p,), restype=None)
                inserted = True
            if placed != self._placed:
                _msg(self._host, b'setFrame:', rect, argtypes=(_CGRect,), restype=None)
        self._placed = placed
        _msg(self._host, b'setHidden:', False, argtypes=(ctypes.c_bool,), restype=None)
        # A drawable acquired in the same turn the view joins the window is
        # never shown. Commit the view, then draw on the retry.
        if inserted:
            _msg(_class(b'CATransaction'), b'flush', restype=None)
            self._defer_present = True
        pixel = (points[0] * self._scale, points[1] * self._scale)
        if inserted or self._drawable_pixels != pixel:
            self._drawable_pixels = pixel
            _msg(self._layer, b'setContentsScale:', self._scale,
                 argtypes=(ctypes.c_double,), restype=None)
            _msg(self._layer, b'setDrawableSize:',
                 _CGSize(pixel[0], pixel[1]),
                 restype=None, argtypes=(_CGSize,))

    def set_visible(self, visible):
        if self._host:
            _msg(self._host, b'setHidden:', not visible, argtypes=(ctypes.c_bool,), restype=None)

    def _release_held(self):
        """Drop the previous drawable now that the run loop has committed it."""
        if self._held_command:
            _msg(self._held_command, b'release', restype=None)
            self._held_command = None
        if self._held_drawable:
            _msg(self._held_drawable, b'release', restype=None)
            self._held_drawable = None

    def _prepare_image(self, image):
        if image is None or image is self._source_image:
            return
        height, width = image.shape[:2]
        if image.shape[2] == 4:
            rgba = np.ascontiguousarray(image)
        else:
            scratch = self._rgba_scratch
            if scratch is None or scratch.shape[0] != height or scratch.shape[1] != width:
                scratch = np.empty((height, width, 4), np.uint8)
                scratch[:, :, 3] = 255
                self._rgba_scratch = scratch
            scratch[:, :, :3] = image
            rgba = scratch
        if self._texture_size != (width, height):
            if self._texture is not None:
                _msg(self._texture, b'release', restype=None)
            descriptor = _msg(
                _class(b'MTLTextureDescriptor'),
                b'texture2DDescriptorWithPixelFormat:width:height:mipmapped:',
                _MTL_PIXEL_RGBA8, width, height, False,
                argtypes=(ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_bool))
            # The default on macOS is managed storage. A CPU upload then stays
            # invisible to the GPU unless it is explicitly synchronized.
            _msg(descriptor, b'setStorageMode:', _MTL_STORAGE_SHARED,
                 argtypes=(ctypes.c_uint64,), restype=None)
            self._texture = _msg(
                self._device, b'newTextureWithDescriptor:', descriptor,
                argtypes=(ctypes.c_void_p,))
            if not self._texture:
                raise RuntimeError('Metal could not create an image texture.')
            self._texture_size = (width, height)
        region = _MTLRegion(_MTLOrigin(0, 0, 0), _MTLSize(width, height, 1))
        _msg(
            self._texture, b'replaceRegion:mipmapLevel:withBytes:bytesPerRow:',
            region, 0,
            rgba.ctypes.data_as(ctypes.c_void_p), width * 4,
            argtypes=(_MTLRegion, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64),
            restype=None)
        self._source_image = image

    def _vertex_buffer(self, blob: bytes):
        if self._vbo is not None and len(blob) <= self._vbo_cap:
            ctypes.memmove(_msg(self._vbo, b'contents'), blob, len(blob))
            _msg(self._vbo, b'didModifyRange:', _NSRange(0, len(blob)),
                 restype=None, argtypes=(_NSRange,))
            return
        if self._vbo is not None:
            _msg(self._vbo, b'release', restype=None)
        self._vbo_cap = max(len(blob), 4096)
        storage = ctypes.create_string_buffer(blob, self._vbo_cap)
        # Keep the padded tail zeroed. create_string_buffer copies ``blob`` and
        # the extra bytes are zero.
        self._vbo = _msg(
            self._device, b'newBufferWithBytes:length:options:',
            ctypes.addressof(storage), self._vbo_cap, _MTL_STORAGE_SHARED,
            argtypes=(ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64))
        if not self._vbo:
            raise RuntimeError('Metal could not create a vertex buffer.')

    def draw(self, width, height, image, picture, overlays, drawable_id=None):
        if drawable_id:
            self._drawable_id = int(drawable_id)
        self._place_overlay(width, height)
        if self._defer_present:
            self._defer_present = False
            return False
        # The previous frame's transaction has committed by the time another
        # draw runs. Releasing it first keeps a drawable free for this frame.
        self._release_held()
        pool = _pool()
        drawable = None
        command = None
        try:
            drawable = _msg(self._layer, b'nextDrawable')
            if not drawable:
                if not self._noted_missing_drawable:
                    self._noted_missing_drawable = True
                    sys.stderr.write('Metal preview: the layer produced no drawable.\n')
                    sys.stderr.flush()
                return False
            drawable = _msg(drawable, b'retain')
            self._prepare_image(image)
            draws, blob = _pack_vertices(picture, overlays)
            if blob:
                self._vertex_buffer(blob)
            desc = _msg(_class(b'MTLRenderPassDescriptor'), b'renderPassDescriptor')
            attachment = _msg(
                _msg(desc, b'colorAttachments'),
                b'objectAtIndexedSubscript:', 0, argtypes=(ctypes.c_uint64,))
            _msg(attachment, b'setTexture:', _msg(drawable, b'texture'),
                 argtypes=(ctypes.c_void_p,), restype=None)
            _msg(attachment, b'setLoadAction:', _MTL_LOAD_CLEAR,
                 argtypes=(ctypes.c_uint64,), restype=None)
            _msg(attachment, b'setStoreAction:', _MTL_STORE,
                 argtypes=(ctypes.c_uint64,), restype=None)
            command = _msg(self._queue, b'commandBuffer')
            if not command:
                return False
            command = _msg(command, b'retain')
            encoder = _msg(
                command, b'renderCommandEncoderWithDescriptor:', desc,
                argtypes=(ctypes.c_void_p,))
            if not encoder:
                return False
            if blob:
                _msg(encoder, b'setVertexBuffer:offset:atIndex:',
                     self._vbo, 0, 0,
                     argtypes=(ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64),
                     restype=None)
                for kind, first, count, color in draws:
                    pipeline = self._texture_pipeline if kind == 'texture' else self._color_pipeline
                    _msg(encoder, b'setRenderPipelineState:', pipeline,
                         argtypes=(ctypes.c_void_p,), restype=None)
                    if kind == 'texture':
                        _msg(encoder, b'setFragmentTexture:atIndex:',
                             self._texture, 0,
                             argtypes=(ctypes.c_void_p, ctypes.c_uint64), restype=None)
                    else:
                        channels = (ctypes.c_float * 4)(*color)
                        _msg(encoder, b'setFragmentBytes:length:atIndex:',
                             ctypes.addressof(channels), 16, 0,
                             argtypes=(ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64),
                             restype=None)
                    _msg(encoder, b'drawPrimitives:vertexStart:vertexCount:',
                         _MTL_TRIANGLE, first, count,
                         argtypes=(ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64),
                         restype=None)
            _msg(encoder, b'endEncoding', restype=None)
            _msg(command, b'presentDrawable:', drawable, argtypes=(ctypes.c_void_p,), restype=None)
            _msg(command, b'commit', restype=None)
            self._held_drawable = drawable
            self._held_command = command
            drawable = None
            command = None
            return True
        finally:
            _msg(pool, b'drain', restype=None)
            if command:
                _msg(command, b'release', restype=None)
            if drawable:
                _msg(drawable, b'release', restype=None)

    def destroy(self):
        self._release_held()
        if self._host:
            try:
                _msg(self._host, b'removeFromSuperview', restype=None)
                _msg(self._host, b'release', restype=None)
            except Exception:
                pass
            self._host = None
        for obj in (
                self._vbo, self._texture, self._color_pipeline, self._texture_pipeline,
                self._library, self._layer, self._queue, self._device, *reversed(self._owned)):
            if obj:
                try:
                    _msg(obj, b'release', restype=None)
                except Exception:
                    pass
        self._vbo = None
        self._texture = None
        self._layer = None
        self._device = None
        self._owned = []

def _ndc(x, y, width, height):
    'Widget pixels to Vulkan NDC. Y grows downward in both.'
    return ((((x / width) * 2.0) - 1.0), (((y / height) * 2.0) - 1.0))
_BBOX_LINE_WIDTH = 3
_BBOX_DASH = (8, 4)


def _segment_quad(x1, y1, x2, y2, width, height, thickness):
    """A screen-space line as two triangles, `thickness` pixels wide."""
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return []
    offset_x = -dy / length * (thickness / 2)
    offset_y = dx / length * (thickness / 2)
    corners = [
        (*_ndc(x1 + offset_x, y1 + offset_y, width, height), 0.0, 0.0),
        (*_ndc(x2 + offset_x, y2 + offset_y, width, height), 0.0, 0.0),
        (*_ndc(x2 - offset_x, y2 - offset_y, width, height), 0.0, 0.0),
        (*_ndc(x1 - offset_x, y1 - offset_y, width, height), 0.0, 0.0),
    ]
    return [corners[index] for index in (0, 1, 2, 0, 2, 3)]


def _dash_segments(x1, y1, x2, y2, dash, gap):
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return
    traveled = 0.0
    pattern = dash + gap
    while traveled < length:
        end = min(traveled + dash, length)
        yield (
            x1 + dx * traveled / length,
            y1 + dy * traveled / length,
            x1 + dx * end / length,
            y1 + dy * end / length,
        )
        traveled += pattern


def _bbox_outline(x1, y1, x2, y2, width, height):
    """White border with black dashes, matching the OpenGL selection."""
    sides = ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1))
    shapes = []
    dash, gap = _BBOX_DASH
    for side in sides:
        quad = _segment_quad(*side, width, height, _BBOX_LINE_WIDTH)
        if quad:
            shapes.append((quad, (1, 1, 1, 1)))
    for side in sides:
        for segment in _dash_segments(*side, dash, gap):
            quad = _segment_quad(*segment, width, height, _BBOX_LINE_WIDTH)
            if quad:
                shapes.append((quad, (0, 0, 0, 1)))
    return shapes


def _speaker_overlays(rect, muted, volume, width, height):
    """The mute button: a flared speaker, waves, or a red X."""
    return _icon_overlays(_animationplayback.speaker_icon(rect, muted, volume), width, height)


def _icon_overlays(icon_shapes, width, height):
    shapes = []
    for shape in icon_shapes:
        kind = shape[0]
        if kind == 'rect':
            _kind, x1, y1, x2, y2, color = shape
            shapes.append((_rect_triangles(x1, y1, x2, y2, width, height), color))
        elif kind == 'poly':
            _kind, points, color = shape
            ndc = [_ndc(x, y, width, height) for x, y in points]
            vertices = []
            for index in range(1, len(ndc) - 1):
                vertices.extend([
                    (*ndc[0], 0.0, 0.0),
                    (*ndc[index], 0.0, 0.0),
                    (*ndc[index + 1], 0.0, 0.0),
                ])
            shapes.append((vertices, color))
        elif kind == 'stroke':
            _kind, points, thickness, color = shape
            vertices = []
            for start, end in zip(points, points[1:]):
                vertices.extend(_segment_quad(*start, *end, width, height, thickness))
            if vertices:
                shapes.append((vertices, color))
    return shapes


def _rect_triangles(x1, y1, x2, y2, width, height):
    corners = [(*_ndc(x1, y1, width, height), 0.0, 0.0), (*_ndc(x2, y1, width, height), 1.0, 0.0), (*_ndc(x2, y2, width, height), 1.0, 1.0), (*_ndc(x1, y2, width, height), 0.0, 1.0)]
    return [corners[index] for index in (0, 1, 2, 0, 2, 3)]
def _picture_triangles(origin_x, origin_y, picture_w, picture_h, width, height):
    corners = [(*_ndc(origin_x, origin_y, width, height), 0.0, 0.0), (*_ndc((origin_x + picture_w), origin_y, width, height), 1.0, 0.0), (*_ndc((origin_x + picture_w), (origin_y + picture_h), width, height), 1.0, 1.0), (*_ndc(origin_x, (origin_y + picture_h), width, height), 0.0, 1.0)]
    return [corners[index] for index in (0, 1, 2, 0, 2, 3)]
class ImageViewerMetal(tk.Frame):
    """
    Preview pane drawn with Metal.

    The macOS console preview. Still images, and finished animations (GIF,
    WebP, APNG, and MP4, including audio) with a timeline. The picture stays
    above the controls. ``DGENERATE_CONSOLE_UI_METAL=0`` leaves macOS on the
    Vulkan preview when that is enabled, or on the Tk canvas.
    """
    def __init__(self, parent, **kwargs):
        # Black until a picture is loaded. An empty background is only used
        # once something is on screen, so a sash drag does not smear.
        kwargs['bg'] = 'black'
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, **kwargs)
        self._is_macos = (platform.system() == 'Darwin')
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 80.0
        self._zoom_step = 1.2
        self._pan_x = 0
        self._pan_y = 0
        self._original_image_array = None
        self._original_image_size = None
        self._image_path = None
        self._base_display_width = None
        self._base_display_height = None
        self._gpu = None
        self._gpu_error = None
        self._bbox_selection_mode = False
        self._bbox_selection_coords_seperator = None
        self._bbox_start_coords = None
        self._bbox_end_coords = None
        self._bbox_widget_start = None
        self._bbox_widget_current = None
        self._is_panning = False
        self._is_alt_panning = False
        self._last_pan_x = 0
        self._last_pan_y = 0
        self.on_error = None
        self.on_info = None
        self._animation = None
        self._loop = True
        self._animation_after = None
        self._uploaded_frame = None
        self._presented_key = None
        self._pending_load = None
        self._present_retries = 0
        self._reported_present_failure = False
        self._retry_after = None
        self._resize_after = None
        self._resize_drawn_at = 0.0
        self._pointer_inside = False
        self._pointer_moved = 0.0
        self._scrubbing = False
        self._resume_after_scrub = False
        self._volume_dragging = False
        self._preview_volume, self._preview_muted = _animationplayback.load_preview_audio()
        _mousewheelbind.bind_mousewheel(self.bind, self._on_mouse_wheel)
        self.bind('<Button-1>', self._on_left_click)
        self.bind('<B1-Motion>', self._on_left_drag)
        self.bind('<ButtonRelease-1>', self._on_left_release)
        self.bind('<Button-2>', self._on_middle_click)
        self.bind('<B2-Motion>', self._on_middle_drag)
        self.bind('<ButtonRelease-2>', self._on_middle_release)
        self.bind('<Motion>', self._on_pointer_motion)
        self.bind('<Enter>', self._on_pointer_enter)
        self.bind('<Leave>', self._on_pointer_leave)
        self.bind('<Key>', self._on_key_press)
        self.bind('<Configure>', self._on_configure)
        self.bind('<Map>', self._on_map)
        self.bind('<Unmap>', self._on_unmap)
        self.bind('<Control-equal>', lambda e: self._zoom_by_factor(self._zoom_step))
        self.bind('<Control-minus>', lambda e: self._zoom_by_factor((1 / self._zoom_step)))
        if self._is_macos:
            self.bind('<Command-equal>', lambda e: self._zoom_by_factor(self._zoom_step))
            self.bind('<Command-minus>', lambda e: self._zoom_by_factor((1 / self._zoom_step)))
        self.focus_set()
        return
    def bind_event(self, event, callback):
        self.bind(event, callback)
        return
    def unbind_event(self, event):
        self.unbind(event)
        return
    def get_image_path(self):
        return self._image_path
    def has_image(self):
        return (self._original_image_array is not None)
    def _sync_backdrop(self):
        """Fill an empty preview. A loaded picture keeps the last presented frame."""
        color = '' if self.has_image() else 'black'
        try:
            if self.cget('bg') != color:
                self.configure(bg=color)
        except tk.TclError:
            return
    def get_coordinates_at_cursor(self, widget_x, widget_y):
        return self._widget_to_image_coordinates(widget_x, widget_y)
    def _ensure_gpu(self):
        if self._gpu is not None or self._gpu_error:
            return self._gpu is not None
        self.update_idletasks()
        if self.winfo_width() <= 1 or self.winfo_height() <= 1:
            return False
        try:
            scale = 1.0
            try:
                scale = max(1.0, float(self.winfo_fpixels('1i')) / 72.0)
            except (tk.TclError, TypeError, ValueError):
                scale = 1.0
            self._gpu = _MetalDevice(
                widget=self,
                drawable_id=int(self.winfo_id()),
                width=max(1, self.winfo_width()),
                height=max(1, self.winfo_height()),
                scale=scale)
            return True
        except Exception as error:
            # The toplevel can be mapped before Tk has stored its NSView.
            # Try again on the next draw instead of giving up with no message.
            message = str(error)
            if 'not on screen' in message or 'no NSView' in message:
                return False
            self._gpu_error = message
            if self.on_error:
                self.on_error(f'Metal preview: {error}')
            return False
    def _picture_key(self, width, height):
        'Identity of what is on screen. Unchanged playback ticks skip the GPU.'
        playhead = None
        playing = None
        if (self._animation is not None):
            playing = self._animation.playing
            if self._timeline_visible():
                duration = (self._animation.duration if (self._animation.duration > 0) else 1.0)
                playhead = int(((width * self._animation.time()) / duration))
        return (id(self._original_image_array), width, height, self._zoom_factor, self._pan_x, self._pan_y, playhead, playing, self._preview_muted, self._preview_volume, self._bbox_widget_start, self._bbox_widget_current)
    def _can_present(self):
        """The widget is on screen and has a real size.

        A pane that was torn off into its own window is unmapped, and Tk
        reports it as a pixel or two. Presenting then builds a tiny swapchain
        and, once the pane returns at the same size, the picture is not drawn again.
        """
        return self.winfo_ismapped() and self.winfo_width() > 1 and self.winfo_height() > 1
    def redraw(self):
        # Configure fires for every pixel of a drag. Rebuilding the swapchain
        # on each one flashes, and a present that fails mid-drag must not be
        # the last attempt or the picture stays gone.
        if self._resize_after is not None:
            return
        self._sync_backdrop()
        if not self._can_present() or not self.has_image():
            if self._gpu is not None:
                self._gpu.set_visible(False)
            return
        if not self._ensure_gpu():
            return
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        key = self._picture_key(width, height)
        if key == self._presented_key:
            return
        picture = None
        if self._base_display_width:
            display_w = self._base_display_width * self._zoom_factor
            display_h = self._base_display_height * self._zoom_factor
            left, top = self._placed_image_origin(display_w, display_h)
            picture = _picture_triangles(left, top, display_w, display_h, width, height)
        overlays = []
        if self._bbox_widget_start and self._bbox_widget_current:
            x1, y1 = self._bbox_widget_start
            x2, y2 = self._bbox_widget_current
            overlays.extend(_bbox_outline(x1, y1, x2, y2, width, height))
        if self._timeline_visible():
            overlays.extend(self._timeline_overlays(width, height))
        try:
            presented = self._gpu.draw(
                width, height, self._original_image_array, picture, overlays, int(self.winfo_id()))
        except Exception as error:
            if self.on_error:
                self.on_error(f'Metal preview: {error}')
            return
        if presented:
            self._presented_key = key
            self._present_retries = 0
            return
        # A present that fails while the sash is still moving must not be the
        # last try. Zoom used to be the only thing that asked again.
        if self._can_present() and self._present_retries < 40:
            self._present_retries += 1
            self._schedule_present_retry()
            return
        if self.on_error and not self._reported_present_failure:
            self._reported_present_failure = True
            self.on_error('Metal preview: the picture did not reach the screen.')
    def _schedule_settled_present(self):
        self.after_idle(self._present_settled)
        self.after(50, self._present_settled)

    def _present_settled(self):
        if not self.has_image() or self._resize_after is not None:
            return
        self._presented_key = None
        self._present_retries = 0
        self.redraw()

    def _schedule_present_retry(self):
        if self._retry_after is not None:
            return
        self._retry_after = self.after(30, self._retry_present)
    def _retry_present(self):
        self._retry_after = None
        if self._resize_after is not None:
            self._schedule_present_retry()
            return
        self._presented_key = None
        self.redraw()

    def _timeline_overlays(self, width, height):
        layout = _animationplayback.control_layout(width, height)
        clip = self._animation
        if layout is None or clip is None:
            return []
        shapes = []

        def rect(box, color):
            shapes.append((_rect_triangles(*box, width, height), color))

        rect(layout['bar'], (0, 0, 0, 0.72))
        track = layout['track']
        rect(track, (1, 1, 1, 0.35))
        duration = clip.duration if clip.duration > 0 else 1.0
        fraction = min(1.0, max(0.0, clip.time() / duration))
        played = track[0] + (track[2] - track[0]) * fraction
        rect((track[0], track[1], played, track[3]), (0.9, 0.15, 0.15, 1))
        rect((played - 3, track[1] - 4, played + 3, track[3] + 4), (1, 1, 1, 1))
        play = layout['play']
        if clip.playing:
            mid = (play[0] + play[2]) / 2
            rect((mid - 8, play[1], mid - 4, play[3]), (1, 1, 1, 1))
            rect((mid + 4, play[1], mid + 8, play[3]), (1, 1, 1, 1))
        else:
            ax, ay = _ndc(play[0] + 2, play[1], width, height)
            bx, by = _ndc(play[0] + 2, play[3], width, height)
            cx, cy = _ndc(play[2], (play[1] + play[3]) / 2, width, height)
            shapes.append(([(ax, ay, 0, 0), (bx, by, 0, 0), (cx, cy, 0, 0)], (1, 1, 1, 1)))
        volume = layout['volume']
        rect(volume, (1, 1, 1, 0.35))
        shown = 0.0 if self._preview_muted else self._preview_volume
        filled = volume[0] + (volume[2] - volume[0]) * shown
        rect((volume[0], volume[1], filled, volume[3]), (1, 1, 1, 1))
        shapes.extend(_icon_overlays(
            _animationplayback.loop_icon(layout['loop'], self._loop), width, height))
        shapes.extend(_icon_overlays(
            _animationplayback.speaker_icon(layout['mute'], self._preview_muted, self._preview_volume),
            width, height))
        return shapes
    def _control_reserve(self):
        if (self._animation is None):
            return 0
        return (_animationplayback.BAR_HEIGHT + _animationplayback.CONTROL_GAP)
    def _content_box(self):
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 0 or height <= 0:
            width, height = 800, 600
        return (0.0, 0.0, float(width), float(max(1, (height - self._control_reserve()))))
    def _placed_image_origin(self, display_width, display_height):
        x = self._content_box()[0]
        y = self._content_box()[1]
        content_width = self._content_box()[2]
        content_height = self._content_box()[3]
        return (((x + ((content_width - display_width) / 2)) + self._pan_x), ((y + ((content_height - display_height) / 2)) + self._pan_y))
    def _calculate_base_display_size(self):
        if (not self.has_image()):
            return
        _x = self._content_box()[0]
        _y = self._content_box()[1]
        content_width = self._content_box()[2]
        content_height = self._content_box()[3]
        img_width = self._original_image_size[0]
        img_height = self._original_image_size[1]
        image_aspect = (img_width / img_height)
        if (image_aspect > (content_width / content_height)):
            self._base_display_width = content_width
            self._base_display_height = (content_width / image_aspect)
            return
        self._base_display_width = (content_height * image_aspect)
        self._base_display_height = content_height
        return
    def _widget_to_image_coordinates(self, widget_x, widget_y):
        if not self.has_image() or not self._base_display_width:
            return None, None
        display_w = self._base_display_width * self._zoom_factor
        display_h = self._base_display_height * self._zoom_factor
        left, top = self._placed_image_origin(display_w, display_h)
        if not (left <= widget_x <= left + display_w and top <= widget_y <= top + display_h):
            return None, None
        img_w, img_h = self._original_image_size
        image_x = int((widget_x - left) / display_w * img_w)
        image_y = int((widget_y - top) / display_h * img_h)
        return max(0, min(image_x, img_w - 1)), max(0, min(image_y, img_h - 1))

    def _image_to_widget_coordinates(self, image_x, image_y):
        if not self.has_image() or not self._base_display_width:
            return None, None
        display_w = self._base_display_width * self._zoom_factor
        display_h = self._base_display_height * self._zoom_factor
        left, top = self._placed_image_origin(display_w, display_h)
        img_w, img_h = self._original_image_size
        return int(left + image_x / img_w * display_w), int(top + image_y / img_h * display_h)
    def load_image(self, image_path, fit=False, view_state=None):
        self._image_path = image_path
        self._pending_load = (image_path, fit, view_state)
        if not self._can_present():
            return
        self._finish_pending_load()
    def _finish_pending_load(self):
        if self._pending_load is None or not self._can_present():
            return False
        image_path, fit, view_state = self._pending_load
        self._pending_load = None
        self._load_image_now(image_path, fit, view_state)
        return True
    def _load_image_now(self, image_path, fit=False, view_state=None):
        self._image_path = image_path
        try:
            self._stop_animation()
            self._presented_key = None
            self._present_retries = 0
            clip = _animationplayback.open_animation(image_path) if _animationplayback.is_animation_path(image_path) else None
            if clip is not None:
                frame = clip.frame()
                if frame is None:
                    clip.close()
                    raise RuntimeError(f'Could not read a frame from "{image_path}"')
                self._animation = clip
                clip.loop = self._loop
                self._original_image_array = frame
                self._uploaded_frame = frame
            else:
                image = PIL.Image.open(image_path)
                if image.mode not in ('RGB', 'RGBA'):
                    image = image.convert('RGB')
                self._original_image_array = np.array(image)
                image.close()
            self._original_image_size = (
                self._original_image_array.shape[1], self._original_image_array.shape[0])
            self._calculate_base_display_size()
            if view_state:
                self._zoom_factor = view_state.get('zoom_factor', 1.0)
                self._pan_x = view_state.get('pan_x', 0)
                self._pan_y = view_state.get('pan_y', 0)
            else:
                self._zoom_factor = 1.0
                self._pan_x = 0
                self._pan_y = 0
            if fit and view_state is None:
                self._calculate_base_display_size()
                self._zoom_factor = 1.0
            self.redraw()
            # The output pane inserts the "Wrote ..." line after this returns,
            # and that paint can cover a single present. A playing clip draws
            # again on its own. A still stays on the previous frame until
            # something else, such as a sash drag, presents once more.
            self._schedule_settled_present()
            if self._animation is not None:
                self._animation.set_gain(0.0 if self._preview_muted else self._preview_volume)
                self._animation.start()
                self._schedule_animation()
        except Exception as error:
            self._stop_animation()
            if self.on_error:
                self.on_error(f'Metal preview: {error}')

    def _on_map(self, _event):
        # The present from before this pane was hidden does not survive being unmapped.
        self._presented_key = None
        if not self._finish_pending_load():
            self._show_current_picture()
        return

    def _on_unmap(self, _event):
        if self._gpu is not None:
            self._gpu.set_visible(False)
        return
    def _show_current_picture(self):
        if not self._can_present() or not self.has_image():
            return
        if self._animation is not None or not self._base_display_width or self._base_display_width <= 1:
            self._calculate_base_display_size()
        self.redraw()
    def _on_configure(self, _event):
        if not self.has_image():
            self._sync_backdrop()
            if self._resize_after is not None:
                try:
                    self.after_cancel(self._resize_after)
                except Exception:
                    pass
                self._resize_after = None
            return
        if not self._can_present():
            return
        self._present_retries = 0
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except Exception:
                pass
        self._resize_after = self.after(32, self._finish_resize)
        # Cover the window often enough that it does not sit black, without
        # rebuilding the swapchain on every pixel of the drag.
        now = time.monotonic()
        if now - self._resize_drawn_at < 0.032:
            return
        self._resize_drawn_at = now
        self._draw_for_resize()
    def _draw_for_resize(self):
        pending = self._resize_after
        self._resize_after = None
        self._presented_key = None
        try:
            if not self._finish_pending_load():
                self._show_current_picture()
        finally:
            if pending is not None and self._resize_after is None:
                self._resize_after = pending
    def _finish_resize(self):
        self._resize_after = None
        self._resize_drawn_at = time.monotonic()
        self._presented_key = None
        if not self._finish_pending_load():
            self._show_current_picture()
        return
    def _schedule_animation(self):
        if self._animation is None or self._animation_after is not None:
            return
        delay = _animationplayback.playback_tick_delay_ms(
            self._animation,
            self._timeline_visible() or self._scrubbing or self._volume_dragging,
        )
        self._animation_after = self.after(delay, self._animation_tick)
    def _animation_tick(self):
        self._animation_after = None
        clip = self._animation
        if clip is None:
            return
        frame = clip.frame()
        if frame is not None and frame is not self._uploaded_frame:
            self._original_image_array = frame
            self._uploaded_frame = frame
        self.redraw()
        if clip.playing or self._pointer_inside or self._scrubbing or self._volume_dragging:
            self._schedule_animation()
    def _stop_animation(self):
        if (self._animation_after is not None):
            try:
                self.after_cancel(self._animation_after)
            except Exception:
                pass
            self._animation_after = None
        clip = self._animation
        self._animation = None
        self._uploaded_frame = None
        self._scrubbing = False
        self._volume_dragging = False
        if (clip is not None):
            clip.close()
            return
        return
    def _timeline_visible(self):
        if self._animation is None or self._bbox_selection_mode:
            return False
        if self._scrubbing or self._volume_dragging:
            return True
        if not self._pointer_inside:
            return False
        if not self._animation.playing:
            return True
        return (time.perf_counter() - self._pointer_moved) < 2.5
    def _on_pointer_enter(self, _event):
        self._pointer_inside = True
        self._pointer_moved = time.perf_counter()
        self.redraw()
        return
    def _on_pointer_leave(self, _event):
        self._pointer_inside = False
        if (not self._scrubbing):
            if (not self._volume_dragging):
                self.redraw()
                return
            return
        return
    def _on_pointer_motion(self, _event):
        self._pointer_inside = True
        self._pointer_moved = time.perf_counter()
        if self._timeline_visible():
            self.redraw()
            return
        return
    def _on_left_click(self, event):
        if (not self.has_image()):
            return
        if (self._animation is not None):
            if (not self._bbox_selection_mode):
                if self._timeline_visible():
                    hit = _animationplayback.control_hit(event.x, event.y, self.winfo_width(), self.winfo_height())
                    if (hit == 'play'):
                        self._animation.toggle()
                        self._schedule_animation()
                        self.redraw()
                        return 'break'
                    if (hit == 'loop'):
                        self._loop = (not self._loop)
                        self._animation.loop = self._loop
                        self.redraw()
                        return 'break'
                    if (hit == 'mute'):
                        self._preview_muted = (not self._preview_muted)
                        self._animation.set_gain((0.0 if self._preview_muted else self._preview_volume))
                        _animationplayback.save_preview_audio(self._preview_volume, self._preview_muted)
                        self.redraw()
                        return 'break'
                    if (hit == 'volume'):
                        self._volume_dragging = True
                        self._set_volume_from_x(event.x)
                        return 'break'
                    if (hit == 'track'):
                        self._scrubbing = True
                        self._resume_after_scrub = self._animation.playing
                        if self._animation.playing:
                            self._animation.pause()
                        self._scrub_to(event.x)
                        return 'break'
        if (event.state & (0x8 if self._is_macos else 0x4)):
            self._is_alt_panning = True
            self._last_pan_x = event.x
            self._last_pan_y = event.y
            return 'break'
        if (not self._bbox_selection_mode):
            return
        image_x = self._widget_to_image_coordinates(event.x, event.y)[0]
        image_y = self._widget_to_image_coordinates(event.x, event.y)[1]
        if (image_x is None):
            return
        self._bbox_start_coords = (image_x, image_y)
        self._bbox_end_coords = (image_x, image_y)
        self._bbox_widget_start = (event.x, event.y)
        self._bbox_widget_current = (event.x, event.y)
        return 'break'
    def _on_left_drag(self, event):
        if self._volume_dragging:
            self._set_volume_from_x(event.x)
            return 'break'
        if self._scrubbing:
            self._scrub_to(event.x)
            return 'break'
        if self._is_alt_panning:
            self._pan_x = (self._pan_x + (event.x - self._last_pan_x))
            self._pan_y = (self._pan_y + (event.y - self._last_pan_y))
            self._last_pan_x = event.x
            self._last_pan_y = event.y
            self.redraw()
            return 'break'
        if self._bbox_selection_mode:
            if (self._bbox_start_coords is not None):
                image_x = self._widget_to_image_coordinates(event.x, event.y)[0]
                image_y = self._widget_to_image_coordinates(event.x, event.y)[1]
                if (image_x is not None):
                    self._bbox_end_coords = (image_x, image_y)
                    self._bbox_widget_current = (event.x, event.y)
                    self.redraw()
        return 'break'
    def _on_left_release(self, event):
        if self._volume_dragging:
            self._set_volume_from_x(event.x)
            self._volume_dragging = False
            _animationplayback.save_preview_audio(self._preview_volume, self._preview_muted)
            return 'break'
        if self._scrubbing:
            self._scrub_to(event.x)
            self._scrubbing = False
            if self._resume_after_scrub:
                if (self._animation is not None):
                    self._animation.toggle()
            self.redraw()
            return 'break'
        self._is_alt_panning = False
        if self._bbox_selection_mode:
            if self._bbox_start_coords:
                if self._bbox_end_coords:
                    self._complete_bbox_selection()
                    return
                return
            return
        return
    def _set_volume_from_x(self, widget_x):
        self._preview_volume = _animationplayback.volume_on_slider(widget_x, self.winfo_width(), self.winfo_height())
        if (self._preview_volume > 0):
            self._preview_muted = False
        if (self._animation is not None):
            self._animation.set_gain((0.0 if self._preview_muted else self._preview_volume))
        self.redraw()
        return
    def _scrub_to(self, widget_x):
        if (self._animation is None):
            return
        self._animation.seek(_animationplayback.time_on_track(widget_x, self.winfo_width(), self.winfo_height(), self._animation.duration))
        frame = self._animation.frame()
        if (frame is not None):
            self._original_image_array = frame
            self._uploaded_frame = frame
        self.redraw()
        return
    def _on_middle_click(self, event):
        self._is_panning = True
        self._last_pan_x = event.x
        self._last_pan_y = event.y
        return
    def _on_middle_drag(self, event):
        if (not self._is_panning):
            return
        self._pan_x = (self._pan_x + (event.x - self._last_pan_x))
        self._pan_y = (self._pan_y + (event.y - self._last_pan_y))
        self._last_pan_x = event.x
        self._last_pan_y = event.y
        self.redraw()
        return
    def _on_middle_release(self, _event):
        self._is_panning = False
        return
    def _on_mouse_wheel(self, event):
        if (not self.has_image()):
            return
        direction = _mousewheelbind.scroll_direction(event)
        if direction == 0:
            return
        zoom_in = direction > 0
        factor = (self._zoom_step if zoom_in else (1 / self._zoom_step))
        self._zoom_by_factor(factor)
        return
    def _zoom_by_factor(self, factor):
        self._zoom_factor = max(self._min_zoom, min(self._max_zoom, (self._zoom_factor * factor)))
        self.redraw()
        return
    def _on_key_press(self, event):
        if (event.keysym == 'Escape'):
            if self._bbox_selection_mode:
                self._cancel_bbox_selection()
                return
        if (event.keysym == 'space'):
            if (self._animation is not None):
                self._animation.toggle()
                self._schedule_animation()
                self.redraw()
                return 'break'
            return
        return
    def start_bbox_selection(self, seperator):
        if (not self.has_image()):
            return
        self._bbox_selection_mode = True
        self._bbox_selection_coords_seperator = seperator
        self._bbox_start_coords = None
        self._bbox_end_coords = None
        if self.on_info:
            self.on_info('Bounding box selection mode started. Left-click and drag to select area. Press Escape to cancel.')
            return
        return
    def _complete_bbox_selection(self):
        x1 = self._bbox_start_coords[0]
        y1 = self._bbox_start_coords[1]
        x2 = self._bbox_end_coords[0]
        y2 = self._bbox_end_coords[1]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        text = self._bbox_selection_coords_seperator.join((str(value) for value in (x1, y1, x2, y2)))
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            if self.on_info:
                self.on_info(f'Bounding box copied to clipboard: {text}')
        except Exception as error:
            if self.on_error:
                self.on_error(f'Metal preview: {error}')
        self._cancel_bbox_selection()
        return
    def _cancel_bbox_selection(self):
        self._bbox_selection_mode = False
        self._bbox_start_coords = None
        self._bbox_end_coords = None
        self._bbox_widget_start = None
        self._bbox_widget_current = None
        self.redraw()
        return
    def copy_path(self):
        if (not self._image_path):
            return
        self.clipboard_clear()
        self.clipboard_append(self._image_path)
        return
    def reset_view(self):
        self._calculate_base_display_size()
        if self._base_display_width:
            self._zoom_factor = (self._original_image_size[0] / self._base_display_width)
        self._pan_x = 0
        self._pan_y = 0
        self.redraw()
        return
    def zoom_to_fit(self):
        self._calculate_base_display_size()
        self._zoom_factor = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self.redraw()
        return
    def get_view_state(self):
        if (not self.has_image()):
            return
        return {'zoom_factor': self._zoom_factor, 'pan_x': self._pan_x, 'pan_y': self._pan_y}
    def set_view_state(self, view_state):
        if (not view_state):
            return
        self._zoom_factor = view_state.get('zoom_factor', 1.0)
        self._pan_x = view_state.get('pan_x', 0)
        self._pan_y = view_state.get('pan_y', 0)
        self.redraw()
        return
    def request_help(self):
        if (not self.on_info):
            return
        _helpdialog.show_help_dialog(title='Metal preview help', help_text='\n'.join([*('Metal preview controls:', '• Mouse wheel: zoom', '• Command+drag or middle-drag: pan', '• Move the pointer over an animation for the timeline', '• Space: pause and resume', '• Loop arrow: replay the clip, or stop at the end when slashed', '• Speaker: mute (red X) or unmute; slider sets volume. Both are remembered')]), parent=self.master, size=(460, 280), position_widget=self.master, dock_to_right=False)
        return
    def cleanup(self):
        self._stop_animation()
        if (self._gpu is not None):
            self._gpu.destroy()
            self._gpu = None
            return
        return
