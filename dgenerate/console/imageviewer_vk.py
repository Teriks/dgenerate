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

"""Vulkan preview pane.

Installed with the ``console_ui_vulkan`` extra. The console uses it on
Windows, Linux, and macOS whenever that package is present.
``DGENERATE_CONSOLE_UI_VULKAN=0`` selects OpenGL, or the Tk canvas when
OpenGL is not installed. Linux presents through X11 (including XWayland).
macOS presents through MoltenVK.
"""
import ctypes
import ctypes.util
import platform
import struct
import time
import tkinter as tk
import typing
import numpy as np
import PIL.Image
import vulkan as vk
from vulkan._vulkan import ffi, lib
import dgenerate.console.animationplayback as _animationplayback
import dgenerate.console.helpdialog as _helpdialog
import dgenerate.console.mousewheelbind as _mousewheelbind
_CAPABILITY = 17
_EXT_IMPORT = 11
_MEMORY_MODEL = 14
_ENTRY = 15
_EXEC_MODE = 16
_DECORATE = 71
_MEMBER_DECORATE = 72
_TYPE_VOID = 19
_TYPE_INT = 21
_TYPE_FLOAT = 22
_TYPE_VECTOR = 23
_TYPE_IMAGE = 25
_TYPE_SAMPLED = 27
_TYPE_STRUCT = 30
_TYPE_POINTER = 32
_TYPE_FUNCTION = 33
_CONSTANT = 43
_FUNCTION = 54
_FUNCTION_END = 56
_VARIABLE = 59
_LOAD = 61
_STORE = 62
_ACCESS = 65
_COMPOSITE = 80
_SAMPLE = 87
_LABEL = 248
_RETURN = 253
class _Spirv:
    def __init__(self):
        self._ops = []
        self._next = 1
        return
    def _id(self):
        value = self._next
        self._next = (self._next + 1)
        return value
    def raw(self, code, *operands):
        self._ops.append((code, operands))
        return
    def result(self, code, *operands):
        ident = self._id()
        self._ops.append((code, (ident, *operands)))
        return ident
    def typed(self, code, type_id, *operands):
        ident = self._id()
        self._ops.append((code, (type_id, ident, *operands)))
        return ident
    def bytecode(self):
        body = []
        for code, operands in self._ops:
            encoded = []
            for operand in operands:
                if isinstance(operand, str):
                    raw = (operand.encode() + b'\x00')
                    raw = (raw + (b'\x00' * ((4 - (len(raw) % 4)) % 4)))
                    encoded.extend(struct.unpack(('<' + ('I' * (len(raw) // 4))), raw))
                    continue
                encoded.append((int(operand) & 4294967295))
            body.append((((1 + len(encoded)) << 16) | code))
            body.extend(encoded)
        header = [119734787, 65536, 0, self._next, 0]
        return struct.pack(('<' + ('I' * (len(header) + len(body)))), *header, *body)
def _vertex_shader():
    spirv = _Spirv()
    spirv.raw(_CAPABILITY, 1)
    spirv.result(_EXT_IMPORT, 'GLSL.std.450')
    spirv.raw(_MEMORY_MODEL, 0, 1)
    void = spirv.result(_TYPE_VOID)
    fn = spirv.result(_TYPE_FUNCTION, void)
    f32 = spirv.result(_TYPE_FLOAT, 32)
    v2 = spirv.result(_TYPE_VECTOR, f32, 2)
    v4 = spirv.result(_TYPE_VECTOR, f32, 4)
    in_ptr = spirv.result(_TYPE_POINTER, 1, v2)
    pos = spirv.result(_VARIABLE, in_ptr, 1)
    uv = spirv.result(_VARIABLE, in_ptr, 1)
    out_uv_ptr = spirv.result(_TYPE_POINTER, 3, v2)
    out_uv = spirv.result(_VARIABLE, out_uv_ptr, 3)
    out_pos_ptr = spirv.result(_TYPE_POINTER, 3, v4)
    gl_pos = spirv.result(_VARIABLE, out_pos_ptr, 3)
    zero = spirv.typed(_CONSTANT, f32, 0)
    one = spirv.typed(_CONSTANT, f32, 1065353216)
    spirv.raw(_ENTRY, 0, 0, 'main', pos, uv, out_uv, gl_pos)
    return _vertex_shader_ids()
def _vertex_shader_ids():
    'NDC position in location 0, UV in location 1, UV passed to location 0.'
    s = _Spirv()
    s.raw(_CAPABILITY, 1)
    s.result(_EXT_IMPORT, 'GLSL.std.450')
    s.raw(_MEMORY_MODEL, 0, 1)
    void = s.result(_TYPE_VOID)
    fn_type = s.result(_TYPE_FUNCTION, void)
    f32 = s.result(_TYPE_FLOAT, 32)
    v2 = s.result(_TYPE_VECTOR, f32, 2)
    v4 = s.result(_TYPE_VECTOR, f32, 4)
    pin = s.result(_TYPE_POINTER, 1, v2)
    pos = s.result(_VARIABLE, pin, 1)
    uv_in = s.result(_VARIABLE, pin, 1)
    pout = s.result(_TYPE_POINTER, 3, v2)
    uv_out = s.result(_VARIABLE, pout, 3)
    ppos = s.result(_TYPE_POINTER, 3, v4)
    gl_pos = s.result(_VARIABLE, ppos, 3)
    zero = s.typed(_CONSTANT, f32, 0)
    one = s.typed(_CONSTANT, f32, 1065353216)
    main = s.typed(_FUNCTION, void, 0, fn_type)
    del main
    del zero
    del one
    del gl_pos
    del uv_out
    del pos
    del uv_in
    return _assemble_vertex()
def _words_from_ops(ops, bound):
    body = []
    for code, operands in ops:
        encoded = []
        for operand in operands:
            if isinstance(operand, str):
                raw = (operand.encode() + b'\x00')
                raw = (raw + (b'\x00' * ((4 - (len(raw) % 4)) % 4)))
                encoded.extend(struct.unpack(('<' + ('I' * (len(raw) // 4))), raw))
                continue
            encoded.append((int(operand) & 4294967295))
        body.append((((1 + len(encoded)) << 16) | code))
        body.extend(encoded)
    header = [119734787, 65536, 0, bound, 0]
    return struct.pack(('<' + ('I' * (len(header) + len(body)))), *header, *body)
def _assemble_vertex():
    ops = [(_CAPABILITY, (1,)), (_EXT_IMPORT, (1, 'GLSL.std.450')), (_MEMORY_MODEL, (0, 1)), (_ENTRY, (0, 16, 'main', 8, 9, 11, 13)), (_DECORATE, (8, 30, 0)), (_DECORATE, (9, 30, 1)), (_DECORATE, (11, 30, 0)), (_DECORATE, (13, 11, 0)), (_TYPE_VOID, (2,)), (_TYPE_FUNCTION, (3, 2)), (_TYPE_FLOAT, (4, 32)), (_TYPE_VECTOR, (5, 4, 2)), (_TYPE_VECTOR, (6, 4, 4)), (_TYPE_POINTER, (7, 1, 5)), (_TYPE_POINTER, (10, 3, 5)), (_TYPE_POINTER, (12, 3, 6)), (_VARIABLE, (7, 8, 1)), (_VARIABLE, (7, 9, 1)), (_VARIABLE, (10, 11, 3)), (_VARIABLE, (12, 13, 3)), (_CONSTANT, (4, 14, 0)), (_CONSTANT, (4, 15, 1065353216)), (_FUNCTION, (2, 16, 0, 3)), (_LABEL, (17,)), (_LOAD, (5, 18, 8)), (_LOAD, (5, 19, 9)), (_STORE, (11, 19)), (_COMPOSITE, (6, 20, 18, 14, 15)), (_STORE, (13, 20)), (_RETURN, ()), (_FUNCTION_END, ())]
    return _words_from_ops(ops, 21)
def _assemble_texture_fragment():
    ops = [(_CAPABILITY, (1,)), (_EXT_IMPORT, (1, 'GLSL.std.450')), (_MEMORY_MODEL, (0, 1)), (_ENTRY, (4, 18, 'main', 8, 10)), (_EXEC_MODE, (18, 7)), (_DECORATE, (8, 30, 0)), (_DECORATE, (10, 30, 0)), (_DECORATE, (16, 34, 0)), (_DECORATE, (16, 33, 0)), (_TYPE_VOID, (2,)), (_TYPE_FUNCTION, (3, 2)), (_TYPE_FLOAT, (4, 32)), (_TYPE_VECTOR, (5, 4, 2)), (_TYPE_VECTOR, (6, 4, 4)), (_TYPE_POINTER, (7, 1, 5)), (_VARIABLE, (7, 8, 1)), (_TYPE_POINTER, (9, 3, 6)), (_VARIABLE, (9, 10, 3)), (_TYPE_IMAGE, (11, 4, 1, 0, 0, 0, 1, 0)), (_TYPE_SAMPLED, (12, 11)), (_TYPE_POINTER, (13, 0, 12)), (_VARIABLE, (13, 16, 0)), (_FUNCTION, (2, 18, 0, 3)), (_LABEL, (19,)), (_LOAD, (5, 20, 8)), (_LOAD, (12, 21, 16)), (_SAMPLE, (6, 22, 21, 20)), (_STORE, (10, 22)), (_RETURN, ()), (_FUNCTION_END, ())]
    return _words_from_ops(ops, 23)
def _assemble_color_fragment():
    ops = [(_CAPABILITY, (1,)), (_EXT_IMPORT, (1, 'GLSL.std.450')), (_MEMORY_MODEL, (0, 1)), (_ENTRY, (4, 16, 'main', 12)), (_EXEC_MODE, (16, 7)), (_DECORATE, (12, 30, 0)), (_DECORATE, (8, 2)), (_MEMBER_DECORATE, (8, 0, 35, 0)), (_TYPE_VOID, (2,)), (_TYPE_FUNCTION, (3, 2)), (_TYPE_FLOAT, (4, 32)), (_TYPE_VECTOR, (5, 4, 4)), (_TYPE_STRUCT, (8, 5)), (_TYPE_POINTER, (9, 9, 8)), (_VARIABLE, (9, 10, 9)), (_TYPE_INT, (6, 32, 1)), (_CONSTANT, (6, 7, 0)), (_TYPE_POINTER, (11, 9, 5)), (_TYPE_POINTER, (13, 3, 5)), (_VARIABLE, (13, 12, 3)), (_FUNCTION, (2, 16, 0, 3)), (_LABEL, (17,)), (_ACCESS, (11, 18, 10, 7)), (_LOAD, (5, 19, 18)), (_STORE, (12, 19)), (_RETURN, ()), (_FUNCTION_END, ())]
    return _words_from_ops(ops, 20)
def shader_modules_are_valid():
    """Create the preview shaders on a real device, then tear the device down."""
    try:
        device = _VulkanDevice.headless()
    except Exception:
        return False
    try:
        device.create_shader(_assemble_vertex())
        device.create_shader(_assemble_texture_fragment())
        device.create_shader(_assemble_color_fragment())
        return True
    except Exception:
        return False
    finally:
        device.destroy()
def _instance_extension_names() -> set[str]:
    names = set()
    for prop in vk.vkEnumerateInstanceExtensionProperties(None):
        name = prop.extensionName
        if isinstance(name, bytes):
            name = name.decode()
        names.add(str(name).split('\x00', 1)[0])
    return names


def _surface_extension_names(windowing: str, available: set[str]) -> tuple[list[str], int]:
    """Instance extensions and create flags for this Tk window system.

    ``windowing`` is Tk's ``tk windowingsystem``: ``win32``, ``x11``, or ``aqua``.
    """
    names = [vk.VK_KHR_SURFACE_EXTENSION_NAME]
    flags = 0
    if windowing == 'win32':
        surface = vk.VK_KHR_WIN32_SURFACE_EXTENSION_NAME
    elif windowing == 'aqua':
        if vk.VK_MVK_MACOS_SURFACE_EXTENSION_NAME in available:
            surface = vk.VK_MVK_MACOS_SURFACE_EXTENSION_NAME
        elif vk.VK_EXT_METAL_SURFACE_EXTENSION_NAME in available:
            surface = vk.VK_EXT_METAL_SURFACE_EXTENSION_NAME
        else:
            raise RuntimeError(
                'MoltenVK is not available. The Vulkan preview on macOS needs it.')
        portability = vk.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        if portability in available:
            names.append(portability)
            flags = vk.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
    elif windowing == 'x11':
        if vk.VK_KHR_XLIB_SURFACE_EXTENSION_NAME in available:
            surface = vk.VK_KHR_XLIB_SURFACE_EXTENSION_NAME
        elif vk.VK_KHR_XCB_SURFACE_EXTENSION_NAME in available:
            surface = vk.VK_KHR_XCB_SURFACE_EXTENSION_NAME
        else:
            raise RuntimeError(
                'This Vulkan loader has no X11 surface. The preview needs X11 or XWayland.')
    else:
        raise RuntimeError(
            f'The Vulkan preview does not support the {windowing} window system.')
    if surface not in available:
        raise RuntimeError(f'{surface} is not available from this Vulkan loader.')
    names.append(surface)
    return names, flags


def _composite_alpha(caps):
    """Pick a swapchain alpha the surface actually supports.

    Opaque is not supported on every compositor. Linux and MoltenVK often
    require inherit instead, and requesting an unsupported bit fails the swapchain.
    """
    supported = int(caps.supportedCompositeAlpha)
    for bit in (
        vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        vk.VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        vk.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        vk.VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    ):
        if supported & int(bit):
            return bit
    return vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR


def _objc():
    libobjc = ctypes.cdll.LoadLibrary('/usr/lib/libobjc.A.dylib')
    libobjc.sel_registerName.restype = ctypes.c_void_p
    libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
    libobjc.objc_getClass.restype = ctypes.c_void_p
    libobjc.objc_getClass.argtypes = [ctypes.c_char_p]
    return libobjc


def _objc_msg(obj, selector, *args, restype=ctypes.c_void_p, argtypes=()):
    objc = _objc()
    send = objc.objc_msgSend
    send.restype = restype
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    return send(obj, objc.sel_registerName(selector), *args)


def _cocoa_set_wants_layer(view: int):
    try:
        _objc_msg(view, b'setWantsLayer:', 1, argtypes=(ctypes.c_int8,))
    except Exception:
        return


def _cocoa_metal_layer(view: int):
    ctypes.cdll.LoadLibrary('/System/Library/Frameworks/QuartzCore.framework/QuartzCore')
    _cocoa_set_wants_layer(view)
    objc = _objc()
    layer_class = objc.objc_getClass(b'CAMetalLayer')
    if not layer_class:
        raise RuntimeError('CAMetalLayer is not available.')
    layer = _objc_msg(layer_class, b'alloc')
    layer = _objc_msg(layer, b'init')
    if not layer:
        raise RuntimeError('CAMetalLayer could not be created.')
    _objc_msg(view, b'setLayer:', layer, argtypes=(ctypes.c_void_p,))
    return layer


class _AppleSurfaceInfo(ctypes.Structure):
    """VkMacOSSurfaceCreateInfoMVK and VkMetalSurfaceCreateInfoEXT share this layout."""
    _fields_ = [
        ('sType', ctypes.c_uint32),
        ('pNext', ctypes.c_void_p),
        ('flags', ctypes.c_uint32),
        ('handle', ctypes.c_void_p),
    ]


class _VulkanDevice:
    'Swapchain and the two pipelines the preview draws with.'
    def __init__(self, hwnd=None, hinstance=None, windowing='win32'):
        self._alive = []
        self.hwnd = hwnd
        self.hinstance = hinstance
        self.windowing = windowing
        self._native_closer = None
        self._extension_names = []
        self.instance = None
        self.surface = None
        self.physical = None
        self.device = None
        self.queue = None
        self.queue_family = 0
        self.swapchain = None
        self.render_pass = None
        self.command_pool = None
        self.command = None
        self.frames = []
        self.extent = (1, 1)
        self.format = None
        self.pipelines = {}
        self.texture = None
        self.texture_view = None
        self.sampler = None
        self.descriptor_pool = None
        self.descriptor = None
        self.descriptor_layout = None
        self._texture_size = None
        self._create_instance()
        if (hwnd is not None):
            self._create_surface()
            self._create_device()
            self._create_swapchain(1, 1)
            self._create_pipelines()
            return
        return
    @classmethod
    def headless(cls):
        device = cls.__new__(cls)
        device._alive = []
        device.hwnd = None
        device.hinstance = None
        device.instance = None
        device.surface = None
        device.physical = None
        device.device = None
        device.queue = None
        device.swapchain = None
        device.frames = []
        device.pipelines = {}
        device.texture = None
        device.texture_view = None
        device.sampler = None
        device.descriptor_pool = None
        device.descriptor_layout = None
        device.descriptor = None
        device.render_pass = None
        device.command_pool = None
        device.command = None
        device._texture_size = None
        device._create_instance(surface=False)
        device._create_device(present=False)
        return device
    def _keep(self, value):
        self._alive.append(value)
        return value
    def _create_instance(self, surface=True):
        app = vk.VkApplicationInfo(pApplicationName='dgenerate', applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0), pEngineName='dgenerate', engineVersion=vk.VK_MAKE_VERSION(1, 0, 0), apiVersion=vk.VK_API_VERSION_1_0)
        flags = 0
        if surface:
            windowing = getattr(self, 'windowing', None) or {
                'Windows': 'win32', 'Darwin': 'aqua'}.get(platform.system(), 'x11')
            self.windowing = windowing
            names, flags = _surface_extension_names(windowing, _instance_extension_names())
        else:
            names = []
        self._extension_names = list(names)
        encoded = [self._keep(ffi.new('char[]', name.encode())) for name in names]
        listed = self._keep(ffi.new('char*[]', encoded)) if names else ffi.NULL
        info = vk.VkInstanceCreateInfo(
            flags=flags, pApplicationInfo=app,
            enabledExtensionCount=len(names), ppEnabledExtensionNames=listed)
        self.instance = vk.vkCreateInstance(info, None)
        return
    def _present_surface(self, info, function_name, address=None):
        encoded = self._keep(ffi.new('char[]', function_name.encode()))
        addr = lib.vkGetInstanceProcAddr(self.instance, encoded)
        if addr == ffi.NULL:
            raise RuntimeError(f'{function_name} is not available')
        # Do not look up vkCreateWin32SurfaceKHR. That symbol is absent from
        # the Linux and macOS loaders, and every surface function has this shape.
        create = ffi.cast(
            'VkResult(*)(struct VkInstance_T *, void *, struct VkAllocationCallbacks *, struct VkSurfaceKHR_T * *)',
            addr)
        if address is None:
            pointer = ffi.cast('void *', ffi.addressof(info))
            self._keep(info)
        else:
            pointer = ffi.cast('void *', address)
            self._keep(info)
        surface = ffi.new('VkSurfaceKHR*')
        result = create(self.instance, pointer, ffi.NULL, surface)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f'{function_name} failed ({result})')
        self.surface = surface[0]
        return
    def _create_surface(self):
        windowing = getattr(self, 'windowing', 'win32')
        if windowing == 'win32':
            info = vk.VkWin32SurfaceCreateInfoKHR(flags=0, hinstance=self.hinstance, hwnd=self.hwnd)
            self._present_surface(info, 'vkCreateWin32SurfaceKHR')
            return
        if windowing == 'aqua':
            self._create_apple_surface()
            return
        if windowing == 'x11':
            self._create_x11_surface()
            return
        raise RuntimeError(
            f'The Vulkan preview does not support the {windowing} window system.')
    def _create_apple_surface(self):
        view = int(self.hwnd or 0)
        if view == 0:
            raise RuntimeError('The preview window is not on screen yet.')
        info = _AppleSurfaceInfo()
        extensions = set(self._extension_names)
        if vk.VK_MVK_MACOS_SURFACE_EXTENSION_NAME in extensions:
            _cocoa_set_wants_layer(view)
            info.sType = vk.VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK
            info.handle = view
            function = 'vkCreateMacOSSurfaceMVK'
        else:
            info.sType = vk.VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT
            info.handle = _cocoa_metal_layer(view)
            function = 'vkCreateMetalSurfaceEXT'
        self._present_surface(info, function, address=ctypes.addressof(info))
        return
    def _create_x11_surface(self):
        library = ctypes.util.find_library('X11')
        if not library:
            raise RuntimeError('libX11 is not installed, so the Vulkan preview cannot attach to the window.')
        x11 = ctypes.cdll.LoadLibrary(library)
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        display = x11.XOpenDisplay(None)
        if not display:
            raise RuntimeError('XOpenDisplay failed. The Vulkan preview needs an X11 display.')
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._native_closer = lambda: x11.XCloseDisplay(display)
        window = int(self.hwnd or 0)
        if vk.VK_KHR_XLIB_SURFACE_EXTENSION_NAME in self._extension_names:
            info = vk.VkXlibSurfaceCreateInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                flags=0, dpy=ffi.cast('struct Display *', display), window=window)
            self._present_surface(info, 'vkCreateXlibSurfaceKHR')
            return
        xcb_library = ctypes.util.find_library('xcb')
        if not xcb_library:
            raise RuntimeError('libxcb is not installed, so the Vulkan preview cannot attach to the window.')
        xcb = ctypes.cdll.LoadLibrary(xcb_library)
        xcb.xcb_connect.restype = ctypes.c_void_p
        xcb.xcb_connect.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        xcb.xcb_disconnect.argtypes = [ctypes.c_void_p]
        connection = xcb.xcb_connect(None, None)
        if not connection:
            raise RuntimeError('xcb_connect failed.')
        previous = self._native_closer
        def _close_both(previous=previous, connection=connection):
            if previous is not None:
                previous()
            xcb.xcb_disconnect(connection)
        self._native_closer = _close_both
        info = vk.VkXcbSurfaceCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
            flags=0, connection=ffi.cast('struct xcb_connection_t *', connection),
            window=window & 0xFFFFFFFF)
        self._present_surface(info, 'vkCreateXcbSurfaceKHR')
        return
    def _device_extension_names(self, present: bool) -> list[str]:
        names = []
        if present:
            names.append(vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME)
        available = set()
        try:
            for prop in vk.vkEnumerateDeviceExtensionProperties(self.physical, None):
                name = prop.extensionName
                if isinstance(name, bytes):
                    name = name.decode()
                available.add(str(name).split('\x00', 1)[0])
        except Exception:
            return names
        portability = vk.VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
        if portability in available:
            names.append(portability)
        return names
    def _create_device(self, present=True):
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if (not devices):
            raise RuntimeError('No Vulkan device is available.')
        self.physical = devices[0]
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical)
        chosen = None
        for index, family in enumerate(families):
            if (not (family.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT)):
                continue
            if present:
                supported = ffi.new('VkBool32*')
                lib.vkGetPhysicalDeviceSurfaceSupportKHR(self.physical, index, self.surface, supported)
                if (not supported[0]):
                    continue
            chosen = index
            break
        if (chosen is None):
            raise RuntimeError('No Vulkan graphics queue can present to this window.')
        self.queue_family = chosen
        priority = self._keep(ffi.new('float[]', [1.0]))
        queue_info = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=chosen, queueCount=1, pQueuePriorities=priority)
        device_extensions = self._device_extension_names(present)
        encoded = [self._keep(ffi.new('char[]', name.encode())) for name in device_extensions]
        listed = self._keep(ffi.new('char*[]', encoded)) if encoded else ffi.NULL
        info = vk.VkDeviceCreateInfo(queueCreateInfoCount=1, pQueueCreateInfos=self._keep(ffi.new('VkDeviceQueueCreateInfo[]', [queue_info])), enabledExtensionCount=len(device_extensions), ppEnabledExtensionNames=listed)
        self.device = vk.vkCreateDevice(self.physical, info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, chosen, 0)
        pool = vk.VkCommandPoolCreateInfo(flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex=chosen)
        self.command_pool = vk.vkCreateCommandPool(self.device, pool, None)
        flight_count = (2 if present else 1)
        alloc = vk.VkCommandBufferAllocateInfo(commandPool=self.command_pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=flight_count)
        buffers = vk.vkAllocateCommandBuffers(self.device, alloc)
        self.command = buffers[0]
        self._flight = []
        self._flight_index = 0
        self._image_slot = {}
        self._widget_size = None
        self._swapchain_dirty = False
        self._retired = None
        self._retired_extra = []
        self._rgba_scratch = None
        if present:
            semaphore_info = vk.VkSemaphoreCreateInfo()
            for command in buffers:
                fence = vk.vkCreateFence(self.device, vk.VkFenceCreateInfo(flags=0), None)
                self._flight.append({'command': command, 'fence': fence, 'fence_list': ffi.new('VkFence[]', [fence]), 'acquire': vk.vkCreateSemaphore(self.device, semaphore_info, None), 'render': vk.vkCreateSemaphore(self.device, semaphore_info, None), 'submitted': False, 'vbo': None, 'vbo_cap': 0, 'vbo_map': None, 'staging': None, 'staging_cap': 0, 'staging_map': None})
        self._vbo = None
        self._vbo_cap = 0
        self._vbo_map = None
        self._staging = None
        self._staging_cap = 0
        self._staging_map = None
        self._source_image = None
        self._pending_source = None
        self._pending_rgba = None
        self._image_layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        return
    def create_shader(self, code):
        info = vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code)
        return vk.vkCreateShaderModule(self.device, info, None)
    def destroy(self):
        if self.device:
            vk.vkDeviceWaitIdle(self.device)
        for pipeline, layout in self.pipelines.values():
            vk.vkDestroyPipeline(self.device, pipeline, None)
            vk.vkDestroyPipelineLayout(self.device, layout, None)
        self.pipelines = {}
        self._destroy_texture()
        if self.sampler:
            vk.vkDestroySampler(self.device, self.sampler, None)
        if self.descriptor_pool:
            vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
        if self.descriptor_layout:
            vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_layout, None)
        self._destroy_swapchain()
        if self.render_pass:
            if self.device:
                vk.vkDestroyRenderPass(self.device, self.render_pass, None)
                self.render_pass = None
        self._destroy_dynamic_buffers()
        for slot in getattr(self, '_flight', []):
            if slot.get('fence'):
                if self.device:
                    vk.vkDestroyFence(self.device, slot['fence'], None)
            for name in ('acquire', 'render'):
                handle = slot.get(name)
                if (not handle):
                    continue
                if (not self.device):
                    continue
                vk.vkDestroySemaphore(self.device, handle, None)
        self._flight = []
        if self.command_pool:
            if self.device:
                vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        if self.device:
            vk.vkDestroyDevice(self.device, None)
        if self.surface:
            if self.instance:
                lib.vkDestroySurfaceKHR(self.instance, self.surface, ffi.NULL)
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)
        closer = getattr(self, '_native_closer', None)
        self._native_closer = None
        if closer is not None:
            try:
                closer()
            except Exception:
                pass
        self.device = None
        self.instance = None
        return
    def _destroy_swapchain(self):
        self._destroy_retired_handles()
        for frame in self.frames:
            vk.vkDestroyFramebuffer(self.device, frame['framebuffer'], None)
            vk.vkDestroyImageView(self.device, frame['view'], None)
        self.frames = []
        if self.swapchain:
            if self.device:
                lib.vkDestroySwapchainKHR(self.device, self.swapchain, ffi.NULL)
                self.swapchain = None
                return
            return
        return
    def _destroy_texture(self):
        if (not self.device):
            return
        if self.texture_view:
            vk.vkDestroyImageView(self.device, self.texture_view, None)
            self.texture_view = None
        if self.texture:
            vk.vkDestroyImage(self.device, self.texture['image'], None)
            vk.vkFreeMemory(self.device, self.texture['memory'], None)
            self.texture = None
        self._texture_size = None
        return
    def _choose_present_mode(self):
        'Mailbox keeps each presented picture for the compositor.\n\nImmediate mode shows it mid-refresh, which reads as stutter. FIFO\nwaits inside the driver on this window and the picture stalls.\n'
        count = ffi.new('uint32_t*', 0)
        lib.vkGetPhysicalDeviceSurfacePresentModesKHR(self.physical, self.surface, count, ffi.NULL)
        if (not count[0]):
            return vk.VK_PRESENT_MODE_FIFO_KHR
        modes = ffi.new('VkPresentModeKHR[]', count[0])
        lib.vkGetPhysicalDeviceSurfacePresentModesKHR(self.physical, self.surface, count, modes)
        available = {modes[index] for index in range(count[0])}
        for mode in (vk.VK_PRESENT_MODE_MAILBOX_KHR, vk.VK_PRESENT_MODE_IMMEDIATE_KHR, vk.VK_PRESENT_MODE_FIFO_KHR):
            if mode in available:
                return mode
        return vk.VK_PRESENT_MODE_FIFO_KHR
    def _retire(self, swapchain, frames):
        """Keep the swapchain that is still on screen until a new frame is presented."""
        if self._retired is not None:
            self._retired_extra.append(self._retired)
        self._retired = (swapchain, frames)
        if len(self._retired_extra) > 2:
            kept = self._retired
            self._retired = None
            self._discard_retired()
            self._retired = kept

    def _destroy_retired_handles(self):
        items = list(getattr(self, '_retired_extra', []))
        retired = getattr(self, '_retired', None)
        if retired is not None:
            items.append(retired)
        self._retired = None
        self._retired_extra = []
        for swapchain, frames in items:
            for frame in frames:
                vk.vkDestroyFramebuffer(self.device, frame['framebuffer'], None)
                vk.vkDestroyImageView(self.device, frame['view'], None)
            if swapchain and self.device:
                lib.vkDestroySwapchainKHR(self.device, swapchain, ffi.NULL)

    def _discard_retired(self):
        if self._retired is None and not self._retired_extra:
            return
        self._wait_gpu()
        self._destroy_retired_handles()

    def _create_swapchain(self, width, height):
        width = max(1, int(width))
        height = max(1, int(height))
        # Leave the current swapchain installed until the new one exists.
        # Destroying it first, on the UI thread, is the black flash while a
        # sash is dragged.
        retired = self.swapchain
        retired_frames = list(self.frames) if retired else []
        caps_ptr = ffi.new('VkSurfaceCapabilitiesKHR*')
        result = lib.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(self.physical, self.surface, caps_ptr)
        if result != vk.VK_SUCCESS:
            if retired:
                return False
            raise RuntimeError(f'vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed ({result})')
        caps = caps_ptr[0]
        extent = caps.currentExtent
        if extent.width == 4294967295 or extent.height == 4294967295:
            extent = vk.VkExtent2D(width=width, height=height)
        if extent.width == 0 or extent.height == 0:
            if retired:
                return False
            raise RuntimeError('Vulkan surface has no size yet.')
        self.extent = (int(extent.width), int(extent.height))
        count = ffi.new('uint32_t*', 0)
        lib.vkGetPhysicalDeviceSurfaceFormatsKHR(self.physical, self.surface, count, ffi.NULL)
        formats = ffi.new('VkSurfaceFormatKHR[]', max(1, count[0]))
        lib.vkGetPhysicalDeviceSurfaceFormatsKHR(self.physical, self.surface, count, formats)
        chosen = formats[0]
        preferred = vk.VK_FORMAT_B8G8R8A8_UNORM
        for index in range(count[0]):
            if formats[index].format == preferred:
                chosen = formats[index]
                break
        self.format = chosen.format
        self._present_mode = self._choose_present_mode()
        image_count = max(int(caps.minImageCount), 3)
        if caps.maxImageCount:
            image_count = min(image_count, int(caps.maxImageCount))
        info = vk.VkSwapchainCreateInfoKHR(surface=self.surface, minImageCount=image_count, imageFormat=chosen.format, imageColorSpace=chosen.colorSpace, imageExtent=vk.VkExtent2D(width=self.extent[0], height=self.extent[1]), imageArrayLayers=1, imageUsage=(vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT), imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE, preTransform=caps.currentTransform, compositeAlpha=_composite_alpha(caps), presentMode=self._present_mode, clipped=1, oldSwapchain=retired if retired else 0)
        swapchain = ffi.new('VkSwapchainKHR*')
        result = lib.vkCreateSwapchainKHR(self.device, ffi.addressof(info), ffi.NULL, swapchain)
        if result != vk.VK_SUCCESS:
            if retired:
                return False
            raise RuntimeError(f'vkCreateSwapchainKHR failed ({result})')
        self._keep(info)
        if self.render_pass is None:
            self._create_render_pass()
        image_count = ffi.new('uint32_t*', 0)
        lib.vkGetSwapchainImagesKHR(self.device, swapchain[0], image_count, ffi.NULL)
        images = ffi.new('VkImage[]', image_count[0])
        lib.vkGetSwapchainImagesKHR(self.device, swapchain[0], image_count, images)
        built = []
        for index in range(image_count[0]):
            view_info = vk.VkImageViewCreateInfo(image=images[index], viewType=vk.VK_IMAGE_VIEW_TYPE_2D, format=self.format, components=vk.VkComponentMapping(r=vk.VK_COMPONENT_SWIZZLE_IDENTITY, g=vk.VK_COMPONENT_SWIZZLE_IDENTITY, b=vk.VK_COMPONENT_SWIZZLE_IDENTITY, a=vk.VK_COMPONENT_SWIZZLE_IDENTITY), subresourceRange=vk.VkImageSubresourceRange(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1, layerCount=1))
            view = vk.vkCreateImageView(self.device, view_info, None)
            attachments = self._keep(ffi.new('VkImageView[]', [view]))
            fb_info = vk.VkFramebufferCreateInfo(renderPass=self.render_pass, attachmentCount=1, pAttachments=attachments, width=self.extent[0], height=self.extent[1], layers=1)
            framebuffer = vk.vkCreateFramebuffer(self.device, fb_info, None)
            built.append({'image': images[index], 'view': view, 'framebuffer': framebuffer})
        self.swapchain = swapchain[0]
        self.frames = built
        self._image_slot = {}
        if retired:
            self._retire(retired, retired_frames)
        return True
    def _create_render_pass(self):
        attachment = vk.VkAttachmentDescription(format=self.format, samples=vk.VK_SAMPLE_COUNT_1_BIT, loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR, storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE, stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE, stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE, initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED, finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        ref = vk.VkAttachmentReference(attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        subpass = vk.VkSubpassDescription(pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS, colorAttachmentCount=1, pColorAttachments=self._keep(ffi.new('VkAttachmentReference[]', [ref])))
        info = vk.VkRenderPassCreateInfo(attachmentCount=1, pAttachments=self._keep(ffi.new('VkAttachmentDescription[]', [attachment])), subpassCount=1, pSubpasses=self._keep(ffi.new('VkSubpassDescription[]', [subpass])))
        self.render_pass = vk.vkCreateRenderPass(self.device, info, None)
        return
    def _create_pipelines(self):
        vertex = self.create_shader(_assemble_vertex())
        texture = self.create_shader(_assemble_texture_fragment())
        color = self.create_shader(_assemble_color_fragment())
        self.pipelines['texture'] = self._make_pipeline(vertex, texture, textured=True)
        self.pipelines['color'] = self._make_pipeline(vertex, color, textured=False)
        vk.vkDestroyShaderModule(self.device, vertex, None)
        vk.vkDestroyShaderModule(self.device, texture, None)
        vk.vkDestroyShaderModule(self.device, color, None)
        self._create_sampler()
        return
    def _make_pipeline(self, vertex, fragment, textured):
        name = self._keep(ffi.new('char[]', b'main'))
        stages = self._keep(ffi.new('VkPipelineShaderStageCreateInfo[2]', [vk.VkPipelineShaderStageCreateInfo(stage=vk.VK_SHADER_STAGE_VERTEX_BIT, module=vertex, pName=name), vk.VkPipelineShaderStageCreateInfo(stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT, module=fragment, pName=name)]))
        binding = vk.VkVertexInputBindingDescription(binding=0, stride=16, inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX)
        attributes = [vk.VkVertexInputAttributeDescription(location=0, binding=0, format=vk.VK_FORMAT_R32G32_SFLOAT, offset=0), vk.VkVertexInputAttributeDescription(location=1, binding=0, format=vk.VK_FORMAT_R32G32_SFLOAT, offset=8)]
        vertex_input = vk.VkPipelineVertexInputStateCreateInfo(vertexBindingDescriptionCount=1, pVertexBindingDescriptions=self._keep(ffi.new('VkVertexInputBindingDescription[]', [binding])), vertexAttributeDescriptionCount=len(attributes), pVertexAttributeDescriptions=self._keep(ffi.new('VkVertexInputAttributeDescription[]', attributes)))
        assembly = vk.VkPipelineInputAssemblyStateCreateInfo(topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        viewport = vk.VkPipelineViewportStateCreateInfo(viewportCount=1, scissorCount=1)
        raster = vk.VkPipelineRasterizationStateCreateInfo(polygonMode=vk.VK_POLYGON_MODE_FILL, cullMode=vk.VK_CULL_MODE_NONE, frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE, lineWidth=1.0)
        multisample = vk.VkPipelineMultisampleStateCreateInfo(rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT)
        blend = vk.VkPipelineColorBlendAttachmentState(blendEnable=1, srcColorBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA, dstColorBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, colorBlendOp=vk.VK_BLEND_OP_ADD, srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE, dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, alphaBlendOp=vk.VK_BLEND_OP_ADD, colorWriteMask=15)
        blending = vk.VkPipelineColorBlendStateCreateInfo(attachmentCount=1, pAttachments=self._keep(ffi.new('VkPipelineColorBlendAttachmentState[]', [blend])))
        dynamic = vk.VkPipelineDynamicStateCreateInfo(dynamicStateCount=2, pDynamicStates=self._keep(ffi.new('VkDynamicState[]', [vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR])))
        if textured:
            binding_info = vk.VkDescriptorSetLayoutBinding(binding=0, descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT)
            layout_info = vk.VkDescriptorSetLayoutCreateInfo(bindingCount=1, pBindings=self._keep(ffi.new('VkDescriptorSetLayoutBinding[]', [binding_info])))
            set_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)
            self.descriptor_layout = set_layout
            layouts = self._keep(ffi.new('VkDescriptorSetLayout[]', [set_layout]))
            pipe_layout_info = vk.VkPipelineLayoutCreateInfo(setLayoutCount=1, pSetLayouts=layouts)
        else:
            push = vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT, offset=0, size=16)
            pipe_layout_info = vk.VkPipelineLayoutCreateInfo(pushConstantRangeCount=1, pPushConstantRanges=self._keep(ffi.new('VkPushConstantRange[]', [push])))
        pipe_layout = vk.vkCreatePipelineLayout(self.device, pipe_layout_info, None)
        created = vk.VkGraphicsPipelineCreateInfo(stageCount=2, pStages=stages, pVertexInputState=vertex_input, pInputAssemblyState=assembly, pViewportState=viewport, pRasterizationState=raster, pMultisampleState=multisample, pColorBlendState=blending, pDynamicState=dynamic, layout=pipe_layout, renderPass=self.render_pass)
        pipeline = vk.vkCreateGraphicsPipelines(self.device, ffi.NULL, 1, self._keep(ffi.new('VkGraphicsPipelineCreateInfo[]', [created])), None)[0]
        return (pipeline, pipe_layout)
    def _create_sampler(self):
        info = vk.VkSamplerCreateInfo(magFilter=vk.VK_FILTER_LINEAR, minFilter=vk.VK_FILTER_LINEAR, addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
        self.sampler = vk.vkCreateSampler(self.device, info, None)
        pool = vk.VkDescriptorPoolCreateInfo(maxSets=1, poolSizeCount=1, pPoolSizes=self._keep(ffi.new('VkDescriptorPoolSize[]', [vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=1)])))
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool, None)
        alloc = vk.VkDescriptorSetAllocateInfo(descriptorPool=self.descriptor_pool, descriptorSetCount=1, pSetLayouts=self._keep(ffi.new('VkDescriptorSetLayout[]', [self.descriptor_layout])))
        self.descriptor = vk.vkAllocateDescriptorSets(self.device, alloc)[0]
        return
    def _memory_type(self, bits, flags, prefer=0):
        props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical)
        fallback = None
        for index in range(props.memoryTypeCount):
            if (not (bits & (1 << index))):
                continue
            have = props.memoryTypes[index].propertyFlags
            if ((have & flags) != flags):
                continue
            if prefer and (have & prefer) == prefer:
                return index
            if fallback is None:
                fallback = index
        if (fallback is None):
            raise RuntimeError('No matching Vulkan memory type.')
        return fallback
    def _destroy_dynamic_buffers(self):
        if (not self.device):
            return
        for slot in getattr(self, '_flight', []):
            for kind in ('vbo', 'staging'):
                mapped = slot.get((kind + '_map'))
                current = slot.get(kind)
                if (mapped is not None):
                    if (current is not None):
                        vk.vkUnmapMemory(self.device, current['memory'])
                        slot[(kind + '_map')] = None
                if (current is None):
                    continue
                vk.vkDestroyBuffer(self.device, current['buffer'], None)
                vk.vkFreeMemory(self.device, current['memory'], None)
                slot[kind] = None
                slot[(kind + '_cap')] = 0
        vbo = getattr(self, '_vbo', None)
        if (getattr(self, '_vbo_map', None) is not None):
            if (vbo is not None):
                vk.vkUnmapMemory(self.device, vbo['memory'])
                self._vbo_map = None
        if (vbo is not None):
            vk.vkDestroyBuffer(self.device, vbo['buffer'], None)
            vk.vkFreeMemory(self.device, vbo['memory'], None)
            self._vbo = None
            self._vbo_cap = 0
        staging = getattr(self, '_staging', None)
        if (getattr(self, '_staging_map', None) is not None):
            if (staging is not None):
                vk.vkUnmapMemory(self.device, staging['memory'])
                self._staging_map = None
        if (staging is not None):
            vk.vkDestroyBuffer(self.device, staging['buffer'], None)
            vk.vkFreeMemory(self.device, staging['memory'], None)
            self._staging = None
            self._staging_cap = 0
            return
        return
    def _wait_slot(self, slot, block=True):
        if (not slot['submitted']):
            return True
        timeout = (18446744073709551615 if block else 0)
        try:
            vk.vkWaitForFences(self.device, 1, slot['fence_list'], vk.VK_TRUE, timeout)
        except vk.VkTimeout:
            return False
        vk.vkResetFences(self.device, 1, slot['fence_list'])
        slot['submitted'] = False
        return True
    def _wait_gpu(self, block=True):
        'Wait until every submitted frame has finished.\n\nUsed when a texture or other shared resource has to be rebuilt. Playback\nitself waits only on the older in-flight frame, which is already done.\n'
        ready = True
        for slot in getattr(self, '_flight', []):
            if not self._wait_slot(slot, block=block):
                ready = False
                if not block:
                    return False
        return ready
    def _mapped_buffer(self, kind, size):
        'A host-visible buffer kept mapped across frames.'
        attr = ('_vbo' if (kind == 'vertex') else '_staging')
        usage = (vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT if (kind == 'vertex') else vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        current = getattr(self, attr)
        cap = getattr(self, (attr + '_cap'))
        if (current is not None):
            if (cap >= size):
                return (current, getattr(self, (attr + '_map')))
        self._wait_gpu()
        if (current is not None):
            vk.vkUnmapMemory(self.device, current['memory'])
            vk.vkDestroyBuffer(self.device, current['buffer'], None)
            vk.vkFreeMemory(self.device, current['memory'], None)
        cap = max(size, (262144 if (kind == 'vertex') else size))
        created = self._buffer(cap, usage, host=True)
        mapped = vk.vkMapMemory(self.device, created['memory'], 0, cap, 0)
        setattr(self, attr, created)
        setattr(self, (attr + '_cap'), cap)
        setattr(self, (attr + '_map'), mapped)
        return (created, mapped)
    def _prepare_image(self, image):
        """Upload only when the viewer hands over a new frame."""
        if image is None or image is self._source_image:
            return False
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
            self._wait_gpu()
            self._destroy_texture()
            self._allocate_texture(width, height)
            self._image_layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        self._pending_rgba = rgba
        self._pending_source = image
        return True
    def _allocate_texture(self, width, height):
        info = vk.VkImageCreateInfo(imageType=vk.VK_IMAGE_TYPE_2D, format=vk.VK_FORMAT_R8G8B8A8_UNORM, extent=vk.VkExtent3D(width=width, height=height, depth=1), mipLevels=1, arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT, tiling=vk.VK_IMAGE_TILING_OPTIMAL, usage=(vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT | vk.VK_IMAGE_USAGE_SAMPLED_BIT), sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE, initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED)
        image = vk.vkCreateImage(self.device, info, None)
        requirements = vk.vkGetImageMemoryRequirements(self.device, image)
        memory = vk.vkAllocateMemory(self.device, vk.VkMemoryAllocateInfo(allocationSize=requirements.size, memoryTypeIndex=self._memory_type(requirements.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)), None)
        vk.vkBindImageMemory(self.device, image, memory, 0)
        view = vk.vkCreateImageView(self.device, vk.VkImageViewCreateInfo(image=image, viewType=vk.VK_IMAGE_VIEW_TYPE_2D, format=vk.VK_FORMAT_R8G8B8A8_UNORM, components=vk.VkComponentMapping(r=vk.VK_COMPONENT_SWIZZLE_IDENTITY, g=vk.VK_COMPONENT_SWIZZLE_IDENTITY, b=vk.VK_COMPONENT_SWIZZLE_IDENTITY, a=vk.VK_COMPONENT_SWIZZLE_IDENTITY), subresourceRange=vk.VkImageSubresourceRange(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1, layerCount=1)), None)
        self.texture = {'image': image, 'memory': memory}
        self.texture_view = view
        self._texture_size = (width, height)
        image_info = vk.VkDescriptorImageInfo(sampler=self.sampler, imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        write = vk.VkWriteDescriptorSet(dstSet=self.descriptor, dstBinding=0, descriptorCount=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, pImageInfo=self._keep(ffi.new('VkDescriptorImageInfo[]', [image_info])))
        vk.vkUpdateDescriptorSets(self.device, 1, self._keep(ffi.new('VkWriteDescriptorSet[]', [write])), 0, ffi.NULL)
        return
    def _record_upload(self, cmd, staging_buffer):
        rgba = self._pending_rgba
        height = rgba.shape[slice(None, 2, None)][0]
        width = rgba.shape[slice(None, 2, None)][1]
        old = self._image_layout
        src_access = (vk.VK_ACCESS_SHADER_READ_BIT if (old == vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) else 0)
        self._barrier(cmd, self.texture['image'], old, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, src_access, vk.VK_ACCESS_TRANSFER_WRITE_BIT, vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, vk.VK_PIPELINE_STAGE_TRANSFER_BIT)
        region = vk.VkBufferImageCopy(imageSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1), imageExtent=vk.VkExtent3D(width=width, height=height, depth=1))
        vk.vkCmdCopyBufferToImage(cmd, staging_buffer, self.texture['image'], vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, ffi.new('VkBufferImageCopy[]', [region]))
        self._barrier(cmd, self.texture['image'], vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, vk.VK_ACCESS_TRANSFER_WRITE_BIT, vk.VK_ACCESS_SHADER_READ_BIT, vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
        self._image_layout = vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        self._pending_rgba = None
        return
    def _barrier(self, cmd, image, old, new, src_access, dst_access, src_stage, dst_stage):
        barrier = vk.VkImageMemoryBarrier(srcAccessMask=src_access, dstAccessMask=dst_access, oldLayout=old, newLayout=new, srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED, dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED, image=image, subresourceRange=vk.VkImageSubresourceRange(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1, layerCount=1))
        vk.vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, ffi.NULL, 0, ffi.NULL, 1, ffi.new('VkImageMemoryBarrier[]', [barrier]))
        return
    def _buffer(self, size, usage, host=False):
        info = vk.VkBufferCreateInfo(size=size, usage=usage, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE)
        buffer = vk.vkCreateBuffer(self.device, info, None)
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        if host:
            flags = (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            prefer = vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        else:
            flags = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            prefer = 0
        memory = vk.vkAllocateMemory(self.device, vk.VkMemoryAllocateInfo(allocationSize=requirements.size, memoryTypeIndex=self._memory_type(requirements.memoryTypeBits, flags, prefer)), None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        return {'buffer': buffer, 'memory': memory}
    def _once(self, record):
        begin = vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vk.vkBeginCommandBuffer(self.command, begin)
        record(self.command)
        vk.vkEndCommandBuffer(self.command)
        buffers = self._keep(ffi.new('VkCommandBuffer[]', [self.command]))
        submit = vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=buffers)
        vk.vkQueueSubmit(self.queue, 1, self._keep(ffi.new('VkSubmitInfo[]', [submit])), ffi.NULL)
        vk.vkQueueWaitIdle(self.queue)
        return
    def _slot_buffer(self, slot, kind, size):
        'Host buffer owned by one in-flight frame, kept mapped.'
        usage = (vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT if (kind == 'vbo') else vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        current = slot[kind]
        if (current is not None):
            if (slot[(kind + '_cap')] >= size):
                return (current, slot[(kind + '_map')])
        if (current is not None):
            vk.vkUnmapMemory(self.device, current['memory'])
            vk.vkDestroyBuffer(self.device, current['buffer'], None)
            vk.vkFreeMemory(self.device, current['memory'], None)
        cap = max(size, (65536 if (kind == 'vbo') else size))
        created = self._buffer(cap, usage, host=True)
        mapped = vk.vkMapMemory(self.device, created['memory'], 0, cap, 0)
        slot[kind] = created
        slot[(kind + '_cap')] = cap
        slot[(kind + '_map')] = mapped
        return (created, mapped)
    def draw(self, width, height, image, picture, overlays):
        'Present one frame. Returns False when nothing reached the screen.'
        if (not self._flight):
            return False
        if (not self._refresh_extent(width, height)):
            return False
        slot = self._flight[self._flight_index]
        if (not self._wait_slot(slot, block=True)):
            return False
        upload = self._prepare_image(image)
        staging = None
        if upload:
            rgba = self._pending_rgba
            nbytes = int(rgba.nbytes)
            staging, mapped = self._slot_buffer(slot, 'staging', nbytes)
            ffi.memmove(mapped, ffi.from_buffer(rgba), nbytes)
        draws, blob = self._pack_vertices(picture, overlays)
        vbo = None
        if blob:
            vbo, mapped = self._slot_buffer(slot, 'vbo', len(blob))
            mapped[:len(blob)] = blob
        index = ffi.new('uint32_t*')
        try:
            result = lib.vkAcquireNextImageKHR(self.device, self.swapchain, ffi.cast('uint64_t', 2000000), slot['acquire'], ffi.NULL, index)
        except vk.VkTimeout:
            return False
        if (result in (vk.VK_TIMEOUT, vk.VK_NOT_READY)):
            return False
        if (result not in (vk.VK_SUCCESS, vk.VK_SUBOPTIMAL_KHR)):
            self._swapchain_dirty = True
            self._create_swapchain(width, height)
            return False
        image_index = index[0]
        previous = self._image_slot.get(image_index)
        if (previous is not None):
            if (previous is not slot):
                if previous['submitted']:
                    self._wait_slot(previous, block=True)
        self._image_slot[image_index] = slot
        self.command = slot['command']
        begin = vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vk.vkBeginCommandBuffer(self.command, begin)
        if upload:
            self._record_upload(self.command, staging['buffer'])
        clear = ffi.new('VkClearValue*')
        area = vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=vk.VkExtent2D(width=self.extent[0], height=self.extent[1]))
        rp = vk.VkRenderPassBeginInfo(renderPass=self.render_pass, framebuffer=self.frames[image_index]['framebuffer'], renderArea=area, clearValueCount=1, pClearValues=clear)
        vk.vkCmdBeginRenderPass(self.command, rp, vk.VK_SUBPASS_CONTENTS_INLINE)
        viewport = vk.VkViewport(width=float(self.extent[0]), height=float(self.extent[1]), maxDepth=1.0)
        vk.vkCmdSetViewport(self.command, 0, 1, ffi.new('VkViewport[]', [viewport]))
        scissor = vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=vk.VkExtent2D(width=self.extent[0], height=self.extent[1]))
        vk.vkCmdSetScissor(self.command, 0, 1, ffi.new('VkRect2D[]', [scissor]))
        if (vbo is not None):
            offset = ffi.new('VkDeviceSize[]', [0])
            vk.vkCmdBindVertexBuffers(self.command, 0, 1, ffi.new('VkBuffer[]', [vbo['buffer']]), offset)
            for kind, first, count, color in draws:
                self._draw_range(kind, first, count, color)
        vk.vkCmdEndRenderPass(self.command)
        vk.vkEndCommandBuffer(self.command)
        buffers = ffi.new('VkCommandBuffer[]', [self.command])
        wait_sem = ffi.new('VkSemaphore[]', [slot['acquire']])
        signal_sem = ffi.new('VkSemaphore[]', [slot['render']])
        wait_stage = ffi.new('VkPipelineStageFlags[]', [vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT])
        submit = vk.VkSubmitInfo(waitSemaphoreCount=1, pWaitSemaphores=wait_sem, pWaitDstStageMask=wait_stage, commandBufferCount=1, pCommandBuffers=buffers, signalSemaphoreCount=1, pSignalSemaphores=signal_sem)
        vk.vkQueueSubmit(self.queue, 1, ffi.new('VkSubmitInfo[]', [submit]), slot['fence'])
        slot['submitted'] = True
        present = vk.VkPresentInfoKHR(waitSemaphoreCount=1, pWaitSemaphores=signal_sem, swapchainCount=1, pSwapchains=ffi.new('VkSwapchainKHR[]', [self.swapchain]), pImageIndices=index)
        present_result = lib.vkQueuePresentKHR(self.queue, ffi.addressof(present))
        self._flight_index = ((self._flight_index + 1) % len(self._flight))
        if upload:
            self._source_image = self._pending_source
        if present_result == vk.VK_ERROR_OUT_OF_DATE_KHR:
            self._swapchain_dirty = True
            return False
        # Suboptimal still put the picture on screen. Recreate on the next draw.
        if present_result == vk.VK_SUBOPTIMAL_KHR:
            self._swapchain_dirty = True
        elif present_result != vk.VK_SUCCESS:
            self._swapchain_dirty = True
            return False
        self._discard_retired()
        return True
    def _refresh_extent(self, width, height):
        desired = (max(1, int(width)), max(1, int(height)))
        if (self.swapchain and not self._swapchain_dirty
                and desired == self.extent and self.extent[0] and self.extent[1]):
            self._widget_size = desired
            return True
        self._widget_size = desired
        caps_ptr = ffi.new('VkSurfaceCapabilitiesKHR*')
        result = lib.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            self.physical, self.surface, caps_ptr)
        if result != vk.VK_SUCCESS:
            return False
        caps = caps_ptr[0]
        if caps.currentExtent.width in (0, 0xFFFFFFFF):
            extent = desired
        else:
            extent = (caps.currentExtent.width, caps.currentExtent.height)
        if extent[0] == 0 or extent[1] == 0:
            return False
        if self._swapchain_dirty or extent != self.extent or not self.swapchain:
            if not self._create_swapchain(*extent):
                return False
            self._swapchain_dirty = False
        return True
    def _pack_vertices(self, picture, overlays):
        chunks = []
        draws = []
        cursor = 0
        if picture:
            array = np.asarray(picture, dtype=np.float32)
            chunks.append(array)
            draws.append(('texture', cursor, len(array), None))
            cursor = (cursor + len(array))
        for corners, color in overlays:
            array = np.asarray(corners, dtype=np.float32)
            chunks.append(array)
            draws.append(('color', cursor, len(array), color))
            cursor = (cursor + len(array))
        if (not chunks):
            return ([], b'')
        return (draws, np.concatenate(chunks).tobytes())
    def _draw_range(self, kind, first, count, color):
        pipeline = self.pipelines[('texture' if (kind == 'texture') else 'color')][0]
        layout = self.pipelines[('texture' if (kind == 'texture') else 'color')][1]
        vk.vkCmdBindPipeline(self.command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline)
        if kind == 'texture':
            vk.vkCmdBindDescriptorSets(self.command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, ffi.new('VkDescriptorSet[]', [self.descriptor]), 0, ffi.NULL)
        else:
            vk.vkCmdPushConstants(self.command, layout, vk.VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, ffi.new('float[]', list(color)))
        vk.vkCmdDraw(self.command, count, 1, first, 0)
        return
    def _begin_frame_trash(self):
        self._trash = []
        return
    def _end_frame_trash(self):
        for item in getattr(self, '_trash', []):
            vk.vkDestroyBuffer(self.device, item['buffer'], None)
            vk.vkFreeMemory(self.device, item['memory'], None)
        self._trash = []
        return
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
class ImageViewerVulkan(tk.Frame):
    '\nPreview pane drawn with Vulkan. This is the console preview when the\n``console_ui_vulkan`` extra is installed.\n\nStill images, and finished animations (GIF, WebP, APNG, and MP4, including\naudio) with a timeline. The speaker draws sound waves, or a red X when\nmuted, beside the volume slider. The picture stays above the\ncontrols. ``DGENERATE_CONSOLE_UI_VULKAN=0`` selects the OpenGL viewer instead.\n'
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
        self.bind('<Control-equal>', lambda e: self._zoom_by_factor(self._zoom_step))
        self.bind('<Control-minus>', lambda e: self._zoom_by_factor((1 / self._zoom_step)))
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
            windowing = str(self.tk.call('tk', 'windowingsystem'))
            hinstance = None
            if windowing == 'win32':
                hinstance = ctypes.windll.kernel32.GetModuleHandleW(None)
            self._gpu = _VulkanDevice(
                hwnd=self.winfo_id(), hinstance=hinstance, windowing=windowing)
            return True
        except Exception as error:
            self._gpu_error = str(error)
            if self.on_error:
                self.on_error(f'Vulkan preview: {error}')
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
        if not self._can_present() or not self._ensure_gpu() or not self.has_image():
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
            presented = self._gpu.draw(width, height, self._original_image_array, picture, overlays)
        except Exception as error:
            if self.on_error:
                self.on_error(f'Vulkan preview: {error}')
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
                self.on_error(f'Vulkan preview: {error}')

    def _on_map(self, _event):
        # The present from before this pane was hidden does not survive being unmapped.
        self._presented_key = None
        if not self._finish_pending_load():
            self._show_current_picture()
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
        if (event.state & 4):
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
        zoom_in = getattr(event, 'delta', 0) > 0 or getattr(event, 'num', 0) == 4
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
                self.on_error(f'Vulkan preview: {error}')
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
        _helpdialog.show_help_dialog(title='Vulkan preview help', help_text='\n'.join([*('Vulkan preview controls:', '• Mouse wheel: zoom', '• Ctrl+drag or middle-drag: pan', '• Move the pointer over an animation for the timeline', '• Space: pause and resume', '• Loop arrow: replay the clip, or stop at the end when slashed', '• Speaker: mute (red X) or unmute; slider sets volume. Both are remembered')]), parent=self.master, size=(460, 280), position_widget=self.master, dock_to_right=False)
        return
    def cleanup(self):
        self._stop_animation()
        if (self._gpu is not None):
            self._gpu.destroy()
            self._gpu = None
            return
        return
