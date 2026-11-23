# Copyright (c) 2026, Teriks
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

"""Themed window frame for Windows and Linux.

The window keeps a real frame so the operating system resizes it.
The stock title bar is removed and this module paints one in the client area.
The painted title does not start a move by itself. On Windows the click is
handed back to the system so the frame and its contents move together.
On Windows the left, right, and bottom borders stay the native resize border.
On Linux the window manager owns the drag. macOS keeps the system frame and
tints its title bar from the window color.
Menus and tooltips set ``overrideredirect`` themselves and stay frameless.
"""

import ctypes
import io
import platform
import sys
import time as _time
import tkinter as tk
import tkinter.font as tkfont

import dgenerate.console.themedialog as _themedialog

_user32 = None
_comctl32 = None
_comctl32_dispatch = None
_dwmapi = None
if platform.system() == 'Windows':
    class _LockedWinDLL(ctypes.WinDLL):
        """Windows calls that keep the interpreter lock.

        The window procedure runs while Tk has released the lock. A nested call
        that releases it again re-enters this procedure with no thread state.
        Queries stay on this loader. ``DefSubclassProc`` is the exception: it has
        to release the lock so Tk can take it back while Windows moves the window.
        """

        _func_flags_ = ctypes.WinDLL._func_flags_ | ctypes._FUNCFLAG_PYTHONAPI

    _user32 = _LockedWinDLL('user32')
    _comctl32 = _LockedWinDLL('comctl32')
    # Releasing the lock here is required. Tk saved its thread state before
    # the event loop and restores that pointer from inside the default
    # procedure. Holding the lock across that call aborts the interpreter.
    _comctl32_dispatch = ctypes.WinDLL('comctl32')
    _dispatch_args = (
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_ssize_t, ctypes.c_ssize_t)
    _comctl32_dispatch.DefSubclassProc.restype = ctypes.c_ssize_t
    _comctl32_dispatch.DefSubclassProc.argtypes = _dispatch_args
    _comctl32.DefSubclassProc.restype = ctypes.c_ssize_t
    _comctl32.DefSubclassProc.argtypes = _dispatch_args
    _dwmapi = _LockedWinDLL('dwmapi')

_CUSTOM = platform.system() in ('Windows', 'Linux')
_installed = []
_pending = None
_patched = False

_CLOSE_HOVER = '#c42b1c'
_CLOSE_HOVER_TEXT = '#ffffff'
# Segoe Fluent Icons, the same marks Windows draws in a title bar. MDL2 is the
# older set of the same four characters. 10px matches the smaller caption marks
# used by other themed windows. The button itself stays 46px wide.
_CAPTION_GLYPH = {
    'min': '\uE921',
    'max': '\uE922',
    'restore': '\uE923',
    'close': '\uE8BB',
}
_CAPTION_FAMILIES = ('Segoe Fluent Icons', 'Segoe MDL2 Assets')
_CAPTION_PX = 10
_CAPTION_BUTTON = 46
_caption_family_name = None
_caption_family_ready = False

# Windows styles. A normal window already has a taskbar button. Dialogs are
# owned tool windows so they do not each get one while they are open.
# Ownership is released for a minimize: Windows parks an owned window at
# the lower left of the monitor instead of the taskbar.
_GWL_STYLE = -16
_GWL_EXSTYLE = -20
_WS_MINIMIZEBOX = 0x00020000
_WS_MAXIMIZEBOX = 0x00010000
_WS_SYSMENU = 0x00080000
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_APPWINDOW = 0x00040000
_SWP_FRAMECHANGED = 0x0020
_SWP_NOREDRAW = 0x0008
_SWP_NOMOVE_NOSIZE_NOZORDER_NOACTIVATE = 0x0013
_WM_SETREDRAW = 0x000B
_WM_SYSCOMMAND = 0x0112
_SC_MINIMIZE = 0xF020
_SC_MAXIMIZE = 0xF030
_SC_RESTORE = 0xF120
_SC_MOVE = 0xF010
_SC_SIZE = 0xF000
# WM_SYSCOMMAND keeps the sizing edge in the low four bits of SC_SIZE.
# Those bits are WMSZ_* values. HT* codes do not fit: HTBOTTOMLEFT is 16,
# and SC_SIZE combined with it is SC_MOVE.
_WMSZ_LEFT = 1
_WMSZ_RIGHT = 2
_WMSZ_TOP = 3
_WMSZ_TOPLEFT = 4
_WMSZ_TOPRIGHT = 5
_WMSZ_BOTTOM = 6
_WMSZ_BOTTOMLEFT = 7
_WMSZ_BOTTOMRIGHT = 8
_SM_REMOTESESSION = 0x1000
_SPI_GETWORKAREA = 0x0030
_WS_THICKFRAME = 0x00040000
_WS_CHILD = 0x40000000
_WM_NCHITTEST = 0x0084
_WM_NCDESTROY = 0x0082
_WM_NCCALCSIZE = 0x0083
_WM_NCLBUTTONDBLCLK = 0x00A3
_WM_NCLBUTTONDOWN = 0x00A1
_WM_SIZE = 0x0005
_WM_WINDOWPOSCHANGED = 0x0047
_SIZE_MINIMIZED = 1
_SIZE_MAXIMIZED = 2
_GWLP_HWNDPARENT = -8
_WM_GETMINMAXINFO = 0x0024
# Past the monitor. Windows otherwise stops the right edge at the primary
# screen's tracking size while the left edge can still move.
_TRACK_LIMIT = 32000
_SM_CYCAPTION = 4
_SM_CXFRAME = 32
_SM_CXPADDEDBORDER = 92
# A size change that must not raise or focus the window.
_SWP_NOZORDER_NOACTIVATE = 0x0014
_DWMWA_WINDOW_CORNER_PREFERENCE = 33
_DWMWA_BORDER_COLOR = 34
_DWMWCP_ROUND = 2
_HTNOWHERE = 0
_HTCLIENT = 1
_HTCAPTION = 2
_HTTRANSPARENT = -1
_HTLEFT = 10
_HTRIGHT = 11
_HTTOP = 12
_HTTOPLEFT = 13
_HTTOPRIGHT = 14
_HTBOTTOM = 15
_HTBOTTOMLEFT = 16
_HTBOTTOMRIGHT = 17

# hwnd -> hook. The callback stays referenced here so it is not collected.
_size_hooks = {}
_size_callback = None
_drag_posted = {}


class _Rect(ctypes.Structure):
    _fields_ = [
        ('left', ctypes.c_long),
        ('top', ctypes.c_long),
        ('right', ctypes.c_long),
        ('bottom', ctypes.c_long),
    ]


class _NcCalcSizeParams(ctypes.Structure):
    _fields_ = [
        ('rgrc', _Rect * 3),
        ('lppos', ctypes.c_void_p),
    ]


_macos_windows = []
_macos_dialogs = []
_macos_link_job = None
_macos_colors = None
_macos_patched = False
_objc_lib = None


class _MacDrawable(ctypes.Structure):
    """Leading fields of Tk's Aqua drawable. ``winfo_id`` is this pointer."""

    _fields_ = [
        ('winPtr', ctypes.c_void_p),
        ('view', ctypes.c_void_p),
    ]


def _macos_objc():
    global _objc_lib
    if _objc_lib is None:
        ctypes.cdll.LoadLibrary('/System/Library/Frameworks/AppKit.framework/AppKit')
        ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
        lib = ctypes.cdll.LoadLibrary('/usr/lib/libobjc.A.dylib')
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]
        lib.objc_getClass.restype = ctypes.c_void_p
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        _objc_lib = lib
    return _objc_lib


def _macos_msg(restype, *argtypes):
    proto = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)
    return proto(('objc_msgSend', _macos_objc()))


def _macos_sel(name):
    return _macos_objc().sel_registerName(name)


def _macos_ns_window(window):
    """The NSWindow for a Tk toplevel, once Aqua has created its view."""
    try:
        drawable = int(window.winfo_id() or 0)
    except (tk.TclError, TypeError, ValueError):
        return None
    if not drawable:
        return None
    view = int(_MacDrawable.from_address(drawable).view or 0)
    if not view:
        return None
    return _macos_msg(ctypes.c_void_p)(view, _macos_sel(b'window'))


def _macos_patch_toplevels():
    global _macos_patched
    if _macos_patched:
        return
    _macos_patched = True
    original = tk.Toplevel.__init__

    def toplevel_init(self, master=None, cnf={}, **kw):
        original(self, master, cnf, **kw)
        _macos_watch(self)

    tk.Toplevel.__init__ = toplevel_init


def _macos_watch(window):
    if window is None:
        return
    if window not in _macos_windows:
        _macos_windows.append(window)
    if getattr(window, '_macos_watched', False):
        _macos_titlebar(window)
        return
    window._macos_watched = True

    def on_map(event, window=window):
        if event.widget is window and window in _macos_windows:
            _macos_titlebar(window)
            _macos_add_dialog(window)

    window.bind('<Map>', on_map, add='+')
    try:
        if int(window.winfo_ismapped()):
            _macos_titlebar(window)
    except tk.TclError:
        pass


def _macos_owner(window):
    """The transient master, or the opener's toplevel as Windows owns it."""
    try:
        path = str(window.tk.call('wm', 'transient', window._w))
        if path:
            return window.nametowidget(path)
        if window.master is None:
            return None
        owner = window.master.winfo_toplevel()
        return None if owner is window else owner
    except (tk.TclError, KeyError):
        return None


def _macos_menu_open(widget):
    """A menu popup holds the grab. Relinking now reorders windows under it."""
    try:
        for path in widget.tk.splitlist(widget.tk.call('grab', 'current')):
            top = widget.tk.call('winfo', 'toplevel', path)
            if int(widget.tk.call('wm', 'overrideredirect', top)):
                return True
    except (tk.TclError, ValueError):
        return False
    return False


def _macos_add_dialog(window):
    """Keep a dialog an AppKit child window of its owner, except to drag it.

    A child window always stays above its owner, as an owned window does on
    Windows. The window server cannot drag a child window by itself, so the
    app moves it on every mouse event and the drag lags. The output window
    is not a child, and its drag is smooth. The dialog is detached while the
    pointer is over its title bar or resize border, before a press there
    can start a drag. Menus and tooltips are overrideredirect and skipped.
    """
    try:
        if not window.winfo_exists() or _app_window(window):
            return
        if int(window.tk.call('wm', 'overrideredirect', window._w)):
            return
    except (tk.TclError, ValueError):
        return
    owner = _macos_owner(window)
    if owner is None:
        return
    window._macos_owner = owner
    if window not in _macos_dialogs:
        _macos_dialogs.append(window)
    if not getattr(window, '_macos_raises', False):
        window._macos_raises = True

        def on_activate(event, window=window):
            if event.widget is window:
                _macos_raise_child(window)

        window.bind('<Activate>', on_activate, add='+')
    _macos_arm_dialog_links(window)


def _macos_ordered_windows():
    """This app's windows, front first."""
    app = _macos_msg(ctypes.c_void_p)(
        _macos_objc().objc_getClass(b'NSApplication'), _macos_sel(b'sharedApplication'))
    windows = _macos_msg(ctypes.c_void_p)(app, _macos_sel(b'orderedWindows'))
    count = int(_macos_msg(ctypes.c_ulong)(windows, _macos_sel(b'count')))
    at = _macos_msg(ctypes.c_void_p, ctypes.c_ulong)
    return [at(windows, _macos_sel(b'objectAtIndex:'), index) for index in range(count)]


def _macos_children_in_front(ns_window, parent):
    """Other child windows of ``parent`` stacked above ``ns_window``, front first."""
    try:
        parent_of = _macos_msg(ctypes.c_void_p)
        front = []
        for window in _macos_ordered_windows():
            if window == ns_window:
                return front
            if window and parent_of(window, _macos_sel(b'parentWindow')) == parent:
                front.append(window)
    except (OSError, AttributeError):
        pass
    return []


def _macos_unlink_children(parent, ns_windows):
    """Remove these child windows of ``parent`` and keep their stacking.

    AppKit orders a removed child behind its old parent, and keeps the
    children that remain directly above the parent. A single dialog cannot
    be detached under a sibling, so the owner's dialogs are detached
    together and stacked again above it in the order they had.
    """
    parent_of = _macos_msg(ctypes.c_void_p)
    linked = [w for w in ns_windows if parent_of(w, _macos_sel(b'parentWindow')) == parent]
    if not linked:
        return
    try:
        stack = [w for w in _macos_ordered_windows() if w in ns_windows]
    except (OSError, AttributeError):
        stack = list(ns_windows)
    unlink = _macos_msg(None, ctypes.c_void_p)
    for window in linked:
        unlink(parent, _macos_sel(b'removeChildWindow:'), window)
    order = _macos_msg(None, ctypes.c_long, ctypes.c_long)
    number = _macos_msg(ctypes.c_long)
    below = parent
    for window in reversed(stack):
        # NSWindowAbove.
        order(window, _macos_sel(b'orderWindow:relativeTo:'), 1,
              int(number(below, _macos_sel(b'windowNumber'))))
        below = window


def _macos_link_children(parent, ns_windows):
    """Make these windows children of ``parent`` in the order they are stacked.

    AppKit puts each added child above the others, so they are added from
    the back. A dialog behind another one stays behind it.
    """
    try:
        stack = [w for w in _macos_ordered_windows() if w in ns_windows]
    except (OSError, AttributeError):
        return
    parent_of = _macos_msg(ctypes.c_void_p)
    unlink = _macos_msg(None, ctypes.c_void_p)
    link = _macos_msg(None, ctypes.c_void_p, ctypes.c_long)
    for window in stack:
        if parent_of(window, _macos_sel(b'parentWindow')) == parent:
            unlink(parent, _macos_sel(b'removeChildWindow:'), window)
    for window in reversed(stack):
        # NSWindowAbove.
        link(parent, _macos_sel(b'addChildWindow:ordered:'), window, 1)


def _macos_raise_child(window):
    """Bring an activated dialog above its sibling dialogs.

    A click does not reorder child windows. Tk relinks a transient dialog
    when it activates, but not one that is only owned.
    """
    if _macos_menu_open(window):
        return
    ns_window = _macos_ns_window(window)
    if not ns_window:
        return
    try:
        parent = _macos_msg(ctypes.c_void_p)(ns_window, _macos_sel(b'parentWindow'))
        if not parent or not _macos_children_in_front(ns_window, parent):
            return
        _macos_msg(None, ctypes.c_void_p)(parent, _macos_sel(b'removeChildWindow:'), ns_window)
        # NSWindowAbove: AppKit puts an added child above its siblings.
        _macos_msg(None, ctypes.c_void_p, ctypes.c_long)(
            parent, _macos_sel(b'addChildWindow:ordered:'), ns_window, 1)
    except (OSError, AttributeError):
        return


def _macos_arm_dialog_links(widget):
    global _macos_link_job
    if _macos_link_job is not None:
        return
    try:
        _macos_link_job = widget._root().after(30, _macos_update_dialog_links)
    except tk.TclError:
        _macos_link_job = None


def _macos_on_frame(window, x, y):
    """The pointer is on the title bar or resize border, not the content."""
    left, top = window.winfo_rootx(), window.winfo_rooty()
    right, bottom = left + window.winfo_width(), top + window.winfo_height()
    if not (left - 8 <= x < right + 8 and top - 40 <= y < bottom + 8):
        return False
    return not (left + 4 <= x < right - 4 and top + 4 <= y < bottom - 4)


def _macos_buttons_down():
    """A mouse button is held, so a drag may be under way.

    A fast drag leaves the pointer off the title bar, and Tk reports the
    window where it was. Relinking then would move the dialog mid-drag.
    """
    try:
        event = _macos_objc().objc_getClass(b'NSEvent')
        return bool(_macos_msg(ctypes.c_ulong)(event, _macos_sel(b'pressedMouseButtons')))
    except (OSError, AttributeError):
        return False


def _macos_shown(window):
    return window.wm_state() == 'normal' and bool(int(window.winfo_viewable()))


def _macos_update_dialog_links():
    global _macos_link_job
    _macos_link_job = None
    for dialog in list(_macos_dialogs):
        try:
            if not dialog.winfo_exists():
                _macos_dialogs.remove(dialog)
        except tk.TclError:
            _macos_dialogs.remove(dialog)
    if not _macos_dialogs:
        return
    anchor = _macos_dialogs[0]
    if not _macos_menu_open(anchor) and not _macos_buttons_down():
        try:
            x, y = anchor.winfo_pointerxy()
        except tk.TclError:
            x = y = -1
        groups = {}
        for dialog in _macos_dialogs:
            owner = getattr(dialog, '_macos_owner', None)
            try:
                if owner is None or not owner.winfo_exists():
                    continue
                if not _macos_shown(dialog) or not int(owner.winfo_viewable()):
                    continue
                on_frame = _macos_on_frame(dialog, x, y)
            except tk.TclError:
                continue
            ns_window = _macos_ns_window(dialog)
            above = _macos_ns_window(owner)
            if not ns_window or not above:
                continue
            group = groups.setdefault(above, [[], False])
            group[0].append(ns_window)
            group[1] = group[1] or on_frame
        parent_of = _macos_msg(ctypes.c_void_p)
        for above, (ns_windows, detach) in groups.items():
            try:
                if detach:
                    _macos_unlink_children(above, ns_windows)
                    continue
                if all(parent_of(w, _macos_sel(b'parentWindow')) == above
                       for w in ns_windows):
                    continue
                _macos_link_children(above, ns_windows)
            except (OSError, AttributeError):
                continue
    _macos_arm_dialog_links(anchor)


def _macos_unwatch(window):
    try:
        _macos_windows.remove(window)
    except ValueError:
        pass


def _macos_bar_color(window):
    if _macos_colors is None:
        return None
    background, _foreground, window_background, window_chrome = _macos_colors
    color = window_chrome or window_background or background
    if not color:
        return None
    red, green, blue = window.winfo_rgb(color)
    return red / 65535, green / 65535, blue / 65535


def _macos_limit_dialog_controls(window, ns_window):
    """Leave a dialog with the close button only.

    The console and a floated output window keep minimize and zoom. A dialog
    stays with the console, so those two buttons are removed. Resizing stays.
    """
    if _app_window(window):
        return
    try:
        mask = int(_macos_msg(ctypes.c_ulong)(ns_window, _macos_sel(b'styleMask')) or 0)
        miniaturizable = 1 << 2
        if mask & miniaturizable:
            _macos_msg(None, ctypes.c_ulong)(
                ns_window, _macos_sel(b'setStyleMask:'), mask & ~miniaturizable)
        for kind in (1, 2):
            button = _macos_msg(ctypes.c_void_p, ctypes.c_ulong)(
                ns_window, _macos_sel(b'standardWindowButton:'), kind)
            if button:
                _macos_msg(None, ctypes.c_bool)(button, _macos_sel(b'setHidden:'), True)
    except (OSError, AttributeError):
        return


def _macos_titlebar(window):
    """Show the theme color through the system title bar.

    The close button and the resize border stay. Minimize and zoom are removed
    from a dialog. The bar stops painting its own material, so the window
    color shows through it, and the title text follows dark or light Aqua so
    it stays readable.
    """
    rgb = None
    try:
        rgb = _macos_bar_color(window)
    except tk.TclError:
        return
    ns_window = _macos_ns_window(window)
    if not ns_window:
        return
    _macos_limit_dialog_controls(window, ns_window)
    if rgb is None:
        return
    red, green, blue = rgb
    try:
        core = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
        core.CGColorCreateGenericRGB.restype = ctypes.c_void_p
        core.CGColorCreateGenericRGB.argtypes = (
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
        core.CFRelease.argtypes = (ctypes.c_void_p,)
        cg = core.CGColorCreateGenericRGB(red, green, blue, 1.0)
        if not cg:
            return
        nscolor = _macos_objc().objc_getClass(b'NSColor')
        color = _macos_msg(ctypes.c_void_p, ctypes.c_void_p)(
            nscolor, _macos_sel(b'colorWithCGColor:'), cg)
        core.CFRelease(cg)
        if not color:
            return
        dark = (0.2126 * red + 0.7152 * green + 0.0722 * blue) < 0.5
        window.tk.call(
            '::tk::unsupported::MacWindowStyle', 'appearance', window._w,
            'darkaqua' if dark else 'aqua')
        _macos_msg(None, ctypes.c_bool)(
            ns_window, _macos_sel(b'setTitlebarAppearsTransparent:'), True)
        _macos_msg(None, ctypes.c_void_p)(
            ns_window, _macos_sel(b'setBackgroundColor:'), color)
        try:
            _macos_msg(None, ctypes.c_long)(
                ns_window, _macos_sel(b'setTitlebarSeparatorStyle:'), 1)
        except (AttributeError, OSError):
            pass
        _macos_limit_dialog_controls(window, ns_window)
    except (tk.TclError, OSError, AttributeError):
        return


def close_leftover_windows(title: str, keep=None) -> None:
    """Hide Aqua windows left on screen after ``wm forget``.

    Forgetting a managed frame puts its view back in the parent, but the
    window it was given stays visible. The next manage builds a new window
    and leaves the old one beside it. ``keep`` is the window that still
    holds the frame.
    """
    if platform.system() != 'Darwin' or not title:
        return
    try:
        lib = _macos_objc()
        app = _macos_msg(ctypes.c_void_p)(
            lib.objc_getClass(b'NSApplication'), _macos_sel(b'sharedApplication'))
        if not app:
            return
        windows = _macos_msg(ctypes.c_void_p)(app, _macos_sel(b'windows'))
        if not windows:
            return
        count = _macos_msg(ctypes.c_ulong)(windows, _macos_sel(b'count'))
        victims = []
        for index in range(int(count)):
            window = _macos_msg(ctypes.c_void_p, ctypes.c_ulong)(
                windows, _macos_sel(b'objectAtIndex:'), index)
            if not window or window == keep or not _macos_msg(ctypes.c_bool)(
                    window, _macos_sel(b'isVisible')):
                continue
            ns_title = _macos_msg(ctypes.c_void_p)(window, _macos_sel(b'title'))
            if not ns_title:
                continue
            raw = _macos_msg(ctypes.c_char_p)(ns_title, _macos_sel(b'UTF8String'))
            if raw and raw.decode() == title:
                victims.append(window)
        for window in victims:
            # orderOut only. close releases the window, and the next
            # Tk update orders that same window back on screen.
            _macos_msg(None, ctypes.c_void_p)(
                window, _macos_sel(b'orderOut:'), None)
    except (OSError, AttributeError, UnicodeError):
        return


def claim_output_window(widget, title: str) -> None:
    """Title the Aqua window that holds ``widget`` and close older copies.

    A later ``wm manage`` shows the log in a new window with no title and
    leaves the previous titled window open.
    """
    if platform.system() != 'Darwin' or widget is None:
        return
    ns_window = _macos_ns_window(widget)
    if ns_window and title:
        try:
            lib = _macos_objc()
            text = _macos_msg(ctypes.c_void_p, ctypes.c_char_p)(
                lib.objc_getClass(b'NSString'),
                _macos_sel(b'stringWithUTF8String:'),
                title.encode())
            if text:
                _macos_msg(None, ctypes.c_void_p)(
                    ns_window, _macos_sel(b'setTitle:'), text)
        except (OSError, AttributeError, UnicodeError):
            pass
    close_leftover_windows(title, keep=ns_window)
    # The managed frame is not a Toplevel, so the usual watch never saw it.
    _macos_titlebar(widget)


def _macos_install(window):
    _macos_patch_toplevels()
    _macos_watch(window)


def _macos_apply(background, foreground, window_background, window_chrome):
    global _macos_colors
    _macos_colors = (background, foreground, window_background, window_chrome)
    for watched in list(_macos_windows):
        _macos_titlebar(watched)


def _pack_options(info):
    options = {}
    for key in ('anchor', 'expand', 'fill', 'side', 'padx', 'pady', 'ipadx', 'ipady'):
        if key in info:
            options[key] = info[key]
    return options


def _packed(window):
    saved = []
    try:
        slaves = list(window.pack_slaves())
    except tk.TclError:
        return saved
    for child in slaves:
        try:
            info = child.pack_info()
            child.pack_forget()
        except tk.TclError:
            continue
        saved.append((child, _pack_options(info)))
    return saved


def _repack(parent, saved):
    for child, options in saved:
        try:
            if not child.winfo_exists():
                continue
            child.pack(in_=parent, **options)
            # The slave stays a child of its original parent. Chrome built
            # around it is stacked later and would cover the log, so the
            # text and its menu never appear. Raise it once it is placed.
            if str(parent) != str(child.master):
                child.lift()
        except tk.TclError:
            pass


def _give_window_methods(window):
    """Let a managed frame use the same window methods as a Toplevel.

    ``wm manage`` teaches Tk the commands. The Python ``Frame`` class does
    not carry them.
    """
    if hasattr(window, 'wm_title'):
        return
    for name in (
            'wm_title', 'title', 'wm_resizable', 'resizable', 'wm_geometry', 'geometry',
            'wm_minsize', 'minsize', 'wm_maxsize', 'maxsize', 'wm_iconify', 'iconify',
            'wm_protocol', 'protocol', 'wm_overrideredirect', 'overrideredirect',
            'wm_deiconify', 'deiconify', 'wm_withdraw', 'withdraw', 'wm_state', 'state'):
        method = getattr(tk.Wm, name, None)
        if method is not None:
            setattr(window, name, method.__get__(window, window.__class__))


def adopt(window) -> None:
    """Theme a frame that ``wm manage`` turned into its own window.

    The frame already holds its content. On macOS the system title bar is
    tinted. On Windows and Linux the content moves under the same painted
    frame the other windows use, and comes back out in ``release``.
    """
    if window is None:
        return
    # This frame's master is the console. Without this it would be treated
    # as a dialog, including on macOS where that removes minimize and zoom.
    window._chrome_standalone = True
    if platform.system() == 'Darwin':
        _macos_watch(window)
        return
    if not _CUSTOM or getattr(window, '_chrome', None):
        return
    saved = _packed(window)
    _give_window_methods(window)
    try:
        install(window)
    except tk.TclError:
        if getattr(window, '_chrome', None):
            _remove(window)
        _repack(window, saved)
        raise
    chrome = getattr(window, '_chrome', None)
    client = getattr(window, '_chrome_client', None)
    if chrome is None or client is None:
        _repack(window, saved)
        return
    chrome['placed'] = True
    _repack(client, saved)


def release(window) -> None:
    """Drop the themed frame from a managed window and restore its content."""
    if window is None:
        return
    if platform.system() == 'Darwin':
        _macos_unwatch(window)
        return
    chrome = getattr(window, '_chrome', None)
    if chrome is None:
        return
    client = chrome.get('client')
    saved = _packed(client) if client is not None else []
    _remove(window)
    _repack(window, saved)


def install(window) -> None:
    """Replace the native frame with themed chrome.

    On macOS the system frame stays, and its title bar is tinted from the
    window color. Elsewhere, content created afterwards is parented under
    the client frame, below the title. No effect when ``window`` already
    has chrome.
    """
    if platform.system() == 'Darwin':
        _macos_install(window)
        return
    if window is None or not _CUSTOM or getattr(window, '_chrome', None):
        return
    _ensure_patched()
    window._chrome_building = True
    try:
        _build(window)
    finally:
        window._chrome_building = False
    _installed.append(window)
    try:
        window.update_idletasks()
    except tk.TclError:
        pass
    if platform.system() == 'Linux' and not _install_linux_frame(window):
        # No window manager to hand the drag to. Fall back to a borderless window.
        window._chrome_applying = True
        try:
            window.overrideredirect(True)
        finally:
            window._chrome_applying = False
    _arm_system_resize(window)
    _schedule_taskbar(window)


def apply(background, foreground, window_background=None, window_chrome=None) -> None:
    """Recolor every themed frame from the window colors."""
    global _pending
    if platform.system() == 'Darwin':
        _macos_apply(background, foreground, window_background, window_chrome)
        return
    if not _CUSTOM:
        return
    _pending = (str(background), str(foreground),
                None if window_background is None else str(window_background),
                None if window_chrome is None else str(window_chrome))
    for window in list(_installed):
        chrome = getattr(window, '_chrome', None)
        if chrome is None:
            continue
        try:
            if not window.winfo_exists():
                continue
        except tk.TclError:
            continue
        _paint(window, _palette(window, *_pending))


def _ensure_patched() -> None:
    global _patched
    if _patched:
        return
    _patched = True
    original_base = tk.BaseWidget.__init__
    original_override = tk.Wm.wm_overrideredirect
    original_resizable = tk.Wm.wm_resizable
    original_toplevel = tk.Toplevel.__init__
    original_withdraw = tk.Wm.wm_withdraw
    original_deiconify = tk.Wm.wm_deiconify
    original_geometry = tk.Wm.wm_geometry

    def base_init(self, master, widgetName, cnf={}, kw={}, extra=()):
        if widgetName != 'toplevel' and master is not None:
            if not getattr(master, '_chrome_building', False):
                client = getattr(master, '_chrome_client', None)
                if client is not None:
                    master = client
        result = original_base(self, master, widgetName, cnf, kw, extra)
        _attach_widget(self)
        return result

    def wm_overrideredirect(self, boolean=None):
        if boolean and getattr(self, '_chrome', None) and not getattr(self, '_chrome_applying', False):
            _remove(self)
        return original_override(self, boolean)

    def wm_resizable(self, width=None, height=None):
        # Tk turns the override-redirect frame off and on while changing this.
        # That call is not a request to drop the themed frame.
        self._chrome_applying = True
        try:
            result = original_resizable(self, width, height)
        finally:
            self._chrome_applying = False
        if width is not None and getattr(self, '_chrome', None):
            _layout_grips(self)
            _rebind_frame(self)
            _refresh_frame(self)
        return result

    def wm_withdraw(self):
        # Menus and tooltips hide themselves. A later reveal must not show them
        # again at the default size.
        if getattr(self, '_chrome_hidden', None) is not None:
            self._chrome_hidden = True
        return original_withdraw(self)

    def wm_deiconify(self):
        if getattr(self, '_chrome_hidden', None) is not None:
            self._chrome_hidden = False
        return original_deiconify(self)

    def wm_geometry(self, value=None):
        result = original_geometry(self, value)
        if isinstance(value, str):
            # Position-only requests start with + or -. The size is what Tk
            # was asked for, before Windows adds the stock caption.
            size = value.split('+', 1)[0].split('-', 1)[0]
            if 'x' in size:
                width, _, height = size.partition('x')
                try:
                    self._chrome_asked = (int(width), int(height))
                except ValueError:
                    pass
        return result

    def toplevel_init(self, master=None, cnf={}, **kw):
        original_toplevel(self, master, cnf, **kw)
        # Tk maps a new window immediately, at a small default size. Hide it
        # before the frame is built so that size is never painted.
        try:
            original_withdraw(self)
        except tk.TclError:
            return
        self._chrome_hidden = False
        install(self)
        try:
            # Owned by the application root. A callback on this window is
            # deleted when the window closes, and the timer still fires.
            self._root().after(0, _reveal_toplevel, self)
        except tk.TclError:
            pass

    tk.BaseWidget.__init__ = base_init
    tk.Wm.wm_overrideredirect = wm_overrideredirect
    tk.Wm.overrideredirect = wm_overrideredirect
    tk.Wm.wm_resizable = wm_resizable
    tk.Wm.resizable = wm_resizable
    tk.Wm.wm_withdraw = wm_withdraw
    tk.Wm.withdraw = wm_withdraw
    tk.Wm.wm_deiconify = wm_deiconify
    tk.Wm.deiconify = wm_deiconify
    tk.Wm.wm_geometry = wm_geometry
    tk.Wm.geometry = wm_geometry
    tk.Toplevel.__init__ = toplevel_init


def _reveal_toplevel(window):
    """Show a toplevel once its owner has had a chance to place it.

    Menus mark themselves hidden before this runs, and stay hidden.
    """
    if getattr(window, '_chrome_hidden', True):
        return
    try:
        if not window.winfo_exists():
            return
        window.deiconify()
    except tk.TclError:
        pass


def _caption_family():
    """Icon font that contains the Windows caption marks, if this machine has it."""
    global _caption_family_name, _caption_family_ready
    if _caption_family_ready:
        return _caption_family_name
    _caption_family_ready = True
    try:
        families = set(tkfont.families())
    except tk.TclError:
        return None
    for name in _CAPTION_FAMILIES:
        if name in families:
            _caption_family_name = name
            return name
    return None


def _caption_pad(family):
    """Side padding that makes each button the width of a Windows caption button."""
    if not family:
        mark = _CAPTION_PX
    else:
        mark = tkfont.Font(family=family, size=-_CAPTION_PX).measure(_CAPTION_GLYPH['close'])
    return max((_CAPTION_BUTTON - mark) // 2, 0)


def _caption_pixels(kind):
    """1px caption mark used when the Windows icon font is not installed."""
    def box(left, top, right, bottom):
        pixels = []
        for x in range(left, right + 1):
            pixels.append((x, top))
            pixels.append((x, bottom))
        for y in range(top + 1, bottom):
            pixels.append((left, y))
            pixels.append((right, y))
        return pixels

    if kind == 'min':
        return [(x, 4) for x in range(0, 10)]
    if kind == 'restore':
        return box(3, 0, 9, 6) + box(0, 3, 6, 9)
    if kind == 'close':
        return [(x, x) for x in range(0, 10)] + [(x, 9 - x) for x in range(0, 10)]
    return box(0, 0, 9, 9)


def _caption_image(master, kind, color):
    image = tk.PhotoImage(master=master, width=_CAPTION_PX, height=_CAPTION_PX)
    image.blank()
    for x, y in _caption_pixels(kind):
        image.put(color, (x, y))
    return image


def _caption_shown(button):
    if button._caption_role == 'max':
        try:
            if button.winfo_toplevel()._chrome.get('zoomed'):
                return 'restore'
        except tk.TclError:
            pass
    return button._caption_role


def _refresh_caption_glyph(button):
    """Show the caption mark for this button in its current color."""
    kind = _caption_shown(button)
    family = button._caption_family
    if family:
        if getattr(button, '_caption_shown', None) == kind:
            return
        button.configure(text=_CAPTION_GLYPH[kind])
        button._caption_shown = kind
        return
    color = str(button.cget('foreground'))
    if getattr(button, '_caption_shown', None) == (kind, color):
        return
    image = _caption_image(button, kind, color)
    # X11 ignores padx on an image label, so the width is set directly.
    # With an image, width is in pixels.
    button.configure(image=image, width=_CAPTION_BUTTON, padx=0)
    button._caption_image = image
    button._caption_shown = (kind, color)


def _build(window):
    palette = _starting_palette(window)
    north = _grip(window, height=1, cursor='size_ns')
    south = _grip(window, height=1, cursor='size_ns')
    center = tk.Frame(window, borderwidth=0, highlightthickness=0)
    west = _grip(center, width=1, cursor='size_we')
    east = _grip(center, width=1, cursor='size_we')
    column = tk.Frame(center, borderwidth=0, highlightthickness=0)
    title = tk.Frame(column, height=32, borderwidth=0, highlightthickness=0)
    title.pack_propagate(False)
    icon = tk.Label(title, borderwidth=0, highlightthickness=0)
    text = tk.Label(title, anchor='w', borderwidth=0, highlightthickness=0, width=1)
    buttons = {}
    family = _caption_family()
    pad = _caption_pad(family)
    actions = [('close', lambda: _close(window))]
    if _app_window(window):
        actions.extend((
            ('max', lambda: _toggle_zoom(window)),
            ('min', lambda: _minimize(window)),
        ))
    for name, action in actions:
        button = tk.Label(
            title, borderwidth=0, highlightthickness=0, takefocus=0, padx=pad,
            text=_CAPTION_GLYPH[name] if family else '',
            font=(family, -_CAPTION_PX) if family else 'TkDefaultFont')
        button._caption_family = family
        button._caption_role = name
        button._caption_image = None
        button._caption_shown = None
        button.pack(side=tk.RIGHT, fill=tk.Y)
        button.bind('<Button-1>', lambda _event, button=button: _press_caption_button(button))
        button.bind(
            '<ButtonRelease-1>',
            lambda _event, button=button, action=action: _release_caption_button(button, action))
        buttons[name] = button
    icon.pack(side=tk.LEFT, padx=(8, 8))
    text.pack(side=tk.LEFT, fill=tk.X, expand=True)
    separator = tk.Frame(column, height=1, borderwidth=0, highlightthickness=0)
    client = tk.Frame(column, borderwidth=0, highlightthickness=0)
    title.pack(side=tk.TOP, fill=tk.X)
    separator.pack(side=tk.TOP, fill=tk.X)
    separator.pack_propagate(False)
    client.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    window._chrome_client = client

    chrome = {
        'widgets': (north, south, center, west, east, column, title, icon, text,
                    separator, client, *buttons.values()),
        'north': north, 'south': south, 'west': west, 'east': east,
        'center': center, 'column': column, 'title': title, 'icon': icon, 'text': text,
        'separator': separator, 'client': client, 'buttons': buttons,
        'wm_title': window.wm_title,
        'zoomed': False,
        'restore': None,
        'resize': (1, 1),
        'job': None,
        'image': _icon_image(window),
        'palette': palette,
        'prior_grab': None,
        'placed': False,
    }
    window._chrome = chrome
    if chrome['image'] is not None:
        icon.configure(image=chrome['image'])
    text.configure(text=window.wm_title() or '')
    _wrap_title(window, text, chrome['wm_title'])
    _bind_drag(window, title)
    _bind_drag(window, text)
    _bind_drag(window, icon)
    if 'max' in buttons:
        title.bind('<Double-Button-1>', lambda _event: _toggle_zoom(window))
        text.bind('<Double-Button-1>', lambda _event: _toggle_zoom(window))
    _bind_grip(window, north, 'n')
    _bind_grip(window, south, 's')
    _bind_grip(window, west, 'w')
    _bind_grip(window, east, 'e')
    # Motion while the button is down is not always delivered. The pointer
    # poll below covers that. This binding covers the events that do arrive.
    # The ids are kept so a frameless window, such as a menu, can drop them
    # when its frame is removed.
    window._chrome_release = window.bind(
        '<ButtonRelease-1>', lambda _event: _release_pointer(window), add='+')
    window._chrome_motion = window.bind(
        '<B1-Motion>', lambda event: _pointer_motion(window, event), add='+')
    window.bind('<Alt-F4>', lambda _event: _close(window), add='+')
    window.bind('<Destroy>', lambda event: _forget(window, event), add='+')
    _redirect_grid(window, client)
    _redirect_padding(window, client)
    _layout_grips(window)
    _paint(window, palette)
    _install_size_hook(window)
    window.bind('<Configure>', lambda event: _sync_size_hook_event(window, event), add='+')


def _cursor(name):
    """Resize pointer names. X11 does not provide the Windows size_* cursors."""
    if platform.system() == 'Windows' or name == 'arrow':
        return name
    return {
        'size_ns': 'sb_v_double_arrow',
        'size_we': 'sb_h_double_arrow',
        'size_nw_se': 'top_left_corner',
        'size_ne_sw': 'top_right_corner',
    }.get(name, name)


def _grip(window, cursor, width=None, height=None):
    grip = tk.Frame(window, borderwidth=0, highlightthickness=0, cursor=_cursor(cursor))
    if width is not None:
        grip.configure(width=width)
    if height is not None:
        grip.configure(height=height)
    grip.pack_propagate(False)
    return grip


def _wrap_title(window, label, original):
    def title(value=None):
        if value is None:
            return original()
        original(value)
        try:
            if label.winfo_exists():
                label.configure(text=value)
        except tk.TclError:
            pass
        return value

    window.wm_title = title
    window.title = title


def _redirect_grid(window, client):
    def rowconfigure(index, cnf={}, **kwargs):
        return client.grid_rowconfigure(index, cnf, **kwargs)

    def columnconfigure(index, cnf={}, **kwargs):
        return client.grid_columnconfigure(index, cnf, **kwargs)

    window.grid_rowconfigure = rowconfigure
    window.rowconfigure = rowconfigure
    window.grid_columnconfigure = columnconfigure
    window.columnconfigure = columnconfigure


def _redirect_padding(window, client):
    """Keep padx/pady inside the frame.

    Dialogs inset their contents with padding on the window. The chrome is a
    child of that window, so the same padding would draw as a thick border.
    """
    original = window.configure

    def configure(cnf=None, **kwargs):
        if isinstance(cnf, str):
            return original(cnf)
        if isinstance(cnf, dict):
            kwargs = {**cnf, **kwargs}
        pad = {key: kwargs.pop(key) for key in ('padx', 'pady') if key in kwargs}
        if pad:
            client.configure(**pad)
        if kwargs or cnf is None and not pad:
            return original(**kwargs) if kwargs else original()

    window.configure = configure
    window.config = configure


def _layout_grips(window):
    chrome = window._chrome
    try:
        horizontal, vertical = window.resizable()
        horizontal, vertical = int(horizontal), int(vertical)
    except tk.TclError:
        horizontal, vertical = 1, 1
    chrome['resize'] = (horizontal, vertical)
    for name in ('north', 'south', 'west', 'east', 'center', 'column'):
        chrome[name].pack_forget()
    chrome['center'].pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    if platform.system() == 'Windows':
        # The native border is the edge. A packed grip would draw a second one.
        chrome['column'].pack(fill=tk.BOTH, expand=True)
        return
    chrome['north'].pack(side=tk.TOP, fill=tk.X)
    chrome['south'].pack(side=tk.BOTTOM, fill=tk.X)
    chrome['west'].pack(side=tk.LEFT, fill=tk.Y)
    chrome['east'].pack(side=tk.RIGHT, fill=tk.Y)
    chrome['column'].pack(fill=tk.BOTH, expand=True)


def _bind_drag(window, widget):
    widget.bind('<Button-1>', lambda event: _press_drag(window, event))
    widget.bind('<B1-Motion>', lambda event: _drag(window, event))


def _bind_grip(window, grip, edge):
    grip.bind('<Motion>', lambda event, edge=edge: _grip_cursor(window, grip, edge, event))
    grip.bind('<Button-1>', lambda event, edge=edge: _press_resize(window, edge, event))
    grip.bind('<B1-Motion>', lambda event: _resize(window, event))


def _grip_cursor(window, grip, edge, event):
    if edge in ('n', 's'):
        width = max(grip.winfo_width(), 1)
        if event.x <= 8:
            chosen = ('nw' if edge == 'n' else 'sw')
            cursor = 'size_ne_sw' if edge == 's' else 'size_nw_se'
        elif event.x >= width - 8:
            chosen = ('ne' if edge == 'n' else 'se')
            cursor = 'size_nw_se' if edge == 's' else 'size_ne_sw'
        else:
            chosen = edge
            cursor = 'size_ns'
    else:
        height = max(grip.winfo_height(), 1)
        if event.y <= 8:
            chosen = 'nw' if edge == 'w' else 'ne'
            cursor = 'size_nw_se' if edge == 'w' else 'size_ne_sw'
        elif event.y >= height - 8:
            chosen = 'sw' if edge == 'w' else 'se'
            cursor = 'size_ne_sw' if edge == 'w' else 'size_nw_se'
        else:
            chosen = edge
            cursor = 'size_we'
    allowed = _allowed_edge(window, chosen)
    grip._edge = allowed
    grip.configure(cursor=_cursor(cursor) if allowed else 'arrow')


def _allowed_edge(window, edge):
    horizontal, vertical = window._chrome['resize']
    kept = ''
    if horizontal and ('w' in edge or 'e' in edge):
        kept += 'w' if 'w' in edge else 'e'
    if vertical and ('n' in edge or 's' in edge):
        kept += 'n' if 'n' in edge else 's'
    return kept


def _context_menu_open():
    """True when a console menu is up.

    Read from the window procedure. That code must not call into Tk.
    """
    module = sys.modules.get('dgenerate.console.thememenu')
    if module is None:
        return False
    return bool(module._open)


def _dismiss_context_menu():
    """Close an open menu from a Tk event. True when one was open."""
    module = sys.modules.get('dgenerate.console.thememenu')
    if module is None or not module._open:
        return False
    module._close_all()
    return True


def _press_caption_button(button):
    button._menu_dismissed = _dismiss_context_menu()
    return 'break'


def _release_caption_button(button, action):
    if getattr(button, '_menu_dismissed', False):
        button._menu_dismissed = False
        return 'break'
    action()


def _frame_origin(window):
    """Position that ``geometry`` reads back.

    On X11 the window manager reparents the frame, and ``winfo_x`` is only the
    offset inside that frame. The root position is the one geometry uses.
    """
    try:
        if platform.system() == 'Linux':
            return window.winfo_rootx(), window.winfo_rooty()
        return window.winfo_x(), window.winfo_y()
    except tk.TclError:
        return 0, 0


def _move_hit(hit, menu_open):
    """Hit code for a painted title.

    ``HTCAPTION`` does not start a move from this subclass. The click has to
    land on the title widget, which hands the move back to Windows. Moving
    the window by setting its geometry leaves the client area behind the frame.
    """
    if hit is None:
        return None
    if menu_open or hit == _HTCAPTION:
        return _HTCLIENT
    return hit


def _press_drag(window, event):
    if _dismiss_context_menu():
        return 'break'
    if platform.system() == 'Windows':
        if window._chrome['zoomed']:
            _toggle_zoom(window)
        hwnd = _hwnd(window)
        if hwnd:
            user32 = _user32
            user32.ReleaseCapture.restype = ctypes.c_int
            user32.ReleaseCapture.argtypes = ()
            user32.ReleaseCapture()
            _begin_system_drag(hwnd, _SC_MOVE | _HTCAPTION)
            return 'break'
    # The click is already owned by Tk. Asking the window manager to grab it
    # does not move the window, so the title tracks the pointer directly.
    if window._chrome['zoomed']:
        _toggle_zoom(window)
        window._chrome['drag'] = (max(window.winfo_width() // 2, 1), 14)
        return
    origin_x, origin_y = _frame_origin(window)
    window._chrome['drag'] = (event.x_root - origin_x, event.y_root - origin_y)
    _track_pointer(window)


def _pointer_motion(window, event):
    chrome = getattr(window, '_chrome', None)
    if not chrome:
        return
    if chrome.get('resize_drag'):
        _resize(window, event)
        return
    _drag(window, event)


def _drag(window, event):
    chrome = getattr(window, '_chrome', None)
    if not chrome:
        return
    origin = chrome.get('drag')
    if not origin:
        return
    window.geometry(f'+{event.x_root - origin[0]}+{event.y_root - origin[1]}')


def _query_pointer():
    """Root pointer position and button mask, or nothing off X11.

    A button press can leave the pointer grabbed, so motion events never
    arrive. The pointer position itself is still readable.
    """
    lib, display = _x11_lib()
    if not lib or not display:
        return None
    root = lib.XRootWindow(display, lib.XDefaultScreen(display))
    root_ret = ctypes.c_ulong()
    child = ctypes.c_ulong()
    root_x = ctypes.c_int()
    root_y = ctypes.c_int()
    win_x = ctypes.c_int()
    win_y = ctypes.c_int()
    buttons = ctypes.c_uint()
    if not lib.XQueryPointer(
            display, root, ctypes.byref(root_ret), ctypes.byref(child),
            ctypes.byref(root_x), ctypes.byref(root_y),
            ctypes.byref(win_x), ctypes.byref(win_y), ctypes.byref(buttons)):
        return None
    return root_x.value, root_y.value, buttons.value


class _PointerEvent:
    def __init__(self, x, y):
        self.x_root = x
        self.y_root = y


def _track_pointer(window):
    """Follow the pointer while button 1 is held."""
    chrome = getattr(window, '_chrome', None)
    if chrome is None or not (chrome.get('drag') or chrome.get('resize_drag')):
        return
    point = _query_pointer()
    if point is None:
        return
    x, y, buttons = point
    if not buttons & (1 << 8):
        chrome['drag'] = None
        chrome['resize_drag'] = None
        return
    event = _PointerEvent(x, y)
    if chrome.get('resize_drag'):
        _resize(window, event)
    elif chrome.get('drag'):
        _drag(window, event)
    try:
        window.after(16, _track_pointer, window)
    except tk.TclError:
        pass


def _press_resize(window, edge, event):
    if _dismiss_context_menu():
        return 'break'
    if window._chrome['zoomed']:
        return
    chosen = _allowed_edge(window, getattr(event.widget, '_edge', edge))
    if not chosen:
        return
    if platform.system() != 'Linux':
        _capture(window)
    origin_x, origin_y = _frame_origin(window)
    window._chrome['resize_drag'] = {
        'edge': chosen,
        'x': event.x_root,
        'y': event.y_root,
        'geom': (origin_x, origin_y, window.winfo_width(), window.winfo_height()),
    }
    _track_pointer(window)


def _resize(window, event):
    drag = window._chrome.get('resize_drag')
    if not drag:
        return
    x, y, width, height = drag['geom']
    dx = event.x_root - drag['x']
    dy = event.y_root - drag['y']
    edge = drag['edge']
    min_w, min_h = _minimum(window)
    if 'e' in edge:
        width = max(min_w, width + dx)
    if 'w' in edge:
        width = max(min_w, width - dx)
        x += drag['geom'][2] - width
    if 's' in edge:
        height = max(min_h, height + dy)
    if 'n' in edge:
        height = max(min_h, height - dy)
        y += drag['geom'][3] - height
    window.geometry(f'{width}x{height}+{x}+{y}')


def _minimum(window):
    try:
        width, height = window.minsize()
        width, height = int(width), int(height)
    except tk.TclError:
        width, height = 0, 0
    return max(width, 160), max(height, 64)


def _capture(window):
    chrome = window._chrome
    try:
        chrome['prior_grab'] = window.grab_current()
    except tk.TclError:
        chrome['prior_grab'] = None
    try:
        window.grab_set()
    except tk.TclError:
        pass


def _release_pointer(window):
    chrome = getattr(window, '_chrome', None)
    if chrome is None:
        return
    chrome['drag'] = None
    chrome['resize_drag'] = None
    prior = chrome.get('prior_grab')
    chrome['prior_grab'] = None
    try:
        if window.grab_current() is window:
            window.grab_release()
    except tk.TclError:
        pass
    if prior is not None and prior is not window:
        try:
            if prior.winfo_exists():
                prior.grab_set()
        except tk.TclError:
            pass


def _toggle_zoom(window):
    if 'max' not in window._chrome['buttons']:
        return
    if platform.system() == 'Windows':
        hwnd = _hwnd(window)
        if hwnd:
            user32 = _user32
            user32.IsZoomed.argtypes = (ctypes.c_void_p,)
            user32.IsZoomed.restype = ctypes.c_int
            command = _SC_RESTORE if user32.IsZoomed(ctypes.c_void_p(hwnd)) else _SC_MAXIMIZE
            _post_syscommand(hwnd, command)
            return
    chrome = window._chrome
    if chrome['zoomed']:
        chrome['zoomed'] = False
        if chrome['restore']:
            window.geometry(chrome['restore'])
        _refresh_caption_glyph(chrome['buttons']['max'])
        return
    chrome['restore'] = window.geometry()
    left, top, width, height = _work_area(window)
    window.geometry(f'{width}x{height}+{left}+{top}')
    chrome['zoomed'] = True
    _refresh_caption_glyph(chrome['buttons']['max'])


def _work_area(window):
    if platform.system() == 'Windows':
        rect = _Rect()
        _user32.SystemParametersInfoW(_SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    return 0, 0, window.winfo_screenwidth(), window.winfo_screenheight()


def _remote_session():
    """True when this Windows session is being viewed over Remote Desktop.

    A border click does not start a move or a size in that session. Both have
    to be posted as system commands. The size command uses ``WMSZ_*`` in its
    low four bits; a raw ``HT*`` code aliases the bottom corners to a move.
    """
    if platform.system() != 'Windows':
        return False
    user32 = _user32
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
    return bool(user32.GetSystemMetrics(_SM_REMOTESESSION))


def _resize_command(hit):
    """``SC_SIZE`` for a border hit, or None when the hit is not an edge.

    Remote Desktop starts the size from this command. The direction lives in
    the low four bits and must be a ``WMSZ_*`` value. ``HTBOTTOMLEFT`` is 16
    and ``HTBOTTOMRIGHT`` is 17, so combining either with ``SC_SIZE`` sets the
    ``SC_MOVE`` bit and the corner drags the window. The other edges land on
    direction codes Windows ignores, so the border does not resize.
    """
    direction = {
        _HTLEFT: _WMSZ_LEFT,
        _HTRIGHT: _WMSZ_RIGHT,
        _HTTOP: _WMSZ_TOP,
        _HTTOPLEFT: _WMSZ_TOPLEFT,
        _HTTOPRIGHT: _WMSZ_TOPRIGHT,
        _HTBOTTOM: _WMSZ_BOTTOM,
        _HTBOTTOMLEFT: _WMSZ_BOTTOMLEFT,
        _HTBOTTOMRIGHT: _WMSZ_BOTTOMRIGHT,
    }.get(int(hit))
    if direction is None:
        return None
    return _SC_SIZE | direction


def _begin_system_drag(hwnd, command, point=0):
    now = _time.monotonic()
    previous = _drag_posted.get(int(hwnd), 0)
    if now - previous < 0.2:
        return
    _drag_posted[int(hwnd)] = now
    _post_syscommand(hwnd, command, point)


def _post_syscommand(hwnd, command, point=0):
    """Queue a system command instead of sending it.

    ``SendMessage`` runs the subclass procedure before the button handler
    returns. That procedure has to release the interpreter lock, and doing
    so from inside the handler leaves no thread state to restore.
    """
    user32 = _user32
    user32.PostMessageW.restype = ctypes.c_int
    user32.PostMessageW.argtypes = (
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_ssize_t, ctypes.c_ssize_t)
    user32.PostMessageW(
        ctypes.c_void_p(hwnd), _WM_SYSCOMMAND, command, int(point))


def _window_long(hwnd, index):
    user32 = _user32
    user32.GetWindowLongPtrW.restype = ctypes.c_void_p
    user32.GetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int)
    return int(user32.GetWindowLongPtrW(ctypes.c_void_p(hwnd), index) or 0)


def _set_window_long(hwnd, index, value):
    user32 = _user32
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.SetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
    user32.SetWindowLongPtrW(ctypes.c_void_p(hwnd), index, ctypes.c_void_p(value))


def _without_size_hook(hwnd, call):
    """Run ``call`` while the themed procedure is not installed on ``hwnd``."""
    hwnd = int(hwnd)
    hook = _size_hooks.get(hwnd)
    comctl = _comctl32 if hook is not None else None
    pointer = _subclass_pointer() if hook is not None else None
    if comctl is not None:
        comctl.RemoveWindowSubclass.restype = ctypes.c_int
        comctl.RemoveWindowSubclass.argtypes = (
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
        comctl.RemoveWindowSubclass(ctypes.c_void_p(hwnd), pointer, 1)
    try:
        return call()
    finally:
        if comctl is not None:
            comctl.SetWindowSubclass.restype = ctypes.c_int
            comctl.SetWindowSubclass.argtypes = (
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ssize_t)
            refdata = ctypes.addressof(hook.native) if getattr(hook, 'native', None) is not None else 0
            comctl.SetWindowSubclass(ctypes.c_void_p(hwnd), pointer, 1, refdata)


def _flush_frame_change(hwnd):
    """Apply a style change without re-entering the themed window procedure.

    The procedure releases the interpreter lock for the default handler.
    A style change from a button handler is still inside that handler, so
    the subclass is lifted for the flush and put back afterwards.
    """
    user32 = _user32
    user32.SetWindowPos.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint)

    def move():
        user32.SetWindowPos(
            ctypes.c_void_p(int(hwnd)), None, 0, 0, 0, 0,
            _SWP_NOMOVE_NOSIZE_NOZORDER_NOACTIVATE | _SWP_NOREDRAW | _SWP_FRAMECHANGED)

    _without_size_hook(hwnd, move)


def _app_window(window):
    """The console and a floated output window.

    A dialog keeps a close button only. Several of them are modal, and the
    themed dialogs were not given minimize or maximize.
    """
    return window.master is None or getattr(window, '_chrome_standalone', False)


def _release_owner_for_minimize(window, hwnd):
    """Let a dialog minimize to the taskbar.

    An owned window has nowhere to go. Windows hides it with its owner, or
    shows a small iconic window at the lower left. The console link is
    released until this dialog itself is restored. The Tk window stays as it
    is: clearing the transient master would rebuild it, which drops the
    contents and leaves a white frame on the taskbar thumbnail.
    """
    chrome = getattr(window, '_chrome', None)
    if chrome is None or chrome.get('minimized_detached'):
        return
    owner = _window_long(hwnd, _GWLP_HWNDPARENT)
    if not owner or owner == int(hwnd):
        return
    chrome['minimized_detached'] = True
    chrome['saw_iconic'] = False
    _set_window_long(hwnd, _GWLP_HWNDPARENT, 0)
    exstyle = _window_long(hwnd, _GWL_EXSTYLE)
    exstyle = (exstyle & ~_WS_EX_TOOLWINDOW) | _WS_EX_APPWINDOW
    _set_window_long(hwnd, _GWL_EXSTYLE, exstyle)
    _flush_frame_change(hwnd)
    try:
        window.after(50, _watch_minimized, window)
    except tk.TclError:
        pass
    return hwnd


def _reattach_after_minimize(window):
    chrome = getattr(window, '_chrome', None)
    if chrome is None:
        return
    chrome['minimized_detached'] = False
    chrome['saw_iconic'] = False
    _apply_taskbar_style(window)
    try:
        window.update_idletasks()
    except tk.TclError:
        pass


def _watch_minimized(window):
    """Put the console back in charge once the dialog leaves the taskbar."""
    chrome = getattr(window, '_chrome', None)
    if chrome is None or not chrome.get('minimized_detached'):
        return
    try:
        if not window.winfo_exists():
            return
    except tk.TclError:
        return
    hwnd = _hwnd(window)
    if not hwnd:
        return
    user32 = _user32
    user32.IsIconic.argtypes = (ctypes.c_void_p,)
    user32.IsIconic.restype = ctypes.c_int
    iconic = bool(user32.IsIconic(ctypes.c_void_p(hwnd)))
    if iconic:
        chrome['saw_iconic'] = True
    elif chrome.get('saw_iconic'):
        _reattach_after_minimize(window)
        return
    try:
        window.after(50, _watch_minimized, window)
    except tk.TclError:
        pass


def _minimize(window):
    if platform.system() == 'Windows':
        hwnd = _hwnd(window)
        if hwnd:
            hwnd = _release_owner_for_minimize(window, hwnd) or hwnd
            _post_syscommand(hwnd, _SC_MINIMIZE)
            return
    try:
        window.iconify()
    except tk.TclError:
        pass


def _close(window):
    try:
        command = window.protocol('WM_DELETE_WINDOW')
    except tk.TclError:
        command = ''
    if command:
        try:
            window.tk.call(command)
            return
        except tk.TclError:
            pass
    try:
        window.destroy()
    except tk.TclError:
        pass


def _schedule_taskbar(window):
    try:
        host = window._root()
        window._chrome['job_host'] = host
        window._chrome['job'] = host.after(1, _realize_taskbar, window)
    except tk.TclError:
        pass
    window.bind('<Map>', lambda event: _on_map(window, event), add='+')


def _cancel_taskbar(window, chrome):
    job = chrome.get('job') if chrome is not None else None
    if job is None:
        return
    host = chrome.get('job_host', window)
    chrome['job'] = None
    try:
        host.after_cancel(job)
    except tk.TclError:
        pass


def _on_map(window, event):
    if event.widget is not window or getattr(window, '_chrome', None) is None:
        return
    if platform.system() == 'Linux':
        _install_linux_frame(window)
    _apply_taskbar_style(window)
    _sync_size_hook(window)
    _refresh_frame(window)
    _correct_opening_height(window)
    chrome = window._chrome
    if chrome.get('placed'):
        return
    try:
        chrome['place_job'] = window.after_idle(lambda: _center_if_default(window))
    except tk.TclError:
        pass


def _center_if_default(window):
    """Center a window the window manager used to place.

    A window the window manager did not place opens at the top-left. Dialogs
    that set their own position are left where they are.
    """
    chrome = getattr(window, '_chrome', None)
    if chrome is None or chrome.get('placed'):
        return
    chrome['placed'] = True
    try:
        if not window.winfo_exists() or window.winfo_x() or window.winfo_y():
            return
        window.update_idletasks()
        width = max(window.winfo_width(), window.winfo_reqwidth())
        height = max(window.winfo_height(), window.winfo_reqheight())
        parent = window.master.winfo_toplevel() if window.master is not None else None
        if parent is None:
            left = (window.winfo_screenwidth() - width) // 2
            top = (window.winfo_screenheight() - height) // 2
        else:
            left = parent.winfo_rootx() + (parent.winfo_width() // 2) - (width // 2)
            top = parent.winfo_rooty() + (parent.winfo_height() // 2) - (height // 2)
        window.geometry(f'+{max(left, 0)}+{max(top, 0)}')
    except tk.TclError:
        pass


def _realize_taskbar(window):
    chrome = getattr(window, '_chrome', None)
    if chrome is None:
        return
    chrome['job'] = None
    _apply_taskbar_style(window)


def _apply_taskbar_style(window):
    """Give the main window a taskbar button. Dialogs stay owned by it."""
    if platform.system() != 'Windows':
        return
    chrome = getattr(window, '_chrome', None)
    if chrome is not None and chrome.get('minimized_detached'):
        return
    hwnd = _hwnd(window)
    if not hwnd:
        return
    user32 = _user32
    user32.GetWindowLongPtrW.restype = ctypes.c_void_p
    user32.GetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int)
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.SetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
    user32.SetWindowPos.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint)
    hwnd_p = ctypes.c_void_p(hwnd)
    style = int(user32.GetWindowLongPtrW(hwnd_p, _GWL_STYLE) or 0)
    exstyle = int(user32.GetWindowLongPtrW(hwnd_p, _GWL_EXSTYLE) or 0)
    root = _app_window(window)
    if root:
        style |= _WS_MINIMIZEBOX | _WS_MAXIMIZEBOX | _WS_SYSMENU
        exstyle = (exstyle & ~_WS_EX_TOOLWINDOW) | _WS_EX_APPWINDOW
    else:
        style = (style & ~(_WS_MINIMIZEBOX | _WS_MAXIMIZEBOX)) | _WS_SYSMENU
        exstyle = (exstyle & ~_WS_EX_APPWINDOW) | _WS_EX_TOOLWINDOW
        owner = _hwnd(window.master.winfo_toplevel())
        if owner and owner != hwnd:
            user32.SetWindowLongPtrW(hwnd_p, -8, ctypes.c_void_p(owner))
    user32.SetWindowLongPtrW(hwnd_p, _GWL_STYLE, ctypes.c_void_p(style))
    user32.SetWindowLongPtrW(hwnd_p, _GWL_EXSTYLE, ctypes.c_void_p(exstyle))
    user32.SetWindowPos(
        hwnd_p, None, 0, 0, 0, 0,
        _SWP_NOMOVE_NOSIZE_NOZORDER_NOACTIVATE | _SWP_FRAMECHANGED)


class _Point(ctypes.Structure):
    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]


class _MinMaxInfo(ctypes.Structure):
    _fields_ = [
        ('ptReserved', _Point),
        ('ptMaxSize', _Point),
        ('ptMaxPosition', _Point),
        ('ptMinTrackSize', _Point),
        ('ptMaxTrackSize', _Point),
    ]


def _window_rect(hwnd):
    rect = _Rect()
    user32 = _user32
    user32.GetWindowRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(_Rect))
    user32.GetWindowRect.restype = ctypes.c_int
    if not user32.GetWindowRect(ctypes.c_void_p(int(hwnd)), ctypes.byref(rect)):
        return None
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return None
    return rect.left, rect.top, rect.right, rect.bottom


def _lparam_point(lparam):
    return (ctypes.c_int16(lparam & 0xFFFF).value,
            ctypes.c_int16((lparam >> 16) & 0xFFFF).value)


def _frame_hwnd(window):
    """The top-level window Windows frames.

    Tk's ``winfo_id`` is a child of that window. The frame is the first
    ancestor that is not itself a child window.
    """
    try:
        hwnd = int(window.winfo_id())
    except tk.TclError:
        return None
    user32 = _user32
    user32.GetParent.restype = ctypes.c_void_p
    user32.GetParent.argtypes = (ctypes.c_void_p,)
    user32.GetWindowLongPtrW.restype = ctypes.c_void_p
    user32.GetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int)
    seen = set()
    while hwnd and hwnd not in seen:
        seen.add(hwnd)
        style = int(user32.GetWindowLongPtrW(ctypes.c_void_p(hwnd), _GWL_STYLE) or 0)
        parent = int(user32.GetParent(ctypes.c_void_p(hwnd)) or 0)
        if not parent or not (style & _WS_CHILD):
            return hwnd
        parent_style = int(user32.GetWindowLongPtrW(ctypes.c_void_p(parent), _GWL_STYLE) or 0)
        if not (parent_style & _WS_CHILD):
            return parent
        hwnd = parent
    return hwnd


def _ensure_thick_frame(hwnd):
    if not hwnd:
        return
    user32 = _user32
    user32.GetWindowLongPtrW.restype = ctypes.c_void_p
    user32.GetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int)
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.SetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
    user32.SetWindowPos.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint)
    hwnd_p = ctypes.c_void_p(int(hwnd))
    style = int(user32.GetWindowLongPtrW(hwnd_p, _GWL_STYLE) or 0)
    if style & _WS_THICKFRAME:
        return
    user32.SetWindowLongPtrW(hwnd_p, _GWL_STYLE, ctypes.c_void_p(style | _WS_THICKFRAME))
    user32.SetWindowPos(
        hwnd_p, None, 0, 0, 0, 0,
        _SWP_NOMOVE_NOSIZE_NOZORDER_NOACTIVATE | _SWP_FRAMECHANGED)


def _rebind_frame(window):
    hook = getattr(window, '_size_hook', None)
    if hook is None:
        return
    outer = _frame_hwnd(window)
    if not outer:
        return
    if outer != hook.hwnd:
        previous = hook.hwnd
        _attach_hwnd(hook, outer)
        hook.hwnd = outer
        if _size_hooks.get(previous) is hook:
            _size_hooks.pop(previous, None)
    chrome = getattr(window, '_chrome', None)
    resize = chrome.get('resize', (1, 1)) if chrome else (1, 1)
    if resize[0] or resize[1]:
        _ensure_thick_frame(outer)


def _release_foreign_subclass(hwnd, window):
    """Drop a child subclass owned by a different themed window.

    A frame that later becomes its own window was subclassed as a child of
    the window it was created in. That registration hides the real frame
    from the caption hook.
    """
    hwnd = int(hwnd)
    hook = _size_hooks.get(hwnd)
    if hook is None or hook.window is window or hook.hwnd == hwnd:
        return
    hook.hwnds.discard(hwnd)
    _size_hooks.pop(hwnd, None)
    if _size_callback is None and _native_proc is None:
        return
    comctl = _comctl32
    comctl.RemoveWindowSubclass.restype = ctypes.c_int
    comctl.RemoveWindowSubclass.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    comctl.RemoveWindowSubclass(ctypes.c_void_p(hwnd), _subclass_pointer(), 1)


def _release_foreign_tree(window):
    def visit(widget):
        try:
            _release_foreign_subclass(int(widget.winfo_id()), window)
            children = widget.winfo_children()
        except (tk.TclError, TypeError, ValueError):
            return
        for child in children:
            visit(child)

    visit(window)


def _claim_frame_hook(window):
    """Subclass the outer frame of a window that already had chrome as a child."""
    if platform.system() != 'Windows' or getattr(window, '_chrome', None) is None:
        return
    if getattr(window, '_size_hook', None) is not None:
        return
    outer = _hwnd(window)
    if not outer:
        return
    existing = _size_hooks.get(int(outer))
    if existing is not None and existing.window is not window:
        return
    _release_foreign_tree(window)
    _install_size_hook(window)


def _arm_system_resize(window):
    """Subclass the real frame and give the client area its captionless shape."""
    if platform.system() != 'Windows':
        return
    _claim_frame_hook(window)
    if getattr(window, '_size_hook', None) is None:
        return
    _rebind_frame(window)
    _subclass_tree(window)
    _refresh_frame(window)


def _refresh_frame(window):
    """Ask Windows to recalculate the client area and recolor the border."""
    if platform.system() != 'Windows':
        return
    hook = getattr(window, '_size_hook', None)
    hwnd = hook.hwnd if hook is not None else _frame_hwnd(window)
    if not hwnd:
        return
    user32 = _user32
    user32.SetWindowPos.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint)
    user32.SetWindowPos(
        ctypes.c_void_p(int(hwnd)), None, 0, 0, 0, 0,
        _SWP_NOMOVE_NOSIZE_NOZORDER_NOACTIVATE | _SWP_FRAMECHANGED)
    _paint_native_border(window)


def _native_caption_extra(hwnd, inset):
    """Height Windows adds for a caption this frame does not keep.

    Tk sizes a window that has a minimum size, or is transient, as if the
    stock caption and resize border were still outside the client. This
    frame takes the caption into the client and keeps only ``inset`` at the
    bottom, so the client opens that much too tall.
    """
    user32 = _user32
    try:
        user32.GetDpiForWindow.restype = ctypes.c_uint
        user32.GetDpiForWindow.argtypes = (ctypes.c_void_p,)
        user32.GetSystemMetricsForDpi.restype = ctypes.c_int
        user32.GetSystemMetricsForDpi.argtypes = (ctypes.c_int, ctypes.c_uint)
        dpi = int(user32.GetDpiForWindow(ctypes.c_void_p(int(hwnd))) or 96)
        caption = int(user32.GetSystemMetricsForDpi(_SM_CYCAPTION, dpi))
        frame = int(user32.GetSystemMetricsForDpi(_SM_CXFRAME, dpi))
        pad = int(user32.GetSystemMetricsForDpi(_SM_CXPADDEDBORDER, dpi))
    except (AttributeError, OSError):
        caption = int(user32.GetSystemMetrics(_SM_CYCAPTION))
        frame = int(user32.GetSystemMetrics(_SM_CXFRAME))
        pad = int(user32.GetSystemMetrics(_SM_CXPADDEDBORDER))
    return caption + 2 * max(frame + pad, 0) - int(inset)


def _correct_opening_height(window):
    """Drop the stock caption out of the height Tk just applied.

    A later geometry change is not padded, and neither is a window the user
    has resized. Only the height that appears on the first map is.
    """
    if platform.system() != 'Windows':
        return
    chrome = getattr(window, '_chrome', None)
    hook = getattr(window, '_size_hook', None)
    if chrome is None or hook is None or chrome.get('opening_sized') or chrome.get('zoomed'):
        return
    try:
        if window.wm_state() != 'normal':
            return
        actual = int(window.winfo_height())
        asked = getattr(window, '_chrome_asked', None)
        target = int(asked[1]) if asked else int(window.winfo_reqheight())
    except (tk.TclError, TypeError, ValueError, IndexError):
        return
    if actual <= 1 or target <= 1:
        return
    chrome['opening_sized'] = True
    extra = _native_caption_extra(hook.hwnd, _edge_inset(hook))
    if extra <= 0 or actual != target + extra:
        return
    rect = _window_rect(hook.hwnd)
    if rect is None:
        return
    left, top, right, bottom = rect
    user32 = _user32
    user32.SetWindowPos.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint)
    user32.SetWindowPos(
        ctypes.c_void_p(int(hook.hwnd)), None,
        int(left), int(top), int(right - left), int(bottom - top - extra),
        _SWP_NOZORDER_NOACTIVATE)


def _native_resize_band(hwnd):
    """The width Windows uses for a normal window's resize border."""
    user32 = _user32
    try:
        user32.GetDpiForWindow.restype = ctypes.c_uint
        user32.GetDpiForWindow.argtypes = (ctypes.c_void_p,)
        user32.GetSystemMetricsForDpi.restype = ctypes.c_int
        user32.GetSystemMetricsForDpi.argtypes = (ctypes.c_int, ctypes.c_uint)
        dpi = int(user32.GetDpiForWindow(ctypes.c_void_p(hwnd)) or 96)
        frame = int(user32.GetSystemMetricsForDpi(_SM_CXFRAME, dpi))
        pad = int(user32.GetSystemMetricsForDpi(_SM_CXPADDEDBORDER, dpi))
    except (AttributeError, OSError):
        frame = int(user32.GetSystemMetrics(_SM_CXFRAME))
        pad = int(user32.GetSystemMetrics(_SM_CXPADDEDBORDER))
    return max(frame + pad, 4)


def _install_size_hook(window):
    """Subclass the frame so the client covers the stock title bar.

    The left, right, and bottom borders stay native, and Windows resizes them.
    """
    if platform.system() != 'Windows':
        return
    hwnd = _hwnd(window)
    if not hwnd or int(hwnd) in _size_hooks:
        return
    comctl = _comctl32
    comctl.SetWindowSubclass.restype = ctypes.c_int
    comctl.SetWindowSubclass.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ssize_t)
    comctl.DefSubclassProc.restype = ctypes.c_ssize_t
    comctl.DefSubclassProc.argtypes = (
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_ssize_t, ctypes.c_ssize_t)
    hook = _SizeHook(window, int(hwnd), _native_resize_band(hwnd))
    _publish_native_insets(hook)
    if not comctl.SetWindowSubclass(
            ctypes.c_void_p(hwnd), _subclass_pointer(), 1, ctypes.addressof(hook.native)):
        return
    _size_hooks[int(hwnd)] = hook
    window._size_hook = hook
    _sync_size_hook(window)


class _SizeHook:
    def __init__(self, window, hwnd, band):
        self.window = window
        self.hwnd = hwnd
        self.native = _NativeChrome()
        self.native.hwnd = int(hwnd)
        self.band = band
        self.hwnds = {hwnd}
        self.left = 0
        self.top = 0
        self.right = 0
        self.bottom = 0
        self.buttons = ()
        self.button_hwnds = ()
        self.sizing = None
        self.min_size = (160, 64)
        self.title_height = 32


def _sync_size_hook_event(window, event):
    if event.widget is window:
        _sync_size_hook(window)


def _sync_size_hook(window):
    hook = getattr(window, '_size_hook', None)
    chrome = getattr(window, '_chrome', None)
    if hook is None or chrome is None:
        return
    _rebind_frame(window)
    try:
        rect = _window_rect(hook.hwnd)
        if rect is None:
            left = window.winfo_rootx()
            top = window.winfo_rooty()
            rect = (left, top, left + window.winfo_width(), top + window.winfo_height())
        hook.left, hook.top, hook.right, hook.bottom = rect
        rects = []
        hwnds = []
        for button in chrome['buttons'].values():
            try:
                hwnds.append(int(button.winfo_id()))
            except tk.TclError:
                continue
            if not button.winfo_ismapped():
                continue
            x = button.winfo_rootx()
            y = button.winfo_rooty()
            rects.append((x, y, x + button.winfo_width(), y + button.winfo_height()))
        hook.buttons = tuple(rects)
        hook.button_hwnds = tuple(hwnds)
        try:
            measured = int(chrome['title'].winfo_height())
            if measured >= 8:
                hook.title_height = measured
        except (tk.TclError, KeyError, TypeError, ValueError):
            pass
        button = chrome.get('buttons', {}).get('max')
        if button is not None:
            _refresh_caption_glyph(button)
        try:
            width, height = window.minsize()
            hook.min_size = (max(int(width), 160), max(int(height), 64))
        except (tk.TclError, TypeError, ValueError):
            hook.min_size = (160, 64)
        _publish_native_insets(hook)
    except tk.TclError:
        pass


def _button_rects(hook):
    rects = list(hook.buttons)
    user32 = _user32
    user32.IsWindowVisible.argtypes = (ctypes.c_void_p,)
    user32.IsWindowVisible.restype = ctypes.c_int
    if not user32.IsWindowVisible(ctypes.c_void_p(hook.hwnd)):
        return rects
    for button_hwnd in hook.button_hwnds:
        button_rect = _window_rect(button_hwnd)
        if button_rect is None:
            continue
        if button_rect[2] - button_rect[0] < 4 or button_rect[3] - button_rect[1] < 4:
            continue
        rects.append(button_rect)
    return rects


def _edge_inset(hook):
    """Native border thickness. A fixed window keeps a 1-pixel edge."""
    chrome = getattr(hook.window, '_chrome', None)
    resize = chrome.get('resize', (1, 1)) if chrome else (1, 1)
    if resize[0] or resize[1]:
        return max(int(hook.band), 1)
    return 1


def _inset_rect(rect, inset, top_inset):
    left, top, right, bottom = (int(value) for value in rect)
    left += inset
    top += top_inset
    right -= inset
    bottom -= inset
    if right < left:
        right = left
    if bottom < top:
        bottom = top
    return left, top, right, bottom


def _apply_client_rect(hwnd, wparam, lparam, hook):
    """Cover the stock title bar and keep the system resize border.

    The top of a restored window is the client area, so the painted title
    sits there. A maximized window insets the top too, or the border covers
    the work area.
    """
    user32 = _user32
    user32.IsZoomed.argtypes = (ctypes.c_void_p,)
    user32.IsZoomed.restype = ctypes.c_int
    inset = _edge_inset(hook)
    top_inset = inset if user32.IsZoomed(ctypes.c_void_p(int(hwnd))) else 0
    if int(wparam):
        rects = (_Rect * 3).from_address(int(lparam))
        src = rects[0]
        left, top, right, bottom = _inset_rect(
            (src.left, src.top, src.right, src.bottom), inset, top_inset)
        rects[0] = _Rect(left, top, right, bottom)
        return
    rect = _Rect.from_address(int(lparam))
    left, top, right, bottom = _inset_rect(
        (rect.left, rect.top, rect.right, rect.bottom), inset, top_inset)
    rect.left, rect.top, rect.right, rect.bottom = left, top, right, bottom


def _title_box(hook, bounds):
    """Title rectangle from the last measured height.

    The window procedure must not ask Tk. A Tk call from there clears the
    thread state Tk saved for the event loop, and the next default-procedure
    call aborts the interpreter.
    """
    left, top, right, bottom = bounds
    inset = _edge_inset(hook)
    height = hook.title_height if hook.title_height >= 8 else 32
    return left + inset, top, right - inset, top + height


def _caption_hit(hook, x, y, bounds):
    """Hit code for the painted title, or None to leave the point alone.

    The top strip of the title resizes the window. The rest drags it.
    Caption buttons stay client area so they still click.
    """
    chrome = getattr(hook.window, '_chrome', None)
    if chrome is None:
        return None
    for button_left, button_top, button_right, button_bottom in _button_rects(hook):
        if button_left <= x < button_right and button_top <= y < button_bottom:
            return None
    left, top, right, bottom = bounds
    if x < left or y < top or x >= right or y >= bottom:
        return None
    _horizontal, vertical = chrome.get('resize', (1, 1))
    inset = _edge_inset(hook)
    box_left, box_top, box_right, box_bottom = _title_box(hook, bounds)
    in_title = box_left <= x < box_right and box_top <= y < box_bottom
    if not in_title:
        return None
    if vertical and not chrome.get('zoomed') and (y - top) < inset:
        return _HTTOP
    return _HTCAPTION


def _paint_native_border(window):
    """Color the native border and keep Windows 11's rounded corners."""
    if platform.system() != 'Windows':
        return
    hook = getattr(window, '_size_hook', None)
    hwnd = hook.hwnd if hook is not None else _frame_hwnd(window)
    if not hwnd:
        return
    chrome = getattr(window, '_chrome', None)
    color = '#000000'
    if chrome is not None:
        color = chrome.get('palette', {}).get('edge', color)
    text = str(color).lstrip('#')
    if len(text) != 6:
        return
    red = int(text[0:2], 16)
    green = int(text[2:4], 16)
    blue = int(text[4:6], 16)
    colorref = red | (green << 8) | (blue << 16)
    try:
        dwmapi = _dwmapi
        dwmapi.DwmSetWindowAttribute.argtypes = (
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint)
        dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long
        painted = ctypes.c_uint(colorref)
        corner = ctypes.c_int(_DWMWCP_ROUND)
        target = ctypes.c_void_p(int(hwnd))
        dwmapi.DwmSetWindowAttribute(
            target, _DWMWA_BORDER_COLOR, ctypes.byref(painted), ctypes.sizeof(painted))
        dwmapi.DwmSetWindowAttribute(
            target, _DWMWA_WINDOW_CORNER_PREFERENCE, ctypes.byref(corner), ctypes.sizeof(corner))
    except (AttributeError, OSError):
        pass


def _note_zoomed(hook, hwnd):
    """Record whether Windows has the frame maximized.

    This only writes a Python value. The window procedure must not touch Tk.
    """
    chrome = getattr(hook.window, '_chrome', None)
    if chrome is None:
        return
    user32 = _user32
    user32.IsZoomed.argtypes = (ctypes.c_void_p,)
    user32.IsZoomed.restype = ctypes.c_int
    chrome['zoomed'] = bool(user32.IsZoomed(ctypes.c_void_p(int(hwnd))))
    _publish_native_insets(hook)


class _NativeChrome(ctypes.Structure):
    """Insets the native frame reads while a drag has the interpreter detached.

    A drag runs inside ``DefSubclassProc``. Windows sends more messages before
    that call returns, and entering Python for those attaches a thread state
    that is already saved. The subclass thunk handles the size message itself
    and forwards everything else without entering Python.
    """

    _fields_ = (
        ('inset', ctypes.c_int),
        ('top_inset', ctypes.c_int),
        ('hwnd', ctypes.c_void_p),
        ('skip_nccalc', ctypes.c_int),
    )


_dispatch_flag = ctypes.c_int(0)
_native_proc = None


def _subclass_machine_code(flag_addr, python_addr, defproc_addr):
    """x64 thunk. A non-zero flag means a default-procedure call is in progress."""
    code = bytearray()

    def emit(*values):
        code.extend(values)

    def imm64(value):
        code.extend(int(value).to_bytes(8, 'little'))

    emit(0x48, 0xB8)
    imm64(flag_addr)
    emit(0x83, 0x38, 0x00)
    jump_python = len(code)
    emit(0x74, 0x00)
    emit(0x48, 0x8B, 0x44, 0x24, 0x30)
    emit(0x48, 0x85, 0xC0)
    jump_default = [len(code)]
    emit(0x74, 0x00)
    emit(0x48, 0x39, 0x48, 0x08)
    jump_default.append(len(code))
    emit(0x75, 0x00)
    emit(0x81, 0xFA, 0x83, 0x00, 0x00, 0x00)
    jump_default.append(len(code))
    emit(0x75, 0x00)
    emit(0x83, 0x78, 0x10, 0x00)
    jump_default.append(len(code))
    emit(0x75, 0x00)
    emit(0x44, 0x8B, 0x10)
    emit(0x44, 0x8B, 0x58, 0x04)
    emit(0x45, 0x01, 0x11)
    emit(0x45, 0x01, 0x59, 0x04)
    emit(0x45, 0x29, 0x51, 0x08)
    emit(0x45, 0x29, 0x51, 0x0C)
    emit(0x31, 0xC0)
    emit(0xC3)
    default_at = len(code)
    emit(0x48, 0xB8)
    imm64(defproc_addr)
    emit(0xFF, 0xE0)
    python_at = len(code)
    emit(0x48, 0xB8)
    imm64(python_addr)
    emit(0xFF, 0xE0)

    def patch(at, target):
        relative = target - (at + 2)
        if not -128 <= relative <= 127:
            raise RuntimeError('subclass thunk jump is out of range')
        code[at + 1] = relative & 0xFF

    patch(jump_python, python_at)
    for at in jump_default:
        patch(at, default_at)
    return bytes(code)


def _subclass_address():
    """The subclass procedure Windows calls. Nested messages stay out of Python."""
    global _size_callback, _native_proc
    if _native_proc:
        return int(_native_proc)
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint,
        ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_size_t, ctypes.c_ssize_t)
    if _size_callback is None:
        _size_callback = callback_type(_size_wndproc)
    python_addr = int(ctypes.cast(_size_callback, ctypes.c_void_p).value or 0)
    if ctypes.sizeof(ctypes.c_void_p) != 8 or not python_addr:
        return python_addr
    kernel32 = ctypes.WinDLL('kernel32')
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleHandleW.argtypes = (ctypes.c_wchar_p,)
    kernel32.GetProcAddress.restype = ctypes.c_void_p
    kernel32.GetProcAddress.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    defproc = kernel32.GetProcAddress(
        kernel32.GetModuleHandleW('comctl32'), b'DefSubclassProc')
    if not defproc:
        return python_addr
    code = _subclass_machine_code(ctypes.addressof(_dispatch_flag), python_addr, int(defproc))
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = (
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint)
    memory = kernel32.VirtualAlloc(None, len(code), 0x3000, 0x40)
    if not memory:
        return python_addr
    ctypes.memmove(memory, code, len(code))
    kernel32.FlushInstructionCache.restype = ctypes.c_int
    kernel32.FlushInstructionCache.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.FlushInstructionCache(kernel32.GetCurrentProcess(), memory, len(code))
    _native_proc = int(memory)
    return _native_proc


def _subclass_pointer():
    return ctypes.c_void_p(_subclass_address())


def _publish_native_insets(hook):
    native = getattr(hook, 'native', None)
    if native is None:
        return
    inset = _edge_inset(hook)
    chrome = getattr(hook.window, '_chrome', None)
    zoomed = bool(chrome.get('zoomed')) if chrome else False
    native.inset = int(inset)
    native.top_inset = int(inset if zoomed else 0)
    native.hwnd = int(hook.hwnd)


def _default_proc(hwnd, msg, wparam, lparam):
    """Forward a message with the interpreter lock released.

    Tk restores its saved thread state from inside this call. The lock has to
    be free, and this function must not have called into Tk first: that clears
    the saved state and the restore aborts the interpreter. The flag tells the
    subclass thunk to keep the messages this call provokes out of Python.
    """
    _dispatch_flag.value += 1
    try:
        return _comctl32_dispatch.DefSubclassProc(hwnd, msg, wparam, lparam)
    finally:
        _dispatch_flag.value -= 1


def _size_wndproc(hwnd, msg, wparam, lparam, subclass_id, refdata):
    key = int(hwnd)
    hook = _size_hooks.get(key)
    if hook is None:
        return _default_proc(hwnd, msg, wparam, lparam)
    if msg == _WM_NCDESTROY:
        _size_hooks.pop(key, None)
        hook.hwnds.discard(key)
        return _default_proc(hwnd, msg, wparam, lparam)
    root = key == hook.hwnd
    if root and msg == _WM_NCCALCSIZE:
        native = getattr(hook, 'native', None)
        if native is not None and native.skip_nccalc:
            return _default_proc(hwnd, msg, wparam, lparam)
        _apply_client_rect(hwnd, wparam, lparam, hook)
        return 0
    if root and msg == _WM_GETMINMAXINFO:
        result = _default_proc(hwnd, msg, wparam, lparam)
        info = ctypes.cast(lparam, ctypes.POINTER(_MinMaxInfo)).contents
        track = info.ptMaxTrackSize
        track.x = max(int(track.x), _TRACK_LIMIT)
        track.y = max(int(track.y), _TRACK_LIMIT)
        info.ptMaxTrackSize = track
        return result
    if root and msg == _WM_WINDOWPOSCHANGED:
        # Maximizing does not deliver WM_SIZE to this frame. The zoomed flag has
        # to be set before the default procedure runs, because that is what
        # tells Tk to fire Configure, and the max button glyph is updated there.
        _note_zoomed(hook, hwnd)
        return _default_proc(hwnd, msg, wparam, lparam)
    if root and msg == _WM_SIZE:
        result = _default_proc(hwnd, msg, wparam, lparam)
        chrome = getattr(hook.window, '_chrome', None)
        if chrome is not None:
            chrome['zoomed'] = int(wparam) == _SIZE_MAXIMIZED
        return result
    if root and msg == _WM_NCLBUTTONDBLCLK and int(wparam) == _HTCAPTION:
        chrome = getattr(hook.window, '_chrome', None) or {}
        if 'max' in chrome.get('buttons', {}):
            _toggle_zoom(hook.window)
        return 0
    if root and msg == _WM_NCLBUTTONDOWN and _remote_session():
        command = _resize_command(int(wparam))
        if command is not None:
            _begin_system_drag(key, command, lparam)
            return 0
    if msg == _WM_NCHITTEST:
        bounds = _window_rect(hook.hwnd)
        if bounds is None:
            return _default_proc(hwnd, msg, wparam, lparam)
        x, y = _lparam_point(lparam)
        # An open menu has to see the click. Caption and border hits are
        # otherwise handled by Windows and never reach Tk.
        menu_open = _context_menu_open()
        if not root:
            hit = _move_hit(_caption_hit(hook, x, y, bounds), menu_open)
            if hit == _HTCLIENT:
                return _HTCLIENT
            if hit is not None:
                return _HTTRANSPARENT
            return _default_proc(hwnd, msg, wparam, lparam)
        chrome = getattr(hook.window, '_chrome', None) or {}
        resize = chrome.get('resize', (1, 1))
        if resize[0] or resize[1]:
            _ensure_thick_frame(hook.hwnd)
        result = _default_proc(hwnd, msg, wparam, lparam)
        if menu_open and result not in (_HTNOWHERE, _HTCLIENT):
            return _HTCLIENT
        if result not in (_HTNOWHERE, _HTCLIENT):
            return result
        hit = _move_hit(_caption_hit(hook, x, y, bounds), menu_open)
        if hit is not None:
            return hit
        return result
    return _default_proc(hwnd, msg, wparam, lparam)


def _attach_hwnd(hook, hwnd):
    hwnd = int(hwnd)
    if hwnd in hook.hwnds or hwnd in _size_hooks:
        return
    comctl = _comctl32
    comctl.SetWindowSubclass.restype = ctypes.c_int
    comctl.SetWindowSubclass.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ssize_t)
    refdata = ctypes.addressof(hook.native) if getattr(hook, 'native', None) is not None else 0
    if not comctl.SetWindowSubclass(ctypes.c_void_p(hwnd), _subclass_pointer(), 1, refdata):
        return
    hook.hwnds.add(hwnd)
    _size_hooks[hwnd] = hook


def _attach_widget(widget):
    if platform.system() != 'Windows':
        return
    try:
        top = widget.winfo_toplevel()
        hwnd = int(widget.winfo_id())
    except tk.TclError:
        return
    hook = getattr(top, '_size_hook', None)
    if hook is None:
        return
    _attach_hwnd(hook, hwnd)


def _subclass_tree(window):
    hook = getattr(window, '_size_hook', None)
    if hook is None:
        return

    def visit(widget):
        try:
            _attach_hwnd(hook, int(widget.winfo_id()))
            children = widget.winfo_children()
        except tk.TclError:
            return
        for child in children:
            visit(child)

    visit(window)


def _remove_size_hook(window):
    hook = getattr(window, '_size_hook', None)
    if hook is None:
        return
    window._size_hook = None
    hwnds = list(hook.hwnds)
    for hwnd in hwnds:
        _size_hooks.pop(hwnd, None)
    hook.hwnds.clear()
    if platform.system() != 'Windows' or (_size_callback is None and _native_proc is None):
        return
    comctl = _comctl32
    comctl.RemoveWindowSubclass.restype = ctypes.c_int
    comctl.RemoveWindowSubclass.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    proc = _subclass_pointer()
    for hwnd in hwnds:
        comctl.RemoveWindowSubclass(ctypes.c_void_p(hwnd), proc, 1)


def _hwnd(window):
    if platform.system() != 'Windows':
        return None
    try:
        widget_id = window.winfo_id()
    except tk.TclError:
        return None
    user32 = _user32
    user32.GetAncestor.restype = ctypes.c_void_p
    user32.GetAncestor.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    root = user32.GetAncestor(ctypes.c_void_p(widget_id), 2)
    return int(root or widget_id)


_x11 = None
_x11_display = None


def _x11_lib():
    """The X11 connection, or nothing when this is not an X11 session."""
    global _x11, _x11_display
    if platform.system() != 'Linux':
        return None, None
    if _x11 is False:
        return None, None
    if _x11 is not None:
        return _x11, _x11_display
    try:
        lib = ctypes.CDLL('libX11.so.6')
    except OSError:
        _x11 = False
        return None, None
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XOpenDisplay.argtypes = (ctypes.c_char_p,)
    lib.XInternAtom.restype = ctypes.c_ulong
    lib.XInternAtom.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int)
    lib.XChangeProperty.restype = ctypes.c_int
    lib.XChangeProperty.argtypes = (
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    lib.XFlush.argtypes = (ctypes.c_void_p,)
    lib.XDefaultScreen.restype = ctypes.c_int
    lib.XDefaultScreen.argtypes = (ctypes.c_void_p,)
    lib.XRootWindow.restype = ctypes.c_ulong
    lib.XRootWindow.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.XQueryPointer.restype = ctypes.c_int
    lib.XQueryPointer.argtypes = (
        ctypes.c_void_p, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint))
    display = lib.XOpenDisplay(None)
    if not display:
        _x11 = False
        return None, None
    _x11 = lib
    _x11_display = display
    return lib, display


def _install_linux_frame(window):
    """Remove the server title bar and advertise a 1-pixel border.

    Returns false when there is no X11 connection, so the caller can keep a
    borderless window instead of showing two title bars.
    """
    lib, display = _x11_lib()
    try:
        xid = int(window.winfo_id())
    except (tk.TclError, ValueError, TypeError):
        return False
    if lib is None or not display or not xid:
        return False
    motif = lib.XInternAtom(display, b'_MOTIF_WM_HINTS', False)
    hints = (ctypes.c_ulong * 5)(3, 1, 0, 0, 0)
    lib.XChangeProperty(
        display, xid, motif, motif, 32, 0, ctypes.cast(hints, ctypes.c_void_p), 5)
    extents_atom = lib.XInternAtom(display, b'_GTK_FRAME_EXTENTS', False)
    cardinal = lib.XInternAtom(display, b'CARDINAL', False)
    extents = (ctypes.c_ulong * 4)(1, 1, 1, 1)
    lib.XChangeProperty(
        display, xid, extents_atom, cardinal, 32, 0,
        ctypes.cast(extents, ctypes.c_void_p), 4)
    lib.XFlush(display)
    return True


def _paint(window, palette):
    chrome = window._chrome
    chrome['palette'] = palette
    edge = palette['edge']
    title = palette['title']
    fg = palette['foreground']
    window.configure(background=edge)
    for name in ('north', 'south', 'west', 'east', 'separator'):
        chrome[name].configure(background=edge)
    chrome['center'].configure(background=edge)
    chrome['title'].configure(background=title)
    chrome['client'].configure(background=palette['background'])
    chrome['icon'].configure(background=title)
    chrome['text'].configure(background=title, foreground=fg)
    for name, button in chrome['buttons'].items():
        button.configure(background=title, foreground=fg)
        button._caption_shown = None
        _refresh_caption_glyph(button)
        _bind_hover(button, name == 'close', palette)
    _paint_native_border(window)


def _bind_hover(button, close, palette):
    hover = _CLOSE_HOVER if close else palette['active']
    hover_text = _CLOSE_HOVER_TEXT if close else palette['foreground']

    def enter(_event):
        button.configure(background=hover, foreground=hover_text)
        _refresh_caption_glyph(button)

    def leave(_event):
        current = button.winfo_toplevel()._chrome['palette']
        button.configure(background=current['title'], foreground=current['foreground'])
        _refresh_caption_glyph(button)

    button.bind('<Enter>', enter)
    button.bind('<Leave>', leave)


def _starting_palette(window):
    if _pending is not None:
        return _palette(window, *_pending)
    background = _themedialog.background() or '#f0f0f0'
    foreground = _themedialog.foreground() or '#000000'
    chrome = _themedialog.edge()
    return _palette(window, background, foreground, background, chrome)


def _palette(window, background, foreground, window_background=None, window_chrome=None):
    field = _rgb(window, background)
    fg = _rgb(window, foreground)
    bg = _rgb(window, window_background or background)
    if window_chrome:
        chrome = _hex(_rgb(window, window_chrome))
    else:
        chrome = _hex(_mix(bg, fg, 0.35))
    return {
        'background': _hex(bg),
        'foreground': _hex(fg),
        'edge': chrome,
        'title': chrome if window_chrome else _hex(bg),
        'active': _hex(_mix(_rgb(window, chrome), fg, 0.26)),
    }


_icon_source = None


def _icon_image(master):
    global _icon_source
    try:
        import PIL.Image
        import PIL.ImageTk
        if _icon_source is None:
            import dgenerate.resources as resources
            image = PIL.Image.open(io.BytesIO(resources.get_icon_data()))
            _icon_source = image.resize((16, 16), PIL.Image.Resampling.LANCZOS)
        return PIL.ImageTk.PhotoImage(_icon_source, master=master)
    except Exception:
        return None


def _unbind_pointer(window):
    for sequence, name in (
            ('<ButtonRelease-1>', '_chrome_release'),
            ('<B1-Motion>', '_chrome_motion')):
        token = getattr(window, name, None)
        if not token:
            continue
        try:
            window.unbind(sequence, token)
        except tk.TclError:
            pass
        try:
            delattr(window, name)
        except AttributeError:
            pass


def _remove(window):
    _remove_size_hook(window)
    chrome = getattr(window, '_chrome', None)
    if chrome is None:
        return
    _unbind_pointer(window)
    _cancel_taskbar(window, chrome)
    window._chrome = None
    window._chrome_client = None
    if window in _installed:
        _installed.remove(window)
    original = chrome.get('wm_title')
    if original is not None:
        window.wm_title = original
        window.title = original
    for name in ('grid_rowconfigure', 'rowconfigure', 'grid_columnconfigure', 'columnconfigure',
                 'configure', 'config'):
        if hasattr(window, name):
            try:
                delattr(window, name)
            except AttributeError:
                pass
    for widget in chrome['widgets']:
        try:
            widget.destroy()
        except tk.TclError:
            pass


def _forget(window, event):
    if event.widget is not window:
        return
    _remove_size_hook(window)
    _unbind_pointer(window)
    chrome = getattr(window, '_chrome', None)
    _cancel_taskbar(window, chrome)
    if chrome is not None and window in _installed:
        _installed.remove(window)
    window._chrome = None
    window._chrome_client = None


def _rgb(widget, color):
    red, green, blue = widget.winfo_rgb(color)
    return red / 65535, green / 65535, blue / 65535


def _mix(start, end, amount):
    return tuple(start[i] + (end[i] - start[i]) * amount for i in range(3))


def _hex(color):
    return '#{:02x}{:02x}{:02x}'.format(*(
        round(min(1, max(0, channel)) * 255) for channel in color
    ))


if _CUSTOM:
    _ensure_patched()
