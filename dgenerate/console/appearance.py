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

"""Aqua dark-mode colors for the classic widgets ttk does not cover.

ttk on the aqua theme follows the system appearance. Classic frames, labels,
and canvases do not, unless they are given a semantic system color. Those
color names keep tracking light and dark mode after they are applied.
"""

import tkinter as tk
import tkinter.ttk as ttk

_PLAIN_BACKGROUND = {
    'white', '#ffffff', '#fff',
    'systembuttonface',
    '#d9d9d9', '#dcdcdc', '#dddddd', '#e0e0e0', '#e8e8e8', '#ececec', '#f0f0f0',
    'grey85', 'gray85', 'grey86', 'gray86',
}
_PLAIN_FOREGROUND = {
    'black', '#000000', '#000',
    'systembuttontext', 'systemwindowtext',
}


def is_aqua(widget) -> bool:
    """True when this Tk was built for macOS Aqua."""
    try:
        return widget.tk.call('tk', 'windowingsystem') == 'aqua'
    except tk.TclError:
        return False


def field_background() -> str:
    """Background for a text field that should follow the system appearance."""
    return 'systemTextBackgroundColor'


def text_foreground() -> str:
    """Foreground for text that should follow the system appearance."""
    return 'systemLabelColor'


def error_foreground() -> str:
    """Foreground for validation errors that remains visible in either appearance."""
    return 'systemRedColor'


def install(widget) -> None:
    """
    Apply dynamic Aqua system colors to a window's classic widgets.

    Tk resolves semantic system colors against the current macOS appearance and
    redraws windows when that appearance changes. No effect on Windows or X11.
    """
    if widget is None or not is_aqua(widget):
        return
    style = ttk.Style(widget)
    layout = style.layout('TCheckbutton')
    style.layout('TCheckbutton', _remove_checkbutton_focus(layout))
    apply(widget.winfo_toplevel())


def install_checkbutton_fit(widget):
    """Remove the ttk.Checkbutton focus ring on every platform.

    Classic Tk checkboxes do not draw a focus ring, but every ttk theme except
    classic adds a ``Checkbutton.focus`` element (vista on Windows,
    ``xpnative``/``clam`` on macOS, ``clam``/``default`` on Linux), so the
    same checkbox renders with an extra ring. Stripping that element from the
    ``TCheckbutton`` layout leaves no focus element, so the checkbox matches
    classic Tk on all platforms. This is global (applied by the console before
    any widgets are created) so it also covers every dialog and ``Toplevel``.
    """
    if widget is None:
        return
    style = ttk.Style(widget)
    style.layout('TCheckbutton', _remove_checkbutton_focus(style.layout('TCheckbutton')))


def _remove_checkbutton_focus(layout):
    result = []
    for element, options in layout:
        children = options.get('children', [])
        child_layout = _remove_checkbutton_focus(children)
        if element == 'Checkbutton.focus':
            parent_options = {key: value for key, value in options.items() if key != 'children'}
            for child_element, child_options in child_layout:
                result.append((child_element, {**parent_options, **child_options}))
            continue

        updated_options = dict(options)
        if 'children' in options:
            updated_options['children'] = child_layout
        result.append((element, updated_options))
    return result


def apply(widget) -> None:
    """Repaint one window's classic containers. No effect outside Aqua."""
    if widget is None or not is_aqua(widget):
        return
    _paint(widget)
    if isinstance(widget, tk.Menu):
        return
    for child in list(widget.winfo_children()):
        apply(child)


def _paint(widget) -> None:
    if isinstance(widget, (tk.Text, tk.Entry, tk.Spinbox, tk.Listbox, tk.Menu, ttk.Widget)):
        return
    if isinstance(widget, tk.PanedWindow):
        _configure(widget, background='systemSeparatorColor')
        return
    if not isinstance(widget, (tk.Tk, tk.Toplevel, tk.Frame, tk.Label, tk.Canvas, tk.Checkbutton, tk.Button)):
        return

    background = _cget(widget, 'background')
    if background is not None and _plain_background(background):
        _configure(widget, background='systemWindowBackgroundColor')
    if isinstance(widget, tk.Label):
        foreground = _cget(widget, 'foreground')
        if foreground is not None and _plain_foreground(foreground):
            _configure(widget, foreground='systemLabelColor')


def _plain_background(value: str) -> bool:
    text = value.lower()
    if text.startswith('system'):
        return False
    return text in _PLAIN_BACKGROUND


def _plain_foreground(value: str) -> bool:
    text = value.lower()
    if text.startswith('system'):
        return False
    return text in _PLAIN_FOREGROUND


def _cget(widget, option: str):
    try:
        return str(widget.cget(option))
    except tk.TclError:
        return None


def _configure(widget, **options) -> None:
    try:
        widget.configure(**options)
    except tk.TclError:
        pass


def install_button_fitting(widget):
    """Fit every ``ttk.Button`` on this window to its text, on all platforms.

    A classic ``tk.Button`` sizes itself to its label, but ``ttk.Button``
    instead uses a theme-defined ``width`` (about ten columns), so the same
    label renders far wider. This rewrites the ``TButton`` layout without a
    ``Button.padding`` element and clears the theme ``width``, so the button
    shrinks to fit its text on Windows, Linux, and macOS alike. A per-instance
    ``width`` or ``style`` (e.g. the spin-box buttons) still overrides this, so
    explicitly-sized buttons are unchanged.
    """
    if widget is None:
        return
    style = ttk.Style(widget)
    style.layout('TButton', _without_button_padding(style.layout('TButton')))
    style.configure('TButton', width=0, anchor='center', justify='center')


def _without_button_padding(layout):
    """Return a copy of a ttk layout with every ``Button.padding`` removed.

    ``Button.padding`` can pin a minimum width in some themes, which would
    defeat the fit-to-text behavior, so it is stripped out while preserving
    its children in place.
    """
    result = []
    for element, options in layout:
        if element == 'Button.padding':
            for child_element, child_options in options.get('children') or ():
                result.append((child_element, dict(child_options)))
            continue
        options = dict(options)
        if options.get('children') is not None:
            options['children'] = _without_button_padding(options['children'])
        result.append((element, options))
    return result
