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

"""Scrollbars colored from the editor theme.

The native ttk scrollbar is drawn by the platform theme (vista, aqua) and
ignores color options. These styles reuse the clam scrollbar elements, which
are drawn from ``troughcolor`` and ``background``. Dialog chrome is colored
separately; these styles only replace the scrollbar.
"""

import tkinter as tk
import tkinter.ttk as ttk

VERTICAL_STYLE = 'Dgenerate.Vertical.TScrollbar'
HORIZONTAL_STYLE = 'Dgenerate.Horizontal.TScrollbar'

_installed_theme = None

_ELEMENTS = (
    ('Vertical.Scrollbar.trough', 'Dgenerate.Vertical.Scrollbar.trough'),
    ('Vertical.Scrollbar.thumb', 'Dgenerate.Vertical.Scrollbar.thumb'),
    ('Horizontal.Scrollbar.trough', 'Dgenerate.Horizontal.Scrollbar.trough'),
    ('Horizontal.Scrollbar.thumb', 'Dgenerate.Horizontal.Scrollbar.thumb'),
)


def install(widget) -> None:
    """Create the clam-backed scrollbar styles for the current ttk theme.

    Elements belong to the theme that was current when they were created.
    Switching themes (dialog theming uses clam) drops them, so this runs
    again after a theme change.
    """
    global _installed_theme
    if widget is None:
        return
    style = ttk.Style(widget)
    theme = style.theme_use()
    if _installed_theme == theme:
        return
    for source, name in _ELEMENTS:
        try:
            style.element_create(name, 'from', 'clam', source)
        except tk.TclError:
            pass
    style.layout(VERTICAL_STYLE, [
        ('Dgenerate.Vertical.Scrollbar.trough', {
            'sticky': 'ns',
            'children': [
                ('Dgenerate.Vertical.Scrollbar.thumb', {
                    'sticky': 'nswe',
                    'unit': '1',
                }),
            ],
        }),
    ])
    style.layout(HORIZONTAL_STYLE, [
        ('Dgenerate.Horizontal.Scrollbar.trough', {
            'sticky': 'ew',
            'children': [
                ('Dgenerate.Horizontal.Scrollbar.thumb', {
                    'sticky': 'nswe',
                    'unit': '1',
                }),
            ],
        }),
    ])
    _installed_theme = theme


def scrollbar(master, orient=tk.VERTICAL, **kwargs):
    """A scrollbar whose colors follow :func:`apply`."""
    install(master)
    vertical = orient in (tk.VERTICAL, 'vertical')
    return ttk.Scrollbar(
        master,
        orient=orient,
        style=VERTICAL_STYLE if vertical else HORIZONTAL_STYLE,
        **kwargs,
    )


def apply(widget, background, foreground) -> None:
    """Color the themed scrollbars from an editor background and foreground."""
    if widget is None:
        return
    install(widget)
    background_rgb = _rgb(widget, background)
    foreground_rgb = _rgb(widget, foreground)
    dark = _luma(background_rgb) < 0.5
    thumb = _mix(background_rgb, foreground_rgb, 0.45 if dark else 0.28)
    active = _mix(background_rgb, foreground_rgb, 0.68 if dark else 0.45)
    trough = _hex_color(background_rgb)
    thumb_color = _hex_color(thumb)
    active_color = _hex_color(active)
    style = ttk.Style(widget)
    for name in (VERTICAL_STYLE, HORIZONTAL_STYLE):
        style.configure(
            name,
            troughcolor=trough,
            background=thumb_color,
            bordercolor=trough,
            lightcolor=thumb_color,
            darkcolor=thumb_color,
            arrowcolor=_hex_color(foreground_rgb),
            # The thumb hatch is -gripcount on the Windows Tk build and
            # -gripsize (default 7.5 points) on the macOS build. A value
            # of 0 skips the strokes on that build.
            gripcount=0,
            gripsize=0,
            borderwidth=0,
            relief='flat',
            width=12,
        )
        style.map(name, background=[
            ('pressed', active_color),
            ('active', active_color),
        ])


def _rgb(widget, color):
    red, green, blue = widget.winfo_rgb(color)
    return red / 65535, green / 65535, blue / 65535


def _mix(start, end, amount):
    return tuple(start[i] + (end[i] - start[i]) * amount for i in range(3))


def _luma(color):
    red, green, blue = color
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _hex_color(color):
    return '#{:02x}{:02x}{:02x}'.format(*(
        round(min(1, max(0, channel)) * 255) for channel in color
    ))
