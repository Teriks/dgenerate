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

"""Editor colors for dialogs.

The platform ttk theme (vista, aqua, default) ignores color options, so
dialogs stay on the system appearance. Clam draws its widgets from style
colors, on macOS as well. The macOS window frame and the system menu bar stay
native; this only recolors the widgets inside the window.
"""

import platform
import tkinter as tk
import tkinter.ttk as ttk

import dgenerate.console.appearance as _appearance

_CUSTOM = platform.system() in ('Windows', 'Linux', 'Darwin')

# Classic colors that mean "the platform default", plus the light card fill
# image-seed dialogs paint by hand. Anything else (black viewers, the sash,
# a validation tint) is left alone. Colors this module has applied are tracked
# separately so a later theme can replace them.
_PLAIN_BACKGROUND = {
    'white', '#ffffff', '#fff',
    'systembuttonface', 'systemwindow', 'systemwindowframe',
    'systembuttonshadow', 'systemhighlight',
    '#d9d9d9', '#dcdcdc', '#dddddd', '#e0e0e0', '#e8e8e8', '#ececec', '#f0f0f0',
    '#f7f7f7', '#d0d0d0',
    'grey85', 'gray85', 'grey86', 'gray86',
}
_PLAIN_FOREGROUND = {
    'black', '#000000', '#000',
    'systembuttontext', 'systemwindowtext',
}

_CLASSIC = {
    'Tk', 'Toplevel', 'Frame', 'Label', 'Canvas', 'Listbox', 'Entry', 'Text',
    'Button', 'Checkbutton', 'Radiobutton', 'Spinbox', 'Menubutton',
}
_FOREGROUND_CLASS = {
    'Label', 'Entry', 'Text', 'Button', 'Checkbutton', 'Radiobutton',
    'Spinbox', 'Listbox', 'Menubutton',
}
_FIELD_CLASS = {'Entry', 'Text', 'Spinbox'}

_colors = None
_owned_background = set()
_owned_foreground = set()


def background():
    """Editor background last applied, or ``None`` when dialogs are not themed."""
    if _colors is None:
        return None
    return _colors['background']


def foreground():
    """Editor foreground last applied, or ``None`` when dialogs are not themed."""
    if _colors is None:
        return None
    return _colors['foreground']


def edge():
    """Border mixed from the editor colors, or ``None`` when dialogs are not themed."""
    if _colors is None:
        return None
    return _colors['edge']


def apply(widget, background, foreground, selectbackground, selectforeground,
          window_background=None, window_chrome=None) -> None:
    """Recolor every dialog from the editor theme.

    ``background`` is the text-field color. ``window_background`` fills the
    window and dialogs, and ``window_chrome`` draws their border. Both default
    to the field color and a mix of the field and text, so a theme can keep a
    white editor without a white window.
    """
    global _colors
    if widget is None or not _CUSTOM:
        return
    field = _hex(_rgb(widget, background))
    fg = _hex(_rgb(widget, foreground))
    sel = _hex(_rgb(widget, selectbackground))
    selfg = _hex(_rgb(widget, selectforeground))
    bg = _hex(_rgb(widget, window_background or field))
    bg_rgb = _rgb(widget, bg)
    fg_rgb = _rgb(widget, fg)
    field_rgb = _rgb(widget, field)
    edge_color = (_hex(_rgb(widget, window_chrome)) if window_chrome
                  else _hex(_mix(bg_rgb, fg_rgb, 0.35)))
    button = _hex(_mix(bg_rgb, fg_rgb, 0.14))
    active = _hex(_mix(bg_rgb, fg_rgb, 0.26))
    pressed = _hex(_mix(bg_rgb, fg_rgb, 0.40))
    disabled = _hex(_mix(fg_rgb, bg_rgb, 0.45))
    # A step darker than the field. The platform disabled color is a light
    # button face, which sits as a gray slab on a dark dialog.
    disabled_field = _hex(_mix(field_rgb, (0, 0, 0), 0.22))
    palette = {
        'background': bg,
        'field': field,
        'foreground': fg,
        'selectbackground': sel,
        'selectforeground': selfg,
        'edge': edge_color,
        'button': button,
        'active': active,
        'pressed': pressed,
        'disabled': disabled,
        'disabled_field': disabled_field,
    }
    _styles(widget, palette)
    _options(widget, palette)
    _walk(_application(widget), palette)
    _colors = palette
    _owned_background.clear()
    _owned_background.update((bg, field, edge_color, button, active, pressed))
    _owned_foreground.clear()
    _owned_foreground.update((fg, disabled, selfg))


def _styles(widget, palette):
    style = ttk.Style(widget)
    if style.theme_use() != 'clam':
        style.theme_use('clam')
    bg = palette['background']
    field = palette['field']
    fg = palette['foreground']
    sel = palette['selectbackground']
    selfg = palette['selectforeground']
    edge_color = palette['edge']
    button = palette['button']
    active = palette['active']
    pressed = palette['pressed']
    disabled = palette['disabled']
    disabled_field = palette['disabled_field']
    style.configure(
        '.',
        background=bg,
        foreground=fg,
        fieldbackground=field,
        selectbackground=sel,
        selectforeground=selfg,
        insertcolor=fg,
        troughcolor=bg,
        bordercolor=edge_color,
        lightcolor=edge_color,
        darkcolor=edge_color,
        arrowcolor=fg,
    )
    style.map(
        '.',
        background=[('disabled', bg), ('active', active)],
        foreground=[('disabled', disabled)],
        selectbackground=[('!focus', edge_color)],
        selectforeground=[('!focus', fg)],
    )
    style.configure(
        'TButton',
        width=0,
        anchor='center',
        padding=(4, 2),
        background=button,
        foreground=fg,
        bordercolor=edge_color,
        lightcolor=button,
        darkcolor=button,
    )
    style.map(
        'TButton',
        background=[('disabled', bg), ('pressed', pressed), ('active', active)],
        foreground=[('disabled', disabled)],
        bordercolor=[('focus', sel)],
    )
    for name in ('Toolbutton', 'TMenubutton'):
        style.configure(
            name,
            background=button,
            foreground=fg,
            bordercolor=edge_color,
        )
        style.map(
            name,
            background=[('disabled', bg), ('pressed', pressed), ('active', active)],
            foreground=[('disabled', disabled)],
        )
    style.configure(
        'DgenerateSpin.TButton',
        padding=0,
        background=button,
        foreground=fg,
        bordercolor=edge_color,
        lightcolor=button,
        darkcolor=button,
    )
    for name in ('TCheckbutton', 'TRadiobutton'):
        style.configure(
            name,
            background=bg,
            foreground=fg,
            indicatorbackground=bg,
            indicatorforeground=fg,
        )
        style.map(
            name,
            background=[('active', active), ('disabled', bg)],
            foreground=[('disabled', disabled)],
            indicatorbackground=[
                ('pressed', active),
                (('!disabled', 'alternate'), sel),
                (('disabled', 'alternate'), edge_color),
                ('disabled', bg),
            ],
        )
    for name in ('TEntry', 'TSpinbox', 'TCombobox'):
        style.configure(
            name,
            fieldbackground=field,
            foreground=fg,
            insertcolor=fg,
            bordercolor=edge_color,
            arrowcolor=fg,
        )
        style.map(
            name,
            fieldbackground=[('readonly', field), ('disabled', disabled_field)],
            foreground=[('disabled', disabled), ('readonly', fg)],
            bordercolor=[('focus', sel)],
            arrowcolor=[('disabled', disabled)],
            background=[('active', active), ('pressed', pressed)],
        )
    style.configure('ComboboxPopdownFrame', background=bg, bordercolor=edge_color)
    style.configure('TLabelframe', background=bg, foreground=fg, bordercolor=edge_color)
    style.configure('TLabelframe.Label', background=bg, foreground=fg)
    style.configure('TNotebook', background=bg, bordercolor=edge_color)
    style.configure('TNotebook.Tab', background=button, foreground=fg, bordercolor=edge_color)
    style.map(
        'TNotebook.Tab',
        background=[('selected', bg), ('active', active)],
        foreground=[('disabled', disabled)],
    )
    style.configure('Treeview', background=field, fieldbackground=field, foreground=fg, bordercolor=edge_color)
    style.map(
        'Treeview',
        background=[('disabled', bg), ('selected', sel)],
        foreground=[('disabled', disabled), ('selected', selfg)],
    )
    style.configure('Heading', background=button, foreground=fg, bordercolor=edge_color)
    style.configure('TSeparator', background=edge_color)
    style.configure('TProgressbar', background=sel, troughcolor=bg, bordercolor=edge_color)
    style.configure('TScale', background=sel, troughcolor=bg, bordercolor=edge_color)
    style.configure(
        'TScrollbar',
        troughcolor=bg,
        background=button,
        bordercolor=bg,
        lightcolor=button,
        darkcolor=button,
        arrowcolor=fg,
        # A long combobox list uses this style, not the editor scrollbar.
        # The clam thumb hatch is -gripcount on Windows and -gripsize on macOS.
        gripcount=0,
        gripsize=0,
    )
    _appearance.install_checkbutton_fit(widget)
    _appearance.install_button_fitting(widget)


def _options(widget, palette):
    bg = palette['background']
    field = palette['field']
    fg = palette['foreground']
    sel = palette['selectbackground']
    selfg = palette['selectforeground']
    edge_color = palette['edge']
    pairs = (
        ('*Frame.background', bg),
        ('*Label.background', bg),
        ('*Label.foreground', fg),
        ('*Canvas.background', bg),
        ('*Toplevel.background', bg),
        ('*Listbox.background', field),
        ('*Listbox.foreground', fg),
        ('*Listbox.selectBackground', sel),
        ('*Listbox.selectForeground', selfg),
        ('*Entry.background', field),
        ('*Entry.foreground', fg),
        ('*Entry.insertBackground', fg),
        ('*Entry.selectBackground', sel),
        ('*Entry.selectForeground', selfg),
        ('*Entry.highlightBackground', edge_color),
        ('*Entry.highlightColor', sel),
        ('*Entry.disabledBackground', palette['disabled_field']),
        ('*Entry.disabledForeground', palette['disabled']),
        ('*Text.background', field),
        ('*Text.foreground', fg),
        ('*Text.insertBackground', fg),
        ('*Text.selectBackground', sel),
        ('*Text.selectForeground', selfg),
        ('*Text.highlightBackground', edge_color),
        ('*Text.highlightColor', sel),
        ('*Button.background', bg),
        ('*Button.foreground', fg),
        ('*Checkbutton.background', bg),
        ('*Checkbutton.foreground', fg),
        ('*Checkbutton.selectColor', bg),
        ('*Radiobutton.background', bg),
        ('*Radiobutton.foreground', fg),
        ('*Radiobutton.selectColor', bg),
        ('*Spinbox.background', field),
        ('*Spinbox.foreground', fg),
        ('*Spinbox.insertBackground', fg),
        ('*Spinbox.highlightBackground', edge_color),
        ('*Spinbox.highlightColor', sel),
        ('*Spinbox.disabledBackground', palette['disabled_field']),
        ('*Spinbox.disabledForeground', palette['disabled']),
        ('*TCombobox*Listbox.background', field),
        ('*TCombobox*Listbox.foreground', fg),
        ('*TCombobox*Listbox.selectBackground', sel),
        ('*TCombobox*Listbox.selectForeground', selfg),
    )
    if platform.system() == 'Darwin':
        # Aqua's entry is a 2px sunken bevel plus a 3px ring. The bevel's
        # light edge is white, so a dark field reads as a thick white frame.
        # A flat 1px ring in the theme edge matches the Windows line.
        pairs += (
            ('*Entry.borderWidth', 0),
            ('*Entry.relief', 'flat'),
            ('*Entry.highlightThickness', 1),
            ('*Entry.highlightColor', edge_color),
            ('*Spinbox.borderWidth', 0),
            ('*Spinbox.relief', 'flat'),
            ('*Spinbox.highlightThickness', 1),
            ('*Spinbox.highlightColor', edge_color),
        )
    for pattern, value in pairs:
        widget.option_add(pattern, value, 100)


def _walk(widget, palette):
    _paint(widget, palette)
    _recolor_combobox_popdown(widget, palette)
    try:
        children = widget.winfo_children()
    except tk.TclError:
        return
    for child in children:
        _walk(child, palette)


def _recolor_combobox_popdown(widget, palette):
    """Recolor a combobox list that is already open.

    Tk builds that list as a child toplevel. It is absent from
    ``winfo_children``, so the normal walk never reaches its listbox, and
    the list keeps the colors it was created with.
    """
    if not isinstance(widget, ttk.Combobox):
        return
    listbox = f'{widget._w}.popdown.f.l'
    try:
        if not widget.tk.getboolean(widget.tk.call('winfo', 'exists', listbox)):
            return
        widget.tk.call(
            listbox, 'configure',
            '-background', palette['field'],
            '-foreground', palette['foreground'],
            '-selectbackground', palette['selectbackground'],
            '-selectforeground', palette['selectforeground'],
        )
    except tk.TclError:
        pass


def _paint(widget, palette):
    try:
        klass = widget.winfo_class()
    except tk.TclError:
        return
    if klass not in _CLASSIC:
        return
    try:
        current = str(widget.cget('background'))
    except tk.TclError:
        return
    if not _paintable(current, _owned_background, _PLAIN_BACKGROUND):
        return
    fill = palette['field'] if klass in _FIELD_CLASS or klass == 'Listbox' else palette['background']
    fg = palette['foreground']
    options = {'background': fill}
    if klass in _FOREGROUND_CLASS:
        try:
            current_fg = str(widget.cget('foreground'))
        except tk.TclError:
            current_fg = ''
        if _paintable(current_fg, _owned_foreground, _PLAIN_FOREGROUND):
            options['foreground'] = fg
    if klass in _FIELD_CLASS:
        options.update(
            insertbackground=fg,
            selectbackground=palette['selectbackground'],
            selectforeground=palette['selectforeground'],
            highlightbackground=palette['edge'],
            highlightcolor=palette['selectbackground'],
            highlightthickness=1,
        )
        if klass in ('Entry', 'Spinbox'):
            options['disabledbackground'] = palette['disabled_field']
            options['disabledforeground'] = palette['disabled']
            if platform.system() == 'Darwin':
                options.update(
                    relief='flat',
                    borderwidth=0,
                    highlightthickness=1,
                    highlightcolor=palette['edge'],
                )
    elif klass == 'Listbox':
        options.update(
            selectbackground=palette['selectbackground'],
            selectforeground=palette['selectforeground'],
        )
    else:
        try:
            highlight = str(widget.cget('highlightbackground'))
        except tk.TclError:
            highlight = ''
        if highlight and _paintable(highlight, _owned_background, _PLAIN_BACKGROUND):
            options['highlightbackground'] = palette['edge']
            options['highlightcolor'] = palette['edge']
    try:
        widget.configure(**options)
    except tk.TclError:
        pass


def _paintable(color, owned, plain):
    text = str(color).lower()
    return text.startswith('system') or text in plain or text in owned


def _application(widget):
    root = widget.winfo_toplevel()
    while root.master is not None:
        root = root.master
    return root


def _rgb(widget, color):
    red, green, blue = widget.winfo_rgb(color)
    return red / 65535, green / 65535, blue / 65535


def _mix(start, end, amount):
    return tuple(start[i] + (end[i] - start[i]) * amount for i in range(3))


def _hex(color):
    return '#{:02x}{:02x}{:02x}'.format(*(
        round(min(1, max(0, channel)) * 255) for channel in color
    ))
