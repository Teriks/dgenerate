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

"""Menu bar and dropdowns for the console.

Windows and Linux draw the bar and its menus, so the editor colors cover
them. The native menu frame, including the white border Windows draws around
a themed dropdown, is not used. macOS keeps Tk's menu bar so it stays in the
system menu bar. Right-click menus are drawn by the console on every platform.
"""

import platform
import tkinter as tk

# The macOS menu bar is the system menu bar. Its dropdowns stay native with it.
# Right-click menus are drawn here on every platform.
_CUSTOM = platform.system() in ('Windows', 'Linux')

_bars = []
_menus = []
_open = []
_chrome = set()
_return_focus = None
_watching = False
_motion_job = None
_pointer_job = None
_foreground_job = None
_switching = False
_opening_click = False
# Aqua can drop the click that should dismiss a menu after a scroll.
# The pointer poll remembers that the buttons came up, then closes when
# one goes down outside the menu.
_buttons_released = False
_colors = {
    'background': '#f0f0f0',
    'foreground': '#000000',
    'activebackground': '#cce8ff',
    'activeforeground': '#000000',
    'disabledforeground': '#6e6e6e',
    'selectcolor': '#000000',
}


class MenuBar(tk.Frame):
    """The top row of menu titles."""

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._slots = []
        self._commands = {}
        self._border = tk.Frame(self, height=1, borderwidth=0, highlightthickness=0)
        self._border.pack(side=tk.BOTTOM, fill=tk.X)
        self._row = tk.Frame(self, borderwidth=0, highlightthickness=0)
        self._row.pack(side=tk.TOP, fill=tk.X)
        _bars.append(self)
        _chrome.add(self)
        _chrome.add(self._row)
        self._paint()

    def add_cascade(self, label, menu):
        button = tk.Label(self._row, text=label, padx=10, pady=4, anchor='w')
        button.pack(side=tk.LEFT)
        button.bind('<Button-1>', lambda event, item=button, child=menu: self._toggle(item, child))
        button.bind('<Enter>', lambda event, item=button, child=menu: self._hover(item, child))
        button.bind('<Leave>', lambda event: self._clear_hot())
        _chrome.add(button)
        menu._bar = self
        menu._bar_button = button
        self._slots.append((button, menu))
        self._paint()

    def add_command(self, label, command=None):
        button = tk.Label(self._row, text=label, padx=10, pady=4, anchor='w')
        button.pack(side=tk.LEFT)

        def invoke(event, action=command, item=button):
            self._paint()
            _close_all()
            if action is not None:
                action()
            return 'break'

        button.bind('<Button-1>', invoke)
        button.bind('<Enter>', lambda event, item=button: self._hover_idle(item))
        button.bind('<Leave>', lambda event: self._clear_hot())
        _chrome.add(button)
        self._commands[button] = command
        self._slots.append((button, None))
        self._paint()

    def open_relative(self, menu, step):
        menus = [child for button, child in self._slots if child is not None]
        if menu not in menus:
            return
        index = (menus.index(menu) + step) % len(menus)
        button = next(item for item, child in self._slots if child is menus[index])
        self._open(button, menus[index])

    def _toggle(self, button, menu):
        if menu in _open and menu._parent is None:
            _close_all()
            return 'break'
        self._open(button, menu)
        return 'break'

    def _hover(self, button, menu):
        if _open:
            self._open(button, menu)
            return
        self._hot = button
        self._paint()

    def _hover_idle(self, button):
        if _open:
            _close_all()
        self._hot = button
        self._paint()

    def _clear_hot(self):
        self._hot = None
        if not _open:
            self._paint()

    def _open(self, button, menu):
        global _switching
        if _switching or (menu in _open and menu._parent is None):
            return
        # Closing the open popup releases its pointer grab. Doing that in the
        # middle of the hover that entered this button, and destroying the
        # popup on the same stack, aborts Tk.
        _switching = True
        try:
            _close_all(restore_focus=False)
            self._active = button
            self._paint()
            x = button.winfo_rootx()
            y = button.winfo_rooty() + button.winfo_height()
            menu.show(x, y)
        finally:
            _switching = False

    def _paint(self):
        background = _colors['background']
        foreground = _colors['foreground']
        active_background = _colors['activebackground']
        active_foreground = _colors['activeforeground']
        edge = _edge()
        self.configure(background=background)
        self._row.configure(background=background)
        self._border.configure(background=edge)
        active = getattr(self, '_active', None)
        hot = getattr(self, '_hot', None)
        for button, menu in self._slots:
            chosen = (button is active and menu in _open) or (not _open and button is hot)
            button.configure(
                background=active_background if chosen else background,
                foreground=active_foreground if chosen else foreground,
            )

    def destroy(self):
        if self in _bars:
            _bars.remove(self)
        _chrome.discard(self)
        _chrome.discard(self._row)
        for button, _menu in self._slots:
            _chrome.discard(button)
        super().destroy()

    def install(self):
        self.pack(side=tk.TOP, fill=tk.X)


class ThemeMenu:
    """A dropdown or right-click menu.

    A menu hung off the macOS system menu bar is a real Tk menu. Every other
    menu, including right-click menus on macOS, is drawn here.
    """

    def __new__(cls, master=None, cnf=None, **kwargs):
        if not _CUSTOM and isinstance(master, tk.Menu):
            if cnf:
                kwargs = {**cnf, **kwargs}
            kwargs.setdefault('tearoff', 0)
            return tk.Menu(master, **kwargs)
        return super().__new__(cls)

    def __init__(self, master=None, cnf=None, **kwargs):
        self.master = master
        self._entries = []
        self._postcommand = None
        self._popup = None
        self._parent = None
        self._bar = None
        self._bar_button = None
        self._active = None
        self._owner_destroy = None
        if isinstance(master, tk.Misc):
            self._owner_destroy = master.bind('<Destroy>', self._on_owner_destroy, add='+')
            _arm_right_click(master)
        _menus.append(self)

    def _on_owner_destroy(self, event):
        # The recipes form rebuilds rows while a menu is up. The popup is a
        # child of the window, so it survives the entry and the watchers
        # then ask a dead widget where its window is.
        if event.widget is not self.master:
            return
        if self in _open or self._popup is not None:
            _close_all()

    def destroy(self):
        master = self.master
        token = self._owner_destroy
        self._owner_destroy = None
        if token is not None and isinstance(master, tk.Misc):
            try:
                master.unbind('<Destroy>', token)
            except tk.TclError:
                pass
        self._hide()
        if self in _menus:
            _menus.remove(self)

    def add_command(self, label='', command=None, accelerator=None, state=tk.NORMAL):
        self._entries.append(_Entry('command', label, command, accelerator, state))

    def add_checkbutton(self, label='', variable=None, accelerator=None, state=tk.NORMAL):
        self._entries.append(_Entry('check', label, None, accelerator, state, variable=variable))

    def add_radiobutton(self, label='', variable=None, value=None, accelerator=None, state=tk.NORMAL):
        self._entries.append(_Entry(
            'radio', label, None, accelerator, state, variable=variable, value=value))

    def add_cascade(self, label='', menu=None, state=tk.NORMAL):
        entry = _Entry('cascade', label, None, None, state, menu=menu)
        self._entries.append(entry)
        if menu is not None:
            menu._owner = self

    def add_separator(self):
        self._entries.append(_Entry('separator', '', None, None, tk.NORMAL))

    def index(self, what):
        if what == 'end':
            return len(self._entries) - 1 if self._entries else None
        raise tk.TclError(f'bad menu index "{what}"')

    def entryconfigure(self, index, **kwargs):
        entry = self._entry(index)
        if 'label' in kwargs and entry.kind != 'separator':
            entry.label = kwargs['label']
            if entry.text is not None:
                entry.text.configure(text=entry.label)
        if 'state' in kwargs:
            entry.state = kwargs['state']
            if entry.row is not None:
                self._paint_entry(entry, entry is self._active)
        if 'command' in kwargs:
            entry.command = kwargs['command']

    entryconfig = entryconfigure

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs = {**cnf, **kwargs}
        if 'postcommand' in kwargs:
            self._postcommand = kwargs['postcommand']

    config = configure

    def cget(self, key):
        if key == 'postcommand':
            return self._postcommand
        try:
            return _colors[key]
        except KeyError:
            raise tk.TclError(f'unknown option "-{key}"') from None

    def tk_popup(self, x, y, entry=''):
        self.show(x, y)

    def grab_release(self):
        """Present for the native-menu calling pattern. The popup keeps its grab.

        A native menu grabs inside tk_popup, and callers release that grab
        in a finally block. Doing the same to the popup would drop the
        pointer, so a click outside would never reach the menu.
        """
        return

    def show(self, x, y, parent=None):
        global _return_focus, _opening_click, _buttons_released
        # The click that opened this menu is still being dispatched. It must
        # not also count as a click outside the menu.
        _opening_click = True
        _buttons_released = False
        if not _open and _return_focus is None:
            try:
                _return_focus = self._toplevel().focus_get()
            except tk.TclError:
                _return_focus = None
        if self._postcommand is not None:
            self._postcommand()
        keep = []
        owner = parent
        while owner is not None:
            keep.append(owner)
            owner = owner._parent
        for menu in list(_open):
            if menu not in keep:
                menu._hide()
        self._parent = parent
        _watch_parent(self._toplevel())
        self._build(x, y)
        if self not in _open:
            _open.append(self)
        _watch(self._popup)
        _arm_foreground_watch()
        _arm_pointer_watch()
        top = self._toplevel()
        if top is not None:
            try:
                top.after_idle(_end_opening_click)
            except tk.TclError:
                _opening_click = False
        else:
            _opening_click = False
        if self._bar is not None and parent is None:
            self._bar._active = self._bar_button
            self._bar._paint()

    def _entry(self, index):
        if isinstance(index, str) and index != 'end':
            for entry in self._entries:
                if entry.label == index:
                    return entry
            raise tk.TclError(f'bad menu entry index "{index}"')
        if index == 'end':
            index = self.index('end')
        try:
            return self._entries[index]
        except (TypeError, IndexError):
            raise tk.TclError(f'bad menu entry index "{index}"') from None

    def _build(self, x, y):
        self._hide_popup()
        master = self._toplevel()
        popup = tk.Toplevel(master)
        popup.overrideredirect(True)
        _floating(popup)
        popup.configure(background=_colors['background'], highlightthickness=1,
                        highlightbackground=_edge(), highlightcolor=_edge(), bd=0)
        popup.withdraw()
        self._popup = popup
        self._active = None
        _chrome.add(popup)
        for entry in self._entries:
            if entry.kind == 'separator':
                line = tk.Frame(popup, height=1, background=_edge(),
                                borderwidth=0, highlightthickness=0)
                line.pack(fill=tk.X, padx=8, pady=4)
                entry.row = line
                continue
            row = tk.Frame(popup, background=_colors['background'],
                           borderwidth=0, highlightthickness=0)
            row.pack(fill=tk.X)
            mark = tk.Label(row, text=self._mark(entry), width=2, anchor='center',
                            padx=2, pady=3)
            mark.grid(row=0, column=0, sticky='nsw')
            text = tk.Label(row, text=entry.label, anchor='w', padx=4, pady=3)
            text.grid(row=0, column=1, sticky='nsew')
            row.grid_columnconfigure(1, weight=1)
            extra = tk.Label(
                row,
                text=entry.accelerator or ('>' if entry.kind == 'cascade' else ''),
                anchor='e',
                padx=10,
                pady=3,
            )
            extra.grid(row=0, column=2, sticky='nse')
            entry.row = row
            entry.text = text
            entry.mark = mark
            entry.extra = extra
            for widget in (row, mark, text, extra):
                widget.bind('<Enter>', lambda event, item=entry: self._hover(item))
                widget.bind('<Motion>', lambda event, item=entry: self._hover(item))
                widget.bind('<Button-1>', lambda event, item=entry: self._activate(item))
                _bind_scroll(widget)
            self._paint_entry(entry, False)
        popup.update_idletasks()
        width = max(popup.winfo_reqwidth(), 160)
        height = popup.winfo_reqheight()
        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        if x + width > screen_w:
            x = max(0, screen_w - width)
        if y + height > screen_h:
            y = max(0, screen_h - height)
        popup.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
        # event.x_root matches this point. Aqua's winfo_rootx for an
        # override-redirect window often does not.
        popup._menu_origin = (int(x), int(y))
        if master is not None:
            try:
                # On X11 a transient override-redirect window is withdrawn
                # by the window manager as soon as it is mapped.
                if popup.tk.call('tk', 'windowingsystem') != 'x11':
                    popup.transient(_popup_owner(popup, master))
            except tk.TclError:
                pass
        popup.bind('<Deactivate>', _on_deactivate)
        popup.deiconify()
        _own_popup(popup, master)
        try:
            popup.focus_set()
        except tk.TclError:
            pass
        _take_pointer(popup)
        for sequence in ('<Button-1>', '<Button-2>', '<Button-3>'):
            popup.bind(sequence, _on_popup_click)
        popup.bind('<Escape>', lambda event: (_close_all(), 'break')[1])
        popup.bind('<Up>', lambda event: self._move(-1))
        popup.bind('<Down>', lambda event: self._move(1))
        popup.bind('<Return>', lambda event: self._activate(self._active) if self._active else None)
        popup.bind('<Left>', lambda event: self._left())
        popup.bind('<Right>', lambda event: self._right())
        _bind_scroll(popup)

    def _hover(self, entry):
        if self._popup is None or entry.kind == 'separator':
            return
        if self._active is entry:
            return
        previous = self._active
        self._active = entry
        if previous is not None and previous is not entry and previous.row is not None:
            self._paint_entry(previous, False)
        self._paint_entry(entry, True)
        if entry.kind == 'cascade' and entry.menu is not None and not _disabled(entry):
            self._open_child(entry)
        else:
            self._close_children()

    def _open_child(self, entry):
        if self._popup is None:
            return
        if entry.menu in _open and entry.menu._parent is self:
            return
        self._close_children()
        if self._popup is None:
            return
        left, top = _popup_origin(self._popup)
        x = left + self._popup.winfo_width() - 1
        y = top + entry.row.winfo_y()
        entry.menu.show(x, y, parent=self)

    def _close_children(self):
        for menu in list(_open):
            owner = menu._parent
            while owner is not None:
                if owner is self:
                    menu._hide()
                    break
                owner = owner._parent

    def _activate(self, entry):
        if entry is None or entry.kind == 'separator' or _disabled(entry):
            return 'break'
        if entry.kind == 'cascade':
            self._open_child(entry)
            return 'break'
        if entry.kind == 'check' and entry.variable is not None:
            entry.variable.set(not bool(entry.variable.get()))
        elif entry.kind == 'radio' and entry.variable is not None:
            entry.variable.set(entry.value)
        action = entry.command
        _close_all()
        if action is not None:
            action()
        return 'break'

    def _move(self, step):
        choices = [entry for entry in self._entries if entry.kind != 'separator']
        if not choices:
            return 'break'
        if self._active not in choices:
            index = 0 if step > 0 else len(choices) - 1
        else:
            index = (choices.index(self._active) + step) % len(choices)
        self._hover(choices[index])
        return 'break'

    def _left(self):
        if self._parent is not None:
            self._hide()
            if self._parent._popup is not None:
                self._parent._popup.focus_set()
        elif self._bar is not None:
            self._bar.open_relative(self, -1)
        return 'break'

    def _right(self):
        if self._active is not None and self._active.kind == 'cascade':
            self._open_child(self._active)
            if self._active.menu._popup is not None:
                self._active.menu._move(1)
        elif self._bar is not None and self._parent is None:
            self._bar.open_relative(self, 1)
        return 'break'

    def _paint_entry(self, entry, active):
        if entry.row is None or entry.kind == 'separator':
            return
        if _disabled(entry):
            foreground = _colors['disabledforeground']
        elif active:
            foreground = _colors['activeforeground']
        else:
            foreground = _colors['foreground']
        background = _colors['activebackground'] if active and not _disabled(entry) else _colors['background']
        entry.mark.configure(text=self._mark(entry))
        entry.row.configure(background=background)
        for widget in (entry.mark, entry.text, entry.extra):
            widget.configure(background=background, foreground=foreground)

    def _mark(self, entry):
        if entry.kind == 'check' and entry.variable is not None and bool(entry.variable.get()):
            return '✓'
        if entry.kind == 'radio' and entry.variable is not None and entry.variable.get() == entry.value:
            return '●'
        return ''

    def _toplevel(self):
        master = self.master
        while master is not None and not isinstance(master, tk.Misc):
            master = getattr(master, 'master', None)
        if master is None:
            return None
        try:
            if not master.winfo_exists():
                return None
            return master.winfo_toplevel()
        except tk.TclError:
            return None

    def _hide_popup(self):
        popup = self._popup
        self._popup = None
        self._active = None
        if popup is not None:
            _chrome.discard(popup)
            _release_popup(popup)
        for entry in self._entries:
            entry.row = None
            entry.text = None
            entry.mark = None
            entry.extra = None

    def _hide(self):
        parent = self._parent
        self._hide_popup()
        if self in _open:
            _open.remove(self)
        self._parent = None
        if self._bar is not None and getattr(self._bar, '_active', None) is self._bar_button:
            self._bar._active = None
            try:
                self._bar._paint()
            except tk.TclError:
                pass
        if parent is not None and parent in _open and parent._popup is not None:
            try:
                parent._popup.grab_set()
                parent._popup.focus_set()
            except tk.TclError:
                pass


class _Entry:
    def __init__(self, kind, label, command, accelerator, state,
                 variable=None, value=None, menu=None):
        self.kind = kind
        self.label = label
        self.command = command
        self.accelerator = accelerator
        self.state = state
        self.variable = variable
        self.value = value
        self.menu = menu
        self.row = None
        self.text = None
        self.mark = None
        self.extra = None

    def invoke(self):
        if self.kind == 'separator' or _disabled(self):
            return
        if self.kind == 'check' and self.variable is not None:
            self.variable.set(not bool(self.variable.get()))
        elif self.kind == 'radio' and self.variable is not None:
            self.variable.set(self.value)
        if self.command is not None:
            self.command()


if not _CUSTOM:
    class MenuBar(tk.Menu):
        """Tk menu bar, attached to the window's system menu on macOS."""

        def __init__(self, master=None, **kwargs):
            super().__init__(master, tearoff=0)
            self._owner = master

        def install(self):
            self._owner.config(menu=self)


def apply(widget, background, foreground, activebackground, activeforeground) -> None:
    """Recolor drawn menus from the editor colors."""
    if widget is None:
        return
    background_rgb = _rgb(widget, background)
    foreground_rgb = _rgb(widget, foreground)
    _colors.update(
        background=_hex_color(background_rgb),
        foreground=_hex_color(foreground_rgb),
        activebackground=_hex_color(_rgb(widget, activebackground)),
        activeforeground=_hex_color(_rgb(widget, activeforeground)),
        disabledforeground=_hex_color(_mix(background_rgb, foreground_rgb, 0.45)),
        selectcolor=_hex_color(foreground_rgb),
    )
    for bar in list(_bars):
        try:
            bar._paint()
        except tk.TclError:
            pass


def _edge():
    return _hex_color(_mix(
        _parse(_colors['background']),
        _parse(_colors['foreground']),
        0.35,
    ))


def _disabled(entry):
    return str(entry.state).lower() in ('disabled', '0')


def _release_popup(popup):
    """Drop the pointer grab and destroy the popup after this event."""
    try:
        if popup.grab_current() is popup:
            popup.grab_release()
    except tk.TclError:
        pass
    try:
        popup.withdraw()
    except tk.TclError:
        pass
    try:
        owner = popup.master.winfo_toplevel() if popup.master is not None else popup
        owner.after(1, lambda popup=popup: _destroy_popup(popup))
    except tk.TclError:
        _destroy_popup(popup)


def _destroy_popup(popup):
    try:
        popup.destroy()
    except tk.TclError:
        pass


def _cancel_owner_job(job):
    if job is None or not _open:
        return
    try:
        top = _open[0]._toplevel()
    except tk.TclError:
        return
    if top is None:
        return
    try:
        top.after_cancel(job)
    except tk.TclError:
        pass


def _close_all(restore_focus=True):
    global _return_focus, _motion_job, _pointer_job, _foreground_job, _buttons_released
    _buttons_released = False
    _cancel_owner_job(_motion_job)
    _motion_job = None
    _cancel_owner_job(_pointer_job)
    _pointer_job = None
    _cancel_owner_job(_foreground_job)
    _foreground_job = None
    for menu in list(_open):
        menu._hide()
    for bar in _bars:
        bar._active = None
        try:
            bar._paint()
        except tk.TclError:
            pass
    focus = _return_focus
    if restore_focus:
        _return_focus = None
        if focus is not None:
            try:
                focus.focus_set()
            except tk.TclError:
                pass


def _watch_parent(top):
    if top is None or getattr(top, '_menu_parent_watch', False):
        return
    top._menu_parent_watch = True
    top.bind('<Deactivate>', _on_deactivate, add='+')
    top.bind('<Unmap>', _on_parent_unmap, add='+')


def _on_deactivate(event):
    if _switching or not _open:
        return
    try:
        event.widget.after_idle(_close_if_background)
    except tk.TclError:
        _close_if_background()


def _on_parent_unmap(event):
    if _switching or not _open:
        return
    widget = event.widget
    try:
        if widget is not widget.winfo_toplevel():
            return
    except tk.TclError:
        return
    for menu in _open:
        if menu._popup is widget:
            return
    _close_all()


def _close_if_background():
    # Opening a menu moves focus off the owner, and that deactivates it.
    # The click has not finished, so this is not the user leaving.
    if _switching or not _open or _opening_click:
        return
    for menu in list(_open):
        popup = menu._popup
        try:
            popup_alive = popup is not None and bool(popup.winfo_exists())
        except tk.TclError:
            popup_alive = False
        top = menu._toplevel()
        if not popup_alive or top is None:
            _close_all()
            return
        if _foreground(popup) or _foreground(top):
            return
    _close_all()


def _floating(popup):
    """Keep an override-redirect menu from taking the key window on Aqua.

    Tk drops mouse-moved events for every window except the key window,
    and a help window is not allowed to become that. Hover is read from
    the pointer position instead.
    """
    try:
        if popup.tk.call('tk', 'windowingsystem') != 'aqua':
            return
        popup.tk.call(
            '::tk::unsupported::MacWindowStyle', 'style', popup._w,
            'help', 'noActivates')
    except tk.TclError:
        pass


def _take_pointer(popup):
    """Hold the pointer so a click outside the menu is delivered here.

    A local grab does that on Windows and X11. On Aqua the click goes to
    the widget underneath, and that widget's binding can swallow it before
    the menu sees it. A global grab is what reaches those clicks.
    """
    try:
        aqua = popup.tk.call('tk', 'windowingsystem') == 'aqua'
    except tk.TclError:
        aqua = False
    try:
        if aqua:
            popup.grab_set_global()
        else:
            popup.grab_set()
    except tk.TclError:
        try:
            popup.grab_set()
        except tk.TclError:
            pass


def _popup_owner(popup, master):
    """The window a menu popup is made transient to.

    On Aqua, showing a transient window makes its owner the key window. A
    menu opened over the console while a dialog has focus would activate
    the console, so the popup belongs to the window that had focus.
    """
    try:
        if popup.tk.call('tk', 'windowingsystem') != 'aqua' or _return_focus is None:
            return master
        top = _return_focus.winfo_toplevel()
        if not top.winfo_exists() or int(top.tk.call('wm', 'overrideredirect', top._w)):
            return master
        return top
    except (tk.TclError, ValueError):
        return master


def _own_popup(popup, master):
    """Keep the menu above its window without floating over other applications.

    Tk's transient flag does not set the owner of an overrideredirect window
    on Windows, so the popup otherwise stays on top after its window is in
    the background.
    """
    if platform.system() != 'Windows' or master is None:
        return
    import ctypes
    user32 = ctypes.windll.user32
    user32.GetAncestor.restype = ctypes.c_void_p
    user32.GetAncestor.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.SetWindowLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
    user32.SetWindowPos.restype = ctypes.c_int
    user32.SetWindowPos.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint)
    try:
        popup_hwnd = user32.GetAncestor(ctypes.c_void_p(popup.winfo_id()), 2) or popup.winfo_id()
        owner_hwnd = user32.GetAncestor(ctypes.c_void_p(master.winfo_id()), 2) or master.winfo_id()
    except tk.TclError:
        return
    user32.SetWindowLongPtrW(ctypes.c_void_p(popup_hwnd), -8, ctypes.c_void_p(owner_hwnd))
    user32.SetWindowPos(ctypes.c_void_p(popup_hwnd), 0, 0, 0, 0, 0, 0x0013)


def _arm_foreground_watch():
    global _foreground_job
    if _foreground_job is not None or not _open:
        return
    try:
        top = _open[0]._toplevel()
    except tk.TclError:
        top = None
    if top is None:
        _close_all()
        return
    try:
        _foreground_job = top.after(100, _watch_foreground)
    except tk.TclError:
        _foreground_job = None
        _close_all()


def _watch_foreground():
    global _foreground_job
    _foreground_job = None
    if _switching or not _open:
        return
    _close_if_background()
    _arm_foreground_watch()


def _foreground(widget):
    """True when this window is the one the user is working in."""
    if widget is None:
        return False
    if platform.system() == 'Windows':
        return _windows_foreground(widget)
    if platform.system() == 'Linux':
        return _linux_foreground(widget)
    if platform.system() == 'Darwin':
        return _aqua_foreground(widget)
    return _focus_foreground(widget)


def _aqua_foreground(widget):
    """True while this app is active.

    A right click does not make its window key, so with a dialog open the
    focus can stay in the dialog while a menu is shown over the console.
    Clicks elsewhere in the app reach the menu's global grab. Focus is
    empty once another application is active.
    """
    try:
        return widget.focus_get() is not None
    except (tk.TclError, KeyError):
        return False


def _focus_foreground(widget):
    try:
        focus = widget.focus_get()
    except tk.TclError:
        return False
    if focus is None:
        return False
    try:
        return focus.winfo_toplevel() is widget.winfo_toplevel()
    except tk.TclError:
        return False


_linux_lib = None
_linux_display = None


def _linux_x11():
    """A private X11 connection, or nothing when X11 is not available.

    The connection is separate from Tk's. It is only used to read which
    window the window manager considers active.
    """
    global _linux_lib, _linux_display
    if _linux_lib is False:
        return None, None
    if _linux_lib is not None:
        return _linux_lib, _linux_display
    import ctypes
    try:
        lib = ctypes.CDLL('libX11.so.6')
    except OSError:
        _linux_lib = False
        return None, None
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XOpenDisplay.argtypes = (ctypes.c_char_p,)
    lib.XInternAtom.restype = ctypes.c_ulong
    lib.XInternAtom.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int)
    lib.XDefaultRootWindow.restype = ctypes.c_ulong
    lib.XDefaultRootWindow.argtypes = (ctypes.c_void_p,)
    lib.XGetWindowProperty.restype = ctypes.c_int
    lib.XGetWindowProperty.argtypes = (
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long, ctypes.c_long,
        ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)))
    lib.XFree.argtypes = (ctypes.c_void_p,)
    lib.XQueryTree.restype = ctypes.c_int
    lib.XQueryTree.argtypes = (
        ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
        ctypes.POINTER(ctypes.c_uint))
    display = lib.XOpenDisplay(None)
    if not display:
        _linux_lib = False
        return None, None
    _linux_lib = lib
    _linux_display = display
    return lib, display


def _net_active_window():
    """The window manager's active window, 0 when nothing is, or nothing when unknown."""
    import ctypes
    lib, display = _linux_x11()
    if lib is None:
        return None
    atom = lib.XInternAtom(display, b'_NET_ACTIVE_WINDOW', False)
    actual = ctypes.c_ulong()
    actual_format = ctypes.c_int()
    count = ctypes.c_ulong()
    remaining = ctypes.c_ulong()
    data = ctypes.POINTER(ctypes.c_ulong)()
    status = lib.XGetWindowProperty(
        display, lib.XDefaultRootWindow(display), atom,
        0, 1, False, 0,
        ctypes.byref(actual), ctypes.byref(actual_format),
        ctypes.byref(count), ctypes.byref(remaining), ctypes.byref(data))
    if status != 0 or not count.value or not data:
        if data:
            lib.XFree(data)
        return None if status != 0 else 0
    window = int(data[0])
    lib.XFree(data)
    return window


def _window_ids(widget):
    """This window and the frame windows the window manager may report instead."""
    import ctypes
    found = []
    try:
        found.append(int(widget.winfo_id()))
    except (tk.TclError, TypeError, ValueError):
        return found
    lib, display = _linux_x11()
    if lib is None:
        return found
    root = lib.XDefaultRootWindow(display)
    window = found[0]
    for _ in range(8):
        root_return = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        if not lib.XQueryTree(
                display, window, ctypes.byref(root_return), ctypes.byref(parent),
                ctypes.byref(children), ctypes.byref(count)):
            break
        if children:
            lib.XFree(children)
        parent_id = int(parent.value)
        if parent_id in (0, root, window):
            break
        found.append(parent_id)
        window = parent_id
    return found


def _linux_is_root(window):
    lib, display = _linux_x11()
    if lib is None or not window:
        return False
    return int(window) == int(lib.XDefaultRootWindow(display))


def _linux_foreground(widget):
    """Whether this application is still the one in front.

    The menu is an override-redirect window. It often never takes focus, so
    an empty Tk focus only means the pointer is in the menu. The window
    manager's active window is what changes when the user leaves. Xming's
    fullscreen screen reports no active client while that menu is up, which
    is the same as not knowing, and the menu stays.
    """
    active = _net_active_window()
    if active is None or not active or _linux_is_root(active):
        if _focus_foreground(widget):
            return True
        try:
            return widget.focus_get() is None
        except tk.TclError:
            return False
    if active in _window_ids(widget):
        return True
    return _focus_foreground(widget)


def _windows_foreground(widget):
    import ctypes
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetAncestor.restype = ctypes.c_void_p
    user32.GetAncestor.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    try:
        hwnd = widget.winfo_id()
    except tk.TclError:
        return False
    foreground = user32.GetForegroundWindow()
    if not foreground:
        return False
    root = user32.GetAncestor(ctypes.c_void_p(hwnd), 2) or hwnd
    return int(foreground) in (int(root), int(hwnd))


def _watch(popup):
    global _watching
    if _watching:
        return
    for sequence in ('<Button-1>', '<Button-2>', '<Button-3>'):
        popup.bind_all(sequence, _on_click, add='+')
    popup.bind_all('<Motion>', _on_motion, add='+')
    popup.bind_all('<Escape>', _on_escape, add='+')
    _watching = True


def _rect(widget):
    """Screen box of a widget as x, y, width, height.

    A menu is placed at a screen point, and that point is what the pointer
    reports. Aqua's winfo_rootx for an override-redirect window often sits
    somewhere else, so a menu row is measured from the placed origin plus
    its position inside its parent.
    """
    try:
        if not widget.winfo_exists():
            return None
        width = widget.winfo_width()
        height = widget.winfo_height()
    except tk.TclError:
        return None
    origin_of = widget if getattr(widget, '_menu_origin', None) is not None else None
    if origin_of is None:
        parent = getattr(widget, 'master', None)
        while parent is not None and getattr(parent, '_menu_origin', None) is None:
            parent = getattr(parent, 'master', None)
        if parent is not None:
            left, top = parent._menu_origin
            chain = []
            current = widget
            while current is not parent:
                chain.append(current)
                current = getattr(current, 'master', None)
                if current is None:
                    chain = None
                    break
            if chain:
                for item in reversed(chain):
                    try:
                        left += item.winfo_x()
                        top += item.winfo_y()
                    except tk.TclError:
                        return None
                return left, top, width, height
    else:
        return origin_of._menu_origin[0], origin_of._menu_origin[1], width, height
    try:
        return widget.winfo_rootx(), widget.winfo_rooty(), width, height
    except tk.TclError:
        return None


def _inside(widget, x, y):
    rect = _rect(widget)
    if rect is None:
        return False
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


_right_was_down = False
_right_job = None


def _context_menu_at(x, y):
    """The context menu under a screen point, or none.

    Bar menus are skipped. The smallest widget wins, so a menu on a field
    inside a larger form is the one that opens.
    """
    best = None
    best_area = None
    for menu in _menus:
        if getattr(menu, '_bar', None) is not None:
            continue
        master = getattr(menu, 'master', None)
        if not isinstance(master, tk.Misc) or not _inside(master, x, y):
            continue
        try:
            if not int(master.winfo_viewable()):
                continue
            area = master.winfo_width() * master.winfo_height()
        except tk.TclError:
            continue
        if best is None or area < best_area:
            best = menu
            best_area = area
    return best


def _deliver_right_click():
    """Open the context menu under the pointer, if a right-click binding did not.

    uv's Tk does not turn a right-click on the editor into ``<Button-3>``
    while a dialog is the active window. python.org's Tk does, and that
    binding opens the menu on its own.
    """
    if _open:
        return
    for menu in _menus:
        master = getattr(menu, 'master', None)
        if not isinstance(master, tk.Misc):
            continue
        try:
            x, y = master.winfo_pointerxy()
        except tk.TclError:
            continue
        target = _context_menu_at(x, y)
        if target is not None:
            target.tk_popup(x, y)
        return


def _right_button_down():
    """Whether the right mouse button is down. Reads it without an AppKit callback."""
    if getattr(_aqua_buttons_down, 'lib', None) is None:
        _aqua_buttons_down()
    lib = getattr(_aqua_buttons_down, 'lib', None)
    if not lib:
        return False
    # kCGMouseButtonRight.
    return bool(lib.CGEventSourceButtonState(0, 1))


def _poll_right_click():
    global _right_was_down, _right_job
    _right_job = None
    down = _right_button_down()
    if down and not _right_was_down:
        _deliver_right_click()
    _right_was_down = down
    _arm_right_click(None)


def _arm_right_click(widget):
    """Poll the right button where uv's Tk swallows ``<Button-3>``.

    The poll runs on Tk's timer, so it keeps the interpreter lock. An AppKit
    event monitor calls back into Python while that lock is released and
    aborts the process.
    """
    global _right_job
    if _right_job is not None or platform.system() != 'Darwin':
        return
    import sys
    if 'Python.framework' in getattr(sys, 'base_prefix', ''):
        return
    if widget is not None:
        try:
            if widget.tk.call('tk', 'windowingsystem') != 'aqua':
                return
        except tk.TclError:
            return
    anchor = widget
    if anchor is None:
        for menu in _menus:
            master = getattr(menu, 'master', None)
            if isinstance(master, tk.Misc):
                anchor = master
                break
    if anchor is None:
        return
    try:
        _right_job = anchor.after(16, _poll_right_click)
    except tk.TclError:
        _right_job = None


def _popup_origin(popup):
    """Screen top-left of a menu in the same space as the pointer.

    On Aqua the placed position is that space. winfo_rootx for an
    override-redirect window often is not, which parks a submenu away
    from the row that opened it.
    """
    placed = getattr(popup, '_menu_origin', None)
    try:
        aqua = popup.tk.call('tk', 'windowingsystem') == 'aqua'
    except tk.TclError:
        aqua = False
    if aqua and placed is not None:
        return placed
    try:
        return popup.winfo_rootx(), popup.winfo_rooty()
    except tk.TclError:
        return placed if placed is not None else (0, 0)


def _local_point(popup, x, y):
    """Where the pointer sits inside a menu, or None when it is outside.

    The pointer is compared with the window's reported origin and with
    the point it was placed at. Whichever of those boxes contains the
    pointer is the one the rows are measured in.
    """
    try:
        if not popup.winfo_exists():
            return None
        width = popup.winfo_width()
        height = popup.winfo_height()
    except tk.TclError:
        return None
    origins = []
    try:
        origins.append((popup.winfo_rootx(), popup.winfo_rooty()))
    except tk.TclError:
        pass
    placed = getattr(popup, '_menu_origin', None)
    if placed is not None and placed not in origins:
        origins.append(placed)
    for left, top in origins:
        local_x = x - left
        local_y = y - top
        if 0 <= local_x < width and 0 <= local_y < height:
            return local_x, local_y
    return None


def _bind_scroll(widget):
    """A scroll must not be delivered to the window under the menu.

    On Aqua that scroll starts a gesture in the window underneath, and the
    menu's grab then swallows the click that should have closed it.
    """

    def hold(_event, widget=widget):
        try:
            popup = widget.winfo_toplevel()
        except tk.TclError:
            return 'break'
        _take_pointer(popup)
        return 'break'

    for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>', '<TouchpadScroll>'):
        try:
            widget.bind(sequence, hold)
        except tk.TclError:
            pass


def _note_buttons(down, x, y):
    """Close when a button goes down outside the menu.

    The opening click is still down until the buttons have been seen up.
    """
    global _buttons_released
    if not _open:
        return
    if not down:
        _buttons_released = True
        return
    if not _buttons_released or _opening_click or _switching:
        return
    if _pointer_over_menu(x, y):
        return
    _close_all()


def _pointer_over_menu(x, y):
    for menu in _open:
        popup = menu._popup
        if popup is not None and _local_point(popup, x, y) is not None:
            return True
    for bar in _bars:
        for button, _menu in getattr(bar, '_slots', ()):
            if _inside(button, x, y):
                return True
    return False


def _reassert_pointer(popup):
    try:
        current = popup.grab_current()
    except tk.TclError:
        return
    for menu in _open:
        if menu._popup is not None and current is menu._popup:
            return
    _take_pointer(popup)


def _aqua_buttons_down():
    """Whether the left, right, or middle button is down on Aqua."""
    lib = getattr(_aqua_buttons_down, 'lib', None)
    if lib is False:
        return False
    if lib is None:
        import ctypes
        try:
            lib = ctypes.CDLL(
                '/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
            lib.CGEventSourceButtonState.restype = ctypes.c_bool
            lib.CGEventSourceButtonState.argtypes = (ctypes.c_int, ctypes.c_uint)
        except (OSError, AttributeError):
            _aqua_buttons_down.lib = False
            return False
        _aqua_buttons_down.lib = lib
    for button in (0, 1, 2):
        if lib.CGEventSourceButtonState(0, button):
            return True
    return False


def _arm_pointer_watch():
    """Follow the pointer while a menu is open on Aqua.

    Mouse-moved events are delivered only to the key window, and this
    menu never becomes the key window. The pointer position is still
    available, so hover does not depend on those events.
    """
    global _pointer_job
    if _pointer_job is not None or not _open:
        return
    popup = _open[0]._popup
    if popup is None:
        return
    try:
        if popup.tk.call('tk', 'windowingsystem') != 'aqua':
            return
        top = _open[0]._toplevel()
    except tk.TclError:
        return
    if top is None:
        return
    try:
        _pointer_job = top.after(16, _poll_pointer)
    except tk.TclError:
        _pointer_job = None


def _poll_pointer():
    global _pointer_job
    _pointer_job = None
    if not _open or _switching:
        return
    popup = None
    for menu in _open:
        if menu._popup is not None:
            popup = menu._popup
            break
    if popup is None:
        return
    try:
        x, y = popup.winfo_pointerxy()
    except tk.TclError:
        _arm_pointer_watch()
        return
    if x >= 0 and y >= 0:
        _track(x, y)
        _note_buttons(_aqua_buttons_down(), x, y)
        if not _open:
            return
        _reassert_pointer(popup)
    _arm_pointer_watch()


def _on_popup_click(event):
    """A click outside the items. The menu holds the pointer, so this click
    does not also land on the editor."""
    x, y = event.x_root, event.y_root
    for bar in list(_bars):
        for button, menu in list(bar._slots):
            if not _inside(button, x, y):
                continue
            if menu is None:
                action = bar._commands.get(button)
                _close_all()
                if action is not None:
                    action()
            else:
                bar._toggle(button, menu)
            return 'break'
    _close_all()
    return 'break'


def _on_motion(event):
    global _motion_job
    if not _open:
        return
    try:
        top = _open[0]._toplevel()
    except tk.TclError:
        top = None
    if top is None:
        _close_all()
        return
    if _motion_job is not None:
        try:
            top.after_cancel(_motion_job)
        except tk.TclError:
            pass
    x, y = event.x_root, event.y_root
    _motion_job = top.after_idle(lambda: _track(x, y))


def _track(x, y):
    global _motion_job
    _motion_job = None
    if not _open:
        return
    for bar in list(_bars):
        for button, menu in bar._slots:
            if menu is None or not _inside(button, x, y):
                continue
            if menu in _open and menu._parent is None:
                return
            bar._open(button, menu)
            return
    for menu in reversed(list(_open)):
        popup = menu._popup
        if popup is None:
            continue
        local = _local_point(popup, x, y)
        if local is None:
            continue
        local_x, local_y = local
        for entry in menu._entries:
            row = entry.row
            if entry.kind == 'separator' or row is None:
                continue
            try:
                row_x, row_y = row.winfo_x(), row.winfo_y()
                row_w, row_h = row.winfo_width(), row.winfo_height()
            except tk.TclError:
                continue
            if row_x <= local_x < row_x + row_w and row_y <= local_y < row_y + row_h:
                if menu._active is not entry:
                    menu._hover(entry)
                return
        return


def _end_opening_click():
    global _opening_click
    _opening_click = False


def _on_click(event):
    if not _open or _opening_click:
        return
    widget = event.widget
    while widget is not None:
        if widget in _chrome:
            return
        widget = getattr(widget, 'master', None)
    _close_all()
    return 'break'


def _on_escape(event):
    if not _open:
        return
    _close_all()
    return 'break'


def _rgb(widget, color):
    red, green, blue = widget.winfo_rgb(color)
    return red / 65535, green / 65535, blue / 65535


def _parse(color):
    text = color.lstrip('#')
    return tuple(int(text[index:index + 2], 16) / 255 for index in (0, 2, 4))


def _mix(start, end, amount):
    return tuple(start[index] + (end[index] - start[index]) * amount for index in range(3))


def _hex_color(color):
    return '#{:02x}{:02x}{:02x}'.format(*(
        round(min(1, max(0, channel)) * 255) for channel in color
    ))
