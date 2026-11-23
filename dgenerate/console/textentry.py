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


import tkinter as tk
import tkinter.font as tkfont
import tkinter.ttk as ttk

import dgenerate.console.appearance as _appearance
import dgenerate.console.thememenu as _thememenu
import dgenerate.console.combobox as _combobox


def _suggestion_pairs(suggestions):
    """Stored value and the text to show for each recipe suggestion.

    A string is shown as itself. ``{"label": "Q4_K_S", "value": "<url>"}``
    shows the label and writes the value. Labels come from the recipe.
    """
    pairs = []
    seen_values = set()
    seen_labels = set()
    for item in suggestions or []:
        if isinstance(item, str):
            value = label = item
        elif isinstance(item, dict):
            value = item.get('value') or ''
            label = item.get('label') or value
        else:
            continue
        if not value or value in seen_values:
            continue
        if label in seen_labels:
            label = value
        seen_values.add(value)
        seen_labels.add(label)
        pairs.append((value, label))
    return pairs


class _LabeledCombo(_combobox.ComboBox):
    """
    Editable dropdown that can show a recipe label instead of the stored value.

    ``variable`` keeps the value the recipe writes. The list shows ``label``
    when the suggestion provides one, and the value itself when it does not.
    """

    def __init__(self, parent, variable, pairs):
        self._value_var = variable
        self._display_var = tk.StringVar(master=parent)
        self._by_label = {}
        self._by_value = {}
        labels = []
        for value, label in pairs:
            self._by_label[label] = value
            self._by_value[value] = label
            labels.append(label)
        self._syncing = False
        super().__init__(
            parent, textvariable=self._display_var, values=labels, state='normal')
        self._display_var.set(self._by_value.get(variable.get(), variable.get()))
        variable.trace_add('write', self._show_value)
        self._display_var.trace_add('write', self._store_display)
        _bind_text_tip(self, self._tip_text)
        self.bind('<<Copy>>', self._copy_stored)

    def _show_value(self, *_args):
        if self._syncing:
            return
        self._syncing = True
        try:
            current = self._value_var.get()
            self._display_var.set(self._by_value.get(current, current))
        finally:
            self._syncing = False

    def _store_display(self, *_args):
        if self._syncing:
            return
        self._syncing = True
        try:
            shown = self._display_var.get()
            self._value_var.set(self._by_label.get(shown, shown))
        finally:
            self._syncing = False

    def _copy_stored(self, _event):
        """Copy the stored value when the whole short name is selected."""
        shown = self._display_var.get()
        stored = self._value_var.get()
        if not stored or stored == shown:
            return
        try:
            selected = self.selection_get() if self.selection_present() else ''
        except tk.TclError:
            return
        if selected != shown:
            return
        self.clipboard_clear()
        self.clipboard_append(stored)
        return 'break'

    def _tip_text(self):
        if self._is_dropdown_open:
            return ''
        stored = self._value_var.get()
        shown = self._display_var.get()
        if not stored:
            return ''
        # A short recipe name hides the value. A value shown as itself still
        # needs the tip when the field clips it.
        if stored != shown or _text_overruns(self, shown):
            return stored
        return ''

    def destroy(self):
        _hide_text_tip(self)
        super().destroy()


def value_entry(parent, textvariable, suggestions=None):
    """
    A text field, or an editable dropdown when more than one suggestion is known.

    The current value stays whatever the caller put in ``textvariable``.
    Suggestions are extra values the user can pick. A suggestion with a
    ``label`` shows that name; the variable still holds ``value``. A typed
    path is kept as typed.
    """
    pairs = _suggestion_pairs(suggestions)
    if len(pairs) > 1 and any(label != value for value, label in pairs):
        return _LabeledCombo(parent, textvariable, pairs)
    if len(pairs) > 1:
        widget = _combobox.ComboBox(
            parent,
            textvariable=textvariable,
            values=[value for value, _label in pairs],
            state='normal')
        _bind_text_tip(widget, lambda widget=widget: _plain_tip_text(widget))
        return widget
    return TextEntry(parent, textvariable=textvariable)


def _plain_tip_text(widget):
    if getattr(widget, '_is_dropdown_open', False):
        return ''
    text = widget.get()
    if text and _text_overruns(widget, text):
        return text
    return ''


def _text_overruns(widget, text):
    """True when ``text`` is wider than the field's text area.

    ``Entry.bbox`` reports an empty box until the widget has been drawn,
    so the width comes from the font instead.
    """
    if not text:
        return False
    area = _text_area_width(widget)
    if area <= 1:
        return False
    return tkfont.Font(font=widget.cget('font')).measure(text) > area


def _text_area_width(widget):
    width = widget.winfo_width()
    if width <= 1:
        return 0
    if isinstance(widget, ttk.Combobox):
        mid = max(widget.winfo_height() // 2, 0)
        x = width - 1
        while x > 0:
            if 'textarea' in str(widget.identify(x, mid)):
                return max(0, x - 2)
            x -= 4
        return max(0, width - 24)
    border = int(widget.cget('borderwidth') or 0)
    highlight = int(widget.cget('highlightthickness') or 0)
    return max(0, width - 2 * (border + highlight + 3))


def _bind_text_tip(widget, text_of):
    widget._text_tip = None
    widget._text_tip_fn = text_of
    widget.bind('<Enter>', lambda _event: _show_text_tip(widget), add='+')
    widget.bind('<Leave>', lambda _event: _hide_text_tip(widget), add='+')
    widget.bind('<ButtonPress>', lambda _event: _hide_text_tip(widget), add='+')
    widget.bind('<Destroy>', lambda _event: _hide_text_tip(widget), add='+')
    if isinstance(widget, ttk.Combobox):
        widget.bind('<<ComboboxDropdown>>', lambda _event: _hide_text_tip(widget), add='+')


def _show_text_tip(widget):
    _hide_text_tip(widget)
    text = widget._text_tip_fn()
    if not text:
        return
    tip = tk.Toplevel(widget)
    tip.wm_overrideredirect(True)
    ttk.Label(
        tip, text=text, justify='left', relief='solid', borderwidth=1,
        padding=(6, 3), wraplength=480
    ).pack()
    tip.update_idletasks()
    width = tip.winfo_reqwidth()
    x = widget.winfo_rootx()
    # Stay under the field. The list opens downward from that same edge,
    # and a tip placed above the field covers it.
    y = widget.winfo_rooty() + widget.winfo_height() + 2
    if x + width > widget.winfo_screenwidth():
        x = max(0, widget.winfo_screenwidth() - width)
    tip.wm_geometry(f'+{x}+{y}')
    widget._text_tip = tip


def _hide_text_tip(widget):
    tip = getattr(widget, '_text_tip', None)
    widget._text_tip = None
    if tip is not None:
        try:
            tip.destroy()
        except tk.TclError:
            pass


class TextEntry(tk.Entry):
    """
    A tk.Entry widget with a right-click context menu providing copy, cut, paste, and delete operations.
    """
    
    def __init__(self, parent, **kwargs):
        if _appearance.native_controls(parent) and not any(
                key in kwargs for key in ('bg', 'background')):
            kwargs['background'] = _appearance.field_background()
        super().__init__(parent, **kwargs)

        self.context_menu = _thememenu.ThemeMenu(self, tearoff=0)
        self.context_menu.add_command(label="Cut", command=self.__context_cut)
        self.context_menu.add_command(label="Copy", command=self.__context_copy)
        self.context_menu.add_command(label="Paste", command=self.__context_paste)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Delete", command=self.__context_delete)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Select All", command=self.__context_select_all)

        self.bind("<Button-3>", self.__show_context_menu)
        _bind_text_tip(self, lambda self=self: _plain_tip_text(self))
    
    def __show_context_menu(self, event):
        self.focus()
        self.__update_menu_state()
        
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
    
    def __update_menu_state(self):
        has_selection = self.selection_present()

        has_clipboard = False
        try:
            self.clipboard_get()
            has_clipboard = True
        except tk.TclError:
            pass

        self.context_menu.entryconfig("Cut", state="normal" if has_selection else "disabled")
        self.context_menu.entryconfig("Copy", state="normal" if has_selection else "disabled")
        self.context_menu.entryconfig("Paste", state="normal" if has_clipboard else "disabled")
        self.context_menu.entryconfig("Delete", state="normal" if has_selection else "disabled")
    
    def __context_cut(self):
        if self.selection_present():
            self.clipboard_clear()
            self.clipboard_append(self.selection_get())
            super().delete(tk.SEL_FIRST, tk.SEL_LAST)
    
    def __context_copy(self):
        if self.selection_present():
            self.clipboard_clear()
            self.clipboard_append(self.selection_get())
    
    def __context_paste(self):
        try:
            text = self.clipboard_get()
            if self.selection_present():
                super().delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.insert(tk.INSERT, text)
        except tk.TclError:
            pass
    
    def __context_delete(self):
        if self.selection_present():
            super().delete(tk.SEL_FIRST, tk.SEL_LAST)
    
    def __context_select_all(self):
        self.select_range(0, tk.END) 