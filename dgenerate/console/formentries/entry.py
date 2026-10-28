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
import tkinter
import tkinter.ttk as ttk

import dgenerate.console.helpdialog as _helpdialog
import dgenerate.console.resources as _resources
import dgenerate.textprocessing as _textprocessing
import typing

ROW_XPAD = (5, 2)
DIVIDER_YPAD = (5, 5)


def replace_first(text, old, new):
    out = []
    replaced = False
    for line in text.splitlines():
        if not replaced:
            r = line.replace(old, new, 1)
            replaced = r != line
        else:
            r = line
        if r.strip():
            out.append(r)
        if not line.strip():
            out.append('')
    return '\n'.join(out)


def shell_quote_if(text, strict: bool = False):
    if text and ' ' in text or "'" in text or '"' in text \
            and '{%' not in text \
            and '{{' not in text \
            and '{#' not in text:
        return _textprocessing.shell_quote(text, strict=strict)
    return text


def _highlight_border(widget):
    return not isinstance(widget, (ttk.Widget, tkinter.Checkbutton))


def valid_colors(widget):
    if _highlight_border(widget):
        widget.config(
            highlightthickness=0)


def invalid_colors(widget):
    if _highlight_border(widget):
        widget.config(
            highlightbackground="red",
            highlightcolor="red",
            highlightthickness=2)


class _Entry:
    NAME = None

    def __init__(self,
                 form,
                 master,
                 row: int,
                 config: dict[str, typing.Any],
                 placeholder: str,
                 widget_rows: int
                 ):
        self.form = form
        self.master = master
        self.widget_rows = widget_rows
        self.row = row
        self.config = config
        self.placeholder = placeholder
        self.optional = config.get('optional', True)

        if not isinstance(self.optional, bool):
            raise RuntimeError('Recipe used optional = str instead of bool.')

        self.arg = config.get('arg', None)
        self.help_arg = config.get('help-arg', None)
        self.before = config.get('before', None)
        self.after = config.get('after', None)
        self._row_inactive = False
        self._saved_widget_states: dict[str, str] = {}

    def set_row_enabled(self, enabled: bool) -> None:
        """Enable or gray out this recipe row. A gray row emits no argument."""
        if enabled != self._row_inactive:
            # Already in the requested state.
            return
        self._row_inactive = not enabled

        def set_widget_enabled(widget):
            try:
                if enabled:
                    # Restore the state the widget had before the row was
                    # grayed out, such as 'readonly' for a combobox.
                    widget.configure(state=self._saved_widget_states.pop(str(widget), 'normal'))
                else:
                    self._saved_widget_states[str(widget)] = str(widget.cget('state'))
                    widget.configure(state='disabled')
            except tkinter.TclError:
                pass
            for child in widget.winfo_children():
                set_widget_enabled(child)

        # An entry may put controls such as spin buttons in sibling widgets,
        # or put several controls on separate rows. Apply the state to every
        # widget laid out in this entry's rows, including nested controls.
        seen = set()
        for row in range(self.row, self.row + self.widget_rows):
            for widget in self.master.grid_slaves(row=row):
                if str(widget) not in seen:
                    seen.add(str(widget))
                    set_widget_enabled(widget)

        if enabled:
            # Entry values may have changed while the row was gray, for example
            # a mutually exclusive checkbox being unchecked, so states the
            # entry manages itself are reapplied from its current values.
            self.sync_widget_states()

    def sync_widget_states(self) -> None:
        """Reapply widget states this entry manages from its own values."""
        pass

    def get_label(self, default, key='label'):
        if self.optional:
            return f'(Optional) {self.config.get(key, default)}'
        return self.config.get(key, default)

    def _template(self, content, new_content):
        if new_content:
            new_content = (self.before if self.before else '') + new_content + (self.after if self.after else '')
        if self.arg and new_content:
            new_content = self.arg + ' ' + new_content
        return replace_first(content, self.placeholder, new_content)

    def is_empty(self) -> bool:
        return False

    def is_active(self) -> bool:
        return not self.is_empty()

    def is_valid(self) -> bool:
        return True

    def invalid(self):
        pass

    def valid(self):
        pass

    def template(self, content):
        return self._template(content, 'OVERRIDE ME')

    def on_form_scroll(self):
        pass

    def argument_help_targets(self) -> list[tuple[tkinter.Widget, str]]:
        """
        Labels that should offer help for a command line option.

        The first item is the label, the second is the option name.
        """
        targets = []
        seen = set()

        def add(widget, option):
            if widget is None or not isinstance(option, str) or not option.startswith('-'):
                return
            identity = id(widget)
            if identity in seen:
                return
            seen.add(identity)
            targets.append((widget, option))

        primary = self.arg or self.help_arg
        add(getattr(self, 'label_widget', None), primary)
        add(getattr(self, 'uri_label', None), primary)
        add(getattr(self, 'dropdown_label', None), primary)
        add(getattr(self, 'switch_label', None), primary)
        add(getattr(self, 'float_label', None), getattr(self, 'float_arg', None))
        add(getattr(self, 'int1_label', None), getattr(self, 'int1_arg', None))
        add(getattr(self, 'int2_label', None), getattr(self, 'int2_arg', None))
        add(getattr(self, '_map_label', None), getattr(self, '_map_arg', None))

        labels = getattr(self, 'label_widgets', None)
        options = getattr(self, 'args', None)
        if labels and options and len(labels) == len(options):
            for label, option in zip(labels, options):
                add(label, option)
        return targets

    def install_argument_help(self) -> None:
        """Bind right-click help on labels that correspond to a command line option."""
        arguments = _resources.get_dgenerate_arguments()
        for label, option in self.argument_help_targets():
            help_text = arguments.get(option)
            if not help_text:
                continue
            _bind_argument_help(label, option, help_text, self.form)


_HELP_TIP = None


def _bind_argument_help(label, option: str, help_text: str, form) -> None:
    label.configure(cursor='question_arrow')

    menu = tkinter.Menu(label, tearoff=0)

    def show_help():
        _hide_help_tip()
        _helpdialog.show_help_dialog(
            form,
            title=option,
            help_text=help_text,
            position_widget=form)

    menu.add_command(label=f'Help for {option}', command=show_help)

    def popup(event):
        _hide_help_tip()
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    label.bind('<Button-3>', popup, add='+')
    label.bind('<Enter>', lambda _event: _show_help_tip(label, 'Right-click for help'), add='+')
    label.bind('<Leave>', lambda _event: _hide_help_tip(), add='+')
    label.bind('<Destroy>', lambda _event: _hide_help_tip(), add='+')


def _show_help_tip(widget, text: str) -> None:
    global _HELP_TIP
    _hide_help_tip()
    tip = tkinter.Toplevel(widget)
    tip.wm_overrideredirect(True)
    tip.wm_attributes('-topmost', True)
    body = ttk.Label(tip, text=text, relief='solid', borderwidth=1, padding=(6, 3))
    body.pack()
    tip.update_idletasks()
    x = widget.winfo_rootx() + widget.winfo_width() - tip.winfo_reqwidth()
    y = widget.winfo_rooty() + widget.winfo_height() + 2
    tip.geometry(f'+{x}+{y}')
    _HELP_TIP = tip


def _hide_help_tip() -> None:
    global _HELP_TIP
    tip = _HELP_TIP
    _HELP_TIP = None
    if tip is not None:
        try:
            tip.destroy()
        except tkinter.TclError:
            pass
