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


def valid_colors(widget):
    if not isinstance(widget, ttk.Combobox) and not isinstance(widget, tkinter.Checkbutton):
        widget.config(
            highlightthickness=0)


def invalid_colors(widget):
    if not isinstance(widget, ttk.Combobox) and not isinstance(widget, tkinter.Checkbutton):
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
        self.before = config.get('before', None)
        self.after = config.get('after', None)

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
