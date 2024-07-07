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
import inspect
import json
import re
import tkinter as tk
import typing

import dgenerate.console.recipes as _recipes
import dgenerate.console.recipeformentries as _recipeformentries


class _RecipesForm(tk.Toplevel):
    def __init__(self, master=None, position: tuple[int, int] = None, width: int = None):
        super().__init__(master)
        self.title('Insert Recipe')

        self._entry_classes = {
            c.NAME: c for c in
            _recipeformentries.__dict__.values()
            if inspect.isclass(c) and issubclass(c, _recipeformentries._Entry)}

        self.minsize(600, 0)

        self._templates = None
        self._dropdown = None
        self._templates_dict: dict[str, str] = _recipes.RECIPES
        self._template_names: list[str] = list(_recipes.RECIPES.keys())

        self._current_template = tk.StringVar(value=self._template_names[0])
        self._entries: typing.List[_recipeformentries._Entry] = []
        self._content: str | None = None

        self._ok: bool = False
        self.transient(master)
        self.grab_set()
        self._create_form()
        self.resizable(True, False)

        self.withdraw()
        self.update_idletasks()

        if position is None:
            window_width = self.master.winfo_width()
            window_height = self.master.winfo_height()
            top_level_width = self.winfo_width()
            top_level_height = self.winfo_height()

            position_top = self.master.winfo_y() + window_height // 2 - top_level_height // 2
            position_left = self.master.winfo_x() + window_width // 2 - top_level_width // 2

            if width is not None:
                self.geometry(f"{width}x{self.winfo_reqheight()}+{position_left}+{position_top}")
            else:
                self.geometry(f"+{position_left}+{position_top}")

        else:
            if width is not None:
                self.geometry("{}x{}+{}+{}".format(width, self.winfo_reqheight(), *position))
            else:
                self.geometry("+{}+{}".format(*position))

        self.deiconify()

    def on_submit(self, callback):
        self._on_submit.append(callback)

    def _apply_templates(self):
        content = self._content
        missing_fields = False

        for callback in self._on_submit:
            callback()

        for entry in self._entries:
            content = entry.template(content)
            if entry.is_empty() and not entry.optional:
                entry.invalid()
                missing_fields = True
            else:
                entry.valid()

        if not missing_fields:
            self._content = content
            self._ok = True
            self.master.focus_set()
            self.destroy()

    def _create_form(self):
        self._on_submit = []
        self._dropdown = tk.OptionMenu(self,
                                       self._current_template, *self._template_names,
                                       command=lambda s: self._update_form(s, preserve_width=True))
        self._dropdown.grid(row=0, column=0, columnspan=3, sticky='ew', padx=(5, 5), pady=(5, 5))
        self._update_form(self._current_template.get())

    def _update_form(self, selection, preserve_width=False):
        for widget in self.winfo_children():
            if widget != self._dropdown:
                widget.destroy()

        self._entries = []
        self._content = self._templates_dict[selection]
        self._templates = re.findall(r'(@(.*?)\[(.*)\])', self._content)
        self.grid_columnconfigure(1, weight=1)

        row_offset = 1
        for template in self._templates:
            ttype = template[1]
            config = template[2]
            if ttype in self._entry_classes:
                try:
                    entry = self._entry_classes[ttype](
                        recipe_form=self,
                        master=self,
                        row=row_offset,
                        config=json.loads(config),
                        placeholder=template[0])
                except json.JSONDecodeError as e:
                    raise RuntimeError(f'json decode error in {ttype}: {e}')
                row_offset += entry.widget_rows
            else:
                raise RuntimeError(f'Unknown template placeholder: {ttype}')

            self._entries.append(entry)

        apply_button = tk.Button(self, text='Insert', command=self._apply_templates)
        apply_button.grid(row=row_offset, column=0, padx=(5, 5), pady=(5, 5), columnspan=3)

        self.update_size(preserve_width)

    def update_size(self, preserve_width=False):
        old_form_width = self.winfo_width()
        if preserve_width:
            self.update_idletasks()
            self.geometry(f"{old_form_width}x{self.winfo_reqheight()}")
        else:
            self.geometry('')

    def get_recipe(self):
        self.wait_window(self)
        return ('\n'.join(line.lstrip() for line in self._content.splitlines())).strip() if self._ok else None


_last_pos = None
_last_width = None


def request_recipe(master):
    global _last_width, _last_pos

    window = _RecipesForm(master, position=_last_pos, width=_last_width)

    og_destroy = window.destroy

    def destroy():
        global _last_width, _last_pos
        _last_width = window.winfo_width()
        _last_pos = _last_size = (window.winfo_x(), window.winfo_y())
        og_destroy()

    window.destroy = destroy

    def on_closing():
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    return window.get_recipe()
