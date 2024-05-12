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

import re
import tkinter as tk
from tkinter import filedialog

import dgenerate.console.recipes as _recipes
import dgenerate.console.resources as _resources


class _RecipesForm(tk.Toplevel):
    def __init__(self, master=None, position=None, width=None):
        super().__init__(master)
        self.title('Insert Recipe')
        self._templates = None
        self._dropdown = None
        self._templates_dict = _recipes.RECIPES
        self._template_names = list(_recipes.RECIPES.keys())
        self._current_template = tk.StringVar(value=self._template_names[0])
        self._entries = []
        self._content = None
        self._ok = False
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
                self.geometry(f"{width}x{self.winfo_height()}+{position_left}+{position_top}")
            else:
                self.geometry(f"+{position_left}+{position_top}")

        else:
            if width is not None:
                self.geometry("{}x{}+{}+{}".format(width, self.winfo_height(), *position))
            else:
                self.geometry("+{}+{}".format(*position))

        self.deiconify()

    @staticmethod
    def _read_template_string(template_string):
        templates = re.findall(r'@(.*?)\[(.*?)\]:\[(.*?)\]:"(.*?)"', template_string)
        return templates, template_string

    @staticmethod
    def _open_file_dialog():
        f = filedialog.askopenfilename()
        if f is None or not f.strip():
            return None
        return f

    @staticmethod
    def _open_file_save_dialog():
        f = filedialog.asksaveasfilename()
        if f is None or not f.strip():
            return None
        return f

    @staticmethod
    def _open_directory_dialog():
        d = filedialog.askdirectory()
        if d is None or not d.strip():
            return None
        return d

    @staticmethod
    def _replace_txt(text, old, new):
        out = []
        for line in text.splitlines():
            r = line.replace(old, new)
            if r.strip():
                out.append(r)
            if not line.strip():
                out.append('')
        return '\n'.join(out)

    def _apply_templates(self):
        missing_fields = False
        have_scheduler = False
        for entry, template in zip(self._entries, self._templates):
            ttype = template[0]
            label = template[1]
            opt = template[2]
            text = template[3]
            entry_text = entry[1].get().strip()
            entry = entry[0]
            replace_str = f'@{ttype}[{label}]:[{opt}]:"{text}"'
            opt = opt.replace('\\n', '\n')

            if 'scheduler' in ttype and entry_text:
                have_scheduler = True

            if 'predictiontype' in ttype:
                if not have_scheduler:
                    self._content = self._replace_txt(self._content,
                                                      replace_str, '')
                    continue
                else:
                    have_scheduler = False

            if ttype.startswith('optional'):
                self._content = self._replace_txt(self._content,
                                                  replace_str, opt.format(entry_text) if entry_text else '')
            else:
                if entry_text:
                    self._content = self._replace_txt(self._content,
                                                      replace_str, opt.format(entry_text))
                    entry.config(highlightthickness=0)
                else:
                    entry.config(highlightbackground="red",
                                 highlightcolor="red",
                                 highlightthickness=2)
                    missing_fields = True
        if not missing_fields:
            self._ok = True
            self.master.focus_set()
            self.destroy()

    def _create_form(self):
        self._dropdown = tk.OptionMenu(self,
                                       self._current_template, *self._template_names,
                                       command=lambda s: self._update_form(s, preserve_width=True))
        self._dropdown.grid(row=0, column=0, columnspan=3, sticky='ew', padx=(5, 5), pady=(5, 5))
        self._update_form(self._current_template.get())

    def _update_form(self, selection, preserve_width=False):
        old_form_width = self.winfo_width()

        for widget in self.winfo_children():
            if widget != self._dropdown:
                widget.destroy()

        self._entries = []
        self._templates, self._content = self._read_template_string(self._templates_dict[selection])
        for i, template in enumerate(self._templates):
            ttype = template[0]
            label = ('(Optional) ' if ttype.startswith('optional') else '') + template[1]
            default_value = template[3]
            label_widget = tk.Label(self, text=label, anchor='e')
            if ttype == 'device':
                devices = _resources.get_cuda_devices()
                text_var = tk.StringVar(value=devices[0])
                entry = tk.OptionMenu(self, text_var, *devices)
            elif ttype.endswith('scheduler'):
                schedulers = _resources.get_karras_schedulers()
                if ttype.startswith('optional'):
                    schedulers = [''] + schedulers
                text_var = tk.StringVar(value=default_value)
                entry = tk.OptionMenu(self, text_var, *schedulers)
            elif ttype.endswith('predictiontype'):
                prediction_types = _resources.get_karras_scheduler_prediction_types()
                if ttype.startswith('optional'):
                    prediction_types = [''] + prediction_types
                text_var = tk.StringVar(value=default_value)
                entry = tk.OptionMenu(self, text_var, *prediction_types)
            elif ttype.startswith('int'):
                text_var = tk.StringVar(value=default_value)
                entry = tk.Spinbox(self, from_=-10000, to=10000,
                                   textvariable=text_var)  # adjust the maximum value as needed
            elif ttype.startswith('float'):
                text_var = tk.StringVar(value=default_value)
                entry = tk.Spinbox(self, from_=-10000, to=10000, format="%.2f", increment=0.01,
                                   textvariable=text_var)
            elif ttype.endswith('string'):
                text_var = tk.StringVar(value=default_value)
                entry = tk.Entry(self, textvariable=text_var)
            elif ttype.endswith('file'):  # filename
                text_var = tk.StringVar(value=default_value)
                entry = tk.Entry(self, textvariable=text_var)

                def select_command(t=text_var, tt=ttype):
                    if 'output' in tt:
                        r = self._open_file_save_dialog()
                    else:
                        r = self._open_file_dialog()
                    if r is not None:
                        t.set(r)

                if 'output' in ttype:
                    button = tk.Button(self, text="Save File",
                                       command=select_command)
                else:
                    button = tk.Button(self, text="Select File",
                                       command=select_command)

                button.grid(row=i + 1, column=2, padx=(2, 5), sticky='w')
            elif ttype.endswith('dir'):  # directory
                text_var = tk.StringVar(value=default_value)
                entry = tk.Entry(self, textvariable=text_var)

                def select_command(t=text_var):
                    r = self._open_directory_dialog()
                    if r is not None:
                        t.set(r)

                button = tk.Button(self, text="Select Directory", command=select_command)

                button.grid(row=i + 1, column=2, padx=(2, 5), sticky='w')
            else:
                raise RuntimeError(f'Unknown template placeholder: {ttype}')

            entry.bind('<Key>', lambda e: entry.config(highlightthickness=0))
            label_widget.grid(row=i + 1, column=0, padx=(5, 2), sticky='e')
            entry.config(width=40)
            entry.grid(row=i + 1, column=1, sticky='ew')

            self.grid_columnconfigure(1, weight=1)
            self._entries.append((entry, text_var))

        apply_button = tk.Button(self, text='Insert', command=self._apply_templates)
        apply_button.grid(row=len(self._templates) + 1, column=0, padx=(5, 5), pady=(5, 5), columnspan=3)

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
