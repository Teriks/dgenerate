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
import json
import re
import shlex
import tkinter as tk
import typing
from tkinter import filedialog

import dgenerate.console.recipes as _recipes
import dgenerate.console.resources as _resources


def _open_file_dialog(*args, **kwargs):
    f = filedialog.askopenfilename(*args, **kwargs)
    if f is None or not f.strip():
        return None
    return f


def _open_file_save_dialog(*args, **kwargs):
    f = filedialog.asksaveasfilename(*args, **kwargs)
    if f is None or not f.strip():
        return None
    return f


def _open_directory_dialog(*args, **kwargs):
    d = filedialog.askdirectory(*args, **kwargs)
    if d is None or not d.strip():
        return None
    return d


def _replace_first(text, old, new):
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


def _shell_quote_if(text):
    if text:
        return shlex.quote(text)
    return text


class _Entry:
    def __init__(self, master, row: int, config: dict[str, typing.Any], placeholder: str, widget_rows: int):
        self.master = master
        self.widget_rows = widget_rows
        self.row = row
        self.config = config
        self.placeholder = placeholder
        self.optional = config.get('optional', True)
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
        return _replace_first(content, self.placeholder, new_content)

    def is_empty(self):
        pass

    def invalid(self):
        pass

    def valid(self):
        pass

    def template(self, content):
        return self._template(content, 'OVERRIDE ME')


class _StringEntry(_Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.text_var = tk.StringVar(
            value=self.config.get('default', ''))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('String'), anchor='e')

        self.entry = tk.Entry(self.master, textvariable=self.text_var)

        self.label_widget.grid(row=self.row, column=0, padx=(5, 2), sticky='e')
        self.entry.grid(row=self.row, column=1, padx=(5, 2), sticky='ew')
        self.entry.bind('<Key>', lambda e: self.valid())

    def invalid(self):
        self.entry.config(highlightbackground="red",
                          highlightcolor="red",
                          highlightthickness=2)

    def valid(self):
        self.entry.config(highlightthickness=0)

    def is_empty(self):
        return self.text_var.get().strip() == ''

    def template(self, content):
        return self._template(content, _shell_quote_if(self.text_var.get().strip()))


class _DeviceEntry(_Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        if self.arg is None:
            self.arg = '--device'

        devices = _resources.get_cuda_devices()

        if self.optional:
            devices = [''] + devices

        self.text_var = tk.StringVar(
            value=self.config.get('default', devices[0]))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('Device'), anchor='e')

        self.entry = tk.OptionMenu(self.master,
                                   self.text_var,
                                   *devices)

        self.label_widget.grid(row=self.row, column=0, padx=(5, 2), sticky='e')
        self.entry.grid(row=self.row, column=1, padx=(5, 2), sticky='ew')

    def invalid(self):
        self.entry.config(highlightbackground="red",
                          highlightcolor="red",
                          highlightthickness=2)

    def valid(self):
        self.entry.config(highlightthickness=0)

    def is_empty(self):
        return self.text_var.get().strip() == ''

    def template(self, content):
        return self._template(content, self.text_var.get().strip())


class _IntEntry(_Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.min = self.config.get('min', float('-inf'))
        self.max = self.config.get('max', float('inf'))

        self.text_var = tk.StringVar(
            value=self.config.get('default', ''))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('Int'), anchor='e')

        def validate_input(event):
            t = self.text_var.get()
            if not t.isdigit():
                self.text_var.set(self.min)
            elif int(t) < self.min:
                self.text_var.set(self.min)
            elif int(t) > self.max:
                self.text_var.set(self.max)

        def increment(delta):
            validate_input(None)
            value = int(self.text_var.get())
            value = max(self.min, min(self.max, value + delta))
            self.text_var.set(value)

        def on_mouse_wheel(event):
            delta = -1 if event.delta < 0 else 1
            increment(delta)

        self.entry = tk.Spinbox(self.master,
                                from_=self.min,
                                to=self.max,
                                textvariable=self.text_var)

        self.entry.bind('<Return>', validate_input)
        self.entry.bind('<FocusOut>', validate_input)
        self.entry.bind('<MouseWheel>', on_mouse_wheel)
        self.master.on_submit(lambda: validate_input(None))

        self.label_widget.grid(row=self.row, column=0, padx=(5, 2), sticky='e')
        self.entry.grid(row=self.row, column=1, padx=(5, 2), sticky='ew')
        self.entry.bind('<Key>', lambda e: self.valid())

    def invalid(self):
        self.entry.config(highlightbackground="red",
                          highlightcolor="red",
                          highlightthickness=2)

    def valid(self):
        self.entry.config(highlightthickness=0)

    def is_empty(self):
        return self.text_var.get().strip() == ''

    def template(self, content):
        return self._template(content, self.text_var.get().strip())


class _FloatEntry(_Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.min = self.config.get('min', float('-inf'))
        self.max = self.config.get('max', float('inf'))

        self.text_var = tk.StringVar(
            value=self.config.get('default', ''))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('Float'), anchor='e')

        def validate_input(event):
            t = self.text_var.get()
            try:
                if float(t) < self.min:
                    self.text_var.set(self.min)
                elif float(t) > self.max:
                    self.text_var.set(self.max)
            except ValueError:
                self.text_var.set(self.min)

        def increment(delta):
            validate_input(None)
            value = float(self.text_var.get())
            value = max(self.min, min(self.max, round(value + (delta*0.01), 2)))
            self.text_var.set(value)

        def on_mouse_wheel(event):
            delta = -1 if event.delta < 0 else 1
            increment(delta)

        self.entry = tk.Spinbox(self.master,
                                from_=self.min,
                                to=self.max,
                                format="%.2f", increment=0.01,
                                textvariable=self.text_var)

        self.entry.bind('<Return>', validate_input)
        self.entry.bind('<FocusOut>', validate_input)
        self.entry.bind('<MouseWheel>', on_mouse_wheel)
        self.master.on_submit(lambda: validate_input(None))


        self.label_widget.grid(row=self.row, column=0, padx=(5, 2), sticky='e')
        self.entry.grid(row=self.row, column=1, padx=(5, 2), sticky='ew')
        self.entry.bind('<Key>', lambda e: self.valid())

    def invalid(self):
        self.entry.config(highlightbackground="red",
                          highlightcolor="red",
                          highlightthickness=2)

    def valid(self):
        self.entry.config(highlightthickness=0)

    def is_empty(self):
        return self.text_var.get().strip() == ''

    def template(self, content):
        return self._template(content, self.text_var.get().strip())


class _FileEntry(_Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.text_var = tk.StringVar(value=self.config.get('default', ''))

        self.label_widget = tk.Label(self.master,
                                     text=self.get_label('File'), anchor='e')

        self.entry = tk.Entry(self.master, textvariable=self.text_var)
        self.select_mode = self.config.get('mode', 'input')
        self.file_types = self.config.get('file-types', None)

        dialog_args = dict()
        if self.file_types == 'models':
            dialog_args['filetypes'] = [('Models', ' *.'.join(_resources.supported_torch_model_formats_open()))]

        def select_command():
            if 'output' in self.select_mode:
                r = _open_file_save_dialog(**dialog_args)
            else:
                r = _open_file_dialog(**dialog_args)
            if r is not None:
                self.text_var.set(r)

        if 'output' in self.select_mode:
            self.button = tk.Button(self.master, text="Save File",
                                    command=select_command)
        else:
            self.button = tk.Button(self.master, text="Select File",
                                    command=select_command)

        self.label_widget.grid(row=self.row, column=0, padx=(5, 2), sticky='e')
        self.entry.grid(row=self.row, column=1, padx=(5, 2), sticky='ew')
        self.button.grid(row=self.row, column=2, padx=(2, 5), sticky='w')
        self.entry.bind('<Key>', lambda e: self.valid())

    def invalid(self):
        self.entry.config(highlightbackground="red",
                          highlightcolor="red",
                          highlightthickness=2)

    def valid(self):
        self.entry.config(highlightthickness=0)

    def is_empty(self):
        return self.text_var.get().strip() == ''

    def template(self, content):
        return self._template(content, _shell_quote_if(self.text_var.get().strip()))


class _DirectoryEntry(_Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.text_var = tk.StringVar(value=self.config.get('default', ''))

        self.label_widget = tk.Label(self.master,
                                     text=self.get_label('Directory'), anchor='e')

        self.entry = tk.Entry(self.master, textvariable=self.text_var)
        self.select_mode = self.config.get('mode', 'input')

        def select_command():
            r = _open_directory_dialog()
            if r is not None:
                self.text_var.set(r)

        self.button = tk.Button(self.master,
                                text="Select Directory", command=select_command)

        self.label_widget.grid(row=self.row, column=0, padx=(5, 2), sticky='e')
        self.entry.grid(row=self.row, column=1, padx=(5, 2), sticky='ew')
        self.button.grid(row=self.row, column=2, padx=(2, 5), sticky='w')
        self.entry.bind('<Key>', lambda e: self.valid())

    def invalid(self):
        self.entry.config(highlightbackground="red",
                          highlightcolor="red",
                          highlightthickness=2)

    def is_empty(self):
        return self.text_var.get().strip() == ''

    def valid(self):
        self.entry.config(highlightthickness=0)

    def template(self, content):
        return self._template(content, _shell_quote_if(self.text_var.get().strip()))


class _KarrasSchedulerEntry(_Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=2)

        if self.arg is None:
            self.arg = '--scheduler'

        schedulers = _resources.get_karras_schedulers()

        if self.optional:
            schedulers = [''] + schedulers

        self.scheduler_name_text_var = \
            tk.StringVar(value=self.config.get('default', schedulers[0]))

        prediction_types = [''] + _resources.get_karras_scheduler_prediction_types()

        self.prediction_type_entry_text_var = \
            tk.StringVar(value=self.config.get('prediction-type', ''))

        scheduler_name_label_text = self.get_label('Scheduler')

        self.scheduler_name_label = tk.Label(
            self.master, text=scheduler_name_label_text, anchor='e')

        self.scheduler_name_entry = tk.OptionMenu(
            self.master, self.scheduler_name_text_var, *schedulers)

        self.prediction_type_label = tk.Label(
            self.master, text=scheduler_name_label_text + ' Prediction Type', anchor='e'
        )

        self.prediction_type_entry = tk.OptionMenu(
            self.master, self.prediction_type_entry_text_var, *prediction_types)

        self.scheduler_name_label.grid(row=self.row, column=0, padx=(5, 2), sticky='e')
        self.scheduler_name_entry.grid(row=self.row, column=1, padx=(5, 2), sticky='ew')

        self.prediction_type_label.grid(row=self.row + 1, column=0, padx=(5, 2), sticky='e')
        self.prediction_type_entry.grid(row=self.row + 1, column=1, padx=(5, 2), sticky='ew')

    def invalid(self):
        self.scheduler_name_entry.config(
            highlightbackground="red",
            highlightcolor="red",
            highlightthickness=2)

    def is_empty(self):
        return self.scheduler_name_text_var.get().strip() == ''

    def valid(self):
        self.scheduler_name_entry.config(highlightthickness=0)

    def template(self, content):
        value = self.scheduler_name_text_var.get().strip()
        pred_type = self.prediction_type_entry_text_var.get().strip()

        if value and pred_type:
            value += f';prediction-type={pred_type}'

        return self._template(content, value)


class _TorchVaeEntry(_Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=2)

        if self.arg is None:
            self.arg = '--vae'

        self.vae_uri_var = \
            tk.StringVar(value=self.config.get('default', ''))

        vae_uri_label_text = self.get_label('VAE')

        self.vae_uri_label = tk.Label(
            self.master, text=vae_uri_label_text, anchor='e')

        self.vae_uri_entry = \
            tk.Entry(self.master, textvariable=self.vae_uri_var)

        vae_types = _resources.get_torch_vae_types()

        self.vae_type_var = \
            tk.StringVar(value=self.config.get('vae-type', vae_types[0]))

        self.vae_type_label = tk.Label(
            self.master, text='VAE Type', anchor='e')

        self.vae_type_entry = tk.OptionMenu(
            self.master, self.vae_type_var, *vae_types)

        def select_command():
            r = _open_file_dialog(
                filetypes=[('Models', ' *.'.join(_resources.supported_torch_model_formats_open()))])
            if r is not None:
                self.vae_uri_var.set(r)

        self.button = tk.Button(self.master, text="Select File",
                                command=select_command)

        self.vae_uri_label.grid(row=self.row, column=0, padx=(5, 2), sticky='e')
        self.vae_uri_entry.grid(row=self.row, column=1, padx=(5, 2), sticky='ew')
        self.button.grid(row=self.row, column=2, padx=(2, 5), sticky='w')

        self.vae_type_label.grid(row=self.row + 1, column=0, padx=(5, 2), sticky='e')
        self.vae_type_entry.grid(row=self.row + 1, column=1, padx=(5, 2), sticky='ew')

        self.vae_uri_entry.bind('<Key>', lambda e: self.valid())

    def invalid(self):
        self.vae_uri_entry.config(
            highlightbackground="red",
            highlightcolor="red",
            highlightthickness=2)

    def is_empty(self):
        return self.vae_uri_entry.get().strip() == ''

    def valid(self):
        self.vae_uri_entry.config(highlightthickness=0)

    def template(self, content):
        vae_uri = self.vae_uri_var.get().strip()
        vae_type = self.vae_type_var.get().strip()

        value = ''
        if vae_uri and vae_type:
            value = f'{vae_type};model={shlex.quote(vae_uri)}'

        return self._template(content, value)


class _RecipesForm(tk.Toplevel):
    def __init__(self, master=None, position: tuple[int, int] = None, width: int = None):
        super().__init__(master)
        self.title('Insert Recipe')

        self._entry_classes = {
            'file': _FileEntry,
            'dir': _DirectoryEntry,
            'karrasscheduler': _KarrasSchedulerEntry,
            'torchvae': _TorchVaeEntry,
            'string': _StringEntry,
            'int': _IntEntry,
            'float': _FloatEntry,
            'device': _DeviceEntry
        }

        self.minsize(600, 0)

        self._templates = None
        self._dropdown = None
        self._templates_dict: dict[str, str] = _recipes.RECIPES
        self._template_names: list[str] = list(_recipes.RECIPES.keys())

        self._current_template = tk.StringVar(value=self._template_names[0])
        self._entries: typing.List[_Entry] = []
        self._content: typing.Optional[str] = None

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
        old_form_width = self.winfo_width()

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
                        master=self, row=row_offset,
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
