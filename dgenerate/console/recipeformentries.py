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
import shlex
import sys
import tkinter as tk
import typing

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.resources as _resources
from dgenerate.console.spinbox import IntSpinbox, RealSpinbox


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


_ROW_PAD = (5, 2)


class _Entry:
    NAME = None

    def __init__(self, recipe_form, master, row: int, config: dict[str, typing.Any], placeholder: str,
                 widget_rows: int):
        self.recipe_form = recipe_form
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
    NAME = 'string'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.text_var = tk.StringVar(
            value=self.config.get('default', ''))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('String'), anchor='e')

        self.entry = tk.Entry(self.master, textvariable=self.text_var)

        self.label_widget.grid(row=self.row, column=0, padx=_ROW_PAD, sticky='e')
        self.entry.grid(row=self.row, column=1, padx=_ROW_PAD, sticky='ew')
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


class _ArgSwitchCheckbox(_Entry):
    NAME = 'switch'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        # initialize BooleanVar with a default value from config
        self.bool_var = tk.BooleanVar(value=self.config.get('default', False))

        # create label widget
        self.label_widget = tk.Label(self.master,
                                     text=self.config.get(
                                         'label',
                                         self.arg if self.arg else 'Switch Option'), anchor='e')

        # create checkbox widget with boolean variable
        self.checkbox = tk.Checkbutton(self.master, variable=self.bool_var)

        # place the widgets in a grid layout
        self.label_widget.grid(row=self.row, column=0, padx=_ROW_PAD, sticky='e')
        self.checkbox.grid(row=self.row, column=1, sticky='w')

    def _is_checked(self):
        return self.bool_var.get()

    def template(self, content):
        return _replace_first(content, self.placeholder, self.arg if self._is_checked() else '')


class _DeviceEntry(_Entry):
    NAME = 'device'

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

        self.label_widget.grid(row=self.row, column=0, padx=_ROW_PAD, sticky='e')
        self.entry.grid(row=self.row, column=1, padx=_ROW_PAD, sticky='ew')

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


class _DropDownEntry(_Entry):
    NAME = 'dropdown'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        if self.arg is None:
            self.arg = ''

        options = [o.strip() for o in self.config.get('options', '').split(',')]

        if self.optional:
            options = [''] + options

        self.text_var = tk.StringVar(
            value=self.config.get('default', options[0]))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('Prompt Weighter'), anchor='e')

        self.entry = tk.OptionMenu(self.master,
                                   self.text_var,
                                   *options)

        self.label_widget.grid(row=self.row, column=0, padx=_ROW_PAD, sticky='e')
        self.entry.grid(row=self.row, column=1, padx=_ROW_PAD, sticky='ew')

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
    NAME = 'int'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.min = self.config.get('min', -sys.maxsize)
        self.max = self.config.get('max', sys.maxsize)

        self.text_var = tk.StringVar(
            value=self.config.get('default', ''))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('Int'), anchor='e')

        def increment(delta):
            value = int(self.text_var.get())
            value = max(self.min, min(self.max, value + delta))
            self.text_var.set(value)

        def on_mouse_wheel(event):
            delta = -1 if event.delta < 0 else 1
            increment(delta)
            return "break"

        self.entry = IntSpinbox(self.master,
                                from_=self.min,
                                to=self.max,
                                textvariable=self.text_var)

        self.entry.bind("<MouseWheel>", on_mouse_wheel)
        self.entry.bind("<Button-4>", on_mouse_wheel)  # For Linux systems
        self.entry.bind("<Button-5>", on_mouse_wheel)  # For Linux systems

        self.label_widget.grid(row=self.row, column=0, padx=_ROW_PAD, sticky='e')
        self.entry.grid(row=self.row, column=1, padx=_ROW_PAD, sticky='ew')
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
    NAME = 'float'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.min = self.config.get('min', float('-inf'))
        self.max = self.config.get('max', float('inf'))

        self.text_var = tk.StringVar(
            value=self.config.get('default', ''))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('Float'), anchor='e')

        def increment(delta):
            cur_value = self.text_var.get().strip()
            value = float(cur_value if cur_value != '' else 0.0)
            value = max(self.min, min(self.max, round(value + (delta * 0.01), 2)))
            self.text_var.set(value)

        def on_mouse_wheel(event):
            delta = -1 if event.delta < 0 else 1
            increment(delta)
            return "break"

        self.entry = tk.Spinbox(self.master,
                                from_=self.min,
                                to=self.max,
                                format="%.2f", increment=0.01,
                                textvariable=self.text_var)

        self.entry.config(validate='all', validatecommand=(self.entry.register(self._validate), '%P'))

        self.entry.bind("<MouseWheel>", on_mouse_wheel)
        self.entry.bind("<Button-4>", on_mouse_wheel)  # For Linux systems
        self.entry.bind("<Button-5>", on_mouse_wheel)  # For Linux systems

        self.label_widget.grid(row=self.row, column=0, padx=_ROW_PAD, sticky='e')
        self.entry.grid(row=self.row, column=1, padx=_ROW_PAD, sticky='ew')
        self.entry.bind('<Key>', lambda e: self.valid())

    def _validate(self, value_if_allowed):
        try:
            # Validate if the input is a valid float string
            float(value_if_allowed)
            return True
        except ValueError:
            # Allow for empty if optional
            if self.optional:
                return value_if_allowed.strip() == ''
            return False

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
    NAME = 'file'

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
            file_globs = ['*.' + ext for ext in _resources.supported_torch_model_formats_open()]
            dialog_args['filetypes'] = [('Models', ' '.join(file_globs))]

        def select_command():
            if 'output' in self.select_mode:
                r = _filedialog.open_file_save_dialog(**dialog_args)
            else:
                r = _filedialog.open_file_dialog(**dialog_args)
            if r is not None:
                self.text_var.set(r)

        if 'output' in self.select_mode:
            self.button = tk.Button(self.master, text="Save File",
                                    command=select_command)
        else:
            self.button = tk.Button(self.master, text="Select File",
                                    command=select_command)

        self.label_widget.grid(row=self.row, column=0, padx=_ROW_PAD, sticky='e')
        self.entry.grid(row=self.row, column=1, padx=_ROW_PAD, sticky='ew')
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
    NAME = 'dir'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.text_var = tk.StringVar(value=self.config.get('default', ''))

        self.label_widget = tk.Label(self.master,
                                     text=self.get_label('Directory'), anchor='e')

        self.entry = tk.Entry(self.master, textvariable=self.text_var)
        self.select_mode = self.config.get('mode', 'input')

        def select_command():
            r = _filedialog.open_directory_dialog()
            if r is not None:
                self.text_var.set(r)

        self.button = tk.Button(self.master,
                                text="Select Directory", command=select_command)

        self.label_widget.grid(row=self.row, column=0, padx=_ROW_PAD, sticky='e')
        self.entry.grid(row=self.row, column=1, padx=_ROW_PAD, sticky='ew')
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
    NAME = 'karrasscheduler'

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

        self.scheduler_name_label.grid(row=self.row, column=0, padx=_ROW_PAD, sticky='e')
        self.scheduler_name_entry.grid(row=self.row, column=1, padx=_ROW_PAD, sticky='ew')

        self.prediction_type_label.grid(row=self.row + 1, column=0, padx=_ROW_PAD, sticky='e')
        self.prediction_type_entry.grid(row=self.row + 1, column=1, padx=_ROW_PAD, sticky='ew')

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
    NAME = 'torchvae'

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
            file_globs = ['*.' + ext for ext in _resources.supported_torch_model_formats_open()]
            r = _filedialog.open_file_dialog(
                filetypes=[('Models', ' *.'.join(file_globs))])
            if r is not None:
                self.vae_uri_var.set(r)

        self.button = tk.Button(self.master, text="Select File",
                                command=select_command)

        self.vae_uri_label.grid(row=self.row, column=0, padx=_ROW_PAD, sticky='e')
        self.vae_uri_entry.grid(row=self.row, column=1, padx=_ROW_PAD, sticky='ew')
        self.button.grid(row=self.row, column=2, padx=(2, 5), sticky='w')

        self.vae_type_label.grid(row=self.row + 1, column=0, padx=_ROW_PAD, sticky='e')
        self.vae_type_entry.grid(row=self.row + 1, column=1, padx=_ROW_PAD, sticky='ew')

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


class _ImageProcessor(_Entry):
    NAME = 'imageprocessor'

    def __init__(self, *args, **kwargs):
        self.schema = _resources.get_schema('imageprocessors')
        max_rows = max(len(args) for args in self.schema.values()) + 1
        super().__init__(*args, **kwargs, widget_rows=max_rows)

        self.declared_optional = self.config.get('optional', True)
        label = f'(Optional) {self.config.get("label", "Image Processor")}' if self.declared_optional else self.config.get(
            "label", "Image Processor")

        self.optional = False
        self.processor_name_var = tk.StringVar(self.master)
        values = list(self.schema.keys())

        self.dropdown_label = tk.Label(self.master, text=label)

        if self.declared_optional:
            self.algorithm_dropdown = tk.OptionMenu(self.master, self.processor_name_var, '', *values,
                                                    command=self._on_processor_change)
        else:
            self.processor_name_var.set(values[0])
            self.algorithm_dropdown = tk.OptionMenu(self.master, self.processor_name_var, *values,
                                                    command=self._on_processor_change)

        self.processor_help_button = tk.Button(self.master, text='Help', command=self._show_help)
        self.dropdown_label.grid(row=self.row, column=0, padx=_ROW_PAD, sticky="e")
        self.algorithm_dropdown.grid(row=self.row, column=1, padx=_ROW_PAD, sticky="ew")

        if not self.declared_optional:
            self._show_help_button()

        self.entries = {}
        self.dynamic_widgets = []
        self.file_arguments = {
            'model': [('Models', ' *.'.join(['*.' + ext for ext in _resources.supported_torch_model_formats_open()]))]}

    def _show_help(self):
        top = tk.Toplevel(self.master)
        top.title(f'Image Processor Help: {self.processor_name_var.get()}')
        top.attributes('-topmost', 1)
        top.attributes('-topmost', 0)

        frame = tk.Frame(top)
        frame.pack(expand=True, fill='both')

        v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        text_widget = tk.Text(frame, wrap=tk.NONE, state='disabled', yscrollcommand=v_scrollbar.set,
                              xscrollcommand=h_scrollbar.set)

        if self.recipe_form.master and hasattr(self.recipe_form.master, '_input_text'):
            bg = self.recipe_form.master._input_text.text.cget('bg')
            fg = self.recipe_form.master._input_text.text.cget('fg')
            text_widget.configure(bg=bg, fg=fg)

        text_widget.config(state='normal')
        text_widget.insert(tk.END, self.current_help_text)
        text_widget.config(state='disabled')

        v_scrollbar.config(command=text_widget.yview)
        h_scrollbar.config(command=text_widget.xview)
        text_widget.pack(expand=True, fill='both')

        top.geometry("800x600")

    def _show_help_button(self):
        self.processor_help_button.grid(row=self.row, column=2, padx=_ROW_PAD, sticky="w")

    def _hide_help_button(self):
        self.processor_help_button.grid_forget()

    def _on_processor_change(self, event):
        for widget in self.dynamic_widgets:
            widget.destroy()

        self.dynamic_widgets.clear()
        self.entries.clear()
        algorithm_name = self.processor_name_var.get()

        if algorithm_name == '':
            self._hide_help_button()
            return

        self._show_help_button()
        parameters = self.schema[algorithm_name]

        for i, (param_name, param_info) in enumerate(parameters.items()):
            if param_name == 'PROCESSOR_HELP':
                self.current_help_text = param_info
                continue

            row = self.row + i + 1

            optional_label_part = '(Optional) ' if param_info.get('optional', False) else ''
            label_text = f"{optional_label_part}{param_name} ({', '.join(param_info['types'])})"

            default_value = param_info.get('default', "")
            param_types = param_info['types']
            optional = param_info.get('optional', False)

            sticky_entry = 'we'
            padx_entry = _ROW_PAD

            if len(param_types) == 1:
                param_type = param_types[0]
                if param_type in ['int', 'float'] and default_value != "":

                    variable = tk.StringVar(value=default_value)

                    increment = 1 if param_type == 'int' else 0.01
                    typ = int if param_type == 'int' else float

                    def on_mouse_wheel(e, var=variable, typ=typ, inc=increment):
                        delta = -1 if e.delta < 0 else 1
                        new_value = round(typ(var.get()) + (delta * inc), 2)
                        var.set(str(new_value))
                        return "break"

                    min_spin = -sys.maxsize if param_type == 'int' else float('-inf')
                    max_spin = sys.maxsize if param_type == 'int' else float('inf')

                    sp_typ = IntSpinbox if param_type == 'int' else RealSpinbox

                    entry = sp_typ(master=self.master,
                                   from_=min_spin,
                                   to=max_spin,
                                   textvariable=variable,
                                   increment=increment,
                                   format='%.15f' if param_type == 'float' else None)

                    entry.optional = optional

                    entry.bind("<MouseWheel>", on_mouse_wheel)
                    entry.bind("<Button-4>", on_mouse_wheel)  # For Linux systems
                    entry.bind("<Button-5>", on_mouse_wheel)  # For Linux systems

                    label_text = f"{optional_label_part}{param_name}"
                elif param_type == 'bool' and default_value != "" and not optional:
                    variable = tk.BooleanVar(value=default_value)
                    entry = tk.Checkbutton(self.master,
                                           variable=variable)
                    label_text = f"{optional_label_part}{param_name}"
                    sticky_entry = 'w'
                    padx_entry = None
                else:
                    variable = tk.StringVar(value=default_value)
                    entry = tk.Entry(self.master, textvariable=variable)
            else:
                variable = tk.StringVar(value=default_value)
                entry = tk.Entry(self.master, textvariable=variable)

            label = tk.Label(self.master, text=label_text)
            self.dynamic_widgets.append(label)
            label.grid(row=row, column=0, padx=_ROW_PAD, sticky="e")

            self.dynamic_widgets.append(entry)

            entry.grid(row=row, column=1, padx=padx_entry, sticky=sticky_entry)

            if param_name in self.file_arguments:
                file_types = self.file_arguments[param_name]
                file_select = tk.Button(self.master, text='Select File',
                                        command=lambda e=entry, f=file_types: self._select_model_command(e, f))
                file_select.grid(row=row, column=2, padx=_ROW_PAD, sticky='w')
                self.dynamic_widgets.append(file_select)

            self.entries[param_name] = (entry, variable, default_value, optional)

    def invalid(self):
        for entry, variable, default_value, optional in self.entries.values():
            if optional is False and not str(variable.get()):
                entry.config(
                    highlightbackground="red",
                    highlightcolor="red",
                    highlightthickness=2)

    def is_empty(self):
        for entry, variable, default_value, optional in self.entries.values():
            if optional is False and not str(variable.get()):
                return True
        return False

    def valid(self):
        for entry, variable, default_value, optional in self.entries.values():
            if optional:
                entry.config(highlightthickness=0)

    @staticmethod
    def _select_model_command(entry, file_types):
        file_path = _filedialog.open_file_dialog(filetypes=file_types)
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    @staticmethod
    def _normalize_value(value):
        value = str(value).lower()
        return {'none': 'None', 'true': 'True', 'false': 'False'}.get(value, value)

    def _format_uri(self):
        algorithm_name = self.processor_name_var.get()
        if not algorithm_name:
            return ''
        uri_parts = [algorithm_name]
        for param_name, (entry, variable, default_value, _) in self.entries.items():
            current_value = self._normalize_value(variable.get())
            if current_value != self._normalize_value(default_value) and current_value.strip():
                uri_parts.append(f"{param_name}={current_value}")
        return ';'.join(uri_parts)

    def template(self, content):
        return self._template(content, self._format_uri())
