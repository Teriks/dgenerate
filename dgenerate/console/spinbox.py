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
import sys
import tkinter as tk
import dgenerate.console.textentry as _t_entry

from dgenerate.console.mousewheelbind import bind_mousewheel, un_bind_mousewheel


class IntSpinbox(_t_entry.TextEntry):
    """
    Spin box for integer values.
    """

    def __init__(self,
                 master=None,
                 textvariable: tk.StringVar = None,
                 from_: int | None = None,
                 to: int | None = None,
                 increment: int = 1,
                 **kwargs):

        from_ = from_ if from_ is not None else -sys.maxsize
        to = to if to is not None else sys.maxsize

        if not isinstance(from_, int):
            raise ValueError('from must be int.')

        if not isinstance(to, int):
            raise ValueError('to must be int.')

        self.display_value = textvariable if textvariable is not None else tk.StringVar()

        self.increment = increment

        self.from_ = from_
        self.to = to

        super().__init__(master, textvariable=self.display_value, **kwargs)
        self._setup_validation()

        bind_mousewheel(self.bind, self._on_mouse_wheel)

        self.bind('<Up>', self._on_up_arrow)
        self.bind('<Down>', self._on_down_arrow)

    def create_spin_buttons(self, master) -> tk.Frame:
        button_frame = tk.Frame(master)
        inc = tk.Button(button_frame, text='+', command=lambda: self.do_increment(1))
        inc.pack(side=tk.LEFT)
        dec = tk.Button(button_frame, text='-', command=lambda: self.do_increment(-1))
        dec.pack(side=tk.RIGHT)
        return button_frame

    def do_increment(self, delta):
        if self.cget('state') == tk.DISABLED:
            return

        try:
            new_value = max(self.from_,
                            min(self.to, int(self.display_value.get()) + (delta * self.increment)))
            self.display_value.set(str(new_value))
        except ValueError:
            self.display_value.set(self.get_midpoint())

    def _on_up_arrow(self, e):
        self.do_increment(1)
        return "break"

    def _on_down_arrow(self, e):
        self.do_increment(-1)
        return "break"

    def _on_mouse_wheel(self, e):
        delta = -1 if e.delta < 0 else 1
        self.do_increment(delta)
        return "break"

    def _setup_validation(self):
        self.config(validate='key', validatecommand=(self.register(self._validate), '%P'))

    def get_midpoint(self) -> str:
        """
        Calculate the midpoint of the range defined by `from_` and `to`.

        :return: Midpoint as a string.
        """
        if self.to == sys.maxsize and self.from_ != -sys.maxsize:
            return str(self.from_)

        if self.from_ == -sys.maxsize and self.to != sys.maxsize:
            return str(self.to)

        return str(round((self.from_ + self.to) / 2))

    def _validate(self, value_if_allowed):

        if value_if_allowed == "":  # Allow empty string for deletion
            return True
        try:
            value = int(value_if_allowed)  # Validate as float

            if value < self.from_ or value > self.to:
                self.delete(0, tk.END)
                self.insert(0, str(max(self.from_,
                                       min(self.to, value))))
                # Validation gets blown out by the edits for
                # god knows what reason, this took ages to figure out
                self._setup_validation()
                return False

            return True
        except ValueError:
            return False

    def is_valid(self):
        display_value = self.display_value.get()
        if display_value == '':
            return True

        try:
            value = int(display_value)  # Validate as int

            if value < self.from_ or value > self.to:
                return False

            return True
        except ValueError:
            return False

    def destroy(self) -> None:
        un_bind_mousewheel(self.unbind)
        super().destroy()


class FloatSpinbox(_t_entry.TextEntry):
    """
    Spinbox for floats where the accuracy of the value retrieved is important.
    """

    def __init__(self,
                 master=None,
                 textvariable: tk.StringVar = None,
                 from_: float | None = None,
                 to: float | None = None,
                 increment: float = 0.01,
                 **kwargs):

        if from_ is None:
            self.from_ = float('-inf')
        else:
            self.from_ = float(from_)

        if to is None:
            self.to = float('inf')
        else:
            self.to = float(to)

        self.real_value = textvariable if textvariable else tk.StringVar()
        self.display_value = tk.StringVar(value=self._format_real_to_display_value())

        # Trace real_value changes to update display_value
        self._real_trace_id = self.real_value.trace_add('write', self._update_display_value)

        self.increment = increment

        super().__init__(master,
                         textvariable=self.display_value,
                         **kwargs)

        self._setup_validation()

        self._display_trace_id = self.display_value.trace_add('write', self._update_real_value)

        bind_mousewheel(self.bind, self._on_mouse_wheel)

        self.bind('<Up>', self._on_up_arrow)
        self.bind('<Down>', self._on_down_arrow)

    def create_spin_buttons(self, master) -> tk.Frame:
        button_frame = tk.Frame(master)
        inc = tk.Button(button_frame, text='+', command=lambda: self.do_increment(1))
        inc.pack(side=tk.LEFT)
        dec = tk.Button(button_frame, text='-', command=lambda: self.do_increment(-1))
        dec.pack(side=tk.RIGHT)
        return button_frame

    def do_increment(self, delta):
        if self.cget('state') == tk.DISABLED:
            return

        if 'inf' in self.display_value.get():
            self.display_value.set('0')
            return

        try:
            new_value = max(self.from_,
                            min(self.to, round(float(self.display_value.get()) + (delta * self.increment), 2)))
            self.display_value.set(str(new_value))
        except ValueError:
            self.display_value.set(self.get_midpoint())

    def _on_up_arrow(self, e):
        self.do_increment(1)
        return "break"

    def _on_down_arrow(self, e):
        self.do_increment(-1)
        return "break"

    def _on_mouse_wheel(self, e):
        delta = -1 if e.delta < 0 else 1
        self.do_increment(delta)
        return "break"

    def _setup_validation(self):
        self.config(validate='all', validatecommand=(self.register(self._validate), '%d', '%P'))

    def _format_real_to_display_value(self):
        """
        Format the display value based on the real_value.
        """
        if self.real_value.get() == '':
            return ''
        return str(float(self.real_value.get()))

    def get_midpoint(self) -> str:
        """
        Calculate the midpoint of the range defined by `from_` and `to`.

        :return: Midpoint as a string.
        """
        if self.to == float('inf') and self.from_ != float('-inf'):
            return str(self.from_)

        if self.from_ == float('-inf') and self.to != float('inf'):
            return str(self.to)

        if self.to == float('inf') and self.from_ == float('-inf'):
            return '0'  # Midpoint of infinite range can be arbitrarily chosen as 0

        return str((self.from_ + self.to) / 2)

    def _validate(self, action, value_if_allowed):
        if value_if_allowed == "":  # Allow empty string for deletion
            return True

        if action != '-1':
            # inserting or deleting
            try:
                is_positive_min = self.from_ >= 0

                pattern = r'^i(nf?)?$|^[0-9.]+$' if is_positive_min else r'^-?i(nf?)?$|^[0-9.-]+$'

                match = re.match(pattern, value_if_allowed)

                return (
                        match is not None
                        and value_if_allowed.count('.') < 2  # At most one decimal point
                        and value_if_allowed.count('-') < 2  # At most one '-' sign
                        and (is_positive_min or value_if_allowed.find('-') in [0, -1])  # '-' at valid position
                )

            except ValueError:
                return False
        else:
            # gain or lose focus
            try:
                value = float(value_if_allowed)  # Validate as float

                if value < self.from_ or value > self.to:
                    self.delete(0, tk.END)
                    self.insert(0, str(max(self.from_,
                                       min(self.to, round(value, 2)))))
                    # Validation gets blown out by the edits for
                    # god knows what reason, this took ages to figure out
                    self._setup_validation()
                    return False

                return True
            except ValueError:
                self.delete(0, tk.END)
                self.insert(0, self.get_midpoint())
                # Validation gets blown out by the edits for
                # god knows what reason, this took ages to figure out
                self._setup_validation()
                return False

    def is_valid(self):
        display_value = self.display_value.get()
        if display_value == '':
            return True

        try:
            value = float(display_value)  # Validate as float

            if value < self.from_ or value > self.to:
                return False

            return True
        except ValueError:
            return False

    def _update_display_value(self, *args):
        """
        Update the display value based on the real value.
        """
        self.display_value.set(self._format_real_to_display_value())

    def _update_real_value(self, *args):
        """
        Update the real value based on the display value.
        """
        self.real_value.trace_remove('write', self._real_trace_id)
        self.real_value.set(self.display_value.get())
        self._real_trace_id = self.real_value.trace_add('write', self._update_display_value)

    def destroy(self) -> None:
        un_bind_mousewheel(self.unbind)
        super().destroy()
