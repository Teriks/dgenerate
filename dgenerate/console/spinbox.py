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


class IntSpinbox(tk.Spinbox):
    """
    Spin box for integer values.
    """

    def __init__(self, master=None, **kwargs):
        # Ensure the from_ and to values are integers
        if 'from_' in kwargs:
            kwargs['from_'] = int(kwargs['from_'])
        if 'to' in kwargs:
            kwargs['to'] = int(kwargs['to'])
        if 'increment' in kwargs:
            kwargs['increment'] = int(kwargs['increment'])

        super().__init__(master, **kwargs)

        self.config(validate='all', validatecommand=(self.register(self._validate), '%P'))

        self.optional = False

    def _validate(self, value_if_allowed):
        try:
            # Validate if the input is a valid float string
            int(value_if_allowed)
            return True
        except ValueError:
            # Allow for empty if optional
            if self.optional:
                return value_if_allowed.strip() == ''
            return False


class RealSpinbox(tk.Spinbox):
    """
    Spinbox for doubles where the accuracy of the value retrieved is important.

    This value cannot be optional.
    """

    def __init__(self, master=None, textvariable: tk.DoubleVar | None = None, **kwargs):

        if 'from_' in kwargs:
            initial_value = float(kwargs['from_'])
        else:
            initial_value = 0.0

        self.real_value = textvariable if textvariable is not None else tk.DoubleVar(value=initial_value)
        self.display_value = tk.StringVar()

        self.display_value.set(self._format_display_value())

        self.real_value.trace_add('write', lambda *a: self.display_value.set(self._format_display_value()))

        self.increment = kwargs.get('increment', 1.0)

        super().__init__(master, textvariable=self.display_value, **kwargs)

        self.config(validate='all', validatecommand=(self.register(self._validate), '%P'))

        self.display_value.trace_add('write', lambda *a: self.real_value.set(
            float(self.display_value.get() if self.display_value.get().strip() else '0')))

    def _format_display_value(self):
        return repr(self.real_value.get())

    def _validate(self, value_if_allowed):
        try:
            float(value_if_allowed)
            return True
        except ValueError:
            return False
