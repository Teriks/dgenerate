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

import tkinter.ttk as ttk
import dgenerate.console.mousewheelbind as _mousewheelbind


class ComboBox(ttk.Combobox):
    """
    A custom Combobox that prevents mousewheel scrolling when the dropdown is not open.
    """

    def __init__(self, *args, **kwargs):

        if 'state' not in kwargs:
            kwargs['state'] = 'readonly'

        super().__init__(*args, **kwargs)
        self._is_dropdown_open = False

        # Bind events to track dropdown state
        self.bind('<<ComboboxDropdown>>', self._on_dropdown_open)
        self.bind('<<ComboboxClosed>>', self._on_dropdown_close)

        # Bind mousewheel events
        _mousewheelbind.bind_mousewheel(self.bind, self._on_mousewheel)

    def _on_dropdown_open(self, event):
        """Called when the dropdown list opens."""
        self._is_dropdown_open = True

    def _on_dropdown_close(self, event):
        """Called when the dropdown list closes."""
        self._is_dropdown_open = False

    def _on_mousewheel(self, event):
        """Handle mousewheel events."""
        if not self._is_dropdown_open:
            return "break"  # Prevent scrolling when dropdown is closed
