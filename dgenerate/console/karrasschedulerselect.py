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

import dgenerate.console.recipesformentries as _entries
import dgenerate.console.pluginuriselector as _pluginuriselector


class _KarrasSchedulerSelect(_pluginuriselector._PluginUriSelect):
    def __init__(self, master=None, position: tuple[int, int] = None):
        super().__init__(
            title='Insert Karras Scheduler URI',
            plugin_entry_class=_entries._KarrasSchedulerEntry,
            master=master,
            position=position
        )


_last_pos = None


def request_uri(master):
    global _last_pos

    window = _KarrasSchedulerSelect(master, position=_last_pos)

    og_destroy = window.destroy

    def destroy():
        global _last_pos
        _last_pos = _last_size = (window.winfo_x(), window.winfo_y())
        og_destroy()

    window.destroy = destroy

    def on_closing():
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    return window.get_uri()
