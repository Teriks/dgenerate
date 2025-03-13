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
import typing

import dgenerate.console.dropdownselectwithhelp as _modalselectwithhelp
import dgenerate.console.resources as _resources

_last_pos = None
_cur_window = None


def request_function(master, insert: typing.Callable[[str], None]):
    global _last_pos, _cur_window

    if _cur_window is not None:
        _cur_window.focus_set()
        return

    _cur_window = _modalselectwithhelp._DropdownSelectWithHelp(
        item_name='Function',
        values_to_help=_resources.get_dgenerate_functions(),
        insert=insert,
        master=master,
        position=_last_pos
    )

    og_destroy = _cur_window.destroy

    # noinspection PyUnresolvedReferences
    def destroy():
        global _last_pos, _cur_window
        _last_pos = _last_size = (_cur_window.winfo_x(), _cur_window.winfo_y())
        _cur_window = None
        og_destroy()

    _cur_window.destroy = destroy

    def on_closing():
        _cur_window.destroy()

    _cur_window.protocol("WM_DELETE_WINDOW", on_closing)

