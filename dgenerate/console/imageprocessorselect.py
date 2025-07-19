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

import dgenerate.console.pluginuriselector as _pluginuriselector
import dgenerate.console.formentries as _entries
import dgenerate.console.util as _util

_dialog_state = _util.DialogState(save_position=True, save_size=True)


class _ImageProcessorSelect(_pluginuriselector._PluginUriSelect):
    def __init__(self,
                 insert: typing.Callable[[str], None],
                 master=None, position: tuple[int, int] | None = None,
                 size: tuple[int, int] | None = None
                 ):
        super().__init__(
            title='Insert Image Processor URI',
            plugin_entry_class=_entries._ImageProcessorEntry,
            master=master,
            position=position,
            insert=insert,
            size=size
        )


def request_uri(master, insert: typing.Callable[[str], None], dialog_state: _util.DialogState | None = None):
    return _util.create_singleton_dialog(
        master=master,
        dialog_class=_ImageProcessorSelect,
        state=_dialog_state if dialog_state is None else dialog_state,
        dialog_kwargs={'insert': insert}
    )
