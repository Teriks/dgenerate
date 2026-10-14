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

import dgenerate.console.formentries.entry as _entry
from dgenerate.console.formentries.argswitchcheckbox import _ArgSwitchCheckbox
from dgenerate.console.formentries.fileentry import _FileEntry


class _ArgSwitchFile(_ArgSwitchCheckbox):
    NAME = 'switchfile'

    _FILE_PLACEHOLDER = '__DGENERATE_SWITCH_FILE__'

    def __init__(self, *args, **kwargs):
        kwargs['widget_rows'] = 2
        super().__init__(*args, **kwargs)

        file_config = self.config.get('file')
        if not isinstance(file_config, dict):
            raise RuntimeError('Recipe used switchfile without a dictionary file configuration.')

        self.file_entry = _FileEntry(
            form=self.form,
            master=self.master,
            row=self.row + 1,
            config=file_config,
            placeholder=self._FILE_PLACEHOLDER
        )

        self.file_implies_switch = self.config.get('file-implies-switch', False)
        if not isinstance(self.file_implies_switch, bool):
            raise RuntimeError('Recipe used file-implies-switch = str instead of bool.')

        # The after-divider belongs to the combined entry in the form layout.
        if 'divider-after' in file_config:
            self.config.setdefault('divider-after', file_config['divider-after'])

        self.bool_var.trace_add('write', self._update_file_enabled)
        self.file_entry.set_enabled(self._is_checked())

    def _update_file_enabled(self, *_args):
        enabled = self._is_checked()
        self.file_entry.set_enabled(enabled)
        if not enabled:
            self.file_entry.valid()

    def is_valid(self):
        if not self._is_checked():
            return True
        if self.file_entry.is_empty() and not self.file_entry.optional:
            return False
        return self.file_entry.is_valid()

    def invalid(self):
        if self._is_checked():
            self.file_entry.invalid()

    def valid(self):
        self.file_entry.valid()

    def template(self, content):
        if not self._is_checked():
            return _entry.replace_first(content, self.placeholder, '')

        file_argument = None
        if not self.file_entry.is_empty():
            file_argument = self.file_entry.template(self._FILE_PLACEHOLDER)

        arguments = []
        if self.arg and not (self.file_implies_switch and file_argument):
            arguments.append(self.arg)
        if file_argument:
            arguments.append(file_argument)

        return _entry.replace_first(content, self.placeholder, '\n'.join(arguments))
