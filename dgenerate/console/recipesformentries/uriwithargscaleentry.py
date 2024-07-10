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


import dgenerate.console.recipesformentries.entry as _entry
import dgenerate.console.recipesformentries.uriwithscaleentry as _uriwithscaleentry


class _UriWithArgScaleEntry(_uriwithscaleentry._UriWithScaleEntry):
    NAME = 'uriwithargscale'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale_arg = self.config.get('scale_arg', '')

    def template(self, content):
        uri = self.uri_var.get().strip()
        scale = self.scale_var.get().strip()

        new_content = ''

        if uri:
            new_content = self.arg if self.arg else ''
            new_content += (' ' if self.arg else '') + _entry.shell_quote_if(uri)
            if scale:
                arg_part = (self.scale_arg if self.scale_arg else '')
                new_content += '\n' + arg_part + (' ' if arg_part else '') + scale

        return _entry.replace_first(content, self.placeholder, new_content)
