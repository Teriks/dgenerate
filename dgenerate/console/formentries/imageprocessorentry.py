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
import typing

import dgenerate.console.resources as _resources
import dgenerate.console.formentries.pluginschemaentry as _schemaentry
import dgenerate.console.util
import dgenerate.textprocessing as _textprocessing


class _ImageProcessorEntry(_schemaentry._PluginSchemaEntry):
    NAME = 'imageprocessor'

    def __init__(self, *args, **kwargs):

        schema = _resources.get_schema('imageprocessors')

        config = kwargs.get('config', {})

        hidden_args = set()
        if config.get('hide-device', False):
            hidden_args.add('device')

        super().__init__(*args,
                         label='Image Processor',
                         hidden_args=hidden_args,
                         help_button=True,
                         schema_help_node='PROCESSOR_HELP',
                         schema=schema, **kwargs)

    def _create_entry_single_type(self,
                                  param_name: str,
                                  param_type: str,
                                  default_value: typing.Any,
                                  optional: bool,
                                  options: list | None,
                                  row: int) -> _schemaentry._PluginArgEntry:

        if options:
            return self._create_dropdown_entry(options, default_value, optional, row)

        created_simple_type, entry = self._create_int_float_bool_entries(
            param_type, default_value, optional, row
        )

        if created_simple_type:
            return entry
        elif 'device' in param_name:
            return self._create_dropdown_entry(
                _resources.get_torch_devices(),
                default_value,
                optional,
                row
            )
        elif 'prompt-weighter' in param_name:
            return self._create_prompt_weighter_entry(row)
        elif re.match(r'.*(image|mask)-processors', param_name):
            raw = self._create_raw_type_entry(param_type, default_value, optional, options, row)

            dialog_state = dgenerate.console.util.DialogState()

            def select_processor(variable):
                import dgenerate.console.imageprocessorselect as _s
                def quote_processor(t: str) -> str:
                    # does not handle more than 1x nesting
                    # but should be sufficient for most cases
                    if ';' in t:
                        if "'" in t:
                            return _textprocessing.quote(t, '"')
                        elif '"' in t:
                            return _textprocessing.quote(t, "'")
                        else:
                            return _textprocessing.quote(t, '"')
                    return t

                window = _s.request_uri(
                    self.form,
                    insert=lambda t: variable.set(quote_processor(t)),
                    dialog_state=dialog_state
                )

                og_window_destroy = window.destroy
                def window_destroy():
                    og_window_destroy()
                    if hasattr(self.form, 'bind_mousewheel'):
                        self.form.bind_mousewheel()
                window.destroy = window_destroy

            raw.insert_text_callback = ('Select', select_processor)

            return raw
        else:
            return self._create_raw_type_entry(
                    param_type, default_value, optional, options, row
                )
