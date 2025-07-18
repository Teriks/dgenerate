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

import dgenerate.console.resources as _resources
import dgenerate.console.recipesformentries.pluginschemaentry as _schemaentry
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

        self._file_in_arguments = {
            'model': _resources.get_file_dialog_args(['models']),
            'mask': _resources.get_file_dialog_args(['images-in']),
            'image': _resources.get_file_dialog_args(['images-in']),
            'param': {'filetypes': [('param', ['*.param'])]}}

        self._file_out_arguments = {
            'output-file': _resources.get_file_dialog_args(['images-out'])
        }

        super().__init__(*args,
                         label='Image Processor',
                         hidden_args=hidden_args,
                         help_button=True,
                         schema_help_node='PROCESSOR_HELP',
                         schema=schema, **kwargs)

    def _apply_file_selects(self, param_name: str, entry: _schemaentry._PluginArgEntry):
        if param_name in self._file_in_arguments:
            entry.file_types = self._file_in_arguments[param_name]
            entry.raw = False
        if param_name in self._file_out_arguments:
            entry.file_types = self._file_out_arguments[param_name]
            entry.file_out = True
            entry.raw = False
        return entry

    def _create_entry_single_type(self,
                                  param_name: str,
                                  param_type: str,
                                  default_value: typing.Any,
                                  optional: bool,
                                  row: int) -> _schemaentry._PluginArgEntry:

        created_simple_type, entry = self._create_int_float_bool_entries(param_type, default_value, optional, row)
        if created_simple_type:
            return entry
        elif 'device' in param_name:
            return self._create_dropdown_entry(
                _resources.get_torch_devices(),
                default_value,
                optional,
                row
            )
        elif param_name in {'image-processors', 'mask-processors'}:
            raw = self._create_raw_type_entry(param_type, default_value, optional, row)

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

                _s.request_uri(self.recipe_form, insert=lambda t: variable.set(quote_processor(t)), dialog_state=dialog_state)

            raw.insert_text_callback = ('Select', select_processor)

            return raw
        else:
            return self._apply_file_selects(
                param_name, self._create_raw_type_entry(param_type, default_value, optional, row))
