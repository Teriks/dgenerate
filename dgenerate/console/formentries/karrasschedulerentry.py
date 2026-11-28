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

import dgenerate.console.formentries.entry as _entry
import dgenerate.console.formentries.pluginschemaentry as _schemaentry
import dgenerate.console.resources as _resources
import dgenerate.console.sigmaschedule as _schedule


class _KarrasSchedulerEntry(_schemaentry._PluginSchemaEntry):
    NAME = 'karrasscheduler'

    def __init__(self, *args, **kwargs):
        schema = _resources.get_schema('karrasschedulers')
        # A recipe filter is the dropdown. The model-type list applies only
        # when the recipe does not name one, so a FlowMatch-only video recipe
        # is not replaced by the wider Karras set.
        config = kwargs.get('config') or {}
        if not config.get('filter'):
            allowed = _schedule.compatible_scheduler_names(
                _entry.recipe_model_type(kwargs.get('form')))
            if allowed is not None:
                allowed = set(allowed)
                schema = {name: value for name, value in schema.items() if name in allowed}

        super().__init__(
            *args,
            label='Karras Scheduler',
            help_button=False,
            schema=schema,
            **kwargs)

        if self.arg is None:
            self.arg = '--scheduler'
        self._name_required = False

    def _choice_names(self) -> list[str]:
        if self.plugin_filter:
            return [name for name in self.schema.keys() if name in self.plugin_filter]
        return list(self.schema.keys())

    def _blank_allowed(self) -> bool:
        return bool(self.declared_optional) and not self._name_required

    def set_name_required(self, required: bool) -> None:
        """Drop the blank choice while a sigma list cannot use the model default."""
        required = bool(required) and bool(self.declared_optional)
        if required == self._name_required:
            return
        self._name_required = required
        if self.dropdown_label is not None:
            label = self.config.get('label', self._label)
            self.dropdown_label.configure(
                text=f'(Optional) {label}' if self._blank_allowed() else label)
        names = self._choice_names()
        if self._blank_allowed():
            self.plugin_dropdown['values'] = ('',) + tuple(names)
        else:
            self.plugin_dropdown['values'] = tuple(names)
        enabled_state = 'disabled' if len(self.plugin_dropdown['values']) < 2 else 'readonly'
        if self._row_inactive:
            self._saved_widget_states[str(self.plugin_dropdown)] = enabled_state
        else:
            self.plugin_dropdown.configure(state=enabled_state)

    def is_empty(self):
        # The form flags an empty plugin field as missing. A blank name is
        # missing only while a sigma list needs a named scheduler.
        if not str(self.plugin_name_var.get()).strip():
            return self._name_required
        return super().is_empty()

    def is_active(self) -> bool:
        return bool(str(self.plugin_name_var.get()).strip())
