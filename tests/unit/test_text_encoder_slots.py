import json
import unittest
from pathlib import Path

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.textencoderschema as _slots

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / 'dgenerate' / 'console' / 'schemas' / 'textencoders.json')


def _checkbutton_labels(widget) -> list[str]:
    labels = []
    for child in widget.winfo_children():
        if child.winfo_class() == 'TCheckbutton':
            labels.append(child.cget('text'))
        labels.extend(_checkbutton_labels(child))
    return labels


class TestTextEncoderSlots(unittest.TestCase):
    def test_defaults_are_omitted_until_a_later_slot_is_set(self):
        from dgenerate.console.formentries.textencodersentry import format_text_encoder_values

        self.assertEqual(format_text_encoder_values([None, None, None]), '')
        self.assertEqual(format_text_encoder_values(['clip', None, None]), 'clip')
        self.assertEqual(format_text_encoder_values([None, None, 't5']), '+ + t5')
        self.assertEqual(format_text_encoder_values(['clip', None, 'null']), 'clip + null')

    def test_schema_covers_model_types(self):
        schema = _slots.text_encoder_schema()
        names = set(_enums.supported_model_type_strings())
        names.add('s-cascade-decoder')
        self.assertEqual(set(schema), names)
        for model_type, slots in schema.items():
            self.assertTrue(slots, model_type)
            for slot in slots:
                self.assertTrue(slot['encoders'], model_type)
                self.assertNotIn('PreTrainedModel', slot['encoders'])

    def test_monolithic_modes_are_optional_and_slot_specific(self):
        schema = _slots.text_encoder_schema()
        self.assertNotIn('modes', schema['sd'][0])
        self.assertNotIn('modes', schema['sdxl'][0])
        self.assertNotIn('modes', schema['wan'][0])
        self.assertEqual(
            [mode['value'] for mode in schema['flux'][0]['modes']],
            ['clip-l'])
        self.assertEqual(
            [mode['value'] for mode in schema['flux'][1]['modes']],
            ['t5-xxl'])
        self.assertEqual(
            [mode['value'] for mode in schema['sd3'][0]['modes']],
            ['clip-l-sd3', 'clip-l-sd35-large'])
        self.assertEqual(
            [mode['value'] for mode in schema['sd3'][1]['modes']],
            ['clip-g-sd3', 'clip-g-sd35-large'])
        self.assertEqual(
            [mode['value'] for mode in schema['sd3'][2]['modes']],
            ['t5-xxl-sd3'])
        self.assertEqual(
            schema['flux'][1]['encoders'],
            ['T5EncoderModel', 'DistillT5EncoderModel'])
        self.assertEqual(
            schema['sd3'][2]['encoders'],
            ['T5EncoderModel', 'DistillT5EncoderModel'])
        self.assertEqual(
            schema['sd3-pix2pix'][2]['encoders'],
            ['T5EncoderModel', 'DistillT5EncoderModel'])
        self.assertEqual(
            schema['flux2'][0]['encoders'][0],
            'Mistral3ForConditionalGeneration')
        self.assertIn('Qwen3ForCausalLM', schema['flux2'][0]['encoders'])
        self.assertEqual(schema['z-image'][0]['encoders'][0], 'Qwen3Model')
        self.assertEqual(schema['if'][0]['encoders'], ['T5EncoderModel'])
        self.assertEqual(schema['kolors'][0]['encoders'], ['ChatGLMModel'])
        self.assertEqual(schema['wan-animate-2'][0]['encoders'], ['UMT5EncoderModel'])
        nullable = [
            (model_type, slot['name'])
            for model_type, slots in schema.items()
            for slot in slots
            if slot.get('nullable')
        ]
        self.assertEqual(
            nullable,
            [('sd3', 'text_encoder_3'), ('sd3-pix2pix', 'text_encoder_3')])

    def test_recipe_entry_joins_slots_and_null(self):
        import tkinter as tk

        from dgenerate.console.formentries.textencodersentry import _TextEncodersEntry

        root = tk.Tk()
        root.withdraw()
        try:
            frame = tk.Frame(root)
            entry = _TextEncodersEntry(
                form=object(),
                master=frame,
                row=0,
                config={'model-type': 'sd3', 'arg': '--text-encoders'},
                placeholder='PLACE')
            entry.bool_vars[2].set(True)
            self.assertEqual(entry.template('PLACE'), '--text-encoders + + null')
            entry.bool_vars[2].set(False)
            entry.uri_vars[0].set(
                'CLIPTextModelWithProjection;model=clip_l;mode=clip-l-sd3')
            rendered = entry.template('PLACE')
            self.assertEqual(
                rendered,
                '--text-encoders CLIPTextModelWithProjection;model=clip_l;mode=clip-l-sd3')
            self.assertTrue(entry.is_empty() is False)
            entry.uri_vars[0].set('')
            self.assertTrue(entry.is_empty())
            entry.uri_vars[0].set('clip_l.safetensors')
            self.assertEqual(
                entry.template('PLACE'),
                '--text-encoders CLIPTextModelWithProjection;model=clip_l.safetensors')
            self.assertEqual(_checkbutton_labels(frame), ['null?'])

            for child in frame.winfo_children():
                child.destroy()
            sd = _TextEncodersEntry(
                form=object(),
                master=frame,
                row=0,
                config={'model-type': 'sd', 'arg': '--text-encoders'},
                placeholder='PLACE')
            self.assertEqual(_checkbutton_labels(frame), [])
            sd.uri_vars[0].set('')
            self.assertEqual(sd.template('PLACE'), '')
        finally:
            root.destroy()

    def test_every_recipe_text_encoder_tag_builds(self):
        import tkinter as tk

        from dgenerate.console.formentries.textencodersentry import _TextEncodersEntry
        from tests.unit.test_console_acceleration_recipes import _tags

        root = tk.Tk()
        root.withdraw()
        try:
            frame = tk.Frame(root)
            recipes = Path(__file__).resolve().parents[2] / 'dgenerate' / 'console' / 'recipes'
            seen = 0
            for path in recipes.glob('*.recipe'):
                for name, config in _tags(path.read_text(encoding='utf-8')):
                    if name != 'textencoders':
                        continue
                    entry = _TextEncodersEntry(
                        form=object(),
                        master=frame,
                        row=0,
                        config=config,
                        placeholder='PLACE')
                    rendered = entry.template('PLACE')
                    defaults = [value for value in (config.get('defaults') or []) if value]
                    if defaults:
                        self.assertTrue(rendered.startswith('--text-encoders '), path.name)
                        for value in defaults:
                            self.assertIn(value, rendered, path.name)
                    else:
                        self.assertEqual(rendered, '', path.name)
                    self.assertGreaterEqual(entry.widget_rows, 1)
                    seen += 1
                    for child in frame.winfo_children():
                        child.destroy()
            self.assertGreater(seen, 100)
        finally:
            root.destroy()

    def test_checked_in_schema_matches_the_builder(self):
        on_disk = json.loads(_SCHEMA_PATH.read_text(encoding='utf-8'))
        self.assertEqual(on_disk, _slots.text_encoder_schema())


if __name__ == '__main__':
    unittest.main()
