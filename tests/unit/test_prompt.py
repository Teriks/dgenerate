import unittest

from dgenerate.prompt import Prompt
from dgenerate.textprocessing import shell_parse


class TestPromptDelimiterEscape(unittest.TestCase):

    def test_unescaped_semicolon_splits(self):
        prompt = Prompt.parse('a sign that says hello; blurry')
        self.assertEqual(prompt.positive, 'a sign that says hello')
        self.assertEqual(prompt.negative, 'blurry')

    def test_escaped_semicolon_stays_in_the_text(self):
        prompt = Prompt.parse(r'a sign that says hello\; world; blurry')
        self.assertEqual(prompt.positive, 'a sign that says hello; world')
        self.assertEqual(prompt.negative, 'blurry')

    def test_escaped_semicolon_in_the_negative_is_literal(self):
        prompt = Prompt.parse(r'horse; wearing a saddle\; rain')
        self.assertEqual(prompt.positive, 'horse')
        self.assertEqual(prompt.negative, 'wearing a saddle; rain')

    def test_a_backslash_before_another_character_stays(self):
        prompt = Prompt.parse(r'a \path; blurry')
        self.assertEqual(prompt.positive, r'a \path')
        self.assertEqual(prompt.negative, 'blurry')

    def test_round_trip_escapes_the_semicolon(self):
        prompt = Prompt.parse(r'hello\; world; blurry')
        self.assertEqual(str(prompt), r'hello\; world; blurry')
        self.assertEqual(Prompt.parse(str(prompt)).positive, 'hello; world')
        self.assertEqual(Prompt.parse(str(prompt)).negative, 'blurry')

    def test_config_shell_keeps_the_escape_for_prompt_parse(self):
        args = shell_parse(r'--prompts "a sign that says hello\; world; blurry"')
        self.assertEqual(args[1], r'a sign that says hello\; world; blurry')
        prompt = Prompt.parse(args[1])
        self.assertEqual(prompt.positive, 'a sign that says hello; world')
        self.assertEqual(prompt.negative, 'blurry')

    def test_format_prompt_escapes_a_semicolon_in_the_text(self):
        from dgenerate.batchprocess.configrunnerbuiltins import format_prompt

        prompt = Prompt.parse(r'a sign that says hello\; world; blurry')
        formatted = format_prompt(prompt)
        parsed = shell_parse(formatted)
        self.assertEqual(len(parsed), 1)
        again = Prompt.parse(parsed[0])
        self.assertEqual(again.positive, 'a sign that says hello; world')
        self.assertEqual(again.negative, 'blurry')

        several = format_prompt([prompt, Prompt.parse('a horse; artwork')])
        values = shell_parse(several)
        self.assertEqual(Prompt.parse(values[0]).positive, 'a sign that says hello; world')
        self.assertEqual(Prompt.parse(values[1]).negative, 'artwork')

    def test_prompt_function_escapes_semicolons_in_both_parts(self):
        from dgenerate.batchprocess.configrunnerbuiltins import prompt

        formatted = prompt('a sign that says hello; world', 'blurry; rain')
        again = Prompt.parse(shell_parse(formatted)[0])
        self.assertEqual(again.positive, 'a sign that says hello; world')
        self.assertEqual(again.negative, 'blurry; rain')

        positive_only = Prompt.parse(shell_parse(prompt('hello; world'))[0])
        self.assertEqual(positive_only.positive, 'hello; world')
        self.assertIsNone(positive_only.negative)

    def test_read_text_decodes_with_the_given_encoding(self):
        import os
        import tempfile

        from dgenerate.batchprocess.batchprocessor import BatchProcessError
        from dgenerate.batchprocess.configrunnerbuiltins import read_text

        fd, path = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        try:
            with open(path, 'w', encoding='cp1252', newline='\n') as handle:
                handle.write('café; rain\r\n')
            self.assertEqual(read_text(path, encoding='cp1252'), 'café; rain\n')
        finally:
            os.remove(path)
        with self.assertRaises(BatchProcessError):
            read_text(path, encoding='cp1252')


if __name__ == '__main__':
    unittest.main()
