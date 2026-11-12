import unittest

from dgenerate.transpile import transpile


class TestTranspileComments(unittest.TestCase):

    def test_top_level_and_tail_comments_are_kept(self):
        source = '\n'.join([
            '# header',
            'stabilityai/sd-turbo --prompts "a cat" --seeds 1 # tail',
            '',
        ])
        output = transpile(source, relaxed=True)
        self.assertIn('# header', output)
        self.assertIn('# tail', output)
        self.assertLess(output.index('# header'), output.index('config.model_path'))
        self.assertLess(output.index('# tail'), output.index('config.model_path'))
        self.assertNotIn('# tail', output.split('config.model_path', 1)[1])

    def test_comments_inside_a_loop_stay_in_the_loop(self):
        source = '\n'.join([
            '{% for i in range(2) %}',
            '# inside',
            'stabilityai/sd-turbo --prompts "a cat" # step',
            '{% endfor %}',
            '',
        ])
        output = transpile(source, relaxed=True)
        loop = output.split('for _loop0, i in enumerate(_items0):', 1)[1]
        loop = loop.split('script.sync()', 1)[0]
        self.assertIn('# inside', loop)
        self.assertIn('# step', loop)
        self.assertEqual(output.count('# inside'), 1)
        self.assertEqual(output.count('# step'), 1)

    def test_strip_removes_top_level_and_loop_comments(self):
        source = '\n'.join([
            '# header',
            '{% for i in range(2) %}',
            '# inside',
            'stabilityai/sd-turbo --prompts "a cat" # step',
            '{% endfor %}',
            '',
        ])
        output = transpile(source, relaxed=True, comments=False)
        self.assertNotIn('# header', output)
        self.assertNotIn('# inside', output)
        self.assertNotIn('# step', output)
        self.assertIn('for _loop0, i in enumerate(_items0):', output)

    def test_hash_in_a_value_is_not_a_comment(self):
        source = r'\set greeting hello \# world' + '\n'
        output = transpile(source, relaxed=True)
        comment_lines = [line.strip() for line in output.splitlines() if line.strip().startswith('#')]
        self.assertNotIn('# world', comment_lines)
        self.assertIn('hello # world', output)

    def test_hash_inside_quotes_stays_in_the_value(self):
        source = 'stabilityai/sd-turbo --prompts "a # b" --seeds 1\n'
        output = transpile(source, relaxed=True)
        comment_lines = [line.strip() for line in output.splitlines() if line.strip().startswith('#')]
        self.assertTrue(all(line.startswith('# Generated') for line in comment_lines))
        self.assertIn('a # b', output)

    def test_version_lines_are_not_comments(self):
        source = '#! dgenerate 6.0.0\n# note\n'
        output = transpile(source, relaxed=True)
        self.assertEqual(output.count('#! dgenerate 6.0.0'), 1)
        self.assertIn('# note', output)
        self.assertIn("script.version_check('#! dgenerate 6.0.0')", output)


if __name__ == '__main__':
    unittest.main()
