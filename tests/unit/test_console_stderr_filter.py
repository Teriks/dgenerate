import unittest

from dgenerate.console.console import _consume_stderr, _flush_held_stderr


_WARNING = (
    b'2026-11-22 17:18:59.188 Python[48847:862564] '
    b'Warning: Window move completed without beginning\n')


class TestStderrFilter(unittest.TestCase):

    def test_drops_the_warning_and_keeps_the_neighbors(self):
        text = b'before\n' + _WARNING + b'after\n'
        out, held = _consume_stderr(text)
        self.assertEqual(out + _flush_held_stderr(held), b'before\nafter\n')
        self.assertNotIn(b'without beginning', out + held)

    def test_holds_a_warning_split_across_reads(self):
        first, held = _consume_stderr(b'ok\n' + b'x' * 80 + b'Warning: Window move ')
        self.assertTrue(first.startswith(b'ok\n'))
        second, held = _consume_stderr(held + b'completed without beginning\nnext\n')
        self.assertNotIn(b'without beginning', first + second + held)
        self.assertEqual((first + second + _flush_held_stderr(held)).endswith(b'next\n'), True)

    def test_flush_releases_an_ordinary_tail(self):
        _out, held = _consume_stderr(b'progress')
        self.assertEqual(_flush_held_stderr(held), held)

    def test_flush_drops_a_warning_with_no_newline(self):
        _out, held = _consume_stderr(b'keep\n' + _WARNING[:-1])
        self.assertEqual(_flush_held_stderr(held), b'')
