import io
import unittest

try:
    import tkinter as tk

    _ROOT = tk.Tk()
    _ROOT.withdraw()
except Exception:
    _ROOT = None

from dgenerate.files import TerminalLineReader

if _ROOT is not None:
    from dgenerate.console.terminaltext import TerminalText

# stderr of nested tqdm bars through a pipe on Windows, captured from a real process
NESTED_TQDM = (
    '\router:   0%|          | 0/2 [00:00<?, ?it/s]\r\n'
    '\rinner:   0%|          | 0/3 [00:00<?, ?it/s]\x1b[A\r\n'
    '\rinner:  33%|###3      | 1/3 [00:00<00:00,  4.99it/s]\x1b[A\r\n'
    '\rinner:  67%|######6   | 2/3 [00:00<00:00,  4.99it/s]\x1b[A\r\n'
    '\rinner: 100%|##########| 3/3 [00:00<00:00,  4.99it/s]\x1b[A'
    '\rinner: 100%|##########| 3/3 [00:00<00:00,  4.99it/s]\r\n'
    '\router:  50%|#####     | 1/2 [00:00<00:00,  1.66it/s]'
    '\router:  50%|#####     | 1/2 [00:00<00:00,  1.66it/s]\r\n'
)


def _chunks(data: str):
    reader = TerminalLineReader(io.BytesIO(data.encode('utf-8')))
    while True:
        line = reader.readline()
        if not line:
            return
        yield line.decode('utf-8')


@unittest.skipIf(_ROOT is None, 'Tk is not available')
class TestTerminalText(unittest.TestCase):
    def setUp(self):
        self.text = tk.Text(_ROOT)
        self.terminal = TerminalText(self.text)

    def tearDown(self):
        self.text.destroy()

    def lines(self):
        return [line.rstrip() for line in self.text.get('1.0', 'end-1c').split('\n')]

    def feed(self, data, stream='stderr', tag=None):
        for chunk in _chunks(data):
            self.terminal.write(chunk, tag=tag, stream=stream)

    def test_nested_tqdm_redraws_in_place(self):
        self.feed(NESTED_TQDM)
        # tqdm closes the inner bar by moving up and redrawing it over the outer
        # bar's line, then redraws the outer bar below, same as a real terminal.
        self.assertEqual(self.lines(), [
            'inner: 100%|##########| 3/3 [00:00<00:00,  4.99it/s]',
            'outer:  50%|#####     | 1/2 [00:00<00:00,  1.66it/s]',
            '',
        ])

    def test_single_bar_and_plain_lines(self):
        self.feed('Loading\n')
        self.feed('\rbar 1\rbar 2\rbar 3\n')
        self.feed('done\n')
        self.assertEqual(self.lines(), ['Loading', 'bar 3', 'done', ''])

    def test_erase_and_color_codes(self):
        self.feed('\x1b[31mred text\x1b[0m\r\x1b[2Kshort\n')
        self.assertEqual(self.lines(), ['short', ''])

    def test_stream_switch_starts_fresh_line(self):
        self.feed('\rbar 50%', stream='stderr')
        self.feed('message\n', stream='stdout')
        self.feed('\rbar 60%', stream='stderr')
        self.assertEqual(self.lines(), ['bar 50%', 'message', 'bar 60%'])

    def test_error_tag_follows_overwrite(self):
        self.feed('\rerror bar', stream='stderr', tag='error')
        self.feed('\rplain line\n', stream='stderr')
        self.assertEqual(self.text.tag_ranges('error'), ())

    def test_cursor_survives_scrollback_trim(self):
        self.feed('a\nb\n\rbar 1')
        self.text.delete('1.0', '2.0')
        self.feed('\rbar 2\n')
        self.assertEqual(self.lines(), ['b', 'bar 2', ''])


if __name__ == '__main__':
    unittest.main()
