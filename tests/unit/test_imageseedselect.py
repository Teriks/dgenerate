import unittest

try:
    import tkinter as tk

    _root = tk.Tk()
    _root.withdraw()
    _root.destroy()
    from dgenerate.console.imageseedselect import _ImageSeedSelect
except Exception:
    tk = None
    _ImageSeedSelect = None


@unittest.skipIf(_ImageSeedSelect is None, 'Tk is not available')
class TestImageSeedSelect(unittest.TestCase):
    def _dialog(self):
        root = tk.Tk()
        root.geometry('900x700+40+40')
        root.update_idletasks()
        inserted = []
        dialog = _ImageSeedSelect(inserted.append, master=root)
        dialog.update_idletasks()
        return root, dialog, inserted

    def _close(self, root, dialog):
        if dialog.winfo_exists():
            dialog.destroy()
        root.destroy()

    def test_several_seeds_and_one_mask(self):
        root, dialog, inserted = self._dialog()
        try:
            dialog._seeds.entries[0].insert(0, 'a.png')
            dialog._seeds.add()
            dialog._seeds.entries[1].insert(0, 'b.png')
            dialog._masks.add()
            dialog._masks.entries[0].insert(0, 'mask.png')
            dialog._insert_click()
            self.assertEqual(inserted, ['images:a.png, b.png;mask.png'])
        finally:
            self._close(root, dialog)

    def test_latents_list(self):
        root, dialog, inserted = self._dialog()
        try:
            dialog._seeds.entries[0].delete(0, tk.END)
            dialog._latents.add()
            dialog._latents.entries[0].insert(0, 'one.pt')
            dialog._latents.add()
            dialog._latents.entries[1].insert(0, 'two.pt')
            dialog._insert_click()
            self.assertEqual(inserted, ['latents:one.pt, two.pt'])
        finally:
            self._close(root, dialog)

    def test_reference_with_inpaint(self):
        root, dialog, inserted = self._dialog()
        try:
            dialog._seeds.entries[0].insert(0, 'seed.png')
            dialog._masks.add()
            dialog._masks.entries[0].insert(0, 'mask.png')
            dialog._references.add()
            dialog._references.entries[0].insert(0, 'earth.jpg')
            dialog._references.add()
            dialog._references.entries[1].insert(0, 'mountain.png')
            dialog._insert_click()
            self.assertEqual(
                inserted,
                ['seed.png;mask=mask.png;reference=earth.jpg, mountain.png'])
        finally:
            self._close(root, dialog)

    def test_reference_requires_seed(self):
        root, dialog, inserted = self._dialog()
        try:
            dialog._seeds.entries[0].delete(0, tk.END)
            dialog._references.add()
            dialog._references.entries[0].insert(0, 'earth.jpg')
            dialog._insert_click()
            self.assertEqual(inserted, [])
        finally:
            self._close(root, dialog)

    def test_control_list(self):
        root, dialog, inserted = self._dialog()
        try:
            dialog._seeds.entries[0].insert(0, 'seed.png')
            dialog._controls.add()
            dialog._controls.entries[0].insert(0, 'c1.png')
            dialog._controls.add()
            dialog._controls.entries[1].insert(0, 'c2.png')
            dialog._insert_click()
            self.assertEqual(inserted, ['seed.png;control=c1.png, c2.png'])
        finally:
            self._close(root, dialog)

    def test_adapter_rows(self):
        root, dialog, inserted = self._dialog()
        try:
            dialog._seeds.entries[0].delete(0, tk.END)
            dialog._adapters.add()
            dialog._adapters.rows[0]['path'].insert(0, 'a.png')
            dialog._adapters.rows[0]['resize'].insert(0, '512')
            dialog._adapters.add()
            dialog._adapters.rows[1]['path'].insert(0, 'b.png')
            dialog._adapters.rows[1]['aspect'].set(False)
            dialog._insert_click()
            self.assertEqual(
                inserted,
                ['adapter:a.png|resize=512 + b.png|aspect=false'])
        finally:
            self._close(root, dialog)

    def test_remove_file_row(self):
        root, dialog, inserted = self._dialog()
        try:
            dialog._seeds.add()
            self.assertEqual(len(dialog._seeds.entries), 2)
            extra = dialog._seeds.entries[1]
            dialog._seeds.remove(dialog._seeds._frames[1], extra)
            self.assertEqual(len(dialog._seeds.entries), 1)
            dialog._seeds.entries[0].insert(0, 'only.png')
            dialog._insert_click()
            self.assertEqual(inserted, ['only.png'])
        finally:
            self._close(root, dialog)


if __name__ == '__main__':
    unittest.main()
