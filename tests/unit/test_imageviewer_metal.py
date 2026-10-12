import platform
import unittest


class TestMetalPreview(unittest.TestCase):
    def test_shader_flips_y_for_metal(self):
        from dgenerate.console.imageviewer_metal import METAL_SHADER
        self.assertIn('float4(v.x, -v.y, 0.0, 1.0)', METAL_SHADER)

    def test_packed_vertices_count_triangles(self):
        from dgenerate.console.imageviewer_metal import _pack_vertices
        draws, blob = _pack_vertices(
            [(0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1)],
            [([(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)], (1, 1, 1, 1))])
        self.assertEqual(draws[0], ('texture', 0, 3, None))
        self.assertEqual(draws[1][0], 'color')
        self.assertEqual(draws[1][1], 3)
        self.assertEqual(draws[1][2], 3)
        self.assertEqual(len(blob), 6 * 16)

    def test_metal_stays_off_away_from_macos(self):
        if platform.system() == 'Darwin':
            self.skipTest('Windows and Linux keep their own preview')
        from dgenerate.console.imageviewer import _use_metal
        self.assertFalse(_use_metal())


if __name__ == '__main__':
    unittest.main()
