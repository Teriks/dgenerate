import ctypes
import unittest

from dgenerate.console.openglframe_darwin import (
    NSOpenGLProfileVersion4_1Core,
    aqua_nsview,
    glsl_source,
    pixel_format_attributes,
    _MacDrawable,
)


class TestMacOpenGLFrame(unittest.TestCase):
    def test_apple_shaders_use_glsl_410(self):
        source = '#version 330 core\nvoid main() {}\n'
        self.assertEqual(
            glsl_source(source, darwin=True),
            '#version 410 core\nvoid main() {}\n')
        self.assertEqual(glsl_source(source, darwin=False), source)

    def test_pixel_format_requests_opengl_41(self):
        attrs = pixel_format_attributes()
        self.assertEqual(attrs[-1], 0)
        self.assertEqual(attrs[1], NSOpenGLProfileVersion4_1Core)
        self.assertIn(5, attrs)

    def test_missing_frame_is_installed_on_darwin_only(self):
        import types

        from dgenerate.console.openglframe_darwin import OpenGLFrame, install_frame

        module = types.SimpleNamespace()
        self.assertFalse(install_frame(module, system='Windows'))
        self.assertFalse(hasattr(module, 'OpenGLFrame'))
        if OpenGLFrame is None:
            self.assertFalse(install_frame(module, system='Darwin'))
            return
        self.assertTrue(install_frame(module, system='Darwin'))
        self.assertIs(module.OpenGLFrame, OpenGLFrame)
        self.assertTrue(install_frame(module, system='Darwin'))

    def test_aqua_nsview_reads_mac_drawable_view(self):
        fake = _MacDrawable(winPtr=0x1111, view=0x2222)
        self.assertEqual(aqua_nsview(ctypes.addressof(fake)), 0x2222)
        with self.assertRaises(RuntimeError):
            aqua_nsview(0)


if __name__ == '__main__':
    unittest.main()
