import ctypes
import unittest


class TestVulkanPreviewShaders(unittest.TestCase):
    def test_shader_bytecode_is_spirv(self):
        try:
            from dgenerate.console.imageviewer_vk import (
                _assemble_color_fragment,
                _assemble_texture_fragment,
                _assemble_vertex,
            )
        except ImportError:
            self.skipTest('console_ui_vulkan extra is not installed')
        for code in (_assemble_vertex(), _assemble_texture_fragment(), _assemble_color_fragment()):
            self.assertEqual(code[:4], b'\x03\x02#\x07')

    def test_surface_extensions_follow_the_window_system(self):
        try:
            import vulkan as vk
            from dgenerate.console.imageviewer_vk import _surface_extension_names
        except ImportError:
            self.skipTest('console_ui_vulkan extra is not installed')
        windows, flags = _surface_extension_names('win32', {
            vk.VK_KHR_SURFACE_EXTENSION_NAME,
            vk.VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        })
        self.assertIn(vk.VK_KHR_WIN32_SURFACE_EXTENSION_NAME, windows)
        self.assertEqual(flags, 0)
        linux, flags = _surface_extension_names('x11', {
            vk.VK_KHR_SURFACE_EXTENSION_NAME,
            vk.VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
        })
        self.assertIn(vk.VK_KHR_XLIB_SURFACE_EXTENSION_NAME, linux)
        self.assertEqual(flags, 0)
        mac_mvk, flags = _surface_extension_names('aqua', {
            vk.VK_KHR_SURFACE_EXTENSION_NAME,
            vk.VK_MVK_MACOS_SURFACE_EXTENSION_NAME,
            vk.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        })
        self.assertIn(vk.VK_MVK_MACOS_SURFACE_EXTENSION_NAME, mac_mvk)
        self.assertIn(vk.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME, mac_mvk)
        self.assertEqual(flags, vk.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR)
        mac_metal, flags = _surface_extension_names('aqua', {
            vk.VK_KHR_SURFACE_EXTENSION_NAME,
            vk.VK_EXT_METAL_SURFACE_EXTENSION_NAME,
            vk.VK_MVK_MACOS_SURFACE_EXTENSION_NAME,
            vk.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        })
        self.assertIn(vk.VK_EXT_METAL_SURFACE_EXTENSION_NAME, mac_metal)
        self.assertNotIn(vk.VK_MVK_MACOS_SURFACE_EXTENSION_NAME, mac_metal)
        self.assertEqual(flags, vk.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR)
        with self.assertRaises(RuntimeError):
            _surface_extension_names('wayland', {vk.VK_KHR_SURFACE_EXTENSION_NAME})

    def test_tk_aqua_nsview_reads_mac_drawable_view(self):
        try:
            from dgenerate.console.imageviewer_vk import _MacDrawable, _tk_aqua_nsview
        except ImportError:
            self.skipTest('console_ui_vulkan extra is not installed')
        fake = _MacDrawable(winPtr=0x1111, view=0x2222)
        view = _tk_aqua_nsview(ctypes.addressof(fake))
        self.assertEqual(view, 0x2222)
        with self.assertRaises(RuntimeError):
            _tk_aqua_nsview(0)


if __name__ == '__main__':
    unittest.main()
