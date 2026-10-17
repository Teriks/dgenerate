import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_INSTALLER_DIR = Path(__file__).resolve().parents[2] / 'installer'
if str(_INSTALLER_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTALLER_DIR))

from network_installer.platform_detection import (  # noqa: E402
    AMD_WINDOWS_MULTIARCH_INDEX,
    GPUInfo,
    _get_torch_cuda_url,
    _get_torch_rocm_url,
    _get_torch_xpu_url,
    _parse_nvidia_smi_cuda_version,
    cap_torch_version_for_xformers,
    is_amd_windows_multiarch_index,
    xformers_version_overrides,
)
from network_installer.setup_analyzer import SetupAnalyzer  # noqa: E402


class TestInstallerTorchIndex(unittest.TestCase):

    def test_nvidia_smi_cuda_umd_version(self):
        header = (
            '| NVIDIA-SMI 610.47                 KMD Version: 610.47'
            '        CUDA UMD Version: 13.3     |'
        )
        self.assertEqual(_parse_nvidia_smi_cuda_version(header), '13.3')
        self.assertEqual(
            _parse_nvidia_smi_cuda_version('CUDA UMD version    : 13.3'),
            '13.3',
        )
        self.assertEqual(_parse_nvidia_smi_cuda_version('CUDA Version: 12.8'), '12.8')

    def test_torch_213_cuda_mapping(self):
        cases = [
            (13, 2, False, 'cu130'),
            (13, 1, False, 'cu130'),
            (13, 0, False, 'cu130'),
            (12, 9, False, 'cu126'),
            (12, 6, False, 'cu126'),
            (12, 4, False, 'cpu'),
            (13, 2, True, 'cu126'),
        ]
        for cuda_major, cuda_minor, legacy, suffix in cases:
            url = _get_torch_cuda_url(2, 13, 0, cuda_major, cuda_minor, legacy)
            self.assertTrue(url.endswith(suffix), (cuda_major, cuda_minor, legacy, url))

    def test_older_torch_cuda_mappings_preserved(self):
        self.assertTrue(_get_torch_cuda_url(2, 8, 0, 13, 0, False).endswith('cu129'))
        self.assertTrue(_get_torch_cuda_url(2, 8, 0, 12, 8, False).endswith('cu128'))
        self.assertTrue(_get_torch_cuda_url(2, 7, 1, 12, 8, False).endswith('cu128'))
        self.assertTrue(_get_torch_cuda_url(2, 7, 1, 12, 6, False).endswith('cu126'))
        self.assertTrue(_get_torch_cuda_url(2, 6, 0, 12, 6, False).endswith('cu126'))
        self.assertTrue(_get_torch_cuda_url(2, 6, 0, 12, 4, False).endswith('cu124'))
        self.assertTrue(_get_torch_cuda_url(2, 5, 0, 12, 4, False).endswith('cu124'))
        self.assertTrue(_get_torch_cuda_url(2, 3, 0, 12, 1, False).endswith('cu121'))

    def test_torch_213_rocm_mapping(self):
        self.assertTrue(_get_torch_rocm_url(2, 13, 0, '7.2').endswith('rocm7.2'))
        self.assertTrue(_get_torch_rocm_url(2, 13, 0, '7.1').endswith('rocm7.1'))
        self.assertTrue(_get_torch_rocm_url(2, 13, 0, '6.3').endswith('/cpu'))

    def test_older_torch_rocm_mappings(self):
        self.assertTrue(_get_torch_rocm_url(2, 8, 0, '6.4').endswith('rocm6.4'))
        self.assertTrue(_get_torch_rocm_url(2, 8, 0, '6.3').endswith('rocm6.3'))
        self.assertTrue(_get_torch_rocm_url(2, 7, 1, '6.3').endswith('rocm6.3'))
        self.assertTrue(_get_torch_rocm_url(2, 6, 0, '6.1').endswith('rocm6.1'))

    def test_xpu_supported_from_25(self):
        self.assertEqual(_get_torch_xpu_url(2, 14, 0), 'https://download.pytorch.org/whl/xpu')
        self.assertEqual(_get_torch_xpu_url(2, 5, 0), 'https://download.pytorch.org/whl/xpu')
        self.assertIsNone(_get_torch_xpu_url(2, 4, 0))

    def test_amd_windows_index_helper(self):
        self.assertTrue(is_amd_windows_multiarch_index(AMD_WINDOWS_MULTIARCH_INDEX))
        self.assertFalse(is_amd_windows_multiarch_index('https://download.pytorch.org/whl/rocm7.2'))
        self.assertFalse(is_amd_windows_multiarch_index(None))

    def test_xformers_caps_torch_newer_than_210(self):
        self.assertEqual(cap_torch_version_for_xformers('2.13.0'), '2.10.0')
        self.assertEqual(cap_torch_version_for_xformers('2.11.0'), '2.10.0')
        self.assertEqual(cap_torch_version_for_xformers('==2.13.0'), '2.10.0')
        self.assertEqual(cap_torch_version_for_xformers('2.10.0'), '2.10.0')
        self.assertEqual(cap_torch_version_for_xformers('2.10.1'), '2.10.1')
        self.assertEqual(cap_torch_version_for_xformers('2.8.0'), '2.8.0')
        self.assertIsNone(cap_torch_version_for_xformers(None))

    def test_xformers_overrides_only_when_torch_is_newer(self):
        self.assertIsNone(xformers_version_overrides('2.8.0', '==0.23.0'))
        self.assertIsNone(xformers_version_overrides('2.10.0', '==0.25.0'))
        self.assertEqual(
            xformers_version_overrides('2.13.0', '==0.29.0'),
            {'torch': '2.10.0', 'torchvision': '0.25.0'},
        )
        self.assertEqual(
            xformers_version_overrides('2.13.0', '==0.29.0', '==2.13.0'),
            {'torch': '2.10.0', 'torchvision': '0.25.0', 'torchaudio': '2.10.0'},
        )

    def test_xformers_uses_capped_torch_index(self):
        analyzer = SetupAnalyzer.__new__(SetupAnalyzer)
        analyzer.torch_version = '2.13.0'
        analyzer.log_callback = lambda message: None
        # The cap is applied before the index lookup, so this uses the torch
        # 2.10 table. CUDA 12.9 stays on cu129, which the current 2.13 row
        # no longer publishes.
        capped = cap_torch_version_for_xformers(analyzer.torch_version)
        self.assertEqual(capped, '2.10.0')
        self.assertTrue(_get_torch_cuda_url(2, 10, 0, 13, 3, False).endswith('cu130'))
        self.assertTrue(_get_torch_cuda_url(2, 10, 0, 12, 9, False).endswith('cu129'))
        self.assertTrue(_get_torch_cuda_url(2, 10, 0, 12, 8, False).endswith('cu128'))
        self.assertTrue(_get_torch_cuda_url(2, 10, 0, 12, 6, False).endswith('cu126'))
        cuda_133 = GPUInfo(has_nvidia=True, cuda_version='13.3')
        cuda_130 = GPUInfo(has_nvidia=True, cuda_version='13.0')
        cuda_129 = GPUInfo(has_nvidia=True, cuda_version='12.9')
        self.assertTrue(analyzer._is_extra_compatible('xformers', cuda_133))
        self.assertTrue(analyzer._is_extra_compatible('xformers', cuda_130))
        self.assertTrue(analyzer._is_extra_compatible('xformers', cuda_129))

        # Releases that already pin torch 2.10 or older keep that pin and its index.
        self.assertEqual(cap_torch_version_for_xformers('2.8.0'), '2.8.0')
        self.assertIsNone(xformers_version_overrides('2.8.0', '==0.23.0'))
        self.assertTrue(_get_torch_cuda_url(2, 8, 0, 12, 8, False).endswith('cu128'))
        self.assertTrue(_get_torch_cuda_url(2, 8, 0, 13, 0, False).endswith('cu129'))
        analyzer.torch_version = '2.8.0'
        self.assertTrue(analyzer._is_extra_compatible(
            'xformers', GPUInfo(has_nvidia=True, cuda_version='12.8')))

        analyzer.torch_version = '2.7.1'
        cuda_118 = GPUInfo(has_nvidia=True, cuda_version='11.8')
        self.assertFalse(analyzer._is_extra_compatible('xformers', cuda_118))
        self.assertTrue(_get_torch_cuda_url(2, 7, 1, 11, 8, False).endswith('cu118'))

    def test_current_release_does_not_ship_xformers_extra(self):
        setup_py = Path(__file__).resolve().parents[2] / 'setup.py'
        text = setup_py.read_text(encoding='utf-8')
        self.assertNotIn("'xformers'", text)
        self.assertNotIn('"xformers"', text)

    def test_vulkan_preview_is_recommended_on_every_platform(self):
        analyzer = SetupAnalyzer.__new__(SetupAnalyzer)
        analyzer.extras = {
            'console_ui_vulkan': ['vulkan', 'miniaudio'],
            'console_ui_opengl': ['pyopengltk'],
        }
        analyzer.log_callback = lambda message: None
        analyzer._log = lambda message: None
        with patch('network_installer.setup_analyzer.platform.system', return_value='Windows'):
            recommended = analyzer.get_recommended_extras(GPUInfo())
        self.assertIn('console_ui_vulkan', recommended)
        self.assertIn('console_ui_opengl', recommended)
        with patch('network_installer.setup_analyzer.platform.system', return_value='Darwin'):
            recommended = analyzer.get_recommended_extras(GPUInfo())
            available = analyzer.get_available_extras(GPUInfo())
        self.assertIn('console_ui_vulkan', recommended)
        self.assertNotIn('console_ui_opengl', recommended)
        self.assertNotIn('console_ui_opengl', available)

    def test_triton_windows_accepts_nvidia_and_rdna3(self):
        from network_installer.platform_detection import (
            amd_rdna_major_from_gfx,
            amd_rdna_major_from_name,
            amd_rdna3_or_newer,
            triton_windows_compatible,
        )

        self.assertEqual(amd_rdna_major_from_gfx('gfx1100'), 3)
        self.assertEqual(amd_rdna_major_from_gfx('gfx1151'), 3)
        self.assertEqual(amd_rdna_major_from_gfx('gfx1201'), 4)
        self.assertEqual(amd_rdna_major_from_gfx('gfx1300'), 5)
        self.assertEqual(amd_rdna_major_from_gfx('gfx1030'), 2)
        self.assertEqual(amd_rdna_major_from_name('AMD Radeon RX 7900 XTX'), 3)
        self.assertEqual(amd_rdna_major_from_name('AMD Radeon RX 9070 XT'), 4)
        self.assertEqual(amd_rdna_major_from_name('AMD Radeon RX 10500 XT'), 5)
        self.assertEqual(amd_rdna_major_from_name('AMD Radeon RX 6800 XT'), 2)
        self.assertTrue(amd_rdna3_or_newer(
            GPUInfo(has_amd=True, amd_gfx='gfx1300')))
        self.assertTrue(amd_rdna3_or_newer(
            GPUInfo(has_amd=True, gpu_name='AMD Radeon RX 11000 XT')))

        analyzer = SetupAnalyzer.__new__(SetupAnalyzer)
        analyzer.log_callback = lambda message: None
        self.assertTrue(analyzer._is_extra_compatible(
            'triton_windows', GPUInfo(has_nvidia=True)))
        self.assertTrue(analyzer._is_extra_compatible(
            'triton_windows',
            GPUInfo(has_amd=True, amd_gfx='gfx1100', amd_rdna_major=3)))
        self.assertTrue(triton_windows_compatible(
            GPUInfo(has_amd=True, gpu_name='AMD Radeon RX 7800 XT')))
        self.assertFalse(analyzer._is_extra_compatible(
            'triton_windows',
            GPUInfo(has_amd=True, amd_gfx='gfx1030', amd_rdna_major=2)))
        self.assertFalse(analyzer._is_extra_compatible(
            'triton_windows', GPUInfo()))


if __name__ == '__main__':
    unittest.main()
