import sys
import unittest
from pathlib import Path

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

    def test_torch_214_cuda_mapping(self):
        cases = [
            (13, 2, False, 'cu132'),
            (13, 1, False, 'cu130'),
            (13, 0, False, 'cu130'),
            (12, 9, False, 'cu126'),
            (12, 6, False, 'cu126'),
            (12, 4, False, 'cpu'),
            (13, 2, True, 'cu126'),
        ]
        for cuda_major, cuda_minor, legacy, suffix in cases:
            url = _get_torch_cuda_url(2, 14, 0, cuda_major, cuda_minor, legacy)
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

    def test_torch_214_rocm_mapping(self):
        self.assertTrue(_get_torch_rocm_url(2, 14, 0, '7.14').endswith('rocm7.14'))
        self.assertTrue(_get_torch_rocm_url(2, 14, 0, '7.2').endswith('rocm7.2'))
        self.assertTrue(_get_torch_rocm_url(2, 14, 0, '6.3').endswith('/cpu'))

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
        self.assertEqual(cap_torch_version_for_xformers('2.14.0'), '2.10.0')
        self.assertEqual(cap_torch_version_for_xformers('2.11.0'), '2.10.0')
        self.assertEqual(cap_torch_version_for_xformers('==2.14.0'), '2.10.0')
        self.assertEqual(cap_torch_version_for_xformers('2.10.0'), '2.10.0')
        self.assertEqual(cap_torch_version_for_xformers('2.10.1'), '2.10.1')
        self.assertEqual(cap_torch_version_for_xformers('2.8.0'), '2.8.0')
        self.assertIsNone(cap_torch_version_for_xformers(None))

    def test_xformers_overrides_only_when_torch_is_newer(self):
        self.assertIsNone(xformers_version_overrides('2.8.0', '==0.23.0'))
        self.assertIsNone(xformers_version_overrides('2.10.0', '==0.25.0'))
        self.assertEqual(
            xformers_version_overrides('2.14.0', '==0.29.0'),
            {'torch': '2.10.0', 'torchvision': '0.25.0'},
        )
        self.assertEqual(
            xformers_version_overrides('2.14.0', '==0.29.0', '==2.14.0'),
            {'torch': '2.10.0', 'torchvision': '0.25.0', 'torchaudio': '2.10.0'},
        )

    def test_xformers_uses_capped_torch_index(self):
        analyzer = SetupAnalyzer.__new__(SetupAnalyzer)
        analyzer.torch_version = '2.14.0'
        analyzer.log_callback = lambda message: None
        # torch 2.14 would select cu132 here; xformers caps that pin at 2.10,
        # which publishes a wheel on cu130.
        cuda_133 = GPUInfo(has_nvidia=True, cuda_version='13.3')
        cuda_130 = GPUInfo(has_nvidia=True, cuda_version='13.0')
        self.assertTrue(analyzer._is_extra_compatible('xformers', cuda_133))
        self.assertTrue(analyzer._is_extra_compatible('xformers', cuda_130))

        analyzer.torch_version = '2.7.1'
        cuda_118 = GPUInfo(has_nvidia=True, cuda_version='11.8')
        self.assertFalse(analyzer._is_extra_compatible('xformers', cuda_118))


if __name__ == '__main__':
    unittest.main()
