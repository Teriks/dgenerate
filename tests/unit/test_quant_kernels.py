import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import dgenerate.messages as messages
import dgenerate.pipelinewrapper.quantkernels as quantkernels
from dgenerate.pipelinewrapper.quantkernels import QuantKernel
from dgenerate.subcommands.kernel_warmup import KernelWarmupSubCommand

_INSTALLER_DIR = Path(__file__).resolve().parents[2] / 'installer'
if str(_INSTALLER_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTALLER_DIR))

from network_installer.uv_installer import UvInstaller  # noqa: E402


class TestQuantKernelSpecs(unittest.TestCase):

    def test_metal_downloads_the_mlx_kernel_only(self):
        specs = quantkernels.quant_kernel_specs('mps')
        self.assertEqual(specs, [QuantKernel(
            'kernels-community/mlx-quantization-metal-kernels', version=2)])

    def test_rocm_downloads_the_aiter_kernel_only(self):
        specs = quantkernels.quant_kernel_specs(
            'cuda', hip=True, capability=(9, 4))
        self.assertEqual(specs, [QuantKernel(
            'kernels-community/aiter-kernels', version=2)])

    def test_linux_cuda_downloads_the_gguf_kernel(self):
        specs = quantkernels.quant_kernel_specs('cuda', capability=(8, 9))
        self.assertEqual(specs, [QuantKernel(
            'Isotr0py/ggml', revision='main', trust_remote_code=True)])

    def test_windows_cuda_requests_the_gguf_kernel(self):
        specs = quantkernels.quant_kernel_specs('cuda', capability=(12, 0))
        self.assertEqual(specs, [QuantKernel(
            'Isotr0py/ggml', revision='main', trust_remote_code=True)])

    def test_old_cuda_skips_the_gguf_kernel(self):
        specs = quantkernels.quant_kernel_specs('cuda', capability=(6, 1))
        self.assertEqual(specs, [])

    def test_cpu_and_xpu_have_no_hub_kernel(self):
        self.assertEqual(quantkernels.quant_kernel_specs('cpu'), [])
        self.assertEqual(quantkernels.quant_kernel_specs('xpu'), [])


class TestQuantKernelDownload(unittest.TestCase):

    def test_download_calls_get_kernel_for_each_spec(self):
        specs = [
            QuantKernel('kernels-community/aiter-kernels', version=2),
            QuantKernel('Isotr0py/ggml', revision='main', trust_remote_code=True),
        ]
        with mock.patch.object(quantkernels, '_get_kernel') as get_kernel, \
                mock.patch.object(messages, 'log'):
            status = quantkernels.download_quant_kernels(specs)
        self.assertEqual(status, 0)
        self.assertEqual(
            get_kernel.call_args_list,
            [mock.call(specs[0]), mock.call(specs[1])])

    def test_connection_failure_uses_the_offline_message(self):
        spec = QuantKernel('kernels-community/mlx-quantization-metal-kernels', version=2)
        with mock.patch.object(
                quantkernels, '_get_kernel', side_effect=ConnectionError('failed')), \
                mock.patch.object(messages, 'error') as error:
            status = quantkernels.download_quant_kernels([spec])
        self.assertEqual(status, 1)
        error.assert_called_once_with(quantkernels.OFFLINE_MESSAGE)

    def test_missing_build_is_skipped(self):
        spec = QuantKernel('Isotr0py/ggml', revision='main', trust_remote_code=True)
        with mock.patch.object(
                quantkernels, '_get_kernel',
                side_effect=FileNotFoundError(
                    'Cannot find a build variant for this system')), \
                mock.patch.object(messages, 'log'), \
                mock.patch.object(messages, 'error') as error:
            status = quantkernels.download_quant_kernels([spec])
        self.assertEqual(status, 0)
        error.assert_not_called()

    def test_offline_mode_refuses_before_downloading(self):
        command = KernelWarmupSubCommand(
            loaded_by_name='kernel-warmup',
            plugin_module_paths=[], args=[], local_files_only=True)
        spec = QuantKernel('kernels-community/aiter-kernels', version=2)
        with mock.patch.object(
                quantkernels, 'current_quant_kernel_specs', return_value=[spec]), \
                mock.patch.object(quantkernels, 'download_quant_kernels') as download, \
                mock.patch.object(messages, 'error') as error:
            status = command()
        self.assertEqual(status, 1)
        error.assert_called_once_with(quantkernels.OFFLINE_MESSAGE)
        download.assert_not_called()


class TestInstallerKernelWarmup(unittest.TestCase):

    def _installer(self):
        logs = []
        installer = UvInstaller(log_callback=logs.append)
        return installer, logs

    def test_old_dgenerate_skips_the_subcommand(self):
        installer, logs = self._installer()
        help_result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='Available sub-commands:\n\n    "image-process"\n    "assistant"\n',
            stderr='')
        with tempfile_exe() as exe, \
                mock.patch.object(
                    installer, 'get_installation_info',
                    return_value=mock.Mock(dgenerate_exe=exe)), \
                mock.patch(
                    'network_installer.subprocess_utils.run_silent',
                    return_value=help_result) as run:
            self.assertTrue(installer.warmup_quant_kernels())
        self.assertEqual(run.call_count, 1)
        self.assertIn('kernel-warmup', ' '.join(logs))
        self.assertIn('Skipping quant kernel download.', ' '.join(logs))

    def test_new_dgenerate_runs_kernel_warmup(self):
        installer, logs = self._installer()
        help_result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='    "kernel-warmup"\n', stderr='')
        warmup_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='Downloaded kernels-community/aiter-kernels.\n',
            stderr='')
        with tempfile_exe() as exe, \
                mock.patch.object(
                    installer, 'get_installation_info',
                    return_value=mock.Mock(dgenerate_exe=exe)), \
                mock.patch(
                    'network_installer.subprocess_utils.run_silent',
                    side_effect=[help_result, warmup_result]) as run:
            self.assertTrue(installer.warmup_quant_kernels())
        warmup_call = run.call_args_list[1]
        self.assertEqual(
            warmup_call.args[0][-2:],
            ['--sub-command', 'kernel-warmup'])
        self.assertIn('Quant kernel download finished.', logs)

    def test_download_failure_does_not_fail_install(self):
        installer, logs = self._installer()
        help_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='"kernel-warmup"\n', stderr='')
        warmup_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout='',
            stderr=quantkernels.OFFLINE_MESSAGE + '\n')
        with tempfile_exe() as exe, \
                mock.patch.object(
                    installer, 'get_installation_info',
                    return_value=mock.Mock(dgenerate_exe=exe)), \
                mock.patch(
                    'network_installer.subprocess_utils.run_silent',
                    side_effect=[help_result, warmup_result]):
            self.assertTrue(installer.warmup_quant_kernels())
        self.assertIn(quantkernels.OFFLINE_MESSAGE, logs)
        self.assertTrue(any('did not complete' in line for line in logs))


class tempfile_exe:
    def __enter__(self):
        import tempfile
        self._handle = tempfile.NamedTemporaryFile(suffix='-dgenerate', delete=False)
        self._handle.close()
        self.path = self._handle.name
        return self.path

    def __exit__(self, exc_type, exc, tb):
        Path(self.path).unlink(missing_ok=True)
