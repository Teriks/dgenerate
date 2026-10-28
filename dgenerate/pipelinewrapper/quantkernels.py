# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Hub kernels downloaded by the GGUF and ComfyUI quant loaders."""

import dataclasses

import dgenerate.messages as _messages


OFFLINE_MESSAGE = (
    'Cannot download platform specific quant kernels, you are not online.')

# ComfyUI quant matmuls. The loader asks for one of these, and only on the
# device that has a Hub kernel. CUDA and XPU use kernels built into PyTorch.
COMFY_HUB_KERNELS = {
    'mlx': ('kernels-community/mlx-quantization-metal-kernels', 2),
    'aiter': ('kernels-community/aiter-kernels', 2),
}

# Diffusers loads this when CUDA GGUF kernels are enabled. Compute capability
# 7.0 is the same gate Diffusers uses. ``kernels`` then picks the build for
# this operating system, so a Windows build is downloaded when one is published.
GGUF_CUDA_KERNEL = 'Isotr0py/ggml'

_CONNECTIVITY_TYPES = {
    'ConnectError', 'ConnectTimeout', 'ReadTimeout', 'WriteTimeout', 'PoolTimeout',
    'NetworkError', 'RemoteProtocolError', 'ConnectTimeoutError', 'MaxRetryError',
    'NewConnectionError', 'NameResolutionError', 'ProxyError', 'OfflineModeIsEnabled',
    'LocalEntryNotFoundError',
}

_CONNECTIVITY_TEXT = (
    'getaddrinfo',
    'name or service not known',
    'temporary failure in name resolution',
    'nodename nor servname',
    'network is unreachable',
    'network unreachable',
    'failed to establish a new connection',
    'connection refused',
    'connection aborted',
    'connection reset',
    'max retries exceeded',
    'timed out',
    'offline mode',
    'outgoing traffic has been disabled',
)


@dataclasses.dataclass(frozen=True)
class QuantKernel:
    """One ``kernels.get_kernel`` call the quant loaders make on this machine."""

    repo: str
    version: int | None = None
    revision: str | None = None
    trust_remote_code: bool = False


def quant_kernel_specs(device_type: str,
                       *,
                       hip: bool = False,
                       capability: tuple[int, int] | None = None) -> list[QuantKernel]:
    """
    Kernels the GGUF and Comfy quant loaders download for ``device_type``.

    Other accelerators are omitted. A CUDA machine does not fetch the Metal
    or ROCm kernels, and a Metal machine does not fetch the CUDA GGUF kernel.
    The GGUF kernel is requested on any CUDA device of compute capability 7.0
    or newer. The Hub build for this operating system is the one that downloads.
    """
    specs = []
    if device_type == 'mps':
        repo, version = COMFY_HUB_KERNELS['mlx']
        specs.append(QuantKernel(repo, version=version))
    elif device_type == 'cuda' and hip:
        repo, version = COMFY_HUB_KERNELS['aiter']
        specs.append(QuantKernel(repo, version=version))
    elif (device_type == 'cuda' and not hip
          and capability is not None and capability[0] >= 7):
        specs.append(QuantKernel(
            GGUF_CUDA_KERNEL, revision='main', trust_remote_code=True))
    return specs


def _exception_chain(exc: BaseException):
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_connectivity_error(exc: BaseException) -> bool:
    for current in _exception_chain(exc):
        if type(current).__name__ in _CONNECTIVITY_TYPES:
            return True
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        text = str(current).lower()
        if any(piece in text for piece in _CONNECTIVITY_TEXT):
            return True
    return False


def _is_unpublished(exc: BaseException) -> bool:
    """True when the Hub has no build of this kernel for the running machine."""
    if _is_connectivity_error(exc):
        return False
    for current in _exception_chain(exc):
        name = type(current).__name__
        if name in {
            'RepositoryNotFoundError', 'RevisionNotFoundError', 'GatedRepoError',
            'DisabledRepoError', 'EntryNotFoundError', 'FileNotFoundError',
        }:
            return True
        if name == 'HfHubHTTPError':
            response = getattr(current, 'response', None)
            status = getattr(response, 'status_code', None)
            if status in (401, 403, 404):
                return True
        text = str(current).lower()
        if ('cannot find a build variant' in text
                or 'repository not found' in text
                or 'version or revision must be specified' in text):
            return True
    return False


def _get_kernel(spec: QuantKernel):
    from kernels import get_kernel
    kwargs = {}
    if spec.version is not None:
        kwargs['version'] = spec.version
    if spec.revision is not None:
        kwargs['revision'] = spec.revision
    if spec.trust_remote_code:
        kwargs['trust_remote_code'] = True
    return get_kernel(spec.repo, **kwargs)


def current_quant_kernel_specs() -> list[QuantKernel]:
    """Kernel downloads for the default accelerator on this computer."""
    import torch

    import dgenerate.torchutil as _torchutil

    device_type = _torchutil.default_device()
    hip = device_type == 'cuda' and bool(getattr(torch.version, 'hip', None))
    capability = None
    if device_type == 'cuda' and torch.cuda.is_available():
        capability = tuple(torch.cuda.get_device_capability())
    return quant_kernel_specs(device_type, hip=hip, capability=capability)


def download_quant_kernels(specs: list[QuantKernel] | None = None) -> int:
    """
    Download ``specs``, or the kernels for this computer when ``specs`` is omitted.

    A missing connection prints :data:`OFFLINE_MESSAGE` and returns 1.
    A kernel with no build for this computer is skipped.

    :return: process status, ``0`` on success
    """
    if specs is None:
        specs = current_quant_kernel_specs()
    if not specs:
        _messages.log('No platform specific quant kernels are used on this device.')
        return 0

    for spec in specs:
        try:
            _get_kernel(spec)
        except Exception as exc:
            if _is_connectivity_error(exc):
                _messages.error(OFFLINE_MESSAGE)
                return 1
            if _is_unpublished(exc):
                _messages.log(
                    f'No build of {spec.repo} is published for this device.')
                continue
            _messages.error(f'Could not download {spec.repo}: {exc}')
            return 1
        _messages.log(f'Downloaded {spec.repo}.')
    return 0
