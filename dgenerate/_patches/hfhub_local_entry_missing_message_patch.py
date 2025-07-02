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


import huggingface_hub
import huggingface_hub.file_download
import huggingface_hub.errors

og1 = huggingface_hub.file_download._hf_hub_download_to_local_dir
og2 = huggingface_hub.file_download._hf_hub_download_to_cache_dir


def _hf_hub_download_to_local_dir(*args, **kwargs):
    status = kwargs.get('local_files_only', False)
    try:
        return og1(*args, **kwargs)
    except huggingface_hub.errors.LocalEntryNotFoundError as e:
        if not status:
            raise e

        raise huggingface_hub.errors.LocalEntryNotFoundError(
            f'Offline mode is enabled, but the file "{kwargs["filename"]}" from Hugging Face repo "{kwargs["repo_id"]}" '
            f'was not found locally. Please ensure the file is available locally or disable offline mode.'
        ) from e
    except huggingface_hub.errors.OfflineModeIsEnabled as e:
        raise huggingface_hub.errors.OfflineModeIsEnabled(
            f'Offline mode is enabled, but the file "{kwargs["filename"]}" from Hugging Face repo "{kwargs["repo_id"]}" '
            f'was not found locally. Please ensure the file is available locally or disable offline mode.'
        ) from e

def _hf_hub_download_to_cache_dir(*args, **kwargs):
    status = kwargs.get('local_files_only', False)
    try:
        return og2(*args, **kwargs)
    except huggingface_hub.errors.LocalEntryNotFoundError as e:
        if not status:
            raise e

        raise huggingface_hub.errors.LocalEntryNotFoundError(
            f'Offline mode is enabled, but the file "{kwargs["filename"]}" from Hugging Face repo "{kwargs["repo_id"]}" '
            f'was not found locally. Please ensure the file is available locally or disable offline mode.'
        ) from e
    except huggingface_hub.errors.OfflineModeIsEnabled as e:
        raise huggingface_hub.errors.OfflineModeIsEnabled(
            f'Offline mode is enabled, but the file "{kwargs["filename"]}" from Hugging Face repo "{kwargs["repo_id"]}" '
            f'was not found locally. Please ensure the file is available locally or disable offline mode.'
        ) from e

huggingface_hub.file_download._hf_hub_download_to_local_dir = _hf_hub_download_to_local_dir
huggingface_hub.file_download._hf_hub_download_to_cache_dir = _hf_hub_download_to_cache_dir