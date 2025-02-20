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
import contextlib
import huggingface_hub


@contextlib.contextmanager
def _with_hf_local_files_only(status: bool):
    """
    Directly patch ``huggingface_hub`` to set the ``local_files_only`` status on ``hf_hub_download``

    :param status: ``local_files_only`` value
    """
    og1 = huggingface_hub.file_download._hf_hub_download_to_local_dir
    og2 = huggingface_hub.file_download._hf_hub_download_to_cache_dir

    try:
        def patch_status1(*args, **kwargs):
            kwargs['local_files_only'] = status
            return og1(*args, **kwargs)

        def patch_status2(*args, **kwargs):
            kwargs['local_files_only'] = status
            return og2(*args, **kwargs)

        huggingface_hub.file_download._hf_hub_download_to_local_dir = patch_status1
        huggingface_hub.file_download._hf_hub_download_to_cache_dir = patch_status2
        yield
    finally:
        huggingface_hub.file_download._hf_hub_download_to_local_dir = og1
        huggingface_hub.file_download._hf_hub_download_to_cache_dir = og2
