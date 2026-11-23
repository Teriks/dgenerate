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

"""Keep OpenCV's AVFoundation class names from colliding with PyAV."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Same length as the originals, so the Mach-O stays aligned.
_REPLACEMENTS = (
    (b'AVFFrameReceiver', b'CV2FrameReceiver'),
    (b'AVFAudioReceiver', b'CV2AudioReceiver'),
)


def rewrite_avfoundation_class_names(data: bytes) -> bytes:
    """Return *data* with OpenCV's duplicate class names renamed."""
    for old, new in _REPLACEMENTS:
        data = data.replace(old, new)
    return data


def _cv2_libavdevice_dylibs():
    spec = importlib.util.find_spec('cv2')
    if spec is None or not spec.submodule_search_locations:
        return []
    found = []
    for location in spec.submodule_search_locations:
        found.extend(Path(location).glob('.dylibs/libavdevice*.dylib'))
    return found


def _sign(path: Path) -> bool:
    signed = subprocess.run(
        ['codesign', '--force', '--sign', '-', str(path)],
        capture_output=True, text=True)
    return signed.returncode == 0


def rename_opencv_avfoundation_classes() -> None:
    """Rename the classes inside OpenCV's bundled libavdevice, then re-sign it.

    pip does not run code after installing a wheel, so a plain
    ``pip install dgenerate`` never reaches the installer's copy of this
    edit. Doing it here, before cv2 is imported, covers that install.
    The edit is a no-op once the names are already changed. A library that
    cannot be written or re-signed is left as it was.
    """
    if sys.platform != 'darwin':
        return
    for dylib in _cv2_libavdevice_dylibs():
        try:
            original = dylib.read_bytes()
        except OSError:
            continue
        updated = rewrite_avfoundation_class_names(original)
        if updated == original:
            continue
        temporary = dylib.with_name(dylib.name + '.tmp')
        try:
            temporary.write_bytes(updated)
            os.replace(temporary, dylib)
        except OSError:
            try:
                temporary.unlink()
            except OSError:
                pass
            continue
        if _sign(dylib):
            continue
        try:
            dylib.write_bytes(original)
            _sign(dylib)
        except OSError:
            pass
