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
import enum
import importlib.resources
import json
import typing

import packaging.version
import requests

import dgenerate.types as _types

__doc__ = """
Package resources, version, pre-release and latest release information, icon, etc.

This module can be imported without incurring a large import overhead.
"""

__version__ = '5.0.0'


def get_icon_path() -> str:
    """
    Get a path to dgenerates .ico icon file.
    :return: file path
    """
    with importlib.resources.path('dgenerate', 'icon.ico') as path:
        return str(path)


def get_icon_data() -> bytes:
    """
    Get dgenerates .ico icon file as an array of bytes.
    :return: bytes
    """
    return importlib.resources.read_binary('dgenerate', 'icon.ico')


class CurrentReleaseInfo:
    version: str
    commit: str
    branch: str
    pre_release: bool

    def __init__(self,
                 version: str,
                 commit: str | None,
                 branch: str | None,
                 pre_release: bool,
                 ):
        if not isinstance(version, str):
            raise TypeError(
                f"CurrentReleaseInfo.version must be a string, got {type(version).__name__}")
        if commit is not None and not isinstance(commit, str):
            raise TypeError(
                f"CurrentReleaseInfo.commit must be a string, got {type(commit).__name__}")
        if branch is not None and not isinstance(branch, str):
            raise TypeError(
                f"CurrentReleaseInfo.branch must be a string, got {type(branch).__name__}")
        if not isinstance(pre_release, bool):
            raise TypeError(
                f"CurrentReleaseInfo.pre_release must be a bool, got {type(pre_release).__name__}")
        self.version = version
        self.commit = commit
        self.branch = branch
        self.pre_release = pre_release

    def json_dump(self, fo: typing.IO[str]):
        json.dump(self.__dict__, fo)

    def json_dumps(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def json_load(cls, fo: typing.IO[str]):
        data = json.load(fo)

        required_keys = {"version", "commit", "branch", "pre_release"}
        missing_keys = required_keys - data.keys()
        extra_keys = data.keys() - required_keys

        if missing_keys:
            raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
        if extra_keys:
            raise ValueError(f"Unexpected keys: {', '.join(extra_keys)}")

        return cls(**data)

    def copy(self):
        return CurrentReleaseInfo(
            **{k: v for k, v in self.__dict__.items() if not callable(v)})


_release_info: CurrentReleaseInfo | None = None


def get_release_info() -> CurrentReleaseInfo:
    """
    Return release information, commit and branch will be ``None``
    inside the development environment.
    """
    global _release_info
    if _release_info is not None:
        return _release_info.copy()
    try:
        with importlib.resources.open_text('dgenerate', f'release.json') as file:
            _release_info = CurrentReleaseInfo.json_load(file)
    except Exception:
        return CurrentReleaseInfo(version(), None, None, False)

    return _release_info.copy()


def version():
    """
    Code version. In the form MAJOR.MINOR.PATCH.
    """

    global __version__

    return __version__


class LatestReleaseInfo:
    """
    Latest release info from github.
    """
    tag_name: str
    release_name: str
    release_url: str

    def __init__(self,
                 tag_name: str,
                 release_name: str,
                 release_url: str):
        self.tag_name = tag_name
        self.release_name = release_name
        self.release_url = release_url

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)


class VersionComparison(enum.Enum):
    """
    Version comparison result.
    """
    V1_NEWER = 0
    V2_NEWER = 1
    SAME = 2


def _ver_extra(ver: packaging.version.Version) -> tuple[str, int] | None:
    if ver.is_postrelease:
        return 'post', ver.post
    if ver.is_devrelease:
        return 'dev', ver.dev
    if ver.is_prerelease:
        return ver.pre
    return None


def compare_versions(version1: str, version2: str) -> VersionComparison:
    """
    Python PEP 440 version comparison utility.

    :param version1: left version
    :param version2: right version
    :return: :py:class:`VersionComparison`
    """
    try:
        version1 = packaging.version.Version(version1)
        version2 = packaging.version.Version(version2)
    except packaging.version.InvalidVersion as e:
        raise ValueError(str(e)) from e

    if version1 > version2:
        return VersionComparison.V1_NEWER
    elif version1 < version2:
        return VersionComparison.V2_NEWER
    else:
        extra1 = _ver_extra(version1)
        extra2 = _ver_extra(version2)

        if extra1 != extra2:
            # fallback, probably not required
            if extra1 is not None and extra2 is None:
                return VersionComparison.V1_NEWER
            elif extra1 is None and extra2 is not None:
                return VersionComparison.V2_NEWER
            elif extra1[1] > extra2[1]:
                return VersionComparison.V1_NEWER
            elif extra1[1] < extra2[1]:
                return VersionComparison.V2_NEWER

        return VersionComparison.SAME


def check_latest_release() -> LatestReleaseInfo | None:
    """
    Get the latest software release for this software.

    :return: :py:class:`ReleaseInfo`
    """

    url = f"https://api.github.com/repos/Teriks/dgenerate/releases/latest"

    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        latest_release = response.json()

        tag_name = latest_release['tag_name']
        release_name = latest_release['name']
        release_url = latest_release['html_url']

        return LatestReleaseInfo(tag_name, release_name, release_url)

    except requests.exceptions.RequestException:
        return None


__all__ = _types.module_all()
