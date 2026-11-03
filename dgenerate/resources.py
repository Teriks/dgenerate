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
import platform
import typing
import urllib.parse

import packaging.version
import requests

import dgenerate.types as _types

__doc__ = """
Package resources, version, pre-release and latest release information, icon, etc.

This module can be imported without incurring a large import overhead.
"""

__version__ = '6.0.0'


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
    network_install: bool

    def __init__(self,
                 version: str,
                 commit: str | None,
                 branch: str | None,
                 pre_release: bool,
                 network_install: bool = False,
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
        if not isinstance(network_install, bool):
            raise TypeError(
                "CurrentReleaseInfo.network_install must be a bool, got "
                f"{type(network_install).__name__}")
        self.version = version
        self.commit = commit
        self.branch = branch
        self.pre_release = pre_release
        self.network_install = network_install

    def json_dump(self, fo: typing.IO[str]):
        json.dump(self.__dict__, fo)

    def json_dumps(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def json_load(cls, fo: typing.IO[str]):
        data = json.load(fo)

        required_keys = {"version", "commit", "branch", "pre_release"}
        allowed_keys = required_keys | {"network_install"}
        missing_keys = required_keys - data.keys()
        extra_keys = data.keys() - allowed_keys

        if missing_keys:
            raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
        if extra_keys:
            raise ValueError(f"Unexpected keys: {', '.join(extra_keys)}")

        network_install = data.get('network_install', False)
        if not isinstance(network_install, bool):
            raise TypeError(
                "CurrentReleaseInfo.network_install must be a bool, got "
                f"{type(network_install).__name__}")

        return cls(
            version=data['version'],
            commit=data['commit'],
            branch=data['branch'],
            pre_release=data['pre_release'],
            network_install=network_install,
        )

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
        return CurrentReleaseInfo(version(), None, None, False, network_install=False)

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

    A single leading ``v`` or ``V`` is ignored so GitHub tags such as
    ``v6.0.0`` compare with ``6.0.0``.

    :param version1: left version
    :param version2: right version
    :return: :py:class:`VersionComparison`
    """
    def _parse(value: str) -> packaging.version.Version:
        text = value.strip()
        if len(text) > 1 and text[0] in 'vV' and text[1].isdigit():
            text = text[1:]
        return packaging.version.Version(text)

    try:
        version1 = _parse(version1)
        version2 = _parse(version2)
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


_GITHUB_REPO = 'Teriks/dgenerate'
_GITHUB_API_HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'User-Agent': 'dgenerate',
}


def is_network_install() -> bool:
    """
    Return whether this install was performed by the network installer.

    True only when ``release.json`` contains ``network_install: true``, which
    ``UvInstaller.generate_release_json`` writes before installing.
    """
    return get_release_info().network_install


class PreReleaseBranchUpdateInfo:
    """
    Result of comparing the installed commit to the tip of a development branch.
    """

    branch: str
    local_commit: str
    latest_commit: str
    ahead_by: int
    branch_url: str
    pre_release_release_url: str

    def __init__(self,
                 branch: str,
                 local_commit: str,
                 latest_commit: str,
                 ahead_by: int,
                 branch_url: str,
                 pre_release_release_url: str):
        self.branch = branch
        self.local_commit = local_commit
        self.latest_commit = latest_commit
        self.ahead_by = ahead_by
        self.branch_url = branch_url
        self.pre_release_release_url = pre_release_release_url


def normalize_git_sha(sha: str) -> str:
    """Return a lowercase stripped git SHA."""
    return sha.strip().lower()


def commits_equal(sha1: str, sha2: str) -> bool:
    """
    Compare two git SHAs, treating a full hash and its unique short prefix as equal.
    """
    a = normalize_git_sha(sha1)
    b = normalize_git_sha(sha2)
    if a == b:
        return True
    if len(a) < len(b):
        return b.startswith(a)
    return a.startswith(b)


def check_pre_release_branch_update(
        local_commit: str,
        branch: str,
) -> PreReleaseBranchUpdateInfo | None:
    """
    Check whether ``branch`` has commits not contained in ``local_commit``.

    Uses the GitHub compare API (``local_commit...branch``).

    :param local_commit: Installed build commit from :py:func:`get_release_info`.
    :param branch: Development branch name from :py:func:`get_release_info`.
    :return: Update info, or ``None`` if the check could not be completed.
    """
    if not local_commit or not branch:
        return None

    local_commit = normalize_git_sha(local_commit)
    branch = branch.strip()
    if not branch:
        return None

    base = urllib.parse.quote(local_commit, safe='')
    head = urllib.parse.quote(branch, safe='')
    url = f'https://api.github.com/repos/{_GITHUB_REPO}/compare/{base}...{head}'

    try:
        response = requests.get(url, headers=_GITHUB_API_HEADERS, timeout=2)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return None

    ahead_by = data.get('ahead_by', 0)
    if not isinstance(ahead_by, int):
        ahead_by = 0

    latest_commit = local_commit
    commits = data.get('commits')
    if isinstance(commits, list) and commits:
        last = commits[-1]
        if isinstance(last, dict):
            sha = last.get('sha')
            if isinstance(sha, str) and sha:
                latest_commit = normalize_git_sha(sha)
    elif ahead_by > 0:
        tip_url = f'https://api.github.com/repos/{_GITHUB_REPO}/commits/{head}'
        try:
            tip_response = requests.get(
                tip_url, headers=_GITHUB_API_HEADERS, timeout=2)
            tip_response.raise_for_status()
            tip_sha = tip_response.json().get('sha')
            if isinstance(tip_sha, str) and tip_sha:
                latest_commit = normalize_git_sha(tip_sha)
        except (requests.exceptions.RequestException, ValueError, KeyError):
            pass

    branch_url = f'https://github.com/{_GITHUB_REPO}/tree/{urllib.parse.quote(branch)}'
    pre_release_release_url = (
        f'https://github.com/{_GITHUB_REPO}/releases/tag/pre-release')

    return PreReleaseBranchUpdateInfo(
        branch=branch,
        local_commit=local_commit,
        latest_commit=latest_commit,
        ahead_by=ahead_by,
        branch_url=branch_url,
        pre_release_release_url=pre_release_release_url,
    )


def check_latest_release() -> LatestReleaseInfo | None:
    """
    Get the latest software release for this software.

    :return: :py:class:`ReleaseInfo`
    """

    url = f'https://api.github.com/repos/{_GITHUB_REPO}/releases/latest'

    try:
        response = requests.get(url, headers=_GITHUB_API_HEADERS, timeout=2)
        response.raise_for_status()
        latest_release = response.json()

        tag_name = latest_release['tag_name']
        release_name = latest_release['name']
        release_url = latest_release['html_url']

        return LatestReleaseInfo(tag_name, release_name, release_url)

    except requests.exceptions.RequestException:
        return None


def installer_asset_prefix() -> str | None:
    """
    Filename prefix for the network installer on this platform, or ``None`` if unsupported.

    Stable release assets are ``dgenerate-installer-{platform}-{version}``
    (version without a leading ``v``). Pre-release assets are
    ``dgenerate-installer-{platform}-{commit}``.
    """
    system = platform.system()
    machine = platform.machine().lower()
    if system == 'Windows':
        return 'dgenerate-installer-win-'
    if system == 'Linux':
        return 'dgenerate-installer-linux-'
    if system == 'Darwin':
        if machine in ('arm64', 'aarch64'):
            return 'dgenerate-installer-macos-'
        return None
    return None


def fetch_release_installer_download(tag: str) -> tuple[str, str] | None:
    """
    Look up this platform's network installer on a GitHub release.

    :param tag: Release tag, such as ``v6.0.0`` or ``pre-release``.
    :return: ``(asset_name, browser_download_url)`` or ``None``.
    """
    prefix = installer_asset_prefix()
    if prefix is None or not tag:
        return None

    quoted = urllib.parse.quote(tag, safe='')
    url = f'https://api.github.com/repos/{_GITHUB_REPO}/releases/tags/{quoted}'
    try:
        response = requests.get(url, headers=_GITHUB_API_HEADERS, timeout=10)
        response.raise_for_status()
        release = response.json()
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return None

    assets = release.get('assets')
    if not isinstance(assets, list):
        return None

    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = asset.get('name')
        download_url = asset.get('browser_download_url')
        if (isinstance(name, str)
                and name.startswith(prefix)
                and isinstance(download_url, str)
                and download_url):
            return name, download_url

    return None


def download_release_installer(download_url: str, dest_path: str) -> bool:
    """
    Download a release installer asset to ``dest_path``.
    """
    try:
        with requests.get(
                download_url,
                headers={**_GITHUB_API_HEADERS, 'Accept': 'application/octet-stream'},
                stream=True,
                timeout=(10, 600)) as response:
            response.raise_for_status()
            with open(dest_path, 'wb') as fo:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fo.write(chunk)
        return True
    except (OSError, requests.exceptions.RequestException):
        return False


__all__ = _types.module_all()
