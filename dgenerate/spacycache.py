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
import email.parser
import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import sys
import types
import typing
import urllib.parse
import zipfile

import requests
import spacy
import tqdm

import dgenerate.filelock as _filelock
import dgenerate.memory as _memory
import dgenerate.types as _types
import dgenerate.exceptions as _d_exceptions

__doc__ = """
Tools for downloading spaCy models to arbitrary locations, compatible with dgenerate's frozen environment.
"""


class SpacyModelNotFoundError(_d_exceptions.ModelNotFoundError):
    """
    Raised when a spacy model cannot be loaded, due to being unable to
    locate it either online or in the cache.
    """
    pass


def _download_whl_file(model_name, url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = _memory.calculate_chunk_size(total_size)

    with open(output_path, "wb") as file, tqdm.tqdm(
            desc=f'Downloading spaCy model "{model_name}"...',
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(block_size):
            file.write(chunk)
            bar.update(len(chunk))


def get_spacy_cache_directory() -> str:
    """
    Get the default spacy model cache directory.

    Or the value of the environmental variable ``DGENERATE_CACHE`` joined with ``spacy``.

    :return: string (directory path)
    """
    user_cache_path = os.environ.get('DGENERATE_CACHE')

    if user_cache_path is not None:
        path = os.path.join(user_cache_path, 'spacy')
    else:
        path = os.path.expanduser(os.path.join('~', '.cache', 'dgenerate', 'spacy'))

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return path


def _get_version(model: str, comp: dict) -> str:
    import spacy.about as _about
    if model not in comp:
        raise SpacyModelNotFoundError(
            f"No compatible package found for '{model}' (spaCy v{_about.__version__})",
        )
    return comp[model][0]


def _clear_cache_directory():
    cache_directory = get_spacy_cache_directory()
    lock_file = os.path.abspath(os.path.join(cache_directory, '.lock'))

    for item in os.listdir(cache_directory):
        item_path = os.path.join(cache_directory, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        elif os.path.abspath(item_path) != lock_file:
            os.remove(item_path)


def _get_compatibility(local_files_only: bool, attempt: int = 0) -> dict:
    import spacy.about as _about
    import spacy.util as _util

    version = _util.get_minor_version(_about.__version__)

    cache_directory = get_spacy_cache_directory()

    compatibility_file = os.path.join(cache_directory, 'compatibility.json')

    if not os.path.exists(compatibility_file):
        if local_files_only or _offline_mode:
            raise SpacyModelNotFoundError(
                f'Could not download spaCy "{_about.__compatibility__}" '
                f'due to offline mode being active.'
            )

        try:
            r = requests.get(_about.__compatibility__)
            r.raise_for_status()

            comp_table = r.json()['spacy']

            with open(compatibility_file, 'w') as file:
                json.dump(comp_table, file)

        except requests.RequestException as e:
            raise SpacyModelNotFoundError(
                f'Could not download spaCy "{_about.__compatibility__}", reason: {e}')

        return comp_table[version]
    else:
        with open(compatibility_file, 'r') as file:
            comp_table = json.load(file)

        if attempt == 0:
            if version not in comp_table:
                _clear_cache_directory()
                # wipe out the cache and download a newer version
                # of the compatibility index
                return _get_compatibility(local_files_only, attempt=1)
            else:
                return comp_table[version]
        if attempt == 1:
            if version not in comp_table:
                raise SpacyModelNotFoundError(
                    f'Current version of spaCy ({version}) '
                    f'not found in "{compatibility_file}"')
            return comp_table[version]


def _install_whl(model_name, filepath, install_dir):
    with zipfile.ZipFile(filepath, 'r') as whl:

        metadata_files = [f for f in whl.namelist() if f.endswith('METADATA') or f.endswith('PKG-INFO')]
        if not metadata_files:
            return []

        with whl.open(metadata_files[0]) as metadata_file:
            metadata_content = metadata_file.read().decode('utf-8')
            metadata = email.parser.Parser().parsestr(metadata_content)
            dependencies = metadata.get_all('Requires-Dist') or []

            package_names = set()

            for dep in dependencies:
                if 'extra ==' in dep:
                    continue
                package_name = re.split(r"[><=!~;\s]", dep, 1)[0].strip()
                package_names.add(package_name)

        missing_dependencies = [i for i in package_names if not importlib.metadata.metadata(i)]

        if missing_dependencies:
            raise SpacyModelNotFoundError(
                f'Cannot install spaCy model "{model_name}" due to '
                f'dependencies not being met: {",".join(missing_dependencies)}')

        whl.extractall(install_dir)


_offline_mode = False


def is_offline_mode() -> bool:
    """
    Check if the global offline mode for the spacy cache is enabled.

    :return: `True` if offline mode is enabled, `False` otherwise.
    """
    global _offline_mode
    return _offline_mode


def enable_offline_mode():
    """
    Enable global offline mode for the spacy cache.

    This will prevent any network requests from being made, and will only use files
    that are already in the spacy cache.
    """
    global _offline_mode
    _offline_mode = True


def disable_offline_mode():
    """
    Disable global offline mode for the spacy cache.

    This will allow network requests to be made again.
    """
    global _offline_mode
    _offline_mode = False


@contextlib.contextmanager
def offline_mode_context(enabled=True):
    """
    Context manager to temporarily enable or disable global offline mode for the spacy cache.

    :param enabled: If `True`, enables offline mode. If `False`, disables it.
    """
    global _offline_mode
    original_mode = _offline_mode
    _offline_mode = enabled
    try:
        yield
    finally:
        _offline_mode = original_mode


def load_spacy_model(
        name: str,
        *,
        vocab: spacy.Vocab | bool = True,
        disable: str | typing.Iterable[str] = spacy.util._DEFAULT_EMPTY_PIPES,
        enable: str | typing.Iterable[str] = spacy.util._DEFAULT_EMPTY_PIPES,
        exclude: str | typing.Iterable[str] = spacy.util._DEFAULT_EMPTY_PIPES,
        config: dict[str, typing.Any] | spacy.Config = spacy.util.SimpleFrozenDict(),
        local_files_only: bool = False
) -> spacy.Language:
    """
    Load a spaCy model, possibly downloading it if needed.

    :param name: Name of the spaCy model.
    :param vocab: A Vocab object. If True, a vocab is created.
    :param disable: Name(s) of pipeline component(s) to disable. Disabled pipes will be loaded but
        won't be run unless explicitly enabled using ``nlp.enable_pipe``.
    :param enable: Name(s) of pipeline component(s) to enable. All other pipes will be disabled but
        can be enabled later using ``nlp.enable_pipe``.
    :param exclude: Name(s) of pipeline component(s) to exclude. Excluded components won't be loaded.
    :param config: Config overrides as a nested dict or a dict keyed by section values in dot notation.
    :param local_files_only: Avoid connecting to the internet? look in the cache only.
    :return: The loaded nlp object. :py:class:`spacy.Language`
    """
    global _offline_mode
    import spacy.cli.download as _download_module
    import spacy.about as _about

    spacy_cache_dir = get_spacy_cache_directory()

    if not isinstance(_download_module, types.ModuleType):
        _download_module = sys.modules[_download_module.__module__]

    with _filelock.temp_file_lock(os.path.join(spacy_cache_dir, '.lock')):

        filename = _download_module.get_model_filename(
            name, _get_version(name, _get_compatibility(local_files_only)))

        model_site_package = os.path.join(spacy_cache_dir, name)

        if not os.path.isdir(model_site_package):

            if local_files_only or _offline_mode:
                raise SpacyModelNotFoundError(
                    f'Cannot find spaCy model "{name}" in the spaCy model cache, '
                    f'offline mode is active and it may need to be downloaded.')

            base_url = _about.__download_url__

            if not base_url.endswith("/"):
                base_url = _about.__download_url__ + "/"

            download_url = urllib.parse.urljoin(base_url, filename)

            whl_download_to = os.path.join(spacy_cache_dir, os.path.basename(filename))

            try:
                _download_whl_file(name, download_url, whl_download_to)
            except requests.RequestException as e:
                raise SpacyModelNotFoundError(
                    f'Unable to downloaded spaCy model "{name}", reason: {e}')

            try:
                _install_whl(name, whl_download_to, spacy_cache_dir)
            except Exception as e:
                raise SpacyModelNotFoundError(
                    f'Unable to extract spaCy model whl file "{whl_download_to}", reason: {e}'
                )

            os.unlink(whl_download_to)

        model_path = os.path.join(os.path.abspath(model_site_package), os.path.dirname(filename))

        return spacy.load(
            model_path,
            vocab=vocab,
            disable=disable,
            enable=enable,
            exclude=exclude,
            config=config
        )


__all__ = _types.module_all()
