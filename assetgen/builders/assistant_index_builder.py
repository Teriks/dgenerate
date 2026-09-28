#!/usr/bin/env python3
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

import os
from pathlib import Path
from typing import Optional


class AssistantIndexBuilder:
    """
    Builds a retrieval index for each embedding model the assistant can select.
    The indexes are packaged with dgenerate, so an install without the examples
    and docs can use any of them.

    Needs xllamacpp, and downloads each embedding model on first use. Run after
    the docs and console schemas are built.
    """

    def __init__(self, project_dir: Optional[Path] = None, force: bool = False):
        """
        :param project_dir: Project directory (defaults to current working directory)
        :param force: Rebuild even if an index matches the current sources.
        """
        self.project_dir = project_dir or Path.cwd()
        self.force = force
        self.output_file = self.project_dir / 'dgenerate' / 'assistant' / 'data' / 'index.npz'

    def build(self):
        """Build an assistant index for every selectable embedding model."""
        import gc

        import dgenerate.assistant.catalog as _catalog
        import dgenerate.assistant.index as _index
        import dgenerate.assistant.models as _models

        os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

        try:
            import xllamacpp  # noqa: F401
        except ImportError:
            print('✗ Warning: xllamacpp is not installed, keeping the existing assistant indexes')
            return

        repo = str(self.project_dir)
        current = _index.fingerprint(repo)
        for spec, _size, _note in _catalog.EMBED_MODELS:
            basename = spec.rsplit('/', 1)[-1]
            output = Path(_index.shipped_index_path(basename))
            existing = _index.Index.read_meta(str(output)) if output.exists() else None
            if not self.force and existing and existing.get('version') == _index.INDEX_VERSION and \
                    existing.get('embed_model') == basename and existing.get('fingerprint') == current:
                print(f'✓ Assistant index for {basename} is up to date')
                continue

            print(f'Building assistant index for {basename}...')
            embedder = _models.Embedder(_models.resolve_gguf(spec))
            try:
                index = _index.Index.build(repo, embedder)
                index.save(str(output))
            finally:
                del embedder
                gc.collect()

            size = output.stat().st_size / (1024 * 1024)
            print(f'✓ Assistant index for {basename}: {len(index.chunks)} chunks, {size:.1f} MiB')

    def get_output_path(self) -> Path:
        """Get the path to the default embedding model's index file."""
        return self.output_file
