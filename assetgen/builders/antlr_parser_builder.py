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

"""Rebuild the config ANTLR parser when a grammar file changes."""

import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional


class AntlrParserBuilder:
    """Generate ``dgenerate.transpile.grammar`` from its ``.g4`` files."""

    ANTLR_VERSION = '4.13.2'
    GRAMMARS = ('ConfigLexer.g4', 'Config.g4')
    OUTPUTS = (
        'Config.py',
        'ConfigLexer.py',
        'ConfigVisitor.py',
        'Config.tokens',
        'ConfigLexer.tokens',
        'Config.interp',
        'ConfigLexer.interp',
    )
    _VISITOR_IMPORT = (
        'if "." in __name__:\n'
        '    from .Config import Config\n'
        'else:\n'
        '    from Config import Config\n'
    )

    def __init__(self, project_dir: Optional[Path] = None):
        self.project_dir = Path(project_dir) if project_dir is not None else Path.cwd()
        self.grammar_dir = self.project_dir / 'dgenerate' / 'transpile' / 'grammar'

    def needs_build(self) -> bool:
        """True when a grammar is newer than a generated file, or a file is missing."""
        outputs = [self.grammar_dir / name for name in self.OUTPUTS]
        grammars = [self.grammar_dir / name for name in self.GRAMMARS]
        if any(not path.is_file() for path in outputs + grammars):
            return True
        newest_grammar = max(path.stat().st_mtime for path in grammars)
        oldest_output = min(path.stat().st_mtime for path in outputs)
        return newest_grammar > oldest_output

    def build(self):
        if not self.needs_build():
            print('Config parser is up to date.')
            return
        print('Config grammar changed; rebuilding the parser.')
        java = self._require_java()
        jar = self._antlr_jar()
        self._generate(java, jar)
        self._ensure_visitor_import()

    def _require_java(self) -> str:
        java = shutil.which('java')
        if not java:
            raise SystemExit(
                'The config grammar changed, but Java is not installed. '
                'Install a JDK and re-run assetgen to rebuild the ANTLR parser.'
            )
        result = subprocess.run(
            [java, '-version'],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            raise SystemExit(f'Java was found but `java -version` failed.\n{detail}')
        return java

    def _antlr_jar(self) -> Path:
        cached = (
            self.project_dir / 'assetgen' / 'cache'
            / f'antlr-{self.ANTLR_VERSION}-complete.jar'
        )
        if cached.is_file() and cached.stat().st_size > 0:
            return cached
        cached.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://www.antlr.org/download/antlr-{self.ANTLR_VERSION}-complete.jar'
        print(f'Downloading {url}')
        try:
            urllib.request.urlretrieve(url, cached)
        except Exception as exc:
            raise SystemExit(f'Could not download the ANTLR tool: {exc}') from exc
        return cached

    def _generate(self, java: str, jar: Path):
        common = [
            java, '-jar', str(jar),
            '-Dlanguage=Python3', '-visitor', '-no-listener',
            '-o', '.',
        ]
        subprocess.run(
            common + ['ConfigLexer.g4'],
            cwd=self.grammar_dir,
            check=True,
        )
        subprocess.run(
            common + ['-lib', '.', 'Config.g4'],
            cwd=self.grammar_dir,
            check=True,
        )

    def _ensure_visitor_import(self):
        """Keep the visitor importable both as a package module and a script."""
        path = self.grammar_dir / 'ConfigVisitor.py'
        text = path.read_text(encoding='utf-8')
        if 'if "." in __name__' in text and 'from .Config import Config' in text:
            return
        replacement = 'from antlr4 import *\n' + self._VISITOR_IMPORT
        if 'from antlr4 import *' in text:
            text = text.replace('from antlr4 import *\n', replacement, 1)
            for line in ('from .Config import Config\n', 'from Config import Config\n'):
                text = text.replace(line, '')
        path.write_text(text, encoding='utf-8')
