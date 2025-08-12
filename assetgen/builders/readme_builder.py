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

from pathlib import Path
from typing import Optional

from ..core.rst_preprocessor import RSTPreprocessor


class ReadmeBuilder:
    """Builder for README files from templates."""

    def __init__(self, preprocessor: RSTPreprocessor, project_dir: Optional[Path] = None):
        """
        Initialize the ReadmeBuilder.

        :param preprocessor: RST preprocessor instance
        :type preprocessor: RSTPreprocessor
        :param project_dir: Project directory (defaults to current working directory)
        :type project_dir: Optional[Path]
        """
        self.preprocessor = preprocessor
        self.project_dir = project_dir or Path.cwd()
        self.template_dir = self.project_dir / 'assetgen' / 'templates' / 'readme'
        self.output_file = self.project_dir / 'README.rst'

    def build(self):
        """Build the README file from template."""
        readme_template = self.template_dir / 'readme.template.rst'
        
        if not readme_template.exists():
            raise FileNotFoundError(f"README template not found: {readme_template}")
        
        print("Building README.rst...")
        
        try:
            self.preprocessor.render_file(
                input_file=readme_template,
                output_file=self.output_file
            )
            print("✓ README.rst built successfully")
        except Exception as e:
            print(f"✗ Error building README.rst: {e}")
            raise

    def get_template_path(self) -> Path:
        """Get the path to the README template."""
        return self.template_dir / 'readme.template.rst'

    def get_output_path(self) -> Path:
        """Get the path to the README output file."""
        return self.output_file 