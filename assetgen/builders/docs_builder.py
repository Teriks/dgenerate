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
from typing import Optional, List, Dict

from ..core.rst_preprocessor import RSTPreprocessor


class DocsBuilder:
    """Builder for documentation files from templates."""

    def __init__(self, preprocessor: RSTPreprocessor, project_dir: Optional[Path] = None):
        """
        Initialize the DocsBuilder.

        :param preprocessor: RST preprocessor instance
        :type preprocessor: RSTPreprocessor
        :param project_dir: Project directory (defaults to current working directory)
        :type project_dir: Optional[Path]
        """
        self.preprocessor = preprocessor
        self.project_dir = project_dir or Path.cwd()
        self.template_dir = self.project_dir / 'assetgen' / 'templates' / 'docs'
        self.output_dir = self.project_dir / 'docs'

    def build(self):
        """Build all documentation files from templates."""
        # Define template mappings
        templates = self._get_template_mappings()
        
        print(f"Building {len(templates)} documentation files...")
        
        processed_count = 0
        for template in templates:
            if template['input'].exists():
                print(f"Building {template['description']}...")
                
                try:
                    self.preprocessor.render_file(
                        input_file=template['input'],
                        output_file=template['output']
                    )
                    print(f"✓ {template['description']} built successfully")
                    processed_count += 1
                except Exception as e:
                    print(f"✗ Error building {template['description']}: {e}")
                    raise
            else:
                print(f"⚠ Template {template['input']} not found, skipping...")
        
        print(f"Successfully processed {processed_count} documentation files")

    def _get_template_mappings(self) -> List[Dict[str, Path]]:
        """Get the template to output file mappings."""
        return [
            {
                'input': self.template_dir / 'manual.template.rst',
                'output': self.output_dir / 'manual.rst',
                'description': 'docs/manual.rst'
            },
            {
                'input': self.template_dir / 'intro.template.rst',
                'output': self.output_dir / 'intro.rst',
                'description': 'docs/intro.rst'
            }
        ]

    def get_template_dir(self) -> Path:
        """Get the documentation templates directory."""
        return self.template_dir

    def get_output_dir(self) -> Path:
        """Get the documentation output directory."""
        return self.output_dir 