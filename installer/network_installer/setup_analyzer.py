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

"""
Setup analyzer for the network installer.

This module provides functionality to analyze dgenerate source code and extract
dependency information, version requirements, and extras from pyproject.toml.
"""

import os
import sys
import tomllib
import traceback
import warnings
from collections.abc import Callable
from importlib.machinery import SourceFileLoader

from packaging import specifiers
from packaging import version as pkg_version


class SetupAnalyzer:

    def __init__(self, setup_py_path: str, log_callback: Callable[[str], None] | None = None):
        self.setup_py_path = setup_py_path
        self.source_dir = os.path.dirname(setup_py_path)
        self.log_callback = log_callback or print
        self.version = None
        self.python_requirement = None
        self.torch_version = None
        self.extras = {}

    def _log(self, message: str):
        """Log message using callback or print"""
        if self.log_callback:
            self.log_callback(f"ANALYZER: {message}")
        else:
            print(f"ANALYZER: {message}")

    def load_setup_as_library(self) -> bool:
        """Load setup.py for version and extras, parse pyproject.toml for Python/torch requirements"""
        try:
            self._log(f"Loading setup.py for version and extras: {self.setup_py_path}")

            # Set up mock functions for setuptools
            self._setup_mock_functions()

            # Change to setup.py directory for relative imports
            original_cwd = os.getcwd()
            os.chdir(self.source_dir)

            try:
                setup_module = SourceFileLoader('setup', self.setup_py_path).load_module()

                # Extract version from setup.py
                if hasattr(setup_module, 'VERSION'):
                    self.version = setup_module.VERSION
                    self._log(f"Found VERSION (uppercase) in setup.py: {self.version}")
                elif hasattr(setup_module, 'version'):
                    self.version = setup_module.version
                    self._log(f"Found version (lowercase) in setup.py: {self.version}")

                # Extract extras from setup.py
                if hasattr(setup_module, 'extras'):
                    self.extras = setup_module.extras
                    self._log(f"Found extras in setup.py: {list(self.extras.keys())}")
                else:
                    # Check if we captured extras from setup() call
                    if hasattr(self, '_captured_extras') and self._captured_extras:
                        self.extras = self._captured_extras
                        self._log(f"Captured extras from setup() call: {list(self.extras.keys())}")
                    else:
                        self.extras = {}
                        self._log("No extras found in setup.py or captured from setup() call")

                # Parse pyproject.toml for Python and torch requirements
                self._parse_pyproject_toml()

                return True

            finally:
                os.chdir(original_cwd)

        except Exception as e:
            self._log(f"Error loading setup.py as library: {e}")
            self._log(f"Traceback: {traceback.format_exc()}")
            return False

    def _setup_mock_functions(self):
        """Set up mock functions for setuptools when in library mode"""
        # Store original modules
        self._original_setuptools = sys.modules.get('setuptools')
        self._original_setuptools_setup = getattr(sys.modules.get('setuptools'), 'setup',
                                                  None) if 'setuptools' in sys.modules else None
        self._original_setuptools_find_packages = getattr(sys.modules.get('setuptools'), 'find_packages',
                                                          None) if 'setuptools' in sys.modules else None

        # Create mock setuptools module
        class MockSetuptools:
            def __init__(self, original_setuptools, analyzer_instance):
                self._original = original_setuptools
                self._analyzer = analyzer_instance
                self._captured_extras = {}
                # Copy all attributes from original setuptools
                if original_setuptools:
                    for attr in dir(original_setuptools):
                        if not attr.startswith('_'):
                            setattr(self, attr, getattr(original_setuptools, attr))

            def setup(self, *args, **kwargs):
                # Just capture package information, don't execute setup
                if 'extras_require' in kwargs:
                    self._captured_extras = kwargs['extras_require']
                    # Store captured extras in analyzer instance
                    self._analyzer._captured_extras = self._captured_extras
                # Do nothing - UV will handle real setup later
                pass

            def find_packages(self, *args, **kwargs):
                # Return empty list - we don't need package discovery during analysis
                return []

        # Replace setuptools module with mock
        mock_setuptools = MockSetuptools(self._original_setuptools, self)
        sys.modules['setuptools'] = mock_setuptools

        # Suppress UserWarnings from setuptools
        warnings.filterwarnings("ignore", category=UserWarning, module="setuptools._distutils.dist")

    def _parse_pyproject_toml(self):
        """Parse poetry/pyproject.toml to extract Python and torch requirements"""
        try:
            pyproject_path = os.path.join(self.source_dir, 'poetry', 'pyproject.toml')
            if not os.path.exists(pyproject_path):
                self._log(f"poetry/pyproject.toml not found at: {pyproject_path}")
                return

            self._log(f"Parsing pyproject.toml: {pyproject_path}")

            with open(pyproject_path, 'rb') as f:
                pyproject_data = tomllib.load(f)

            # First, extract Python requirement from pyproject.toml
            pyproject_python_req = None
            if 'tool' in pyproject_data and 'poetry' in pyproject_data['tool']:
                poetry_config = pyproject_data['tool']['poetry']

                if 'dependencies' in poetry_config and 'python' in poetry_config['dependencies']:
                    pyproject_python_req = poetry_config['dependencies']['python']
                    self._log(f"Found Python requirement in pyproject.toml (Poetry format): {pyproject_python_req}")
                else:
                    self._log("No Python requirement found in pyproject.toml dependencies")
            else:
                self._log("No Poetry configuration found in pyproject.toml")

            # Now check if we need to override with hardcoded requirements
            hardcoded_requirement = self._get_hardcoded_python_requirement(pyproject_python_req)
            if hardcoded_requirement:
                self.python_requirement = hardcoded_requirement
                self._log(f"Hardcoded Python requirement for version {self.version}: {self.python_requirement}")
            elif pyproject_python_req:
                # Convert Poetry format to setup.py format
                self.python_requirement = self._convert_poetry_to_setup_format(pyproject_python_req)
                self._log(f"Converted to setup.py format: {self.python_requirement}")

            # Parse poetry.lock for exact torch version
            self._parse_poetry_lock_for_torch()

        except Exception as e:
            self._log(f"Error parsing pyproject.toml: {e}")
            self._log(f"Traceback: {traceback.format_exc()}")

    def _get_hardcoded_python_requirement(self, pyproject_python_req: str | None = None) -> str | None:
        """Get hardcoded Python requirement for specific version ranges"""
        if not self.version:
            return None

        try:
            parsed_version = pkg_version.parse(self.version)

            # Python 3.10 for versions 0.18.1 to 1.0.0 inclusive
            if pkg_version.parse("0.18.1") <= parsed_version <= pkg_version.parse("1.0.0"):
                return ">=3.10,<3.11"

            # For versions 1.1.0 to 4.5.1, check if pyproject.toml has >=3.10,<3.13 or >=3.10,<4
            # If so, return >=3.12,<3.13 to force Python 3.12
            if pkg_version.parse("1.1.0") <= parsed_version <= pkg_version.parse("4.5.1"):
                # Check if pyproject.toml has >=3.10,<4
                if pyproject_python_req and pyproject_python_req in [">=3.10,<4", ">=3.10,<4.0", ">=3.10,<4.0.0"]:
                    self._log(
                        f"Found {pyproject_python_req} in pyproject.toml for version {self.version}, forcing Python 3.12")
                    return ">=3.12,<3.13"

            return None

        except Exception:
            # If version parsing fails, don't hardcode
            return None

    def _parse_poetry_lock_for_torch(self):
        """Parse poetry/poetry.lock to get the exact resolved torch version"""
        try:
            lock_path = os.path.join(self.source_dir, 'poetry', 'poetry.lock')
            if not os.path.exists(lock_path):
                self._log(f"poetry/poetry.lock not found at: {lock_path}")
                return

            self._log(f"Parsing poetry.lock for torch version: {lock_path}")

            with open(lock_path, 'rb') as f:
                lock_data = tomllib.load(f)

            # Find torch package in the lock file
            if 'package' in lock_data:
                for package in lock_data['package']:
                    if package.get('name') == 'torch':
                        torch_version = package.get('version')
                        if torch_version:
                            self._log(f"Found exact torch version in poetry.lock: {torch_version}")
                            self.torch_version = torch_version
                            return

                self._log("torch package not found in poetry.lock")
            else:
                self._log("No 'package' section found in poetry.lock")

        except Exception as e:
            self._log(f"Error parsing poetry.lock: {e}")
            self._log(f"Traceback: {traceback.format_exc()}")

    def _convert_poetry_to_setup_format(self, poetry_requirement: str) -> str:
        """Convert Poetry version requirement format to setup.py format"""
        if not poetry_requirement or poetry_requirement.strip() == "":
            return ""

        requirement = poetry_requirement.strip()

        # Handle Poetry caret requirements (^)
        if requirement.startswith('^'):
            version = requirement[1:]
            try:
                parsed_version = pkg_version.parse(version)
                # ^3.11 means >=3.11,<4.0
                major = parsed_version.release[0]
                return f">={version},<{major + 1}.0"
            except:
                return requirement

        # Handle Poetry tilde requirements (~)
        elif requirement.startswith('~'):
            version = requirement[1:]
            try:
                parsed_version = pkg_version.parse(version)
                # ~3.11 means >=3.11,<3.12
                if len(parsed_version.release) >= 2:
                    major, minor = parsed_version.release[0], parsed_version.release[1]
                    return f">={version},<{major}.{minor + 1}"
                else:
                    return f">={version}"
            except:
                return requirement

        # Handle exact version (no operator)
        elif not any(requirement.startswith(op) for op in ['>=', '<=', '>', '<', '==', '!=']):
            # Check if it looks like a version number
            if any(c.isdigit() for c in requirement):
                return f"=={requirement}"
            else:
                return requirement

        # Already in setup.py format (>=, <=, >, <, ==, !=)
        else:
            return requirement

    def cleanup(self):
        """Cleanup any temporary files or modifications"""
        try:
            # Restore original setuptools module if we modified it
            if hasattr(self, '_original_setuptools'):
                if self._original_setuptools is not None:
                    sys.modules['setuptools'] = self._original_setuptools
                else:
                    # If there was no original setuptools, remove our mock
                    if 'setuptools' in sys.modules:
                        del sys.modules['setuptools']
                self._log("Restored original setuptools module")

        except Exception as e:
            self._log(f"Error during cleanup: {e}")
            self._log(f"Traceback: {traceback.format_exc()}")

    def get_available_extras(self):
        """Get available extras for the current platform."""
        # Only return extras that have descriptions
        descriptions = self.get_extra_descriptions()
        return {k: v for k, v in self.extras.items() if k in descriptions}

    def get_recommended_extras(self, gpu_info=None):
        """
        Get list of recommended extras for the current platform.
        
        :pa
        ram gpu_info: Optional GPU information from platform detection
        :return: List of recommended extra names
        """
        descriptions = self.get_extra_descriptions()
        available_extras = [k for k in self.extras.keys() if k in descriptions]

        # If no GPU info provided, return all available extras
        if gpu_info is None:
            return available_extras

        # Make intelligent recommendations based on GPU capabilities
        recommended = []

        for extra in available_extras:
            if extra == 'gpt4all' and 'gpt4all_cuda' in available_extras:
                # Only recommend gpt4all if CUDA is not available
                if not (gpu_info.has_nvidia and gpu_info.cuda_version):
                    recommended.append(extra)
                    self._log(f"Recommending gpt4all (CPU-only) - no CUDA available")
                else:
                    self._log(
                        f"Not recommending gpt4all (CPU-only) - CUDA available, will recommend gpt4all_cuda instead")
            elif extra == 'gpt4all_cuda':
                # Recommend gpt4all_cuda if CUDA is available
                if gpu_info.has_nvidia and gpu_info.cuda_version:
                    recommended.append(extra)
                    self._log(f"Recommending gpt4all_cuda - CUDA {gpu_info.cuda_version} detected")
                else:
                    self._log(f"Not recommending gpt4all_cuda - no CUDA available")
            elif extra == 'triton_windows':
                # Only recommend triton_windows if NVIDIA GPU is available
                if gpu_info.has_nvidia:
                    recommended.append(extra)
                    self._log(f"Recommending triton_windows - NVIDIA GPU detected")
                else:
                    self._log(f"Not recommending triton_windows - no NVIDIA GPU available")
            else:
                # For all other extras, recommend them
                recommended.append(extra)

        return recommended

    def get_extra_descriptions(self):
        """Get human-readable descriptions for extras."""
        return {
            'bitsandbytes': 'Quantization library for faster inference with reduced memory usage',
            'ncnn': 'High-performance neural network inference framework (Used for ncnn-upscaler image processor)',
            'gpt4all': 'Local large language model support (CPU-only)',
            'gpt4all_cuda': 'CUDA-accelerated GPT4All for NVIDIA GPUs (Linux/Windows)',
            'console_ui_opengl': 'OpenGL accelerated Console UI image viewer.',
            'triton_windows': 'Triton support for Windows (NVIDIA)'
        }

    def get_python_requirement(self):
        """Get Python version requirement."""
        return self.python_requirement

    def get_recommended_python_version(self):
        """Get the recommended Python version to use based on requirements."""
        if not self.python_requirement:
            self._log("No Python requirement found, using default 3.13")
            return "3.13"  # Default fallback

        try:
            spec = specifiers.SpecifierSet(self.python_requirement)
            self._log(f"Parsing Python requirement: '{self.python_requirement}' -> spec: {spec}")

            # Test Python versions from newest to oldest to find the highest compatible
            candidate_versions = ["3.14", "3.13", "3.12", "3.11", "3.10"]

            for version in candidate_versions:
                try:
                    # Create a version object and check if it's compatible with the spec
                    ver = pkg_version.Version(version)
                    if ver in spec:
                        self._log(f"Found compatible Python version: {version}")
                        return version
                    else:
                        self._log(f"Python {version} not compatible with spec {spec}")
                except Exception as e:
                    self._log(f"Error checking Python {version}: {e}")
                    continue

            # If no candidate matches, return the latest stable
            self._log("No compatible Python version found, using fallback 3.13")
            return "3.13"

        except Exception as e:
            self._log(f"Error parsing Python requirements: {e}")
            return "3.13"  # Fallback

    def get_torch_version(self):
        """Get torch version requirement."""
        return self.torch_version
