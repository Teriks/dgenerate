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

"""
Shared type definitions for the dgenerate network installer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InstallationInfo:
    """Information about a dgenerate installation."""

    install_base: str
    venv_dir: str
    scripts_dir: str
    dgenerate_exe: str
    installer_type: str = "uv"
    version: Optional[str] = None
    extras: Optional[list[str]] = None


@dataclass
class ExistingInstallation:
    """Information about an existing dgenerate installation."""

    exists: bool
    installer_type: str = "uv"
    path: Optional[str] = None
    version: Optional[str] = None
    installation_info: Optional[InstallationInfo] = None


@dataclass
class InstallationResult:
    """Result of an installation operation."""

    success: bool
    desktop_shortcut_created: bool = False
    installation_info: Optional[InstallationInfo] = None
    error: Optional[str] = None

    @classmethod
    def success_result(
            cls,
            installation_info: InstallationInfo,
            desktop_shortcut_created: bool = False
    ) -> 'InstallationResult':
        """Create a successful installation result."""
        return cls(
            success=True,
            desktop_shortcut_created=desktop_shortcut_created,
            installation_info=installation_info
        )

    @classmethod
    def failure_result(cls, error: str) -> 'InstallationResult':
        """Create a failed installation result."""
        return cls(
            success=False,
            error=error
        )
