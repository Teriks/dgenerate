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

__doc__ = """
Utilities for writing directives.
"""

import argparse
import typing

import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing


class DirectiveArgumentParser(argparse.ArgumentParser):
    """
    An argparse argument parser which does not call ``sys.exit`` and provides
    the return code from parsing as a queryable value instead.
    """

    class _ExitException(Exception):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_code = None
        self.formatter_class = _textprocessing.ArgparseParagraphFormatter

    def parse_args(
            self,
            args: typing.Sequence[str] | None = None,
            namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        self.return_code = None
        try:
            return super().parse_args(args, namespace)
        except self._ExitException:
            return argparse.Namespace()

    def parse_known_args(
            self, args: typing.Sequence[str] | None = None,
            namespace: argparse.Namespace | None = None
    ) -> tuple[argparse.Namespace, list[str]]:
        self.return_code = None
        try:
            return super().parse_known_args(args, namespace)
        except self._ExitException:
            return argparse.Namespace(), []

    def parse_intermixed_args(
            self,
            args: typing.Sequence[str] | None = None,
            namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        self.return_code = None
        try:
            return super().parse_intermixed_args(args, namespace)
        except self._ExitException:
            return argparse.Namespace()

    def parse_known_intermixed_args(
            self, args: typing.Sequence[str] | None = None,
            namespace: argparse.Namespace | None = None
    ) -> tuple[argparse.Namespace, list[str]]:
        self.return_code = None
        try:
            return super().parse_known_intermixed_args(args, namespace)
        except self._ExitException:
            return argparse.Namespace(), []

    def exit(self, status=0, message=None):
        if self.return_code is not None:
            return
        if message is not None:
            if status != 0:
                _messages.error(message.rstrip())
            else:
                _messages.log(message.rstrip())
        self.return_code = status
        raise self._ExitException()
