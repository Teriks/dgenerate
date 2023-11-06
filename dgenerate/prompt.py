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
import typing


class Prompt:
    """
    Represents a combined positive and optional negative prompt split by a delimiter character.
    """

    def __init__(self,
                 positive: typing.Optional[str] = None,
                 negative: typing.Optional[str] = None,
                 delimiter: str = ';'):
        """
        :param positive: positive prompt component
        :param negative: negative prompt component
        :param delimiter: delimiter for stringification
        """
        self.positive = positive
        self.negative = negative
        self.delimiter = delimiter

    def __str__(self):
        if self.positive and self.negative:
            return f'{self.positive}{self.delimiter} {self.negative}'
        elif self.positive:
            return self.positive
        else:
            return ''

    def __repr__(self):
        return f"'{str(self)}'"

    @staticmethod
    def parse(value: str, delimiter=';') -> 'Prompt':
        """
        Parse the positive and negative prompt from a string and return a prompt object.

        :param value: the string
        :param delimiter: The prompt delimiter character

        :raise: :py:class:`ValueError` if value is None

        :return: :py:class:`.Prompt` (returns self)
        """
        if value is None:
            raise ValueError('Input string may not be None.')

        parse = value.split(delimiter, 1)
        if len(parse) == 1:
            positive = parse[0].strip()
            negative = None
        elif len(parse) == 2:
            positive = parse[0].strip()
            negative = parse[1].strip()
        else:
            positive = None
            negative = None

        return Prompt(positive=positive,
                      negative=negative,
                      delimiter=delimiter)
