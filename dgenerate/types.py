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

import dgenerate.prompt as _prompt

Size = typing.Tuple[int, int]
Sizes = typing.List[Size]
OptionalSize = typing.Optional[Size]
OptionalSizes = typing.Optional[Sizes]
Coordinate = typing.Tuple[int, int]
OptionalCoordinate = typing.Optional[Coordinate]
CoordinateList = typing.List[Coordinate]
OptionalCoordinateList = typing.Optional[CoordinateList]
Paths = typing.List[str]
OptionalPaths = typing.Optional[Paths]
Path = str
Name = str
OptionalPath = typing.Optional[Path]
OptionalName = typing.Optional[Name]
Integer = int
Integers = typing.List[int]
OptionalInteger = typing.Optional[int]
OptionalIntegers = typing.Optional[Integers]
Float = float
Floats = typing.List[float]
OptionalFloat = typing.Optional[float]
OptionalFloats = typing.Optional[Floats]
Version = typing.Tuple[int, int, int]
OptionalPrompt = typing.Optional[_prompt.Prompt]
Prompts = typing.List[_prompt.Prompt]
OptionalPrompts = typing.Optional[Prompts]
OptionalString = typing.Optional[str]


def is_type_or_optional(hinted_type, comparison_type):
    if hinted_type == comparison_type:
        return True

    origin = typing.get_origin(hinted_type)

    if origin == comparison_type:
        return True
    if origin == typing.Union:
        union_args = typing.get_args(hinted_type)
        if len(union_args) == 2:
            for a in union_args:
                if typing.get_origin(a) == comparison_type:
                    return True
    return False
