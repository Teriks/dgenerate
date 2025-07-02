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

import dgenerate.textprocessing as _textprocessing


def yolo_filters_parse(
        class_filter: int | str | list | tuple | set | None,
        index_filter: int | list | tuple | set | None,
        argument_error: typing.Callable
) -> tuple[set, set]:
    """
    Parse YOLO class and index filter arguments and return sets of values

    :param class_filter: class filter
    :param index_filter: index filter
    :param argument_error: raise on argument error
    :return: tuple of sets of class and index filter values
    """
    def filter_t(i):
        if _textprocessing.is_quoted(i):
            return _textprocessing.unquote(i)
        if isinstance(i, str):
            try:
                val = int(i)
            except ValueError:
                return i  # It's a non-numeric string
            
            if val < 0:
                raise argument_error('class-filter ID values must be greater than or equal to 0.')
            return val
        return i

    # Parse class filter
    class_filter_out = None
    if class_filter is not None:
        if not isinstance(class_filter, (list, tuple, set)):
            if isinstance(class_filter, str) and ',' in class_filter:
                try:
                    class_filter = _textprocessing.tokenized_split(class_filter, separator=',')
                    class_filter = [filter_t(i) for i in class_filter]
                except Exception as e:
                    raise argument_error(f'Argument "class-filter": {e}')
            else:
                class_filter = [class_filter]

        class_filter_out = set()
        for item in class_filter:
            # Check for negative integers in direct input
            if isinstance(item, int) and item < 0:
                raise argument_error('class-filter ID values must be greater than or equal to 0.')
            class_filter_out.add(item)

    # Parse index filter
    index_filter_out = None
    if index_filter is not None:
        if isinstance(index_filter, int):
            if index_filter < 0:
                raise argument_error('index-filter values must be greater than or equal to 0.')
            index_filter = [index_filter]

        index_filter_out = set()
        try:
            for i in index_filter:
                val = int(i)
                if val < 0:
                    raise argument_error('index-filter values must be greater than or equal to 0.')
                index_filter_out.add(val)
        except ValueError:
            raise argument_error('index-filter values must be integers.')
        except TypeError:
            raise argument_error('index-filter values must be iterable.')

    return class_filter_out, index_filter_out
