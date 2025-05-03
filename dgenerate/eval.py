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

import asteval

__doc__ = """
Safe expression parsing with asteval.
"""


def safe_builtins() -> dict:
    """
    Return a dictionary / symbol table of basic python builtins
    that are considered safe and useful with ``asteval``.

    :return: symbol table
    """
    return {
        'abs': abs,
        'all': all,
        'any': any,
        'ascii': ascii,
        'bin': bin,
        'bool': bool,
        'bytearray': bytearray,
        'bytes': bytes,
        'callable': callable,
        'chr': chr,
        'complex': complex,
        'dict': dict,
        'divmod': divmod,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'format': format,
        'frozenset': frozenset,
        'getattr': getattr,
        'hasattr': hasattr,
        'hash': hash,
        'hex': hex,
        'int': int,
        'iter': iter,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'next': next,
        'object': object,
        'oct': oct,
        'ord': ord,
        'pow': pow,
        'range': range,
        'repr': repr,
        'reversed': reversed,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'type': type,
        'zip': zip,
    }


def standard_interpreter(
        symtable: dict | None = None,
        with_listcomp: bool = True,
        with_dictcomp: bool = True,
        with_setcomp: bool = True,
        with_ifexpr: bool = True,
        use_numpy: bool = False
) -> asteval.Interpreter:
    """
    Return a default safe interpreter from ``asteval``.

    Nothing that does not exist in symtable will be usable,
    if you provide no symtable, no functions / variables will be
    present.

    All forms of assignment, import, etc. are disabled.

    :param symtable: Symbol table
    :param with_listcomp: Allow list comprehension?
    :param with_dictcomp: Allow dict comprehension?
    :param with_setcomp: Allow set comprehension?
    :param with_ifexpr: Allow ternary statements?
    :param use_numpy: Import numpy functions directly, without namespace?
    :return: The interpreter
    """
    interpreter = asteval.Interpreter(
        minimal=True,
        with_listcomp=with_listcomp,
        with_dictcomp=with_dictcomp,
        with_setcomp=with_setcomp,
        with_ifexp=with_ifexpr,
        use_numpy=use_numpy,
        builtins_readonly=True,
        symtable=symtable if symtable is not None else dict()
    )

    if 'print' in interpreter.symtable:
        del interpreter.symtable['print']

    return interpreter
