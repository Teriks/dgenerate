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
import ast
import shutil


class ConceptPathParseError(Exception):
    pass


class ConceptPath:
    def __init__(self, model, args):
        self.concept = model
        self.args = args

    def __str__(self):
        return f"{self.concept}: {self.args}"


class ConceptPathParser:

    def __init__(self, concept_name, known_args=None):
        self.known_args = known_args
        self.concept_name = concept_name

    def parse_concept_path(self, strin):
        args = dict()
        parts = strin.split(';')
        parts = iter(parts)
        concept = parts.__next__()
        for i in parts:
            vals = i.split('=', 1)
            if not vals:
                raise ConceptPathParseError(f'Error parsing path arguments for '
                                            f'{self.concept_name} concept "{concept}", Empty argument space, '
                                            f'stray semicolon?')
            if len(vals) == 1:
                raise ConceptPathParseError(f'Error parsing path arguments for '
                                            f'{self.concept_name} concept "{concept}", missing value '
                                            f'assignment for argument {vals[0]}.')
            name = vals[0].strip().lower()
            if self.known_args is not None and name not in self.known_args:
                raise ConceptPathParseError(
                    f'Unknown path argument "{name}" for {self.concept_name} concept "{concept}", '
                    f'valid arguments: {", ".join(list(self.known_args))}')

            if name in args:
                raise ConceptPathParseError(
                    f'Duplicate argument "{name}" provided for {self.concept_name} concept "{concept}".')

            try:
                args[name] = unquote(vals[1])
            except SyntaxError as e:
                raise ConceptPathParseError(f'Syntax Error parsing argument {name} for '
                                            f'{self.concept_name} concept "{concept}": {e}')
        return ConceptPath(concept, args)


def oxford_comma(elements, conjunction):
    cnt = len(elements)
    elements = (str(i) for i in elements)
    if cnt == 1:
        return next(elements)
    if cnt == 2:
        return next(elements) + f' {conjunction} ' + next(elements)
    output = ''
    for idx, element in enumerate(elements):
        if idx == cnt - 1:
            output += f', {conjunction} {element}'
        elif idx == 0:
            output += element
        else:
            output += f', {element}'
    return output


def long_text_wrap_width():
    return min(shutil.get_terminal_size(fallback=(150, 0))[0], 150)


def underline(msg):
    return msg + '\n' + ('=' * min(len(max(msg.split('\n'), key=len)), long_text_wrap_width()))


def quote(strin):
    return f'"{strin}"'


def unquote(strin):
    strin = strin.strip(' ')
    if strin.startswith('"') or strin.startswith('\''):
        return str(ast.literal_eval('r' + strin))
    else:
        # Is an unquoted string
        return str(strin.strip(' '))


def dashdown(strin):
    return strin.replace('-', '_')


def dashup(strin):
    return strin.replace('_', '-')


def contains_space(strin):
    return any(c.isspace() for c in strin)


def quote_spaces(list_of_strin):
    vals = [] if isinstance(list_of_strin, list) else tuple()
    for v in list_of_strin:
        if isinstance(v, list):
            vals.append(quote_spaces(v))
        elif isinstance(v, tuple):
            vals.append(tuple(quote_spaces(v)))
        elif contains_space(v):
            vals.append(quote(v))
        else:
            vals.append(v)
    return vals


def justify_left(strin):
    return '\n'.join(line.strip() if not line.isspace() else line for line in strin.split('\n'))


def debug_format_args(args_dict, value_transformer=None):
    def _value_transformer(key, value):
        if value_transformer is not None:
            return value_transformer(key, value)
        return value

    return str(
        {k: str(_value_transformer(k, v)) if len(str(_value_transformer(k, v))) < 256
        else v.__class__.__name__ for k, v in args_dict.items()})
