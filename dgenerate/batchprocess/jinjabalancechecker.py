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


import jinja2
import jinja2.lexer
import re


class JinjaBalanceChecker:
    def __init__(self, env: jinja2.Environment = jinja2.Environment()):
        self.env = env
        self._lexer = jinja2.lexer.Lexer(env)

        # state for control structure recognition
        self._token_stack = []
        self._block_balancing_stack = []
        self._cur_name = None

        # State for incremental lexing
        self._pos = 0
        self._lineno = 1
        self._lex_state_stack = ["root"]
        self._lex_balancing_stack = []
        self._newlines_stripped = 0
        self._line_starting = True
        self._buffer = ''

    def put_source(self, source):
        """Tokenizes the input character-by-character incrementally."""
        self._buffer += source

        for lineno, token_type, token_value in self._tokeniter():
            if token_type == jinja2.lexer.TOKEN_BLOCK_BEGIN:
                self._token_stack.append(jinja2.lexer.TOKEN_BLOCK_BEGIN)

            elif token_type == jinja2.lexer.TOKEN_NAME:
                if self._token_stack and self._token_stack[-1] == jinja2.lexer.TOKEN_BLOCK_BEGIN:
                    self._cur_name = token_value
                    self._token_stack.append(jinja2.lexer.TOKEN_NAME)

            elif token_type == jinja2.lexer.TOKEN_BLOCK_END:
                if self._token_stack and self._token_stack[-1] == jinja2.lexer.TOKEN_NAME:
                    self._handle_block_end()
                    self._token_stack.clear()

    def _tokeniter(self):
        while self._buffer:
            source_length = len(self._buffer)
            matched = False

            for regex, tokens, new_state in self._lexer.rules[self._lex_state_stack[-1]]:
                m = regex.match(self._buffer, self._pos)

                if m is None:
                    continue

                matched = True

                if self._lex_balancing_stack and tokens in (
                        jinja2.lexer.TOKEN_VARIABLE_END,
                        jinja2.lexer.TOKEN_BLOCK_END,
                        jinja2.lexer.TOKEN_LINESTATEMENT_END,
                ):
                    continue

                if isinstance(tokens, tuple):
                    groups = m.groups()

                    if isinstance(tokens, jinja2.lexer.OptionalLStrip):
                        text = groups[0]
                        strip_sign = next(g for g in groups[2::2] if g is not None)

                        if strip_sign == "-":
                            stripped = text.rstrip()
                            self._newlines_stripped = text[len(stripped):].count("\n")
                            groups = [stripped, *groups[1:]]
                        elif (
                                strip_sign != "+"
                                and self._lexer.lstrip_blocks
                                and not m.groupdict().get(jinja2.lexer.TOKEN_VARIABLE_BEGIN)
                        ):
                            l_pos = text.rfind("\n") + 1

                            if l_pos > 0 or self._line_starting:
                                if re.fullmatch(r"\s*", text[l_pos:]):
                                    groups = [text[:l_pos], *groups[1:]]

                    for idx, token in enumerate(tokens):
                        if isinstance(token, jinja2.lexer.Failure):
                            raise token(self._lineno, None)
                        elif token == "#bygroup":
                            for key, value in m.groupdict().items():
                                if value is not None:
                                    yield self._lineno, key, value
                                    self._lineno += value.count("\n")
                                    break
                            else:
                                raise RuntimeError(
                                    f"{regex!r} wanted to resolve the token dynamically"
                                    " but no group matched"
                                )
                        else:
                            data = groups[idx]
                            if data or token not in jinja2.lexer.ignore_if_empty:
                                yield self._lineno, token, data
                            self._lineno += data.count("\n") + self._newlines_stripped
                            self._newlines_stripped = 0

                else:
                    data = m.group()

                    if tokens == jinja2.lexer.TOKEN_OPERATOR:
                        if data == "{":
                            self._lex_balancing_stack.append("}")
                        elif data == "(":
                            self._lex_balancing_stack.append(")")
                        elif data == "[":
                            self._lex_balancing_stack.append("]")
                        elif data in ("}", ")", "]"):
                            if not self._lex_balancing_stack:
                                raise jinja2.exceptions.TemplateSyntaxError(
                                    f"unexpected '{data}'", self._lineno, None, None
                                )
                            expected_op = self._lex_balancing_stack.pop()
                            if expected_op != data:
                                raise jinja2.exceptions.TemplateSyntaxError(
                                    f"unexpected '{data}', expected '{expected_op}'",
                                    self._lineno,
                                    None,
                                    None,
                                )

                    if data or tokens not in jinja2.lexer.ignore_if_empty:
                        yield self._lineno, tokens, data

                    self._lineno += data.count("\n")

                self._line_starting = m.group()[-1:] == "\n"
                pos2 = m.end()

                if new_state is not None:
                    if new_state == "#pop":
                        self._lex_state_stack.pop()
                    elif new_state == "#bygroup":
                        for key, value in m.groupdict().items():
                            if value is not None:
                                self._lex_state_stack.append(key)
                                break
                        else:
                            raise RuntimeError(
                                f"{regex!r} wanted to resolve the new state dynamically"
                                f" but no group matched"
                            )
                    else:
                        self._lex_state_stack.append(new_state)

                elif pos2 == self._pos:
                    raise RuntimeError(
                        f"{regex!r} yielded empty string without stack change"
                    )

                self._pos = pos2
                break

            if not matched:
                if self._pos >= source_length:
                    self._buffer = ''
                    self._pos = 0
                    return

                raise jinja2.exceptions.TemplateSyntaxError(
                    f"unexpected char {self._buffer[self._pos]!r} at {self._pos}", self._lineno, None, None
                )

            # Update buffer and reset position if fully processed
            if self._pos >= source_length:
                self._buffer = ''
                self._pos = 0

    def _handle_block_end(self):
        if self._cur_name:
            if self._cur_name in {'endif', 'endfor', 'endblock', 'endmacro', 'endfilter', 'endcall'}:
                self._balance_end_block(self._cur_name)
            elif self._cur_name in {'if', 'for', 'block', 'macro', 'filter', 'call'}:
                self._block_balancing_stack.append(self._cur_name)
            elif self._cur_name in {'else', 'elif'}:
                pass

    def _balance_end_block(self, end_name):
        expected_start = end_name[3:]

        while self._block_balancing_stack and self._block_balancing_stack[-1] != expected_start:
            self._block_balancing_stack.pop()

        if self._block_balancing_stack:
            self._block_balancing_stack.pop()

    def is_balanced(self):
        return len(self._lex_state_stack) == 1 and len(self._token_stack) == 0 and len(self._block_balancing_stack) == 0
