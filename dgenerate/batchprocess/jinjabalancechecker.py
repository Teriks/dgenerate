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


class JinjaBalanceChecker:
    def __init__(self, env: jinja2.Environment = jinja2.Environment()):
        self.source = ''
        self.env = env
        self.stack = []
        self.balance_stack = []
        self.cur_name = None

    def put_source(self, source):
        """Tokenizes the input character-by-character incrementally."""

        self.stack = []
        self.balance_stack = []
        self.cur_name = None

        self.source += source

        try:
            for token in self.env.lex(self.source):
                token_type = token[1]
                token_value = token[2]

                if token_type == jinja2.lexer.TOKEN_BLOCK_BEGIN:
                    self.stack.append(jinja2.lexer.TOKEN_BLOCK_BEGIN)

                elif token_type == jinja2.lexer.TOKEN_NAME:
                    if self.stack and self.stack[-1] == jinja2.lexer.TOKEN_BLOCK_BEGIN:
                        self.cur_name = token_value
                        self.stack.append(jinja2.lexer.TOKEN_NAME)

                elif token_type == jinja2.lexer.TOKEN_BLOCK_END:
                    if self.stack and self.stack[-1] == jinja2.lexer.TOKEN_NAME:
                        self._handle_block_end()
                        self.stack.clear()

        except jinja2.lexer.TemplateSyntaxError:
            pass

    def _handle_block_end(self):
        if self.cur_name:
            if self.cur_name in {'endif', 'endfor', 'endblock', 'endmacro', 'endfilter', 'endcall'}:
                self._balance_end_block(self.cur_name)
            elif self.cur_name in {'if', 'for', 'block', 'macro', 'filter', 'call'}:
                self.balance_stack.append(self.cur_name)
            elif self.cur_name in {'else', 'elif'}:
                # Handle 'else' and 'elif' without popping the 'if' block
                pass

    def _balance_end_block(self, end_name):
        expected_start = end_name[3:]  # Remove 'end' prefix

        while self.balance_stack and self.balance_stack[-1] != expected_start:
            self.balance_stack.pop()

        if self.balance_stack:
            self.balance_stack.pop()

    def is_balanced(self):
        return len(self.stack) == 0 and len(self.balance_stack) == 0