// Copyright (c) 2026, Teriks
//
// dgenerate is distributed under the following BSD 3-Clause License
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
//    the documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

lexer grammar ConfigLexer;

tokens { JOPEN_EXPR, JCLOSE_EXPR, JCLOSE, STRING }

@lexer::members {
at_line_start = True
quote = None

def _toggle_quote(self, mark):
    self.quote = None if self.quote == mark else (mark if self.quote is None else self.quote)
    self.at_line_start = False

def _closer_ahead(self):
    chars = []
    index = 1
    while index <= 24:
        code = self._input.LA(index)
        if code < 0 or code in (10, 13):
            break
        chars.append(chr(code))
        index += 1
    rest = ''.join(chars).lstrip()
    for word in ('endfor', 'endif', 'elif', 'else', 'endset', 'endfilter', 'endwith'):
        if rest.startswith(word):
            nxt = rest[len(word):len(word) + 1]
            return nxt in ('', ' ', '\t', '%')
    return False
}

CLOSER: '{%' {self._closer_ahead()}? {self.at_line_start = False;} -> pushMode(JINJA);
JOPEN: '{%' {self.at_line_start = False;} -> pushMode(JINJA);
JEXPR: '{{' {self.at_line_start = False;} -> type(JOPEN_EXPR), pushMode(JINJA);

LINE_DASH: {self.at_line_start}? '-' {self.at_line_start = False;};

LEADING_WS: {self.at_line_start}? [ \t]+ -> skip;

COMMENT: {self.quote is None}? '#' ~[\r\n]* {self.at_line_start = False;};

QUOTE
    : '"' {self._toggle_quote('"');}
    | '\'' {self._toggle_quote("'");}
    ;

HASH_CHAR: {self.quote is not None}? '#' {self.at_line_start = False;};

BACKSLASH_CONT: '\\' [ \t]* '\r'? '\n' {self.at_line_start = True; self.quote = None;} -> skip;

NEWLINE: '\r'? '\n' {self.at_line_start = True; self.quote = None;};

TEXT
    : {not self.at_line_start}? (
          ~["'#{\\\r\n]
        | '{' ~[{%]
        | '\\' ~[\r\n]
      )+ {self.at_line_start = False;}
    | {self.at_line_start}? (
          ~["'#{\\\r\n \t\-]
        | '{' ~[{%]
        | '\\' ~[\r\n]
      )+ {self.at_line_start = False;}
    ;

mode JINJA;

JCLOSE_STMT: '%}' -> type(JCLOSE), popMode;
JCLOSE_EXPR: '}}' -> type(JCLOSE_EXPR), popMode;

JWS: [ \t\r\n]+ -> skip;

FOR: 'for';
IN: 'in';
IF: 'if';
ELIF: 'elif';
ELSE: 'else';
ENDIF: 'endif';
ENDFOR: 'endfor';
SET: 'set';
ENDSET: 'endset';
FILTER: 'filter';
ENDFILTER: 'endfilter';
WITH: 'with';
ENDWITH: 'endwith';
AND: 'and';
OR: 'or';
NOT: 'not';
IS: 'is';

EQ: '==';
ASSIGN: '=';
NEQ: '!=';
LE: '<=';
GE: '>=';
LT: '<';
GT: '>';

STAR2: '**';
SLASH2: '//';
PIPE: '|';
TILDE: '~';
PLUS: '+';
MINUS: '-';
STAR: '*';
SLASH: '/';
PERCENT: '%';
DOT: '.';
COMMA: ',';
COLON: ':';
LPAREN: '(';
RPAREN: ')';
LBRACK: '[';
RBRACK: ']';
LBRACE: '{';
RBRACE: '}';

NUMBER: [0-9]+ ('.' [0-9]+)?;

ID: [A-Za-z_] [A-Za-z0-9_]*;

JSTRING_DQ: '"' ('\\' . | ~["\\\r\n])* '"' -> type(STRING);
JSTRING_SQ: '\'' ('\\' . | ~['\\\r\n])* '\'' -> type(STRING);
