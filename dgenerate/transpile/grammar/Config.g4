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

// dgenerate config, including the Jinja these configs actually use:
// {% if %} {% elif %} {% else %} {% endif %} {% for %} {% endfor %}
// {% set %} {% endset %} {% filter %} {% endfilter %} {% with %} {% endwith %}
// and {{ expressions }} with calls, attributes, filters, tuples, and/or/not, and comparisons.
parser grammar Config;

options { tokenVocab = ConfigLexer; }

config: element* EOF;

element
    : NEWLINE
    | stmt
    ;

// A physical line, plus continuation lines that start with '-'.
// A trailing backslash is consumed by the lexer and does not produce a newline.
stmt: piece+ continuation* (NEWLINE | EOF);

continuation: NEWLINE LINE_DASH piece*;

piece
    : TEXT
    | QUOTE
    | HASH_CHAR
    | COMMENT
    | LINE_DASH
    | jinjaExpr
    | jinjaFor
    | jinjaIf
    | jinjaSet
    | jinjaFilter
    | jinjaWith
    ;

jinjaExpr: JOPEN_EXPR jExpr JCLOSE_EXPR;

// Inline loops are pieces on the same line. Block loops are later lines.
jinjaFor
    : JOPEN FOR jTargets IN jExpr JCLOSE
      piece*
      element*
      forElse?
      CLOSER ENDFOR JCLOSE
    ;

forElse: CLOSER ELSE JCLOSE element*;

jinjaIf
    : JOPEN IF jExpr JCLOSE ifBody
      elifPart*
      elsePart?
      CLOSER ENDIF JCLOSE
    ;

ifBody: element*;

elifPart: CLOSER ELIF jExpr JCLOSE element*;

elsePart: CLOSER ELSE JCLOSE element*;

// {% set name = expr %} and {% set name %}...{% endset %}
jinjaSet
    : JOPEN SET jTargets ASSIGN jExpr JCLOSE
    | JOPEN SET jTargets JCLOSE piece* element* CLOSER ENDSET JCLOSE
    ;

// {% filter name %}...{% endfilter %} renders the body, then that text is config.
jinjaFilter
    : JOPEN FILTER jPostfix JCLOSE piece* element* CLOSER ENDFILTER JCLOSE
    ;

// {% with name = expr %}...{% endwith %}
jinjaWith
    : JOPEN WITH jAssign (COMMA jAssign)* JCLOSE piece* element* CLOSER ENDWITH JCLOSE
    ;

jAssign: jTargets ASSIGN jExpr;

jTargets: ID (COMMA ID)*;

jExpr: jOr (IF jOr ELSE jExpr)?;

jOr: jAnd (OR jAnd)*;

jAnd: jNot (AND jNot)*;

jNot: NOT jNot | jComparison;

jComparison: jConcat (jCmpOp jConcat?)*;

jCmpOp: EQ | NEQ | LE | GE | LT | GT | IN | NOT IN | IS NOT? ID;

jConcat: jSum (TILDE jSum)*;

jSum: jTerm ((PLUS | MINUS) jTerm)*;

jTerm: jFactor ((STAR | SLASH | SLASH2 | PERCENT) jFactor)*;

jFactor: (PLUS | MINUS) jFactor | jPower;

jPower: jPostfix (STAR2 jFactor)?;

jPostfix: jPrimary jSuffix*;

jSuffix
    : DOT ID
    | LBRACK jExpr RBRACK
    | LPAREN jArgs? RPAREN
    | PIPE ID (LPAREN jArgs? RPAREN)?
    ;

jPrimary
    : ID
    | NUMBER
    | STRING
    | LPAREN jParenItems? RPAREN
    | LBRACK jList? RBRACK
    | LBRACE jDict? RBRACE
    ;

// (1 + 2) is grouping. (1, 2) and (1,) are tuples. () is an empty tuple.
jParenItems: jExpr (COMMA jExpr)* COMMA?;

jList: jExpr (COMMA jExpr)* COMMA?;

jDict: jDictPair (COMMA jDictPair)* COMMA?;

jDictPair: jExpr COLON jExpr;

jArgs: jArg (COMMA jArg)* COMMA?;

jArg: ID ASSIGN jExpr | jExpr;
