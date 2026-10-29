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

import json
import os
import re
import socket
import sys
import typing

import numpy as np

from dgenerate.assistant.catalog import (
    CHAT_MODELS, DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, is_downloaded, resolve_sampling)
from dgenerate.assistant.link_tools import LINK_TOOL_DEFINITIONS, redact_url, run_tool

# Qwen3-Embedding scores 1 to 5 percent lower on retrieval without a query instruction.
_QUERY_INSTRUCTION = (
    'Given a request for a dgenerate config script, retrieve the documentation, '
    'option help, and example configs needed to write it'
)

_THINK_BLOCK = re.compile(
    r'<(?:think|redacted_thinking)>\s*(.*?)\s*</(?:think|redacted_thinking)>',
    re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r'<(?:think|redacted_thinking)>', re.IGNORECASE)
_CONFIG_MARK = re.compile(r'```|^[ \t]*[\w.-]+/[\w.-]+', re.MULTILINE)


def answer_text(content: str, reasoning: str = '') -> str:
    """
    The part of a reply that can contain the config.

    A finished thought is skipped when the answer after it has the config.
    An unfinished thought is not a config unless a config is already in it.
    """
    blocks = [block.strip() for block in _THINK_BLOCK.findall(content)]
    rest = _THINK_BLOCK.sub('', content).strip()
    if _THINK_OPEN.search(rest):
        if _CONFIG_MARK.search(rest):
            return rest
        if _CONFIG_MARK.search(reasoning):
            return reasoning
        return ''
    if _CONFIG_MARK.search(rest):
        return rest
    for block in blocks:
        if _CONFIG_MARK.search(block):
            return block
    if _CONFIG_MARK.search(reasoning):
        return reasoning
    return rest


def reply_token_limit(n_ctx: int, requested: int = 0) -> int:
    """
    How many tokens the reply may use.

    ``requested`` of 0 means no cap. Generation then stops when the context window
    is full. That window is already reserved, so this does not use more memory.
    A positive request is a ceiling, and it is not larger than the window.
    """
    if requested > 0:
        return min(requested, n_ctx)
    return -1


def _chunk_dict(chunk) -> dict | None:
    if isinstance(chunk, (bytes, bytearray)):
        chunk = chunk.decode('utf-8', errors='replace')
    if isinstance(chunk, str):
        chunk = chunk.strip()
        if chunk.startswith('data:'):
            chunk = chunk[5:].strip()
        if not chunk or chunk == '[DONE]':
            return None
        try:
            chunk = json.loads(chunk)
        except json.JSONDecodeError:
            return None
    return chunk if isinstance(chunk, dict) else None


class _ThoughtDisplay:
    """
    Streams the thought to the status stream and keeps the answer for the config.
    """

    _CLOSERS = ('</' + 'think>', '</' + 'redacted_thinking>')

    def __init__(self, thinking: bool, out):
        self._thinking = thinking
        self._out = out
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._carry = ''
        self._closed = not thinking
        self._saw_reasoning = False
        self._pending = ''
        self._announced = False

    def _emit(self, text: str) -> None:
        if self._thinking and not self._announced:
            self._out.write('Thinking:\n')
            self._announced = True
        self._pending += text
        while True:
            newline = self._pending.find('\n')
            if newline != -1:
                self._out.write(self._pending[:newline + 1])
                self._pending = self._pending[newline + 1:]
                continue
            if len(self._pending) >= 120:
                self._out.write(self._pending[:120] + '\n')
                self._pending = self._pending[120:]
                continue
            break
        self._out.flush()

    def _flush_pending(self) -> None:
        if self._pending:
            self._out.write(self._pending + '\n')
            self._pending = ''
            self._out.flush()

    def add(self, reasoning: str, content: str) -> None:
        if reasoning:
            self._saw_reasoning = True
            self._reasoning.append(reasoning)
            self._emit(reasoning)
        if not content:
            return
        self._content.append(content)
        if self._closed or self._saw_reasoning:
            return
        self._carry += content
        lower = self._carry.lower()
        for closer in self._CLOSERS:
            at = lower.find(closer)
            if at == -1:
                continue
            self._emit(self._carry[:at] + '\n')
            self._carry = ''
            self._closed = True
            return
        hold = 24
        if len(self._carry) > hold:
            self._emit(self._carry[:-hold])
            self._carry = self._carry[-hold:]

    def finish(self) -> str:
        if self._thinking and self._carry and not self._closed:
            self._emit(self._carry)
            self._carry = ''
        self._flush_pending()
        return answer_text(''.join(self._content), ''.join(self._reasoning)).strip()


class ModelError(Exception):
    pass


def resolve_gguf(spec: str, offline: bool = False) -> str:
    """
    Return a local path for a GGUF model.

    :param spec: A path to a ``.gguf`` file, or ``org/repo/file.gguf`` on Hugging Face.
    :param offline: Only look in the Hugging Face cache.
    """
    if os.path.isfile(spec):
        return os.path.abspath(spec)

    parts = spec.replace('\\', '/').split('/')
    if len(parts) < 3 or not spec.lower().endswith('.gguf'):
        raise ModelError(
            f'Model "{spec}" is not a .gguf file on disk or an org/repo/file.gguf Hugging Face path.')

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download('/'.join(parts[:2]), '/'.join(parts[2:]), local_files_only=offline)
    except Exception as e:
        raise ModelError(f'Could not download "{spec}": {e}') from e


def _free_localhost_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _create_server(model_path: str, n_ctx: int, gpu_layers: int, verbose: bool, embedding: bool):
    try:
        import xllamacpp
    except (ImportError, OSError) as e:
        raise ModelError(
            'xllamacpp is not installed. Install it with: pip install dgenerate[xllamacpp]') from e

    params = xllamacpp.CommonParams()
    params.model.path = model_path
    params.n_ctx = n_ctx
    params.n_gpu_layers = gpu_layers
    params.n_parallel = 1
    params.hostname = '127.0.0.1'
    params.port = _free_localhost_port()
    # llama.cpp defaults to info (3), which prints model load and timing lines on stderr. 1 keeps errors.
    params.verbosity = 3 if verbose else 1
    params.show_timings = verbose

    if embedding:
        params.embedding = True
        params.pooling_type = xllamacpp.llama_pooling_type.LLAMA_POOLING_TYPE_LAST
        # Embedding inputs must fit in one micro-batch.
        params.n_batch = n_ctx
        params.n_ubatch = n_ctx
    else:
        params.use_jinja = True

    try:
        return xllamacpp.Server(params)
    except Exception as e:
        raise ModelError(f'Could not load "{model_path}": {e}') from e


def _check_response(response: dict, what: str) -> dict:
    if isinstance(response, dict) and 'error' in response:
        error = response['error']
        message = error.get('message', error) if isinstance(error, dict) else error
        raise ModelError(f'{what} failed: {message}')
    return response


def _parse_completion_response(raw) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode('utf-8', errors='replace')
    if isinstance(raw, str):
        raw = json.loads(raw)
    return _check_response(raw, 'Chat completion')


def _message_text(message: dict) -> tuple[str, str]:
    content = message.get('content') or ''
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'text':
                parts.append(part.get('text') or '')
            elif isinstance(part, str):
                parts.append(part)
        content = ''.join(parts)
    reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
    return str(content), str(reasoning)


def _submit_config(arguments: str | dict) -> tuple[bool, str, int]:
    from dgenerate.assistant.check import consider_submission
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return False, f'Invalid tool arguments JSON: {e}', 0
    config = ((arguments or {}).get('config') or '')
    return consider_submission(config)


def _tool_log_args(arguments: str) -> str:
    text = (arguments or '').strip()
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and data.get('url'):
        data = dict(data)
        data['url'] = redact_url(str(data['url']))
        text = json.dumps(data, ensure_ascii=False)
    if len(text) <= 200:
        return text
    return text[:200] + '...'


def _error_text(failure) -> str:
    if isinstance(failure, dict):
        return str(failure.get('message', failure))
    return str(failure)


def _is_context_overflow(failure) -> bool:
    text = _error_text(failure)
    return 'exceeds the available context' in text or 'larger than the max context size' in text


def _tools_rejected(failure) -> bool:
    """Whether a server error means this build cannot accept a tools request."""
    if _is_context_overflow(failure):
        return False
    text = _error_text(failure).lower()
    return any(token in text for token in (
        'tool_choice', 'tool call', 'tool_call', 'tools', 'function calling'))


def _warn_tools(reason) -> None:
    sys.stderr.write(f'warning: link tools unavailable ({_error_text(reason)}); continuing without them.\n')
    sys.stderr.flush()


class Embedder:
    """
    Qwen3-Embedding served through xllamacpp.
    """

    def __init__(self, model_path: str, n_ctx: int = 8192, gpu_layers: int = -1, verbose: bool = False):
        self.model_path = model_path
        self.max_chars = n_ctx * 2
        self._server = _create_server(model_path, n_ctx, gpu_layers, verbose, embedding=True)

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = _check_response(
            self._server.handle_embeddings({'input': [t[:self.max_chars] for t in texts]}),
            'Embedding')
        data = sorted(response['data'], key=lambda d: d['index'])
        vectors = np.asarray([d['embedding'] for d in data], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def embed_documents(self, texts: list[str], batch_size: int = 8, progress=None) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), batch_size):
            vectors.append(self._embed(texts[start:start + batch_size]))
            if progress:
                progress(min(start + batch_size, len(texts)), len(texts))
        return np.concatenate(vectors, axis=0)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([f'Instruct: {_QUERY_INSTRUCTION}\nQuery:{text}'])[0]


class ChatModel:
    """
    A Qwen3.8 chat model served through xllamacpp.
    """

    def __init__(self,
                 model_path: str,
                 n_ctx: int = 32768,
                 gpu_layers: int = -1,
                 think: bool = False,
                 effort: str = 'medium',
                 verbose: bool = False):
        self.model_path = model_path
        self.think = think
        self.effort = effort if effort in ('low', 'medium', 'xhigh') else 'medium'
        self._server = _create_server(model_path, n_ctx, gpu_layers, verbose, embedding=False)

    def complete(self,
                 messages: list[dict],
                 max_tokens: int = -1,
                 temperature: float | None = None,
                 top_p: float | None = None,
                 top_k: int | None = None,
                 min_p: float | None = None,
                 presence_penalty: float | None = None,
                 repeat_penalty: float | None = None,
                 link_tools: bool = False,
                 tools: list | None = None,
                 on_tool: typing.Callable[[str], None] | None = None,
                 max_tool_rounds: int = 6,
                 _shrink: bool = False) -> str:
        # None keeps the Qwen3.8 recommendation for this thinking mode.
        sampling = resolve_sampling(
            self.think,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty)
        self.config_was_submitted = False
        self.submitted_invocations = 0
        if tools is None and link_tools:
            tools = [item for item in LINK_TOOL_DEFINITIONS
                     if (item.get('function') or {}).get('name') == 'fetch_link']
        active = tools
        if active:
            return self._complete_with_link_tools(
                messages,
                max_tokens=max_tokens,
                sampling=sampling,
                on_tool=on_tool,
                max_tool_rounds=max_tool_rounds,
                tools=active,
                _shrink=_shrink)
        return self._complete_streaming(
            messages,
            max_tokens=max_tokens,
            sampling=sampling,
            _shrink=_shrink)

    def _complete_streaming(self,
                            messages: list[dict],
                            max_tokens: int,
                            sampling: dict,
                            _shrink: bool = False) -> str:
        template_kwargs = {'enable_thinking': self.think}
        if self.think:
            template_kwargs['reasoning_effort'] = self.effort
        body = {
            'messages': messages,
            'max_tokens': max_tokens,
            **sampling,
            'chat_template_kwargs': template_kwargs,
            'stream': True,
        }
        display = _ThoughtDisplay(self.think, sys.stderr)
        error = {}

        def on_chunk(chunk):
            data = _chunk_dict(chunk)
            if not data:
                return
            if 'error' in data:
                error['value'] = data['error']
                return
            choices = data.get('choices') or []
            if not choices:
                return
            delta = choices[0].get('delta') or {}
            display.add(delta.get('reasoning_content') or delta.get('reasoning') or '',
                        delta.get('content') or '')

        self._server.handle_chat_completions(body, on_chunk)
        if 'value' in error:
            return self._raise_or_shrink(
                messages, max_tokens, sampling, _shrink, error['value'],
                link_tools=False)
        return display.finish()

    def _complete_with_link_tools(self,
                                  messages: list[dict],
                                  max_tokens: int,
                                  sampling: dict,
                                  on_tool: typing.Callable[[str], None] | None,
                                  max_tool_rounds: int,
                                  tools: list,
                                  _shrink: bool) -> str:
        # Tool turns are appended to the caller's list so a later repair still has the fetch.
        template_kwargs = {'enable_thinking': self.think}
        if self.think:
            template_kwargs['reasoning_effort'] = self.effort
        for _round in range(max_tool_rounds):
            body = {
                'messages': messages,
                'max_tokens': max_tokens,
                **sampling,
                'chat_template_kwargs': template_kwargs,
                'tools': tools,
                'tool_choice': 'auto',
                'stream': False,
            }
            try:
                raw = self._server.handle_chat_completions(body)
            except Exception as e:
                if _round == 0 and _tools_rejected(e):
                    _warn_tools(e)
                    return self._complete_streaming(messages, max_tokens, sampling, _shrink=_shrink)
                raise ModelError(f'Chat completion failed: {e}') from e
            try:
                response = _parse_completion_response(raw)
            except json.JSONDecodeError as e:
                raise ModelError(f'Chat completion failed: {e}') from e
            except ModelError as e:
                if _is_context_overflow(e):
                    return self._raise_or_shrink(
                        messages, max_tokens, sampling, _shrink, e,
                        tools=tools, on_tool=on_tool, max_tool_rounds=max_tool_rounds)
                if _round == 0 and _tools_rejected(e):
                    _warn_tools(e)
                    return self._complete_streaming(messages, max_tokens, sampling, _shrink=_shrink)
                raise
            choices = response.get('choices') or []
            if not choices:
                raise ModelError('Chat completion returned no choices')
            message = choices[0].get('message') or {}
            tool_calls = message.get('tool_calls') or []
            content, reasoning = _message_text(message)
            if self.think and reasoning.strip():
                sys.stderr.write('Thinking:\n')
                sys.stderr.write(reasoning.strip() + '\n')
                sys.stderr.flush()
            if not tool_calls:
                return answer_text(content, reasoning).strip()
            messages.append({'role': 'assistant', 'content': message.get('content') or None,
                             'tool_calls': tool_calls})
            submitted = None
            for call in tool_calls:
                fn = call.get('function') or {}
                name = fn.get('name') or ''
                args = fn.get('arguments') or ''
                if on_tool:
                    on_tool(f'{name} {_tool_log_args(args)}')
                if name == 'submit_config':
                    accepted, payload, invocations = _submit_config(args)
                    result = 'Accepted.' if accepted else payload
                    if accepted:
                        submitted = payload
                        self.config_was_submitted = True
                        self.submitted_invocations = invocations
                else:
                    result = run_tool(name, args)
                if on_tool:
                    on_tool(f'  -> {_truncate_tool_result(result)}')
                messages.append({
                    'role': 'tool',
                    'tool_call_id': call.get('id') or '',
                    'content': result,
                })
            if submitted is not None:
                return submitted
        raise ModelError(f'Tool loop exceeded {max_tool_rounds} rounds without a final answer')

    def _raise_or_shrink(self, messages, max_tokens, sampling, _shrink, failure, *,
                         link_tools: bool = False,
                         tools: list | None = None,
                         on_tool: typing.Callable[[str], None] | None = None,
                         max_tool_rounds: int = 6):
        message = _error_text(failure)
        if _is_context_overflow(failure):
            if len(messages) > 2:
                last_user = next((m for m in reversed(messages) if m['role'] == 'user'), messages[-1])
                return self.complete(
                    [messages[0], last_user],
                    max_tokens=max_tokens,
                    link_tools=link_tools,
                    tools=tools,
                    on_tool=on_tool,
                    max_tool_rounds=max_tool_rounds,
                    _shrink=_shrink, **sampling)
            if not _shrink and messages[-1].get('role') == 'user' and len(messages[-1].get('content') or '') > 12000:
                user = dict(messages[-1])
                user['content'] = user['content'][:12000] + '\n[... truncated]'
                return self.complete(
                    [messages[0], user],
                    max_tokens=max_tokens,
                    link_tools=link_tools,
                    tools=tools,
                    on_tool=on_tool,
                    max_tool_rounds=max_tool_rounds,
                    _shrink=True, **sampling)
        raise ModelError(f'Chat completion failed: {message}')


def _truncate_tool_result(text: str, limit: int = 120) -> str:
    line = text.strip().splitlines()[0] if text.strip() else '(empty)'
    if len(line) <= limit:
        return line
    return line[:limit] + '...'
