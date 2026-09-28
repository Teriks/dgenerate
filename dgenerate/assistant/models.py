import json
import os
import re
import socket
import sys

import numpy as np

from dgenerate.assistant.catalog import CHAT_MODELS, DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, is_downloaded

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
                 temperature: float = 0.3,
                 top_p: float = 0.9,
                 top_k: int = 20,
                 _shrink: bool = False) -> str:
        template_kwargs = {'enable_thinking': self.think}
        if self.think:
            template_kwargs['reasoning_effort'] = self.effort
        body = {
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
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
            failure = error['value']
            message = failure.get('message', failure) if isinstance(failure, dict) else failure
            if 'exceeds the available context' in str(message) or 'larger than the max context size' in str(message):
                if len(messages) > 2:
                    last_user = next((m for m in reversed(messages) if m['role'] == 'user'), messages[-1])
                    return self.complete(
                        [messages[0], last_user],
                        max_tokens=max_tokens,
                        temperature=temperature, top_p=top_p, top_k=top_k, _shrink=_shrink)
                if not _shrink and messages[-1].get('role') == 'user' and len(messages[-1].get('content') or '') > 12000:
                    user = dict(messages[-1])
                    user['content'] = user['content'][:12000] + '\n[... truncated]'
                    return self.complete(
                        [messages[0], user],
                        max_tokens=max_tokens,
                        temperature=temperature, top_p=top_p, top_k=top_k, _shrink=True)
            raise ModelError(f'Chat completion failed: {message}')
        return display.finish()
