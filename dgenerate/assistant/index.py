import collections
import dataclasses
import functools
import hashlib
import json
import math
import os
import re

import numpy as np

import dgenerate.assistant.corpus as _corpus

INDEX_VERSION = 18

SHIPPED_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'index.npz')
"""The index built by ``python -m assetgen.build --target assistant-index`` and packaged with dgenerate."""

# Option names like --ic-lora stay whole so exact flag mentions score in BM25.
_TOKEN = re.compile(r'--?[a-z0-9][a-z0-9-]*|[a-z0-9]+(?:[._][a-z0-9]+)*')

_RRF_K = 60

# Score multiplier for examples that need input media when the request names none.
_INPUT_EXAMPLE_PENALTY = 0.6


class AssistantIndexError(Exception):
    pass


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def fingerprint(repo: str) -> str:
    """
    Hash of the corpus source files. Line endings are normalized so a Windows
    checkout matches an index built on Linux.
    """
    h = hashlib.sha1(f'{INDEX_VERSION}\n'.encode())
    for path in _corpus.source_files(repo):
        with open(path, 'rb') as f:
            content = f.read().replace(b'\r\n', b'\n')
        h.update(os.path.relpath(path, repo).replace('\\', '/').encode() + b'\0')
        h.update(hashlib.sha1(content).digest())
    return h.hexdigest()


@functools.cache
def known_options() -> tuple[str, ...]:
    """
    Every dgenerate option string, including short aliases.
    """
    import dgenerate.arguments as _arguments
    return tuple(sorted({o for action in _arguments._actions for o in action.option_strings}))


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [collections.Counter(d) for d in documents]
        self.lengths = np.asarray([len(d) for d in documents], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if len(documents) else 0.0
        df = collections.Counter(t for counts in self.tf for t in counts)
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        result = np.zeros(len(self.tf), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg_length, 1e-6))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            freq = np.asarray([counts.get(term, 0) for counts in self.tf], dtype=np.float32)
            result += idf * freq * (self.k1 + 1) / (freq + norm)
        return result


@dataclasses.dataclass
class SearchHit:
    chunk: _corpus.Chunk
    score: float


class Index:
    def __init__(self, chunks: list[_corpus.Chunk], vectors: np.ndarray, meta: dict):
        self.chunks = chunks
        # float16 on disk ranks identically to float32, search runs in float32.
        self.vectors = vectors.astype(np.float32)
        self.meta = meta
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25([tokenize(c.search_text()) for c in chunks])
        self.input_examples = np.asarray(
            [c.kind == 'example' and _corpus.uses_media(c.text) for c in chunks], dtype=bool)

    @property
    def known_options(self) -> tuple[str, ...]:
        return known_options()

    def argument(self, option: str) -> _corpus.Chunk | None:
        return self.by_id.get(f'argument:{option}:0')

    def scheduler(self, name: str) -> _corpus.Chunk | None:
        return self.by_id.get(f'scheduler:{name}:0')

    def search(self, query: str, query_vector: np.ndarray, limit: int = 80,
               has_inputs: bool = True) -> list[SearchHit]:
        """
        :param has_inputs: whether the request names input media. When it does not,
            examples that transform an input image or video rank lower.
        """
        if query_vector.shape[-1] != self.vectors.shape[1]:
            raise AssistantIndexError(
                f'This embedding is {int(query_vector.shape[-1])} dimensions, and the index was built '
                f'with {self.meta.get("embed_model")} ({self.vectors.shape[1]} dimensions).')
        dense = self.vectors @ query_vector
        lexical = self.bm25.scores(tokenize(query))
        fused = np.zeros(len(self.chunks), dtype=np.float64)
        for scores in (dense, lexical):
            order = np.argsort(-scores)
            fused[order] += 1.0 / (_RRF_K + np.arange(1, len(order) + 1))
        if not has_inputs:
            fused[self.input_examples] *= _INPUT_EXAMPLE_PENALTY
        for i, chunk in enumerate(self.chunks):
            if chunk.kind == 'guide':
                fused[i] *= 1.35
        best = np.argsort(-fused)[:limit]
        return [SearchHit(self.chunks[i], float(fused[i])) for i in best]

    def save(self, path: str):
        """
        Write the index to one compressed ``.npz`` file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        chunks = json.dumps([dataclasses.asdict(c) for c in self.chunks], separators=(',', ':'))
        meta = json.dumps(self.meta, separators=(',', ':'))
        temp = path + '.tmp.npz'
        np.savez_compressed(
            temp,
            vectors=self.vectors.astype(np.float16),
            chunks=np.frombuffer(chunks.encode('utf-8'), dtype=np.uint8),
            meta=np.frombuffer(meta.encode('utf-8'), dtype=np.uint8))
        os.replace(temp, path)

    @staticmethod
    def read_meta(path: str) -> dict | None:
        try:
            with np.load(path, allow_pickle=False) as data:
                return json.loads(data['meta'].tobytes().decode('utf-8'))
        except (OSError, KeyError, ValueError):
            return None

    @classmethod
    def load(cls, path: str) -> 'Index':
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data['meta'].tobytes().decode('utf-8'))
            chunks = [_corpus.Chunk(**c) for c in json.loads(data['chunks'].tobytes().decode('utf-8'))]
            vectors = data['vectors']
        return cls(chunks, vectors, meta)

    @classmethod
    def build(cls, repo: str, embedder, log=print) -> 'Index':
        chunks = _corpus.collect(repo)
        log(f'Embedding {len(chunks)} chunks from {repo}')

        def progress(done, total):
            if done == total or done % 128 == 0:
                log(f'  {done}/{total}')

        vectors = embedder.embed_documents([c.search_text() for c in chunks], progress=progress)
        meta = {
            'version': INDEX_VERSION,
            'embed_model': os.path.basename(embedder.model_path),
            'fingerprint': fingerprint(repo),
        }
        return cls(chunks, vectors, meta)


def shipped_index_path(embed_basename: str) -> str:
    """
    The packaged index for an embedding model. The default model keeps the
    historical name ``index.npz``. Every other packaged model is
    ``{model}.npz`` in the same directory.
    """
    data = os.path.dirname(SHIPPED_INDEX)
    stem = embed_basename[:-5] if embed_basename.lower().endswith('.gguf') else embed_basename
    named = os.path.join(data, stem + '.npz')
    if os.path.isfile(named):
        return named
    meta = Index.read_meta(SHIPPED_INDEX) if os.path.isfile(SHIPPED_INDEX) else None
    if meta and meta.get('embed_model') == embed_basename:
        return SHIPPED_INDEX
    return named


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_DIR = os.path.dirname(_PACKAGE_DIR)


def corpus_repo() -> str | None:
    """
    The checkout that holds ``examples/`` and the assistant guide, when this
    process is running from one. An install without the examples cannot rebuild an index.
    """
    guide = os.path.join(_PACKAGE_DIR, 'assistant', 'data', 'guide.rst')
    if os.path.isdir(os.path.join(_REPO_DIR, 'examples')) and os.path.isfile(guide):
        return _REPO_DIR
    return None


def index_cache_dir() -> str:
    """Where indexes for an embedding model that was not packaged are kept."""
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
    else:
        base = os.environ.get('XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache'))
    return os.path.join(base, 'dgenerate', 'assistant')


def _cache_path(embed_basename: str) -> str:
    name = embed_basename[:-5] + '.npz' if embed_basename.lower().endswith('.gguf') else embed_basename + '.npz'
    return os.path.join(index_cache_dir(), name)


def choose_index(embed_basename: str, shipped_meta: dict | None, cache_meta: dict | None,
                 repo_fingerprint: str | None) -> str:
    """
    ``shipped``, ``cache``, or ``build``. A cache with no repo to compare against is
    used as-is when its version and embedding model match.
    """
    if (shipped_meta and shipped_meta.get('version') == INDEX_VERSION
            and shipped_meta.get('embed_model') == embed_basename):
        return 'shipped'
    if (cache_meta and cache_meta.get('version') == INDEX_VERSION
            and cache_meta.get('embed_model') == embed_basename
            and (repo_fingerprint is None or cache_meta.get('fingerprint') == repo_fingerprint)):
        return 'cache'
    return 'build'


def load_index(embedder, log=print) -> Index:
    """
    Load the index that matches ``embedder``. A packaged index is used when it was
    built with that model. Otherwise a cached index is used, or built from the
    checkout on first use.
    """
    basename = os.path.basename(embedder.model_path)
    shipped_path = shipped_index_path(basename)
    shipped_meta = Index.read_meta(shipped_path) if os.path.isfile(shipped_path) else None
    repo = corpus_repo()
    repo_fingerprint = fingerprint(repo) if repo else None
    cache_path = _cache_path(basename)
    cache_meta = Index.read_meta(cache_path) if os.path.isfile(cache_path) else None
    choice = choose_index(basename, shipped_meta, cache_meta, repo_fingerprint)
    if choice == 'shipped':
        return Index.load(shipped_path)
    if choice == 'cache':
        log(f'Loading the index for {basename}')
        return Index.load(cache_path)
    if repo is None:
        raise AssistantIndexError(
            f'dgenerate has no assistant index for {basename}. '
            'Build it with: python -m assetgen.build --target assistant-index')
    log(f'Building an index for {basename}.')
    index = Index.build(repo, embedder, log=log)
    index.save(cache_path)
    return index
