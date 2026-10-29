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

"""
The models the assistant can use. Standard library only, the console imports this.
"""

import glob
import importlib.util
import os

DEFAULT_CHAT_MODEL = 'unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf'

# Every non-BF16 quant of unsloth's Qwen 3.8 chat GGUF. 3.8's open dense model is the 27B.
# Left out: BF16 files, 3.8 2.4T (too big), 3.8 Flash-Next (new architecture, folder-split quants).
# (Hugging Face path, download size in GB, who it suits), smallest first.
# GPU labels are a rough fit: file size plus room for context.
CHAT_MODELS = [
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf', 6.2, 'Qwen 3.8, 8 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_M.gguf', 6.7, 'Qwen 3.8, 8 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf', 7.3, 'Qwen 3.8, 8 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_S.gguf', 8.4, 'Qwen 3.8, 12 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf', 9.8, 'Qwen 3.8, 12 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf', 10.9, 'Qwen 3.8, 12 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf', 12.0, 'Qwen 3.8, 16 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf', 13.1, 'Qwen 3.8, 16 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf', 14.3, 'Qwen 3.8, 16 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_S.gguf', 15.4, 'Qwen 3.8, 16 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf', 16.1, 'Qwen 3.8, 24 GB GPUs'),
    (DEFAULT_CHAT_MODEL, 16.5, 'Qwen 3.8, 24 GB GPUs, default'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_1.gguf', 17.5, 'Qwen 3.8, 24 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf', 17.6, 'Qwen 3.8, 24 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q5_K_S.gguf', 18.7, 'Qwen 3.8, 24 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q5_K_M.gguf', 19.8, 'Qwen 3.8, 24 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q5_K_XL.gguf', 20.9, 'Qwen 3.8, 24 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf', 22.0, 'Qwen 3.8, 24 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_M.gguf', 23.1, 'Qwen 3.8, 24 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_L.gguf', 24.2, 'Qwen 3.8, 32 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_XL.gguf', 25.3, 'Qwen 3.8, 32 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_L.gguf', 28.0, 'Qwen 3.8, 32 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf', 29.0, 'Qwen 3.8, 32 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf', 31.5, 'Qwen 3.8, 32 GB GPUs'),
]


DEFAULT_EMBED_MODEL = 'Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf'

# Qwen3.8 template values. Extra high is what the model uses when effort is omitted.
REASONING_EFFORTS = ('low', 'medium', 'xhigh')
DEFAULT_REASONING_EFFORT = 'medium'

# Qwen3.8-27B sampling from the Unsloth model card. Thinking and non-thinking differ.
# repeat_penalty is the card's repetition_penalty. min_p is 0 on purpose: llama.cpp
# otherwise uses 0.05, and top_k otherwise uses 40.
# https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
_SAMPLING_THINKING = {
    'temperature': 1.0,
    'top_p': 0.95,
    'top_k': 20,
    'min_p': 0.0,
    'presence_penalty': 0.0,
    'repeat_penalty': 1.0,
}
_SAMPLING_INSTRUCT = {
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 20,
    'min_p': 0.0,
    'presence_penalty': 1.5,
    'repeat_penalty': 1.0,
}


def sampling_defaults(think: bool) -> dict:
    """Recommended Qwen3.8 sampling for thinking or non-thinking mode."""
    return dict(_SAMPLING_THINKING if think else _SAMPLING_INSTRUCT)


def resolve_sampling(think: bool, **overrides) -> dict:
    """
    Recommended sampling, with any value the caller set left as given.

    ``None`` means the caller did not set that value.
    """
    values = sampling_defaults(think)
    for name, value in overrides.items():
        if value is not None:
            values[name] = value
    return values


# Qwen3-Embedding GGUFs. They all use last-token pooling, which the assistant embedder sets.
# Each one has an index built by assetgen and packaged in assistant/data.
EMBED_MODELS = [
    (DEFAULT_EMBED_MODEL, 0.6, 'default'),
    ('Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf', 4.3, 'stronger retrieval'),
    ('Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf', 8.1, 'strongest retrieval'),
]


def chat_model_specs() -> tuple[str, ...]:
    """Hugging Face paths of the chat models the assistant can load."""
    return tuple(spec for spec, _size, _note in CHAT_MODELS)


def embed_model_specs() -> tuple[str, ...]:
    """Hugging Face paths of the embedding models that have a packaged index."""
    return tuple(spec for spec, _size, _note in EMBED_MODELS)


def xllamacpp_installed() -> bool:
    """Whether the xllamacpp extra is installed. Does not import it."""
    return importlib.util.find_spec('xllamacpp') is not None


def _hub_cache() -> str:
    # Resolved the way huggingface_hub.constants resolves it.
    hf_home = os.path.expanduser(os.environ.get(
        'HF_HOME', os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.join('~', '.cache')), 'huggingface')))
    return os.path.expanduser(os.environ.get(
        'HF_HUB_CACHE', os.environ.get('HUGGINGFACE_HUB_CACHE', os.path.join(hf_home, 'hub'))))


def is_downloaded(spec: str) -> bool:
    """
    Whether a model is a local file or already in the Hugging Face cache.
    """
    if os.path.isfile(spec):
        return True
    parts = spec.replace('\\', '/').split('/')
    if len(parts) < 3:
        return False
    repo_folder = 'models--' + '--'.join(parts[:2])
    snapshots = os.path.join(_hub_cache(), repo_folder, 'snapshots')
    return any(os.path.isfile(p) for p in glob.glob(os.path.join(glob.escape(snapshots), '*', *parts[2:])))
