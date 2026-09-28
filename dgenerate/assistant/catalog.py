"""
The models the assistant can use. Standard library only, the console imports this.
"""

import glob
import importlib.util
import os

DEFAULT_CHAT_MODEL = 'unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf'

# Qwen chat GGUFs that fit a desktop. 3.8's open dense model is the 27B.
# 3.5 also has a 9B, and 3.5 and 3.6 have a 35B mixture-of-experts.
# (Hugging Face path, download size in GB, who it suits), by version, smallest first.
CHAT_MODELS = [
    ('unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf', 5.7, 'Qwen 3.5, 8 GB GPUs'),
    ('unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-UD-IQ2_XXS.gguf', 8.6, 'Qwen 3.5, 12 GB GPUs'),
    ('unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-IQ4_XS.gguf', 15.0, 'Qwen 3.5, 16 GB GPUs'),
    ('unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf', 16.7, 'Qwen 3.5, 24 GB GPUs'),
    ('unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf', 17.5, 'Qwen 3.5 MoE, 16 GB GPUs'),
    ('unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q6_K.gguf', 22.5, 'Qwen 3.5, 24 GB GPUs, higher quality'),
    ('unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q8_0.gguf', 28.6, 'Qwen 3.5, 32 GB GPUs'),
    ('unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-IQ2_XXS.gguf', 9.4, 'Qwen 3.6, 12 GB GPUs'),
    ('unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-IQ4_XS.gguf', 15.4, 'Qwen 3.6, 16 GB GPUs'),
    ('unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf', 16.8, 'Qwen 3.6, 24 GB GPUs'),
    ('unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf', 17.7, 'Qwen 3.6 MoE, 16 GB GPUs'),
    ('unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q6_K.gguf', 22.5, 'Qwen 3.6, 24 GB GPUs, higher quality'),
    ('unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q8_0.gguf', 28.6, 'Qwen 3.6, 32 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf', 9.0, 'Qwen 3.8, 12 GB GPUs'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf', 15.7, 'Qwen 3.8, 16 GB GPUs'),
    (DEFAULT_CHAT_MODEL, 16.5, 'Qwen 3.8, 24 GB GPUs, default'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf', 22.0, 'Qwen 3.8, 24 GB GPUs, higher quality'),
    ('unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf', 29.0, 'Qwen 3.8, 32 GB GPUs'),
]

DEFAULT_EMBED_MODEL = 'Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf'

# Qwen3.8 template values. Extra high is what the model uses when effort is omitted.
REASONING_EFFORTS = ('low', 'medium', 'xhigh')
DEFAULT_REASONING_EFFORT = 'medium'

# Qwen3-Embedding GGUFs. They all use last-token pooling, which the assistant embedder sets.
# Each one has an index built by assetgen and packaged in assistant/data.
EMBED_MODELS = [
    (DEFAULT_EMBED_MODEL, 0.6, 'default'),
    ('Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf', 4.3, 'stronger retrieval'),
    ('Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf', 8.1, 'strongest retrieval'),
]


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
