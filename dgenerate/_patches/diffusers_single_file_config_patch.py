# Copyright (c) 2023, Teriks
# BSD 3-Clause License

import pathlib
import os

import diffusers.loaders.single_file as _df_single_file
import diffusers.utils.hub_utils as _hub_utils
import huggingface_hub
from huggingface_hub import snapshot_download as _hf_snapshot_download

_HUB_CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "pipelinewrapper" / "hub_configs"

# Store original functions for fallback
_original_hf_hub_download = huggingface_hub.hf_hub_download


def _is_offline(kwargs: dict) -> bool:
    if kwargs.get("local_files_only", False):
        return True
    if _hub_utils.HF_HUB_OFFLINE or _hub_utils.HF_HUB_DISABLE_TELEMETRY:
        return True
    return False


def _has_vendored_configs(repo_id: str) -> bool:
    """Check if we have vendored configs for this repo_id."""
    if not _HUB_CONFIGS_DIR.exists():
        return False
    try:
        repo_dir = _HUB_CONFIGS_DIR / f"models--{repo_id.replace('/', '--')}"
        return repo_dir.exists() and repo_dir.is_dir()
    except Exception:
        return False


def _get_vendored_config_path(repo_id: str) -> str:
    """
    Return the path to vendored config directory for the given repo_id.
    The vendored configs are already in the correct structure.
    """
    repo_dir_name = f"models--{repo_id.replace('/', '--')}"
    vendored_repo_path = _HUB_CONFIGS_DIR / repo_dir_name

    if not vendored_repo_path.exists():
        raise FileNotFoundError(f"No vendored configs found for {repo_id}")

    return str(vendored_repo_path)


def _snapshot_download_for_single_file(repo_id: str, *args, **kwargs) -> str:
    """
    Replacement for diffusers.loaders.single_file.snapshot_download that resolves configs
    from vendored hub_configs when possible, especially for config-only requests.
    Falls back to Hugging Face snapshot_download otherwise.
    """
    allow_patterns = kwargs.get('allow_patterns')

    # Check if this is likely a config-only request (based on allow_patterns)
    is_config_request = (allow_patterns and
                         all(any(ext in pattern for ext in ['.json', '.txt', '.model'])
                             for pattern in allow_patterns))

    # If we have vendored configs and this is offline or a config-only request
    if _has_vendored_configs(repo_id):
        if _is_offline(kwargs) or is_config_request:
            try:
                return _get_vendored_config_path(repo_id)
            except Exception:
                # If vendored configs fail and we're offline, we have to fail
                if _is_offline(kwargs):
                    raise
                # Otherwise fall back to network

    # Fallback to original HH snapshot_download
    if _hf_snapshot_download is None:
        raise RuntimeError("huggingface_hub.snapshot_download unavailable")
    return _hf_snapshot_download(repo_id, *args, **kwargs)


_df_single_file.snapshot_download = _snapshot_download_for_single_file


def _patched_hf_hub_download(repo_id: str, filename: str, *args, **kwargs) -> str:
    """
    Patched hf_hub_download that uses vendored configs when available in offline mode.
    This handles config file downloads for single-file checkpoints.
    """
    # Only intercept config-like files when offline and vendored configs exist
    if (_is_offline(kwargs) and 
        _has_vendored_configs(repo_id) and 
        _is_config_like_file(filename)):
        
        try:
            vendored_path = _get_vendored_config_path(repo_id)
            subfolder = kwargs.get('subfolder', '')
            
            if subfolder:
                config_file_path = os.path.join(vendored_path, subfolder, filename)
            else:
                config_file_path = os.path.join(vendored_path, filename)
            
            if os.path.exists(config_file_path):
                return config_file_path
        except Exception:
            pass  # Fall back to original behavior
    
    # Fallback to original function
    return _original_hf_hub_download(repo_id, filename, *args, **kwargs)


def _is_config_like_file(filename: str) -> bool:
    """Check if a filename looks like a config file we might have vendored."""
    config_files = {
        'model_index.json',
        'config.json', 
        'scheduler_config.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'vocab.json',
        'tokenizer.json',
        'merges.txt',
        'preprocessor_config.json',
        'text_config.json',
        'processor_config.json',
        'spiece.model'
    }
    return filename in config_files


# Apply the patch
huggingface_hub.hf_hub_download = _patched_hf_hub_download
