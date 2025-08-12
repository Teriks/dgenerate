# Copyright (c) 2023, Teriks
# BSD 3-Clause License

import os
import json
from typing import Optional, Dict, Any

import diffusers.utils.hub_utils as hub_utils
import diffusers.models.modeling_utils as modeling_utils
from huggingface_hub import model_info, snapshot_download


from requests import HTTPError


_ORIG_GET_CHECKPOINT_SHARD_FILES = hub_utils._get_checkpoint_shard_files


def _patched_get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    proxies=None,
    local_files_only=False,
    token=None,
    user_agent=None,
    revision=None,
    subfolder="",
    dduf_entries: Optional[Dict[str, Any]] = None,
):
    """
    Wrapper that avoids calling model_info when offline/local-only to prevent network usage.
    The rest of the behavior mirrors diffusers.utils.hub_utils._get_checkpoint_shard_files.
    """
    if dduf_entries:
        if index_filename not in dduf_entries:
            raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")
    else:
        if not os.path.isfile(index_filename):
            raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    if dduf_entries:
        index = json.loads(dduf_entries[index_filename].read_text())
    else:
        with open(index_filename, "r") as f:
            index = json.loads(f.read())

    original_shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()
    shards_path = os.path.join(pretrained_model_name_or_path, subfolder)

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path) or dduf_entries:
        shard_filenames = [os.path.join(shards_path, f) for f in original_shard_filenames]
        for shard_file in shard_filenames:
            if dduf_entries:
                if shard_file not in dduf_entries:
                    raise FileNotFoundError(
                        f"{shards_path} does not appear to have a file named {shard_file} which is "
                        "required according to the checkpoint index."
                    )
            else:
                if not os.path.exists(shard_file):
                    raise FileNotFoundError(
                        f"{shards_path} does not appear to have a file named {shard_file} which is "
                        "required according to the checkpoint index."
                    )
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    allow_patterns = original_shard_filenames
    if subfolder is not None:
        allow_patterns = [os.path.join(subfolder, p) for p in allow_patterns]

    ignore_patterns = ["*.json", "*.md"]

    # Only call model_info when not offline/local-only
    use_local_only = local_files_only or hub_utils.HF_HUB_OFFLINE or hub_utils.HF_HUB_DISABLE_TELEMETRY
    if not use_local_only:
        model_files_info = model_info(pretrained_model_name_or_path, revision=revision, token=token)
        for shard_file in original_shard_filenames:
            shard_file_present = any(shard_file in k.rfilename for k in model_files_info.siblings)
            if not shard_file_present:
                raise EnvironmentError(
                    f"{shards_path} does not appear to have a file named {shard_file} which is "
                    f"required according to the checkpoint index."
                )

    try:
        cached_folder = snapshot_download(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            proxies=proxies,
            local_files_only=use_local_only,
            token=token,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            user_agent=user_agent,
        )
        if subfolder is not None:
            cached_folder = os.path.join(cached_folder, subfolder)

    # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
    # we don't have to catch them here. We have also dealt with EntryNotFoundError.
    except HTTPError as e:
        raise EnvironmentError(
            f"We couldn't connect to '{hub_utils.HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {pretrained_model_name_or_path}. You should try"
            " again after checking your internet connection."
        ) from e

    cached_filenames = [os.path.join(cached_folder, f) for f in original_shard_filenames]

    # Validate cached files exist
    for cached_file in cached_filenames:
        if not os.path.exists(cached_file):
            if use_local_only:
                raise FileNotFoundError(
                    f"Shard file {os.path.basename(cached_file)} is missing from the local cache "
                    f"({cached_folder}) and cannot be downloaded in offline mode. "
                    f"Please ensure all required checkpoint files are cached locally."
                )
            else:
                raise FileNotFoundError(
                    f"Shard file {cached_file} was not properly downloaded or is missing from cache."
                )

    return cached_filenames, sharded_metadata


# Apply patch in both locations (module and imported alias inside modeling_utils)
hub_utils._get_checkpoint_shard_files = _patched_get_checkpoint_shard_files
modeling_utils._get_checkpoint_shard_files = _patched_get_checkpoint_shard_files

