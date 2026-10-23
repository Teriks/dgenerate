#!/usr/bin/env python3
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

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download, HfApi
except ImportError:
    snapshot_download = None
    HfApi = None


class HfConfigsBuilder:
    """Builder for downloading Hugging Face model configuration files."""

    def __init__(self, project_dir: Optional[Path] = None):
        """
        Initialize the HfConfigsBuilder.

        :param project_dir: Project directory (defaults to current working directory)
        :type project_dir: Optional[Path]
        """
        self.project_dir = project_dir or Path.cwd()
        self.output_dir = self.project_dir / 'dgenerate' / 'pipelinewrapper' / 'hub_configs'

        # Maps repository -> [source repository, hub commit sha, download patterns hash]
        # that was last downloaded. A repository is skipped when all three match
        # and the files are still present.
        # This file lives in hub_configs itself, so it is committed and packaged
        # with the configs it describes and the two always agree.
        self.revisions_file = self.output_dir / 'revisions.json'
        
        # List of repositories to download from
        # These are the repositories that diffusers actually uses for single file loading
        self.repositories = [
            ### Model type: SD1x/2x
            # sd1.5
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            # sd2.1
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-2-inpainting",
            # sd1.5 controlnets
            "lllyasviel/control_v11p_sd15_canny",
            #
            ### Model type: SDXL
            # SDXL
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            # SDXL Controlnets (Covers all SDXL controlnet types)
            "diffusers/controlnet-canny-sdxl-1.0",
            "diffusers/controlnet-canny-sdxl-1.0-mid",
            "diffusers/controlnet-canny-sdxl-1.0-small",
            #
            ### Model type: Flux
            # Flux
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-Fill-dev",
            "black-forest-labs/FLUX.1-Depth-dev",
            #
            ### Model type: Flux.2
            # GGUF loading reads the transformer config from these repos.
            "black-forest-labs/FLUX.2-dev",
            "black-forest-labs/FLUX.2-klein-4B",
            "black-forest-labs/FLUX.2-klein-base-9B",
            "black-forest-labs/FLUX.2-klein-9b-kv",
            #
            ### Model type: Z-Image
            # Base and Turbo share one transformer config. GGUF uses Turbo.
            "Tongyi-MAI/Z-Image",
            "Tongyi-MAI/Z-Image-Turbo",
            #
            ### Model type: Qwen-Image
            "Qwen/Qwen-Image",
            "Qwen/Qwen-Image-Edit",
            "Qwen/Qwen-Image-Edit-2509",
            "Qwen/Qwen-Image-Layered",
            # Qwen Image 2.1 family
            "Qwen/Qwen-Image-2.1",
            "Qwen/Qwen-Image-2.1-Turbo",
            #
            ### Model type: LTX
            "Lightricks/LTX-Video",
            "Lightricks/LTX-2.5-Diffusers",
            #
            ### Model type: Wan
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
            "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "Wan-AI/Wan2.1-VACE-14B-diffusers",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "Wan-AI/Wan2.2-Animate-14B-Diffusers",
            "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
            "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
            #
            # Hunyuan Video 1.5
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
            # Kandinsky 5 Video
            "kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers",
            "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers",
            "kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers",
            "kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers",
            # Motif Video
            "Motif-Technologies/Motif-Video-2B",
            # Krea 2
            "krea/Krea-2-Raw",
            "krea/Krea-2-Turbo",
            ### Model type: SD3x
            # SD3m
            "stabilityai/stable-diffusion-3-medium-diffusers",
            # SD3.5
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3.5-medium",
            #
            ### Model type: Pix2Pix
            # Pix2Pix
            "timbrooks/instruct-pix2pix",
            # Kandinsky Image models
            "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers",
            "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers",
            #
            ### Model type: Upscaler
            "stabilityai/stable-diffusion-x4-upscaler",
            #
            ### Model type: Stable Cascade
            # Stable Cascade
            "stabilityai/stable-cascade",
            "stabilityai/stable-cascade-prior",
        ]

        # Repositories that diffusers still refers to by their original id, but which
        # have moved on the hub. They are saved under the original id so that diffusers
        # single file loading finds them, and downloaded from where they live now.
        self.moved_repositories = {
            "stabilityai/stable-diffusion-2-1": "sd2-community/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-2-inpainting": "sd2-community/stable-diffusion-2-inpainting",
        }

    def _load_revisions(self) -> dict:
        try:
            return json.loads(self.revisions_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}

    def _save_revisions(self, revisions: dict):
        self.revisions_file.parent.mkdir(parents=True, exist_ok=True)
        self.revisions_file.write_text(json.dumps(revisions, indent=2, sort_keys=True), encoding='utf-8')

    def build(self):
        """Download essential JSON configuration files from Hugging Face repositories."""
        print("Downloading Hugging Face model configuration files...")


        # Create the target directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Target directory: {self.output_dir}")
        print(f"Number of repositories: {len(self.repositories)}")
        print("-" * 60)

        success_count = 0
        failed_repos = []

        if HfApi is None or snapshot_download is None:
            raise RuntimeError('huggingface_hub is required to build hub configs.')

        # Download only essential .json config files
        essential_patterns = [
            "model_index.json",           # Main model configuration
            "config.json",                # Top-level config (for ControlNets and other models)
            "*/config.json",              # Component configs (unet, vae, text_encoder, etc.)
            "*/scheduler_config.json",    # Scheduler configuration
            "*/tokenizer_config.json",    # Tokenizer configuration
            "*/special_tokens_map.json",  # Tokenizer special tokens
            "*/vocab.json",               # Tokenizer vocabulary
            "*/tokenizer.json",           # Tokenizer data
            # Ensure BPE/SentencePiece tokenizers are complete
            "merges.txt",
            "*/merges.txt",
            "**/merges.txt",
            "spiece.model",
            "*/spiece.model",
            "**/spiece.model",
            "*/preprocessor_config.json", # Feature extractor config
            "*.safetensors.index.json",   # Model weight indices
            "*.safetensors.index.fp16.json",  # FP16 model weight indices
            # Diffusers single-file loading requests every json, txt, and
            # sentencepiece model. Processor and chat-template files live here.
            "**/*.json",
            "**/*.txt",
            "**/*.model",
        ]

        # Changing the download patterns invalidates every recorded revision
        patterns_key = hashlib.sha256(
            json.dumps(essential_patterns).encode('utf-8')).hexdigest()[:16]

        revisions = self._load_revisions()
        api = HfApi()

        # Download .json config files from each repository
        for i, repo in enumerate(self.repositories, 1):
            print(f"[{i}/{len(self.repositories)}] Processing: {repo}")
            
            try:
                # Create a directory structure: models--org--name
                repo_dir_name = f"models--{repo.replace('/', '--')}"
                repo_path = self.output_dir / repo_dir_name
                source_repo = self.moved_repositories.get(repo, repo)

                # One cheap metadata request, skip the download if the hub
                # commit is the one we already have on disk
                hub_sha = api.repo_info(repo_id=source_repo).sha
                if (hub_sha and revisions.get(repo) == [source_repo, hub_sha, patterns_key]
                        and any(repo_path.rglob("*.json"))):
                    print(f"  ✓ Up to date ({hub_sha[:8]}), skipping")
                    success_count += 1
                    print()
                    continue

                snapshot_download(
                    repo_id=source_repo,
                    allow_patterns=essential_patterns,
                    local_dir=repo_path
                )
                
                # Remove any .cache directories that might have been created
                cache_dirs = list(repo_path.rglob(".cache"))
                for cache_dir in cache_dirs:
                    if cache_dir.is_dir():
                        shutil.rmtree(cache_dir)
                        print(f"  ✓ Removed checkpoint .cache directory")
                
                # Count downloaded JSON files
                json_files = list(repo_path.rglob("*.json"))
                print(f"  ✓ Downloaded {len(json_files)} JSON files")
                success_count += 1

                if hub_sha:
                    revisions[repo] = [source_repo, hub_sha, patterns_key]
                    self._save_revisions(revisions)

            except Exception as e:
                print(f"  ✗ Failed to download from {repo}: {str(e)}")
                failed_repos.append(repo)
                continue
            
            print()

        # Summary
        print("=" * 60)
        print(f"Download Summary:")
        print(f"  Successfully processed: {success_count}/{len(self.repositories)} repositories")
        
        if failed_repos:
            print(f"  Failed repositories:")
            for repo in failed_repos:
                print(f"    - {repo}")
            print("✗ Some repositories failed to download")
        else:
            print(f"  ✓ All repositories downloaded successfully!")
        
        print(f"\nJSON config files are now available in: {self.output_dir}")

    def get_output_path(self) -> Path:
        """Get the path to the hub configs output directory."""
        return self.output_dir
