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
Fetch CivitAI and Hugging Face model pages for the assistant link tool.
"""

from __future__ import annotations

import functools
import json
import posixpath
import re
import urllib.parse

import dgenerate.webcache as _webcache

_LINK_URL = re.compile(
    r'https?://(?:www\.)?(?:civitai\.com|civitai\.red|huggingface\.co)/[^\s\)\]>"\']+',
    re.IGNORECASE)
_HF_PATH_KINDS = frozenset({'tree', 'blob', 'raw', 'resolve'})
_WEIGHT_SUFFIXES = ('.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf')
_TEXT_SUFFIXES = ('.md', '.txt', '.json', '.yaml', '.yml')
# A repo can hold dozens of weights plus a note each. Images and caption
# dumps are not paths the config can name.
_PATH_LIST_CHARS = 12000
_NOTE_CHARS = 4000

# CivitAI baseModel enum (their constants.ts), lowercase, for bases dgenerate can run.
# Illustrious, NoobAI, and Pony are SDXL checkpoints. Hunyuan 1 is Hunyuan-DiT.
# "Hunyuan Video" on CivitAI is the original video model, not HunyuanVideo 1.5.
_CIVITAI_BASE_TO_MODEL_TYPE = {
    'sd 1.4': 'sd',
    'sd 1.5': 'sd',
    'sd 1.5 lcm': 'sd',
    'sd 1.5 hyper': 'sd',
    'sd 2.0': 'sd',
    'sd 2.0 768': 'sd',
    'sd 2.1': 'sd',
    'sd 2.1 768': 'sd',
    'sd 2.1 unclip': 'sd',
    'sdxl 0.9': 'sdxl',
    'sdxl 1.0': 'sdxl',
    'sdxl 1.0 lcm': 'sdxl',
    'sdxl 1.0 base': 'sdxl',
    'sdxl distilled': 'sdxl',
    'sdxl turbo': 'sdxl',
    'sdxl lightning': 'sdxl',
    'sdxl hyper': 'sdxl',
    'pony': 'sdxl',
    'pony v7': 'sdxl',
    'ponyv7': 'sdxl',
    'illustrious': 'sdxl',
    'noobai': 'sdxl',
    'sd 3': 'sd3',
    'sd 3.5': 'sd3',
    'sd 3.5 medium': 'sd3',
    'sd 3.5 large': 'sd3',
    'sd 3.5 large turbo': 'sd3',
    'stable cascade': 's-cascade',
    'flux.1 s': 'flux',
    'flux.1 d': 'flux',
    'flux.1 dev': 'flux',
    'flux.1 schnell': 'flux',
    'flux.1 krea': 'flux',
    'flux.1 kontext': 'flux-kontext',
    'flux.2 d': 'flux2',
    'flux.2 klein 9b': 'flux2',
    'flux.2 klein 9b-base': 'flux2',
    'flux.2 klein 4b': 'flux2',
    'flux.2 klein 4b-base': 'flux2',
    'qwen': 'qwen-image',
    'qwen 2.1': 'qwen-image-21',
    'qwen image 2.1': 'qwen-image-21',
    'qwen image edit': 'qwen-image-edit',
    'qwen image layered': 'qwen-image-layered',
    'qwen image': 'qwen-image',
    'zimageturbo': 'z-image',
    'zimagebase': 'z-image',
    'z image': 'z-image',
    'krea 2': 'krea2',
    'krea 2 turbo': 'krea2',
    'kolors': 'kolors',
    'hunyuan 1': 'hydit',
    'ltxv': 'ltx-video',
    'ltx video': 'ltx-video',
    'ltxv2': 'ltx2',
    'ltxv 2': 'ltx2',
    'ltxv 2.3': 'ltx2',
    'ltxv 2.5': 'ltx2',
    'wan video': 'wan',
    'wan video 1.3b t2v': 'wan',
    'wan video 14b t2v': 'wan',
    'wan video 14b i2v 480p': 'wan',
    'wan video 14b i2v 720p': 'wan',
    'wan video 2.1': 'wan',
    'wan video 2.2': 'wan',
    'wan video 2.2 i2v-a14b': 'wan',
    'wan video 2.2 t2v-a14b': 'wan',
    'wan video 2.2 ti2v-5b': 'wan',
}

# Real CivitAI bases with no dgenerate pipeline. A version title must not reclassify them.
_CIVITAI_UNSUPPORTED = frozenset({
    'hunyuan video',
    'wan video 2.5 t2v',
    'wan video 2.5 i2v',
    'flux.3',
    'flux.3 klein',
    'flux.3 klein-base',
    'flux 3 video',
    'anima',
    'chroma',
    'hidream',
    'mochi',
    'cogvideox',
    'lumina',
    'auraflow',
})

# A version name may select a finer pipeline only inside the base's family.
_CIVITAI_FINER = {
    'flux': frozenset({'flux-kontext', 'flux-fill'}),
    'flux2': frozenset({'flux2-klein-kv'}),
    'wan': frozenset({'wan-animate', 'wan-animate-2'}),
    'qwen-image': frozenset({'qwen-image-edit', 'qwen-image-layered', 'qwen-image-21'}),
    'sdxl': frozenset({'sdxl-pix2pix'}),
    'sd3': frozenset({'sd3-pix2pix'}),
    'sd': frozenset({'pix2pix'}),
}

# Checkpoint that supplies the text encoder and VAE for a CivitAI baseModel.
# Flux.1 S is handled separately; Flux.1 D is the default flux repo.
_BASE_CHECKPOINT = {
    'sd': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
    'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
    'sdxl-pix2pix': 'diffusers/sdxl-instructpix2pix-768',
    'pix2pix': 'timbrooks/instruct-pix2pix',
    'flux': 'black-forest-labs/FLUX.1-dev',
    'flux-fill': 'black-forest-labs/FLUX.1-Fill-dev',
    'flux-kontext': 'black-forest-labs/FLUX.1-Kontext-dev',
    'flux2': 'black-forest-labs/FLUX.2-dev',
    'flux2-klein-kv': 'black-forest-labs/FLUX.2-klein-9b-kv',
    'sd3': 'stabilityai/stable-diffusion-3.5-large',
    'sd3-pix2pix': 'BleachNick/SD3_UltraEdit_freeform',
    'krea2': 'krea/Krea-2-Raw',
    's-cascade': 'stabilityai/stable-cascade-prior',
    'qwen-image': 'Qwen/Qwen-Image',
    'qwen-image-edit': 'Qwen/Qwen-Image-Edit',
    'qwen-image-layered': 'Qwen/Qwen-Image-Layered',
    'qwen-image-21': 'Qwen/Qwen-Image-2.1',
    'z-image': 'Tongyi-MAI/Z-Image-Turbo',
    'ltx2': 'Lightricks/LTX-2.5-Diffusers',
    'ltx-video': 'Lightricks/LTX-Video',
    'wan': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
    'wan-animate': 'Wan-AI/Wan2.2-Animate-14B-Diffusers',
    'wan-animate-2': 'Wan-AI/Wan2.2-Animate-2-14B-Diffusers',
    'kolors': 'Kwai-Kolors/Kolors-diffusers',
    'hydit': 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
    'hyimage-21': 'hunyuanvideo-community/HunyuanImage-2.1-Diffusers',
    'hyimage-21-refiner': 'hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers',
    'hyvideo-15': 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v',
    'kandinsky5-image': 'kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers',
    'kandinsky5-video': 'kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers',
    'motif-video': 'Motif-Technologies/Motif-Video-2B',
    'if': 'DeepFloyd/IF-I-M-v1.0',
    'ifs': 'DeepFloyd/IF-II-M-v1.0',
    'upscaler-x4': 'stabilityai/stable-diffusion-x4-upscaler',
    'upscaler-x2': 'stabilityai/sd-x2-latent-upscaler',
}

_README_CHARS = 3500
_SUMMARY_CHARS = 12000
# Weight paths plus the README. The file list has to survive the cap.
# Truncating it drops every path past the first screenful.
_HF_SUMMARY_CHARS = 18000


def find_link_urls(text: str) -> list[str]:
    """Unique CivitAI / Hugging Face URLs in ``text``, in order."""
    seen = set()
    urls = []
    for match in _LINK_URL.finditer(text):
        url = match.group(0).rstrip('.,;')
        key = url.lower()
        if key not in seen:
            seen.add(key)
            urls.append(url)
    return urls


def redact_url(url: str) -> str:
    """Remove query tokens from a URL for logs and tool summaries."""
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    params.pop('token', None)
    query = urllib.parse.urlencode(params, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=query))


def _truncate(text: str, limit: int) -> str:
    text = (text or '').strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + '\n[... truncated]'


# Longest first so "sd 3.5" wins over "sd 3".
_CIVITAI_BASE_FRAGMENTS = tuple(sorted(
    _CIVITAI_BASE_TO_MODEL_TYPE.items(), key=lambda item: len(item[0]), reverse=True))

# First path segment on huggingface.co that is not a model repo.
_HF_NON_MODEL = frozenset({'datasets', 'spaces', 'collections', 'posts'})


def _norm_label(label: str) -> str:
    """Lowercase, and treat underscores as hyphens. Hub repos use both."""
    return (label or '').lower().replace('_', '-')


def _specific_model_type(label: str) -> str | None:
    """
    ``--model-type`` from a repo id, version name, or file name.

    Community Hub spellings are included. A later architecture does not inherit
    an earlier one's name: HunyuanImage 3.0 is not 2.1, Kandinsky 6 is not 5,
    and the original HunyuanVideo is not HunyuanVideo 1.5. ``LTXV`` without a
    2 is LTX-Video; ``LTXV2`` and ``ltxv-2.3`` are LTX-2.
    """
    label = _norm_label(label)
    if not label.strip():
        return None
    if re.search(r'wan[\w.-]*animate-2|wan-animate-2|wan animate 2', label):
        return 'wan-animate-2'
    if re.search(r'wan[\w.-]*animate(?!-2)|wan-animate(?!-2)|wan animate(?! 2)', label):
        return 'wan-animate'
    if 'kontext' in label:
        return 'flux-kontext'
    if 'flux-fill' in label or 'fill-dev' in label:
        return 'flux-fill'
    if re.search(r'klein[\w.-]*kv|klein kv', label):
        return 'flux2-klein-kv'
    if re.search(r'flux\.2|flux2|flux-2\b|flux 2', label):
        return 'flux2'
    # FLUX.1-Krea-dev is Flux.1. The repo name wins over a card that says "Krea 2".
    if re.search(r'flux\.1\b|flux1\b', label):
        return 'flux'
    if re.search(r'krea-2\b|krea2\b|krea 2\b', label):
        return 'krea2'
    if 'flux' in label:
        return 'flux'
    # ltxv-2b is the 2B LTX-Video checkpoint. ltxv-2.3 and LTXV2 are LTX-2.
    if re.search(r'ltx-2|ltx2(?!\d)|ltxv2|ltxv-2\.\d', label):
        return 'ltx2'
    if re.search(r'ltx-video|ltxvideo|\bltxv\b|ltx video', label):
        return 'ltx-video'
    if re.search(r'qwen-image-2\.1|qwen-image-21(?!\d)|qwen image 2\.1', label):
        return 'qwen-image-21'
    if re.search(r'qwen-image-edit|qwen image edit', label):
        return 'qwen-image-edit'
    if re.search(r'qwen-image-layered|qwen image layered', label):
        return 'qwen-image-layered'
    if re.search(r'qwen-image|qwen image', label):
        return 'qwen-image'
    if re.search(r'hunyuan-?image|hyimage', label) and not re.search(r'3\.0|image-3', label):
        if 'refiner' in label:
            return 'hyimage-21-refiner'
        return 'hyimage-21'
    if re.search(r'hunyuan-?dit|\bhydit\b|hunyuan dit', label):
        return 'hydit'
    if re.search(r'hunyuan-?video-?1\.5|hunyuanvideo1\.5|hyvideo-15|\bhyvideo\b', label):
        return 'hyvideo-15'
    if re.search(r'kandinsky-5|kandinsky5', label) and not re.search(r'kandinsky-6|kandinsky6', label):
        if re.search(r't2v|i2v', label):
            return 'kandinsky5-video'
        if re.search(r't2i|i2i', label):
            return 'kandinsky5-image'
    if re.search(r'motif-video|motif video', label):
        return 'motif-video'
    if ('ultraedit' in label and 'sd3' in label) or (
            'pix2pix' in label and re.search(r'\bsd3\b|stable-diffusion-3', label)):
        return 'sd3-pix2pix'
    if 'pix2pix' in label and re.search(r'\bsdxl\b|stable-diffusion-xl', label):
        return 'sdxl-pix2pix'
    if 'pix2pix' in label:
        return 'pix2pix'
    if 'x4-upscaler' in label:
        return 'upscaler-x4'
    if re.search(r'x2-latent-upscaler|sd-x2-latent', label):
        return 'upscaler-x2'
    if re.search(r'stable-cascade|stablecascade', label):
        return 's-cascade'
    if re.search(r'\bkolors\b', label):
        return 'kolors'
    if re.search(r'z-image|zimage|z image', label) and 'omni' not in label:
        return 'z-image'
    if re.search(r'if-ii\b', label):
        return 'ifs'
    if re.search(r'if-i-', label) or ('deepfloyd' in label and re.search(r'\bif\b', label)):
        return 'if'
    if re.search(r'stable-diffusion-3|\bsd3\b|sd3\.5', label):
        return 'sd3'
    if re.search(r'stable-diffusion-xl|\bsdxl\b|\bsd-xl\b', label):
        return 'sdxl'
    if re.search(r'wan\d|wan-\d|wan-ai|\bwan\b', label):
        return 'wan'
    if re.search(r'stable-diffusion-(?:v1|[12]\b)|sd-1\.?5|sd-2\.[01]', label):
        return 'sd'
    return None


def _family_model_type(blob: str) -> str | None:
    return _specific_model_type(blob)


def _civitai_base_type(key: str) -> str | None:
    if key in _CIVITAI_BASE_TO_MODEL_TYPE:
        return _CIVITAI_BASE_TO_MODEL_TYPE[key]
    for fragment, model_type in _CIVITAI_BASE_FRAGMENTS:
        if fragment in key:
            return model_type
    return None


def _guess_model_type(base_model: str | None, name: str, description: str) -> str | None:
    """
    Map publisher metadata to a dgenerate ``--model-type``.

    CivitAI baseModel is the family. The selected version name can narrow it
    (Kontext, Fill, Animate). A mention of those words only in the description,
    or in the publish title, does not.
    """
    key = (base_model or '').strip().lower()
    if key in _CIVITAI_UNSUPPORTED:
        return None
    from_name = _specific_model_type((name or '').lower())
    if base_model:
        mapped = _civitai_base_type(key)
        if mapped:
            if from_name in _CIVITAI_FINER.get(mapped, ()):
                return from_name
            return mapped
        if from_name:
            return from_name
        return _specific_model_type(key)
    if from_name:
        return from_name
    blob = ' '.join(x for x in (name, (description or '')[:500]) if x).lower()
    return _family_model_type(blob)


# Checkpoint for one CivitAI base. These are the diffsusers repos the examples load.
_CIVITAI_CHECKPOINT = {
    'flux.1 s': 'black-forest-labs/FLUX.1-schnell',
    'flux.1 schnell': 'black-forest-labs/FLUX.1-schnell',
    'flux.1 d': 'black-forest-labs/FLUX.1-dev',
    'flux.1 dev': 'black-forest-labs/FLUX.1-dev',
    'flux.1 krea': 'black-forest-labs/FLUX.1-Krea-dev',
    'flux.1 kontext': 'black-forest-labs/FLUX.1-Kontext-dev',
    'flux.2 d': 'black-forest-labs/FLUX.2-dev',
    'flux.2 klein 4b': 'black-forest-labs/FLUX.2-klein-4B',
    'flux.2 klein 4b-base': 'black-forest-labs/FLUX.2-klein-base-4B',
    'flux.2 klein 9b': 'black-forest-labs/FLUX.2-klein-9B',
    'flux.2 klein 9b-base': 'black-forest-labs/FLUX.2-klein-base-9B',
    'qwen': 'Qwen/Qwen-Image',
    'qwen 2.1': 'Qwen/Qwen-Image-2.1',
    'zimageturbo': 'Tongyi-MAI/Z-Image-Turbo',
    'zimagebase': 'Tongyi-MAI/Z-Image',
    'krea 2': 'krea/Krea-2-Raw',
    'krea 2 turbo': 'krea/Krea-2-Turbo',
    'ltxv': 'Lightricks/LTX-Video',
    'ltxv2': 'Lightricks/LTX-2',
    'ltxv 2.3': 'Lightricks/LTX-2.3',
    'ltxv 2.5': 'Lightricks/LTX-2.5-Diffusers',
    'wan video': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
    'wan video 1.3b t2v': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
    'wan video 14b t2v': 'Wan-AI/Wan2.1-T2V-14B-Diffusers',
    'wan video 14b i2v 480p': 'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers',
    'wan video 14b i2v 720p': 'Wan-AI/Wan2.1-I2V-14B-720P-Diffusers',
    'wan video 2.1': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
    'wan video 2.2': 'Wan-AI/Wan2.2-T2V-A14B-Diffusers',
    'wan video 2.2 i2v-a14b': 'Wan-AI/Wan2.2-I2V-A14B-Diffusers',
    'wan video 2.2 t2v-a14b': 'Wan-AI/Wan2.2-T2V-A14B-Diffusers',
    'wan video 2.2 ti2v-5b': 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
    'sd 3': 'stabilityai/stable-diffusion-3-medium-diffusers',
    'sd 3.5': 'stabilityai/stable-diffusion-3.5-large',
    'sd 3.5 medium': 'stabilityai/stable-diffusion-3.5-medium',
    'sd 3.5 large': 'stabilityai/stable-diffusion-3.5-large',
    'sd 3.5 large turbo': 'stabilityai/stable-diffusion-3.5-large-turbo',
    'hunyuan 1': 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
    'kolors': 'Kwai-Kolors/Kolors-diffusers',
    'stable cascade': 'stabilityai/stable-cascade-prior',
}


def _checkpoint_repo(base_model: str | None, model_type: str | None) -> str | None:
    key = (base_model or '').strip().lower()
    if key in _CIVITAI_UNSUPPORTED:
        return None
    if key in _CIVITAI_CHECKPOINT:
        return _CIVITAI_CHECKPOINT[key]
    if not model_type:
        return None
    return _BASE_CHECKPOINT.get(model_type)


def _civitai_host(url: str) -> str | None:
    if not _webcache.is_civitai_host(url):
        return None
    host = (urllib.parse.urlparse(url).hostname or '').lower()
    return host.removeprefix('www.')


def _civitai_model_id(url: str) -> tuple[str, str | None]:
    """Return ``(model_id, version_id_or_none)`` from a CivitAI page or download URL."""
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    version = (query.get('modelVersionId') or query.get('modelversionid') or [None])[0]
    download = re.match(r'^/api/download/models/(\d+)$', parsed.path)
    if download:
        return '', download.group(1)
    parts = parsed.path.strip('/').split('/')
    if len(parts) >= 2 and parts[0] == 'models' and parts[1].isdigit():
        return parts[1], version
    return '', version


def _browser_headers() -> dict[str, str]:
    # Lower case so this replaces huggingface_hub's own user-agent entry.
    # Desktop Chrome only: mobile agents often get 403 from Cloudflare.
    import dgenerate.filecache as _filecache

    return {
        'user-agent': _filecache.user_agent(),
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en-US,en;q=0.9',
    }


def _fetch_json(url: str, timeout: float = 30.0) -> dict:
    import requests

    response = requests.get(url, headers=_browser_headers(), timeout=timeout)
    response.raise_for_status()
    return response.json()


def _civitai_hosts(preferred: str) -> tuple[str, ...]:
    """Preferred host first, then the other CivitAI mirror."""
    other = 'civitai.red' if preferred == 'civitai.com' else 'civitai.com'
    return preferred, other


def _civitai_get(path: str, preferred_host: str) -> dict:
    """
    GET a CivitAI API path. On 403 or 429, try the other host once.
    """
    import requests

    errors = []
    for host in _civitai_hosts(preferred_host):
        url = f'https://{host}{path}'
        try:
            return _fetch_json(url)
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, 'status_code', None)
            errors.append(f'{host}: {e}')
            if status not in (403, 429):
                raise
        except requests.exceptions.RequestException as e:
            errors.append(f'{host}: {e}')
    raise requests.exceptions.HTTPError(
        'CivitAI request failed on both hosts: ' + '; '.join(errors))


def _civitai_version(host: str, version_id: str) -> dict:
    return _civitai_get(f'/api/v1/model-versions/{version_id}', host)


def _civitai_model(host: str, model_id: str) -> dict:
    return _civitai_get(f'/api/v1/models/{model_id}', host)


def _pick_civitai_version(data: dict, version_id: str | None) -> dict | None:
    versions = data.get('modelVersions') or []
    if not versions:
        return None
    if version_id:
        for version in versions:
            if str(version.get('id')) == str(version_id):
                return version
    return versions[0]


def _on_request_host(page_url: str, download_url: str) -> str:
    """
    Keep a download URL on the host the user asked for.

    The API often returns ``civitai.com`` links for a ``civitai.red`` page.
    Both hosts take ``CIVIT_AI_TOKEN``.
    """
    page = urllib.parse.urlparse(page_url)
    download = urllib.parse.urlparse(download_url)
    if not _webcache.is_civitai_host(download_url) or not page.hostname:
        return download_url
    return urllib.parse.urlunparse(download._replace(netloc=page.hostname))


def _query_value(url: str, *names: str) -> str | None:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    for name in names:
        values = query.get(name)
        if values and values[0]:
            return values[0]
    return None


def _version_line(version: dict) -> str:
    return (f'id {version.get("id")} "{version.get("name") or "Unnamed"}": '
            f'baseModel {version.get("baseModel") or "(none)"}')


def _format_civitai(url: str, model: dict, version: dict, *, version_was_requested: bool) -> str:
    model_type_name = model.get('type') or 'Unknown'
    # Artifact type (Checkpoint, LORA) is not a base model.
    base = version.get('baseModel') or ''
    name = version.get('name') or model.get('name') or 'Unnamed'
    description = _truncate(version.get('description') or model.get('description') or '', 4000)
    trained = version.get('trainedWords') or []
    # Version name can refine the base (Kontext). The publish title cannot:
    # it names every tab on the page.
    suggested = _guess_model_type(base or None, name, '')
    repo = _checkpoint_repo(base, suggested)
    selected_file = _query_value(url, 'fileId', 'fileid')
    others = [v for v in (model.get('modelVersions') or [])
              if str(v.get('id')) != str(version.get('id'))]
    lines = [
        f'URL: {redact_url(url)}',
        f'Publish title: {model.get("name") or name}',
        'The publish title lists every version on the page. It does not choose one. '
        'baseModel on the selected version does.',
    ]
    if version_was_requested:
        lines.append(f'Selected by modelVersionId {version.get("id")}.')
    else:
        lines.append('This URL has no modelVersionId. Selected the first version in the API response.')
    lines += [
        f'Selected version: {_version_line(version)}',
        f'CivitAI artifact type: {model_type_name}',
        f'baseModel: {base or "(none)"}',
    ]
    if suggested:
        lines.append(f'Suggested dgenerate --model-type: {suggested}')
    if repo:
        lines.append(
            f'Checkpoint repo for that baseModel: {repo}. '
            'Put this on the invocation line. A LoRA or embedding is --loras or an embedding, not this line.')
    if others:
        lines.append('Other versions on this page (not selected, do not use):')
        lines.extend(f'  - {_version_line(v)}' for v in others[:8])
    if trained:
        lines.append(
            'trainedWords (write these in --prompts, not as a URI argument): '
            + ', '.join(str(word) for word in trained[:20]))
        lines.append(
            '--loras accepts only scale, revision, subfolder, weight-name, and component. '
            'token= is not a LoRA argument. It is only valid on --textual-inversions.')
    downloads = []
    saw_selected = False
    for file in version.get('files') or []:
        dl = file.get('downloadUrl')
        if not dl:
            continue
        kind = file.get('type') or 'Model'
        selected = selected_file is not None and str(file.get('id')) == str(selected_file)
        if selected:
            saw_selected = True
        mark = ' [selected by fileId]' if selected else ''
        downloads.append(f'  - {kind}{mark}: {redact_url(_on_request_host(url, dl))}')
    if downloads:
        lines.append('Download URLs. Copy them with no token= query. dgenerate appends CIVIT_AI_TOKEN from the environment:')
        lines.extend(downloads[:8])
    if selected_file and not saw_selected:
        lines.append(f'Requested fileId {selected_file} was not in this version file list.')
    if description:
        lines.append('Publisher description:')
        lines.append(description)
    return '\n'.join(lines)


def fetch_civitai_link(url: str) -> str:
    host = _civitai_host(url)
    if not host:
        return f'Not a CivitAI URL: {redact_url(url)}'
    model_id, version_id = _civitai_model_id(url)
    try:
        if version_id and not model_id:
            version = _civitai_version(host, version_id)
            parent_id = version.get('modelId') or (version.get('model') or {}).get('id')
            if not parent_id:
                return _format_civitai(url, {'name': '', 'type': ''}, version, version_was_requested=True)
            model = _civitai_model(host, str(parent_id))
            return _format_civitai(url, model, version, version_was_requested=True)
        if not model_id:
            return f'Could not parse CivitAI model id from: {redact_url(url)}'
        model = _civitai_model(host, model_id)
        version = _pick_civitai_version(model, version_id)
        if not version:
            return f'No versions on CivitAI model {model_id}'
        return _format_civitai(
            url, model, version, version_was_requested=bool(version_id))
    except Exception as e:
        return (
            f'Failed to fetch CivitAI link {redact_url(url)}: {e}. '
            'Do not call fetch_link on this URL again. Write the config from the '
            'request and the retrieved examples.')


def _parse_hf_url(url: str) -> dict | None:
    """
    Repo id, URL kind, revision, and repo-relative path.

    ``tree``, ``blob``, ``raw``, and ``resolve`` URLs name a folder or a file.
    Anything else under the repo is the repo root. Dataset, space, and
    collection URLs set ``non_model``.
    """
    parts = [urllib.parse.unquote(part) for part in urllib.parse.urlparse(url).path.split('/') if part]
    if parts and parts[0].lower() == 'models':
        parts = parts[1:]
    if len(parts) < 2:
        return None
    org = parts[0]
    if org.lower() in _HF_NON_MODEL:
        return {'non_model': org.lower().rstrip('s')}
    kind = 'repo'
    revision = None
    path = ''
    if len(parts) >= 4 and parts[2] in _HF_PATH_KINDS:
        kind = parts[2]
        revision = parts[3]
        path = '/'.join(parts[4:])
    return {'repo_id': f'{org}/{parts[1]}', 'kind': kind, 'revision': revision, 'path': path}


def _hf_text_file(repo_id: str, filename: str, revision: str | None, limit: int) -> str:
    try:
        from huggingface_hub import hf_hub_download

        kwargs = {'headers': _browser_headers()}
        if revision:
            kwargs['revision'] = revision
        local = hf_hub_download(repo_id, filename, **kwargs)
        with open(local, encoding='utf-8', errors='replace') as handle:
            return _truncate(handle.read(), limit)
    except Exception as e:
        return f'(could not read {filename}: {e})'


def _hf_repo_files(repo_id: str, revision: str | None, info) -> list[str]:
    """Every file path. Sibling names from ``model_info`` are only a fallback."""
    try:
        import huggingface_hub

        kwargs = {'revision': revision} if revision else {}
        return list(huggingface_hub.HfApi(headers=_browser_headers()).list_repo_files(repo_id, **kwargs))
    except Exception:
        return [sibling.rfilename for sibling in (getattr(info, 'siblings', None) or [])]


def _weight_uri(repo_id: str, filepath: str) -> str:
    """One quoted ``--loras`` value. The space in a folder name stays inside the quotes."""
    folder, filename = posixpath.split(filepath)
    if folder:
        body = f'{repo_id};subfolder={folder};weight-name={filename}'
    else:
        body = f'{repo_id};weight-name={filename}'
    return f'"{body}"'


def _append_paths(lines: list[str], title: str, paths: list[str], budget: int) -> None:
    lines.append(f'{title} ({len(paths)}):')
    if not paths:
        lines.append('(none)')
        return
    used = 0
    shown = 0
    for path in paths:
        extra = len(path) + 1
        if shown and used + extra > budget:
            break
        lines.append(path)
        used += extra
        shown += 1
    if shown < len(paths):
        lines.append(
            f'{len(paths) - shown} more not shown. '
            'Call fetch_link on the tree URL of the folder you need.')


def _scope_files(files: list[str], kind: str, path: str) -> list[str]:
    """Folder and file URLs list that directory. The repo root lists the repo."""
    if kind == 'tree' and path:
        prefix = path.strip('/') + '/'
        return [name for name in files if name == path.strip('/') or name.startswith(prefix)]
    if kind in ('blob', 'raw', 'resolve') and path:
        folder = posixpath.dirname(path)
        return [name for name in files if posixpath.dirname(name) == folder]
    return files


def _split_listing(files: list[str]) -> tuple[list[str], list[str], str]:
    weights = []
    notes = []
    omitted: dict[str, int] = {}
    for name in files:
        lower = name.lower()
        if lower.endswith(_WEIGHT_SUFFIXES):
            weights.append(name)
        elif lower.endswith('.md'):
            notes.append(name)
        else:
            ext = posixpath.splitext(lower)[1] or '(no extension)'
            omitted[ext] = omitted.get(ext, 0) + 1
    weights.sort(key=str.casefold)
    notes.sort(key=str.casefold)
    omitted_text = ', '.join(f'{count} {ext}' for ext, count in sorted(omitted.items()))
    return weights, notes, omitted_text


def _readme_path(notes: list[str]) -> str | None:
    for note in notes:
        if posixpath.basename(note).lower() == 'readme.md' and not posixpath.dirname(note):
            return note
    return None


def _note_paths_to_read(kind: str, path: str, notes: list[str]) -> list[str]:
    """
    Text to put in the tool result.

    A repo URL is the README. A folder or file URL is the note beside that
    weight; the README is appended by the caller so a direct file link still
    has it, after the trigger word.
    """
    if kind in ('blob', 'raw', 'resolve') and path:
        if path.lower().endswith(_TEXT_SUFFIXES):
            return [path]
        folder = posixpath.dirname(path)
        return [note for note in notes if posixpath.dirname(note) == folder][:4]
    if kind == 'tree' and path:
        folder = path.strip('/')
        return [note for note in notes if posixpath.dirname(note) == folder][:4]
    readme = _readme_path(notes)
    return [readme] if readme else []


def _format_huggingface(
        url: str,
        repo_id: str,
        *,
        kind: str,
        path: str,
        pipeline: str,
        gated: bool,
        tags: list[str],
        card_text: str,
        files: list[str],
        texts: dict[str, str],
        suggested: str | None) -> str:
    scoped = _scope_files(files, kind, path)
    weights, notes, omitted = _split_listing(scoped)
    lines = [
        f'URL: {redact_url(url)}',
        f'Repo: {repo_id}',
    ]
    if kind == 'tree' and path:
        lines.append(f'Folder: {path}')
    elif kind in ('blob', 'raw', 'resolve') and path:
        lines.append(f'File: {path}')
    lines.append(
        'The repo id is only organization/name. Copy a weight as subfolder= and weight-name= '
        'inside one quoted value. An unquoted space splits the line into two tokens.')
    if kind in ('repo', 'tree') and not (kind == 'tree' and path):
        lines.append(
            'For the subject in the request, call fetch_link once on the raw URL of the '
            f'markdown note listed beside that weight: https://huggingface.co/{repo_id}/raw/main/<note path>. '
            'Use the trigger word in that note. Do not fetch the tree page.')
    lines += [
        f'pipeline_tag: {pipeline or "(none)"}',
        f'gated: {bool(gated)}',
        f'tags: {", ".join(tags) if tags else "(none)"}',
    ]
    if suggested:
        lines.append(f'Suggested dgenerate --model-type: {suggested}')
    copyable = weights
    if kind in ('blob', 'raw', 'resolve') and path:
        copyable = [name for name in weights if posixpath.dirname(name) == posixpath.dirname(path)] or weights
    elif kind == 'tree' and path:
        folder = path.strip('/')
        copyable = [name for name in weights if posixpath.dirname(name) == folder] or weights
    if len(copyable) == 1 or (path and len(copyable) <= 3):
        lines.append('Copy this --loras value:')
        lines.extend(f'--loras {_weight_uri(repo_id, name)}' for name in copyable[:3])
    elif weights:
        # A real first file would be a valid LoRA for the wrong subject.
        lines.append(
            'Example shape only, not a file to copy: '
            '--loras "organization/name;subfolder=Folder/Name With Spaces;weight-name=file.safetensors". '
            'Fill subfolder and weight-name from the weight path the request names.')
    if path and kind in ('blob', 'raw', 'resolve') and path not in files:
        lines.append(f'{path} is not a file in {repo_id}. Weight files in the repo:')
    where = 'this folder' if path else 'the repo'
    _append_paths(lines, f'Weight files in {where}', weights, _PATH_LIST_CHARS)
    _append_paths(lines, f'Markdown notes in {where}', notes, _PATH_LIST_CHARS)
    if omitted:
        lines.append(f'Not listed ({where}): {omitted}. These are not model weights.')
    if card_text:
        lines.append('Model card (subset): ' + card_text)
    for name, body in texts.items():
        if body:
            lines.append(f'{name}:')
            lines.append(body)
    return '\n'.join(lines)


def fetch_huggingface_link(url: str) -> str:
    target = _parse_hf_url(url)
    if target is None:
        return f'Could not parse Hugging Face repo from: {redact_url(url)}'
    if target.get('non_model'):
        return (
            f'{redact_url(url)} is a Hugging Face {target["non_model"]} link, not a model repo. '
            'fetch_link only reads model repos.')
    repo_id = target['repo_id']
    try:
        import huggingface_hub

        kwargs = {'revision': target['revision']} if target['revision'] else {}
        info = huggingface_hub.HfApi(headers=_browser_headers()).model_info(repo_id, **kwargs)
        tags = list(info.tags or [])[:30]
        pipeline = getattr(info, 'pipeline_tag', None) or ''
        gated = bool(getattr(info, 'gated', False))
        card = getattr(info, 'card_data', None)
        card_dict = card.to_dict() if card is not None and hasattr(card, 'to_dict') else {}
        card_text = _truncate(json.dumps(card_dict, ensure_ascii=False), 1500) if card_dict else ''
        files = _hf_repo_files(repo_id, target['revision'], info)
        scoped = _scope_files(files, target['kind'], target['path'])
        _weights, notes, _omitted = _split_listing(scoped)
        _, all_notes, _all_omitted = _split_listing(files)
        to_read = _note_paths_to_read(target['kind'], target['path'], notes)
        readme = _readme_path(all_notes)
        if readme and readme not in to_read:
            to_read.append(readme)
        texts = {}
        for name in to_read:
            limit = _README_CHARS if posixpath.basename(name).lower() == 'readme.md' else _NOTE_CHARS
            texts[name] = _hf_text_file(repo_id, name, target['revision'], limit)
        readme = texts.get('README.md', '')
        suggested = _guess_model_type(None, f'{repo_id} {target["path"]}', readme + ' ' + ' '.join(tags))
        return _format_huggingface(
            url, repo_id,
            kind=target['kind'], path=target['path'],
            pipeline=pipeline, gated=gated, tags=tags, card_text=card_text,
            files=files, texts=texts, suggested=suggested)
    except Exception as e:
        return f'Failed to fetch Hugging Face repo {repo_id}: {e}'


_FETCH_CACHE: dict[str, str] = {}


def fetch_link(url: str) -> str:
    """
    Fetch a CivitAI or Hugging Face model URL and return a text summary for the model.

    Identical URLs in one process reuse the first result, so a tool retry does
    not hit a rate-limited API again.
    """
    url = url.strip()
    cache_key = redact_url(url).lower()
    if cache_key in _FETCH_CACHE:
        return _FETCH_CACHE[cache_key]
    if _civitai_host(url):
        text = fetch_civitai_link(url)
    elif 'huggingface.co' in url.lower():
        text = fetch_huggingface_link(url)
    else:
        text = (
            f'Unsupported URL: {redact_url(url)}. '
            'Only civitai.com, civitai.red, and huggingface.co model links are supported.')
    limit = _HF_SUMMARY_CHARS if 'huggingface.co' in url.lower() else _SUMMARY_CHARS
    text = _truncate(text, limit)
    _FETCH_CACHE[cache_key] = text
    return text


def run_tool(name: str, arguments: str | dict) -> str:
    """Execute one assistant tool by name."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return f'Invalid tool arguments JSON: {e}'
    arguments = arguments or {}
    if name == 'fetch_link':
        url = arguments.get('url') or ''
        if not url:
            return 'Missing required argument: url'
        return fetch_link(url)
    if name == 'check_config':
        from dgenerate.assistant.check import check_config_text
        return check_config_text(arguments.get('config') or '')
    return f'Unknown tool: {name}'


# OpenAI-style tool list for llama.cpp chat completions.
LINK_TOOL_DEFINITIONS = [{
    'type': 'function',
    'function': {
        'name': 'fetch_link',
        'description': (
            'Fetch metadata for a CivitAI (civitai.com or civitai.red) model page or download URL, '
            'or a Hugging Face model, folder, or file URL. A repo URL lists weight and markdown '
            'paths and includes the README. A raw or blob URL for a markdown note returns that note, '
            'including its trigger word. Use this for --model-type, base model, trained words, '
            'download URLs, weight paths, or publisher usage notes.'),
        'parameters': {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'Full https URL to the model page or file on CivitAI or Hugging Face.',
                },
            },
            'required': ['url'],
        },
    },
}, {
    'type': 'function',
    'function': {
        'name': 'check_config',
        'description': (
            'Check a complete dgenerate config without loading a diffusion model. '
            'Returns the problems to fix, including memory flags for this GPU. '
            'Call this with the full config before the final reply, then again after each fix.'),
        'parameters': {
            'type': 'object',
            'properties': {
                'config': {
                    'type': 'string',
                    'description': 'The complete config text, including the shebang lines.',
                },
            },
            'required': ['config'],
        },
    },
}, {
    'type': 'function',
    'function': {
        'name': 'submit_config',
        'description': (
            'Submit the finished dgenerate config. This is how the config is returned. '
            'Do not also write it in the chat message. The config is checked first. '
            'If this returns problems, fix them and call submit_config again.'),
        'parameters': {
            'type': 'object',
            'properties': {
                'config': {
                    'type': 'string',
                    'description': 'The complete config text, including the shebang lines.',
                },
            },
            'required': ['config'],
        },
    },
}]


def tool_definitions(*, fetch: bool, check: bool) -> list:
    """The tools this request should offer. Order is fetch, then the config check."""
    chosen = []
    for item in LINK_TOOL_DEFINITIONS:
        name = item['function']['name']
        if name == 'fetch_link' and fetch:
            chosen.append(item)
        elif name in ('check_config', 'submit_config') and check:
            chosen.append(item)
    return chosen
