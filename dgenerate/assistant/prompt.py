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

import collections.abc
import dataclasses
import difflib
import enum
import functools
import os
import re
import types
import typing

import dgenerate.assistant.corpus as _corpus
import dgenerate.assistant.index as _index
import dgenerate.textprocessing as _textprocessing

CONTEXT_BUDGET = 40000
MAX_EXAMPLES = 4
MAX_GUIDES = 8
MAX_GUIDE_CHARS = 2800
MAX_EXAMPLE_CHARS = 6000
MAX_ARGUMENT_CHARS = 1200
MAX_REFERENCE_CHARS = 1500

_OPTION = re.compile(r'(?<![\w-])(--?[a-zA-Z][a-zA-Z0-9-]*)')
_OPTION_LINE = re.compile(r'^\s*(--?[a-zA-Z][a-zA-Z0-9-]*)', re.MULTILINE)
_SCHEDULER = re.compile(r'\b([A-Z]\w*Scheduler)\b')
_FENCE = re.compile(r'^\s*```')
_MEDIA_PATH = re.compile(r'(?<![\w/:.])(?:\.{1,2}[/\\])*(?:examples[/\\])?media[/\\]([\w.-]+)')

_FILE_EXT = (r'png|jpe?g|webp|gif|bmp|tiff?|mp4|webm|mov|mkv|avi|apng|wav|mp3|flac|ogg|'
             r'txt|json|safetensors|gguf|ckpt|pt|bin')
# A quoted path, or an unquoted token ending in a known file extension.
_REQUEST_FILE = re.compile(
    rf'"([^"\n]+\.(?:{_FILE_EXT}))"|\'([^\'\n]+\.(?:{_FILE_EXT}))\'|'
    rf'(?<![\w"\'])((?:[A-Za-z]:)?[^\s"\',;()]+\.(?:{_FILE_EXT}))(?![\w])',
    re.IGNORECASE)

SHEBANG = '#! /usr/bin/env dgenerate --file'

SYSTEM_PROMPT = """\
You write dgenerate config scripts. dgenerate is a command line tool for \
generating images, video, and audio with diffusion models. A config (.dgen file) \
is a script that runs one or more dgenerate invocations.

Config syntax:
- The first two lines are "{shebang}" and "#! dgenerate {version}".
- Lines starting with # are comments. Put a comment before the model path or \
after the last option. A # comment between options ends the invocation. Never \
put a comment between arguments.
- An invocation starts with the model path on its own line. Each following line \
starting with - is an option of that invocation. A blank line, a # comment, or \
any other line that does not start with - ends it.
- Values containing spaces or ; are quoted: --prompts "a red fox in the snow; blurry, low quality".
- A prompt is split at its first ; into a positive and a negative prompt. Everything \
before the ; is what the output should show: subject, action, style, lighting, camera, \
quality. Everything after it is only what the output must not show, like "blurry, \
low quality, distorted". Separate phrases with commas, never with ;. A prompt has at \
most one ;, and none if no negative prompt is needed. Models that do not use a \
negative prompt (check the examples) get no ; at all.
- Directives start with a backslash, for example \\set name value, \\setp name expr, \\print text.
- Jinja2 templates work anywhere: {{{{ variable }}}}, {{% if %}}...{{% endif %}}, {{% for %}}...{{% endfor %}}.
- A line that starts with {{% is one template continuation (a heredoc): \
the whole {{% if %}}...{{% endif %}} or {{% for %}}...{{% endfor %}} \
is rendered first, then the result is run. Nested if/for stay in that \
same render. \\set, \\setp, \\sete, \\download, and \\gen_seeds inside \
the block have not run yet, so {{{{ name }}}} in the same block is \
empty. last_images inside the block is the list from before the \
block, including later iterations of a {{% for %}}. When CIVIT_AI_TOKEN \
or HF_TOKEN is required, run the gating check only once at the top of the \
config file (after the shebang and version line): \\set the token on the \
first lines, then one early exit: {{% if not token.strip() %}} \\print Set HF_TOKEN environmental \
variable or pass --auth-token. \\exit {{% endif %}} (CivitAI: \
CIVIT_AI_TOKEN, no --auth-token). Do not repeat that check later in the \
file, inside {{% for %}}, or per invocation. After that, \\set model and the \
invocation are at the top level. Never wrap \\set, \\download, or a \
generation in {{% if token %}}. Official .dgen files say "to run this \
example" because they are examples; you are writing a config the user \
will run, so that \\print is a real instruction, never an example. \
When the repo is gated, write exactly: \\print Set HF_TOKEN environmental \
variable or pass --auth-token. When the config downloads from civitai.com or \
civitai.red, \
write exactly: \\print Set CIVIT_AI_TOKEN environmental variable. \
A Hugging Face repo does not use CIVIT_AI_TOKEN. Public checkpoints get \
no HF_TOKEN block. Put a per-item loop variable on the option line \
({{% for image in last_images %}} on --image-seeds) when you only \
need to expand a list.
- Environment variables expand as $VAR, ${{VAR}} or %VAR%.

Rules:
- Use only options, URI arguments, model types, and plugin names that appear in the \
reference material. Never invent an option. If something is not supported, say so in \
a # comment instead of guessing.
- Base the config on the closest example and keep its model, --model-type, --dtype, \
and required settings unless the request asks otherwise. Copy the memory flags from \
the Machine section row for that model onto the invocation, even when the closest \
example omits them or hides them in a total_memory check. A row that is only \
--quantizer is required. A row that is only --model-sequential-offload is required. \
Write that flag as its own line. Do not copy the example's \\setp gpu_memory_gib \
total_memory, {{% if not have_cuda() %}}, \\set offload, \\set optimization, \
or --model-type model {{{{ optimization }}}}. \
Klein 4B is not Flux.2 dev and not Klein 9B. A GGUF, FP8, FP4, NVFP4, or INT8 \
transformer uses the prequantized row. Follow the documented constraints on output \
size, frame counts, and guidance.
- Copy Hugging Face repo names (org/name) exactly as they appear in the reference material. \
Never invent a repo or change its organization or name. Do not use --model-type z-image-omni; \
Z-Image-Omni-Base is not released on Hugging Face yet.
- Inpainting models (repos with "inpainting" in the name) only work with --image-seeds \
"image.png;mask.png". A step that generates from a prompt alone needs a regular model of the \
same family, for example stabilityai/stable-diffusion-xl-base-1.0 for SDXL.
- HF_TOKEN is only for gated checkpoints. Omit it for public repos, even when \
the closest example has a token block. Public: SD 1.5, SD 2.1, SDXL (base, \
refiner, inpainting), FLUX.1-schnell, Kolors, Stable Cascade, the upscalers, \
pix2pix, Tongyi-MAI/Z-Image, Tongyi-MAI/Z-Image-Turbo, \
Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32, Qwen/Qwen-Image, \
black-forest-labs/FLUX.2-klein-4B. \
Gated: FLUX.1-dev, FLUX.1-Krea-dev, FLUX.1-Fill-dev, FLUX.1-Kontext-dev, \
black-forest-labs/FLUX.2-dev, black-forest-labs/FLUX.2-klein-9B, \
black-forest-labs/FLUX.2-klein-base-9B, black-forest-labs/FLUX.2-klein-9b-kv, \
SD3, SD3.5, \
LTX-2.5, LTX-Video, DeepFloyd IF. For a gated repo, \\set token %HF_TOKEN% once at the top, \
then {{% if not token.strip() and not '--auth-token' in injected_args %}} \
\\print Set HF_TOKEN environmental variable or pass --auth-token. \
\\exit {{% endif %}} once only. Do not wrap the generation. Do not duplicate \
the check in a loop or a second block. Do not write "example" \
in that \\print. CIVIT_AI_TOKEN is only for a civitai.com or civitai.red download \
link. \
Omit it when every model is a Hugging Face repo or a local file. For a \
CivitAI link: \\set civit_ai_token %CIVIT_AI_TOKEN% once at the top, then \
{{% if not civit_ai_token.strip() %}} \\print Set CIVIT_AI_TOKEN \
environmental variable. \\exit {{% endif %}} once only. No --auth-token check. \
Do not put token= or {{{{ civit_ai_token }}}} on the download URL. dgenerate \
appends CIVIT_AI_TOKEN from the environment to a civitai.com or civitai.red \
/api/download link that does not already have token=. Examples that write \
?token={{{{ civit_ai_token }}}} are not the pattern to copy.
- The Machine section is the memory plan for this computer, one row per model. \
Copy that row's flags onto the invocation. Do not write a total_memory \
check or an {{% if %}} block that picks the flags. SD3 full checkpoints use \
8-bit, not 4-bit. Z-Image full checkpoints are not quantized. Qwen-Image 2.1, \
Krea 2, Wan, LTX-2.5, HunyuanImage 2.1, HunyuanVideo, Kandinsky 5, and Motif \
stay on sequential offload with no quantizer. Hunyuan-DiT (--model-type hydit) \
gets no quantizer and no offload. An SDNQ checkpoint does not get a second --quantizer. \
A GGUF, FP8, FP4, NVFP4, or INT8 transformer uses the prequantized row. \
SD 1.5 and SDXL are not quantized: keep --model-cpu-offload when the closest \
example uses it.
- Paths marked (input file) in the request are inputs, for example for --image-seeds. \
Copy them exactly. They are never output paths: --output-path is a new directory name \
for the results. Paths in the examples that start with path/to/ are placeholders. If the request needs an input file it did not name, use a \
placeholder like path/to/input.png and say in a # comment that it must be replaced.
- To feed one invocation's results into a later one (refine, upscale, img2img, image to video, \
\\image_process), use the template variables set after every invocation: {{{{ last_images }}}} \
and {{{{ last_animations }}}} list the files it wrote. Write --image-seeds {{{{ quote(last_images) }}}}, \
or {{{{ quote(first(last_images)) }}}} for a single file, and "latents: {{{{ quote(last_images) }}}}" \
when the earlier invocation used --image-format pt. They are replaced by the next invocation, so \
save a result for later with \\set name {{{{ quote(first(last_images)) }}}}. Never guess the file \
names dgenerate writes. Every other option of the previous invocation is a last_ variable too, for \
example --prompts {{{{ format_prompt(last_prompts) }}}}. Use only variables and functions listed in \
the template reference.
- The prompts in the examples are short placeholders. Write a new, detailed prompt for the \
request in the style its model understands (see the prompt guide in the reference material), \
keeping everything the user asked for. Text that should appear in the image goes in single \
quotes inside the prompt, like a sign that says 'OPEN'. \
Prompt weighting ((word:1.3), word+) only works with --prompt-weighter. \
Without it those marks are literal characters. --prompt-weighter with a \
plain sentence and no (phrase:1.3) / ((phrase)) / word+ does nothing; \
write those marks on the subjects that matter, or omit the weighter. \
sd-embed is Automatic1111 / CivitAI syntax and the default, including \
SD3. compel is InvokeAI word+ / word++ syntax. Use a weighter when the \
user emphasizes a subject or pastes weighted syntax; do not decorate \
every quality word. Flux, flux-fill, flux-kontext, and flux2 never take a \
negative prompt: no ; in --prompts. flux2-klein-kv does not either. \
z-image, qwen-image, qwen-image-edit, and qwen-image-layered do. \
--guidance-scales on qwen-image is true CFG, not embedded guidance. \
--qwen-guidance-scale is that embedded guidance; omit it to leave it unset. \
--flux2-caption-upsample-temperature is full Flux.2 only. \
--flux2-text-encoder-out-layers, --z-image-cfg-normalization, and \
--z-image-cfg-truncation are optional and default to the pipeline. \
Z-Image ControlNet takes scale only; start and end other than 0 and 1 are an error. \
Do not pass --quantizer sdnq to Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32; \
that checkpoint is already quantized. --prompt-weighter is not supported \
on flux2, z-image, or qwen-image.
- Add short # comments explaining the choices that matter, above the model path, \
never between options. Never repeat the same comment. If you are unsure how to \
finish a step, write the invocation anyway; do not stall in a comment loop.
- A .gguf file is only a --transformer (or --unet, or --wan-second-transformer) \
replacement for Flux, SD3, Flux.2, Z-Image, Qwen-Image, LTX, or Wan. The first \
line is still the Hugging Face repo that supplies the VAE and text encoders, \
never the .gguf path. Do not set quantizer= on that URI and do not use \
--quantizer gguf. Flux.2 Klein, Qwen-Image, Z-Image, LTX-2.5, and Wan are \
detected from the file, including Comfy layouts, so no config= is required. \
--gen-seeds is a count of images; use 1 unless asked for more. \
--seeds 42 is the fixed seed 42.
- last_images is the stills the last step wrote. last_animations is its video \
file. Image models (including Flux Kontext and Flux Fill) and image-to-video \
always take last_images or the user's photo, never last_animations.
- output-file= on a processor only writes a debug image. Do not load that path \
in a later invocation; keep --control-image-processors on the ControlNet step.
- --image-format pt writes latents. Do not follow it with --image-seed-strengths; \
that option is img2img on a PNG/JPEG. Hires fix is generate a small image, then \
img2img larger. A two-step SDXL refine writes latents plus --denoising-end, then \
the refiner repo with --denoising-start. Prefer --sdxl-refiner on the base step.
- --output-size on an upscaler is the size fed in, not the final size. Do not \
shrink a 1024 image to 256 before x4.
- --schedulers help only prints names. Never use it as a generation.
- Reply with exactly one ```dgen code block containing the complete config, and nothing after it.
"""

LINK_TOOLS_SYSTEM = """\
Link tool:
- Call fetch_link for CivitAI or Hugging Face URLs in the Request section or the current \
editor config. After a Hugging Face repo result, you may call it once more on a raw or blob URL \
for a markdown note listed there. Do not call it for links that appear only in the reference \
material or examples, and do not call it on a tree page when the file list is already present.
- Call it on each request URL before writing the config, unless that exact URL was already \
fetched earlier in this conversation. If fetch_link fails, do not call it again for that URL. \
Write the config from the request and the retrieved examples.
- Treat fetch_link output as authoritative for CivitAI artifact type, baseModel, trainedWords, \
suggested --model-type, the checkpoint repo, download URLs, publisher usage notes, and Hugging Face \
weight paths. baseModel on the selected version chooses the family. Words in the publish title are \
the other versions on that page, not this link. A file marked [selected by fileId] is the file the \
URL asked for. Copy a download URL with no token= query. dgenerate appends CIVIT_AI_TOKEN from the \
environment. A Hugging Face repo id is only organization/name. Copy a listed weight as subfolder= \
and weight-name= inside one quoted value. An unquoted space splits that value into two tokens. \
Use the trigger word from the note beside that weight.
- Map LoRA and TextualInversion publishes to the suggested base --model-type plus --loras or \
embeddings, not a checkpoint line alone.
"""

CHECK_TOOL_SYSTEM = """\
Config check:
- Call check_config with the complete config text to see the problems. Do not pass a file path.
- Send the finished config by calling submit_config with that same text. \
Do not also write a ```dgen block. submit_config is how the config is returned.
- submit_config checks the config. If it lists problems, fix them and call submit_config again. \
When it accepts the config, stop.
- Do not paste a tool result into the config. Do not call these tools on an example \
from the reference material.
"""

_CLIP_GUIDE = (
    'Comma separated phrases, most important first: the subject and what it is doing, the setting, '
    'the medium or art style, lighting, color, camera, lens or composition, then a few quality words '
    'like "highly detailed, sharp focus". Keep it under about 60 words unless --prompt-weighter is '
    'set (sd-embed or compel), which also lifts the 77-token CLIP cutoff. To emphasize a subject, '
    'use --prompt-weighter sd-embed and (subject:1.3), or compel and subject+. Without a weighter, '
    'parentheses are plain text. Add a negative prompt with the usual defects, like "blurry, low '
    'quality, deformed, extra fingers, watermark, text".')
_SENTENCE_GUIDE = (
    'Two to four natural sentences that describe the scene in detail, then the style and lighting. '
    'A negative prompt is optional and short.')
_EDIT_GUIDE = (
    'An instruction that states the change and what to keep, like "replace the background with a snowy '
    'forest, keep the person\'s face and pose unchanged". No negative prompt.')

# How to write --prompts for each --model-type, added next to the request for the model types
# of the chosen examples.
PROMPT_GUIDES = {
    'sd': _CLIP_GUIDE,
    'sdxl': _CLIP_GUIDE,
    'kolors': _CLIP_GUIDE + ' Kolors also understands full sentences.',
    's-cascade': _CLIP_GUIDE,
    'sd3': _SENTENCE_GUIDE,
    'if': _SENTENCE_GUIDE,
    'flux': ('Natural sentences describing the subject, its appearance and action, the setting, '
             'composition, lighting, mood, and the style or camera. No negative prompt.'),
    'flux-fill': 'Describe the whole image as it should look, including what fills the masked area. '
                 'No negative prompt.',
    'flux-kontext': _EDIT_GUIDE,
    'flux2': ('Natural sentences describing the subject, its appearance and action, the setting, '
              'composition, lighting, mood, and the style or camera. No negative prompt. '
              'Full Flux.2 and Klein text-to-image take --image-seeds as reference images, '
              'with no mask and no strength. Klein inpaint takes an image and a mask, '
              'and reference= in that seed is an extra reference.'),
    'z-image': ('Natural sentences. Turbo uses guidance 0. Base uses a normal guidance scale. '
                'A short negative prompt is optional.'),
    'qwen-image': ('Natural sentences. --guidance-scales is true CFG. '
                   'A short negative prompt is optional; without one, guidance above 1 '
                   'still applies classifier-free guidance.'),
    'flux2-klein-kv': ('Natural sentences. No negative prompt and no guidance scale. '
                       '--image-seeds with no mask are reference images.'),
    'qwen-image-edit': ('An instruction that says what should change. The image seed is '
                        'the picture to edit, with no strength. A mask makes it inpaint. '
                        'images: a.png, b.png is edit-plus.'),
    'qwen-image-layered': ('A short description of the image. The image seed is required. '
                           'The result is several layers. --qwen-layered-layers defaults to 4.'),
    'pix2pix': _EDIT_GUIDE,
    'sdxl-pix2pix': _EDIT_GUIDE,
    'sd3-pix2pix': _EDIT_GUIDE,
    'wan': ('One paragraph of three to six plain sentences in the order things happen: the main action '
            'first, then specific movements and gestures, how the people and objects look, the setting, '
            'the camera angle and movement, and the lighting. Describe only what fits in the '
            'clip\'s length. --video-lengths is that length in seconds. --video-fps defaults to 16. '
            'Add a negative after one ; : "Bright tones, overexposed, static, blurred details, '
            'subtitles, worst quality, low quality, ugly, deformed, extra fingers, still picture". '
            '--output-size width and height must be divisible by 16.'),
    'wan-animate': ('A short description of how the character looks. The motion comes from the pose '
                    'or driving clip, not from a long action paragraph. No --video-lengths. '
                    'Guidance stays at 1. --video-fps defaults to 30.'),
    'wan-animate-2': ('A short description of how the character looks and the background. '
                      'The motion comes from wan-driving=, not from the prompt. '
                      'No --video-lengths. --video-fps defaults to 24.'),
    'ltx2': ('One paragraph of three to six plain sentences in the order things happen: the main action '
            'first, then specific movements and gestures, how the people and objects look, the setting, '
            'the camera angle and movement, and the lighting and color. Describe only what fits in the '
            'clip\'s length. --video-lengths is that length in seconds and snaps to 8k+1 frames. '
            '--video-fps defaults to 24. Width and height must be divisible by 32. '
            'LTX-2.5 (Lightricks/LTX-2.5-Diffusers) also makes audio, so end with a sentence '
            'about what is heard, like the rain, footsteps, or music, and put spoken words in single quotes. '
            'A negative prompt is optional, like "worst quality, inconsistent motion, blurry, jittery, '
            'distorted".'),
    'ltx-video': ('One paragraph of three to six plain sentences in the order things happen: the main action '
                  'first, then specific movements, how the people and objects look, the setting, the camera '
                  'angle and movement, and the lighting. Describe only what fits in the clip\'s length. '
                  '--video-lengths is that length in seconds and snaps to 8k+1 frames. --video-fps defaults '
                  'to 24. Width and height must be divisible by 32. No audio. A negative prompt is optional, '
                  'like "worst quality, inconsistent motion, blurry, jittery, distorted".'),
    'upscaler-x2': 'A short description of what the image shows.',
    'upscaler-x4': 'A short description of what the image shows.',
}

# Always in the context, so the model does not have to guess a repo when the examples use another family.
MODEL_TABLE = """\
### reference: models
The standard Hugging Face repo for each model family. Use the one the request names, and the \
examples for everything else it needs.
- Stable Diffusion 1.5: stable-diffusion-v1-5/stable-diffusion-v1-5, --model-type sd (the default). Public, no HF_TOKEN.
- Stable Diffusion 2.1: sd2-community/stable-diffusion-2-1, --model-type sd. Public, no HF_TOKEN.
- SDXL: stabilityai/stable-diffusion-xl-base-1.0, --model-type sdxl --variant fp16 --dtype float16. \
Public, no HF_TOKEN. Refine in the same invocation with --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0
- SDXL inpainting (needs --image-seeds "image;mask"): diffusers/stable-diffusion-xl-1.0-inpainting-0.1, \
--model-type sdxl --variant fp16 --dtype float16. Public, no HF_TOKEN.
- Stable Diffusion 3 Medium: stabilityai/stable-diffusion-3-medium-diffusers, --model-type sd3 --variant fp16. Gated, needs HF_TOKEN.
- Stable Diffusion 3.5: stabilityai/stable-diffusion-3.5-large or stabilityai/stable-diffusion-3.5-medium, --model-type sd3. Gated, needs HF_TOKEN.
- Stable Cascade: stabilityai/stable-cascade-prior, --model-type s-cascade, \
with --s-cascade-decoder "stabilityai/stable-cascade;dtype=float16". Public, no HF_TOKEN.
- Flux Schnell: black-forest-labs/FLUX.1-schnell, 4 steps, guidance 0, --model-type flux --dtype bfloat16. Public, no HF_TOKEN.
- Flux Dev: black-forest-labs/FLUX.1-dev, --model-type flux --dtype bfloat16. Gated, needs HF_TOKEN.
- Flux Krea Dev: black-forest-labs/FLUX.1-Krea-dev, --model-type flux --dtype bfloat16, guidance 4.5, 28 steps. \
Photographic, aesthetic finetune of Flux Dev. Gated, needs HF_TOKEN. \
This is not Krea 2.
- Krea 2: krea/Krea-2-Raw, --model-type krea2 --dtype bfloat16. Public, no HF_TOKEN. \
A Krea 2 LoRA uses this repo on the invocation line and the LoRA on --loras. \
Not FLUX.1-Krea-dev, and not Flux.2.
- Flux inpainting and outpainting: black-forest-labs/FLUX.1-Fill-dev, --model-type flux-fill. Gated, needs HF_TOKEN.
- Flux image editing: black-forest-labs/FLUX.1-Kontext-dev, --model-type flux-kontext. Gated, needs HF_TOKEN.
- Flux.2: black-forest-labs/FLUX.2-dev, --model-type flux2 --dtype bfloat16, guidance 4, 50 steps. \
Gated, needs HF_TOKEN. --image-seeds with no mask are reference images, not img2img, and have no strength. \
Output size snaps to a multiple of 16. \
GGUF: --transformer https://huggingface.co/city96/FLUX.2-dev-gguf/blob/main/flux2-dev-Q4_K_S.gguf (hidden width 6144).
- Flux.2 Klein 9B distilled: black-forest-labs/FLUX.2-klein-9B, --model-type flux2 --dtype bfloat16. \
Gated, needs HF_TOKEN. 4 steps, guidance 1. This is not FLUX.2-klein-base-9B. \
A CivitAI base "Flux.2 Klein 9B" uses this repo. \
- Flux.2 Klein 9B base: black-forest-labs/FLUX.2-klein-base-9B, --model-type flux2 --dtype bfloat16. \
Gated, needs HF_TOKEN. Guidance 4, 50 steps. \
Text-to-image references are --image-seeds with no mask. \
Inpaint uses --image-seeds with a mask, and reference= is image_reference. \
Distilled Klein ignores guidance above 1. Caption upsampling is full Flux.2 only. \
Klein 4B is public: black-forest-labs/FLUX.2-klein-4B, no HF_TOKEN, 4 steps. \
GGUF width 3072: --transformer https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/blob/main/flux-2-klein-4b-Q4_K_S.gguf. \
GGUF width 4096: --transformer https://huggingface.co/unsloth/FLUX.2-klein-base-9B-GGUF/blob/main/flux-2-klein-base-9b-Q4_K_S.gguf \
with parent black-forest-labs/FLUX.2-klein-base-9B.
- Z-Image Turbo: Tongyi-MAI/Z-Image-Turbo, --model-type z-image --dtype bfloat16, 8 steps, guidance 0. \
Public, no HF_TOKEN. Img2img strength 0.6. Inpaint strength 1.
- Z-Image Turbo SDNQ: Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32, --model-type z-image --dtype bfloat16, \
9 steps, guidance 0. Public, no HF_TOKEN. Do not pass --quantizer; it is already uint4 SVDQuant.
- Z-Image Base: Tongyi-MAI/Z-Image, --model-type z-image --dtype bfloat16, guidance 4, 28 to 50 steps. \
Public, no HF_TOKEN. \
The 2-step LoRA is --loras "alibaba-pai/Z-Image-Fun-Lora-Distill;weight-name=Z-Image-Fun-Lora-Distill-2-Steps-2603.safetensors;scale=1", \
2 steps, guidance 1. \
GGUF: --transformer https://huggingface.co/jayn7/Z-Image-GGUF/blob/main/z_image-Q4_K_S.gguf. \
The loader uses the Turbo transformer config because the shapes match.
- Z-Image Turbo GGUF: parent Tongyi-MAI/Z-Image-Turbo, \
--transformer https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/blob/main/z_image_turbo-Q4_K_S.gguf, \
8 steps, guidance 0. Public, no HF_TOKEN.
- Z-Image ControlNet union: --control-nets on z-image, one union file, scale 0.75. \
Text-to-image uses https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors \
with Tongyi-MAI/Z-Image-Turbo. Inpaint uses the 2.0 file from \
alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0. No img2img ControlNet.
- Qwen-Image: Qwen/Qwen-Image, --model-type qwen-image --dtype bfloat16, guidance 4, 50 steps. \
Public, no HF_TOKEN. --guidance-scales is true_cfg_scale. Img2img and inpaint strength 0.6. \
ControlNet text-to-image is --control-nets InstantX/Qwen-Image-ControlNet-Union. \
Inpaint is image;mask with InstantX/Qwen-Image-ControlNet-Inpainting (not Union). \
GGUF: --transformer https://huggingface.co/QuantStack/Qwen-Image-GGUF/blob/main/Qwen_Image-Q4_K_S.gguf.
- Flux.2 Klein KV: black-forest-labs/FLUX.2-klein-9b-kv, --model-type flux2-klein-kv --dtype bfloat16. \
Gated, needs HF_TOKEN. Reference images are --image-seeds with no mask and no strength. No guidance scale. \
GGUF: --transformer https://huggingface.co/QuantStack/FLUX.2-Klein-9B-KV-GGUF/blob/main/Flux-2-Klein-9B-KV-Q4_K_S.gguf. \
Width 4096 uses the Klein 9B transformer config, which matches this checkpoint.
- Qwen-Image Edit: Qwen/Qwen-Image-Edit, --model-type qwen-image-edit. The image seed is required. \
Qwen/Qwen-Image-Edit-2509 is edit-plus. A mask on the edit checkpoint is edit-inpaint. \
GGUF: --transformer https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF/blob/main/Qwen_Image_Edit-Q4_K_S.gguf. \
Edit uses the same transformer module as Qwen-Image.
- Qwen-Image Layered: Qwen/Qwen-Image-Layered, --model-type qwen-image-layered. One image in, several layers out. \
--qwen-layered-layers 4 and --qwen-layered-resolution 640 are the defaults. \
GGUF: --transformer https://huggingface.co/unsloth/Qwen-Image-Layered-GGUF/blob/main/qwen-image-layered-Q4_K_S.gguf. \
The extra addition embedding selects the layered config.
- Kolors: Kwai-Kolors/Kolors-diffusers, --model-type kolors --variant fp16. Public, no HF_TOKEN.
- Hunyuan-DiT: Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers, --model-type hydit --dtype float16, \
guidance 5, 50 steps, 1024. Public, no HF_TOKEN. No quantizer and no offload. \
This is not HunyuanVideo (--model-type hyvideo-15) and not HunyuanImage.
- HunyuanImage 2.1: hunyuanvideo-community/HunyuanImage-2.1-Diffusers, --model-type hyimage-21 \
--dtype bfloat16, guidance 3.5, 50 steps, 2048, --model-sequential-offload. No quantizer, \
including a GGUF or FP8 transformer. Public, no HF_TOKEN. \
Refiner: hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers, --model-type hyimage-21-refiner, \
the same offload. Distilled: hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers, \
still --model-type hyimage-21.
- DeepFloyd IF: DeepFloyd/IF-I-M-v1.0, --model-type if --variant fp16. Gated, needs HF_TOKEN.
- Animate a still or make a clip: Lightricks/LTX-2.5-Diffusers, --model-type ltx2, \
--guidance-scales 1, --video-fps 24, --model-sequential-offload, --animation-format mp4. \
Gated, needs HF_TOKEN. Clip length and frame rate are --video-lengths (seconds) and --video-fps. \
--video-lengths snaps to 8k+1 frames. Width and height must be divisible by 32. \
LTX-only options use the --ltx- prefix. \
When the user names a duration, use that many seconds, however long. When they do not, \
omit --video-lengths so the duration head chooses it. \
--ltx-latent-upscale is the two-stage pass in that same generation; --output-size must be divisible by 64. \
The last frame is last-frame= on --image-seeds, at strength 1. strength= is the condition weight. \
--image-seed-strengths fills LTX groups that omit strength=. \
Describe the sound in the prompt. Lightricks/LTX-Video is the older video-only model; use it only when the user names it. \
Distilled GGUF: --transformer https://huggingface.co/realrebelai/LTX-2.5_GGUFs/blob/main/LTX-2.5-Distilled-Q4_K_S.gguf, \
guidance 1, and no --inference-steps. The parent repo stays Lightricks/LTX-2.5-Diffusers. \
Dev (full) GGUF: --transformer https://huggingface.co/vantagewithai/LTX-2.5-GGUF/blob/main/dev/ltx-2.5-22b-dev-transformer-Q4_K_S.gguf, \
guidance 3, audio guidance 7, STG 1 on block 28, modality 3, and \
--scheduler FlowMatchEulerDiscreteScheduler;use-dynamic-shifting=true;shift-terminal=0.1 with --inference-steps 30.
- Wan video: Wan-AI/Wan2.1-T2V-1.3B-Diffusers, --model-type wan, --dtype bfloat16, \
--guidance-scales 5, --video-fps 16, --output-size 832x480, --animation-format mp4. \
Public, no HF_TOKEN. --video-lengths is seconds and snaps to 4k+1 frames \
(5 seconds at 16 fps is 81). Width and height must be divisible by 16. \
Put the negative after ; in --prompts. \
A still is image-to-video on Wan-AI/Wan2.1-I2V-14B-480P-Diffusers. \
last-frame= needs Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers at 1280x720. \
A Wan 2.1 I2V repo only embeds the first frame. \
A video seed path is video-to-video and rejects --video-lengths; trim with frame-end=. \
control=, mask=, and reference= are VACE on Wan-AI/Wan2.1-VACE-1.3B-diffusers. \
Wan 2.2 MoE is Wan-AI/Wan2.2-T2V-A14B-Diffusers with --wan-low-noise-guidance-scales. \
The VAE stays float32 and is not a --quantizer-map target. \
GGUF: --transformer https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/blob/main/wan2.1-t2v-14b-Q4_K_S.gguf \
with parent Wan-AI/Wan2.1-T2V-14B-Diffusers. \
Wan-Animate is --model-type wan-animate, Wan-AI/Wan2.2-Animate-14B-Diffusers, a character still plus \
wan-pose= and wan-face=, or wan-driving= with --wan-animate-preprocess. --video-fps 30. \
Leaving steps at 30 selects 20 and leaving guidance at 5 selects 1. --video-lengths is rejected. \
replace mode needs wan-background= and mask=. \
Wan-Animate-2 is --model-type wan-animate-2, a character still plus wan-driving= only. \
Wan-AI/Wan2.2-Animate-2-14B-Diffusers uses 40 steps. The distilled repo uses 10. \
--video-fps 24. --video-lengths is rejected. No pose, face, or background keywords.
- 4x upscaling with diffusion: stabilityai/stable-diffusion-x4-upscaler, --model-type upscaler-x4 --variant fp16. Public, no HF_TOKEN.
- 2x latent upscaling: stabilityai/sd-x2-latent-upscaler, --model-type upscaler-x2. Public, no HF_TOKEN.
- Instruction editing: timbrooks/instruct-pix2pix (--model-type pix2pix), \
diffusers/sdxl-instructpix2pix-768 (--model-type sdxl-pix2pix). Public, no HF_TOKEN.
Only use a --variant that the examples use with that repo."""

_MODEL_TYPE = re.compile(r'--model-type\s+"?([\w-]+)')

REPAIR_PROMPT = """\
dgenerate rejected the config:

{problems}

{references}Fix these problems and reply with the complete corrected config in one ```dgen code block."""


def options_in(text: str) -> list[str]:
    seen = []
    for option in _OPTION.findall(text):
        if option not in seen:
            seen.append(option)
    return seen


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + '\n[... truncated]'


def _section(chunk: _corpus.Chunk, limit: int) -> str:
    return f'### {chunk.kind}: {chunk.title}\n{_truncate(chunk.text, limit)}'


@dataclasses.dataclass
class RequestFile:
    written: str
    """The path as the user wrote it."""
    config_path: str
    """The path to put in the config, relative to the directory the config runs from."""
    exists: bool


def request_files(request: str, cwd: str, run_dir: str) -> list[RequestFile]:
    """
    File paths named in the request. Relative paths are taken as relative to
    ``cwd`` and rewritten relative to ``run_dir``. URLs are left out.
    """
    files = []
    for match in _REQUEST_FILE.finditer(request):
        written = next(g for g in match.groups() if g)
        if '://' in written or any(f.written == written for f in files):
            continue
        full = os.path.normpath(os.path.join(cwd, os.path.expanduser(written)))
        if os.path.isabs(written):
            config_path = written
        else:
            try:
                config_path = os.path.relpath(full, run_dir)
            except ValueError:
                config_path = full
            if config_path.replace('\\', '/').startswith('../../../'):
                config_path = full
        files.append(RequestFile(written, config_path.replace('\\', '/'), os.path.exists(full)))
    return files


def apply_request_files(config: str, files: list[RequestFile]) -> str:
    """
    Replace request paths the model copied verbatim with their run directory relative form.
    """
    for f in files:
        if f.written != f.config_path:
            config = re.sub(rf'(?<![\w./\\-]){re.escape(f.written)}(?![\w.-])',
                            lambda _: f.config_path, config)
    return config


def placeholder_media_paths(text: str) -> str:
    """
    Turn ``../../media/file`` paths in examples into ``path/to/file`` so the
    model does not treat the example media as files the user has.
    """
    return _MEDIA_PATH.sub(lambda m: f'path/to/{m.group(1)}', text)


_WEIGHT_REQUEST = re.compile(
    r'prompt[- ]weight|\bsd-embed\b|\bcompel\b|\bweight(?:ed|ing)\b', re.IGNORECASE)
_FLUX_REQUEST = re.compile(r'\bflux\b', re.IGNORECASE)
_COMPEL_REQUEST = re.compile(r'\bcompel\b', re.IGNORECASE)
_GATED_HF = re.compile(
    r'flux\.1-dev|\bflux[ -]?dev\b|flux\.1[ .-]*d\b|flux\.1-krea|(?<![\w/])krea(?!/)(?!\s*-?\s*2\b)|'
    r'flux[ -]?fill|flux\.1-fill|\bkontext\b|'
    r'flux\.2(?!-klein-(?:base-)?4b)|flux-2(?!-klein-(?:base-)?4b)|'
    r'flux2(?!-klein-(?:base-)?4b)|flux 2(?! klein(?: base)? 4b)|'
    r'klein-9b|klein 9b|'
    r'\bsd3\b|\bsd\s*3|stable-diffusion-3|stable diffusion 3|'
    r'\bltx2\b|\bltx\b|deep\s*floyd',
    re.IGNORECASE)
_PUBLIC_HF = re.compile(
    r'\bsdxl\b|stable diffusion xl|\bsd\s*1\.?5\b|\bschnell\b|\bkolors\b|'
    r'\bcascade\b|\bpix2pix\b|\bupscal',
    re.IGNORECASE)
_CIVITAI_REQUEST = re.compile(r'\bcivit\s*\.?ai\b|\bcivitai\.(?:com|red)\b', re.IGNORECASE)


def wants_prompt_weighting(request: str) -> bool:
    """Whether the request asked for sd-embed, compel, or prompt weighting."""
    return bool(_WEIGHT_REQUEST.search(request))


def _example_source(chunk: _corpus.Chunk) -> str:
    return (chunk.source or '').replace('\\', '/')


def first_draft_recipes(request: str, editor: str | None = None,
                        machine: tuple[bool, float, bool, bool] | None = None) -> str:
    """
    Rules placed after the retrieved examples so the first draft does not copy
    their ``\\print`` wording, a Flux negative, or a weighter with no marks.
    """
    lines = [
        '### write the config from these rules',
        'The examples and the manual say "to run this example" because those files are examples.',
        'Do not copy that sentence.',
        'Never put a # comment between options. A comment there ends the invocation. '
        'Put comments above the model path.',
        'HF_TOKEN is only for a gated checkpoint. A public repo gets no token block, '
        'even when an example for a different repo has one.',
        'Public, no HF_TOKEN: SD 1.5, SD 2.1, SDXL, FLUX.1-schnell, Kolors, Stable Cascade, '
        'Krea 2 (krea/Krea-2-Raw, --model-type krea2), Wan-AI, upscalers, pix2pix, '
        'Hunyuan-DiT (Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers, --model-type hydit), '
        'HunyuanImage 2.1 (hunyuanvideo-community/HunyuanImage-2.1-Diffusers, --model-type hyimage-21). '
        'Krea 2 is not FLUX.1-Krea-dev. Hunyuan-DiT is not HunyuanVideo and not HunyuanImage. '
        'Klein 4B (black-forest-labs/FLUX.2-klein-4B) is public.',
        'Gated, HF_TOKEN required: FLUX.1-dev, FLUX.1-Krea-dev, FLUX.1-Fill-dev, FLUX.1-Kontext-dev, '
        'black-forest-labs/FLUX.2-dev, black-forest-labs/FLUX.2-klein-9B, '
        'black-forest-labs/FLUX.2-klein-base-9B, black-forest-labs/FLUX.2-klein-9b-kv, '
        'SD3, SD3.5, '
        'LTX-2.5, LTX-Video, DeepFloyd IF. '
        'FLUX.2-klein-9B is the distilled 9B checkpoint. It is gated. It is not the 4B repo and not the base-9B repo.',
        'For a gated repo the \\print is exactly:',
        r'    \print Set HF_TOKEN environmental variable or pass --auth-token.',
        'Put the HF_TOKEN \\set and that single {% if %} \\print \\exit {% endif %} block '
        'only once at the top of the config (after the shebang lines), not inside loops '
        'or before each invocation.',
        'CIVIT_AI_TOKEN is only for a civitai.com or civitai.red link. A Hugging Face repo or a local file does not use it.',
        'That \\print is exactly: \\print Set CIVIT_AI_TOKEN environmental variable.',
        'Put the CIVIT_AI_TOKEN \\set and that single early-exit only once at the top of the config.',
        'Do not put token= or {{ civit_ai_token }} on the download URL. dgenerate appends '
        'CIVIT_AI_TOKEN from the environment. Copy the download URL without a token query.',
        'Neither token \\print contains the word "example".',
    ]
    if re.search(r'\b(animat\w*|video|ltx|wan|clip)\b', request, re.IGNORECASE):
        lines += [
            'Unless the user names Wan or LTX-Video, animate with Lightricks/LTX-2.5-Diffusers, '
            '--guidance-scales 1, --video-fps 24, --model-sequential-offload, --animation-format mp4. '
            'Width and height must be divisible by 32.',
            'Wan text or image to video is --model-type wan, a Wan-AI repo, --guidance-scales 5, '
            '--video-fps 16, --output-size 832x480, and a negative prompt after ;. Public, no HF_TOKEN. '
            'Wan-Animate is --model-type wan-animate. Wan-Animate-2 is --model-type wan-animate-2. '
            'Do not pass --video-lengths to either.',
            'LTX-only options use the --ltx- prefix, including '
            '--ltx-audio-guidance-scales, and --ltx-ic-lora. --video-lengths is seconds and snaps to 8k+1 frames. '
            'If the user names a duration, use that many seconds. '
            'If they do not, omit --video-lengths so the duration head chooses it. '
            'Do not turn a frame count such as 97 into the length unless they asked for 97 seconds. '
            'For a sharper clip, --ltx-latent-upscale in the same generation; --output-size must be divisible by 64.',
            'The last frame is last-frame= on --image-seeds, at strength 1. strength= is the condition weight. '
            '--image-seed-strengths fills LTX groups that omit strength=.',
            'Image-to-video uses --image-seeds {{ quote(first(last_images)) }} and describes the sound.',
        ]
    if _GATED_HF.search(request):
        lines.append(
            'This request uses a gated checkpoint. Include one HF_TOKEN early-exit at the top of the config only.')
    elif _PUBLIC_HF.search(request):
        lines.append('This request uses a public checkpoint. Do not add HF_TOKEN.')
    if _CIVITAI_REQUEST.search(request):
        lines.append(
            'This request downloads from CivitAI. Include one CIVIT_AI_TOKEN early-exit at the top of the config only.')
    else:
        lines.append('This request has no CivitAI link. Do not add CIVIT_AI_TOKEN.')
    if _FLUX_REQUEST.search(request):
        lines.append(
            'This request uses Flux. --prompts is the description only: no ";", no negative prompt.')
    if wants_prompt_weighting(request):
        if _COMPEL_REQUEST.search(request):
            lines += [
                'This request asks for compel weighting. Write the marks on this request\'s subjects:',
                '--prompt-weighter compel',
                '--prompts "(main subject)+ , (second subject)++"',
                'word+ and (phrase)+ are compel. A plain sentence with --prompt-weighter does nothing.',
            ]
        else:
            lines += [
                'This request asks for prompt weighting. Write sd-embed marks on this request\'s subjects:',
                '--prompt-weighter sd-embed',
                '--prompts "((requested style)) of a (main subject:1.3), (second subject:1.2), setting"',
                '(phrase:1.3) and ((phrase)) are sd-embed. word+ is compel, not sd-embed. '
                'A plain sentence with --prompt-weighter does nothing.',
            ]
    hint = memory_draft(request, editor, machine)
    if hint:
        lines.append(hint)
    return '\n'.join(lines)


def prefer_weighting_examples(request: str, examples: list[_corpus.Chunk],
                              index: _index.Index) -> list[_corpus.Chunk]:
    """
    Put a prompt-weighting example first when the request asks for weights,
    so the first draft copies ``(phrase:1.3)`` instead of a plain sentence.
    """
    if not wants_prompt_weighting(request):
        return examples
    already = [c for c in examples if 'prompt_weighting' in _example_source(c)]
    rest = [c for c in examples if 'prompt_weighting' not in _example_source(c)]
    if already:
        return already + rest
    want_flux = 'flux' in request.lower()
    want_compel = bool(re.search(r'\bcompel\b', request, re.IGNORECASE))
    best, best_score = None, -1
    for chunk in index.chunks:
        if chunk.kind != 'example':
            continue
        src = _example_source(chunk)
        if 'prompt_weighting' not in src:
            continue
        score = 0
        if want_flux and 'flux' in src:
            score += 3
        if want_compel and 'compel' in src:
            score += 2
        if not want_compel and 'sd-embed' in src:
            score += 2
        if 'llm4gen' in src:
            score -= 2
        if score > best_score:
            best, best_score = chunk, score
    if best is None:
        return examples
    return [best] + [c for c in rest if _example_source(c) != _example_source(best)][:MAX_EXAMPLES - 1]


def _kind(hint) -> str:
    """
    A short name for a template variable's type.
    """
    args = [a for a in typing.get_args(hint) if a is not type(None)]
    if typing.get_origin(hint) in (typing.Union, types.UnionType) and len(args) == 1:
        hint = args[0]
    origin = typing.get_origin(hint) or hint
    if origin is bool:
        return 'bool'
    if origin is str:
        return 'str'
    if origin in (int, float):
        return 'number'
    if origin is tuple:
        return 'tuple'
    if origin is dict:
        return 'dict'
    if isinstance(origin, type) and issubclass(origin, enum.Enum):
        return 'enum'
    if isinstance(origin, type) and issubclass(origin, collections.abc.Iterable):
        return 'list'
    return 'value'


@functools.cache
def template_reference() -> str:
    """
    Every template variable and function a config can use, read from dgenerate's config runner.
    """
    import dgenerate.batchprocess as _batchprocess

    runner = _batchprocess.ConfigRunner()
    variables = runner._generate_template_variables_with_types()
    last = ', '.join(f'{name} ({_kind(hint)})' for name, (hint, _) in variables.items() if name.startswith('last_'))
    other = ', '.join(name for name in variables if not name.startswith('last_'))
    functions = ', '.join(sorted(runner.template_functions))
    return (
        '### reference: template variables and functions\n'
        'After every invocation, each of its options is available to later lines as a last_ variable. '
        'Options that take several values are lists, unset options are None or empty. '
        'last_images and last_animations list the files it wrote, \\image_process sets them too.\n'
        f'{last}\n\n'
        f'Other variables: {other}\n\n'
        f'Template functions: {functions}\n\n'
        'Common uses:\n'
        '--image-seeds {{ quote(last_images) }}\n'
        '--image-seeds {{ quote(first(last_animations)) }}\n'
        '--prompts {{ format_prompt(last_prompts) }}\n'
        "--seeds {{ last_seeds | join(' ') }} --seeds-to-images (pairs each seed with the image it made, "
        'without --seeds-to-images every seed runs on every image)\n'
        '\\set still {{ quote(first(last_images)) }}\n'
        '\\image_process {{ quote(first(last_images)) }} --output upscaled.png '
        '--processors upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/'
        'realesr-general-x4v3.pth')


def _matching_guides(request: str, hits: list[_index.SearchHit], index: _index.Index) -> list[_corpus.Chunk]:
    """
    Guide sections that match the request, retrieved hits first, then title-word overlap.
    """
    chosen, seen = [], set()
    for chunk in index.chunks:
        if chunk.kind != 'guide' or chunk.id in seen:
            continue
        title = chunk.title.lower()
        if 'first draft' in title or (
                wants_prompt_weighting(request) and 'prompt weighting' in title):
            seen.add(chunk.id)
            chosen.append(chunk)
    for hit in hits:
        if hit.chunk.kind == 'guide' and hit.chunk.id not in seen:
            seen.add(hit.chunk.id)
            chosen.append(hit.chunk)
    query = set(_index.tokenize(request))
    extra = []
    for chunk in index.chunks:
        if chunk.kind != 'guide' or chunk.id in seen:
            continue
        extra.append((len(query & set(_index.tokenize(chunk.title))), chunk))
    extra.sort(key=lambda item: item[0], reverse=True)
    for overlap, chunk in extra:
        if overlap < 2 or len(chosen) >= MAX_GUIDES:
            break
        chosen.append(chunk)
    return chosen[:MAX_GUIDES]


def build_context(request: str, hits: list[_index.SearchHit], index: _index.Index) -> str:
    """
    Pick the closest examples, help for every option and scheduler they and the request use,
    the template reference, and the best matching documentation, within
    :data:`CONTEXT_BUDGET` characters.
    """
    examples, references = [], []
    seen_sources = set()
    for hit in hits:
        chunk = hit.chunk
        if chunk.kind == 'example':
            if chunk.source not in seen_sources and len(examples) < MAX_EXAMPLES:
                seen_sources.add(chunk.source)
                examples.append(chunk)
        elif chunk.kind != 'guide':
            references.append(chunk)

    examples = prefer_weighting_examples(request, examples, index)
    sections = [placeholder_media_paths(_section(c, MAX_EXAMPLE_CHARS)) for c in examples]
    for chunk in _matching_guides(request, hits, index):
        sections.append(_section(chunk, MAX_GUIDE_CHARS))
    sections.append(template_reference())
    used = sum(len(s) for s in sections)

    included = set()
    wanted_options = options_in(request)
    for chunk in examples:
        wanted_options += [o for o in _OPTION_LINE.findall(chunk.text) if o not in wanted_options]
    wanted = [index.argument(o) for o in wanted_options]
    for text in [request] + [c.text for c in examples]:
        wanted += [index.scheduler(name) for name in _SCHEDULER.findall(text)]
    for chunk in wanted:
        if chunk is None or chunk.id in included:
            continue
        section = _section(chunk, MAX_ARGUMENT_CHARS)
        if used + len(section) > CONTEXT_BUDGET * 0.8:
            break
        included.add(chunk.id)
        sections.append(section)
        used += len(section)

    for chunk in references:
        if chunk.id in included:
            continue
        section = _section(chunk, MAX_REFERENCE_CHARS)
        if used + len(section) > CONTEXT_BUDGET:
            continue
        included.add(chunk.id)
        sections.append(section)
        used += len(section)

    sections.append(MODEL_TABLE)

    guide = prompt_guide(examples)
    if guide:
        sections.append(guide)

    return '\n\n'.join(sections)


def prompt_guide(examples: list[_corpus.Chunk]) -> str:
    """
    How to write the prompt for the model types the examples use. Invocations
    without --model-type are Stable Diffusion.
    """
    model_types = []
    for chunk in examples:
        for model_type in _MODEL_TYPE.findall(chunk.text) or ['sd']:
            if model_type in PROMPT_GUIDES and model_type not in model_types:
                model_types.append(model_type)
    if not model_types:
        return ''
    lines = [f'- --model-type {t}: {PROMPT_GUIDES[t]}' for t in model_types]
    return '### reference: prompt guide\nHow to write --prompts for each model type:\n' + '\n'.join(lines)


def user_message(request: str, context: str, files: list[RequestFile],
                 editor: str | None = None,
                 machine: tuple[bool, float, bool, bool] | None = None) -> str:
    for f in files:
        request = re.sub(rf'(?<![\w./\\-]){re.escape(f.written)}(?![\w.-])',
                         lambda _: f'"{f.config_path}" (input file)', request)
    if machine is None:
        machine = machine_state()
    body = (f'Reference material from the dgenerate documentation and examples:\n\n{context}\n\n')
    if editor and editor.strip():
        body += (
            'Current config in the editor. Apply the request as an edit of this config. '
            'Return the complete revised config. Keep every line the request does not ask to change. '
            'Do not wrap a line, and do not insert a blank line inside an invocation: a blank line '
            'ends it, and the next option line is then run with no model path. Leave every {{ }} '
            'expression on the line it already occupies.\n\n'
            f'{editor.strip()}\n\n'
            f'Request:\n{request}\n')
        hint = memory_draft(request, editor, machine)
        if hint:
            body += '\n' + hint + '\n'
        return body
    return body + f'Request:\n{request}\n\n{first_draft_recipes(request, machine=machine)}'


def inputs_used_as_output(config: str, files: list[RequestFile]) -> list[str]:
    """
    Request input files the model wrote into an output option.
    """
    problems = []
    for match in re.finditer(r'^\s*(-o|-op|--output-path|--output-prefix)\s+(.+)$', config, re.MULTILINE):
        value = match.group(2).strip().strip('"\'')
        for f in files:
            if value in (f.config_path, f.written):
                problems.append(f'{match.group(1)} is set to the input file {value}. Use a new '
                                f'output directory name instead, for example a short name for the result.')
    return problems


_POSITIVE_QUALITY = re.compile(
    r'\b(highly detailed|sharp focus|masterpiece|best quality|high quality|ultra detailed|'
    r'high resolution|[48]k)\b', re.IGNORECASE)


def split_prompts(config: str) -> list[str]:
    """
    Prompt values that put positive text in the negative prompt: more than one ``;``,
    or quality words after the ``;``.
    """
    problems = []
    for match in re.finditer(r'(?<![\w-])(--[\w-]*prompts)[ \t]+(.+)$', config, re.MULTILINE):
        try:
            values = _textprocessing.shell_parse(
                match.group(2), expand_home=False, expand_vars=False, expand_glob=False)
        except _textprocessing.ShellParseSyntaxError:
            values = [match.group(2)]
        for value in values:
            if value.startswith('-'):
                break
            text = re.sub(r'<[^<>]*>', '', value)
            if text.count(';') > 1:
                problems.append(
                    f'{match.group(1)} "{value}" has more than one ;. Only the first ; is the split, '
                    f'so everything after it becomes the negative prompt. Join the positive phrases '
                    f'with commas, and after the one ; keep only what the output must not show.')
                continue
            negative = text.partition(';')[2]
            quality = sorted({m.group(1).lower() for m in _POSITIVE_QUALITY.finditer(negative)})
            if quality:
                problems.append(
                    f'{match.group(1)} "{value}" puts {", ".join(quality)} after the ;, in the negative '
                    f'prompt, so the model avoids them. Move them before the ; and keep only defects '
                    f'like "blurry, low quality" after it.')
    return problems


@dataclasses.dataclass(frozen=True)
class MemorySpec:
    """Flags one model needs on this computer. ``nothing`` means the examples load it as-is."""

    offload: str | None = None
    second_offload: str | None = None
    quantizer: str | None = None
    quantizer_map: tuple[str, ...] | None = None
    forbid_quantizer: bool = False
    nothing: bool = False


def _quant_uri(cuda: bool, bitsandbytes: bool, sdnq: bool) -> str | None:
    """4-bit quantizer URI. bitsandbytes is preferred over sdnq. CPU has neither."""
    if not cuda:
        return None
    if bitsandbytes:
        return 'bnb;bits=4'
    if sdnq:
        return 'sdnq;type=int4'
    return None


def _spec_text(spec: MemorySpec) -> str:
    parts = []
    if spec.offload == 'sequential':
        parts.append('--model-sequential-offload')
    elif spec.offload == 'cpu':
        parts.append('--model-cpu-offload')
    if spec.second_offload == 'cpu':
        parts.append('--second-model-cpu-offload')
    if spec.quantizer:
        quantizer = f'--quantizer {spec.quantizer}'
        if spec.quantizer_map:
            quantizer += ' --quantizer-map ' + ' '.join(spec.quantizer_map)
        parts.append(quantizer)
    if parts:
        return ' and '.join(parts)
    if spec.nothing:
        return 'no offload and no quantizer'
    return ''


def _memory_spec(kind: str | None, cuda: bool, gib: float, bitsandbytes: bool, sdnq: bool) -> MemorySpec | None:
    """
    The example's memory flags for one kind of checkpoint.

    Cutoffs are the ``gpu_memory_gib`` tests in ``examples/``. SD3 full checkpoints
    use 8-bit above 24 GiB and never sdnq. Z-Image full checkpoints offload at
    16 GiB and below and are not quantized. Klein 4B quantizes above 12 GiB.
    Flux.2 dev and Klein 9B quantize above 24 GiB.
    """
    quantizer = _quant_uri(cuda, bitsandbytes, sdnq)
    if kind in ('sequential', 'ltx-full', 'flux-comfy'):
        return MemorySpec('sequential', forbid_quantizer=True)
    if kind in ('ltx-gguf', 'packed-cpu'):
        return MemorySpec('cpu', forbid_quantizer=True)
    if kind == 'cascade':
        return MemorySpec('cpu', second_offload='cpu', forbid_quantizer=True)
    if kind == 'kolors':
        if bitsandbytes and cuda and gib > 15:
            return MemorySpec(quantizer='bnb;bits=4')
        if bitsandbytes and cuda and gib > 7:
            return MemorySpec(quantizer='bnb;bits=4;bits4-quant-type=nf4;bits4-use-double-quant=True')
        if sdnq and cuda and gib > 15:
            return MemorySpec(quantizer='sdnq;type=int4')
        return MemorySpec('sequential')
    if kind == 'sd3-full':
        if cuda and bitsandbytes and gib > 24:
            return MemorySpec(quantizer='bnb;bits=8')
        return MemorySpec('sequential')
    if kind == 'sd3-pre':
        mapped = quantizer if quantizer and gib > 15 else None
        return MemorySpec('cpu', quantizer=mapped, quantizer_map=('text_encoder_3',) if mapped else None)
    if kind == 'flux-full':
        if quantizer and gib > 15:
            return MemorySpec(quantizer=quantizer)
        return MemorySpec('sequential')
    if kind == 'flux-control':
        if quantizer and gib > 24:
            return MemorySpec(quantizer=quantizer)
        return MemorySpec('sequential')
    if kind == 'flux-gguf':
        mapped = quantizer if quantizer and gib > 15 else None
        return MemorySpec('cpu', quantizer=mapped, quantizer_map=('text_encoder',) if mapped else None)
    if kind == 'flux2-heavy':
        if quantizer and gib > 24:
            return MemorySpec(quantizer=quantizer)
        return MemorySpec('sequential')
    if kind == 'klein4-full':
        if quantizer and gib > 12:
            return MemorySpec(quantizer=quantizer)
        return MemorySpec('sequential')
    if kind == 'flux2-pre-dev':
        mapped = quantizer if quantizer and gib > 16 else None
        return MemorySpec('sequential', quantizer=mapped, quantizer_map=('text_encoder',) if mapped else None)
    if kind == 'flux2-pre-klein':
        mapped = quantizer if quantizer and gib > 12 else None
        return MemorySpec('sequential', quantizer=mapped, quantizer_map=('text_encoder',) if mapped else None)
    if kind == 'qwen-full':
        if quantizer and gib > 24:
            return MemorySpec(quantizer=quantizer)
        return MemorySpec('sequential')
    if kind == 'qwen-pre':
        mapped = quantizer if quantizer and gib > 16 else None
        return MemorySpec('sequential', quantizer=mapped, quantizer_map=('text_encoder',) if mapped else None)
    if kind == 'zimage-full':
        if not cuda or gib <= 16:
            return MemorySpec('sequential', forbid_quantizer=True)
        return MemorySpec(nothing=True, forbid_quantizer=True)
    if kind == 'zimage-pre':
        mapped = quantizer if quantizer and gib > 12 else None
        offload = 'sequential' if (not cuda or gib <= 16) else None
        return MemorySpec(offload, quantizer=mapped, quantizer_map=('text_encoder',) if mapped else None)
    if kind == 'zimage-sdnq':
        if not cuda or gib <= 16:
            return MemorySpec('cpu', forbid_quantizer=True)
        return MemorySpec(nothing=True, forbid_quantizer=True)
    if kind == 'hydit':
        return MemorySpec(nothing=True, forbid_quantizer=True)
    return None


_SEQUENTIAL_TYPES = frozenset({
    'krea2', 'wan', 'wan-animate', 'wan-animate-2', 'ltx2',
    'hyvideo-15', 'motif-video', 'kandinsky5-image', 'kandinsky5-video',
    'hyimage-21', 'hyimage-21-refiner', 'qwen-image-21',
    'if', 'ifs', 'ifs-img2img',
})
_UNCHECKED_TYPES = frozenset({
    'sd', 'pix2pix', 'sdxl', 'sdxl-pix2pix', 'upscaler-x2', 'upscaler-x4',
    '', 'z-image-omni',
})
_FLUX_TYPES = frozenset({'flux', 'flux-fill', 'flux-kontext'})
_QWEN_TYPES = frozenset({'qwen-image', 'qwen-image-edit', 'qwen-image-layered'})
_PREQUANTIZED = re.compile(r'\.gguf|fp8|fp4|nvfp4|int8|q\d-k|q4-0|q8-0', re.IGNORECASE)


def _norm_blob(*parts: str | None) -> str:
    return ' '.join(part or '' for part in parts).lower().replace('_', '-')


def _is_prequantized(*parts: str | None) -> bool:
    return bool(_PREQUANTIZED.search(_norm_blob(*parts)))


def memory_kind(model_type: str, model_path: str, transformer: str | None = None,
                text_encoders: collections.abc.Iterable[str] | None = None,
                control_loras: collections.abc.Iterable[str] | None = None) -> str | None:
    """Which memory row an invocation uses. ``None`` means the examples set no flags."""
    mt = (model_type or '').lower()
    if mt in _UNCHECKED_TYPES:
        return None
    if mt in _SEQUENTIAL_TYPES:
        return 'sequential'
    if mt == 'ltx-video':
        return 'ltx-gguf' if _is_prequantized(transformer) else 'ltx-full'
    if mt == 's-cascade':
        return 'cascade'
    if mt == 'kolors':
        return 'kolors'
    if mt == 'hydit':
        return 'hydit'
    prequantized = _is_prequantized(transformer)
    text_prequantized = _is_prequantized(*(text_encoders or ()))
    names = _norm_blob(model_path, transformer)
    if mt == 'z-image':
        if 'sdnq' in _norm_blob(model_path):
            return 'zimage-sdnq'
        return 'zimage-pre' if prequantized else 'zimage-full'
    if mt in ('sd3', 'sd3-pix2pix'):
        return 'sd3-pre' if prequantized else 'sd3-full'
    if mt in _FLUX_TYPES:
        if prequantized and text_prequantized:
            return 'packed-cpu'
        if prequantized and '.gguf' in _norm_blob(transformer):
            return 'flux-gguf'
        if prequantized:
            return 'flux-comfy'
        if any(control_loras or ()):
            return 'flux-control'
        return 'flux-full'
    if mt in ('flux2', 'flux2-klein-kv'):
        klein = 'klein' in names or mt == 'flux2-klein-kv'
        if prequantized and text_prequantized:
            return 'packed-cpu'
        if prequantized and klein:
            return 'flux2-pre-klein'
        if prequantized:
            return 'flux2-pre-dev'
        # klein-base-4B is still the 4B checkpoint. The hyphen is not always before 4b.
        if 'klein' in names and re.search(r'4b', names):
            return 'klein4-full'
        return 'flux2-heavy'
    if mt in _QWEN_TYPES:
        return 'qwen-pre' if prequantized else 'qwen-full'
    return None


_KIND_LABEL = {
    'sequential': 'Krea 2, Wan, LTX-2.5, Qwen-Image 2.1, Hunyuan, Kandinsky, Motif, or DeepFloyd',
    'ltx-full': 'LTX-Video full checkpoint',
    'ltx-gguf': 'LTX-Video GGUF transformer',
    'cascade': 'Stable Cascade',
    'kolors': 'Kolors',
    'sd3-full': 'SD3 or SD3 pix2pix full checkpoint',
    'sd3-pre': 'SD3 prequantized transformer',
    'flux-full': 'Flux.1 full checkpoint',
    'flux-control': 'Flux.1 Control LoRA',
    'flux-gguf': 'Flux.1 GGUF transformer',
    'flux-comfy': 'Flux.1 FP8 or Comfy transformer',
    'flux2-heavy': 'Flux.2 dev, Klein 9B, or Klein KV full checkpoint',
    'klein4-full': 'Flux.2 Klein 4B full checkpoint (--model-type flux2, not flux)',
    'flux2-pre-dev': 'Flux.2 dev prequantized transformer',
    'flux2-pre-klein': 'Flux.2 Klein prequantized transformer',
    'packed-cpu': 'Flux.2 with a quantized transformer and a quantized text encoder',
    'qwen-full': 'Qwen-Image, Edit, or Layered full checkpoint',
    'qwen-pre': 'Qwen-Image prequantized transformer',
    'zimage-full': 'Z-Image full checkpoint',
    'zimage-pre': 'Z-Image prequantized transformer',
    'hydit': 'Hunyuan-DiT (--model-type hydit, not hyvideo-15 or hyimage-21)',
    'zimage-sdnq': 'Z-Image SDNQ checkpoint',
}


def _request_prequantized(text: str) -> bool:
    return bool(re.search(r'\.gguf|\bfp8\b|\bfp4\b|nvfp4|\bint8\b|q\d-k', text))


def _request_kinds(request: str) -> list[str]:
    """Memory rows named by the request, most specific model first."""
    text = request.lower().replace('_', '-')
    pre = _request_prequantized(text)
    found: list[str] = []

    def add(kind: str) -> None:
        if kind not in found:
            found.append(kind)

    # klein-base-4B is still the 4B checkpoint. The hyphen is not always before 4b.
    klein4 = bool(re.search(r'klein', text) and re.search(r'4b', text))
    flux2 = bool(re.search(r'flux\.2|flux2|flux-2|flux 2', text))
    if klein4:
        add('flux2-pre-klein' if pre else 'klein4-full')
    elif flux2:
        if 'klein' in text:
            add('flux2-pre-klein' if pre else 'flux2-heavy')
        else:
            add('flux2-pre-dev' if pre else 'flux2-heavy')
    elif re.search(r'\bflux\b|flux\.1|flux1', text):
        if re.search(r'control[- ]lora', text):
            add('flux-control')
        elif 'gguf' in text:
            add('flux-gguf')
        elif pre:
            add('flux-comfy')
        else:
            add('flux-full')
    if re.search(r'z-image|zimage|z image', text):
        if 'sdnq' in text:
            add('zimage-sdnq')
        elif pre:
            add('zimage-pre')
        else:
            add('zimage-full')
    if re.search(r'qwen-image-2\.1|qwen-image-21(?!\d)|qwen image 2\.1', text):
        add('sequential')
    elif re.search(r'qwen-image|qwen image', text):
        add('qwen-pre' if pre else 'qwen-full')
    if re.search(r'\bsd3\b|sd3\.5|stable-diffusion-3|stable diffusion 3', text):
        add('sd3-pre' if pre else 'sd3-full')
    if re.search(r'\bkolors\b', text):
        add('kolors')
    if re.search(r'\bkrea\s*-?\s*2\b|krea2', text) and not re.search(r'flux\.1\b|flux1\b', text):
        add('sequential')
    if re.search(r'ltx-2|ltx2(?!\d)|ltxv2|ltxv-2\.\d', text):
        add('sequential')
    elif re.search(r'ltx-video|ltxvideo|\bltxv\b|ltx video', text):
        add('ltx-gguf' if 'gguf' in text else 'ltx-full')
    if re.search(r'\bwan\b|wan\d|wan-\d', text):
        add('sequential')
    if re.search(r'hunyuan-?dit|\bhydit\b|hunyuan dit', text):
        add('hydit')
    image21 = bool(re.search(r'hunyuan-?image|hyimage', text)) and not re.search(r'3\.0|image-3', text)
    video15 = bool(re.search(r'hunyuan-?video-?1\.5|hunyuanvideo1\.5|hyvideo', text))
    kandinsky5 = bool(re.search(r'kandinsky-5|kandinsky5', text)) and not re.search(
        r'kandinsky-6|kandinsky6', text)
    if image21 or video15 or kandinsky5 or re.search(r'motif-video|\bmotif\b|deepfloyd|\bif-i-', text):
        add('sequential')
    elif re.search(r'\bcascade\b', text):
        add('cascade')
    return found


def _config_kinds(config: str) -> list[str]:
    """Memory rows of the invocations in an open config."""
    kinds: list[str] = []
    model_path = ''
    model_type = ''
    transformer = ''
    text_encoders: list[str] = []
    control: list[str] = []

    def finish() -> None:
        nonlocal model_path, model_type, transformer, text_encoders, control
        if model_path or model_type:
            kind = memory_kind(model_type, model_path, transformer or None, text_encoders, control)
            if kind and kind not in kinds:
                kinds.append(kind)
        model_path = model_type = transformer = ''
        text_encoders = []
        control = []

    for raw in config.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith(('#', '\\', '{%', '{{')):
            continue
        if stripped.startswith('--'):
            name, _, value = stripped.partition(' ')
            if name == '--model-type':
                model_type = value.split()[0] if value else ''
            elif name == '--transformer':
                transformer = value
            elif name == '--text-encoders':
                text_encoders.append(value)
            elif name == '--control-loras':
                control.append(value)
            continue
        finish()
        head, _, rest = stripped.partition(' ')
        model_path = head
        if '--model-type' in rest:
            model_type = rest.split('--model-type', 1)[1].split()[0]
    finish()
    return kinds


def memory_draft(request: str, editor: str | None,
                 machine: tuple[bool, float, bool, bool] | None) -> str:
    """
    The exact flags for the model this request is writing, placed after the examples
    so the first draft copies them instead of the example's memory ``{% if %}``.
    """
    if machine is None:
        machine = machine_state()
    cuda, gib, bitsandbytes, sdnq = machine
    kinds = _request_kinds(request) or _config_kinds(editor or '')
    rows = []
    for kind in kinds:
        spec = _memory_spec(kind, cuda, gib, bitsandbytes, sdnq)
        if spec is None:
            continue
        flags = _spec_text(spec)
        label = _KIND_LABEL.get(kind, kind)
        if not flags or flags == 'no offload and no quantizer':
            rows.append(f'- {label}: do not add --quantizer or an offload flag.')
        elif spec.forbid_quantizer:
            rows.append(f'- {label}: write {flags}. Do not pass --quantizer.')
        else:
            rows.append(f'- {label}: write {flags}.')
    if not rows:
        return ''
    return (
        'Memory flags for this request. Write each flag as its own line under the model path. '
        'Do not copy the example memory block: no total_memory, no \\setp gpu_memory_gib, '
        'no {% if have_cuda() %}, no \\set offload, no \\set optimization, '
        'and no {{ offload }} or {{ optimization }} on the --model-type line. '
        f'This GPU has {gib:.1f} GiB.\n' + '\n'.join(rows))


def _offload_ok(required: str | None, sequential: bool, cpu: bool, group: bool) -> bool:
    if required is None:
        return True
    if required == 'sequential':
        return sequential or group
    if required == 'cpu':
        return cpu or sequential or group
    return False


def _quantizer_ok(spec: MemorySpec, quantizer: str | None, quantizer_map: collections.abc.Iterable[str] | None) -> bool:
    if not spec.quantizer:
        return True
    if not quantizer or spec.quantizer not in quantizer:
        return False
    mapped = set(quantizer_map or ())
    if spec.quantizer_map:
        return set(spec.quantizer_map).issubset(mapped)
    if not mapped:
        return True
    return bool(mapped & {'transformer', 'unet', 'transformer_2'})


def memory_problem(model_type: str, model_path: str, *,
                   transformer: str | None = None,
                   text_encoders: collections.abc.Iterable[str] | None = None,
                   control_loras: collections.abc.Iterable[str] | None = None,
                   quantizer: str | None = None,
                   quantizer_map: collections.abc.Iterable[str] | None = None,
                   sequential: bool = False, cpu: bool = False, group: bool = False,
                   second_sequential: bool = False, second_cpu: bool = False,
                   cuda: bool, gib: float, bitsandbytes: bool, sdnq: bool) -> str | None:
    """
    What to tell the model when an invocation does not match this GPU's row.

    An offload satisfies a quantizer-only row, because it also fits. A quantizer
    does not satisfy an offload row. Text-encoder-only quantization does not
    satisfy a global quantizer.
    """
    spec = _memory_spec(memory_kind(model_type, model_path, transformer, text_encoders, control_loras),
                        cuda, gib, bitsandbytes, sdnq)
    if spec is None:
        return None
    if spec.forbid_quantizer and quantizer:
        flags = _spec_text(spec)
        extra = f' Write {flags}.' if flags and flags != 'no offload and no quantizer' else ''
        return (f'{model_path or model_type} must not be quantized on this GPU. '
                f'Do not pass --quantizer.{extra}')
    if spec.nothing and (sequential or cpu or group or second_sequential or second_cpu):
        return (f'{model_path or model_type} must not use offload on this GPU. '
                'Do not pass an offload flag.')
    offload_ok = _offload_ok(spec.offload, sequential, cpu, group)
    second_ok = _offload_ok(spec.second_offload, second_sequential, second_cpu, False)
    quantizer_ok = _quantizer_ok(spec, quantizer, quantizer_map)
    any_offload = sequential or cpu or group
    if spec.offload and spec.quantizer:
        fits = offload_ok and quantizer_ok and second_ok
    elif spec.offload:
        fits = offload_ok and second_ok
    elif spec.quantizer:
        fits = quantizer_ok or any_offload
    elif spec.second_offload:
        fits = second_ok
    else:
        fits = True
    if fits:
        return None
    return (f'{model_path or model_type} is missing the memory flags for this GPU ({gib:.1f} GiB). '
            f'Write {_spec_text(spec)} on that invocation.')


def offload_plan(cuda: bool, gib: float, bitsandbytes: bool, sdnq: bool) -> str:
    """
    Concrete memory flags for this computer, one row per model family.

    Each row is the branch the examples take at this memory size. A prequantized
    transformer keeps its offload and may quantize only the text encoder.
    """
    def row(kind: str) -> str:
        spec = _memory_spec(kind, cuda, gib, bitsandbytes, sdnq)
        return _spec_text(spec) if spec is not None else ''

    zimage_sdnq = row('zimage-sdnq')
    if 'quantizer' not in zimage_sdnq:
        zimage_sdnq += '. Do not pass --quantizer'
    lines = [
        '- krea2, wan, wan-animate, wan-animate-2, ltx2, hyvideo-15, motif-video, '
        'kandinsky5-image, kandinsky5-video, hyimage-21, hyimage-21-refiner, qwen-image-21, '
        f'if, ifs, ifs-img2img: {row("sequential")}. No quantizer, including a GGUF or FP8 file.',
        f'- ltx-video, full checkpoint: {row("ltx-full")}. No quantizer.',
        f'- ltx-video, prequantized transformer: {row("ltx-gguf")}. No quantizer.',
        f'- s-cascade: {row("cascade")}. No quantizer.',
        '- hydit: no quantizer and no offload.',
        '- sd, pix2pix, sdxl, sdxl-pix2pix, upscaler-x2, upscaler-x4: no quantizer. '
        '--model-cpu-offload only when the closest example uses it.',
        f'- kolors: {row("kolors")}.',
        f'- sd3, sd3-pix2pix, full checkpoint: {row("sd3-full")}.',
        f'- sd3, prequantized transformer: {row("sd3-pre")}.',
        f'- flux, flux-fill, flux-kontext, full checkpoint: {row("flux-full")}.',
        f'- flux, flux-fill, flux-kontext, control lora: {row("flux-control")}.',
        f'- flux, flux-fill, flux-kontext, GGUF transformer: {row("flux-gguf")}.',
        f'- flux, flux-fill, flux-kontext, FP8 or Comfy transformer: {row("flux-comfy")}. No quantizer.',
        f'- flux2 dev, flux2 klein 9b, flux2-klein-kv, full checkpoint: {row("flux2-heavy")}.',
        f'- flux2 klein 4b, full checkpoint: {row("klein4-full")}.',
        f'- flux2 dev, prequantized transformer: {row("flux2-pre-dev")}.',
        '- flux2 klein 4b, flux2 klein 9b, flux2-klein-kv, prequantized transformer: '
        f'{row("flux2-pre-klein")}.',
        '- flux2, transformer and text encoder both already quantized: --model-cpu-offload. No quantizer.',
        f'- qwen-image, qwen-image-edit, qwen-image-layered, full checkpoint: {row("qwen-full")}.',
        '- qwen-image, qwen-image-edit, qwen-image-layered, prequantized transformer: '
        f'{row("qwen-pre")}.',
        f'- z-image, full checkpoint: {row("zimage-full")}.',
        f'- z-image, prequantized transformer: {row("zimage-pre")}.',
        f'- z-image, SDNQ checkpoint: {zimage_sdnq}.',
    ]
    return '\n'.join(lines)


def machine_state() -> tuple[bool, float, bool, bool]:
    """CUDA, accelerator GiB, bitsandbytes, and sdnq for this computer."""
    import dgenerate.batchprocess.configrunnerbuiltins as _builtins

    cuda = bool(_builtins.have_cuda())
    try:
        gib = float(_builtins.total_memory(unit='gib'))
    except Exception:
        gib = 0.0
    return cuda, gib, bool(_builtins.have_feature('bitsandbytes')), bool(_builtins.have_feature('sdnq'))


def machine_report() -> str:
    """
    Accelerator memory and installed quantizers, plus one offload line per family.
    """
    import dgenerate.batchprocess.configrunnerbuiltins as _builtins

    device = _builtins.default_device()
    cuda, gib, bitsandbytes, sdnq = machine_state()
    quantizers = [name for name, present in (('bitsandbytes', bitsandbytes), ('sdnq', sdnq)) if present]
    plan = offload_plan(cuda, gib, bitsandbytes, sdnq)
    return (
        '### machine\n'
        f'Default device: {device}. CUDA: {cuda}. XPU: {_builtins.have_xpu()}. '
        f'MPS: {_builtins.have_mps()}.\n'
        f'Accelerator memory: {gib:.1f} GiB.\n'
        f'Quantizers installed: {", ".join(quantizers) if quantizers else "none"}.\n'
        'Copy that row\'s flags onto every invocation of that family. '
        'Do not write a total_memory check. '
        'An invocation that leaves those flags off does not fit this GPU.\n'
        f'{plan}'
    )


def system_message(version: str, link_tools: bool = False, machine: str = '',
                   check_tool: bool = False) -> str:
    text = SYSTEM_PROMPT.format(shebang=SHEBANG, version=version)
    if check_tool:
        text += '\n' + CHECK_TOOL_SYSTEM
    if link_tools:
        text += '\n' + LINK_TOOLS_SYSTEM
    if machine:
        text += '\n\n' + machine
    return text


def extract_config(reply: str, version: str) -> str:
    """
    Pull the config out of a model reply and make sure it starts with the shebang lines.
    """
    lines = reply.strip('\n').splitlines()
    # Everything from the first fence to the last, because replies sometimes nest one fence inside another.
    fences = [i for i, line in enumerate(lines) if _FENCE.match(line)]
    if fences:
        lines = lines[fences[0] + 1:fences[-1] if len(fences) > 1 else None]
        lines = [line for line in lines if not _FENCE.match(line)]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and lines[0].startswith('#!'):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return lift_comments_between_options(
        '\n'.join([SHEBANG, f'#! dgenerate {version}', ''] + lines).rstrip() + '\n')


def lift_comments_between_options(config: str) -> str:
    """
    Move a ``#`` comment that sits inside an invocation to just above its model path.

    A comment between options ends the invocation. Small models keep writing them
    there, and asking those models to move the line does not stick.
    """
    output: list[str] = []
    in_invocation = False
    model_index: int | None = None
    gap: list[str] = []

    def comments_were_between_options() -> None:
        nonlocal gap, model_index
        if gap and model_index is not None:
            for comment in reversed(gap):
                output.insert(model_index, comment)
            model_index += len(gap)
        gap = []

    def comments_stay() -> None:
        nonlocal gap
        output.extend(gap)
        gap = []

    def end_invocation() -> None:
        nonlocal in_invocation, model_index
        comments_stay()
        in_invocation = False
        model_index = None

    for raw in config.splitlines():
        stripped = raw.strip()
        if not stripped:
            end_invocation()
            output.append(raw)
            continue
        if stripped.startswith('#'):
            if in_invocation:
                gap.append(raw)
            else:
                output.append(raw)
            continue
        if stripped.startswith(('\\', '{%', '{{', '{#')):
            end_invocation()
            output.append(raw)
            continue
        if stripped.startswith('-'):
            comments_were_between_options()
            output.append(raw)
            continue
        end_invocation()
        model_index = len(output)
        output.append(raw)
        in_invocation = True
    end_invocation()
    text = '\n'.join(output)
    if config.endswith('\n'):
        text += '\n'
    return text


def unknown_options(config: str, known: list[str]) -> list[tuple[str, list[str]]]:
    """
    Option lines whose option is not a dgenerate option, with close matches.
    """
    known_set = set(known)
    result = []
    in_directive = False
    for line in config.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(('#', '{%', '{#')):
            continue
        match = _OPTION_LINE.match(line)
        if match is None:
            # Options continuing a \directive line belong to the directive, which the full check validates.
            in_directive = stripped.startswith('\\')
            continue
        option = match.group(1)
        if in_directive:
            continue
        if option not in known_set and option not in (o for o, _ in result):
            result.append((option, difflib.get_close_matches(option, known, n=3, cutoff=0.6)))
    return result


def repair_message(problems: list[str], suggested: list[str], index: _index.Index) -> str:
    references = []
    for option in suggested:
        chunk = index.argument(option)
        if chunk is not None:
            references.append(_section(chunk, MAX_ARGUMENT_CHARS))
    reference_text = ('Help for options that may be what you meant:\n\n' + '\n\n'.join(references) + '\n\n') \
        if references else ''
    return REPAIR_PROMPT.format(problems='\n'.join(f'- {p}' for p in problems), references=reference_text)
