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

import base64
import io
import os
import typing

import PIL.Image

import dgenerate.mediainput as _mediainput
import dgenerate.prompt as _prompt
import dgenerate.webcache as _webcache


def process_prompts_batched(
        prompts: _prompt.Prompts,
        part: str,
        generator: typing.Callable[[list[str]], list[str]]
) -> list[_prompt.Prompt]:
    """
    Process a list of prompts using a text generation function which can accept a batch of strings.

    This handles processing of the positive and negative prompt selectively, and
    reconstruction of the prompt objects after processing.

    :param prompts: Input prompts.
    :param part: Prompt parts to process, "both", "positive", "negative".
    :param generator: The text processing function, should accept a list of strings, and return a list of strings
    :return: List of generated prompt objects.
    """

    generated_pos_prompts = [p.positive for p in prompts]
    generated_neg_prompts = [p.negative for p in prompts]

    if part in {'both', 'positive'}:
        non_empty_idx = [idx for idx, p in enumerate(prompts) if p.positive]

        pos_prompts = [p.positive for p in prompts if p.positive]

        if pos_prompts:
            generated = generator(pos_prompts)

            for idx, non_empty_idx in enumerate(non_empty_idx):
                generated_pos_prompts[non_empty_idx] = generated[idx]

    if part in {'both', 'negative'}:
        non_empty_idx = [idx for idx, p in enumerate(prompts) if p.negative]

        neg_prompts = [p.negative for p in prompts if p.negative]

        if neg_prompts:
            generated = generator(neg_prompts)

            for idx, non_empty_idx in enumerate(non_empty_idx):
                generated_neg_prompts[non_empty_idx] = generated[idx]

    output = []
    for idx, (generated_pos_prompt, generated_neg_prompt) in \
            enumerate(zip(generated_pos_prompts, generated_neg_prompts)):
        prompt_obj = _prompt.Prompt(
            positive=generated_pos_prompt,
            negative=generated_neg_prompt,
            delimiter=prompts[idx].delimiter
        )

        # We need to preserve the embedded diffusion
        # arguments from the original incoming prompt
        # that were parsed out by dgenerate
        prompt_obj.copy_embedded_args_from(prompts[idx])

        # append the generated prompt to the expanded
        # output list of prompts
        output.append(prompt_obj)

    return output


def image_file_argument() -> dict:
    """
    ``FILE_ARGS`` descriptor for a prompt-upscaler image argument.
    """
    extensions = ['*.' + ext for ext in _mediainput.get_supported_image_formats()]
    return {'mode': 'in', 'filetypes': [('Images', extensions)]}


def _image_mimetype_ok(mime: str) -> bool:
    mime = mime.split(';', 1)[0].strip().lower()
    return mime.startswith('image/') or mime == 'application/octet-stream'


def resolve_input_image(image: str, *, local_files_only: bool) -> str:
    """
    Return a local path for an image path or URL.

    :param image: File path or downloadable URL.
    :param local_files_only: Only use a cached download.
    :raises ValueError: The image could not be found or downloaded.
    """
    if _webcache.is_downloadable_url(image):
        try:
            _, path = _webcache.create_web_cache_file(
                image,
                mime_acceptable_desc='image files',
                mimetype_is_supported=_image_mimetype_ok,
                local_files_only=local_files_only,
            )
        except Exception as e:
            raise ValueError(
                f'Could not download argument "image": "{image}", error: {e}') from e
        return path

    if not os.path.exists(image):
        raise ValueError(
            f'Argument "image" must be a file on disk or a URL. No such file: "{image}".')
    return os.path.abspath(image)


def load_rgb_image(path: str) -> PIL.Image.Image:
    """
    Open an image and return a detached RGB copy. Animated files use the first frame.
    """
    try:
        with PIL.Image.open(path) as img:
            rgb = img.convert('RGB')
            rgb.load()
            return rgb
    except Exception as e:
        raise ValueError(
            f'Argument "image" is not a readable image file: "{path}", error: {e}') from e


def png_data_url(image: PIL.Image.Image) -> str:
    """
    Encode an RGB image as a PNG data URL for an OpenAI-style image block.
    """
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return 'data:image/png;base64,' + encoded
