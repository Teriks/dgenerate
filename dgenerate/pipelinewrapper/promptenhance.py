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

"""Rewrite a prompt with the text encoder already loaded for the pipeline."""

from __future__ import annotations

import collections.abc
import contextlib
import itertools
import os
import re

import langcodes
import torch

import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.util as _putil

# Built-in rewrite rules. The length and the image wording are filled in
# for the encoder that will run them.
# An English-leaning BPE holds about three words in four tokens.
# Qwen and ChatGLM hold about one English word in two tokens, and a
# Chinese, Japanese, or Korean character is often one token.
_WORDS_PER_TOKEN = 0.75
_MULTILINGUAL_WORDS_PER_TOKEN = 0.5
_MULTILINGUAL_CHARACTERS_PER_TOKEN = 1.0

MAX_NEW_TOKENS = 512
MIN_NEW_TOKENS = 64


def _multilingual_tokenizer(model_type, encoder_name: str | None) -> bool:
    """True when the tokenizer is built for Chinese as well as English."""
    if encoder_name:
        if encoder_name.startswith(('Qwen', 'ChatGLM', 'GLM')):
            return True
        if encoder_name.startswith('Mistral'):
            return True
        return False
    if model_type is None:
        return False
    return bool(
        _enums.model_type_is_z_image_family(model_type)
        or _enums.model_type_is_krea2(model_type)
        or _enums.model_type_is_qwen_image(model_type)
        or _enums.model_type_is_qwen_image_edit(model_type)
        or _enums.model_type_is_qwen_image_layered(model_type)
        or _enums.model_type_is_qwen_image_21(model_type)
        or model_type == _enums.ModelType.FLUX2_KLEIN_KV)


def _words_for_tokens(tokens: int, *, model_type=None, encoder_name: str | None = None) -> int:
    """English words that fit in ``tokens`` on this tokenizer."""
    if tokens <= 0:
        return 0
    rate = (_MULTILINGUAL_WORDS_PER_TOKEN
            if _multilingual_tokenizer(model_type, encoder_name)
            else _WORDS_PER_TOKEN)
    return max(1, int(round(tokens * rate)))


def _characters_for_tokens(tokens: int, *, model_type=None,
                           encoder_name: str | None = None) -> int | None:
    """CJK characters that fit in ``tokens``, or None for an English-only budget."""
    if model_type is not None and _enhance_english_only(model_type):
        return None
    if not _multilingual_tokenizer(model_type, encoder_name):
        return None
    if tokens <= 0:
        return 0
    return max(1, int(round(tokens * _MULTILINGUAL_CHARACTERS_PER_TOKEN)))


def _clamp_span(floor: int, ceiling: int, preferred_lo: int, preferred_hi: int) -> tuple[int, int]:
    """Keep a preferred word band inside the sampler's floor and ceiling."""
    ceiling = max(floor, ceiling)
    lo = min(max(preferred_lo, floor), ceiling)
    hi = min(max(preferred_hi, lo), ceiling)
    return lo, hi


def _about_count(lo: int, hi: int, unit: str) -> str:
    if hi <= 0:
        return f'a short prompt'
    if lo <= 0:
        return f'at most {hi} {unit}'
    if lo == hi:
        return f'about {lo} {unit}'
    return f'about {lo} to {hi} {unit}'


def _measure(min_tokens: int, max_tokens: int, preferred_lo: int, preferred_hi: int,
             preferred_several: int, *, model_type=None, encoder_name: str | None = None):
    """Word band, and a CJK character band when this tokenizer needs one."""
    floor = _words_for_tokens(min_tokens, model_type=model_type, encoder_name=encoder_name)
    ceiling = max(floor, _words_for_tokens(max_tokens, model_type=model_type, encoder_name=encoder_name))
    lo, hi = _clamp_span(floor, ceiling, preferred_lo, preferred_hi)
    several = min(ceiling, max(hi, preferred_several))
    words = _about_count(lo, hi, 'words')
    characters = _characters_for_tokens(max_tokens, model_type=model_type, encoder_name=encoder_name)
    if characters is None:
        return words, several, None, None
    char_floor = _characters_for_tokens(min_tokens, model_type=model_type, encoder_name=encoder_name) or 0
    char_ceiling = max(char_floor, characters)
    char_lo, char_hi = _clamp_span(char_floor, char_ceiling, preferred_lo, preferred_hi)
    char_several = min(char_ceiling, max(char_hi, preferred_several))
    return words, several, _about_count(char_lo, char_hi, 'characters'), char_several


def _with_script(words: str, characters: str | None) -> str:
    if characters is None:
        return words
    return f'{words}, or {characters} in Chinese, Japanese, or Korean'


def _keep_user_language(model_type) -> bool:
    """``--model-type`` values whose built-in enhance prompt keeps the user's language."""
    if model_type is None:
        return False
    return bool(
        _enums.model_type_is_qwen_image(model_type)
        or _enums.model_type_is_qwen_image_edit(model_type)
        or _enums.model_type_is_qwen_image_layered(model_type)
        or _enums.model_type_is_qwen_image_21(model_type)
        or _enums.model_type_is_flux2(model_type)
        or _enums.model_type_is_flux2_klein_kv(model_type))


def _enhance_english_only(model_type) -> bool:
    """Z-Image and Krea 2 get English captions even for non-English prompts."""
    if model_type is None:
        return False
    return bool(
        _enums.model_type_is_z_image_family(model_type)
        or _enums.model_type_is_krea2(model_type))


def _language_lead(model_type) -> str:
    if _keep_user_language(model_type):
        return 'Write in the language the user writes in'
    return 'Write in English whatever language the user writes in'


_KEEP_USER_LANGUAGE_RULE = (
    '- Match the language of the user\'s request. Do not translate the rewrite into another language. '
    'Sign or label wording in another script stays exactly as the user wrote it.')


_SIGN_LITERAL = re.compile(
    r'[「『][^」』]*[」』]|«[^»]*»|"[^"]*"')

_QWEN25_LANG_DETECTOR = None


def _qwen25_lang_detector():
    global _QWEN25_LANG_DETECTOR
    if _QWEN25_LANG_DETECTOR is None:
        from fast_langdetect import LangDetectConfig, LangDetector
        _QWEN25_LANG_DETECTOR = LangDetector(
            LangDetectConfig(model='auto', max_input_length=None))
    return _QWEN25_LANG_DETECTOR


def _detect_with_fast_langdetect(body: str) -> str | None:
    try:
        hits = _qwen25_lang_detector().detect(body, k=1)
    except Exception as error:
        _messages.debug_log(f'Pipeline prompt enhance language detection failed: {error}')
        return None
    if not hits:
        return None
    lang = hits[0].get('lang')
    return lang if isinstance(lang, str) and lang else None


def _detect_prompt_language(text: str) -> str | None:
    """Top ISO 639-1 code for the user prompt, ignoring literal sign or label text."""
    body = _prompt_body_without_sign_literals(text).strip()
    if not body:
        return None
    return _detect_with_fast_langdetect(body)


def _prompt_body_without_sign_literals(text: str) -> str:
    return _SIGN_LITERAL.sub(' ', text)


def _is_qwen25_vl_encoder(encoder_name: str | None) -> bool:
    return bool(encoder_name and encoder_name.startswith('Qwen2_5_VL'))


def _normalize_detected_language(code: str | None) -> str | None:
    if not code:
        return None
    try:
        return langcodes.get(code).language
    except langcodes.tag_parser.LanguageTagError:
        return code.split('-', 1)[0] if code else None


_QWEN25_USER_TURN_PREFIX_SOURCE = (
    'Write your entire answer in the same language as the user request below. '
    'Do not translate your answer into any other language.\n\n')

_QWEN25_ENGLISH_USER_TURN_PREFIX = (
    'The request below is in English. Write your entire answer in English only. '
    'Do not write Chinese, Japanese, or Korean.\n\n')

_QWEN25_TRANSLATED_PREFIX_CACHE: dict[tuple[str, bool], str] = {}


def _qwen25_translated_user_prefix(lang: str, local_files_only: bool) -> str:
    if lang == 'en':
        return _QWEN25_ENGLISH_USER_TURN_PREFIX
    key = (lang, local_files_only)
    cached = _QWEN25_TRANSLATED_PREFIX_CACHE.get(key)
    if cached is not None:
        return cached
    import dgenerate.translators as _translators
    try:
        translator = _translators.ArgosTranslator('en', lang, local_files_only=local_files_only)
        translated = translator.translate(_QWEN25_USER_TURN_PREFIX_SOURCE)[0].strip()
        if not translated.endswith('\n'):
            translated += '\n'
        translated += '\n'
        _QWEN25_TRANSLATED_PREFIX_CACHE[key] = translated
        return translated
    except (_translators.TranslatorLoadError, _translators.TranslationError) as error:
        _messages.debug_log(
            f'Pipeline prompt enhance: could not translate language prefix to {lang}: {error}')
        return _QWEN25_USER_TURN_PREFIX_SOURCE


def _qwen25_user_language_prefix(prompt: str, *, local_files_only: bool = False) -> str:
    lang = _normalize_detected_language(_detect_prompt_language(prompt))
    if not lang:
        return _QWEN25_USER_TURN_PREFIX_SOURCE
    return _qwen25_translated_user_prefix(lang, local_files_only)


def _qwen25_prefix_already_present(prompt: str) -> bool:
    if prompt.startswith((
            _QWEN25_ENGLISH_USER_TURN_PREFIX,
            _QWEN25_USER_TURN_PREFIX_SOURCE)):
        return True
    for cached in _QWEN25_TRANSLATED_PREFIX_CACHE.values():
        if prompt.startswith(cached):
            return True
    return False


def _encoder_gets_user_language_prefix(encoder_name: str | None) -> bool:
    """Encoders that follow a multilingual user-turn reminder (Qwen-Image Qwen2.5-VL)."""
    if not encoder_name:
        return False
    return _is_qwen25_vl_encoder(encoder_name)


def _encoder_gets_english_user_prefix(encoder_name: str | None, model_type) -> bool:
    if not _enhance_english_only(model_type):
        return False
    return encoder_name in {'Qwen3Model', 'Qwen3VLModel'}


def _rewrite_user_prompt(prompt: str, *, model_type, encoder_name: str | None,
                         local_files_only: bool = False) -> str:
    """Prepend a language reminder on the user turn when it helps the encoder."""
    if _keep_user_language(model_type) and _encoder_gets_user_language_prefix(encoder_name):
        if _qwen25_prefix_already_present(prompt):
            return prompt
        prefix = _qwen25_user_language_prefix(prompt, local_files_only=local_files_only)
        return prefix + prompt
    if _encoder_gets_english_user_prefix(encoder_name, model_type):
        if prompt.startswith(_QWEN25_ENGLISH_USER_TURN_PREFIX):
            return prompt
        return _QWEN25_ENGLISH_USER_TURN_PREFIX + prompt
    return prompt


_IMAGE_TEXT_RULE = (
    '- If the user asks for text in the image (a sign, title or label), copy that wording exactly '
    'as the user wrote it, including quotation marks, guillemets, or corner brackets such as '
    '「OPEN」 or «ABIERTO», and say where it is and how it looks. Never translate sign or label text '
    'and never say which language it is written in. '
    'Otherwise write nothing about text.')


def _t2i_length_line(min_tokens: int, max_tokens: int, *,
                     model_type=None, encoder_name: str | None = None) -> str:
    words, several, characters, char_several = _measure(
        min_tokens, max_tokens, 80, 150, 250,
        model_type=model_type, encoder_name=encoder_name)
    line = (
        f'- {_language_lead(model_type)}, '
        f'{_with_script(words, characters)}, in the present tense, as a description of the image, '
        'not as instructions.')
    extras = []
    word_hi = _span_hi(words)
    if several > word_hi:
        extras.append(f'about {several} words')
    if characters is not None and char_several > _span_hi(characters):
        extras.append(f'about {char_several} characters')
    if extras:
        line += (
            ' A scene with several people or many objects may need up to '
            + ' or '.join(extras)
            + ': give each one its own sentence (how it looks, where it is, what it does).')
    return line


def _span_hi(phrase: str) -> int:
    numbers = [int(piece) for piece in phrase.split() if piece.isdigit()]
    return numbers[-1] if numbers else 0


def _edit_length_line(min_tokens: int, max_tokens: int, *,
                      model_type=None, encoder_name: str | None = None) -> str:
    words, several, characters, char_several = _measure(
        min_tokens, max_tokens, 12, 35, 60,
        model_type=model_type, encoder_name=encoder_name)
    one = _with_script(words, characters)
    lead = _language_lead(model_type)
    if _span_hi(words) == several and (characters is None or _span_hi(characters) == char_several):
        return (
            f'- {lead}, as one paragraph '
            f'with no resolution or aspect ratio: {one}.')
    longer = f'up to about {several} words'
    if characters is not None and char_several > _span_hi(characters):
        longer += f' or {char_several} characters'
    return (
        f'- {lead}, as one paragraph with no '
        f'resolution or aspect ratio: {one} for one change; when the user '
        f'asks for several changes, one sentence for each, {longer}.')


def _brief_encoder(model_type, encoder_name: str | None) -> bool:
    """Tied Qwen encoders follow a short instruction more reliably."""
    if encoder_name in {'Qwen3Model', 'Qwen3VLModel'}:
        return True
    if _is_qwen25_vl_encoder(encoder_name):
        return True
    if encoder_name:
        return False
    if model_type is None:
        return False
    return bool(
        _enums.model_type_is_z_image_family(model_type)
        or _enums.model_type_is_krea2(model_type))


def _encoder_sees_images(model_type, encoder_name: str | None) -> bool:
    """True when this text encoder can take pixels, not only a SigLIP or text tower."""
    if encoder_name:
        if encoder_name.startswith(('Qwen2', 'Qwen3VLFor', 'Mistral')):
            return True
        return False
    if model_type is None:
        return False
    return bool(
        _enums.model_type_is_qwen_image_edit(model_type)
        or _enums.model_type_is_qwen_image_layered(model_type)
        or _enums.model_type_is_qwen_image_21(model_type)
        or model_type == _enums.ModelType.FLUX2)


def _uses_image_placeholders(model_type) -> bool:
    """Qwen edit models expect ``<image1>`` labels. Flux does not."""
    if model_type is None:
        return False
    return bool(
        _enums.model_type_is_qwen_image_edit(model_type)
        or _enums.model_type_is_qwen_image_layered(model_type)
        or _enums.model_type_is_qwen_image_21(model_type))


def _wants_edit_prompt(model_type, encoder_name: str | None, has_image: bool) -> bool:
    if not has_image or model_type is None:
        return False
    family = (
        _enums.model_type_is_qwen_image_edit(model_type)
        or _enums.model_type_is_qwen_image_layered(model_type)
        or _enums.model_type_is_qwen_image_21(model_type)
        or _enums.model_type_is_flux2_family(model_type))
    return bool(family and _encoder_sees_images(model_type, encoder_name))


def t2i_system_prompt(min_tokens: int, max_tokens: int, *, brief: bool = False,
                      model_type=None, encoder_name: str | None = None) -> str:
    """Text-to-image prompt, with the length set from this tokenizer's budget."""
    length = _t2i_length_line(
        min_tokens, max_tokens, model_type=model_type, encoder_name=encoder_name)
    keep = _keep_user_language(model_type)
    if brief:
        opening = (
            'You write prompts for an image generation model. Expand the user\'s request into one paragraph '
            + ('in the user\'s language ' if keep else 'in English ')
            + 'that describes the finished image.')
        lines = [
            opening,
            '- Keep everything the user specified and never contradict it. If the request is already detailed, keep it and add only what is missing. Leave out size and quality words.',
            '- Counts are exact. "three X and one Y" means three X and one Y.',
            '- Name the medium near the start. If the user names none, use a photograph for a real-world scene.',
            '- Describe the light, the colours, and where things are. No quality tags such as masterpiece or highly detailed.',
            _IMAGE_TEXT_RULE,
            length,
            'Output only the prompt.',
        ]
        if keep:
            lines.insert(1, _KEEP_USER_LANGUAGE_RULE)
        return '\n'.join(lines)
    if keep:
        opening = (
            'You write prompts for an image generation model. Expand the user\'s request into one detailed paragraph '
            'in the user\'s language that describes the finished image: subject, setting, composition, lighting and style.')
        meaning = (
            '- Keep the user\'s language. A "red" cat in Italian, French, or Spanish means a ginger cat; '
            'describe that coat in the user\'s language.')
    else:
        opening = (
            'You write prompts for an image generation model. Expand the user\'s request into one detailed English '
            'paragraph that describes the finished image: subject, setting, composition, lighting and style.')
        meaning = '- Translate the meaning, not word by word. A "red" cat in Italian, French or Spanish is a ginger cat.'
    lines = [
        opening,
        '- Keep everything the user specified about the image (objects, counts, colours, positions, style) and never contradict it. If the request is already detailed, keep all of it and only add what is missing. Leave out size and quality words (4K, HD, 16:9, resolution): the app sets the size.',
        '- Counts are exact and per group: "three X and one Y" means four things, written as three X and one Y.',
        meaning,
        '- Name the medium and style near the start. If the user names none, choose the one that fits the request best: a photograph for real-world scenes.',
        '- Always describe the light: its source, direction and quality.',
        '- Be concrete: specific colours that suit this scene, materials, where things are in the frame. Never write vague words like "various items".',
        '- People: only what is visible (life stage, expression, pose, hair, clothing). No brand names unless the user gave them.',
        _IMAGE_TEXT_RULE,
        '- Describe only what is seen, with no quality tags (masterpiece, highly detailed). When the user says what the image is for, keep what that means for the look and write it as visible choices (how dark, how much empty space, how bold), not as the name of the device or use.',
        length,
        'Output only the prompt.',
    ]
    if keep:
        lines.insert(1, _KEEP_USER_LANGUAGE_RULE)
    return '\n'.join(lines)


def edit_system_prompt(min_tokens: int, max_tokens: int, *, placeholders: bool,
                       model_type=None, encoder_name: str | None = None) -> str:
    """Edit prompt. Qwen edit models get ``<imageN>`` labels. Flux does not."""
    if placeholders:
        images = (
            '- With several input images, refer to them as <image1>, <image2>, ...: '
            'start with Place or Combine, say what is taken from each image and where it goes in the result.')
    else:
        images = (
            '- With several input images, name them in order as the first image and the second image: '
            'start with Place or Combine, say what is taken from each and where it goes in the result.')
    keep = _keep_user_language(model_type)
    if keep:
        opening = (
            'You write prompts for an image editing model. You see the input image and the user\'s instruction. '
            'Rewrite the instruction as one precise editing instruction in the user\'s language.')
    else:
        opening = (
            'You write prompts for an image editing model. You see the input image and the user\'s instruction. '
            'Rewrite the instruction as one precise English editing instruction.')
    lines = [
        opening,
        '- Start with the operation (Change, Replace, Remove, Add, Turn, Place, ...) and say exactly what changes, where it is in the image and what it becomes. Make the change complete and clearly visible, not faint.',
        '- Describe in detail only what changes. For everything else end with one sentence such as "Keep everything else in the image unchanged." Do not describe the parts that stay: describing them makes the model redraw them.',
        '- People keep their face and identity, and the image keeps its medium (photo, painting, illustration), unless the user asks to change them.',
        '- Base every detail on what is visible in the image and do not invent objects. Do only what the user asked: no extra fixes or additions.',
        '- When something is removed or moved, its shadow and reflection go with it; say briefly what fills the empty area so the result looks natural.',
        '- Text in the image: write the exact new text as the user gave it, including any brackets or guillemets, and keep the existing lettering style and position unless the user asks otherwise.',
        '- For a style or mood change (painting, season, night, ...), apply it fully to the whole image and end with "Keep the composition and the subjects unchanged." instead.',
        images,
        _edit_length_line(
            min_tokens, max_tokens, model_type=model_type, encoder_name=encoder_name),
        'Output only the instruction.',
    ]
    if keep:
        lines.insert(1, _KEEP_USER_LANGUAGE_RULE)
    return '\n'.join(lines)


T2I_SYSTEM_PROMPT = t2i_system_prompt(MIN_NEW_TOKENS, MAX_NEW_TOKENS)
EDIT_SYSTEM_PROMPT = edit_system_prompt(MIN_NEW_TOKENS, MAX_NEW_TOKENS, placeholders=True)


class PromptEnhanceOverrides:
    """Optional ``--pipeline-prompt-enhance-*`` generation settings.

    Built from CLI or :py:class:`dgenerate.renderloopconfig.RenderLoopConfig`
    fields ``pipeline_prompt_enhance_temperature``, ``pipeline_prompt_enhance_top_p``,
    ``pipeline_prompt_enhance_top_k``, ``pipeline_prompt_enhance_presence_penalty``,
    ``pipeline_prompt_enhance_max_new_tokens``, ``pipeline_prompt_enhance_min_new_tokens``,
    and ``pipeline_prompt_enhance_gpu_headroom``. Unset attributes keep the
    per-model defaults from :func:`sampling_for` and :func:`generation_limits`.
    Setting any of these (or ``--pipeline-prompt-enhance-seeds``) turns
    ``--pipeline-prompt-enhance`` on with the built-in system prompt when that
    flag is omitted; see :func:`ensure_pipeline_prompt_enhance_enabled`.
    """

    __slots__ = (
        'temperature', 'top_p', 'top_k', 'presence_penalty',
        'max_new_tokens', 'min_new_tokens', 'gpu_headroom')

    def __init__(self,
                 temperature: float | None = None,
                 top_p: float | None = None,
                 top_k: int | None = None,
                 presence_penalty: float | None = None,
                 max_new_tokens: int | None = None,
                 min_new_tokens: int | None = None,
                 gpu_headroom: float | None = None):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.gpu_headroom = gpu_headroom

    @classmethod
    def from_args(cls, args) -> 'PromptEnhanceOverrides':
        return cls(
            temperature=getattr(args, 'pipeline_prompt_enhance_temperature', None),
            top_p=getattr(args, 'pipeline_prompt_enhance_top_p', None),
            top_k=getattr(args, 'pipeline_prompt_enhance_top_k', None),
            presence_penalty=getattr(args, 'pipeline_prompt_enhance_presence_penalty', None),
            max_new_tokens=getattr(args, 'pipeline_prompt_enhance_max_new_tokens', None),
            min_new_tokens=getattr(args, 'pipeline_prompt_enhance_min_new_tokens', None),
            gpu_headroom=getattr(args, 'pipeline_prompt_enhance_gpu_headroom', None),
        )

    def any_set(self) -> bool:
        return any(getattr(self, name) is not None for name in self.__slots__)


def pipeline_prompt_enhance_requested(args) -> bool:
    """True when any ``--pipeline-prompt-enhance*`` CLI field is set on *args*."""
    if getattr(args, 'pipeline_prompt_enhance', None) is not None:
        return True
    if getattr(args, 'pipeline_prompt_enhance_seeds', None):
        return True
    return PromptEnhanceOverrides.from_args(args).any_set()


def ensure_pipeline_prompt_enhance_enabled(config) -> None:
    """Set ``pipeline_prompt_enhance`` to ``True`` when only sub-options were given.

    Applies when :func:`pipeline_prompt_enhance_requested` is true and
    ``pipeline_prompt_enhance`` is still ``None``.
    """
    if config.pipeline_prompt_enhance is None and pipeline_prompt_enhance_requested(config):
        config.pipeline_prompt_enhance = True


def generation_limits(model_type=None, encoder_name: str | None = None) -> tuple[int, int]:
    """(max_new_tokens, min_new_tokens) for one rewrite."""
    if model_type is not None and _enums.get_model_type_enum(model_type) == _enums.ModelType.LTX2:
        return 600, 0
    if _is_qwen25_vl_encoder(encoder_name):
        return MAX_NEW_TOKENS, 32
    return MAX_NEW_TOKENS, MIN_NEW_TOKENS


def merged_sampling(model_type, encoder_name: str | None,
                    overrides: PromptEnhanceOverrides | None) -> Sampling:
    base = _QWEN if model_type is None else sampling_for(model_type, encoder_name)
    if overrides is None:
        return base
    return Sampling(
        temperature=base.temperature if overrides.temperature is None else overrides.temperature,
        top_p=base.top_p if overrides.top_p is None else overrides.top_p,
        top_k=base.top_k if overrides.top_k is None else overrides.top_k,
        presence_penalty=(
            base.presence_penalty if overrides.presence_penalty is None
            else overrides.presence_penalty),
    )


def merged_generation_limits(model_type, encoder_name: str | None,
                             overrides: PromptEnhanceOverrides | None) -> tuple[int, int]:
    max_new, min_new = generation_limits(model_type, encoder_name)
    if overrides is None:
        return max_new, min_new
    if overrides.max_new_tokens is not None:
        max_new = overrides.max_new_tokens
    if overrides.min_new_tokens is not None:
        min_new = overrides.min_new_tokens
    return max_new, min_new


def _override_check_limits(model_type, encoder_name: str | None) -> tuple[int, int]:
    """Token budget used to reject an impossible ``--pipeline-prompt-enhance`` range.

    Once the text encoder is known, its own floor applies. Before that, Qwen2.5-VL's
    floor of 32 is the lowest non-LTX floor, so a budget that encoder can run is not
    rejected while the pipeline is still unloaded. LTX is selected by model type.
    """
    if encoder_name is not None:
        return generation_limits(model_type, encoder_name)
    if (model_type is not None
            and _enums.get_model_type_enum(model_type) == _enums.ModelType.LTX2):
        return generation_limits(model_type, None)
    return MAX_NEW_TOKENS, 32


def check_generation_overrides(overrides: PromptEnhanceOverrides | None, a_namer,
                               model_type=None, encoder_name: str | None = None) -> None:
    if overrides is None or not overrides.any_set():
        return
    if overrides.max_new_tokens is not None and overrides.max_new_tokens < 1:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{a_namer("pipeline_prompt_enhance_max_new_tokens")} must be at least 1.')
    if overrides.min_new_tokens is not None and overrides.min_new_tokens < 0:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{a_namer("pipeline_prompt_enhance_min_new_tokens")} must be at least 0.')
    eff_max, eff_min = _override_check_limits(model_type, encoder_name)
    if overrides.max_new_tokens is not None:
        eff_max = overrides.max_new_tokens
    if overrides.min_new_tokens is not None:
        eff_min = overrides.min_new_tokens
    if eff_min > eff_max:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{a_namer("pipeline_prompt_enhance_min_new_tokens")} must not exceed '
            f'{a_namer("pipeline_prompt_enhance_max_new_tokens")}.')
    if overrides.top_p is not None and not (0 < overrides.top_p <= 1):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{a_namer("pipeline_prompt_enhance_top_p")} must be greater than 0 and at most 1.')
    if overrides.top_k is not None and overrides.top_k < 0:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{a_namer("pipeline_prompt_enhance_top_k")} must be at least 0.')
    if overrides.temperature is not None and overrides.temperature < 0:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{a_namer("pipeline_prompt_enhance_temperature")} must be at least 0.')
    if overrides.gpu_headroom is not None and overrides.gpu_headroom < 0:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{a_namer("pipeline_prompt_enhance_gpu_headroom")} must be at least 0.')


def ltx_enhance_prompt_kwargs(pipe, overrides: PromptEnhanceOverrides | None) -> dict:
    """Extra keyword arguments for ``LTX2Pipeline.enhance_prompt``."""
    if overrides is None or not overrides.any_set():
        return {}
    from diffusers.pipelines.ltx2.utils import (
        GEMMA3_PROMPT_ENHANCEMENT_CONFIG,
        GEMMA4_PROMPT_ENHANCEMENT_CONFIG,
    )
    dedicated = getattr(pipe, 'prompt_enhancer', None) is not None
    config = GEMMA4_PROMPT_ENHANCEMENT_CONFIG if dedicated else GEMMA3_PROMPT_ENHANCEMENT_CONFIG
    max_new, min_new = merged_generation_limits(_enums.ModelType.LTX2, None, overrides)
    kwargs = {}
    if overrides.max_new_tokens is not None:
        kwargs['max_new_tokens'] = max_new
    generation = dict(config.generation_kwargs)
    touched = False
    if overrides.temperature is not None:
        generation['temperature'] = overrides.temperature
        generation['do_sample'] = overrides.temperature > 0
        touched = True
    if overrides.top_p is not None:
        generation['top_p'] = overrides.top_p
        generation['do_sample'] = True
        touched = True
    if overrides.top_k is not None:
        touched = True
        if overrides.top_k > 0:
            generation['top_k'] = overrides.top_k
        else:
            generation.pop('top_k', None)
    if overrides.presence_penalty is not None:
        generation['presence_penalty'] = overrides.presence_penalty
        touched = True
    if overrides.min_new_tokens is not None and overrides.min_new_tokens > 0:
        generation['min_new_tokens'] = overrides.min_new_tokens
        touched = True
    elif overrides.min_new_tokens == 0:
        generation.pop('min_new_tokens', None)
        touched = True
    if touched:
        kwargs['generation_kwargs'] = generation
    return kwargs


def attach_ltx_token_counter(pipe, enhance_kwargs: dict) -> '_TokenCounter':
    """Add a one-line token counter to Diffusers ``enhance_prompt`` keyword arguments.

    ``enhance_prompt`` replaces its default generation settings when
    ``generation_kwargs`` is supplied, so the streamer is merged into those
    defaults rather than sent alone.
    """
    from diffusers.pipelines.ltx2.utils import GEMMA4_PROMPT_ENHANCEMENT_CONFIG
    enhancer = getattr(pipe, 'prompt_enhancer', None)
    name = type(enhancer).__name__ if enhancer is not None else 'prompt enhancer'
    max_new = enhance_kwargs.get(
        'max_new_tokens', GEMMA4_PROMPT_ENHANCEMENT_CONFIG.max_new_tokens)
    counter = _TokenCounter(name, int(max_new))
    generation = dict(enhance_kwargs.get(
        'generation_kwargs', GEMMA4_PROMPT_ENHANCEMENT_CONFIG.generation_kwargs))
    generation['streamer'] = _CountStreamer(counter)
    enhance_kwargs['generation_kwargs'] = generation
    return counter


def _ltx_gemma_enhance_generation(pipe, overrides: PromptEnhanceOverrides | None,
                                  *, dedicated_enhancer: bool) -> tuple[int, dict]:
    """``max_new_tokens`` and ``generation_kwargs`` matching Diffusers ``enhance_prompt``."""
    from diffusers.pipelines.ltx2.utils import (
        GEMMA3_PROMPT_ENHANCEMENT_CONFIG,
        GEMMA4_PROMPT_ENHANCEMENT_CONFIG,
    )
    config = GEMMA4_PROMPT_ENHANCEMENT_CONFIG if dedicated_enhancer else GEMMA3_PROMPT_ENHANCEMENT_CONFIG
    if overrides is not None and overrides.max_new_tokens is not None:
        max_new, _min_new = merged_generation_limits(_enums.ModelType.LTX2, None, overrides)
    else:
        max_new = config.max_new_tokens
    generation = dict(config.generation_kwargs)
    if overrides is None or not overrides.any_set():
        return max_new, generation
    if overrides.temperature is not None:
        generation['temperature'] = overrides.temperature
        generation['do_sample'] = overrides.temperature > 0
    if overrides.top_p is not None:
        generation['top_p'] = overrides.top_p
        generation['do_sample'] = True
    if overrides.top_k is not None:
        if overrides.top_k > 0:
            generation['top_k'] = overrides.top_k
        else:
            generation.pop('top_k', None)
    if overrides.presence_penalty is not None:
        generation['presence_penalty'] = overrides.presence_penalty
    if overrides.min_new_tokens is not None and overrides.min_new_tokens > 0:
        generation['min_new_tokens'] = overrides.min_new_tokens
    elif overrides.min_new_tokens == 0:
        generation.pop('min_new_tokens', None)
    return max_new, generation


def _generate_ltx_enhance(pipe, encoder, prompt: str, system_prompt: str, image, seed, counter,
                          overrides: PromptEnhanceOverrides | None,
                          *, dedicated_enhancer: bool = False) -> str:
    """LTX text-encoder or Gemma enhancer rewrite, matching Diffusers input prep and decoding."""
    from diffusers.pipelines.ltx2.prompt_enhancement import (
        _pad_inputs_for_attention_alignment,
        _prepare_enhance_image,
        clean_response as ltx_clean_response,
    )
    processor = getattr(pipe, 'processor', None)
    if processor is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            'LTX prompt enhance needs the pipeline processor.')
    max_new_tokens, generation_kwargs = _ltx_gemma_enhance_generation(
        pipe, overrides, dedicated_enhancer=dedicated_enhancer)
    counter.maximum = max_new_tokens
    if image is None:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'user prompt: {prompt}'},
        ]
        enhance_image = None
    else:
        enhance_image = _prepare_enhance_image(image)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': f'User Raw Input Prompt: {prompt}.'},
                ],
            },
        ]
    template = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    model_inputs = processor(text=template, images=enhance_image, return_tensors='pt')
    pad_token_id = (
        processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0)
    model_inputs = _pad_inputs_for_attention_alignment(
        model_inputs, pad_token_id=pad_token_id)
    input_device = _embedding_device(encoder)
    model_inputs = {
        key: value.to(input_device) if torch.is_tensor(value) else value
        for key, value in model_inputs.items()
    }
    encoder.eval()
    streamer = _CountStreamer(counter)
    counter.show()
    try:
        with torch.inference_mode(), _forked_seed(seed):
            generated_sequences = encoder.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                **generation_kwargs)
    finally:
        counter.finish()
    input_len = model_inputs['input_ids'].shape[-1]
    generated_ids = [sequence[input_len:] for sequence in generated_sequences]
    decoded = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    cleaned = ltx_clean_response(decoded[0]) if decoded else ''
    if not cleaned:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--pipeline-prompt-enhance produced an empty prompt.')
    return cleaned

_GENERATE_KEYS = {
    'input_ids', 'attention_mask', 'position_ids', 'pixel_values',
    'image_grid_thw', 'pixel_values_videos', 'video_grid_thw', 'mm_token_type_ids',
}


_FILE_PROMPT = re.compile(r'^file:\s*(.*)\Z', re.DOTALL)


def system_prompt_file_path(value) -> str | None:
    """Path named by ``file:path``, or ``None`` when *value* is not a file prompt.

    ``file:`` with no path raises :py:exc:`.UnsupportedPipelineConfigError`.
    """
    if not isinstance(value, str):
        return None
    match = _FILE_PROMPT.match(value.strip())
    if match is None:
        return None
    path = (match.group(1) or '').strip()
    if not path:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--pipeline-prompt-enhance file: needs a file path.')
    return path


def normalize_pipeline_prompt_enhance_flag(value):
    """Coerce blank ``--pipeline-prompt-enhance`` text to ``True`` (built-in prompt)."""
    if isinstance(value, str) and not value.strip():
        return True
    return value


def check_system_prompt_file(value) -> None:
    """Raise when a ``file:`` system prompt does not name an existing file."""
    path = system_prompt_file_path(value)
    if path is None:
        return
    if not os.path.isfile(path):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'--pipeline-prompt-enhance file does not exist: "{path}"')


def resolve_system_prompt(value):
    """Return a system prompt string, reading ``file:`` values from disk.

    ``file:path`` names a UTF-8 text file (paths with spaces must be shell-quoted
    on the command line). Any other string is the prompt itself. Blank text means
    the built-in prompt.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return True
    path = system_prompt_file_path(text)
    if path is None:
        return text
    try:
        with open(path, encoding='utf-8') as handle:
            loaded = handle.read().strip()
    except OSError as error:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'--pipeline-prompt-enhance could not read "{path}": {error}') from error
    if not loaded:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'--pipeline-prompt-enhance file "{path}" is empty.')
    return loaded


def auto_system_prompt(model_type, has_image: bool, *,
                       encoder_name: str | None = None,
                       overrides: PromptEnhanceOverrides | None = None,
                       image_count: int = 1) -> str:
    """Built-in prompt for this model, encoder, and token budget.

    The length is the sampler's minimum and maximum. Multilingual enhance is selected
    by ``--model-type`` (Qwen-Image family, ``flux2``, ``flux2-klein-kv``). Z-Image and
    Krea 2 rewrite in English. Token budgets may include a CJK character count when the
    rewrite is not forced to English.
    Edit wording is used only when this encoder can see the picture.
    Qwen edit models are told to say ``<image1>``. Flux names the images in order.
    Tied Qwen encoders get the shorter text-to-image prompt.
    Qwen2.5-VL uses that shorter prompt, cooler sampling, and a user-turn language reminder.
    """
    max_tokens, min_tokens = merged_generation_limits(model_type, encoder_name, overrides)
    if _wants_edit_prompt(model_type, encoder_name, has_image):
        return edit_system_prompt(
            min_tokens, max_tokens,
            placeholders=_uses_image_placeholders(model_type) and image_count >= 1,
            model_type=model_type, encoder_name=encoder_name)
    return t2i_system_prompt(
        min_tokens, max_tokens,
        brief=_brief_encoder(model_type, encoder_name),
        model_type=model_type, encoder_name=encoder_name)


def _log_text_only_encoder() -> None:
    _messages.log(
        'Pipeline prompt enhance: this text encoder cannot read images, '
        'so the rewrite is text-only.')


@contextlib.contextmanager
def _forked_seed(seed):
    """Seed ``generate()`` without changing the process-wide RNG afterward.

    Transformers 5 ``generate`` reads the default generators. A local
    ``torch.Generator`` is not accepted. ``fork_rng`` puts the seed back.
    """
    with torch.random.fork_rng():
        torch.manual_seed(int(seed or 0))
        yield


class _PresencePenaltyLogitsProcessor:
    """Subtract a penalty once for each token already produced, matching ``_sample``."""

    def __init__(self, penalty: float, prompt_length: int):
        self.penalty = float(penalty)
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if not self.penalty:
            return scores
        for row in range(input_ids.shape[0]):
            produced = input_ids[row, self.prompt_length:]
            if produced.numel() == 0:
                continue
            for token_id in {int(token) for token in produced.tolist()}:
                if 0 <= token_id < scores.shape[-1]:
                    scores[row, token_id] -= self.penalty
        return scores


def _head_generate_kwargs(sampling: Sampling, max_new_tokens: int, min_new_tokens: int,
                          prompt_length: int) -> dict:
    """``generate`` kwargs for one rewrite.

    Temperature 0 is greedy and omits sampling flags. ``top_k`` 0 is passed
    through so a model ``generation_config`` cannot turn top-k back on.
    """
    kwargs = {
        'max_new_tokens': max_new_tokens,
        'min_new_tokens': min_new_tokens,
        'do_sample': sampling.temperature > 0,
    }
    if sampling.temperature > 0:
        kwargs['temperature'] = sampling.temperature
        kwargs['top_p'] = sampling.top_p
        kwargs['top_k'] = sampling.top_k if sampling.top_k else 0
    if sampling.presence_penalty:
        kwargs['logits_processor'] = [
            _PresencePenaltyLogitsProcessor(sampling.presence_penalty, prompt_length)]
    return kwargs


class Sampling:
    """Token sampling for one text-encoder family."""

    def __init__(self, temperature: float, top_p: float, top_k: int, presence_penalty: float):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty


# Qwen non-thinking chat settings.
_QWEN = Sampling(0.7, 0.8, 20, 1.5)
# Qwen2.5-VL drifts to Chinese on English prompts; cooler than Qwen3 defaults.
_QWEN25_VL = Sampling(0.35, 0.85, 20, 1.5)
# Mistral Instruct.
_MISTRAL = Sampling(0.7, 0.95, 50, 0.0)


def sampling_for(model_type, encoder_name: str | None = None) -> Sampling:
    """Sampling settings for ``--pipeline-prompt-enhance`` on this model type.

    Flux.2 dev is Mistral. Flux.2 Klein uses the same ``flux2`` model type with
    a Qwen3 encoder, so the encoder class selects the Qwen settings there.
    """
    model_type = _enums.get_model_type_enum(model_type)
    if model_type is not None and _enhance_english_only(model_type):
        if encoder_name is None or encoder_name.startswith('Qwen3'):
            return _QWEN25_VL
    if model_type == _enums.ModelType.FLUX2 and encoder_name and encoder_name.startswith('Qwen'):
        return _QWEN
    if model_type is not None and _keep_user_language(model_type):
        if encoder_name is None or encoder_name.startswith('Qwen3'):
            return _QWEN25_VL
    if model_type == _enums.ModelType.FLUX2:
        return _MISTRAL
    if _is_qwen25_vl_encoder(encoder_name):
        return _QWEN25_VL
    return _QWEN


_PROMPT_LABEL = re.compile(
    r'(?:image\s+prompt|editing\s+instruction|instruction|prompt)\s*:\s*',
    re.IGNORECASE)

_INSTRUCTION_ECHO = re.compile(
    r"You write prompts for an image generation model\.[^.]*\.\s*"
    r"|Expand the user's request into one paragraph[^.]*\.\s*"
    r"|Output only the prompt\.?\s*",
    re.IGNORECASE)

_META_TAIL = re.compile(
    r'\s*(?:towards the end of the prompt\.?|at the end of the prompt\.?)\s*$',
    re.IGNORECASE)


_USER_PREFIX_ECHO = re.compile(
    r'^(?:Write your entire answer|The request below is in English|'
    r'将您的整个回答|把你的整个回答|回答はユーザー|'
    r'전체 답변을|Répondez entièrement|Escribe toda tu respuesta)'
    r'[^\n]*(?:\n[^\n]*){0,3}\n*',
    re.IGNORECASE)


def _iter_user_language_prefixes():
    yield _QWEN25_ENGLISH_USER_TURN_PREFIX
    yield _QWEN25_USER_TURN_PREFIX_SOURCE
    for cached in _QWEN25_TRANSLATED_PREFIX_CACHE.values():
        yield cached


def _strip_echoed_user_language_prefix(text: str) -> str:
    cleaned = text.strip()
    for prefix in sorted(_iter_user_language_prefixes(), key=len, reverse=True):
        piece = prefix.strip()
        if not piece:
            continue
        if cleaned.startswith(piece):
            cleaned = cleaned[len(piece):].strip()
        elif cleaned == piece.rstrip('.'):
            cleaned = ''
    cleaned = _USER_PREFIX_ECHO.sub('', cleaned).strip()
    return cleaned


_GLUED_TAIL = re.compile(
    r'(?<=[.!?])\s*[A-Za-z]{0,4}opyright\w*\b.*$',
    re.IGNORECASE)
_LEAKED_SIGN_SLOT = re.compile(
    r'[_\s]*SIGN\d+[_\s]*',
    re.IGNORECASE)


def _strip_decode_garbage(text: str) -> str:
    if not text:
        return text
    text = text.replace('\ufffd', ' ').replace('\uFFFD', ' ')
    text = _GLUED_TAIL.sub('', text)
    text = _LEAKED_SIGN_SLOT.sub(' ', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


def _strip_instruction_echo(text: str) -> str:
    cleaned = _INSTRUCTION_ECHO.sub('', text)
    cleaned = _META_TAIL.sub('', cleaned)
    return cleaned.strip()


def clean_response(text: str) -> str | None:
    """Drop a leading label, a thinking trace, and quotes wrapped around the whole answer."""
    if text is None:
        return None
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = _strip_instruction_echo(cleaned.strip())
    cleaned = _strip_echoed_user_language_prefix(cleaned)
    cleaned = _strip_decode_garbage(cleaned).strip()
    # ChatGLM sometimes opens with a short line in another language, then a label.
    lines = cleaned.splitlines()
    while len(lines) > 1 and not re.search(r'[A-Za-z]', lines[0]):
        lines.pop(0)
    cleaned = '\n'.join(lines).strip()
    label = _PROMPT_LABEL.search(cleaned)
    if label is not None and label.start() <= 40:
        cleaned = cleaned[label.end():].strip()
    if (len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"'
            and '"' not in cleaned[1:-1]):
        cleaned = cleaned[1:-1].strip()
    cleaned = _drop_repetitive_tail(cleaned)
    if cleaned.startswith('- '):
        cleaned = cleaned[2:].strip()
    elif cleaned.startswith('* '):
        cleaned = cleaned[2:].strip()
    return cleaned or None


_LATIN_LANGS = frozenset({'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'sv', 'da', 'no'})
_CJK_LANGS = frozenset({'zh', 'ja', 'ko'})

_TRAILING_CJK = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]+$')

_LATIN_INLINE = re.compile(
    r'\s+(?:[A-Za-z]+(?:\s+(?:and|or|和)\s+[A-Za-z]+)*)')


def _language_compatible(source_lang: str | None, output_lang: str | None) -> bool:
    if not source_lang or not output_lang:
        return True
    if source_lang == output_lang:
        return True
    if source_lang in _CJK_LANGS and output_lang in _CJK_LANGS:
        return source_lang == output_lang
    return False


def _source_script_present(source: str, lang: str) -> bool:
    if lang == 'ja':
        return bool(re.search(r'[\u3040-\u30ff]', source))
    if lang == 'ko':
        return bool(re.search(r'[\uac00-\ud7af]', source))
    if lang == 'zh':
        return bool(re.search(r'[\u4e00-\u9fff]', source))
    if lang in _LATIN_LANGS:
        return bool(re.search(r'[A-Za-z]', source))
    return True


def _output_keeps_source_script(source: str, enhanced: str, lang: str) -> bool:
    if lang in _LATIN_LANGS:
        latin = len(re.findall(r'[A-Za-z]', enhanced))
        cjk = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', enhanced))
        if not _source_script_present(source, lang):
            return latin >= max(12, cjk * 2)
    if not _source_script_present(source, lang):
        return True
    if lang == 'ja':
        return bool(re.search(r'[\u3040-\u30ff]', enhanced))
    if lang == 'ko':
        return bool(re.search(r'[\uac00-\ud7af]', enhanced))
    if lang == 'zh':
        return bool(re.search(r'[\u4e00-\u9fff]', enhanced))
    if lang in _LATIN_LANGS:
        latin = len(re.findall(r'[A-Za-z]', enhanced))
        cjk = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', enhanced))
        return latin >= max(12, cjk * 2)
    return True


def _trim_mixed_script_pollution(enhanced: str, lang: str) -> str:
    text = enhanced.strip()
    if lang in _LATIN_LANGS:
        text = _TRAILING_CJK.sub('', text).strip()
        return text
    if lang in _CJK_LANGS:
        text = _LATIN_INLINE.sub('', text)
        match = re.search(r'[A-Za-z]{12,}', text)
        if match and match.start() >= 20:
            text = text[:match.start()].strip()
        return text
    return text


def _translate_prompt_to_english(text: str, *, local_files_only: bool = False,
                                 lang: str | None = None) -> str:
    if lang is None:
        lang = _normalize_detected_language(_detect_prompt_language(text))
    else:
        lang = _normalize_detected_language(lang)
    if lang in (None, 'en'):
        return text.strip()
    import dgenerate.translators as _translators
    try:
        translator = _translators.ArgosTranslator(lang, 'en', local_files_only=local_files_only)
    except (_translators.TranslatorLoadError, _translators.TranslationError) as error:
        _messages.debug_log(f'Pipeline prompt enhance: Argos English fallback failed: {error}')
        return text.strip()
    parts: list[str] = []
    last = 0
    for match in _SIGN_LITERAL.finditer(text):
        parts.append(_translate_chunk(translator, text[last:match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(_translate_chunk(translator, text[last:]))
    return re.sub(r'\s+', ' ', ''.join(parts)).strip()


def _translate_chunk(translator, chunk: str) -> str:
    if not chunk.strip():
        return chunk
    import dgenerate.translators as _translators
    try:
        return translator.translate(chunk)[0]
    except (_translators.TranslatorLoadError, _translators.TranslationError) as error:
        _messages.debug_log(f'Pipeline prompt enhance: Argos English fallback failed: {error}')
        return chunk


def _english_fallback(source: str, reason: str, *, local_files_only: bool = False) -> str:
    _messages.debug_log(
        f'Pipeline prompt enhance: {reason}; using an English translation of the original prompt.')
    return _translate_prompt_to_english(source, local_files_only=local_files_only)


def finalize_multilingual_rewrite(source: str, enhanced: str, model_type, *,
                                  local_files_only: bool = False) -> str:
    """Validate rewrite language; English-only families target English captions."""
    source = source.strip()
    enhanced = enhanced.strip()
    if _enhance_english_only(model_type):
        return _finalize_english_rewrite(
            source, enhanced, model_type=model_type, local_files_only=local_files_only)
    if not _keep_user_language(model_type):
        return enhanced
    if not source or not enhanced:
        return enhanced or source
    src_lang = _normalize_detected_language(_detect_prompt_language(source))
    if not src_lang:
        return enhanced
    out_lang = _normalize_detected_language(_detect_prompt_language(enhanced))
    if not _language_compatible(src_lang, out_lang):
        return _fallback_to_source_prompt(
            source,
            f'rewrite looks like {out_lang or "unknown"}, expected {src_lang}')
    if not _output_keeps_source_script(source, enhanced, src_lang):
        return _fallback_to_source_prompt(
            source, f'rewrite dropped the {src_lang} script')
    return _trim_mixed_script_pollution(enhanced, src_lang)


def _finalize_english_rewrite(source: str, enhanced: str, *, model_type=None,
                              local_files_only: bool = False) -> str:
    text = _strip_decode_garbage(_trim_mixed_script_pollution(enhanced, 'en')).strip()
    if not text:
        return _english_fallback(source, 'rewrite was empty after cleanup',
                                 local_files_only=local_files_only)
    if not _output_keeps_source_script('', text, 'en'):
        out_lang = _normalize_detected_language(_detect_prompt_language(text))
        return _english_fallback(
            source, f'rewrite was not English ({out_lang or "wrong script"})',
            local_files_only=local_files_only)
    return text


_MIN_USABLE_REWRITE_CHARS = 12


def _rewrite_is_usable(source: str, cleaned: str | None, model_type=None) -> bool:
    if not cleaned or not cleaned.strip():
        return False
    if _USER_PREFIX_ECHO.match(cleaned.strip()):
        return False
    if len(source.strip()) >= _MIN_USABLE_REWRITE_CHARS and len(cleaned.strip()) < _MIN_USABLE_REWRITE_CHARS:
        return False
    if model_type is not None and _enhance_english_only(model_type):
        return _output_keeps_source_script('', cleaned, 'en')
    src_lang = _normalize_detected_language(_detect_prompt_language(source))
    if src_lang in _CJK_LANGS:
        if not _output_keeps_source_script(source, cleaned, src_lang):
            return False
    return True


def _fallback_to_source_prompt(source: str, reason: str) -> str:
    _messages.debug_log(f'Pipeline prompt enhance: {reason}; keeping the original prompt.')
    return source.strip()


def _drop_repetitive_tail(text: str) -> str:
    """Trim ChatGLM list spam and other copy-pasted lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return text.strip()
    if len(lines) >= 2 and len(set(lines[1:])) == 1:
        return lines[0]
    for index in range(1, len(lines) - 1):
        if lines[index] == lines[index - 1] == lines[index + 1]:
            return '\n'.join(lines[:index])
    repeat = max(lines, key=lines.count)
    if lines.count(repeat) >= 3 and len(repeat) >= 12:
        first = lines.index(repeat)
        if first == 0:
            return repeat
        return '\n'.join(lines[:first])
    return text.strip()


def enhance_diffusion_arguments(wrapper, user_args) -> None:
    """Rewrite ``user_args.prompt`` with the pipeline text encoder, in place."""
    setting = user_args.pipeline_prompt_enhance
    if not setting or _enums.get_model_type_enum(wrapper.model_type) == _enums.ModelType.LTX2:
        return
    prompt = user_args.prompt
    if prompt is None or not (prompt.positive or '').strip():
        return
    images = _condition_images(user_args)
    system_prompt = setting if isinstance(setting, str) else None
    pipe = wrapper._pipeline
    offload = bool(
        wrapper.model_cpu_offload or wrapper.model_sequential_offload or wrapper.model_group_offload)
    overrides = PromptEnhanceOverrides.from_args(user_args)
    rewritten = rewrite_prompt(
        pipe,
        prompt.positive,
        system_prompt,
        images,
        wrapper.device,
        user_args.pipeline_prompt_enhance_seed
        if user_args.pipeline_prompt_enhance_seed is not None else user_args.seed,
        local_files_only=bool(getattr(wrapper, '_local_files_only', False)),
        offload=offload,
        model_type=wrapper.model_type,
        overrides=overrides)
    _messages.log(f'Pipeline prompt enhance: {rewritten}')
    revised = prompt.__class__.copy(prompt)
    revised.positive = rewritten
    user_args.prompt = revised


def _pipeline_tokenizer(pipe):
    """Tokenizer registered on the pipeline, or the one owned by its processor.

    Qwen-Image 2.1 has ``processor`` (a ``Qwen3VLProcessor``) and no
    ``tokenizer`` attribute. The tokenizer is ``processor.tokenizer``.
    """
    tokenizer = getattr(pipe, 'tokenizer', None)
    if tokenizer is not None:
        return tokenizer
    processor = getattr(pipe, 'processor', None)
    if processor is None:
        return None
    return getattr(processor, 'tokenizer', None)


def rewrite_prompt(pipe,
                   prompt: str,
                   system_prompt: str | None,
                   images,
                   device,
                   seed,
                   local_files_only: bool = False,
                   offload: bool = False,
                   model_type=None,
                   overrides: PromptEnhanceOverrides | None = None,
                   ltx_enhance: bool = False) -> str:
    encoder = getattr(pipe, 'text_encoder', None)
    tokenizer = _pipeline_tokenizer(pipe)
    if encoder is None or tokenizer is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--pipeline-prompt-enhance needs the pipeline text encoder and tokenizer.')
    name = type(encoder).__name__
    check_generation_overrides(
        overrides,
        lambda attr: '--' + attr.replace('_', '-'),
        model_type,
        encoder_name=name)
    has_images = bool(images)
    if system_prompt is None and not ltx_enhance:
        image_count = len(images or [])
        system_prompt = auto_system_prompt(
            model_type, has_images, encoder_name=name, overrides=overrides,
            image_count=image_count)
    sampling = merged_sampling(model_type, name, overrides)
    max_new, _min_new = merged_generation_limits(model_type, name, overrides)
    text = None
    for resident in (True, False):
        counter = _TokenCounter(name, max_new)
        try:
            with _fast_encoder(
                    encoder, device, offload, resident=resident,
                    headroom_bytes=_headroom_bytes(overrides)):
                counter.show()
                text = _rewrite_with_encoder(
                    pipe, encoder, tokenizer, name, prompt, system_prompt, images, seed,
                    local_files_only, counter, sampling, model_type, overrides,
                    ltx_enhance=ltx_enhance)
            break
        except torch.cuda.OutOfMemoryError:
            if not resident:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _messages.log(
                'Pipeline prompt enhance: ran out of GPU memory during the rewrite, using offload.')
    if text is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--pipeline-prompt-enhance produced an empty prompt.')
    source = prompt.strip()
    cleaned = clean_response(text)
    if not _rewrite_is_usable(source, cleaned, model_type):
        if _enhance_english_only(model_type):
            return _english_fallback(
                source, 'rewrite was empty or too short after cleanup',
                local_files_only=local_files_only)
        return _fallback_to_source_prompt(
            source, 'rewrite was empty or too short after cleanup')
    result = finalize_multilingual_rewrite(
        source, cleaned, model_type, local_files_only=local_files_only)
    return result


def _rewrite_with_encoder(pipe, encoder, tokenizer, name, prompt, system_prompt, images, seed,
                          local_files_only, counter, sampling, model_type,
                          overrides: PromptEnhanceOverrides | None,
                          *, ltx_enhance: bool = False) -> str:
    max_new, min_new = merged_generation_limits(model_type, name, overrides)
    prompt = _rewrite_user_prompt(
        prompt, model_type=model_type, encoder_name=name,
        local_files_only=local_files_only)
    if ltx_enhance:
        image = images[0] if images else None
        return _generate_ltx_enhance(
            pipe, encoder, prompt, system_prompt, image, seed, counter, overrides)
    if name == 'Qwen3Model':
        if images:
            _log_text_only_encoder()
        return _generate_tied(
            encoder, tokenizer, encoder.embed_tokens, prompt, system_prompt, seed, counter,
            sampling, max_new_tokens=max_new, min_new_tokens=min_new)
    if name == 'Qwen3VLModel':
        if images:
            _log_text_only_encoder()
        language = encoder.language_model
        return _generate_tied(
            language, tokenizer, language.embed_tokens,
            prompt, system_prompt, seed, counter, sampling,
            max_new_tokens=max_new, min_new_tokens=min_new)
    if callable(getattr(encoder, 'generate', None)):
        return _generate_with_head(
            pipe, encoder, tokenizer, prompt, system_prompt, images, seed,
            local_files_only, counter, sampling,
            max_new_tokens=max_new, min_new_tokens=min_new,
            model_type=model_type, encoder_name=name)
    raise _pipelines.UnsupportedPipelineConfigError(
        f'--pipeline-prompt-enhance cannot generate text with {name}.')


def _condition_images(user_args):
    images = []
    for image in user_args.images or ():
        if image is None:
            continue
        if isinstance(image, (list, tuple)):
            image = image[0] if image else None
        if image is not None and not torch.is_tensor(image):
            images.append(image)
    return images


_LAYER_CLASS_NAMES = {
    'GLMBlock',
    'GPT2Block',
    'GPTNeoBlock',
    'GPTNeoXLayer',
    'GPTJBlock',
    'BloomBlock',
}


def _is_decoder_layer(module) -> bool:
    name = type(module).__name__
    return name in _LAYER_CLASS_NAMES or name.endswith('DecoderLayer')


def _resident_bytes(module, device: torch.device) -> int:
    seen = set()
    total = 0
    for tensor in module.parameters():
        if tensor.device.type != device.type or tensor.data_ptr() in seen:
            continue
        seen.add(tensor.data_ptr())
        total += tensor.numel() * tensor.element_size()
    return total


def _module_nbytes(module) -> int:
    seen = set()
    total = 0
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        if tensor is None or tensor.numel() == 0:
            continue
        if tensor.device.type == 'meta' or tensor.data_ptr() == 0:
            total += tensor.numel() * tensor.element_size()
            continue
        if tensor.data_ptr() in seen:
            continue
        seen.add(tensor.data_ptr())
        total += tensor.numel() * tensor.element_size()
    return total


def _contained(name: str, parents: list[str]) -> bool:
    return any(name == parent or name.startswith(parent + '.') for parent in parents)


def _placement_units(model):
    """Embedding, decoder layers, then the norm and head, in forward order."""
    layers = _drop_descendants(
        [(name, module) for name, module in model.named_modules() if _is_decoder_layer(module)])
    layer_names = [name for name, _ in layers]
    embeddings = []
    norms = []
    heads = []
    for name, module in model.named_modules():
        if _contained(name, layer_names):
            continue
        cls = type(module).__name__
        if cls in {'Embedding', 'EmbeddingBag'} or name.endswith(('embed_tokens', 'word_embeddings', 'embedding')):
            embeddings.append((name, module))
        elif name == 'norm' or name.endswith(('.norm', 'final_layernorm')):
            norms.append((name, module))
        elif name.endswith(('lm_head', 'output_layer')) or cls == 'lm_head':
            heads.append((name, module))
    embeddings = _drop_descendants(embeddings)
    return embeddings, layers, _drop_descendants(norms), _drop_descendants(heads)


def _drop_descendants(units):
    names = [name for name, _ in units]
    return [(name, module) for name, module in units
            if not any(name.startswith(other + '.') for other in names)]


def _fresh_bytes(module, assigned: dict) -> int:
    total = 0
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        if tensor is None or tensor.numel() == 0:
            continue
        pointer = None if tensor.device.type == 'meta' else tensor.data_ptr()
        if pointer and pointer in assigned:
            continue
        total += tensor.numel() * tensor.element_size()
    return total


def _remember(module, device_name: str, assigned: dict):
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        if tensor is None or tensor.device.type == 'meta' or tensor.data_ptr() == 0:
            continue
        assigned.setdefault(tensor.data_ptr(), device_name)


def _modules_for_gpu(model, budget: int) -> list[str]:
    """Names of a GPU prefix that fits in ``budget`` bytes.

    Layers stay in order. The first layer that does not fit, and everything
    after it, stays on the CPU, so the activation crosses the bus once.
    """
    if budget <= 0:
        return []
    embeddings, layers, norms, heads = _placement_units(model)
    text_embeddings = [(name, module) for name, module in embeddings if not _is_vision(name)]
    vision_embeddings = [(name, module) for name, module in embeddings if _is_vision(name)]
    assigned = {}
    chosen = []

    def take(name, module) -> bool:
        pointers = [
            tensor.data_ptr()
            for tensor in itertools.chain(module.parameters(), module.buffers())
            if tensor is not None and tensor.device.type != 'meta' and tensor.data_ptr() != 0]
        if pointers and all(pointer in assigned for pointer in pointers):
            if assigned[pointers[0]] == 'gpu':
                chosen.append(name)
            return True
        size = _fresh_bytes(module, assigned)
        if size > budget_left[0]:
            _remember(module, 'cpu', assigned)
            return False
        budget_left[0] -= size
        _remember(module, 'gpu', assigned)
        chosen.append(name)
        return True

    budget_left = [budget]
    for name, module in text_embeddings:
        take(name, module)
    fit_all_layers = True
    for name, module in layers:
        if not take(name, module):
            fit_all_layers = False
            break
    if fit_all_layers:
        for name, module in norms + heads + vision_embeddings:
            take(name, module)
    return chosen


def _is_vision(name: str) -> bool:
    lower = name.lower()
    return 'visual' in lower or 'vision' in lower or 'image_tower' in lower


def _headroom_bytes(overrides: PromptEnhanceOverrides | None) -> int:
    """Bytes of free GPU memory left unused on top of the 10 percent reserve."""
    if overrides is None or overrides.gpu_headroom is None:
        return _DEFAULT_GPU_HEADROOM_BYTES
    return int(overrides.gpu_headroom * (1024 ** 3))


# Activations, the KV cache, and the one CPU-stored layer visiting the GPU.
_DEFAULT_GPU_HEADROOM_BYTES = 2 * 1024 ** 3


def _gpu_budget_bytes(device: torch.device, headroom_bytes: int | None = None) -> int:
    free = 0
    if device.type == 'cuda' and torch.cuda.is_available():
        index = device.index
        free = int(torch.cuda.mem_get_info(0 if index is None else index)[0])
    elif device.type == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        index = device.index
        free = int(torch.xpu.mem_get_info(0 if index is None else index)[0])
    if free <= 0:
        return 0
    extra = _DEFAULT_GPU_HEADROOM_BYTES if headroom_bytes is None else max(0, headroom_bytes)
    reserve = max(512 * 1024 ** 2, free // 10) + extra
    if reserve >= free:
        return 0
    return free - reserve


def _is_weight_cache(value) -> bool:
    if type(value).__name__.endswith('Cache'):
        return True
    return (
        isinstance(value, tuple) and len(value) > 4 and bool(value)
        and isinstance(value[0], tuple))


def _move_activation(value, device: torch.device):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _is_weight_cache(value):
        return value
    if torch.is_tensor(value):
        return value if value.device == device else value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_activation(item, device) for item in value)
    if isinstance(value, list):
        return [_move_activation(item, device) for item in value]
    return value


def _place_layer_cache(module, cache, device: torch.device):
    """Keep this decoder layer's KV cache on the device that will run it."""
    if cache is None:
        return
    index = getattr(module, 'layer_idx', None)
    if index is None:
        attention = getattr(module, 'self_attn', None)
        index = getattr(attention, 'layer_idx', None) if attention is not None else None
    layers = getattr(cache, 'layers', None)
    if index is None or not layers or index >= len(layers):
        return
    layer = layers[index]
    for name in ('keys', 'values'):
        tensor = getattr(layer, name, None)
        if torch.is_tensor(tensor) and tensor.numel() and tensor.device != device:
            setattr(layer, name, tensor.to(device))
    if hasattr(layer, 'device'):
        layer.device = device


def _move_shared_kv(value, device: torch.device):
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if isinstance(item, tuple):
                value[key] = tuple(
                    tensor.to(device) if torch.is_tensor(tensor) and tensor.device != device else tensor
                    for tensor in item)
        return value
    return _move_activation(value, device)


class _LayerDeviceHook:
    """Move activations onto the layer's device and leave its KV cache there."""

    no_grad = False

    def __init__(self, device: torch.device):
        self.device = torch.device(device)

    def init_hook(self, module):
        return module

    def pre_forward(self, module, *args, **kwargs):
        args = tuple(_move_activation(value, self.device) for value in args)
        for key, value in list(kwargs.items()):
            if key in ('past_key_values', 'past_key_value'):
                _place_layer_cache(module, value, self.device)
                continue
            if key == 'shared_kv_states':
                kwargs[key] = _move_shared_kv(value, self.device)
                continue
            kwargs[key] = _move_activation(value, self.device)
        return args, kwargs

    def post_forward(self, module, output):
        return output

    def detach_hook(self, module):
        return module


class _ExecuteOnDeviceHook(_LayerDeviceHook):
    """Run a CPU-stored layer on the GPU, then return its weights to CPU.

    Gemma bf16 layers produce junk if the matmul itself runs on the CPU.
    Weights that do not fit stay in CPU memory and visit the GPU for one
    forward only, while the resident prefix stays on the GPU.
    """

    def __init__(self, device: torch.device, home: torch.device):
        super().__init__(device)
        self.home = torch.device(home)

    def pre_forward(self, module, *args, **kwargs):
        _move_module(module, self.device)
        return super().pre_forward(module, *args, **kwargs)

    def post_forward(self, module, output):
        _move_module(module, self.home)
        return output


def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def _rotary_modules(encoder):
    return [
        (name, module) for name, module in encoder.named_modules()
        if 'RotaryEmbedding' in type(module).__name__]


def _buffer_device(module):
    device = _module_device(module)
    if device is not None:
        return device
    try:
        return next(module.buffers()).device
    except StopIteration:
        return None


def _text_embedding_device(embeddings):
    for name, module in embeddings:
        if _is_vision(name):
            continue
        device = _module_device(module)
        if device is not None:
            return device
    return None


def _move_language_rotary(encoder, embeddings):
    """Put language RoPE on the text embedding. Leave vision RoPE where it is."""
    text_device = _text_embedding_device(embeddings)
    if text_device is None:
        return
    for name, module in _rotary_modules(encoder):
        if _is_vision(name):
            continue
        module.to(text_device)


def _hook_rotary_inputs(encoder):
    """Move RoPE inputs onto the frequency buffer before the matmul.

    ``generate()`` places ``position_ids`` on ``model.device``. A vision
    encoder's first parameter is the tower, so those ids stay on CPU while
    language RoPE has moved with the text embedding.
    """
    from accelerate.hooks import add_hook_to_module
    for _, module in _rotary_modules(encoder):
        home = _buffer_device(module)
        if home is None:
            continue
        add_hook_to_module(module, _LayerDeviceHook(home))


def _hook_resident_layers(model, units, compute_device: torch.device):
    from accelerate.hooks import add_hook_to_module
    covered = {id(module) for _, module in units}
    targets = [module for _, module in units]
    for child in model.children():
        if id(child) in covered or any(id(module) in covered for module in child.modules()):
            continue
        if any(tensor.numel() for tensor in child.parameters()):
            targets.append(child)
    for module in targets:
        home = _module_device(module)
        if home is None:
            continue
        if home.type == compute_device.type:
            add_hook_to_module(module, _LayerDeviceHook(home))
        else:
            add_hook_to_module(module, _ExecuteOnDeviceHook(compute_device, home))


def _embedding_device(model) -> torch.device:
    embeddings, _, _, _ = _placement_units(model)
    for _, module in embeddings:
        try:
            return next(module.parameters()).device
        except StopIteration:
            continue
    return next(model.parameters()).device


def _placement_log(message: str, *, debug: bool) -> None:
    if debug:
        _messages.debug_log(message)
        return
    _messages.log(message)


def _log_layer_split(encoder, target: torch.device, layers, label: str, *, debug: bool = False) -> None:
    gpu_layers = 0
    for _, layer in layers:
        try:
            gpu_layers += next(layer.parameters()).device.type == target.type
        except StopIteration:
            continue
    resident = _memory.bytes_best_human_unit(_resident_bytes(encoder, target))
    if gpu_layers >= len(layers):
        _placement_log(
            f'{label}: '
            f'{gpu_layers} of {len(layers)} layers on {target} ({resident}).',
            debug=debug)
        return
    _placement_log(
        f'{label}: '
        f'{gpu_layers} of {len(layers)} layers on {target} ({resident}), '
        f'{len(layers) - gpu_layers} stored on CPU and run on {target}.',
        debug=debug)


@contextlib.contextmanager
def _real_weights():
    """Move weights that may be cached outside an enclosing inference block.

    ``torch.inference_mode(False)`` nested inside inference mode creates
    ordinary tensors. A clone made while still inside the outer block does not.
    """
    with torch.inference_mode(False):
        yield


def _move_module(module, device) -> None:
    """Move a module and leave ordinary tensors, even inside inference mode."""
    with _real_weights():
        module.to(device)
        _release_inference_tensors_now(module)


def _release_inference_tensors(module) -> None:
    """Turn inference-mode weights back into ordinary tensors before hooks detach.

    Denoising runs under ``torch.inference_mode``. Sequential offload then
    keeps those tensors in its weight map. Removing the hook later builds a
    parameter with ``requires_grad=True``, which inference tensors reject.
    """
    with _real_weights():
        _release_inference_tensors_now(module)


def _release_inference_tensors_now(module) -> None:
    for submodule in module.modules():
        for name, param in list(submodule.named_parameters(recurse=False)):
            if param.is_inference():
                submodule._parameters[name] = torch.nn.Parameter(
                    param.detach().clone(), requires_grad=False)
            elif param.requires_grad:
                param.requires_grad_(False)
        hook = getattr(submodule, '_hf_hook', None)
        weights = getattr(hook, 'weights_map', None) if hook is not None else None
        if weights is None:
            continue
        # PrefixedDataset.__iter__ yields the full name, but __getitem__ adds
        # the module prefix again. Look up the parameter's own name.
        for key, _param in submodule.named_parameters(recurse=False):
            _replace_inference_weight(weights, key)
        for key, _buffer in submodule.named_buffers(recurse=False):
            _replace_inference_weight(weights, key)


def _replace_inference_weight(weights, key: str) -> None:
    try:
        value = weights[key]
    except Exception:
        return
    if not torch.is_tensor(value) or not value.is_inference():
        return
    cloned = value.detach().clone()
    try:
        weights[key] = cloned
        return
    except Exception:
        pass
    dataset = getattr(weights, 'dataset', None)
    prefix = getattr(weights, 'prefix', None)
    if dataset is None or prefix is None:
        return
    full = f'{prefix}{key}'
    state = getattr(dataset, 'state_dict', None)
    if isinstance(state, dict):
        state[full] = cloned
        return
    if isinstance(dataset, dict):
        dataset[full] = cloned


def _count_gpu_layers(layers, target: torch.device) -> int:
    gpu_layers = 0
    for _, layer in layers:
        try:
            gpu_layers += next(layer.parameters()).device.type == target.type
        except StopIteration:
            continue
    return gpu_layers


@contextlib.contextmanager
def _fast_encoder(encoder, device, offload: bool, resident: bool = True,
                  headroom_bytes: int | None = None,
                  label: str = 'Pipeline prompt enhance',
                  debug: bool = False):
    """Keep as many whole layers as fit on the GPU for the rewrite.

    The rest of the encoder stays in CPU memory and is moved to the GPU for
    each forward, then returned. Sequential offload is put back before
    denoising, so the image model still loads one layer at a time. A dedicated
    enhancer that was only parked on CPU uses the same split and is returned
    to CPU afterward.
    """
    if not resident:
        yield
        return
    target = torch.device(device) if device is not None else torch.device('cpu')
    original_to = getattr(encoder, '_DGENERATE_ORIGINAL_TO_DISABLED', None)
    if target.type == 'cpu' or _putil.is_loaded_in_8bit_bnb(encoder) or not offload:
        with _on_device(encoder, device):
            yield
        return
    from accelerate.hooks import remove_hook_from_module
    hooked = original_to is not None
    move = original_to if hooked else encoder.to
    if hooked:
        with _real_weights():
            _release_inference_tensors_now(encoder)
            remove_hook_from_module(encoder, recurse=True)
            original_to('cpu')
            _release_inference_tensors_now(encoder)

    def _park():
        with _real_weights():
            _release_inference_tensors_now(encoder)
            remove_hook_from_module(encoder, recurse=True)
            move('cpu')
            _release_inference_tensors_now(encoder)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hooked:
            _restore_encoder_offload(encoder, target)

    budget = _gpu_budget_bytes(target, headroom_bytes)
    if _module_nbytes(encoder) <= budget:
        try:
            with _real_weights():
                move(target)
                _release_inference_tensors_now(encoder)
        except torch.cuda.OutOfMemoryError:
            _park()
            _placement_log(
                f'{label}: the model does not fit on the GPU, using offload.',
                debug=debug)
            yield
            return
        _, layers, _, _ = _placement_units(encoder)
        if layers:
            _log_layer_split(encoder, target, layers, label, debug=debug)
        else:
            _placement_log(
                f'{label}: the model is on {target} for this rewrite.',
                debug=debug)
        try:
            yield
        finally:
            _park()
        return
    embeddings, layers, norms, heads = _placement_units(encoder)
    if not layers:
        _park()
        _placement_log(
            f'{label}: the model does not fit on the GPU, using offload.',
            debug=debug)
        yield
        return
    names = _modules_for_gpu(encoder, budget)
    if not names:
        _park()
        _placement_log(
            f'{label}: the model does not fit on the GPU, using offload.',
            debug=debug)
        yield
        return
    by_name = dict(embeddings + layers + norms + heads)
    try:
        with _real_weights():
            for name in names:
                by_name[name].to(target)
            # Language RoPE follows the text embedding. Vision RoPE stays with
            # the vision tower: generate() parks position ids on the tower's
            # device, which is CPU when the tower does not fit.
            _move_language_rotary(encoder, embeddings)
            _release_inference_tensors_now(encoder)
    except torch.cuda.OutOfMemoryError:
        _park()
        _placement_log(
            f'{label}: the model does not fit on the GPU, using offload.',
            debug=debug)
        yield
        return
    if _count_gpu_layers(layers, target) == 0:
        _park()
        _placement_log(
            f'{label}: the model does not fit on the GPU, using offload.',
            debug=debug)
        yield
        return
    _hook_resident_layers(encoder, embeddings + layers + norms + heads, target)
    # generate() builds position ids on model.device. For a vision encoder that
    # is the CPU tower, while language RoPE now lives with the text embedding.
    _hook_rotary_inputs(encoder)
    _log_layer_split(encoder, target, layers, label, debug=debug)
    try:
        yield
    finally:
        _park()


def _restore_encoder_offload(encoder, device):
    import accelerate
    # The caller may still be inside inference mode. Offload copies made
    # there would be what the cached encoder keeps.
    with _real_weights():
        accelerate.cpu_offload(
            encoder, device,
            offload_buffers=len(getattr(encoder, '_parameters', {})) > 0,
            preload_module_classes=_pipelines._sequential_offload_preload_classes(encoder))
        _pipelines._disable_to(encoder)
        _pipelines._set_sequential_cpu_offload_flag(encoder, True)


@contextlib.contextmanager
def _on_device(module, device):
    if module is None or _putil.is_loaded_in_8bit_bnb(module):
        yield
        return
    try:
        previous = next(module.parameters()).device
    except StopIteration:
        yield
        return
    target = torch.device(device)
    moved = previous != target
    if moved:
        _move_module(module, target)
    try:
        yield
    finally:
        if moved:
            _move_module(module, previous)


class _TokenCounter:
    """Redraw ``Pipeline prompt enhance: Encoder, N out of max MAX tokens`` on one line."""

    def __init__(self, encoder_name: str, maximum: int):
        self.encoder_name = encoder_name
        self.maximum = maximum
        self.count = 0
        self._open = False

    def show(self):
        self._write()

    def add(self, n: int = 1):
        self.count += n
        self._write()

    def finish(self):
        if not self._open:
            return
        self._open = False
        if not (_messages.LEVEL & _messages.INFO):
            return
        file = _messages.get_message_file()
        file.write('\n')
        file.flush()

    def _write(self):
        if not (_messages.LEVEL & _messages.INFO):
            return
        file = _messages.get_message_file()
        file.write(
            f'\rPipeline prompt enhance: {self.encoder_name}, {self.count} out of max {self.maximum} tokens')
        self._open = True
        file.flush()


class _CountStreamer:
    def __init__(self, counter: _TokenCounter):
        self._counter = counter
        self._skip_prompt = True

    def put(self, value):
        if self._skip_prompt:
            self._skip_prompt = False
            return
        if hasattr(value, 'ndim') and value.ndim > 1:
            value = value[0]
        count = int(value.shape[-1]) if hasattr(value, 'shape') else 1
        self._counter.add(count)

    def end(self):
        self._counter.finish()


def _eos_ids(config) -> set[int]:
    eos = getattr(config, 'eos_token_id', None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(token) for token in eos}
    return {int(eos)}


def _sample(logits: torch.Tensor, generator: torch.Generator, sampling: Sampling,
           seen: collections.abc.Iterable[int] | None = None,
           blocked: collections.abc.Iterable[int] | None = None) -> int:
    scores = logits.detach().float().reshape(-1).cpu()
    scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
    if seen and sampling.presence_penalty:
        for token_id in set(seen):
            if 0 <= token_id < scores.numel():
                scores[token_id] -= sampling.presence_penalty
    if blocked:
        for token_id in blocked:
            if 0 <= token_id < scores.numel():
                scores[token_id] = -1e4
    if sampling.temperature <= 0:
        return int(torch.argmax(scores))
    scores = (scores / sampling.temperature).clamp(-1e4, 1e4)
    if sampling.top_k and 0 < sampling.top_k < scores.numel():
        values, indices = torch.topk(scores, sampling.top_k)
        filtered = torch.full_like(scores, float('-inf'))
        filtered.scatter_(0, indices, values)
        scores = filtered
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0).clamp(min=0)
    if sampling.top_p < 1:
        ordered, order = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(ordered, dim=-1)
        remove = cumulative > sampling.top_p
        remove[0] = False
        ordered = ordered.masked_fill(remove, 0).clamp(min=0)
        total = ordered.sum()
        if not torch.isfinite(total) or float(total) <= 0:
            return int(torch.argmax(scores))
        ordered = ordered / total
        pick = int(torch.multinomial(ordered, 1, generator=generator))
        return int(order[pick])
    total = probs.sum()
    if not torch.isfinite(total) or float(total) <= 0:
        return int(torch.argmax(scores))
    return int(torch.multinomial(probs / total, 1, generator=generator))


def _resident_weight(embed):
    """Embedding matrix, including when sequential offload has left the parameter on ``meta``."""
    weight = embed.weight
    if weight.device.type != 'meta':
        return weight
    hook = getattr(embed, '_hf_hook', None)
    weights = getattr(hook, 'weights_map', None) if hook is not None else None
    if weights is not None and 'weight' in weights:
        return weights['weight']
    raise _pipelines.UnsupportedPipelineConfigError(
        '--pipeline-prompt-enhance cannot read the offloaded text-encoder embedding.')


def _on_real_device(tensor, like):
    # ``.to('meta')`` drops the data. Offload hooks move real CPU tensors themselves.
    if tensor is None or like.device.type == 'meta':
        return tensor
    return tensor.to(like.device)


def _tied_logits(last: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
    # The embedding stands in for a missing language-model head. bf16 matmul
    # overflows to inf, and multinomial then rejects the probabilities.
    last = last.detach().to(device=embed_weight.device, dtype=torch.float32)
    return torch.nn.functional.linear(last, embed_weight.detach().float())


def _generate_tied(model, tokenizer, embed, prompt, system_prompt, seed, counter, sampling,
                   max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS) -> str:
    embed_weight = _resident_weight(embed)
    encoded = _chat_inputs(tokenizer, prompt, system_prompt, None)
    input_ids = _on_real_device(encoded['input_ids'], embed_weight)
    attention_mask = _on_real_device(encoded.get('attention_mask'), embed_weight)
    produced = []
    eos = _eos_ids(getattr(model, 'config', None))
    generator = torch.Generator(device='cpu').manual_seed(int(seed or 0))
    model.eval()
    past = None
    use_cache = True
    step_ids = input_ids
    try:
        with torch.inference_mode():
            for step in range(max_new_tokens):
                kwargs = {
                    'input_ids': step_ids,
                    'attention_mask': attention_mask,
                    'use_cache': use_cache,
                }
                if past is not None:
                    kwargs['past_key_values'] = past
                try:
                    output = model(**kwargs)
                except TypeError:
                    use_cache = False
                    past = None
                    output = model(
                        input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                hidden = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output[0]
                if hidden.shape[0] == step_ids.shape[-1] and hidden.shape[0] != step_ids.shape[0]:
                    last = hidden[-1]
                else:
                    last = hidden[:, -1]
                logits = _tied_logits(last, embed_weight)
                token = _sample(
                    logits, generator, sampling, seen=produced,
                    blocked=eos if len(produced) < min_new_tokens else None)
                produced.append(token)
                counter.add()
                if token in eos:
                    break
                next_id = torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, next_id], dim=-1)
                if attention_mask is not None:
                    ones = torch.ones(
                        (attention_mask.shape[0], 1),
                        device=attention_mask.device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, ones], dim=-1)
                if use_cache:
                    past = getattr(output, 'past_key_values', None)
                    if past is None:
                        use_cache = False
                        step_ids = input_ids
                    else:
                        step_ids = next_id
                else:
                    step_ids = input_ids
    finally:
        counter.finish()
    return tokenizer.decode(produced, skip_special_tokens=True)


def _generate_with_head(pipe, encoder, tokenizer, prompt, system_prompt, images, seed,
                        local_files_only: bool, counter, sampling,
                        max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS,
                        model_type=None, encoder_name: str | None = None) -> str:
    processor = tokenizer
    user_prompt = prompt
    if images:
        processor = _vision_processor(pipe, encoder, tokenizer, local_files_only)
    encoded = _chat_inputs(processor, user_prompt, system_prompt, images)
    if images and 'pixel_values' not in encoded and 'image_grid_thw' not in encoded:
        _messages.log(
            'Pipeline prompt enhance could not attach the reference image. '
            'The rewrite is text-only.')
        if system_prompt == EDIT_SYSTEM_PROMPT:
            system_prompt = T2I_SYSTEM_PROMPT
        encoded = _chat_inputs(tokenizer, user_prompt, system_prompt, None)
    inputs = {
        key: value.to(_embedding_device(encoder))
        if torch.is_tensor(value) else value
        for key, value in encoded.items() if key in _GENERATE_KEYS}
    prompt_width = inputs['input_ids'].shape[-1]
    generate_args = _head_generate_kwargs(
        sampling, max_new_tokens, min_new_tokens, prompt_width)
    generate_args['streamer'] = _CountStreamer(counter)
    encoder.eval()
    try:
        with torch.inference_mode(), _forked_seed(seed):
            sequences = encoder.generate(**inputs, **generate_args)
    finally:
        counter.finish()
    new_tokens = sequences[0, prompt_width:]
    decoder = processor.tokenizer if hasattr(processor, 'tokenizer') else tokenizer
    return decoder.decode(new_tokens, skip_special_tokens=True)


def _vision_processor(pipe, encoder, tokenizer, local_files_only: bool):
    cached = getattr(pipe, '_dgenerate_enhance_processor', None)
    if cached is not None:
        return cached
    processor = getattr(pipe, 'processor', None)
    if processor is not None and getattr(processor, 'tokenizer', None) is not None:
        return processor
    name = getattr(encoder, 'name_or_path', None)
    if not name:
        config = getattr(encoder, 'config', None)
        name = getattr(config, '_name_or_path', None) if config is not None else None
    if not name:
        return tokenizer
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(name, local_files_only=local_files_only)
    except Exception as error:
        _messages.debug_log(
            f'Prompt enhancement is text-only; image processor was not loaded: {error}')
        return tokenizer
    pipe._dgenerate_enhance_processor = processor
    return processor


def _apply_chat_template(tokenizer, messages, **kwargs):
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _ltx_enhance_user_content(prompt: str, has_image: bool):
    if has_image:
        return [
            {'type': 'image'},
            {'type': 'text', 'text': f'User Raw Input Prompt: {prompt}.'},
        ]
    return f'user prompt: {prompt}'


def _chat_inputs(tokenizer, prompt: str, system_prompt: str, images, *, ltx_enhance: bool = False):
    if ltx_enhance:
        user = _ltx_enhance_user_content(prompt, bool(images))
    else:
        user = prompt
        if images:
            user = [{'type': 'image'} for _ in images]
            user.append({'type': 'text', 'text': prompt})
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user},
    ]
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise _pipelines.UnsupportedPipelineConfigError(
            '--pipeline-prompt-enhance needs a tokenizer with a chat template.')
    text = _apply_chat_template(
        tokenizer, messages, tokenize=False, add_generation_prompt=True)
    if images:
        try:
            encoded = tokenizer(text=text, images=list(images), return_tensors='pt')
        except TypeError:
            encoded = tokenizer(text, return_tensors='pt')
    else:
        try:
            encoded = _apply_chat_template(
                tokenizer, messages, tokenize=True, add_generation_prompt=True,
                return_tensors='pt', return_dict=True)
        except TypeError:
            encoded = tokenizer(text, return_tensors='pt')
    if not isinstance(encoded, dict) and hasattr(encoded, 'keys'):
        encoded = dict(encoded)
    if torch.is_tensor(encoded):
        encoded = {'input_ids': encoded}
    return encoded
