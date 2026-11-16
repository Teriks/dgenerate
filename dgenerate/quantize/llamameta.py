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

"""llama.cpp hyperparameters, tokenizer, and chat template for a ``--llama`` GGUF."""

import json
import os
import re

import dgenerate.messages as _messages


_BYTE_TOKEN = re.compile(r'<0x[0-9A-Fa-f]{2}>')
_MAX_VOCAB = 1_000_000


def tie_output(state: dict, config: dict | None) -> dict:
    """Copy ``token_embd.weight`` to ``output.weight`` when the checkpoint ties them.

    llama.cpp reads the output projection as its own tensor. A diffusers text
    encoder that ties the embeddings leaves ``lm_head`` out of the file.
    """
    text = _text_config(config)
    tied = bool(text.get('tie_word_embeddings') or (config or {}).get('tie_word_embeddings'))
    embedding = state.get('token_embd.weight')
    if not tied or embedding is None or 'output.weight' in state:
        return state
    copied = dict(state)
    copied['output.weight'] = embedding
    _messages.log('Tied token embeddings are also stored as output.weight.')
    return copied


def llama_sidecar(config: dict | None, source_dir: str | None, stem: str):
    """A function that writes metadata onto a GGUF writer, or ``None``.

    Hyperparameters come from ``config.json``. The tokenizer and chat template
    come from the component folder, or from a sibling ``tokenizer`` folder.
    A tokenizer that is not BPE is skipped. The chat template is still stored.
    """
    text = _text_config(config)
    directories = _tokenizer_directories(source_dir, stem)
    tokenizer_config = _read_json(_find(directories, 'tokenizer_config.json')) or {}
    template = _chat_template(directories, tokenizer_config)
    vocab_size = _first_int(text, 'vocab_size')
    vocabulary = _sentencepiece_vocabulary(directories, tokenizer_config, vocab_size)
    if vocabulary is None:
        vocabulary = _bpe_vocabulary(
            directories, tokenizer_config, _model_type(config, text), vocab_size)
    if not text and template is None and vocabulary is None:
        return None

    noted = []

    def apply(writer) -> None:
        _write_hparams(writer, text)
        _write_deepstack(writer, config)
        if vocabulary is not None:
            _write_vocabulary(writer, vocabulary, tokenizer_config)
        if template is not None:
            writer.add_chat_template(template)

    # Build the note from a dry description so a writer is not required to log.
    _describe(text, vocabulary, template, noted)
    if noted:
        _messages.log('llama.cpp metadata: ' + ', '.join(noted) + '.')
    return apply if noted else None


def _describe(text, vocabulary, template, noted) -> None:
    if _has_hparams(text):
        noted.append('hyperparameters')
    if vocabulary is not None:
        noted.append(f'{len(vocabulary[0])} tokens')
    if template is not None:
        noted.append('chat template')


def _has_hparams(text: dict) -> bool:
    if _first_int(
            text, 'num_hidden_layers', 'num_layers', 'hidden_size', 'd_model',
            'intermediate_size', 'd_ff', 'num_attention_heads', 'num_heads',
            'num_key_value_heads', 'head_dim', 'max_position_embeddings', 'vocab_size') is not None:
        return True
    if isinstance(text.get('rms_norm_eps'), (int, float)) or isinstance(text.get('rope_theta'), (int, float)):
        return True
    scaling = text.get('rope_scaling')
    sections = scaling.get('mrope_section') if isinstance(scaling, dict) else None
    return isinstance(sections, list) and bool(sections)


def _text_config(config: dict | None) -> dict:
    if not config:
        return {}
    text = config.get('text_config')
    if isinstance(text, dict):
        return text
    return config


def _model_type(config: dict | None, text: dict) -> str:
    if config and config.get('model_type'):
        return str(config['model_type'])
    return str(text.get('model_type') or '')


def _tokenizer_directories(source_dir: str | None, stem: str) -> list[str]:
    if not source_dir:
        return []
    found = [source_dir]
    name = stem.replace('\\', '/').rstrip('/').split('/')[-1]
    sibling = None
    if name == 'text_encoder':
        sibling = 'tokenizer'
    elif name.startswith('text_encoder_'):
        sibling = 'tokenizer_' + name[len('text_encoder_'):]
    if sibling:
        path = os.path.join(os.path.dirname(source_dir), sibling)
        if os.path.isdir(path):
            found.append(path)
    return found


def _find(directories: list[str], name: str) -> str | None:
    for directory in directories:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def _read_json(path: str | None):
    if not path:
        return None
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _chat_template(directories: list[str], tokenizer_config: dict):
    payload = _read_json(_find(directories, 'chat_template.json'))
    template = None
    if isinstance(payload, str):
        template = payload
    elif isinstance(payload, dict):
        template = payload.get('chat_template')
    if template is None:
        template = tokenizer_config.get('chat_template')
    if template is None:
        jinja = _find(directories, 'chat_template.jinja')
        if jinja:
            with open(jinja, encoding='utf-8') as handle:
                template = handle.read()
    if isinstance(template, str) and template.strip():
        return template
    if isinstance(template, list) and template:
        return template
    return None


def _sentencepiece_vocabulary(directories, tokenizer_config: dict, vocab_size: int | None):
    """SentencePiece vocab from ``tokenizer.model``. Gemma 3 ships one beside a BPE json."""
    path = _find(directories, 'tokenizer.model') or _find(directories, 'spiece.model')
    if not path:
        return None
    try:
        from sentencepiece import SentencePieceProcessor
    except ImportError:
        _messages.warning(
            'sentencepiece is not installed, so tokenizer.model was not written.')
        return None
    import gguf

    processor = SentencePieceProcessor()
    processor.LoadFromFile(path)
    size = vocab_size if isinstance(vocab_size, int) else processor.vocab_size()
    tokens = [f'[PAD{index}]' for index in range(size)]
    scores = [-10000.0] * size
    types = [int(gguf.TokenType.UNUSED)] * size
    for token_id in range(min(processor.vocab_size(), size)):
        tokens[token_id] = processor.IdToPiece(token_id)
        scores[token_id] = float(processor.GetScore(token_id))
        if processor.IsUnknown(token_id):
            kind = gguf.TokenType.UNKNOWN
        elif processor.IsControl(token_id):
            kind = gguf.TokenType.CONTROL
        elif processor.IsUnused(token_id):
            kind = gguf.TokenType.UNUSED
        elif processor.IsByte(token_id):
            kind = gguf.TokenType.BYTE
        else:
            kind = gguf.TokenType.NORMAL
        types[token_id] = int(kind)
    added = _read_json(_find(directories, 'added_tokens.json'))
    if isinstance(added, dict):
        for key, token_id in added.items():
            if isinstance(token_id, int) and 0 <= token_id < size:
                tokens[token_id] = str(key)
                scores[token_id] = -1000.0
                types[token_id] = int(gguf.TokenType.USER_DEFINED)
    decoder = tokenizer_config.get('added_tokens_decoder') or {}
    if isinstance(decoder, dict):
        for raw_id, token_data in decoder.items():
            if not isinstance(token_data, dict):
                continue
            try:
                token_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if token_id < 0 or token_id >= size:
                continue
            content = str(token_data.get('content') or '')
            if token_data.get('special'):
                types[token_id] = int(gguf.TokenType.CONTROL)
            else:
                content = content.replace('\u2581', ' ')
                types[token_id] = int(gguf.TokenType.USER_DEFINED)
            scores[token_id] = -1000.0
            tokens[token_id] = content
    return tokens, types, None, 'default', 'llama', scores


def _bpe_vocabulary(directories, tokenizer_config: dict, model_type: str, vocab_size: int | None):
    path = _find(directories, 'tokenizer.json')
    if path:
        payload = _read_json(path)
        model = payload.get('model') or {}
        vocab = model.get('vocab')
        if isinstance(vocab, list) and vocab and isinstance(vocab[0], (list, tuple)):
            tokens = [str(item[0]) for item in vocab]
            scores = [float(item[1]) if len(item) > 1 else 0.0 for item in vocab]
            if isinstance(vocab_size, int) and vocab_size > len(tokens):
                for index in range(len(tokens), vocab_size):
                    tokens.append(f'<unused{index}>')
                    scores.append(0.0)
            return tokens, None, None, None, 't5', scores
        if model.get('type') != 'BPE':
            kind = model.get('type') or 'unknown'
            _messages.warning(
                f'Tokenizer model {kind!r} is not BPE, so its vocabulary was not written. '
                'The chat template and hyperparameters are still stored when they are present.')
            return None
        vocab = vocab or {}
        merges = _merges(model.get('merges') or [])
        added = payload.get('added_tokens') or []
    else:
        vocab_path = _find(directories, 'vocab.json')
        merges_path = _find(directories, 'merges.txt')
        if not vocab_path or not merges_path:
            return None
        vocab = _read_json(vocab_path)
        merges = _merges_file(merges_path)
        added = []
    tokens, types = _token_table(vocab, added, vocab_size)
    if tokens is None:
        return None
    pre = _tokenizer_pre(str(tokenizer_config.get('tokenizer_class') or ''), model_type)
    return tokens, types, merges, pre


def _merges(raw) -> list[str]:
    merges = []
    for item in raw:
        if isinstance(item, str):
            merges.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            merges.append(f'{item[0]} {item[1]}')
    return merges


def _merges_file(path: str) -> list[str]:
    merges = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            merges.append(line)
    return merges


def _token_table(vocab: dict, added: list, vocab_size: int | None):
    import gguf

    by_id = {}
    special = set()
    defined = set()
    for token, index in vocab.items():
        if isinstance(index, int):
            by_id[index] = str(token)
    for item in added:
        if not isinstance(item, dict) or not isinstance(item.get('id'), int):
            continue
        index = item['id']
        by_id[index] = str(item.get('content') or '')
        if item.get('special'):
            special.add(index)
        else:
            defined.add(index)
    if not by_id:
        return None, None
    size = max(by_id) + 1
    if isinstance(vocab_size, int) and size > vocab_size:
        extra = [index for index in by_id if index >= vocab_size]
        if extra:
            _messages.warning(
                f'{len(extra)} tokenizer id'
                f'{"" if len(extra) == 1 else "s"} sit past vocab_size {vocab_size} and were left out.')
            by_id = {index: token for index, token in by_id.items() if index < vocab_size}
            special = {index for index in special if index < vocab_size}
            defined = {index for index in defined if index < vocab_size}
            size = vocab_size
    elif isinstance(vocab_size, int) and vocab_size > size:
        size = vocab_size
    if size > _MAX_VOCAB:
        _messages.warning(
            f'Tokenizer vocabulary ends at id {size - 1}, so it was not written.')
        return None, None
    tokens = []
    types = []
    for index in range(size):
        token = by_id.get(index)
        if token is None:
            tokens.append(f'<unused{index}>')
            types.append(int(gguf.TokenType.UNUSED))
        else:
            tokens.append(token)
            types.append(_token_type(token, index, special, defined))
    return tokens, types


def _token_type(token: str, index: int, special: set, defined: set) -> int:
    import gguf

    if index in special or _BYTE_TOKEN.fullmatch(token):
        if _BYTE_TOKEN.fullmatch(token):
            return int(gguf.TokenType.BYTE)
        return int(gguf.TokenType.CONTROL)
    if index in defined:
        return int(gguf.TokenType.USER_DEFINED)
    return int(gguf.TokenType.NORMAL)


def _tokenizer_pre(tokenizer_class: str, model_type: str) -> str | None:
    name = tokenizer_class.lower()
    model = model_type.lower()
    if 'qwen' in name or model.startswith('qwen'):
        return 'qwen2'
    if 'gemma' in name or model.startswith('gemma'):
        # This llama.cpp build has no 'gemma' pre-tokenizer. Gemma 3 uses SentencePiece.
        return None
    if 'llama' in name or model.startswith('llama') or model.startswith('mistral'):
        return 'llama-bpe'
    return None


def _write_hparams(writer, text: dict) -> None:
    layers = _first_int(text, 'num_hidden_layers', 'num_layers')
    hidden = _first_int(text, 'hidden_size', 'd_model')
    forward = _first_int(text, 'intermediate_size', 'd_ff')
    heads = _first_int(text, 'num_attention_heads', 'num_heads')
    kv = _first_int(text, 'num_key_value_heads') or heads
    context = _first_int(text, 'max_position_embeddings', 'n_positions')
    vocab = _first_int(text, 'vocab_size')
    if layers is not None:
        writer.add_block_count(layers)
    if hidden is not None:
        writer.add_embedding_length(hidden)
    gemma4 = writer.arch == 'gemma4'
    if forward is not None and not gemma4:
        writer.add_feed_forward_length(forward)
    if heads is not None:
        writer.add_head_count(heads)
    if kv is not None:
        writer.add_head_count_kv(kv)
    head_dim = _first_int(text, 'head_dim')
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads
    if head_dim and not gemma4:
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)
    rms = text.get('rms_norm_eps', text.get('layer_norm_epsilon'))
    if isinstance(rms, (int, float)):
        writer.add_layer_norm_rms_eps(float(rms))
    theta = text.get('rope_theta')
    if not isinstance(theta, (int, float)):
        rope = text.get('rope_parameters')
        if isinstance(rope, dict):
            theta = rope.get('rope_theta')
    if isinstance(theta, (int, float)) and not gemma4:
        writer.add_rope_freq_base(float(theta))
    if context is not None:
        writer.add_context_length(context)
    elif writer.arch in {'t5', 't5encoder'}:
        # UMT5 configs omit n_positions. The transformers default is 512.
        writer.add_context_length(512)
    epsilon = text.get('layer_norm_epsilon')
    if isinstance(epsilon, (int, float)):
        writer.add_float32(f'{writer.arch}.attention.layer_norm_epsilon', float(epsilon))
    buckets = text.get('relative_attention_num_buckets')
    if isinstance(buckets, int):
        writer.add_uint32(f'{writer.arch}.attention.relative_buckets_count', buckets)
    decoder_start = text.get('decoder_start_token_id')
    if isinstance(decoder_start, int):
        writer.add_uint32(f'{writer.arch}.decoder_start_token_id', decoder_start)
    if gemma4:
        _write_gemma4(writer, text, head_dim, layers, forward)
    elif writer.arch == 'gemma3':
        _write_gemma3(writer, text)
    else:
        per_layer = text.get('hidden_size_per_layer_input')
        if isinstance(per_layer, int):
            writer.add_uint32(f'{writer.arch}.embedding_length_per_layer_input', per_layer)
    if vocab is not None:
        writer.add_vocab_size(vocab)
    _write_mrope(writer, text)
    if writer.arch in {'qwen35', 'qwen35moe'}:
        _write_qwen35(writer, text)


def _mrope_sections(text: dict) -> list | None:
    """``mrope_section`` from ``rope_parameters`` or the older ``rope_scaling``."""
    for key in ('rope_parameters', 'rope_scaling'):
        rope = text.get(key)
        sections = rope.get('mrope_section') if isinstance(rope, dict) else None
        if isinstance(sections, list) and sections and all(isinstance(item, int) for item in sections):
            return list(sections)
    return None


def _write_mrope(writer, text: dict) -> None:
    sections = _mrope_sections(text)
    if writer.arch in {'qwen35', 'qwen35moe'} and sections is None:
        # llama.cpp treats this key as required and uses this default when the
        # checkpoint leaves the field out.
        sections = [11, 11, 10, 0]
    if not sections:
        return
    import gguf
    # llama.cpp reads four mrope sections. Qwen configs store three and leave the last at 0.
    if writer.arch in {'qwen2vl', 'qwen3vl', 'qwen3vlmoe', 'qwen35', 'qwen35moe'} and len(sections) < 4:
        sections = list(sections) + [0] * (4 - len(sections))
    writer.add_key_value(
        gguf.Keys.Rope.DIMENSION_SECTIONS.format(arch=writer.arch),
        sections,
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.INT32)


def _write_qwen35(writer, text: dict) -> None:
    """Gated-delta metadata llama.cpp's qwen35 loader requires."""
    kernel = text.get('linear_conv_kernel_dim')
    state = text.get('linear_key_head_dim')
    groups = text.get('linear_num_key_heads')
    rank = text.get('linear_num_value_heads')
    value_dim = text.get('linear_value_head_dim')
    if all(isinstance(item, int) for item in (kernel, state, groups, rank, value_dim)):
        writer.add_uint32(f'{writer.arch}.ssm.conv_kernel', int(kernel))
        writer.add_uint32(f'{writer.arch}.ssm.state_size', int(state))
        writer.add_uint32(f'{writer.arch}.ssm.group_count', int(groups))
        writer.add_uint32(f'{writer.arch}.ssm.time_step_rank', int(rank))
        writer.add_uint32(f'{writer.arch}.ssm.inner_size', int(value_dim) * int(rank))
    layer_types = text.get('layer_types')
    if isinstance(layer_types, list) and layer_types:
        writer.add_array(
            f'{writer.arch}.attention.recurrent_layers',
            [item == 'linear_attention' for item in layer_types])
    interval = text.get('full_attention_interval', 4)
    if isinstance(interval, int):
        writer.add_uint32(f'{writer.arch}.full_attention_interval', interval)


def gemma4_mmproj_metadata(config: dict | None, tensors: dict | None = None):
    """CLIP metadata llama.cpp reads for a Gemma 4 vision and audio mmproj, or ``None``."""
    if not isinstance(config, dict):
        return None
    kind = str(config.get('model_type') or '').lower().replace('-', '_')
    if not (kind.startswith('gemma4') or kind.startswith('gemma_4')):
        return None
    vision = config.get('vision_config') if isinstance(config.get('vision_config'), dict) else None
    audio = config.get('audio_config') if isinstance(config.get('audio_config'), dict) else None
    if not isinstance(vision, dict) or 'hidden_size' not in vision:
        return None
    names = tensors or {}
    unified = any('patch_norm' in key or key.startswith('v.patch_norm') for key in names)
    text = _text_config(config)

    def apply(writer) -> None:
        writer.add_bool('clip.has_vision_encoder', True)
        writer.add_string(
            'clip.vision.projector_type', 'gemma4uv' if unified else 'gemma4v')
        writer.add_uint32('clip.vision.embedding_length', int(vision['hidden_size']))
        forward = vision.get('intermediate_size')
        if isinstance(forward, int):
            writer.add_uint32('clip.vision.feed_forward_length', forward)
        layers = vision.get('num_hidden_layers')
        if isinstance(layers, int):
            writer.add_uint32('clip.vision.block_count', layers)
        heads = vision.get('num_attention_heads')
        if isinstance(heads, int):
            writer.add_uint32('clip.vision.attention.head_count', heads)
        head_dim = vision.get('head_dim')
        if isinstance(head_dim, int):
            writer.add_uint32('clip.vision.attention.head_dim', head_dim)
        projection = _first_int(text, 'hidden_size', 'd_model')
        if isinstance(projection, int):
            writer.add_uint32('clip.vision.projection_dim', projection)
        eps = vision.get('rms_norm_eps', vision.get('layer_norm_eps', 1e-6))
        if isinstance(eps, (int, float)):
            writer.add_float32('clip.vision.attention.layer_norm_epsilon', float(eps))
        # llama.cpp requires an image size and does not read it for Gemma 4.
        writer.add_uint32('clip.vision.image_size', 224)
        patch = vision.get('patch_size', 16)
        if isinstance(patch, int):
            writer.add_uint32('clip.vision.patch_size', patch)
        pooling = vision.get('pooling_kernel_size')
        if isinstance(pooling, int):
            writer.add_uint32('clip.vision.projector.scale_factor', pooling)
        import gguf
        # The loader rejects an mmproj whose mean and std arrays are empty.
        # Gemma 4's processor leaves normalization off, so the values are 0 and 1.
        writer.add_key_value(
            'clip.vision.image_mean', [0.0, 0.0, 0.0],
            gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32)
        writer.add_key_value(
            'clip.vision.image_std', [1.0, 1.0, 1.0],
            gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32)
        if isinstance(audio, dict) and isinstance(audio.get('hidden_size'), int):
            writer.add_bool('clip.has_audio_encoder', True)
            writer.add_string(
                'clip.audio.projector_type', 'gemma4ua' if unified else 'gemma4a')
            hidden = int(audio['hidden_size'])
            writer.add_uint32('clip.audio.embedding_length', hidden)
            writer.add_uint32('clip.audio.feed_forward_length', hidden * 4)
            audio_layers = audio.get('num_hidden_layers')
            if isinstance(audio_layers, int):
                writer.add_uint32('clip.audio.block_count', audio_layers)
            audio_heads = audio.get('num_attention_heads')
            if isinstance(audio_heads, int):
                writer.add_uint32('clip.audio.attention.head_count', audio_heads)
            audio_projection = audio.get('output_proj_dims', projection)
            if isinstance(audio_projection, int):
                writer.add_uint32('clip.audio.projection_dim', audio_projection)
            mel = audio.get('input_feat_size', 128)
            if isinstance(mel, int):
                writer.add_uint32('clip.audio.num_mel_bins', mel)
            audio_eps = audio.get('rms_norm_eps', audio.get('layer_norm_eps', 1e-6))
            if isinstance(audio_eps, (int, float)):
                writer.add_float32('clip.audio.attention.layer_norm_epsilon', float(audio_eps))
        _messages.log(
            f'gemma4{"uv" if unified else "v"} mmproj metadata for llama.cpp.')

    return apply


def _vision_norm(
        source_dir: str | None,
        mean_default: list[float] | None = None,
        std_default: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    mean_default = list(mean_default or [0.5, 0.5, 0.5])
    std_default = list(std_default or mean_default)
    path = os.path.join(source_dir, 'preprocessor_config.json') if source_dir else ''
    data = _read_json(path) if path and os.path.isfile(path) else None
    if not isinstance(data, dict):
        return mean_default, std_default
    mean = data.get('image_mean')
    std = data.get('image_std')
    if not _float_list(mean) or not _float_list(std) or len(mean) < 3 or len(std) < 3:
        return mean_default, std_default
    return [float(item) for item in mean], [float(item) for item in std]


def _float_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, (int, float)) for item in value)


def qwen3vl_mmproj_metadata(config: dict | None, source_dir: str | None = None):
    """clip metadata llama.cpp reads for a Qwen2.5-VL or Qwen3-VL mmproj, or ``None``."""
    vision = (config or {}).get('vision_config') if isinstance(config, dict) else None
    kind = _model_type(config, _text_config(config)).lower().replace('-', '_')
    qwen3 = kind.startswith('qwen3_vl') or kind.startswith('qwen3_5') or kind.startswith('qwen35')
    qwen2 = kind.startswith('qwen2')
    if not isinstance(vision, dict) or not vision.get('depth') or not (qwen3 or qwen2):
        return None
    projector = 'qwen3vl_merger' if qwen3 else (
        'qwen2.5vl_merger' if 'qwen2_5' in kind or 'qwen25' in kind else 'qwen2vl_merger')

    def apply(writer) -> None:
        writer.add_string('clip.projector_type', projector)
        writer.add_bool('clip.has_vision_encoder', True)
        if qwen3:
            writer.add_bool('clip.use_gelu', True)
        else:
            # The tower is SwiGLU. The merger GELU is selected by the projector graph.
            writer.add_bool('clip.use_silu', True)
        patch = int(vision.get('patch_size') or 16)
        positions = int(vision.get('num_position_embeddings') or 0)
        image_size = int(positions ** 0.5 * patch) if positions else (0 if qwen3 else 560)
        if image_size:
            writer.add_uint32('clip.vision.image_size', image_size)
        writer.add_uint32('clip.vision.patch_size', patch)
        writer.add_uint32('clip.vision.embedding_length', int(vision['hidden_size']))
        writer.add_uint32('clip.vision.feed_forward_length', int(vision['intermediate_size']))
        text = _text_config(config)
        projection = vision.get('out_hidden_size')
        if not isinstance(projection, int):
            projection = _first_int(text, 'hidden_size', 'd_model')
        if isinstance(projection, int):
            writer.add_uint32('clip.vision.projection_dim', projection)
        depth = int(vision['depth'])
        writer.add_uint32('clip.vision.block_count', depth)
        writer.add_uint32('clip.vision.attention.head_count', int(vision['num_heads']))
        merge = vision.get('spatial_merge_size')
        if isinstance(merge, int):
            writer.add_uint32('clip.vision.spatial_merge_size', merge)
        if qwen3:
            indexes = {int(item) for item in vision.get('deepstack_visual_indexes') or []}
            writer.add_array('clip.vision.is_deepstack_layers', [index in indexes for index in range(depth)])
        full = vision.get('fullatt_block_indexes') or []
        if projector == 'qwen2.5vl_merger':
            step = int(full[1]) - int(full[0]) if len(full) >= 2 else 8
            writer.add_uint32('clip.vision.n_wa_pattern', step)
        text = _text_config(config)
        rms = text.get('rms_norm_eps', vision.get('layer_norm_eps', 1e-6))
        if isinstance(rms, (int, float)):
            writer.add_float32('clip.vision.attention.layer_norm_epsilon', float(rms))
        if qwen3:
            mean, std = _vision_norm(source_dir)
        else:
            mean, std = _vision_norm(
                source_dir,
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711])
        import gguf
        writer.add_key_value(
            'clip.vision.image_mean', mean, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32)
        writer.add_key_value(
            'clip.vision.image_std', std, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32)
        _messages.log(f'{projector} mmproj metadata for llama.cpp.')

    return apply


def _write_gemma3(writer, text: dict) -> None:
    """Sliding-window metadata from llama.cpp's Gemma3Model.set_gguf_parameters."""
    window = text.get('sliding_window')
    pattern = text.get('sliding_window_pattern', text.get('_sliding_window_pattern'))
    layer_types = text.get('layer_types')
    if isinstance(window, int) and window > 0 and pattern != 1:
        writer.add_sliding_window(int(window))
        if isinstance(layer_types, list) and layer_types:
            writer.add_sliding_window_pattern([item == 'sliding_attention' for item in layer_types])
        elif isinstance(pattern, int):
            writer.add_sliding_window_pattern(int(pattern))
    local = text.get('rope_local_base_freq')
    if isinstance(local, (int, float)):
        writer.add_rope_freq_base_swa(float(local))
    softcap = text.get('final_logit_softcapping')
    if isinstance(softcap, (int, float)) and softcap:
        writer.add_final_logit_softcapping(float(softcap))


def _write_gemma4(writer, text: dict, head_dim: int | None, layers: int | None, forward: int | None) -> None:
    """Metadata from llama.cpp's Gemma4Model.set_gguf_parameters."""
    head_dim = head_dim or 256
    global_head = _first_int(text, 'global_head_dim') or head_dim
    writer.add_key_length(global_head)
    writer.add_value_length(global_head)
    writer.add_key_length_swa(head_dim)
    writer.add_value_length_swa(head_dim)
    shared = text.get('num_kv_shared_layers')
    if isinstance(shared, int):
        writer.add_shared_kv_layers(shared)
    window = text.get('sliding_window')
    if isinstance(window, int):
        writer.add_sliding_window(int(window))
    layer_types = text.get('layer_types')
    if isinstance(layer_types, list) and layer_types:
        writer.add_sliding_window_pattern([item == 'sliding_attention' for item in layer_types])
    per_layer = text.get('hidden_size_per_layer_input')
    writer.add_embedding_length_per_layer_input(int(per_layer) if isinstance(per_layer, int) else 0)
    if isinstance(forward, int):
        if text.get('use_double_wide_mlp') and isinstance(shared, int) and layers:
            first = layers - shared
            writer.add_feed_forward_length([
                forward if index < first else forward * 2 for index in range(layers)])
        else:
            writer.add_feed_forward_length(forward)
    rope = text.get('rope_parameters') if isinstance(text.get('rope_parameters'), dict) else {}
    full = rope.get('full_attention') if isinstance(rope.get('full_attention'), dict) else rope
    sliding = rope.get('sliding_attention') if isinstance(rope.get('sliding_attention'), dict) else {}
    theta = full.get('rope_theta', 1_000_000.0)
    if isinstance(theta, (int, float)):
        writer.add_rope_freq_base(float(theta))
    sliding_theta = sliding.get('rope_theta')
    if isinstance(sliding_theta, (int, float)):
        writer.add_rope_freq_base_swa(float(sliding_theta))
    writer.add_rope_dimension_count(int(global_head))
    partial = rope.get('partial_rotary_factor', 1.0)
    if not isinstance(partial, (int, float)):
        partial = 1.0
    writer.add_rope_dimension_count_swa(int(head_dim * float(partial)))
    softcap = text.get('final_logit_softcapping')
    if isinstance(softcap, (int, float)) and softcap:
        writer.add_final_logit_softcapping(float(softcap))


def _write_deepstack(writer, config: dict | None) -> None:
    if writer.arch not in {'qwen3vl', 'qwen3vlmoe'}:
        return
    vision = (config or {}).get('vision_config') if isinstance(config, dict) else None
    indexes = (vision or {}).get('deepstack_visual_indexes') if isinstance(vision, dict) else None
    if isinstance(indexes, list):
        writer.add_uint32(f'{writer.arch}.n_deepstack_layers', len(indexes))


def _write_vocabulary(writer, vocabulary, tokenizer_config: dict) -> None:
    scores = None
    model_name = 'gpt2'
    if len(vocabulary) > 4:
        tokens, types, merges, pre, model_name, scores = vocabulary
    else:
        tokens, types, merges, pre = vocabulary
    if writer.arch == 'gemma4':
        model_name = 'gemma4'
    writer.add_tokenizer_model(model_name)
    if pre:
        writer.add_tokenizer_pre(pre)
    writer.add_token_list(tokens)
    if types:
        writer.add_token_types(types)
    if scores:
        writer.add_token_scores(scores)
    if merges:
        writer.add_token_merges(merges)
    _write_special(writer, tokens, tokenizer_config)
    flag = tokenizer_config.get('add_bos_token')
    tokenizer_class = str(tokenizer_config.get('tokenizer_class') or '')
    # LlamaTokenizer adds BOS when the config omits the flag. Qwen states the flag.
    if not isinstance(flag, bool) and 'llama' in tokenizer_class.lower():
        flag = True
    if writer.arch == 'gemma4':
        flag = True
    if isinstance(flag, bool):
        writer.add_add_bos_token(flag)
    if writer.arch in {'gemma3', 'gemma4'}:
        writer.add_add_space_prefix(False)


def _write_special(writer, tokens: list[str], tokenizer_config: dict) -> None:
    by_text = {}
    for index, token in enumerate(tokens):
        by_text.setdefault(token, index)
    eos = _token_content(tokenizer_config.get('eos_token'))
    bos = _token_content(tokenizer_config.get('bos_token'))
    pad = _token_content(tokenizer_config.get('pad_token'))
    unk = _token_content(tokenizer_config.get('unk_token'))
    if bos in by_text:
        writer.add_bos_token_id(by_text[bos])
    if eos in by_text:
        writer.add_eos_token_id(by_text[eos])
        writer.add_eot_token_id(by_text[eos])
    if pad in by_text:
        writer.add_pad_token_id(by_text[pad])
    if unk in by_text:
        writer.add_unk_token_id(by_text[unk])


def _token_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get('content')
    return None


def _first_int(source: dict, *names: str) -> int | None:
    for name in names:
        value = source.get(name)
        if isinstance(value, int):
            return value
    return None
