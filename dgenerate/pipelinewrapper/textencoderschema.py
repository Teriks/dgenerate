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
Text encoder schema for each ``--model-type``.

The console schema builder writes this to ``textencoders.json``. Encoder class
names come from the pipeline ``__init__`` annotations. ``mode`` is not part of
that signature: it is only for a monolithic CLIP or T5 file, and only the modes
that match that slot are listed. Leaving ``mode`` unset is the usual case
(a diffusers folder, a GGUF, or a pre-quantized file).
"""

import types
import typing


def _class_names(annotation) -> list[str]:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        names: list[str] = []
        for arg in typing.get_args(annotation):
            names.extend(_class_names(arg))
        return names
    name = getattr(annotation, '__name__', '')
    if name in ('', 'NoneType', 'PreTrainedModel'):
        return []
    return [name]


def _is_text_encoder_slot(name: str) -> bool:
    return name == 'text_encoder' or (
        name.startswith('text_encoder_') and name[len('text_encoder_'):].isdigit())


def _copy_slots(model_types: tuple[str, ...], slots: dict[str, typing.Any]) -> dict:
    copied = {}
    for model_type in model_types:
        for slot, value in slots.items():
            copied[(model_type, slot)] = value
    return copied


# Pipeline annotations name one class. These are accepted stand-ins, or the
# classes a generic ``PreTrainedModel`` slot actually loads.
_ENCODER_REPLACEMENTS = {
    **_copy_slots(
        ('flux', 'flux-fill', 'flux-kontext'),
        {'text_encoder_2': ['T5EncoderModel', 'DistillT5EncoderModel']}),
    **_copy_slots(
        ('z-image', 'z-image-omni'),
        {'text_encoder': ['Qwen3Model', 'Qwen3ForCausalLM']}),
    ('flux2', 'text_encoder'): [
        'Mistral3ForConditionalGeneration', 'Qwen3ForCausalLM'],
    ('wan-animate-2', 'text_encoder'): ['UMT5EncoderModel'],
}

_LABELS = {
    **_copy_slots(('sd', 'pix2pix', 'upscaler-x2', 'upscaler-x4'), {'text_encoder': 'CLIP'}),
    **_copy_slots(('sdxl', 'sdxl-pix2pix'), {
        'text_encoder': 'CLIP-L',
        'text_encoder_2': 'CLIP-G',
    }),
    ('kolors', 'text_encoder'): 'ChatGLM',
    **_copy_slots(('s-cascade', 's-cascade-decoder'), {'text_encoder': 'CLIP'}),
    **_copy_slots(('sd3', 'sd3-pix2pix'), {
        'text_encoder': 'CLIP-L',
        'text_encoder_2': 'CLIP-G',
        'text_encoder_3': 'T5-XXL',
    }),
    **_copy_slots(('flux', 'flux-fill', 'flux-kontext'), {
        'text_encoder': 'CLIP-L',
        'text_encoder_2': 'T5-XXL',
    }),
    ('flux2', 'text_encoder'): 'Mistral 3 or Qwen3',
    ('flux2-klein-kv', 'text_encoder'): 'Qwen3',
    **_copy_slots(('z-image', 'z-image-omni'), {'text_encoder': 'Qwen3'}),
    **_copy_slots(
        ('qwen-image', 'qwen-image-edit', 'qwen-image-layered'),
        {'text_encoder': 'Qwen2.5-VL'}),
    ('qwen-image-21', 'text_encoder'): 'Qwen3-VL',
    ('krea2', 'text_encoder'): 'Qwen3-VL',
    **_copy_slots(('kandinsky5-image', 'kandinsky5-video'), {
        'text_encoder': 'Qwen2.5-VL',
        'text_encoder_2': 'CLIP',
    }),
    ('hydit', 'text_encoder'): 'BERT',
    ('hydit', 'text_encoder_2'): 'T5',
    ('hyimage-21', 'text_encoder'): 'Qwen2.5-VL',
    ('hyimage-21', 'text_encoder_2'): 'T5',
    ('hyimage-21-refiner', 'text_encoder'): 'Qwen2.5-VL',
    ('ltx2', 'text_encoder'): 'Gemma',
    ('ltx-video', 'text_encoder'): 'T5',
    **_copy_slots(('wan', 'wan-animate', 'wan-animate-2'), {'text_encoder': 'UMT5'}),
    ('hyvideo-15', 'text_encoder'): 'Qwen2.5-VL',
    ('hyvideo-15', 'text_encoder_2'): 'T5',
    ('motif-video', 'text_encoder'): 'T5Gemma',
    **_copy_slots(('if', 'ifs', 'ifs-img2img'), {'text_encoder': 'T5'}),
}

# Slots the pipeline still runs when the encoder is not loaded. SD3 writes
# zeros for a missing T5. The other slots are required; a type hint of
# ``None`` is not enough, because the encode path still calls the module.
_NULLABLE = {
    ('sd3', 'text_encoder_3'),
    ('sd3-pix2pix', 'text_encoder_3'),
}

# Optional monolithic single-file modes. Absent means the slot has no mode.
_MODES = {
    **_copy_slots(('flux', 'flux-fill', 'flux-kontext'), {
        'text_encoder': [
            {'value': 'clip-l', 'label': 'Monolithic Flux CLIP-L'},
        ],
        'text_encoder_2': [
            {'value': 't5-xxl', 'label': 'Monolithic Flux T5-XXL'},
        ],
    }),
    **_copy_slots(('sd3', 'sd3-pix2pix'), {
        'text_encoder': [
            {'value': 'clip-l-sd3', 'label': 'Monolithic SD3 / SD3.5 medium CLIP-L'},
            {'value': 'clip-l-sd35-large', 'label': 'Monolithic SD3.5 large CLIP-L'},
        ],
        'text_encoder_2': [
            {'value': 'clip-g-sd3', 'label': 'Monolithic SD3 / SD3.5 medium CLIP-G'},
            {'value': 'clip-g-sd35-large', 'label': 'Monolithic SD3.5 large CLIP-G'},
        ],
        'text_encoder_3': [
            {'value': 't5-xxl-sd3', 'label': 'Monolithic SD3 / SD3.5 T5-XXL'},
        ],
    }),
}


def _pipeline_classes() -> dict[str, list[type]]:
    import diffusers

    import dgenerate.extras.ultraedit as _ultraedit
    import dgenerate.pipelinewrapper.enums as _enums

    classes = {
        'sd': [diffusers.StableDiffusionPipeline],
        'pix2pix': [diffusers.StableDiffusionInstructPix2PixPipeline],
        'upscaler-x2': [diffusers.StableDiffusionLatentUpscalePipeline],
        'upscaler-x4': [diffusers.StableDiffusionUpscalePipeline],
        'if': [diffusers.IFPipeline],
        'ifs': [diffusers.IFSuperResolutionPipeline],
        'ifs-img2img': [diffusers.IFImg2ImgSuperResolutionPipeline],
        'sdxl': [diffusers.StableDiffusionXLPipeline],
        'sdxl-pix2pix': [diffusers.StableDiffusionXLInstructPix2PixPipeline],
        'kolors': [diffusers.KolorsPipeline],
        's-cascade': [diffusers.StableCascadePriorPipeline],
        's-cascade-decoder': [diffusers.StableCascadeDecoderPipeline],
        'sd3': [diffusers.StableDiffusion3Pipeline],
        'sd3-pix2pix': [_ultraedit.StableDiffusion3InstructPix2PixPipeline],
        'flux': [diffusers.FluxPipeline],
        'flux-fill': [diffusers.FluxFillPipeline],
        'flux-kontext': [diffusers.FluxKontextPipeline],
        'flux2': [diffusers.Flux2Pipeline, diffusers.Flux2KleinPipeline],
        'flux2-klein-kv': [diffusers.Flux2KleinKVPipeline],
        'z-image': [diffusers.ZImagePipeline],
        'z-image-omni': [diffusers.ZImageOmniPipeline],
        'qwen-image': [diffusers.QwenImagePipeline],
        'qwen-image-edit': [diffusers.QwenImageEditPipeline],
        'qwen-image-layered': [diffusers.QwenImageLayeredPipeline],
        'qwen-image-21': [diffusers.QwenImage21Pipeline],
        'krea2': [diffusers.Krea2Pipeline],
        'kandinsky5-image': [diffusers.Kandinsky5T2IPipeline],
        'hydit': [diffusers.HunyuanDiTPipeline],
        'hyimage-21': [diffusers.HunyuanImagePipeline],
        'hyimage-21-refiner': [diffusers.HunyuanImageRefinerPipeline],
        'ltx2': [diffusers.LTX2Pipeline],
        'ltx-video': [diffusers.LTXPipeline],
        'wan': [diffusers.WanPipeline],
        'wan-animate': [diffusers.WanAnimatePipeline],
        'wan-animate-2': [],
        'hyvideo-15': [diffusers.HunyuanVideo15Pipeline],
        'motif-video': [diffusers.MotifVideoPipeline],
        'kandinsky5-video': [diffusers.Kandinsky5T2VPipeline],
    }
    missing = [name for name in _enums.supported_model_type_strings() if name not in classes]
    missing += [name for name in ('s-cascade-decoder',) if name not in classes]
    if missing:
        raise RuntimeError(
            'text encoder schema is missing pipeline classes for: ' + ', '.join(missing))
    return classes


_SCHEMA: dict[str, list[dict]] | None = None


def text_encoder_schema() -> dict[str, list[dict]]:
    """
    ``--model-type`` to ordered text encoder slots.

    Each slot is ``name``, ``label``, ``encoders``, and optional ``modes``
    and ``nullable``. ``modes`` are monolithic checkpoint modes. They are
    omitted when that slot is not loaded that way. ``nullable`` is set only
    when the pipeline still runs with that encoder left unloaded.
    """
    global _SCHEMA
    if _SCHEMA is not None:
        return _SCHEMA
    schema: dict[str, list[dict]] = {}
    for model_type, pipeline_classes in _pipeline_classes().items():
        found: dict[str, list[str]] = {}
        for pipeline_class in pipeline_classes:
            hints = typing.get_type_hints(pipeline_class.__init__)
            for name, annotation in hints.items():
                if not _is_text_encoder_slot(name):
                    continue
                bucket = found.setdefault(name, [])
                for encoder in _class_names(annotation):
                    if encoder not in bucket:
                        bucket.append(encoder)
        for (slot_model_type, slot_name), encoders in _ENCODER_REPLACEMENTS.items():
            if slot_model_type == model_type:
                found[slot_name] = list(encoders)
        slots = []
        for name in sorted(found):
            encoders = found[name]
            if not encoders:
                continue
            slot = {
                'name': name,
                'label': _LABELS.get((model_type, name), encoders[0]),
                'encoders': encoders,
            }
            modes = _MODES.get((model_type, name))
            if modes:
                slot['modes'] = modes
            if (model_type, name) in _NULLABLE:
                slot['nullable'] = True
            slots.append(slot)
        if not slots:
            raise RuntimeError(f'{model_type} has no text encoder slots.')
        schema[model_type] = slots
    _SCHEMA = schema
    return schema
