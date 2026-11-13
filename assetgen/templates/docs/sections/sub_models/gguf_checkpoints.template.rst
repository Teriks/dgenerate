.. _gguf-checkpoints:

GGUF Checkpoints
================

dgenerate loads pre-quantized ``.gguf`` diffusion transformers and text
encoders, including city96 files and ComfyUI GGUF files. Like a ComfyUI
quantized ``.safetensors`` file, a GGUF file is a file format, not a
``--quantizer`` backend: point ``--transformer`` (or the model path,
``--wan-second-transformer``, or ``--text-encoders``) at the file and it is
recognized from its tensors. Do not set ``quantizer=`` on that URI, and do
not use ``--quantizer gguf``. Loading requires the ``gguf`` package.

Quantized linears stay packed, so a transformer needs roughly the size of
its file. Each matmul dequantizes its weight. On CUDA, ROCm, and XPU with
Triton installed, including the ``triton-windows`` extra, Q4_0, Q4_1, Q5_0,
Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, and IQ4_XS unpack in Triton
and multiply with a PyTorch GEMM. PyTorch dequantizes the weight when Triton
is not installed.

ComfyUI files prefix tensor names with ``model.diffusion_model.``. That
prefix is removed on load. A Comfy linear stored transposed stays packed.
Dequantization uses the Triton unpack on CUDA, ROCm, and XPU, then the weight
is transposed.

Transformers
------------

Point ``--transformer`` at the file or a Hugging Face blob link. The
repository on the model line still supplies the VAE and text encoders. Wan
2.2 passes the low-noise expert through ``--wan-second-transformer``.

@EXAMPLE[@PROJECT_DIR/examples/flux/gguf/flux-dev-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/flux2/gguf/dev-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/wan/gguf/text-to-video-config.dgen]

As the Model
------------

A transformer-only ``.gguf`` can be the model itself. dgenerate reads the
tensor names and shapes, picks the Diffusers repository below, and loads the
text encoders, tokenizers, and VAE from there. ``--text-encoders`` and
``--vae`` still override that. ``+`` keeps a detected encoder and ``null``
skips one. A file that already contains text encoders keeps those encoders.

@EXAMPLE[@PROJECT_DIR/examples/flux/gguf/flux-dev-as-model-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/stablediffusion3/gguf/sd3-medium-as-model-config.dgen]

The repository is chosen from the tensors. ``--model-type`` picks a sibling
when those tensors match more than one pipeline. The file name picks a
sibling when the tensors do not: ``turbo`` selects a distilled Z-Image,
Krea 2, or Qwen-Image 2.1 repository, and a Wan or HunyuanVideo 1.5 name
selects task and resolution.

* SD3, SD3.5 medium, and SD3.5 large use the Diffusers fingerprint.
* Flux.1 dev, schnell, and fill use the Diffusers fingerprint.
  ``--model-type flux-kontext`` selects Kontext.
* Flux.2 dev is hidden width 6144, ``black-forest-labs/FLUX.2-dev``.
  Klein 4B is width 3072. Klein 9B is width 4096,
  ``black-forest-labs/FLUX.2-klein-base-9B``. ``--model-type flux2-klein-kv``
  selects Klein KV from that same width.
* Z-Image is recognized from ``cap_embedder`` and ``adaLN``. Base and Turbo
  share one shape, so the module config is Turbo. A file name that says
  Z-Image and does not say turbo uses ``Tongyi-MAI/Z-Image``.
* Qwen-Image is recognized from ``time_text_embed``. Edit uses this same
  module. ``--model-type qwen-image-edit`` selects edit, and ``2509`` in the
  file name selects edit-plus. ``addition_t_embedding`` selects
  ``Qwen/Qwen-Image-Layered``. A Qwen-Image file is not a layered file.
* Qwen-Image 2.1 is recognized from ``txt_in.text_norm``. The config is
  ``Qwen/Qwen-Image-2.1``, or the Turbo repository when the file name says
  turbo.
* Krea 2 is recognized from ``txtfusion`` or ``first.weight``. The config is
  ``krea/Krea-2-Raw``, or Turbo when the file name says turbo.
* Kandinsky 5 video is recognized from ``visual_transformer_blocks``. Hidden
  width 1792 is Lite and 4096 is Pro.
* LTX-2 and LTX-2.5 both have ``audio_patchify_proj``.
  ``keyframes_abs_pos_embedding`` or ``prompt_adaln_single`` selects
  ``Lightricks/LTX-2.5-Diffusers``. Without those tensors the file is the
  19B LTX-2 transformer and uses ``Lightricks/LTX-2`` (Gemma 3 and the
  2-layer connectors). LTX-Video 0.9.1, 0.9.5, and 0.9.7 use the Diffusers
  fingerprint.
* HunyuanVideo 1.5 is recognized from ``cond_type_embedding``. The file name
  selects 480p or 720p, and text-to-video or image-to-video.
* HunyuanImage 2.1 is recognized from ``txt_in.input_embedder`` and
  ``img_in.proj``. ``byt5_in`` with guidance selects the distilled config.
  Guidance without ``byt5_in`` selects the refiner. Otherwise the config is
  the base model.
* Motif-Video is recognized from a text cross-attention on the single stream,
  and uses ``Motif-Technologies/Motif-Video-2B``.
* Wan is recognized from ``patch_embedding`` width, plus VACE, Animate,
  Animate-2, and first-last-frame keys. Wan 2.1 image-to-video keeps an image
  embedder. Wan 2.2 image-to-video does not. A Wan 2.2 file name selects the
  mixture-of-experts repository.

Text Encoders
-------------

Point ``--text-encoders`` at a ``.gguf`` file and name the encoder class.
There is no ``mode`` and no ``subfolder``. The configuration is matched from
the file's vocabulary, hidden size, and layer count. Do not set ``quantizer``.

llama.cpp architectures ``t5``, ``t5encoder``, ``llama``, ``qwen2vl``,
``qwen3``, ``qwen3vl``, ``gemma3``, and ``gemma4`` are accepted. llama.cpp
tensor names are renamed to Hugging Face names, and the quantized linears
stay packed. A file that already stores Hugging Face or ComfyUI names,
including the LTX-2.5 Gemma 4 GGUF, is loaded as stored. T5 keeps its
feed-forward output projection dense, because T5 runs that projection in
float32. Gemma norms written by llama.cpp are converted back to the
Hugging Face values.

The class is the one the pipeline expects: ``T5EncoderModel`` and
``UMT5EncoderModel`` for T5 and UMT5, and the language-model class for Qwen,
Mistral, and Gemma. See :ref:`specifying-text-encoders`.

@EXAMPLE[@PROJECT_DIR/examples/flux/gguf/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/wan/gguf/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/flux2/gguf/klein-4b-text-encoder-config.dgen]

Other Options
-------------

``--loras`` load on top of the packed weights. The adapters stay active
instead of being fused into the weights. ``--lora-disk-fuse`` cannot merge
into a GGUF transformer.

``--model-cpu-offload`` and ``--model-sequential-offload`` move a GGUF
transformer. ``--model-group-offload`` does not stream its packed weights. It
places the transformer on the run device and still streams the other modules,
such as the VAE. ``--torch-compile`` leaves GGUF weights eager and warns.

A ``--quantizer`` on a full-precision text encoder still applies. It does not
apply to the GGUF file.

More examples are under ``examples/*/gguf``, for instance
`examples/flux/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/flux/gguf>`_,
`examples/flux2/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/flux2/gguf>`_,
`examples/z_image/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/z_image/gguf>`_,
`examples/qwen_image/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/qwen_image/gguf>`_,
`examples/ltx2/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx2/gguf>`_,
and `examples/wan/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/wan/gguf>`_.
