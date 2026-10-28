.. _comfyui-quantized-checkpoints:

ComfyUI Quantized Checkpoints
=============================

dgenerate loads pre-quantized ``.safetensors`` checkpoints written for ComfyUI,
including the official files from ComfyUI, Black Forest Labs, and Lightricks,
and the fine-tunes published in the same formats. Like a ``.gguf`` file, such
a checkpoint is a file format, not a ``--quantizer`` backend: point
``--transformer`` (or the model path, or ``--text-encoders``) at the file and it
is recognized from its contents.

Supported formats:

* ``float8_e4m3fn`` and ``float8_e5m2``, with a per tensor or per row scale
  (``fp8_scaled``, ``fp8mixed``), including older files that name their scales
  ``scale_weight`` and ``scale_input``
* ``int8_tensorwise``, with or without ConvRot (``int8_convrot``)
* ``nvfp4`` (``fp4``, ``nvfp4``, ``fp4_mixed``), including NVIDIA ModelOpt exports
  without ComfyUI markers, such as the LTX-2 and Flux.2 FP4 files
* ``mxfp8``, FP8 E4M3 with one E8M0 scale per 32 inputs, swizzled the same way
  as NVFP4 block scales, padded to a multiple of 32
* ``convrot_w4a4``, signed INT4 packed two values per byte, with ConvRot
* ``asym_w4a8_int8``, 4-bit codes with a codebook, a group scale
  (``weight_s_rel``), a channel scale (``weight_s_channel``), and ConvRot.
  ``weight_codebook`` and ``weight_correction`` are used when the file has them
* ``w6a8_int8``, the same group scale, channel scale, and ConvRot, with uniform
  6-bit codes and no codebook
* Plain FP8 files with no scales (``fp8_e4m3fn``)

The format and group sizes come from each layer's ``comfy_quant`` marker, or
from the Kitchen ``_quantization_metadata`` entry when the file has no markers.

Quantized linears stay packed in memory, so a model needs roughly the size of
its file. Diffusers checkpoint conversion often splits fused weights, for
instance the attention qkv projection of Flux, SD3, Z-Image, and
HunyuanVideo 1.5. Those weights are split while still packed.

Transformers
------------

Point ``--transformer`` at the file or a Hugging Face blob link. The repository
still supplies the VAE and text encoders. Do not set ``quantizer=`` on that URI.
This is the simplest way to use one of these files.

@EXAMPLE[@PROJECT_DIR/examples/krea2/comfy-quant/raw-int8-convrot-config.dgen]

Wan 2.2 MoE takes the low noise expert through ``--wan-second-transformer``.

@EXAMPLE[@PROJECT_DIR/examples/wan/comfy-quant/moe-fp8-scaled-config.dgen]

As the Model
------------

A ComfyUI file can also be the model itself. An all-in-one checkpoint that
bundles text encoders and a VAE needs nothing else; its quantized text encoders
are dequantized as they load.

@EXAMPLE[@PROJECT_DIR/examples/stablediffusion3/comfy-quant/sd3.5-medium-single-file-config.dgen]

A file holding only the transformer loads its text encoders and VAE from the
matching Diffusers repository. ``--text-encoders`` and ``--vae`` still override
that, and ``null`` still skips an encoder.

@EXAMPLE[@PROJECT_DIR/examples/z-image/comfy-quant/single-file-config.dgen]

Text Encoders
-------------

ComfyUI quantized text encoder files load through ``--text-encoders``, and their
linears stay packed. Monolithic CLIP and T5 files use the ``mode`` argument as usual.
T5 keeps its feed-forward output projection dense, as T5 runs it in float32.

Language model text encoders (``Qwen3ForCausalLM``, ``Qwen3Model``,
``Mistral3ForConditionalGeneration``, ``Qwen2_5_VLForConditionalGeneration``, and
others) load from a single file with no ``mode`` and no ``subfolder``. The model
configuration is matched from the file against the configurations dgenerate ships
with. ComfyUI trims some of these files to the layers a pipeline reads, for example
the Flux.2 dev Mistral encoder keeps 30 of its 40 layers; dgenerate builds the
trimmed model to match. The ``quantizer`` argument is not supported for these files.

@EXAMPLE[@PROJECT_DIR/examples/flux2/comfy-quant/klein-4b-fp8-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/flux/comfy-quant/kontext-fp8-scaled-config.dgen]

Matrix Multiplication
---------------------

Where the hardware has a matching kernel, quantized layers run as low precision
matmuls. Otherwise each weight is decoded to the activation dtype for the matmul,
which works on any device, CPU included.

=========  ==========================================================================
Format     Hardware kernels, in the order they are tried
=========  ==========================================================================
INT8       CUDA (sm75+) and XPU: ``torch._int_mm``. ROCm: AITER ``gemm_a8w8``,
           then ``torch._int_mm``. MPS: MLX ``quantized_matmul``, then
           ``torch._weight_int8pack_mm``
FP8        CUDA (sm89+) and XPU: ``torch._scaled_mm``. ROCm MI300, MI350, and
           RDNA4: AITER ``gemm_a8w8``, then ``torch._scaled_mm``. MPS: MLX
           ``quantized_matmul``
NVFP4      CUDA Blackwell: block scaled ``scaled_mm``. MPS: MLX ``quantized_matmul``
=========  ==========================================================================

The CUDA, ROCm, and XPU kernels quantize the activations on the fly (W8A8, W4A4),
as ComfyUI does. Layers a file marks ``full_precision_matrix_mult``, and plain FP8
files, skip those kernels and decode instead, again as in ComfyUI. The MPS kernels
decode the weights inside the matmul and keep the activations as they are, so they
serve those layers too.

MXFP8, INT4 ConvRot, W4A8, and W6A8 are decoded to the activation dtype for the matmul.
That runs on CUDA, ROCm, XPU, MPS, and CPU. The tensor-core kernels ComfyUI uses
for them (Blackwell for MXFP8, SM 7.5 and newer for INT4, SM 8 and newer for W4A8)
are not called.

The AITER (ROCm) and MLX (MPS) kernels are downloaded from the Hugging Face Hub
through the ``kernels`` package, ``kernels-community/aiter-kernels`` and
``kernels-community/mlx-quantization-metal-kernels``. They are skipped when
``DIFFUSERS_DISABLE_REMOTE_CODE`` is set, or when they cannot be loaded.

The first call of each kernel for a layer shape is compared against the decoded
matmul for accuracy and speed. A kernel that fails or disagrees is turned off for
the device, and one that is slower is skipped for that shape; the next candidate
is tried in either case. Set the environment variable
``DGENERATE_COMFY_QUANT_SOFTWARE=1`` to always decode.

Other Options
-------------

``--loras`` load on top of the packed weights. The adapters stay active instead
of being fused into the weights, with the same result.

A ``--quantizer`` that applies to the transformer, for example the global
``--quantizer`` of the video pipelines, dequantizes the file first and quantizes
the dense weights with that backend.

Packed weights are ordinary integer parameters, so ``--model-cpu-offload``,
``--model-sequential-offload``, and ``--model-group-offload`` move them like any
other weights.

More examples are under ``examples/*/comfy-quant``, for instance
`examples/flux2/comfy-quant <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/flux2/comfy-quant>`_,
`examples/qwen-image/comfy-quant <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/qwen-image/comfy-quant>`_,
`examples/wan/comfy-quant <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/wan/comfy-quant>`_,
and `examples/ltx2/comfy-quant <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx2/comfy-quant>`_.
