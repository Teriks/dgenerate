Torch Compile
=============

``--torch-compile`` compiles repeated denoiser blocks, ControlNet blocks, and
VAE decoder blocks. The first use of each waits while the graph builds. A
later change of resolution or frame count compiles once more, then reuses
that graph. A compiled pipeline and an eager pipeline are separate cache
entries. TorchDynamo, Inductor, and Triton stay quiet unless ``-v`` is
set. Compile errors still print.

Module names on ``--torch-compile`` choose which components are compiled.
With no names, the set above is used. Accepted names are ``unet``,
``transformer``, ``transformer_2``, ``controlnet``, ``vae``,
``text_encoder``, ``text_encoder_2``, ``text_encoder_3``, ``image_encoder``,
``prompt_upsampler``, ``prompt_enhancer``, and ``connectors``.
``prompt_upsampler`` and ``prompt_enhancer`` are aliases. Either name compiles
the FLUX.3 prompt upsampler or the LTX Gemma enhancer, whichever that pipeline
has. Text encoders and that upsampler compile their decoder layers.
``image_encoder`` is compiled only when named here. The ``--image-encoder``
URI has no ``compile`` argument.

``compile=true`` on ``--unet``, ``--vae``, ``--text-encoders``,
``--control-nets``, ``--transformer``, or ``--prompt-upsampler`` compiles that
loaded module even when ``--torch-compile`` is omitted, and even when the
named modules omit it. A compiled prompt upsampler is still layer-split for
the rewrite, so the first ``generate()`` can build the graph again.

CUDA and XPU need Triton. On macOS, PyTorch 2.13 and newer compile these
blocks to a Metal kernel. CPU stays eager. BitsAndBytes, SDNQ, GGUF, and
other quantized weights stay eager. Stable Diffusion 3 compiles its joint
transformer blocks. Stable Cascade compiles its res, timestep, and attention
blocks. The LTX diffusion decoder stays eager. Console recipes for the models
that can compile include one Torch Compile control, left off. Its component
field is available while the checkbox is on. Leave that field empty to keep
the default set.

Weights streamed from disk skip ``--torch-compile``, including a URI
``compile=true``. A ``compile=true`` URI without ``--torch-compile`` is still
compiled when the weights are streamed from disk.

Wan-Animate-2 compiles its transformer blocks and VAE blocks on every load,
with or without ``--torch-compile``. The transformer compile keeps flex attention
block-sparse. The pipeline cache stores one compiled Wan-Animate-2 entry
either way. It ignores module names on ``--torch-compile`` and a URI
``compile=true``.

``DGENERATE_TORCH_COMPILE`` defaults to ``1``. ``\env DGENERATE_TORCH_COMPILE=0``
leaves every model eager, including Wan-Animate-2. dgenerate warns when a
compile was skipped. Wan-Animate-2 then builds the full attention score
matrix, which can run out of memory at video resolution. Set it in the
environment or in ``init.dgen``.
