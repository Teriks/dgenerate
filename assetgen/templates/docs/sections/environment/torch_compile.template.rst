Torch Compile
=============

``--torch-compile`` compiles repeated denoiser blocks, ControlNet blocks, and
VAE decoder blocks. The first use of each waits while the graph builds. A
later change of resolution or frame count compiles once more, then reuses
that graph. A compiled pipeline and an eager pipeline are separate cache
entries. TorchDynamo, Inductor, and Triton stay quiet unless ``-v`` is
set. Compile errors still print.

CUDA and XPU need Triton. On macOS, PyTorch 2.13 and newer compile these
blocks to a Metal kernel. CPU stays eager. BitsAndBytes, SDNQ, GGUF, and
other quantized weights stay eager. Stable Diffusion 3 compiles its joint
transformer blocks. Stable Cascade compiles its res, timestep, and attention
blocks. Text encoders, image encoders, and the LTX diffusion decoder stay
eager. Console recipes for the models that can compile include a Torch Compile
checkbox, left off.

Wan-Animate-2 compiles its transformer blocks and VAE blocks on every load,
with or without the flag. The transformer compile keeps flex attention
block-sparse. The pipeline cache stores one compiled Wan-Animate-2 entry
either way.

``DGENERATE_TORCH_COMPILE`` defaults to ``1``. ``\env DGENERATE_TORCH_COMPILE=0``
leaves every model eager, including Wan-Animate-2. dgenerate warns when a
compile was skipped. Wan-Animate-2 then builds the full attention score
matrix, which can run out of memory at video resolution. Set it in the
environment or in ``init.dgen``.
