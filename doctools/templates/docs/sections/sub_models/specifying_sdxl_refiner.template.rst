Specifying an SDXL Refiner
==========================

When the main model is an SDXL model and ``--model-type sdxl`` is specified,
you may specify a refiner model with ``--sdxl-refiner``.

You can provide a path to a huggingface repo/blob link, folder on disk, or a model file
on disk such as a .pt, .pth, .bin, .ckpt, or .safetensors file.

This argument is parsed in much the same way as the argument ``--vae``, except the
model is the first value specified.

Loading arguments available when specifying a refiner are: ``revision``, ``variant``, ``subfolder``, and ``dtype``

The only named argument compatible with loading a .safetensors or other file directly off disk is ``dtype``

The other named arguments are available when loading from a huggingface repo/blob link,
or folder that may or may not be a local git repository on disk.

.. code-block:: bash

    #!/usr/bin/env bash

    # Basic usage of SDXL with a refiner

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"



If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner "stabilityai/stable-diffusion-xl-refiner-1.0;revision=main" \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


If you wish to specify a weights variant IE: load ``pytorch_model.<variant>.safetensors``, from a huggingface
repository that has variants of the same model, use the named argument ``variant``. By default this
value is the same as ``--variant`` unless you override it.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner "stabilityai/stable-diffusion-xl-refiner-1.0;variant=fp16" \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    #!/usr/bin/env bash

    # This is only a syntax example

    dgenerate huggingface/sdxl_model --model-type sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner "huggingface/sdxl_refiner;subfolder=repo_subfolder"


If you want to select the model precision, use the named argument ``dtype``. By
default this value is the same as ``--dtype`` unless you override it. Accepted
values are the same as ``--dtype``, IE: 'float32', 'float16', 'auto'

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner "stabilityai/stable-diffusion-xl-refiner-1.0;dtype=float16" \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


If you are loading a .safetensors or other file from a path on disk, simply do:

.. code-block:: bash

    #!/usr/bin/env bash

    # This is only a syntax example

    dgenerate huggingface/sdxl_model --model-type sdxl \
    --sdxl-refiner my_refinermodel.safetensors


When preforming inpainting or when using `ControlNets <#specifying-control-nets>`_, the
refiner will automatically operate in edit mode instead of cooperative denoising mode.
Edit mode can be forced in other situations with the option ``--sdxl-refiner-edit``.

Edit mode means that the refiner model is accepting the fully (or mostly) denoised output
of the main model generated at the full number of inference steps specified, and acting
on it with an image strength (image seed strength) determined by (1.0 - high-noise-fraction).

The output latent from the main model is renoised with a certain amount of noise determined
by the strength, a lower number means less noise and less modification of the latent output
by the main model.

This is similar to what happens when using dgenerate in img2img with a standalone model,
technically it is just img2img, however refiner models are better at enhancing details
from the main model in this use case.