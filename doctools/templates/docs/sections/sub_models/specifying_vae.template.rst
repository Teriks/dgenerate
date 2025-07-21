Specifying a VAE
================

To specify a VAE directly use ``--vae``.

VAEs are supported for these model types:

    * ``--model-type sd``
    * ``--model-type pix2pix``
    * ``--model-type upscaler-x2``
    * ``--model-type upscaler-x4``
    * ``--model-type sdxl``
    * ``--model-type sdxl-pix2pix``
    * ``--model-type kolors``
    * ``--model-type sd3``
    * ``--model-type flux``

The URI syntax for ``--vae`` is ``AutoEncoderClass;model=(huggingface repository slug/blob link or file/folder path)``

Named arguments when loading a VAE are separated by the ``;`` character and are not positional,
meaning they can be defined in any order.

Loading arguments available when specifying a VAE are: ``model``, ``revision``, ``variant``, ``subfolder``, and ``dtype``

The only named arguments compatible with loading a .safetensors or other model file
directly off disk are ``model`` and ``dtype``

The other named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk.

Available encoder classes are:

* AutoencoderKL
* AsymmetricAutoencoderKL (Does not support ``--vae-slicing`` or ``--vae-tiling``)
* AutoencoderTiny
* ConsistencyDecoderVAE

The AutoencoderKL encoder class accepts huggingface repository slugs/blob links,
.pt, .pth, .bin, .ckpt, and .safetensors files. Other encoders can only accept huggingface
repository slugs/blob links, or a path to a folder on disk with the model
configuration and model file(s).


.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --vae "AutoencoderKL;model=stabilityai/sd-vae-ft-mse" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``,
``subfolder`` is required in this example as well because the VAE model file exists in a subfolder
of the specified huggingface repository.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --revision fp16 \
    --dtype float16 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;revision=fp16;subfolder=vae" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you wish to specify a weights variant IE: load ``pytorch_model.<variant>.safetensors``, from a huggingface
repository that has variants of the same model, use the named argument ``variant``. this value does NOT default to the value
``--variant`` to prevent errors during common use cases. If you wish to select a variant you must specify it in the URI.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --variant fp16 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;subfolder=vae;variant=fp16" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;subfolder=vae" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you want to specify the model precision, use the named argument ``dtype``,
accepted values are the same as ``--dtype``, IE: ``float32``, ``float16``, ``auto``

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --revision fp16 \
    --dtype float16 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;revision=fp16;subfolder=vae;dtype=float16" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you are loading a .safetensors or other file from a path on disk, only the ``model``, and ``dtype``
arguments are available.

.. code-block:: bash

    #!/usr/bin/env bash

    # These are only syntax examples

    dgenerate huggingface/diffusion_model \
    --vae "AutoencoderKL;model=my_vae.safetensors" \
    --prompts "Syntax example"

    dgenerate huggingface/diffusion_model \
    --vae "AutoencoderKL;model=my_vae.safetensors;dtype=float16" \
    --prompts "Syntax example"
