Specifying Textual Inversions (embeddings)
==========================================

One or more Textual Inversion models (otherwise known as embeddings) may be specified with ``--textual-inversions``

Textual inversions are supported for these model types:

    * ``--model-type sd``
    * ``--model-type pix2pix``
    * ``--model-type upscaler-x4``
    * ``--model-type sdxl``
    * ``--model-type sdxl-pix2pix``
    * ``--model-type flux`` (``txt2img``, ``txt2img + ControlNets``, ``inpainting + ControlNets`` only)
    * ``--model-type flux-fill`` (``inpainting`` only)

You can provide a huggingface repository slug, .pt, .pth, .bin, .ckpt, or .safetensors files.
Blob links are not accepted, for that use ``subfolder`` and ``weight-name`` described below.

Arguments pertaining to the loading of each textual inversion model may be specified in the same
way as when using ``--loras`` minus the scale argument.

Available arguments are: ``token``,  ``revision``, ``subfolder``, and ``weight-name``

Named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk, when loading directly from a .safetensors file
or other file from a path on disk they should not be used.

The ``token`` argument may be used to override the prompt token value, which is the text token
in the prompt that triggers the inversion, textual inversions for stable diffusion usually
include this token value in the model itself, for instance in the example below the token
for ``Isometric_Dreams-1000.pt`` is ``Isometric_Dreams``.

The token value used for SDXL and Flux models is a bit different, a default
value is not provided in the model file. If you do not provide a token value, dgenerate
will assign the tokens default value to the filename of the model with any spaces converted to
underscores, and with the file extension removed.


.. code-block:: bash

    #!/usr/bin/env bash

    # Load a textual inversion from a huggingface repository specifying it's name in the repository
    # as an argument

    dgenerate Duskfallcrew/isometric-dreams-sd-1-5  \
    --textual-inversions "Duskfallcrew/IsometricDreams_TextualInversions;weight-name=Isometric_Dreams-1000.pt" \
    --scheduler KDPM2DiscreteScheduler \
    --inference-steps 30 \
    --guidance-scales 7 \
    --prompts "a bright photo of the Isometric_Dreams, a tv and a stereo in it and a book shelf, a table, a couch,a room with a bed"


You can change the ``token`` value to affect the prompt token used to trigger the embedding

.. code-block:: bash

    #!/usr/bin/env bash

    # Load a textual inversion from a huggingface repository specifying it's name in the repository
    # as an argument

    dgenerate Duskfallcrew/isometric-dreams-sd-1-5  \
    --textual-inversions "Duskfallcrew/IsometricDreams_TextualInversions;weight-name=Isometric_Dreams-1000.pt;token=<MY_TOKEN>" \
    --scheduler KDPM2DiscreteScheduler \
    --inference-steps 30 \
    --guidance-scales 7 \
    --prompts "a bright photo of the <MY_TOKEN>, a tv and a stereo in it and a book shelf, a table, a couch,a room with a bed"


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    #!/usr/bin/env bash

    # This is a non working example as I do not know of a repo that utilizes revisions with
    # textual inversion weights :) this is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --textual-inversions "huggingface/ti_repo;revision=main"


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    #!/usr/bin/env bash

    # This is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --textual-inversions "huggingface/ti_repo;subfolder=repo_subfolder;weight-name=ti_model.safetensors"


If you are loading a .safetensors or other file from a path on disk, simply do:

.. code-block:: bash

    #!/usr/bin/env bash

    # This is only a syntax example

    dgenerate Lykon/dreamshaper-8 \
    --prompts "Syntax example" \
    --textual-inversions "my_ti_model.safetensors"
