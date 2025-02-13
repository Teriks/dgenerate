Deterministic Output
====================

If you generate an image you like using a random seed, you can later reuse that seed in another generation.

Updates to the backing model may affect determinism in the generation.

Output images have a name format that starts with the seed, IE: ``s_(seed here)_ ...png``

Reusing a seed has the effect of perfectly reproducing the image in the case that all
other parameters are left alone, including the model version.

You can output a configuration file for each image / animation produced that will reproduce it
exactly using the option ``--output-configs``, that same information can be written to the
metadata of generated PNG files using the option ``--output-metadata`` and can be read back
with ImageMagick for example as so:

.. code-block:: bash

    #!/usr/bin/env bash

    magick identify -format "%[Property:DgenerateConfig]" generated_file.png

Generated configuration can be read back into dgenerate via a pipe or file redirection.

.. code-block:: bash

    #!/usr/bin/env bash

    # DO NOT DO THIS IF THE IMAGE IS UNTRUSTED, SUCH AS IF IT IS SOMEONE ELSE'S IMAGE!
    # VERIFY THAT THE METADATA CONTENT OF THE IMAGE IS NOT MALICIOUS FIRST,
    # USING THE IDENTIFY COMMAND ALONE

    magick identify -format "%[Property:DgenerateConfig]" generated_file.png | dgenerate

    dgenerate < generated-config.dgen

Specifying a seed directly and changing the prompt slightly, or parameters such as image seed strength
if using a seed image, guidance scale, or inference steps, will allow for generating variations close
to the original image which may possess all the original qualities about the image that you liked as well as
additional qualities.  You can further manipulate the AI into producing results that you want with this method.

Changing output resolution will drastically affect image content when reusing a seed to the point where trying to
reuse a seed with a different output size is pointless.

The following command demonstrates manually specifying two different seeds to try: ``1234567890``, and ``9876543210``

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --seeds 1234567890 9876543210 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512