Negative Prompt
===============

In order to specify a negative prompt, each prompt argument is split
into two parts separated by ``;``

The prompt text occurring after ``;`` is the negative influence prompt.

To attempt to avoid rendering of a saddle on the horse being ridden, you
could for example add the negative prompt ``saddle`` or ``wearing a saddle``
or ``horse wearing a saddle`` etc.


.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse; horse wearing a saddle" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


Multiple Prompts
================

Multiple prompts can be specified one after another in quotes in order
to generate images using multiple prompt variations.

The following command generates 10 uniquely named images using two
prompts and five random seeds ``(2x5)``

5 of them will be from the first prompt and 5 of them from the second prompt.

All using 50 inference steps, and 10 for guidance scale value.


.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" "an astronaut riding a donkey" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512