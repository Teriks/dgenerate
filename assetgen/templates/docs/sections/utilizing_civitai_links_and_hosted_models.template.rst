Utilizing CivitAI links and Other Hosted Models
===============================================

Any model accepted by dgenerate that can be specified as a single file
inside of a URI (or otherwise) can be specified by a URL to a model file.
dgenerate will attempt to download the file from the URL directly, store it in
the web cache, and then use it.

You may also use the ``\download`` config directive to assist in pre
downloading other resources from the internet. The directive has the ability
to specify arbitrary storage locations. See: `The \\download directive`_

You can also use the ``download()`` template function for similar
purposes. See: `The download() template function`_

In the case of CivitAI you can use this to bake models into your script
that will be automatically downloaded for you, you just need a CivitAI
account and API token to download models.

Your API token can be created on this page: https://civitai.com/user/account

Near the bottom of the page in the section: ``API Keys``

You can use the `civitai-links <Sub Command: civitai-links_>`_ sub-command to fetch the necessary model
links from a CivitAI model page. You may also use this sub-command in the form of the config
directive ``\civitai_links`` from a config file or the Console UI.

You can also `(Right Click) -> Copy Link Address` on a CivitAI models download link to get the necessary URL.

If you plan to download many large models to the web cache in this manner you may wish
to adjust the global cache expiry time so that they exist in the cache longer than the default of 12 hours.

You can see how to change the cache expiry time in this section `File Cache Control`_

If you set the environmental variable ``CIVIT_AI_TOKEN``, your token will be appended to
CivitAI API links automatically, this example appends it manually.

.. code-block:: bash

    #!/usr/bin/env bash

    # Download the main model from civitai using an api token

    # https://civitai.com/models/122822?modelVersionId=133832

    TOKEN=your_api_token_here

    MODEL="https://civitai.com/api/download/models/133832?type=Model&format=SafeTensor&size=full&fp=fp16&token=$TOKEN"

    dgenerate $MODEL \
    --model-type sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --sdxl-high-noise-fractions 0.8 \
    --guidance-scales 8 \
    --inference-steps 40 \
    --prompts "a fluffy cat playing in the grass"


This method can be used for VAEs, LoRAs, ControlNets, and Textual Inversions
as well, whenever single file loads are supported by the argument.