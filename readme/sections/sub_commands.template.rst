Sub Commands
============

dgenerate implements additional functionality through the option ``--sub-command``.

For a list of available sub-commands use ``--sub-command-help``, which by default
will list available sub-command names.

For additional information on a specific sub-command use ``--sub-command-help NAME``

Multiple sub-command names can be specified to ``--sub-command-help`` if desired.

All sub-commands respect the ``--plugin-modules`` and ``--verbose`` arguments
even if their help output does not specify them, these arguments are handled
by dgenerate and not the sub-command.

Sub Command: image-process
--------------------------

The ``image-process`` sub-command can be used to run image processors implemented
by dgenerate on any file of your choosing including animated images and videos.

It has a similar but slightly different design/usage to the main dgenerate
command itself.

It can be used to run canny edge detection, openpose, etc. on any image or
video/animated file that you want.

The help output of ``image-process`` is as follows:


@COMMAND_OUTPUT[dgenerate --no-stdin --sub-command image-process --help]


Overview of specifying ``image-process`` inputs and outputs

.. code-block:: bash

    #!/usr/bin/env bash

    # Overview of specifying outputs, image-process can do simple operations
    # like resizing images and forcing image alignment with --align, without the
    # need to specify any other processing operations with --processors. Running
    # image-process on an image with no other arguments simply aligns it to 8 pixels,
    # given the defaults for its command line arguments

    # More file formats than .png are supported for static image output, all
    # extensions mentioned in the image-process --help documentation for --frame-format
    # are supported, the supported formats are identical to that mentioned in the --image-format
    # option help section of dgenerate's --help output

    # my_file.png -> my_file_processed_1.png

    dgenerate --sub-command image-process my_file.png --resize 512x512

    # my_file.png -> my_file.png (overwrite)

    dgenerate --sub-command image-process my_file.png --resize 512x512 --output-overwrite

    # my_file.png -> my_file.png (overwrite)

    dgenerate --sub-command image-process my_file.png -o my_file.png --resize 512x512 --output-overwrite

    # my_file.png -> my_dir/my_file_processed_1.png

    dgenerate --sub-command image-process my_file.png -o my_dir/ --resize 512x512 --no-aspect

    # my_file_1.png -> my_dir/my_file_1_processed_1.png
    # my_file_2.png -> my_dir/my_file_2_processed_2.png

    dgenerate --sub-command image-process my_file_1.png my_file_2.png -o my_dir/ --resize 512x512

    # my_file_1.png -> my_dir_1/my_file_1_processed_1.png
    # my_file_2.png -> my_dir_2/my_file_2_processed_2.png

    dgenerate --sub-command image-process my_file_1.png my_file_2.png \
    -o my_dir_1/ my_dir_2/ --resize 512x512

    # my_file_1.png -> my_dir_1/renamed.png
    # my_file_2.png -> my_dir_2/my_file_2_processed_2.png

    dgenerate --sub-command image-process my_file_1.png my_file_2.png \
    -o my_dir_1/renamed.png my_dir_2/ --resize 512x512


A few usage examples with processors:

.. code-block:: bash

    #!/usr/bin/env bash

    # image-process can support any input format that dgenerate itself supports
    # including videos and animated files. It also supports all output formats
    # supported by dgenerate for writing videos/animated files, and images.

    # create a video rigged with OpenPose, frames will be rendered to the directory "output" as well.

    dgenerate --sub-command image-process my-video.mp4 \
    -o output/rigged-video.mp4 --processors "openpose;include-hand=true;include-face=true"

    # Canny edge detected video, also using processor chaining to mirror the frames
    # before they are edge detected

    dgenerate --sub-command image-process my-video.mp4 \
    -o output/canny-video.mp4 --processors mirror "canny;blur=true;threshold-algo=otsu"


Sub Command: civitai-links
--------------------------

The ``civitai-links`` sub-command can be used to list the hard links for models available on a CivitAI model page.

These links can be used directly with dgenerate, it will automatically download the model for you.

You only need to select which models you wish to use from the links listed by this command.

See: `Utilizing CivitAI links and Other Hosted Models`_ for more information about how to use these links.

To get direct links to CivitAI models you can use the ``civitai-links`` sub-command
or the ``\civitai_links`` directive inside of a config to list all available models
on a CivitAI model page.

For example:

.. code-block:: bash

    #!/usr/bin/env bash

    # get links for the Crystal Clear XL model on CivitAI

    dgenerate --sub-command civitai-links "https://civitai.com/models/122822?modelVersionId=133832"

    # you can also automatically append your API token to the end of the URLs with --token
    # some models will require that you authenticate to download, this will add your token
    # to the URL for you

    dgenerate --sub-command civitai-links "https://civitai.com/models/122822?modelVersionId=133832" --token $MY_API_TOKEN


This will list every model link on the page, with title, there may be many model links
depending on what the page has available for download.

Output from the above example:

.. code-block:: text

    Models at: https://civitai.com/models/122822?modelVersionId=133832
    ==================================================================

    CCXL (Model): https://civitai.com/api/download/models/133832?format=SafeTensor&size=full&fp=fp16


Sub Command: auto1111-metadata
------------------------------

The ``auto1111-metadata`` sub-command can be used to add Automatic1111 style metadata to an image
generated by dgenerate.

You must use the dgenerate options ``--output-configs`` or ``--output-metadata`` for this to work.

The dgenerate option ``--output-configs`` will write the generation config to a file in the output directory,
which can be read by this sub-command.

Alternatively, you can use the dgenerate option ``--output-metadata`` to write the metadata directly to the image file,
which can then be read out of the PNG metadata (DgenerateConfig), or EXIF UserComment if you are using jpeg.

This sub-command supports PNG and JPEG files only.

This sub-command also exists as the config directive: ``\auto1111_metadata``.

The help output of ``image-process`` is as follows:

@COMMAND_OUTPUT[dgenerate --no-stdin --sub-command auto1111-metadata --help]


Example of using the ``auto1111-metadata`` sub-command with ``--output-metadata``


.. code-block:: bash

    #!/usr/bin/env bash

    # in this example, we are using --output-metadata to write the dgenerate config to the image file itself.

    dgenerate stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --inference-steps 30 \
    --guidance-scales 5 \
    --clip-skips 0 \
    --seeds 0 \
    --output-metadata \
    --output-path output \
    --output-size 512x512 \
    --prompts "hello world!"

    # Make a copy of the image, with Automatic1111 metadata added to it

    dgenerate --sub-command auto1111-metadata output\s_0_g_5-0_i_30_cs_0_step_1.png --output image_with_auto1111_metadata.png

    # Overwrite the image, dgenerates original metadata (the config) will be lost

    dgenerate --sub-command auto1111-metadata output\s_0_g_5-0_i_30_cs_0_step_1.png


You can also use the output from ``--output-configs`` for this task


.. code-block:: bash

    #!/usr/bin/env bash

    # in this example, we are using --output-configs to write the dgenerate config to a file next to the image.

    dgenerate stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --inference-steps 30 \
    --guidance-scales 5 \
    --clip-skips 0 \
    --seeds 0 \
    --output-configs \
    --output-path output \
    --output-size 512x512 \
    --prompts "hello world!"

    # Make a copy of the image, with Automatic1111 metadata added to it

    dgenerate --sub-command auto1111-metadata output\s_0_g_5-0_i_30_cs_0_step_1.png --config output\s_0_g_5-0_i_30_cs_0_step_1.dgen --output image_with_auto1111_metadata.png

    # Overwrite the image, dgenerates original metadata (the config) will be lost

    dgenerate --sub-command auto1111-metadata output\s_0_g_5-0_i_30_cs_0_step_1.png --config output\s_0_g_5-0_i_30_cs_0_step_1.dgen



Sub Command: to-diffusers
--------------------------

The ``to-diffusers`` sub-command can be used to convert single file diffusion model checkpoints from CivitAI
and elsewhere into diffusers format (a folder on disk with configuration).

This can be useful if you want to load a single file checkpoint with quantization.

You may also save models loaded from Hugging Face repos.

This sub-command also exists as the config directive: ``\to_diffusers``

In memory caching / memoization is disabled for this command to prevent unnecessary resource usage,
the models involved with the loaded pipeline are garbage collected immediately after the conversion happens.

.. code-block:: text

    #!/usr/bin/env bash

    # convert a CivitAI checkpoint (https://civitai.com/models/2711/21-sd-modern-buildings-style-md)
    # into a diffusers compatible model folder, containing separate checkpoint files for each
    # model component and related configuration

    dgenerate --sub-command to-diffusers \
    "https://civitai.com/api/download/models/3002?type=Model&format=PickleTensor&size=full&fp=fp16" \
    --model-type torch \
    --dtypes float16 float32 \
    --output modern_buildings


The help output of ``to-diffusers`` is as follows:

@COMMAND_OUTPUT[dgenerate --no-stdin --sub-command to-diffusers --help]


Sub Command: prompt-upscale
---------------------------

The ``prompt-upscale`` sub-command can be use to run on prompt texts without invoking image generation.

See: `Prompt Upscaling`_ for more information about prompt upscaling plugins.

This sub-command is designed in the same vein as ``dgenerate --sub-command image-process`` and the ``\image_process`` directive.

This sub-command also exists as the config directive: ``\prompt_upscale``

It allows you to output the prompts in various formats such as plain text, or structured json, toml, and yaml.

Prompts can be written to a file or printed to stdout, and in the case of the config directive ``\prompt_upscale``
they can also be written to a config template variable as a python list.

A comprehensive example of the ``\prompt_upscale`` config directive which might be helpful for understanding
this sub-commands functionality is available in the `examples folder <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/config_directives/prompt_upscale/prompt-upscale-directive-config.dgen>`_.

.. code-block:: text

    #!/usr/bin/env bash

    # upscale two prompts with magic prompt
    # using the default accelerator for your system
    # and print them as structured yaml to stdout

    dgenerate --sub-command prompt-upscale \
    "a cat sitting on a bench in a park" \
    "a dog sitting on a bench in a park" \
    --upscaler magicprompt;variations=10 -of yaml

The help output of ``prompt-upscale`` is as follows:

@COMMAND_OUTPUT[dgenerate --no-stdin --sub-command prompt-upscale --help]

