Inpainting With Auto Crop
=========================

The inpaint crop feature provides built-in functionality for automatically cropping to mask bounds during
inpainting operations. This allows inpainting at higher effective resolutions for better quality results by
processing only the relevant masked region at full resolution, then pasting the result back onto the original image.

The simplest way to enable inpaint cropping is with the ``--inpaint-crop`` argument:

.. code-block:: bash

    #!/usr/bin/env bash
    
    dgenerate stabilityai/stable-diffusion-xl-base-1.0 \
    --model-type sdxl \
    --image-seeds "examples/media/horse1.jpg;examples/media/horse1-mask.jpg" \
    --inpaint-crop \
    --output-size 1024 \
    --prompts "a pink horse from a fantasy world"

This will automatically crop the input image and mask to the bounds of the mask area (with 50 pixels of padding by default),
process the cropped region at the specified output resolution (aspect correct, fixed width), and paste the generated
result back onto the original uncropped image.

The inpaint crop arguments are:

* ``--inpaint-crop`` / ``-ic`` - Enable cropping to mask bounds for inpainting
* ``--inpaint-crop-paddings`` / ``-icp`` - (Combinatorial) Specify padding values around mask bounds (default: 50)
* ``--inpaint-crop-feathers`` / ``-icf`` - (Combinatorial) Apply feathering for smooth blending when pasting back
* ``--inpaint-crop-masked`` / ``-icm`` - Use mask when pasting to replace only masked areas

Important limitations:

* Cannot be used with image seed batching (``--image-seeds`` with multiple images/masks in the definition), see `Batching Input Images and Inpaint Masks`_ for details
* However, ``--batch-size > 1`` is supported for generating multiple variations of a single crop
* ``--inpaint-crop-feathers`` and ``--inpaint-crop-masked`` are mutually exclusive

Padding formats for ``--inpaint-crop-paddings``:

* ``32`` - 32px uniform padding on all sides  
* ``10x20`` - 10px horizontal, 20px vertical padding
* ``10x20x30x40`` - 10px left, 20px top, 30px right, 40px bottom padding

@EXAMPLE[@PROJECT_DIR/examples/inpaint_autocrop/hi-res-auto-cropped-config.dgen]

You can also use automatic mask detection with `SAM <Segment Anything Mask Generation>`_ or `YOLO <YOLO Detection Processor>`_ for dynamic masking scenarios:

@EXAMPLE[@PROJECT_DIR/examples/inpaint_autocrop/hi-res-auto-cropped-sam-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/inpaint_autocrop/hi-res-auto-cropped-yolo-config.dgen]

The inpaint crop functionality provides a built-in alternative to manually using image
processors for the same effect. The manual approach using ``crop-to-mask`` and ``paste``
processors offers more granular control but requires more complex configuration,
while the built-in ``--inpaint-crop`` is simpler and more compatible with animated
inputs and automatic feature detection.

For example, this functionality can be duplicated for images with image processors,
but not easily for animations:

@EXAMPLE[@PROJECT_DIR/examples/inpaint_autocrop/processors-hi-res-auto-cropped-config.dgen]