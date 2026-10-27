Face restoration
================

The ``face-restore`` image processor detects faces with facexlib (RetinaFace),
aligns each face crop, runs a restoration model, and pastes the results back
into the image. It requires the ``facexlib`` package (included in a normal
dgenerate install).

Restoration weights are Spandrel ``FaceSR`` models. GFPGAN (v1.2 through v1.4),
CodeFormer (with the ``spandrel_extra_arches`` extra), and RestoreFormer are
supported. Pass a local path or HTTP/HTTPS URL to the ``model`` argument.

Use ``face-restore`` in ``--post-processors`` or any ``--*-image-processors``
option, or run it on files with ``--sub-command image-process`` or the
``\image_process`` directive. See `Image Processors`_, `Upscaling Images`_, and
`Sub Command: image-process`_.

Use ``--image-processor-help face-restore`` for the full argument list:

@COMMAND_OUTPUT[dgenerate --no-stdin --image-processor-help face-restore]

The ``fidelity`` argument applies to CodeFormer only (0 favors a cleaner
reconstruction, 1 stays closer to the input face). GFPGAN and RestoreFormer
ignore it.

Face restoration models often reject half precision. When ``dtype=float16`` or
``bfloat16`` is unsupported, dgenerate falls back to ``float32``.

``detector-device`` can run RetinaFace on a different device than the
restoration model (``device``). ``face-size`` defaults to 512 and must stay 512
for GFPGAN and CodeFormer.

To upscale first and then restore faces in one generation call, list
``upscaler`` before ``face-restore`` in the same ``--post-processors``
specification.

GFPGAN
------

@EXAMPLE[@PROJECT_DIR/examples/image_process/face-restore-gfpgan-config.dgen]

CodeFormer
----------

@EXAMPLE[@PROJECT_DIR/examples/image_process/face-restore-codeformer-config.dgen]

RestoreFormer
-------------

@EXAMPLE[@PROJECT_DIR/examples/image_process/face-restore-restoreformer-config.dgen]
