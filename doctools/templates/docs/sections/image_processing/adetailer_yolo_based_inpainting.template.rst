Adetailer (YOLO based inpainting)
=================================

The adetailer compositing algorithm can be used with YOLO detection models for automated inpainting
of features detected in generated or arbitrary images.

This can be done in one of two ways, as a ``--post-processors`` step using the ``adetailer`` image
processor, using the previously executed diffusion pipeline for inpainting on generated output.

Or on arbitrary images by specifying detector URIs to ``--adetailer-detectors`` with any supported model
type.

Currently adetailer supports these model types:

    * ``--model-type sd``
    * ``--model-type sdxl``
    * ``--model-type kolors``
    * ``--model-type sd3``
    * ``--model-type flux``
    * ``--model-type flux-fill``


In effect, adetailer is supported by most pipelines that support any kind of inpainting, except for
DeepFloyd related model types.

dgenerate's adetailer implementation can be used to very selectively detail specific detections, the
implementation includes features that allow using different prompts and settings on different faces / hands in
an image etc. Allowing for pinpoint use on images with multiple characters or objects of interest.


Adetailer Image Processor
-------------------------

The adetailer image processor can only be executed after an image has been generated with a diffusion pipeline,
as it reuses the last executed pipelines modules to inpaint an image.

It does not necessarily have to be used with ``--post-processors`` as long as a diffusion based image operation
has taken place prior with a supported ``--model-type`` value involved.

The adetailer image processor has many options and it is recommended to take a look at the output of
``dgenerate --image-processor-help adetailer`` and view the examples located at
`examples/adetailer/post_processor <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/adetailer/post_processor>`_
for usage information.


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # quick example showing the adetailer post processor
    # applying detailing to (hopefully) a single face
    # in a generate image, using bingsu's original
    # adetailer YOLO models

    # note that any YOLO model can be used for
    # detection of features, leading to many
    # possible use cases

    stabilityai/stable-diffusion-xl-base-1.0
    --model-type sdxl
    --dtype float16
    --variant fp16
    --inference-steps 30
    --guidance-scales 7
    --clip-skips 0
    --gen-seeds 1
    --output-path sdxl
    --output-size 1024x1024
    --post-processors adetailer;\
                      model=Bingsu/adetailer;\
                      weight-name=face_yolov8n.pt;\
                      prompt="image of emma watson";\
                      negative-prompt="nsfw, blurry, disfigured";\
                      prompt-weighter=sd-embed;\
                      guidance-scale=7;\
                      inference-steps=30;\
                      strength=0.4
    --prompts "full body photo of emma watson in black clothes, \
               night city street, bokeh; pencil drawing, black and white, \
               greyscale, poorly drawn, bad anatomy, wrong anatomy, extra limb, \
               missing limb, floating limbs, disconnected limbs, mutation, mutated, \
               ugly, disgusting, amputation"


Processor arguments such as ``index-filter`` can be used to only include certain detections
on a per-detector basis. The detections have a deterministic order based on position, that
mimics the order of english words on a page, i.e. from left to right, top to bottom, sorted
by the top left corner of the detection area.

The processor can be chained together with another adetailer processor definition to
inpaint multiple types of objects in an image, or different detection indices separately.

.. code-block::

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    stabilityai/stable-diffusion-xl-base-1.0
    --model-type sdxl
    --variant fp16
    --dtype float16
    --inference-steps 30
    --guidance-scales 8
    --output-path multi-subject-config
    --output-size 1024
    --seeds 55307998457041
    --prompts "a woman and a man standing next to each other"
    --post-processors \

    # womans face on the left comes first
    adetailer;\
    model=Bingsu/adetailer;\
    weight-name=face_yolov8n.pt;\
    prompt="the face of a woman";\
    detector-padding=50;\
    guidance-scale=7;\
    seed=0;\
    index-filter=0;\
    inference-steps=30;\
    strength=0.7 \

    # mans face on the right comes second.
    # note the space before the last line
    # continuation character above, that
    # separates these two processors with
    # a single space.

    adetailer;\
    model=Bingsu/adetailer;\
    weight-name=face_yolov8n.pt;\
    prompt="the face of a man";\
    detector-padding=50;\
    guidance-scale=7;\
    seed=0;\
    index-filter=1;\
    inference-steps=30;\
    strength=0.7


The processor argument "class-filter" can also be used to filter to only detections with
a certain class ID or class name in the model, this filter acts before "index-filter",
"class-filter" can be specified as a mix of IDs (integers) or direct names, this is useful
when using models that can detect multiple classes of objects. This is described in the
adetailer processor help output below.

@COMMAND_OUTPUT[dgenerate --no-stdin --image-processor-help adetailer]

Adetailer Pipeline
------------------

The secondary usage of adetailer with dgenerate is on arbitrary input images, this can be useful
for editing existing images on disk or even the output of previous dgenerate invocations.

You can also refine images with different models than the
original generation, or with  different model types all together.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # first we should generate an image that we want to refine
    # with adetailer using some model, or operation that affects
    # the last_images template variable, even \image_process will
    # do this

    stabilityai/stable-diffusion-xl-base-1.0
    --model-type sdxl
    --dtype float16
    --variant fp16
    --inference-steps 30
    --guidance-scales 7
    --gen-seeds 1
    --output-path advanced-postprocess
    --output-prefix unrefined
    --output-size 1024
    --prompts "full body photo of emma watson in black clothes, \
               night city street, bokeh; pencil drawing, black and white, \
               greyscale, poorly drawn, bad anatomy, wrong anatomy, extra limb, \
               missing limb, floating limbs, disconnected limbs, mutation, mutated, \
               ugly, disgusting, amputation"

    # make every image from the last generation an --image-seeds value
    # that gets passed to the adetailer step, the settings
    # for the adetailer process are far more configurable this
    # way, but this cannot be done as a one liner on the command
    # line as with --post-processors

    # we can for instance, use the combinatorial arguments of dgenerate
    # to make variations, and also run the SDXL refiner as a final step,
    # where as with --post-processors, the SDXL refiner always runs before
    # adetailer

    # we can also choose any model type and model that we want to use with
    # adetailer, even different models than the model that generated the initial image,
    # as long as that model supports inpainting

    # this means we can apply this postprocess to the output of models that do not
    # support adetailer if desired, such as Stable Cascade etc.

    # now, combinatorially refine 8 variants using different settings for adetailer
    # so that we can observe differences in the output

    stabilityai/stable-diffusion-xl-base-1.0
    --model-type sdxl
    --variant fp16
    --dtype float16
    --image-seeds {{ quote(last_images) }}
    --inference-steps 30
    --guidance-scales 7
    --adetailer-detectors Bingsu/adetailer;weight-name=face_yolov8n.pt
    --adetailer-mask-blurs 4 8
    --adetailer-mask-dilations 4 8
    --image-seed-strengths 0.4 0.7
    --output-path advanced-postprocess
    --output-prefix refined
    --model-cpu-offload # save some memory, a lot of models are being used
    --prompts "image of emma watson; nsfw, blurry, disfigured"


Almost all of the arguments of the ``adetailer`` image processor exist as URI arguments
when specifying detectors with ``--adetailer-detectors`` including ``class-filter`` and ``index-filter``

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # edit multiple adetailer YOLO detected features in one go on an arbitrary image
    # using detector URI arguments to override the prompt and selected detection
    # index

    # the URI prompts are weighted with --prompt-weighter sd-embed

    stabilityai/stable-diffusion-xl-base-1.0
    --model-type sdxl
    --variant fp16
    --dtype float16
    --inference-steps 30
    --image-seeds ../../../media/americangothic.jpg
    --image-seed-strengths 0.5
    --guidance-scales 8
    --adetailer-detectors \
        Bingsu/adetailer;weight-name=face_yolov8n.pt;prompt="a smiling woman's face";negative-prompt="ugly, deformed";index-filter=0 \
        Bingsu/adetailer;weight-name=face_yolov8n.pt;prompt="a smiling mans face";negative-prompt="ugly, deformed";index-filter=1
    --adetailer-mask-blurs 16
    --adetailer-mask-shapes circle
    --adetailer-detector-paddings 40
    --prompt-weighter sd-embed
    --output-path multi-subject-config


YOLO Detection Processor
------------------------

The ``yolo`` image processor can be used to preview YOLO model detections in an image.

It will draw boxes around detections on the image, displaying the detection
index, class ID, and class name of the detection.

It can also draw the outlines of the detection masks, if the model supports
generating masks.

This can be used to determine what is being classified in an image when
preparing to use the ``adetailer`` image processor.

The detection behavior mimics that of the ``adetailer`` image processor
and adetailer pipeline mode.

This processor also supports generating black and white masks from
detections, in an identical way to the ``adetailer`` image processors
internal inpaint mask generation, minus the gaussian blur and dilation
steps, which can be applied using separate image processors and image
processor chaining if desired.

@COMMAND_OUTPUT[dgenerate --no-stdin --image-processor-help yolo]

