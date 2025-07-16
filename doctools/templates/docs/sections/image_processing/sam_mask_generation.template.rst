Segment Anything Mask Generation
--------------------------------

Segment anything promptable mask generation and preview is supported through
the ``u-sam`` image processor.  This processor is implemented with Ultralytics,
and not to be mistaken for the ``sam`` processor which is to be used for
generating input compatible with legacy SAM ControlNet models.

This processor can be prompted with point image coordinates, or bounding boxes,
or both. It will select the most likely object you want to segment given the prompts,
and then draw outlines around them in preview mode, or generate a black and white
mask in masks mode.

This can be used to quickly generate inpaint masks in an interactive way, especially
when combined with the Console UI's coordinate / bounding box selection utilities
provided in the image preview pane context menu.


@COMMAND_OUTPUT[dgenerate --no-stdin --image-processor-help u-sam]

You can combine this processor with the ``paste`` processor to preform
manual / interactive adetailer like editing.

@EXAMPLE[@PROJECT_DIR/examples/adetailer/manual_with_sam/config.dgen]