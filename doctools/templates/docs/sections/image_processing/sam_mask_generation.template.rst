Segment Anything Mask Generation
================================

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

You can combine this processor with the ``crop`` and ``paste`` processor to perform
manual / interactive adetailer like editing.

@EXAMPLE[@PROJECT_DIR/examples/sam/manual-adetailer-config.dgen]


YOLO + SAM Automated Segmentation
=================================

For automated object detection and segmentation, the ``yolo-sam`` image processor
combines YOLO object detection with SAM segmentation in a single step. This processor
first uses a YOLO model to detect objects and generate bounding boxes, then uses
those bounding boxes as prompts for the SAM model to create precise segmentation masks.

This is particularly useful for workflows where you want to automatically detect and
segment all instances of specific object classes without manual intervention. The
processor supports all YOLO filtering options (confidence thresholds, class filters,
index filters) and can generate either annotated preview images or composite masks.

The ``yolo-sam`` processor is especially valuable for:

* Video processing workflows: Consistent automated segmentation across video frames
* Adapting detection-only YOLO models: Adding segmentation capabilities to YOLO models that only provide bounding boxes
* Creating masks for specific detected objects using class / index filtering

@COMMAND_OUTPUT[dgenerate --no-stdin --image-processor-help yolo-sam]
