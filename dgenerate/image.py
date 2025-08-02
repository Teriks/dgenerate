# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math
import typing

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFilter
import PIL.ImageChops
import cv2
import numpy
import piexif
import piexif.helper
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

__doc__ = """
Image operations commonly used by dgenerate.
"""


def is_image(obj) -> bool:
    """
    Check if an object is a PIL Image.

    :param obj: object to check
    :return: ``True`` if the object is a ``PIL.Image.Image``
    """
    return isinstance(obj, PIL.Image.Image)


def resize_image_calc(old_size: _types.Size,
                      new_size: _types.OptionalSize,
                      aspect_correct: bool = False,
                      align: int | None = None):
    """
    Calculate the new dimensions for a requested resize of an image..


    :param old_size: The old image size
    :param new_size: The new image size

    :param aspect_correct: preserve aspect ratio?

    :param align: Ensure returned size is aligned to this value.

    :return: calculated new size
    """

    if align is not None and align < 1:
        raise ValueError('align value may not be less than 1.')

    if new_size is None:
        if align is not None:
            return align_by(old_size, align=align)
        return old_size

    if align is not None:
        new_size = align_by(new_size, align=align)

    if old_size == new_size:
        return old_size

    if aspect_correct:
        width = new_size[0]
        w_percent = (width / float(old_size[0]))
        hsize = int((float(old_size[1]) * float(w_percent)))
        if align is not None:
            return width, hsize - hsize % align
        return width, hsize
    else:
        return new_size


def is_power_of_two(iterable: typing.Iterable[int]) -> bool:
    """
    Check if all elements are a power of 2.

    :param iterable: Elements to test

    :return: bool
    """
    for n in iterable:
        if n <= 0:
            return False
        if not (n & (n - 1)) == 0:
            return False
    return True


def nearest_power_of_two(iterable: typing.Iterable[int]) -> tuple:
    """
    Round all elements to the nearest power of two and return a tuple.

    :param iterable: Elements to round

    :return: tuple(...)
    """

    result = []
    for number in iterable:
        if number <= 0:
            result.append(0)
            continue

        lower_power_of_two = 2 ** int(math.log2(number))
        higher_power_of_two = 2 ** (int(math.log2(number)) + 1)

        if abs(number - lower_power_of_two) < abs(number - higher_power_of_two):
            result.append(lower_power_of_two)
        else:
            result.append(higher_power_of_two)

    return tuple(result)


def is_aligned(iterable: typing.Iterable[int], align: int) -> bool:
    """
    Check if all elements are aligned by a specific value.

    :param iterable: Elements to test
    :param align: The alignment value, ``None`` indicates no alignment.

    :return: bool
    """

    if align is None:
        return True

    if align < 1:
        raise ValueError('align value may not be less than 1.')

    return all(i % align == 0 for i in iterable)


def align_by(iterable: typing.Iterable[int], align: int) -> tuple:
    """
    Align all elements by a value and return a tuple

    :param iterable: Elements to align
    :param align: The alignment value, ``None`` indicates no alignment.

    :return: tuple(...)
    """

    if align is None:
        align = 1

    if align < 1:
        raise ValueError('align value may not be less than 1.')

    return tuple(i - i % align for i in iterable)


def copy_img(img: PIL.Image.Image):
    """
    Copy a :py:class:`PIL.Image.Image` while preserving its filename attribute.

    :param img: the image
    :return: a copy of the image
    """
    c = img.copy()

    if hasattr(img, 'filename'):
        c.filename = img.filename

    return c

def normalize_padding_value(padding: str | int | tuple[int, int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Normalize a padding value.

    This value can be a string, e.g. ``"10"``, or ``"10x10"``, or ``"10x10x10x10"``

    It can also be specified as a python ``int`` or ``tuple``

    Multidimensional padding values are laid out as: ``LEFTxTOPxRIGHTxBOTTOM``, or ``WIDTHxHEIGHT``

    This is the same all across dgenerate.

    :raise ValueError: If the padding value is specified incorrectly.

    :param padding: Padding value
    :return: Normalized padding (4 tuple of int)
    """

    # Parse padding argument
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)  # left, top, right, bottom
    else:
        if isinstance(padding, str):
            try:
                padding_dims = _textprocessing.parse_dimensions(str(padding))
            except ValueError as e:
                raise ValueError(f'Could not parse the padding value: {padding}') from e
        else:
            padding_dims = padding

        if len(padding_dims) == 1:
            # Uniform padding
            p = padding_dims[0]
            padding = (p, p, p, p)
        elif len(padding_dims) == 2:
            # Width x Height padding
            width_pad, height_pad = padding_dims
            padding = (width_pad, height_pad, width_pad, height_pad)
        elif len(padding_dims) == 4:
            # Left x Top x Right x Bottom padding
            padding = tuple(padding_dims)
        else:
            raise ValueError(
                'Padding value must be 1, 2, or 4 dimensional. '
                'Use format: "10" (uniform), "10x20" (width x height), '
                'or "5x10x5x15" (left x top x right x bottom)')

    return padding

def find_mask_bounds(
        img: PIL.Image.Image,
        padding: str | int | tuple[int, int] | tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """
    Find the bounding box of white pixels in the mask. If no bounding box can be found, return ``None``.

    :raise ValueError: If the padding value is specified incorrectly.

    :param img: The mask image (PIL Image)
    :param padding: Bounding box padding value, see: :py:func:`normalize_padding_value` for accepted values.
    :return: Tuple of (left, top, right, bottom) bounds, or ``None`` if no white pixels found.
    """

    # bit map grayscale
    if img.mode != 'L':
        img = img.convert('L')

    # Convert to numpy array
    mask_array = numpy.array(img)

    # Find coordinates of white pixels (assuming white is > 127)
    white_coords = numpy.where(mask_array > 127)

    if len(white_coords[0]) == 0:
        # No white pixels found
        return None

    # Get bounding box
    top = int(numpy.min(white_coords[0]))
    bottom = int(numpy.max(white_coords[0]))
    left = int(numpy.min(white_coords[1]))
    right = int(numpy.max(white_coords[1]))

    # Apply padding
    pad_left, pad_top, pad_right, pad_bottom = normalize_padding_value(padding)

    # final bounds with padding
    left = max(0, left - pad_left)
    top = max(0, top - pad_top)
    right = min(img.width, right + pad_right + 1)  # +1 because right bound is exclusive
    bottom = min(img.height, bottom + pad_bottom + 1)  # +1 because bottom bound is exclusive

    return left, top, right, bottom


def best_pil_resampling(old_size: _types.Size, new_size: _types.Size) -> PIL.Image.Resampling:
    """
    Auto-select the best PIL resampling setting for a resize operation.

    :param old_size: (tuple) Source image size (width, height).
    :param new_size: (tuple) Destination image size (width, height).
    :return: (PIL.Image.Resampling) Best resampling method.
    """
    scale_x = new_size[0] / old_size[0]
    scale_y = new_size[1] / old_size[1]
    scale_factor = min(scale_x, scale_y)  # Use the smallest scale factor

    if scale_factor > 1:
        # Upscaling
        return PIL.Image.Resampling.BICUBIC if scale_factor < 3 else PIL.Image.Resampling.LANCZOS
    elif scale_factor < 1:
        # Downscaling
        return PIL.Image.Resampling.LANCZOS
    else:
        # No scaling
        return PIL.Image.Resampling.NEAREST


def best_cv2_resampling(old_size: _types.Size, new_size: _types.Size) -> int:
    """
    Auto-select the best OpenCV resampling setting for a resize operation.

    :param old_size: (tuple) Source image shape (height, width, channels).
    :param new_size: (tuple) Destination image shape (height, width).
    :return: (int) Best OpenCV interpolation method.
    """
    scale_x = new_size[1] / old_size[1]
    scale_y = new_size[0] / old_size[0]
    scale_factor = min(scale_x, scale_y)  # Use the smallest scale factor

    if scale_factor > 1:
        # Upscaling
        return cv2.INTER_CUBIC if scale_factor < 3 else cv2.INTER_LANCZOS4
    elif scale_factor < 1:
        # Downscaling
        return cv2.INTER_AREA
    else:
        # No scaling
        return cv2.INTER_NEAREST


def cv2_resize_image(img: numpy.ndarray,
                     size: _types.OptionalSize,
                     aspect_correct: bool = False,
                     align: int | None = None,
                     algo: typing.Optional[int] = None):
    """
    Resize a :py:class:`numpy.ndarray` image and return a copy.

    This function always returns a copy even if the images size did not change.

    The new image dimension will always be forcefully aligned by ``align``,
    specifying ``None`` or ``1`` indicates no alignment.

    The filename attribute is preserved.

    :param img: the image to resize
    :param size: requested new size for the image, may be ``None``.
    :param aspect_correct: preserve aspect ratio?
    :param align: Force alignment by this amount of pixels.
    :param algo: cv2 resampling algorithm
    :return: the resized image
    """
    in_height = img.shape[0]
    in_width = img.shape[1]

    new_size = resize_image_calc(old_size=(in_width, in_height),
                                 new_size=size,
                                 aspect_correct=aspect_correct,
                                 align=align)
    if img.size == new_size:
        # probably less costly
        return numpy.copy(img)

    if algo is None:
        algo = best_cv2_resampling((in_width, in_height), new_size)

    r = cv2.resize(img, new_size, interpolation=algo)

    return r


def resize_image(img: PIL.Image.Image,
                 size: _types.OptionalSize,
                 aspect_correct: bool = False,
                 align: int | None = None,
                 algo: typing.Optional[PIL.Image.Resampling] = None):
    """
    Resize a :py:class:`PIL.Image.Image` and return a copy.

    This function always returns a copy even if the images size did not change.

    The new image dimension will always be forcefully aligned by ``align``,
    specifying ``None`` or ``1`` indicates no alignment.

    The filename attribute is preserved.

    :param img: the image to resize
    :param size: requested new size for the image, may be ``None``.
    :param aspect_correct: preserve aspect ratio?
    :param align: Force alignment by this amount of pixels.
    :param algo: Resampling algorithm
    :return: the resized image
    """
    new_size = resize_image_calc(old_size=img.size,
                                 new_size=size,
                                 aspect_correct=aspect_correct,
                                 align=align)
    if img.size == new_size:
        # probably less costly
        return copy_img(img)

    if algo is None:
        algo = best_pil_resampling(img.size, new_size)

    r = img.resize(new_size, algo)

    if hasattr(img, 'filename'):
        r.filename = img.filename

    return r


def paste_with_feather(
        background: PIL.Image.Image,
        foreground: PIL.Image.Image,
        location: tuple[int, int] | tuple[int, int, int, int] | list[int],
        feather: int = 30,
        shape: str = 'rectangle'
) -> PIL.Image.Image:
    """
    Composite an image onto a background with feathered (soft) edges.

    Creates smooth, blended transitions between foreground and background images
    by applying Gaussian blur to a mask, eliminating hard edges. The feathering
    effect is achieved by shrinking the mask and then blurring it.

    :param background: The background image to paste onto. Will be converted to RGBA mode.
    :param foreground: The foreground image to paste. Will be resized to fit the specified location.
    :param location: Specifies where to place the image. 2 elements (x, y) for top-left corner
                     offset using input_img original size, 4 elements (x0, y0, x1, y1) for
                     bounding box coordinates, or None to center with margin based on feather width.
    :param feather: The desired width of the feathered edge in pixels.
    :param shape: The shape of the mask. ``r`` / ``rect`` / ``rectangle`` for rectangular
        mask, ``c`` / ``circle`` / ``ellipse`` for circular.
    :returns: The composite image with feathered edges in the mode (channels) of the background image.
    :raises ValueError: If location is provided but doesn't contain 2 or 4 elements.
                        If shape is not recognized.
    """

    input_mode = background.mode
    background = background.convert("RGBA")

    inset = feather // 2
    blur_radius = max(1, feather / 6.0) if feather > 0 else 0

    margin = math.ceil(feather)

    if len(location) == 2:
        paste_offset = tuple(location)
        paste_size = foreground.size
    elif len(location) == 4:
        x0, y0, x1, y1 = location
        paste_offset = (x0, y0)
        paste_size = (x1 - x0, y1 - y0)
    else:
        raise ValueError("location must be a tuple/list of length 2 or 4")

    mask_size = (paste_size[0] + 2 * margin, paste_size[1] + 2 * margin)
    mask = PIL.Image.new("L", mask_size, 0)
    draw = PIL.ImageDraw.Draw(mask)

    mask_shape = _textprocessing.parse_basic_mask_shape(shape)

    if mask_shape == _textprocessing.BasicMaskShape.ELLIPSE:
        cx = mask_size[0] // 2
        cy = mask_size[1] // 2
        r = min(paste_size) // 2 - inset
        if r < 0:
            r = 0
        bbox = (cx - r, cy - r, cx + r, cy + r)
        draw.ellipse(bbox, fill=255)
    elif mask_shape == _textprocessing.BasicMaskShape.RECTANGLE:
        left = margin + inset
        top = margin + inset
        right = margin + paste_size[0] - inset
        bottom = margin + paste_size[1] - inset

        if left >= right or top >= bottom:
            left = top = right = bottom = 0
        draw.rectangle((left, top, right, bottom), fill=255)
    else:
        raise ValueError(f'Unknown mask shape: {shape}')

    if blur_radius > 0:
        blurred_mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=blur_radius))
    else:
        blurred_mask = mask

    input_rgba = foreground.convert("RGBA").resize(
        paste_size, PIL.Image.Resampling.LANCZOS
    )

    input_with_margin = PIL.Image.new("RGBA", mask_size, (0, 0, 0, 0))
    input_with_margin.paste(input_rgba, (margin, margin))

    if input_rgba.mode == 'RGBA':
        existing_alpha = input_rgba.split()[-1]
        alpha_with_margin = PIL.Image.new("L", mask_size, 0)
        alpha_with_margin.paste(existing_alpha, (margin, margin))

        combined_alpha = PIL.ImageChops.multiply(alpha_with_margin, blurred_mask)
        input_with_margin.putalpha(combined_alpha)
    else:
        input_with_margin.putalpha(blurred_mask)

    composite = background.copy()

    adjusted_offset = (paste_offset[0] - margin, paste_offset[1] - margin)

    composite.paste(input_with_margin, adjusted_offset, input_with_margin)

    return composite.convert(input_mode)


def letterbox_image(img: PIL.Image,
                    box_size: _types.Padding,
                    box_is_padding: bool = False,
                    box_color: str | int | float | tuple[int, int, int] | tuple[float, float, float] | None = None,
                    inner_size: _types.Size = None,
                    aspect_correct: bool = True):
    """
    Letterbox an image on to a colored background.

    :param img: The image to letterbox
    :param box_size: Size of the outer letterbox, or padding values.
        - If ``box_is_padding=False``:
            - (int) both width and height equal to this integer
            - (width, height) tuple for final letterbox size
        - If ``box_is_padding=True``: Can be either:
            - (padding) for uniform padding
            - (horizontal_padding, vertical_padding) for uniform padding
            - (left, top, right, bottom) for individual padding on each side
    :param box_is_padding: The ``box_size`` argument should be interpreted as padding?
    :param box_color: What color to use for the letter box background, the default is black.
        This should be specified as a HEX color code, or as a 3 tuple of integer or floating
        point RGB values, or as a single integer or float representing all color channels.
    :param inner_size: The size of the inner image
    :param aspect_correct: Should the size of the inner image be aspect correct?
    :return: The letterboxed image
    """
    if inner_size is None:
        inner_size = img.size

    if box_is_padding:
        if isinstance(box_size, int):
            # Single integer: uniform padding on all sides
            padding_left = padding_top = box_size
            final_box_size = (inner_size[0] + 2 * box_size,
                              inner_size[1] + 2 * box_size)
        elif len(box_size) == 2:
            # Two values: (horizontal_padding, vertical_padding)
            horizontal_padding, vertical_padding = box_size
            final_box_size = (inner_size[0] + 2 * horizontal_padding,
                              inner_size[1] + 2 * vertical_padding)
            padding_left = horizontal_padding
            padding_top = vertical_padding
        elif len(box_size) == 4:
            # Four values: (left, top, right, bottom)
            padding_left, padding_top, padding_right, padding_bottom = box_size
            final_box_size = (inner_size[0] + padding_left + padding_right,
                              inner_size[1] + padding_top + padding_bottom)
        else:
            raise ValueError("box_size must be an int, 2-tuple, or 4-tuple when box_is_padding=True")
    else:
        if isinstance(box_size, int):
            # Single integer: square letterbox
            final_box_size = (box_size, box_size)
        else:
            # Two values: (width, height)
            final_box_size = box_size
        # Calculate padding for centering when not in padding mode
        padding_left = (final_box_size[0] - inner_size[0]) // 2
        padding_top = (final_box_size[1] - inner_size[1]) // 2

    # Ensure inner_size fits within the final box
    inner_size = (min(inner_size[0], final_box_size[0]),
                  min(inner_size[1], final_box_size[1]))

    if box_color is None:
        box_color = 0

    letterbox = PIL.Image.new('RGB', final_box_size, box_color)

    img = resize_image(img, inner_size, aspect_correct=aspect_correct)

    if box_is_padding and (isinstance(box_size, int) or len(box_size) == 4):
        # Use the specific padding values for positioning
        x = padding_left
        y = padding_top
    else:
        # Center the image (original behavior for 2-tuple padding and non-padding mode)
        x = (final_box_size[0] - img.size[0]) // 2
        y = (final_box_size[1] - img.size[1]) // 2

    letterbox.paste(img, (x, y))

    if hasattr(img, 'filename'):
        letterbox.filename = img.filename

    return letterbox


def to_rgb(img: PIL.Image.Image):
    """
    Convert a :py:class:`PIL.Image.Image` to RGB format while preserving its filename attribute.

    :param img: the image
    :return: a converted copy of the image
    """
    c = img.convert('RGB')
    if hasattr(img, 'filename'):
        c.filename = img.filename
    return c


def get_filename(img: PIL.Image.Image):
    """
    Get the :py:attr:`PIL.Image.Image.filename` attribute or "NO_FILENAME" if it does not exist.

    :param img: :py:class:`PIL.Image.Image`
    :return: filename string or "NO_FILENAME"
    """

    if hasattr(img, 'filename'):
        return img.filename
    return 'NO_FILENAME'


def create_jpeg_exif_with_user_comment(comment: str) -> bytes:
    """
    Return JPEG EXIF data with a user comment field, this can be used with ``PIL.Image.save(img, exif=...)``.

    This function is specifically for saving JPEG files only.

    :return: EXIF data (bytes)
    """
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = \
        piexif.helper.UserComment.dump(
            comment, encoding="unicode"
        )
    return piexif.dump(exif_dict)


def read_jpeg_exif_user_comment(img: PIL.Image.Image) -> str | None:
    """
    Read the user comment field from a JPEG EXIF data, this can be used with ``PIL.Image.open(img)``.

    This function is specifically for JPEG files only.

    :param img: :py:class:`PIL.Image.Image`
    :return: user comment string or empty string if not found
    """
    if "exif" not in img.info:
        return None

    exif_dict = piexif.load(img.info["exif"])

    user_comment_raw = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)

    if user_comment_raw:
        user_comment = piexif.helper.UserComment.load(user_comment_raw)
        return user_comment
    else:
        return None
