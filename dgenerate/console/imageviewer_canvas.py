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

import platform as _std_platform
import tkinter as tk
import typing

import PIL.Image
import PIL.ImageDraw
import PIL.ImageTk
import cv2
import numpy as np

import dgenerate.console.mousewheelbind as _mousewheelbind
import dgenerate.console.helpdialog as _helpdialog


class ImageViewerCanvas(tk.Frame):
    """
    A zoomable image viewer widget with mouse wheel zoom, panning, 
    bounding box selection, and coordinate conversion functionality.
    """

    def __init__(self, parent, **kwargs):
        """
        Initialize ImageViewer with viewport cropping optimization for performance.
        """
        super().__init__(parent, **kwargs)

        # Platform detection for macOS-specific behavior
        self._is_macos = _std_platform.system() == 'Darwin'

        # Initialize zoom and pan state
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 80.0
        self._zoom_step = 1.2
        self._pan_x = 0
        self._pan_y = 0

        # Image state - store as numpy array for performance
        self._original_image_array = None  # numpy array
        self._original_image_size = None  # (width, height) tuple
        self._image_path = None
        self._cached_photo_img = None
        self._cached_zoom_level = None

        # Display state
        self._display_width = 0
        self._display_height = 0
        self._display_x_offset = 0
        self._display_y_offset = 0

        # Bounding box selection state
        self._bbox_selection_mode = False
        self._bbox_selection_coords_seperator = None  # 'csv' or 'x'
        self._bbox_start_coords = None
        self._bbox_end_coords = None
        self._bbox_widget_start = None  # Widget coordinates for overlay
        self._bbox_widget_current = None

        # Pan state - enhanced for cross-platform modifier+drag panning
        self._is_panning = False
        self._is_alt_panning = False  # For Cmd+drag on macOS, Ctrl+drag elsewhere
        self._last_pan_x = 0
        self._last_pan_y = 0

        # Performance optimization state
        self._pending_update = None
        self._crop_offset = (0, 0)  # Offset when using viewport cropping
        self._panning_needs_uncropped = False  # Flag to disable cropping during panning
        self._cached_uncropped_image = None  # Cache uncropped image for repeated panning
        self._cached_uncropped_zoom = None  # Zoom level of cached uncropped image

        self._bbox_line_width = 3

        # bounding box dash pattern (black portion)
        # 4 pixels on, 8 pixels off
        self._bbox_dash_pattern = (8, 4)

        # Callbacks
        self.on_coordinates_changed = None  # Callback for coordinate updates
        self.on_error = None  # Callback for error messages
        self.on_info = None  # Callback for info messages


        # UI

        self.configure(bg='black')

        # Create the canvas for image and overlay rendering
        self._canvas = tk.Canvas(self, bg='black', highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas items for drawing
        self._canvas_image_item = None
        self._canvas_bbox_items = []

        # Bind events to canvas
        self._canvas.bind('<Configure>', self._on_canvas_configure)

        _mousewheelbind.bind_mousewheel(self._canvas.bind, self._on_mouse_wheel)

        # Mouse events for bounding box selection
        self._canvas.bind('<Button-1>', self._on_left_click)
        self._canvas.bind('<B1-Motion>', self._on_left_drag)
        self._canvas.bind('<ButtonRelease-1>', self._on_left_release)

        # Mouse events for panning (middle mouse button)
        self._canvas.bind('<Button-2>', self._on_middle_click)
        self._canvas.bind('<B2-Motion>', self._on_middle_drag)
        self._canvas.bind('<ButtonRelease-2>', self._on_middle_release)

        # Note: Panning is now handled via modifier+drag (Cmd+drag on macOS, Ctrl+drag elsewhere)
        # This provides better cross-platform consistency and full 2D panning

        # Context menu
        self._canvas.bind('<Button-3>', self._on_right_click)

        # Key bindings (need to set focus for these to work)
        self._canvas.bind('<Key>', self._on_key_press)
        self._canvas.focus_set()

        # Enhanced keyboard shortcuts with macOS support
        self._setup_keyboard_shortcuts()

    def _setup_keyboard_shortcuts(self):
        """Setup platform-specific keyboard shortcuts"""
        # zoom + -
        self._canvas.bind('<Control-equal>', lambda e: self._zoom_by_factor(self._zoom_step))
        self._canvas.bind('<Control-minus>', lambda e: self._zoom_by_factor(1 / self._zoom_step))

        # macOS Command key equivalents
        if self._is_macos:
            self._canvas.bind('<Command-equal>', lambda e: self._zoom_by_factor(self._zoom_step))
            self._canvas.bind('<Command-minus>', lambda e: self._zoom_by_factor(1 / self._zoom_step))


    def _zoom_by_factor(self, factor):
        """Zoom by a specific factor"""
        if not self.has_image():
            return

        old_zoom = self._zoom_factor
        new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom_factor * factor))

        if new_zoom != old_zoom:
            self._zoom_factor = new_zoom
            # Clear uncropped cache since zoom level changed
            self._cached_uncropped_image = None
            self._cached_uncropped_zoom = None
            self._update_display()

    def _pan_by_offset(self, dx, dy):
        """Pan by a specific pixel offset"""
        if not self.has_image():
            return

        self._pan_x += dx
        self._pan_y += dy
        self._update_display()

    def _fast_resize_image(self, target_width: int, target_height: int) -> PIL.Image.Image:
        """Fast image resizing with viewport cropping optimization"""
        if not self.has_image():
            return None

        img_width, img_height = self._original_image_size
        canvas_width = self._canvas.winfo_width()
        canvas_height = self._canvas.winfo_height()

        # Use viewport cropping when zoomed in for better performance
        work_array = self._original_image_array
        work_width = target_width
        work_height = target_height
        self._crop_offset = (0, 0)

        # Apply viewport cropping for zoom levels > 1.0 (but not during panning)
        if self._zoom_factor > 1.0 and not self._panning_needs_uncropped:
            # Calculate what portion of the original image is actually visible
            visible_left = max(0, -self._display_x_offset) / self._zoom_factor
            visible_top = max(0, -self._display_y_offset) / self._zoom_factor
            visible_right = min(img_width, visible_left + canvas_width / self._zoom_factor)
            visible_bottom = min(img_height, visible_top + canvas_height / self._zoom_factor)

            # Convert to integer pixel coordinates with padding for smooth panning
            padding = 50  # Fixed padding in original image pixels
            crop_left = int(max(0, visible_left - padding))
            crop_top = int(max(0, visible_top - padding))
            crop_right = int(min(img_width, visible_right + padding))
            crop_bottom = int(min(img_height, visible_bottom + padding))

            # Only crop if we're viewing a significantly smaller portion
            crop_width = crop_right - crop_left
            crop_height = crop_bottom - crop_top

            if crop_width < img_width * 0.8 or crop_height < img_height * 0.8:
                # Crop the original array to the visible region
                work_array = self._original_image_array[crop_top:crop_bottom, crop_left:crop_right]

                # Calculate target size for the cropped region
                work_width = int(crop_width * self._zoom_factor)
                work_height = int(crop_height * self._zoom_factor)

                # Store crop offset for positioning
                self._crop_offset = (crop_left, crop_top)

        # Choose interpolation method based on scale factor
        scale_factor = work_width / work_array.shape[1]

        if scale_factor > 1:
            # Upscaling - use cubic for good quality/speed balance
            interpolation = cv2.INTER_CUBIC
        elif scale_factor < 1:
            # Downscaling - use area for best quality
            interpolation = cv2.INTER_AREA
        else:
            # No scaling needed
            interpolation = cv2.INTER_NEAREST

        # Perform the resize on the (potentially cropped) array
        resized_array = cv2.resize(work_array, (work_width, work_height), interpolation=interpolation)

        # Convert to PIL Image only for PhotoImage creation
        return PIL.Image.fromarray(resized_array)

    def load_image(self, image_path: str, fit: bool=False, view_state: typing.Optional[typing.Dict] = None):
        """Load an image from the given path and store as numpy array for performance"""
        try:
            # Clear any existing bounding box selection
            if self._bbox_selection_mode:
                self._cancel_bbox_selection()

            # Clear cache to avoid memory leaks and ensure fresh start
            self._cached_photo_img = None
            self._cached_zoom_level = None

            # Load image with PIL first for format support
            pil_image = PIL.Image.open(image_path)

            # Ensure image is in RGB mode for compatibility
            if pil_image.mode not in ['RGB', 'RGBA']:
                pil_image = pil_image.convert('RGB')

            # Convert to numpy array immediately - this is our primary storage format
            self._original_image_array = np.array(pil_image)
            self._original_image_size = (pil_image.width, pil_image.height)  # (width, height)

            # Close PIL image as we no longer need it
            pil_image.close()

            self._image_path = image_path

            # Reset zoom and pan (unless view_state will override)
            if view_state is None:
                self._zoom_factor = 1.0
                self._pan_x = 0
                self._pan_y = 0

            # Clear uncropped cache for new image
            self._cached_uncropped_image = None
            self._cached_uncropped_zoom = None

            # Auto-fit to viewing area if requested
            if fit:
                # Calculate zoom to fit before updating display
                canvas_width = self._canvas.winfo_width()
                canvas_height = self._canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    img_width, img_height = self._original_image_size
                    width_ratio = canvas_width / img_width
                    height_ratio = canvas_height / img_height
                    
                    self._zoom_factor = min(width_ratio, height_ratio, 1.0)  # Don't zoom in beyond 100%

            # Restore view state if provided
            if view_state is not None:
                self.set_view_state(view_state)

            # Update display
            self._update_display()

        except Exception as e:
            if self.on_error:
                self.on_error(f"Error loading image: {e}")

    def get_image_path(self) -> typing.Optional[str]:
        """Get the path of the currently loaded image"""
        return self._image_path

    def has_image(self) -> bool:
        """Check if an image is currently loaded"""
        return self._original_image_array is not None

    def _on_canvas_configure(self, event):
        """Handle canvas resize events - invalidate viewport cropping cache"""
        if not self.has_image():
            return

        # Canvas size changed, so any cached cropped image is now invalid
        # The viewport dimensions changed, requiring new crop calculations
        self._cached_photo_img = None
        self._cached_zoom_level = None
        self._crop_offset = (0, 0)

        # Regenerate display with new viewport
        self._update_display()

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel zoom with improved cross-platform support and fast OpenCV resizing"""
        if not self.has_image():
            return

        if self._bbox_selection_mode:
            self._end_bbox_selection()

        # Get mouse position relative to the canvas
        mouse_x = event.x
        mouse_y = event.y

        # Determine zoom direction with improved cross-platform compatibility
        zoom_in = False

        # Use platform-specific handling for reliable mouse wheel detection
        platform_name = _std_platform.system()

        if platform_name == 'Linux' and hasattr(event, 'num'):
            # Linux: Button-4 = scroll up = zoom in, Button-5 = scroll down = zoom out
            if event.num == 4:
                zoom_in = True
            elif event.num == 5:
                zoom_in = False
            else:
                # Unknown button number
                return
        elif hasattr(event, 'delta') and event.delta != 0:
            # Windows and macOS: positive delta = scroll up = zoom in
            # Also fallback for Linux systems that use delta
            zoom_in = event.delta > 0
        elif hasattr(event, 'num'):
            # Fallback for Linux if delta isn't available
            if event.num == 4:
                zoom_in = True
            elif event.num == 5:
                zoom_in = False
            else:
                return
        else:
            # Unhandled event type
            return

        # Calculate new zoom factor
        old_zoom = self._zoom_factor
        if zoom_in:
            new_zoom = min(self._max_zoom, self._zoom_factor * self._zoom_step)
        else:
            new_zoom = max(self._min_zoom, self._zoom_factor / self._zoom_step)

        if new_zoom == old_zoom:
            return

        # Convert mouse position to image coordinates before zoom
        image_x, image_y = self._widget_to_image_coordinates(mouse_x, mouse_y)
        if image_x is None or image_y is None:
            return

        # Update zoom factor
        self._zoom_factor = new_zoom
        # Clear uncropped cache since zoom level changed
        self._cached_uncropped_image = None
        self._cached_uncropped_zoom = None

        # Adjust pan to keep the same image point under the mouse cursor
        # Convert the same image coordinates back to widget coordinates after zoom
        self._update_display_metrics()
        new_widget_x, new_widget_y = self._image_to_widget_coordinates(image_x, image_y)

        if new_widget_x is not None and new_widget_y is not None:
            # Adjust pan to maintain cursor position
            self._pan_x += mouse_x - new_widget_x
            self._pan_y += mouse_y - new_widget_y

        # Update display
        self._update_display()

    def _on_left_click(self, event):
        """Handle left mouse click for bounding box selection or alternative panning on macOS"""
        if not self.has_image():
            return

        # Check for modifier+drag panning (works on all platforms)
        if ((self._is_macos and event.state & 0x8) or  # Command key (0x8) on macOS
                (not self._is_macos and event.state & 0x4)):  # Ctrl key (0x4) on Windows/Linux
            self._is_alt_panning = True
            self._last_pan_x = event.x
            self._last_pan_y = event.y
            self._canvas.configure(cursor="fleur")
            # Switch to uncropped image for smooth panning
            self._panning_needs_uncropped = True
            # Only clear cache if zoom level changed since last uncropped pan
            if self._cached_uncropped_zoom != self._zoom_factor:
                self._cached_photo_img = None  # Force uncropped image generation
                self._cached_zoom_level = None
            elif self._cached_uncropped_image is not None:
                # Reuse cached uncropped image from same zoom level
                self._cached_photo_img = self._cached_uncropped_image
                self._cached_zoom_level = self._zoom_factor
            return "break"

        # Regular bounding box selection behavior
        if not self._bbox_selection_mode:
            return

        # Convert to image coordinates
        image_x, image_y = self._widget_to_image_coordinates(event.x, event.y)
        if image_x is None or image_y is None:
            return

        # Store both image and widget coordinates
        self._bbox_start_coords = (image_x, image_y)
        self._bbox_end_coords = (image_x, image_y)
        self._bbox_widget_start = (event.x, event.y)
        self._bbox_widget_current = (event.x, event.y)

        # Start drawing the overlay
        self._draw_bbox_overlay()

        return "break"

    def _on_left_drag(self, event):
        """Handle left mouse drag for bounding box selection or alternative panning on macOS"""
        if not self.has_image():
            return

        # Handle alternative panning
        if self._is_alt_panning:
            # Calculate pan delta
            dx = event.x - self._last_pan_x
            dy = event.y - self._last_pan_y

            # Update pan position
            self._pan_x += dx
            self._pan_y += dy

            # Update last position
            self._last_pan_x = event.x
            self._last_pan_y = event.y

            # Update display
            self._update_display()
            return "break"

        # Regular bounding box selection behavior
        if not self._bbox_selection_mode or self._bbox_start_coords is None:
            return

        # Convert to image coordinates
        image_x, image_y = self._widget_to_image_coordinates(event.x, event.y)
        if image_x is None or image_y is None:
            return

        # Update both image and widget coordinates
        self._bbox_end_coords = (image_x, image_y)
        self._bbox_widget_current = (event.x, event.y)

        # Update the overlay
        self._draw_bbox_overlay()

        return "break"

    def _on_left_release(self, event):
        """Handle left mouse release for bounding box selection or alternative panning on macOS"""
        if not self.has_image():
            return

        # Handle alternative panning release
        if self._is_alt_panning:
            self._is_alt_panning = False
            self._canvas.configure(cursor="")
            # End panning mode but keep uncropped image cached
            self._panning_needs_uncropped = False
            # Don't clear cache here - keep uncropped image for future panning
            return "break"

        # Regular bounding box selection behavior
        if not self._bbox_selection_mode or self._bbox_start_coords is None:
            return

        # Convert to image coordinates
        image_x, image_y = self._widget_to_image_coordinates(event.x, event.y)
        if image_x is None or image_y is None:
            return

        # Update final coordinates
        self._bbox_end_coords = (image_x, image_y)
        self._bbox_widget_current = (event.x, event.y)

        # Complete the selection
        self._complete_bbox_selection()

        return "break"

    def _on_middle_click(self, event):
        """Handle middle mouse click for panning"""
        if not self.has_image():
            return

        self._is_panning = True
        self._last_pan_x = event.x
        self._last_pan_y = event.y
        self._canvas.configure(cursor="fleur")
        # Switch to uncropped image for smooth panning
        self._panning_needs_uncropped = True
        # Only clear cache if zoom level changed since last uncropped pan
        if self._cached_uncropped_zoom != self._zoom_factor:
            self._cached_photo_img = None  # Force uncropped image generation
            self._cached_zoom_level = None
        elif self._cached_uncropped_image is not None:
            # Reuse cached uncropped image from same zoom level
            self._cached_photo_img = self._cached_uncropped_image
            self._cached_zoom_level = self._zoom_factor

    def _on_middle_drag(self, event):
        """Handle middle mouse drag for panning"""
        if not self._is_panning or not self.has_image():
            return

        # Calculate pan delta
        dx = event.x - self._last_pan_x
        dy = event.y - self._last_pan_y

        # Update pan position
        self._pan_x += dx
        self._pan_y += dy

        # Update last position
        self._last_pan_x = event.x
        self._last_pan_y = event.y

        # Update display
        self._update_display()

    def _on_middle_release(self, event):
        """Handle middle mouse release for panning"""
        self._is_panning = False
        self._canvas.configure(cursor="")
        # End panning mode but keep uncropped image cached
        self._panning_needs_uncropped = False
        # Don't clear cache here - keep uncropped image for future panning

    def _on_right_click(self, event):
        """Handle right mouse click for context menu"""
        # This will be handled by the parent class
        pass

    def _on_key_press(self, event):
        """Handle key press events with macOS support"""
        if event.keysym == 'Escape':
            if self._bbox_selection_mode:
                self._cancel_bbox_selection()

    def request_help(self):
        """Show help information for macOS users"""
        if self.on_info:
            if self._is_macos:
                help_text = [
                    "Image Viewer Controls:\n",
                    "• Mouse wheel: Zoom in/out",
                    "• Cmd+drag: Pan image in any direction",
                    "• Middle click+drag: Alternative panning",
                    "• Cmd+/Cmd-: Zoom in/out",
                    "• Escape: Cancel bounding box selection"
                ]
            else:
                help_text = [
                    "Image Viewer Controls:\n",
                    "• Mouse wheel: Zoom in/out",
                    "• Ctrl+drag: Pan image in any direction",
                    "• Middle click+drag: Alternative panning",
                    "• Ctrl+/Ctrl-: Zoom in/out",
                    "• Escape: Cancel bounding box selection"
                ]

            _helpdialog.show_help_dialog(
                title='Image viewer help',
                help_text='\n'.join(help_text),
                parent=self.master,
                size=(400,300),
                position_widget=self.master,
                dock_to_right=False
            )

    def _update_display_metrics(self):
        """Update display metrics without redrawing"""
        if not self.has_image():
            return

        canvas_width = self._canvas.winfo_width()
        canvas_height = self._canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        image_width, image_height = self._original_image_size

        # Calculate display size based on zoom
        zoomed_width = int(image_width * self._zoom_factor)
        zoomed_height = int(image_height * self._zoom_factor)

        # Calculate centering offsets
        self._display_x_offset = (canvas_width - zoomed_width) // 2 + self._pan_x
        self._display_y_offset = (canvas_height - zoomed_height) // 2 + self._pan_y

        self._display_width = zoomed_width
        self._display_height = zoomed_height

    def _update_display(self):
        """Update the canvas display with optimized caching to eliminate lag"""
        if not self.has_image():
            self._canvas.delete("all")
            self._canvas_image_item = None
            self._canvas_bbox_items = []
            self._cached_photo_img = None
            self._cached_zoom_level = None
            return

        canvas_width = self._canvas.winfo_width()
        canvas_height = self._canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        self._update_display_metrics()

        try:
            # Only resize when zoom actually changes, not for pure panning
            cache_key = self._zoom_factor

            zoom_changed = (self._cached_zoom_level is None or
                            self._cached_zoom_level != cache_key)

            # Only call the expensive resize operation when zoom changes
            if zoom_changed or self._cached_photo_img is None:
                # Create the scaled image using viewport cropping + OpenCV
                target_width = self._display_width
                target_height = self._display_height

                if target_width > 0 and target_height > 0:
                    scaled_image = self._fast_resize_image(target_width, target_height)

                    self._cached_photo_img = PIL.ImageTk.PhotoImage(scaled_image)
                    self._cached_zoom_level = cache_key

                    # Cache uncropped image for repeated panning at this zoom level
                    if self._panning_needs_uncropped:
                        self._cached_uncropped_image = self._cached_photo_img
                        self._cached_uncropped_zoom = self._zoom_factor

                # Update or create the canvas image item
                if self._canvas_image_item is None:
                    # Create image item once
                    self._canvas_image_item = self._canvas.create_image(
                        0, 0, anchor="nw", image=self._cached_photo_img
                    )
                else:
                    # Efficiently update image content without recreating item
                    self._canvas.itemconfig(self._canvas_image_item, image=self._cached_photo_img)

            # Fast positioning update (no image processing)
            if self._canvas_image_item is not None:
                # Use calculated offsets for all zoom levels
                # Adjust position if we used viewport cropping
                if hasattr(self, '_crop_offset') and self._crop_offset != (0, 0):
                    crop_x, crop_y = self._crop_offset
                    adjusted_x = self._display_x_offset + int(crop_x * self._zoom_factor)
                    adjusted_y = self._display_y_offset + int(crop_y * self._zoom_factor)
                    self._canvas.coords(self._canvas_image_item, adjusted_x, adjusted_y)
                else:
                    self._canvas.coords(self._canvas_image_item,
                                        self._display_x_offset, self._display_y_offset)

            # Keep a reference to prevent garbage collection
            self._canvas.image = self._cached_photo_img

            # Clear and redraw bounding box overlay if active (fast operation)
            for item in self._canvas_bbox_items:
                self._canvas.delete(item)
            self._canvas_bbox_items = []

            if self._bbox_selection_mode and self._bbox_widget_start and self._bbox_widget_current:
                self._draw_bbox_overlay()

        except Exception as e:
            if self.on_error:
                self.on_error(f"Error updating display: {e}")

    def _widget_to_image_coordinates(self, widget_x: int, widget_y: int) -> typing.Tuple[
        typing.Optional[int], typing.Optional[int]]:
        """Convert widget coordinates to image coordinates"""
        if not self.has_image():
            return None, None

        # Check if click is within the displayed image bounds
        if (widget_x < self._display_x_offset or
                widget_x >= self._display_x_offset + self._display_width or
                widget_y < self._display_y_offset or
                widget_y >= self._display_y_offset + self._display_height):
            return None, None

        # Convert to image coordinates
        relative_x = widget_x - self._display_x_offset
        relative_y = widget_y - self._display_y_offset

        img_width, img_height = self._original_image_size
        image_x = int((relative_x / self._display_width) * img_width)
        image_y = int((relative_y / self._display_height) * img_height)

        # Ensure coordinates are within bounds
        image_x = max(0, min(image_x, img_width - 1))
        image_y = max(0, min(image_y, img_height - 1))

        return image_x, image_y

    def _image_to_widget_coordinates(self, image_x: int, image_y: int) -> typing.Tuple[
        typing.Optional[int], typing.Optional[int]]:
        """Convert image coordinates to widget coordinates"""
        if not self.has_image():
            return None, None

        # Convert image coordinates to widget coordinates
        img_width, img_height = self._original_image_size
        widget_x = int((image_x / img_width) * self._display_width) + self._display_x_offset
        widget_y = int((image_y / img_height) * self._display_height) + self._display_y_offset

        return widget_x, widget_y

    def get_coordinates_at_cursor(self, widget_x: int, widget_y: int) -> typing.Tuple[
        typing.Optional[int], typing.Optional[int]]:
        """Get image coordinates at the given widget position"""
        return self._widget_to_image_coordinates(widget_x, widget_y)

    def start_bbox_selection(self, seperator: str):
        """Start bounding box selection mode"""
        if not self.has_image():
            return

        self._bbox_selection_mode = True
        self._bbox_selection_coords_seperator = seperator
        self._bbox_start_coords = None
        self._bbox_end_coords = None

        if self.on_info:
            self.on_info(
                "Bounding box selection mode started. Left-click and drag to select area. Press Escape to cancel.")

    def _draw_dashed_line(self, x1, y1, x2, y2, fill, width, dash):
        import math

        if isinstance(dash, (list, tuple)) and len(dash) == 2:
            dash_length, gap_length = dash
        else:
            raise ValueError("Dash must be a tuple/list of (dash_length, gap_length)")

        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0:
            return []

        items = []
        step = dash_length + gap_length
        num_steps = int(dist // step)

        for i in range(num_steps + 1):
            start = i * step / dist
            end = min((i * step + dash_length) / dist, 1.0)

            sx = x1 + dx * start
            sy = y1 + dy * start
            ex = x1 + dx * end
            ey = y1 + dy * end

            items.append(self._canvas.create_line(
                sx, sy, ex, ey, fill=fill, width=width, capstyle='butt'))

        return items


    def _draw_bbox_overlay(self):
        """Draw crisp bounding box overlay on canvas with high-contrast outline effect"""
        if (not self._bbox_selection_mode or
                self._bbox_widget_start is None or
                self._bbox_widget_current is None):
            return

        # Clear existing bbox items
        for item in self._canvas_bbox_items:
            self._canvas.delete(item)
        self._canvas_bbox_items = []

        # Get widget coordinates
        x1, y1 = self._bbox_widget_start
        x2, y2 = self._bbox_widget_current

        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # First, draw white outline (continuous, slightly thicker for contrast)
        white_top = self._canvas.create_line(
            x1, y1, x2, y1,
            fill='white', width=self._bbox_line_width
        )
        white_right = self._canvas.create_line(
            x2, y1, x2, y2,
            fill='white',
            width=self._bbox_line_width
        )
        white_bottom = self._canvas.create_line(
            x2, y2, x1, y2,
            fill='white',
            width=self._bbox_line_width
        )
        white_left = self._canvas.create_line(
            x1, y2, x1, y1,
            fill='white',
            width=self._bbox_line_width
        )

        black_top = self._draw_dashed_line(
            x1, y1, x2, y1,
            fill='black',
            width=self._bbox_line_width,
            dash=self._bbox_dash_pattern
        )
        black_right = self._draw_dashed_line(
            x2, y1, x2, y2,
            fill='black',
            width=self._bbox_line_width,
            dash=self._bbox_dash_pattern
        )
        black_bottom = self._draw_dashed_line(
            x2, y2, x1, y2,
            fill='black',
            width=self._bbox_line_width,
            dash=self._bbox_dash_pattern
        )
        black_left = self._draw_dashed_line(
            x1, y2, x1, y1,
            fill='black',
            width=self._bbox_line_width,
            dash=self._bbox_dash_pattern
        )

        # Store canvas items for cleanup (white lines first, then black lines on top)
        self._canvas_bbox_items = [white_top, white_right, white_bottom, white_left,
                                   *black_top, *black_right, *black_bottom, *black_left]

    def _complete_bbox_selection(self):
        """Complete bounding box selection and copy to clipboard"""
        if (self._bbox_start_coords is None or
                self._bbox_end_coords is None or
                self._bbox_selection_coords_seperator is None):
            return

        x1, y1 = self._bbox_start_coords
        x2, y2 = self._bbox_end_coords

        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1


        # Format the bounding box coordinates
        bbox_text = self._bbox_selection_coords_seperator.join(str(c) for c in [x1,y1,x2,y2])

        # Copy to clipboard
        try:
            self.clipboard_clear()
            self.clipboard_append(bbox_text)
            if self.on_info:
                self.on_info(f"Bounding box copied to clipboard: {bbox_text}")
        except Exception as e:
            if self.on_error:
                self.on_error(f"Failed to copy bounding box to clipboard: {e}")

        # End selection
        self._end_bbox_selection()

    def _end_bbox_selection(self):
        """End bounding box selection mode"""
        self._bbox_selection_mode = False
        self._bbox_selection_coords_seperator = None
        self._bbox_start_coords = None
        self._bbox_end_coords = None
        self._bbox_widget_start = None
        self._bbox_widget_current = None

        # Clean up canvas overlay items
        for item in self._canvas_bbox_items:
            self._canvas.delete(item)
        self._canvas_bbox_items = []

    def _cancel_bbox_selection(self):
        """Cancel bounding box selection"""
        if self._bbox_selection_mode:
            if self.on_info:
                self.on_info("Bounding box selection cancelled.")
            self._end_bbox_selection()

    def copy_coordinates(self, widget_x: int, widget_y: int, separator: str):
        """Copy coordinates at the given position to clipboard"""
        image_x, image_y = self._widget_to_image_coordinates(widget_x, widget_y)
        if image_x is None or image_y is None:
            return

        coordinate_text = f"{image_x}{separator}{image_y}"

        try:
            self.clipboard_clear()
            self.clipboard_append(coordinate_text)
            if self.on_info:
                self.on_info(f"Coordinates copied to clipboard: {coordinate_text}")
        except Exception as e:
            if self.on_error:
                self.on_error(f"Failed to copy coordinates to clipboard: {e}")

    def copy_path(self):
        """Copy image path to clipboard"""
        if self._image_path is None:
            return

        try:
            import pathlib
            self.clipboard_clear()
            self.clipboard_append(pathlib.Path(self._image_path).as_posix())
            if self.on_info:
                self.on_info(f"Image path copied to clipboard: {self._image_path}")
        except Exception as e:
            if self.on_error:
                self.on_error(f"Failed to copy image path to clipboard: {e}")

    def reset_view(self):
        """Reset zoom and pan to default"""
        self._zoom_factor = 1.0
        self._pan_x = 0
        self._pan_y = 0
        # Clear uncropped cache since zoom level changed
        self._cached_uncropped_image = None
        self._cached_uncropped_zoom = None
        self._update_display()

    def zoom_to_fit(self):
        """Zoom to fit the image in the current view"""
        if not self.has_image():
            return

        canvas_width = self._canvas.winfo_width()
        canvas_height = self._canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Calculate zoom to fit
        img_width, img_height = self._original_image_size
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height

        self._zoom_factor = min(width_ratio, height_ratio, 1.0)  # Don't zoom in beyond 100%
        self._pan_x = 0
        self._pan_y = 0
        # Clear uncropped cache since zoom level changed
        self._cached_uncropped_image = None
        self._cached_uncropped_zoom = None

        self._update_display()

    def get_zoom_info(self) -> str:
        """Get current zoom information as a string"""
        if not self.has_image():
            return "No image"
        return f"{self._zoom_factor:.1%}"

    def get_view_state(self) -> typing.Optional[typing.Dict]:
        """Get current zoom and pan state"""
        if not self.has_image() or not self.winfo_viewable():
            return None
        return {
            'zoom_factor': self._zoom_factor,
            'pan_x': self._pan_x,
            'pan_y': self._pan_y
        }

    def set_view_state(self, view_state: typing.Dict):
        """Set zoom and pan state"""
        if not self.has_image() or not view_state:
            return

        self._zoom_factor = view_state.get('zoom_factor', 1.0)
        self._pan_x = view_state.get('pan_x', 0)
        self._pan_y = view_state.get('pan_y', 0)

        # Clear uncropped cache since zoom level may have changed
        self._cached_uncropped_image = None
        self._cached_uncropped_zoom = None

        self._update_display()

    def bind_event(self, event: str, callback):
        """Bind an event to the image viewer widget"""
        self._canvas.bind(event, callback)

    def unbind_event(self, event: str):
        """Unbind an event from the image viewer widget"""
        self._canvas.unbind(event)

    def cleanup(self):
        """Clean up resources"""
        # Clear image data (numpy arrays are garbage collected automatically)
        self._original_image_array = None
        self._original_image_size = None

        # Clear cache
        self._cached_photo_img = None
        self._cached_zoom_level = None
        self._cached_uncropped_image = None
        self._cached_uncropped_zoom = None

        # Clean up canvas
        self._canvas.delete("all")
        self._canvas_image_item = None
        self._canvas_bbox_items = []

        # End any active selection
        if self._bbox_selection_mode:
            self._end_bbox_selection() 