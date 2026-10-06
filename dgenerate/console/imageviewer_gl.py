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
import time
import typing

import PIL.Image
import numpy as np
import pyopengltk
import OpenGL.GL as gl

import dgenerate.console.animationplayback as _animationplayback
import dgenerate.console.mousewheelbind as _mousewheelbind
import dgenerate.console.helpdialog as _helpdialog


class ImageViewerGL(pyopengltk.OpenGLFrame):
    """
    Hardware-accelerated preview using OpenGL with pyopengltk.

    Still images, and finished animations (GIF, WebP, APNG, and MP4, including
    audio) with a timeline kept below the picture. The speaker draws sound
    waves, or a red X when muted, beside the volume slider. Used when
    ``DGENERATE_CONSOLE_UI_VULKAN=0``, on macOS by default, or when the
    Vulkan extra is not installed.
    """

    def __init__(self, parent, **kwargs):
        """Initialize OpenGL-accelerated ImageViewer"""
        super().__init__(parent, **kwargs)
        # pyopengltk clears the background so a picture is not painted over.
        # An empty preview has nothing to keep, and a sash drag smears unless
        # Tk fills it.
        self.configure(bg='black')

        # Platform detection for macOS-specific behavior
        self._is_macos = _std_platform.system() == 'Darwin'

        # Initialize zoom and pan state
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 80.0
        self._zoom_step = 1.2
        self._pan_x = 0
        self._pan_y = 0

        # Image state
        self._original_image_array = None  # numpy array
        self._original_image_size = None  # (width, height) tuple
        self._image_path = None
        self._gl_texture_id = None

        # Base display size - maintains fixed size for clipping behavior
        self._base_display_width = None
        self._base_display_height = None

        # OpenGL state
        self._shader_program = None
        self._overlay_shader_program = None
        self._vao = None
        self._overlay_vao = None
        self._vbo = None
        self._overlay_vbo = None
        self._ebo = None
        self._scale_uniform = None
        self._translate_uniform = None
        self._texture_uniform = None
        self._overlay_color_uniform = None
        self._overlay_viewport_uniform = None
        self._overlay_dash_size_uniform = None
        self._overlay_gap_size_uniform = None
        self._overlay_is_dashed_uniform = None
        self._gl_initialized = False

        # Pending operations system for handling calls before OpenGL is ready
        self._pending_load_operation = None  # (image_path, fit, view_state) tuple or None
        self._pending_view_state = None  # View state dict or None to apply after widget is configured

        # Bounding box selection state
        self._bbox_selection_mode = False
        self._bbox_selection_coords_seperator = None
        self._bbox_start_coords = None
        self._bbox_end_coords = None
        self._bbox_widget_start = None
        self._bbox_widget_current = None

        # Pan state
        self._is_panning = False
        self._is_alt_panning = False
        self._last_pan_x = 0
        self._last_pan_y = 0

        self._bbox_line_width = 3
        self._bbox_dash_pattern = (8, 4)

        # Callbacks
        self.on_coordinates_changed = None
        self.on_error = None
        self.on_info = None

        # Playback state. Set only while this pane is showing an animated file.
        # The picture is fitted above the timeline so the controls do not cover it.
        self._animation = None
        self._loop = True
        self._resize_after = None
        self._resize_drawn_at = 0.0
        self._animation_after = None
        self._uploaded_frame = None
        self._pointer_inside = False
        self._pointer_moved = 0.0
        self._scrubbing = False
        self._resume_after_scrub = False
        self._volume_dragging = False
        self._preview_volume, self._preview_muted = _animationplayback.load_preview_audio()

        # Bind events
        _mousewheelbind.bind_mousewheel(self.bind, self._on_mouse_wheel)
        self.bind('<Button-1>', self._on_left_click)
        self.bind('<B1-Motion>', self._on_left_drag)
        self.bind('<ButtonRelease-1>', self._on_left_release)
        self.bind('<Motion>', self._on_pointer_motion)
        self.bind('<Enter>', self._on_pointer_enter)
        self.bind('<Leave>', self._on_pointer_leave)
        self.bind('<Button-2>', self._on_middle_click)
        self.bind('<B2-Motion>', self._on_middle_drag)
        self.bind('<ButtonRelease-2>', self._on_middle_release)
        self.bind('<Key>', self._on_key_press)
        self.bind('<Configure>', self._on_configure)
        self.focus_set()

        self._setup_keyboard_shortcuts()

    def _setup_keyboard_shortcuts(self):
        """Setup platform-specific keyboard shortcuts"""
        self.bind('<Control-equal>', lambda e: self._zoom_by_factor(self._zoom_step))
        self.bind('<Control-minus>', lambda e: self._zoom_by_factor(1 / self._zoom_step))

        if self._is_macos:
            self.bind('<Command-equal>', lambda e: self._zoom_by_factor(self._zoom_step))
            self.bind('<Command-minus>', lambda e: self._zoom_by_factor(1 / self._zoom_step))

    def _sync_backdrop(self):
        """Fill an empty preview. A loaded picture keeps the last presented frame."""
        color = '' if self.has_image() else 'black'
        try:
            if self.cget('bg') != color:
                self.configure(bg=color)
        except tk.TclError:
            return

    def _on_configure(self, event):
        """Handle window resize events"""
        if not self.has_image():
            self._sync_backdrop()
            if self._resize_after is not None:
                try:
                    self.after_cancel(self._resize_after)
                except Exception:
                    pass
                self._resize_after = None
            if getattr(self, '_gl_initialized', False) and self.winfo_width() > 1 and self.winfo_height() > 1:
                self.redraw()
            return
        if not getattr(self, '_gl_initialized', False):
            return
        if self.winfo_width() <= 1 or self.winfo_height() <= 1:
            return
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except Exception:
                pass
        self._resize_after = self.after(32, self._finish_resize)
        now = time.monotonic()
        if now - self._resize_drawn_at < 0.032:
            return
        self._resize_drawn_at = now
        self._draw_for_resize()

    def _draw_for_resize(self):
        pending = self._resize_after
        self._resize_after = None
        try:
            if self._animation is not None:
                self._calculate_base_display_size()
            self.redraw()
        finally:
            if pending is not None and self._resize_after is None:
                self._resize_after = pending

    def _present_settled(self):
        if not self.has_image() or self._resize_after is not None:
            return
        self.redraw()

    def _finish_resize(self):
        self._resize_after = None
        self._resize_drawn_at = time.monotonic()
        if self.winfo_width() <= 1 or self.winfo_height() <= 1:
            return
        if self._animation is not None:
            self._calculate_base_display_size()
        self.redraw()

    def tkResize(self, width, height):
        """Handle resize event from pyopengltk"""
        if self._resize_after is not None:
            return
        # Only update viewport if OpenGL is initialized
        if hasattr(self, '_gl_initialized') and self._gl_initialized:
            if width > 0 and height > 0:
                try:
                    gl.glViewport(0, 0, width, height)

                    # Apply pending view state now that we have proper dimensions
                    if self._pending_view_state is not None:
                        view_state_to_apply = self._pending_view_state
                        self._pending_view_state = None
                        self.set_view_state(view_state_to_apply)
                        return  # Let set_view_state trigger its own redraw

                    self.redraw()
                except gl.GLError:
                    # OpenGL context might not be ready
                    pass

    def tkMap(self, evt):
        """Called on <Map>"""
        super().tkMap(evt)
        # Try to apply pending view state when widget becomes visible
        self._try_apply_pending_view_state()
        self.redraw()

    def tkExpose(self, evt):
        """Called on <Expose>"""
        if self._resize_after is not None:
            return
        # The base handler draws and then swaps again, which shows a cleared buffer.
        if not getattr(self, '_gl_initialized', False):
            return
        self._try_apply_pending_view_state()
        self.redraw()

    def initgl(self):
        """Initialize OpenGL resources"""
        # Enable blending for overlays
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        # Create shader program
        self._shader_program = self._create_shader_program()
        gl.glUseProgram(self._shader_program)

        # Get uniform locations
        self._scale_uniform = gl.glGetUniformLocation(self._shader_program, b'scale')
        self._translate_uniform = gl.glGetUniformLocation(self._shader_program, b'translate')
        self._texture_uniform = gl.glGetUniformLocation(self._shader_program, b'texture_sampler')

        # Create overlay shader program for bounding box
        self._overlay_shader_program = self._create_overlay_shader_program()
        self._overlay_color_uniform = gl.glGetUniformLocation(self._overlay_shader_program, b'line_color')
        self._overlay_viewport_uniform = gl.glGetUniformLocation(self._overlay_shader_program, b'viewport_size')
        self._overlay_dash_size_uniform = gl.glGetUniformLocation(self._overlay_shader_program, b'dash_size')
        self._overlay_gap_size_uniform = gl.glGetUniformLocation(self._overlay_shader_program, b'gap_size')
        self._overlay_is_dashed_uniform = gl.glGetUniformLocation(self._overlay_shader_program, b'is_dashed')

        # Create quad geometry
        vertices = np.array([
            # position (x, y), texture coords (u, v)
            -1.0, -1.0, 0.0, 1.0,  # bottom-left -> bottom of texture (flipped)
             1.0, -1.0, 1.0, 1.0,  # bottom-right -> bottom of texture (flipped)
             1.0,  1.0, 1.0, 0.0,  # top-right -> top of texture (flipped)
            -1.0,  1.0, 0.0, 0.0   # top-left -> top of texture (flipped)
        ], dtype=np.float32)

        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        # Create VAO and VBO
        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        self._vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        self._ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        # Set vertex attributes
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 4 * 4, None)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, 4 * 4, gl.GLvoidp(2 * 4))
        gl.glEnableVertexAttribArray(1)

        # Create overlay VAO/VBO for bounding box lines
        self._overlay_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._overlay_vao)

        self._overlay_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._overlay_vbo)
        # Pre-allocate buffer for 8 vertices (4 lines, 2 vertices each)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 8 * 2 * 4, None, gl.GL_DYNAMIC_DRAW)

        # Set vertex attributes for overlay (just position)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, None)
        gl.glEnableVertexAttribArray(0)

        # Unbind VAO
        gl.glBindVertexArray(0)

        # Mark OpenGL as initialized
        self._gl_initialized = True

        # Create texture for any image that was loaded before OpenGL was ready
        if self._original_image_array is not None:
            self._create_texture_from_array(self._original_image_array)

        # Process any pending operations
        if self._pending_load_operation is not None:
            image_path, fit, view_state = self._pending_load_operation
            self._pending_load_operation = None
            # Perform the load operation now that OpenGL is ready
            self._load_image_immediate(image_path, fit, view_state)

    def _create_shader_program(self):
        """Create and compile shader program"""
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec2 position;
        layout (location = 1) in vec2 tex_coord;
        
        uniform vec2 scale;
        uniform vec2 translate;
        
        out vec2 frag_tex_coord;
        
        void main() {
            // Simple 2D transformation: scale then translate
            vec2 transformed_pos = position * scale + translate;
            gl_Position = vec4(transformed_pos, 0.0, 1.0);
            frag_tex_coord = tex_coord;
        }
        """

        fragment_shader_source = """
        #version 330 core
        in vec2 frag_tex_coord;
        out vec4 frag_color;
        
        uniform sampler2D texture_sampler;
        
        void main() {
            frag_color = texture(texture_sampler, frag_tex_coord);
        }
        """

        def compile_shader(source, shader_type):
            shader = gl.glCreateShader(shader_type)
            gl.glShaderSource(shader, source)
            gl.glCompileShader(shader)

            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                error = gl.glGetShaderInfoLog(shader).decode()
                raise RuntimeError(f"Shader compilation failed: {error}")
            return shader

        vertex_shader = compile_shader(vertex_shader_source, gl.GL_VERTEX_SHADER)
        fragment_shader = compile_shader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)

        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            error = gl.glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Shader linking failed: {error}")

        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)

        return program

    @staticmethod
    def _create_overlay_shader_program():
        """Create shader program for overlay rendering (bounding box)"""
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec2 position;
        
        out vec2 screen_pos;
        
        uniform vec2 viewport_size;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            // Convert NDC to screen coordinates for dash pattern
            screen_pos = (position * 0.5 + 0.5) * viewport_size;
        }
        """

        fragment_shader_source = """
        #version 330 core
        in vec2 screen_pos;
        out vec4 frag_color;
        
        uniform vec4 line_color;
        uniform float dash_size;
        uniform float gap_size;
        uniform bool is_dashed;
        
        void main() {
            if (is_dashed) {
                // Calculate position along the line (using screen coordinates)
                float pattern_length = dash_size + gap_size;
                float pos = mod(length(screen_pos), pattern_length);
                
                // Discard fragment if in gap
                if (pos > dash_size) {
                    discard;
                }
            }
            
            frag_color = line_color;
        }
        """

        def compile_shader(source, shader_type):
            shader = gl.glCreateShader(shader_type)
            gl.glShaderSource(shader, source)
            gl.glCompileShader(shader)

            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                error = gl.glGetShaderInfoLog(shader).decode()
                raise RuntimeError(f"Shader compilation failed: {error}")
            return shader

        vertex_shader = compile_shader(vertex_shader_source, gl.GL_VERTEX_SHADER)
        fragment_shader = compile_shader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)

        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            error = gl.glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Shader linking failed: {error}")

        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)

        return program

    def redraw(self):
        """Redraw the OpenGL scene"""
        if self._resize_after is not None:
            return
        self._sync_backdrop()
        # Make sure OpenGL is initialized
        if not hasattr(self, '_gl_initialized') or not self._gl_initialized:
            return
        if self.winfo_ismapped():
            try:
                self.tkMakeCurrent()
            except Exception:
                return

        # Update viewport to match widget size - force this every time for PanedWindow compatibility
        width = self.winfo_width()
        height = self.winfo_height()
        if width > 0 and height > 0:
            try:
                # Always force viewport update to handle PanedWindow sizing issues
                gl.glViewport(0, 0, width, height)

                # Apply pending view state now that viewport is properly set up
                if self._pending_view_state is not None:
                    view_state_to_apply = self._pending_view_state
                    self._pending_view_state = None
                    self.set_view_state(view_state_to_apply)
                    # Early return to let set_view_state trigger its own redraw
                    return

            except gl.GLError:
                # OpenGL context might not be ready yet
                return
        else:
            # Widget doesn't have proper dimensions yet
            return

        if not self.has_image():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            self.tkSwapBuffers()
            return

        # Validate image data consistency before rendering
        if (self._original_image_array is not None and
            self._original_image_size is not None and
            len(self._original_image_array.shape) >= 2):
            array_height, array_width = self._original_image_array.shape[:2]
            size_width, size_height = self._original_image_size
            if array_width != size_width or array_height != size_height:
                if self.on_error:
                    self.on_error(f"Image viewer: Detected image data/size mismatch during redraw: array {array_width}x{array_height} vs size {size_width}x{size_height}")
                return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Keep the picture out of the control strip and the gap above it.
        reserve = self._control_reserve()
        scissor = reserve > 0 and height > reserve and width > 0
        if scissor:
            gl.glEnable(gl.GL_SCISSOR_TEST)
            gl.glScissor(0, reserve, width, height - reserve)

        # Validate shader program before use
        if self._shader_program is None:
            if scissor:
                gl.glDisable(gl.GL_SCISSOR_TEST)
            if self.on_error:
                self.on_error("Image viewer: Shader program is None, cannot render")
            self.tkSwapBuffers()
            return

        gl.glUseProgram(self._shader_program)

        # Calculate and set scale and translate uniforms
        scale, translate = self._calculate_transform()
        gl.glUniform2f(self._scale_uniform, scale[0], scale[1])
        gl.glUniform2f(self._translate_uniform, translate[0], translate[1])

        # Bind texture
        if self._gl_texture_id is not None:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._gl_texture_id)
            gl.glUniform1i(self._texture_uniform, 0)

        # Validate VAO before use
        if self._vao is None:
            if scissor:
                gl.glDisable(gl.GL_SCISSOR_TEST)
            if self.on_error:
                self.on_error("Image viewer: VAO is None, cannot render")
            self.tkSwapBuffers()
            return

        # Draw quad
        gl.glBindVertexArray(self._vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)

        if scissor:
            gl.glDisable(gl.GL_SCISSOR_TEST)

        # Draw bounding box overlay if in selection mode
        if self._bbox_selection_mode:
            self._draw_bbox_overlay()
        elif self._timeline_visible():
            self._draw_timeline()

        self.tkSwapBuffers()

    def _draw_bbox_overlay(self):
        """Draw bounding box overlay in screen space"""
        if (not self._bbox_selection_mode or
                self._bbox_widget_start is None or
                self._bbox_widget_current is None):
            return

        # Get widget dimensions
        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 0 or height <= 0:
            return

        # Get widget coordinates
        x1, y1 = self._bbox_widget_start
        x2, y2 = self._bbox_widget_current

        # Convert to normalized device coordinates (-1 to 1)
        # Note: Y is flipped in OpenGL
        ndc_x1 = (x1 / width) * 2.0 - 1.0
        ndc_y1 = -((y1 / height) * 2.0 - 1.0)
        ndc_x2 = (x2 / width) * 2.0 - 1.0
        ndc_y2 = -((y2 / height) * 2.0 - 1.0)

        # Create line vertices for the bounding box
        # Build dashed lines by creating segments
        vertices = []

        # Parameters for dash pattern
        dash_pixels = self._bbox_dash_pattern[0]
        gap_pixels = self._bbox_dash_pattern[1]

        # Helper function to add dashed line segments
        def add_dashed_line(x1, y1, x2, y2):
            # Calculate line length in pixels
            dx_pixels = abs(x2 - x1) * width / 2
            dy_pixels = abs(y2 - y1) * height / 2
            line_length = np.sqrt(dx_pixels**2 + dy_pixels**2)

            if line_length == 0:
                return

            # Calculate dash pattern
            pattern_length = dash_pixels + gap_pixels
            num_patterns = int(line_length / pattern_length)

            # Direction vector
            dx = x2 - x1
            dy = y2 - y1

            # Add segments
            for i in range(num_patterns + 1):
                start_t = i * pattern_length / line_length
                end_t = min((i * pattern_length + dash_pixels) / line_length, 1.0)

                if start_t >= 1.0:
                    break

                seg_x1 = x1 + dx * start_t
                seg_y1 = y1 + dy * start_t
                seg_x2 = x1 + dx * end_t
                seg_y2 = y1 + dy * end_t

                vertices.extend([seg_x1, seg_y1, seg_x2, seg_y2])

        # Add all four sides with dashed pattern
        add_dashed_line(ndc_x1, ndc_y1, ndc_x2, ndc_y1)  # Top
        add_dashed_line(ndc_x2, ndc_y1, ndc_x2, ndc_y2)  # Right
        add_dashed_line(ndc_x2, ndc_y2, ndc_x1, ndc_y2)  # Bottom
        add_dashed_line(ndc_x1, ndc_y2, ndc_x1, ndc_y1)  # Left

        if not vertices:
            return

        line_vertices = np.array(vertices, dtype=np.float32)
        num_vertices = len(vertices) // 2

        # Validate overlay resources before use
        if (self._overlay_shader_program is None or
            self._overlay_vao is None or
            self._overlay_vbo is None):
            if self.on_error:
                self.on_error("Image viewer: Overlay shader resources not available, cannot draw bounding box")
            return

        # Update the overlay VBO with the line vertices
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._overlay_vbo)
        # Resize buffer if needed
        if line_vertices.nbytes > 8 * 2 * 4:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, line_vertices.nbytes, line_vertices, gl.GL_DYNAMIC_DRAW)
        else:
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, line_vertices.nbytes, line_vertices)

        # Use overlay shader program
        gl.glUseProgram(self._overlay_shader_program)
        gl.glBindVertexArray(self._overlay_vao)

        # First, draw solid white lines (background, same width as black lines)
        # Create solid line vertices for white background
        solid_vertices = []
        # Add all four sides as solid lines
        solid_vertices.extend([ndc_x1, ndc_y1, ndc_x2, ndc_y1])  # Top
        solid_vertices.extend([ndc_x2, ndc_y1, ndc_x2, ndc_y2])  # Right
        solid_vertices.extend([ndc_x2, ndc_y2, ndc_x1, ndc_y2])  # Bottom
        solid_vertices.extend([ndc_x1, ndc_y2, ndc_x1, ndc_y1])  # Left

        solid_line_vertices = np.array(solid_vertices, dtype=np.float32)

        # Draw white solid background
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._overlay_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, solid_line_vertices.nbytes, solid_line_vertices)

        gl.glLineWidth(self._bbox_line_width)
        gl.glUniform4f(self._overlay_color_uniform, 1.0, 1.0, 1.0, 1.0)
        gl.glUniform2f(self._overlay_viewport_uniform, width, height)
        gl.glUniform1f(self._overlay_dash_size_uniform, 0)
        gl.glUniform1f(self._overlay_gap_size_uniform, 0)
        gl.glUniform1i(self._overlay_is_dashed_uniform, 0)
        gl.glDrawArrays(gl.GL_LINES, 0, 8)

        # Then draw black dashed lines on top (same width)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, line_vertices.nbytes, line_vertices)
        gl.glUniform4f(self._overlay_color_uniform, 0.0, 0.0, 0.0, 1.0)
        gl.glDrawArrays(gl.GL_LINES, 0, num_vertices)

        # Restore line width
        gl.glLineWidth(1.0)

    def _calculate_transform(self):
        """Calculate simple 2D scale and translate for image display"""
        if not self.has_image():
            return (1.0, 1.0), (0.0, 0.0)

        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 0 or height <= 0:
            return (1.0, 1.0), (0.0, 0.0)

        # Use base display size for clipping behavior (instead of fitting to current widget size)
        if self._base_display_width is None or self._base_display_height is None:
            # Fallback: calculate base display size if not set
            self._calculate_base_display_size()

        # Apply zoom to base display size
        display_width = self._base_display_width * self._zoom_factor
        display_height = self._base_display_height * self._zoom_factor

        # Convert to normalized device coordinates (-1 to 1).
        # The picture is centered in the area above the control bar, not the whole widget.
        scale_x = display_width / width
        scale_y = display_height / height
        left, top = self._placed_image_origin(display_width, display_height)
        center_x = left + display_width / 2
        center_y = top + display_height / 2
        translate_x = ((center_x - width / 2) * 2.0) / width
        translate_y = -((center_y - height / 2) * 2.0) / height

        return (scale_x, scale_y), (translate_x, translate_y)

    def _control_reserve(self) -> int:
        """Pixels at the bottom kept clear of the picture while an animation is open."""
        if self._animation is None:
            return 0
        return _animationplayback.BAR_HEIGHT + _animationplayback.CONTROL_GAP

    def _content_box(self) -> tuple[float, float, float, float]:
        """Widget rectangle the picture is fitted into. Y grows downward."""
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 0 or height <= 0:
            width, height = 800, 600
        content_height = max(1, height - self._control_reserve())
        return 0.0, 0.0, float(width), float(content_height)

    def _placed_image_origin(self, display_width: float, display_height: float) -> tuple[float, float]:
        x, y, content_width, content_height = self._content_box()
        left = x + (content_width - display_width) / 2 + self._pan_x
        top = y + (content_height - display_height) / 2 + self._pan_y
        return left, top

    def _calculate_base_display_size(self):
        """Calculate and store the base display size for the image"""
        if not self.has_image():
            return

        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 0 or height <= 0:
            # Use a default size if widget dimensions aren't available yet
            width = 800
            height = 600

        img_width, img_height = self._original_image_size
        _x, _y, content_width, content_height = self._content_box()
        # Fall back to the widget size when it has not been mapped yet.
        if self.winfo_width() <= 0 or self.winfo_height() <= 0:
            content_width, content_height = float(width), float(height)

        # Fit inside the picture area, leaving the control strip and its gap clear.
        widget_aspect = content_width / content_height
        image_aspect = img_width / img_height

        if image_aspect > widget_aspect:
            # Image is wider than the picture area - fit to width
            self._base_display_width = content_width
            self._base_display_height = content_width / image_aspect
        else:
            # Image is taller than the picture area - fit to height
            self._base_display_width = content_height * image_aspect
            self._base_display_height = content_height

    def _make_gl_current(self) -> bool:
        """Bind this viewer's GL context before texture uploads.

        A torn-off preview window is a second OpenGLFrame. Leaving that
        context current makes glTexSubImage2D see the wrong texture object
        and raise GL_INVALID_VALUE (1281) for an otherwise valid frame.
        """
        if not self._gl_initialized:
            return False
        if not self.winfo_ismapped():
            return False
        try:
            self.tkMakeCurrent()
        except Exception:
            return False
        return True

    def _create_texture_from_array(self, img_array):
        """Create OpenGL texture from numpy array"""
        try:
            if not self._make_gl_current():
                return

            # Query maximum texture size supported by GPU
            max_texture_size = gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)

            height, width = img_array.shape[:2]

            # Check if image dimensions exceed GPU limits
            if width > max_texture_size or height > max_texture_size:
                error_msg = (f"Image viewer: Image size ({width}x{height}) exceeds GPU maximum texture size "
                           f"({max_texture_size}x{max_texture_size}). Cannot load texture.")
                if self.on_error:
                    self.on_error(error_msg)
                return

            # Clean up existing texture first and set to None immediately
            if self._gl_texture_id is not None:
                gl.glDeleteTextures([self._gl_texture_id])
                self._gl_texture_id = None

            # Ensure image is in RGB format. Decoder frames can be row-padded
            # views; OpenGL reads them as tightly packed and then rejects the upload.
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                texture_data = np.ascontiguousarray(img_array)
                format = gl.GL_RGB
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                texture_data = np.ascontiguousarray(img_array)
                format = gl.GL_RGBA
            else:
                # Convert grayscale to RGB
                texture_data = np.ascontiguousarray(np.stack([img_array] * 3, axis=-1))
                format = gl.GL_RGB

            # No need to flip - texture coordinates are properly set

            # Generate new texture ID
            self._gl_texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._gl_texture_id)

            # Fix texture alignment issues for images with widths not divisible by 4
            # Set unpack alignment to 1 to handle arbitrary row widths correctly
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, 0)

            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, width, height, 0,
                          format, gl.GL_UNSIGNED_BYTE, texture_data)

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        except gl.GLError as e:
            # If texture creation failed, make sure ID is nullified
            if self._gl_texture_id is not None:
                try:
                    gl.glDeleteTextures([self._gl_texture_id])
                except:
                    pass  # Ignore errors during cleanup
                self._gl_texture_id = None
            if self.on_error:
                self.on_error(f"Image viewer: OpenGL error creating texture: {e}")
        except Exception as e:
            # If texture creation failed, make sure ID is nullified
            if self._gl_texture_id is not None:
                try:
                    gl.glDeleteTextures([self._gl_texture_id])
                except:
                    pass  # Ignore errors during cleanup
                self._gl_texture_id = None
            if self.on_error:
                self.on_error(f"Image viewer: Error creating texture: {e}")

    def load_image(self, image_path: str, fit: bool = False, view_state: typing.Optional[typing.Dict] = None):
        """Load an image and create OpenGL texture, storing pending operation if OpenGL not ready"""

        # immediately set even if not displayed yet
        self._image_path = image_path

        if self._gl_initialized:
            # OpenGL is ready, load immediately
            self._load_image_immediate(image_path, fit, view_state)
        else:
            # Store the operation for when OpenGL is ready
            self._pending_load_operation = (image_path, fit, view_state)



    def _load_image_immediate(self, image_path: str, fit: bool = False, view_state: typing.Optional[typing.Dict] = None):
        """Internal method to load an image immediately (OpenGL must be ready)"""
        try:
            self._stop_animation()
            # Clear ALL pending operations to prevent conflicts and race conditions
            self._pending_view_state = None
            self._pending_load_operation = None

            if self._bbox_selection_mode:
                self._cancel_bbox_selection()

            # Clean up any existing OpenGL texture to prevent conflicts
            if self._gl_texture_id is not None:
                gl.glDeleteTextures([self._gl_texture_id])
                self._gl_texture_id = None

            # Clear base display size to recalculate for new image
            self._base_display_width = None
            self._base_display_height = None

            clip = None
            if _animationplayback.is_animation_path(image_path):
                clip = _animationplayback.open_animation(image_path)
            if clip is not None:
                frame = clip.frame()
                if frame is None:
                    clip.close()
                    raise RuntimeError(f'Could not read a frame from "{image_path}"')
                self._animation = clip
                clip.loop = self._loop
                self._original_image_array = frame
                self._original_image_size = (frame.shape[1], frame.shape[0])
                self._uploaded_frame = frame
            else:
                # Load image with PIL
                pil_image = PIL.Image.open(image_path)
                if pil_image.mode not in ['RGB', 'RGBA']:
                    pil_image = pil_image.convert('RGB')

                # Convert to numpy array
                self._original_image_array = np.array(pil_image)
                self._original_image_size = (pil_image.width, pil_image.height)
                pil_image.close()

            # Validate that image data and size are consistent
            if len(self._original_image_array.shape) >= 2:
                array_height, array_width = self._original_image_array.shape[:2]
                size_width, size_height = self._original_image_size
                if array_width != size_width or array_height != size_height:
                    raise RuntimeError(f"Image data/size mismatch: array {array_width}x{array_height} vs size {size_width}x{size_height}")

            # Calculate base display size for clipping behavior
            self._calculate_base_display_size()

            # Apply view state immediately if provided to prevent visual flashing
            if view_state is not None:
                self._zoom_factor = view_state.get('zoom_factor', 1.0)
                self._pan_x = view_state.get('pan_x', 0)
                self._pan_y = view_state.get('pan_y', 0)
            else:
                # Reset zoom and pan to defaults only if no view_state provided
                self._zoom_factor = 1.0
                self._pan_x = 0
                self._pan_y = 0

            # Auto-fit if requested (but only if no view_state provided)
            if fit and view_state is None:
                self._fit_to_widget()

            # Force widget to get proper dimensions, especially important for PanedWindows
            self.update_idletasks()
            self.update()  # Force immediate processing of pending events

            # Ensure OpenGL viewport is properly set up with current widget dimensions
            width = self.winfo_width()
            height = self.winfo_height()
            if width > 1 and height > 1:
                try:
                    gl.glViewport(0, 0, width, height)
                except gl.GLError:
                    pass

            # Create OpenGL texture (we know OpenGL is initialized here)
            self._create_texture_from_array(self._original_image_array)

            # If view state wasn't applied above due to widget sizing, store for later
            if view_state is not None and (width <= 1 or height <= 1):
                # Widget not ready, store for later - will be applied when widget becomes visible
                self._pending_view_state = view_state

            # Trigger redraw
            self.redraw()
            # The output pane paints after this returns and can cover one
            # buffer swap. Present again once that layout has settled.
            self.after_idle(self._present_settled)
            self.after(50, self._present_settled)

            if self._animation is not None:
                if self._animation.has_audio and not self._animation.audio_active and self.on_info:
                    detail = self._animation.audio_error or 'no playback device is available'
                    self.on_info(f'Image viewer: audio will not play ({detail}).')
                self._animation.set_gain(self._preview_gain())
                self._animation.start()
                self._schedule_animation()

        except Exception as e:
            self._stop_animation()
            if self.on_error:
                self.on_error(f"Image viewer: Error loading image: {e}")

    def _try_apply_pending_view_state(self):
        """Attempt to apply pending view state if the widget is ready"""
        if self._pending_view_state is None:
            return

        # Check if widget has proper dimensions now
        width = self.winfo_width()
        height = self.winfo_height()
        if width > 1 and height > 1:
            # Force viewport setup before applying view state
            try:
                gl.glViewport(0, 0, width, height)
            except gl.GLError:
                pass

            view_state = self._pending_view_state
            self._pending_view_state = None
            self.set_view_state(view_state)

    def _fit_to_widget(self):
        """Calculate zoom to fit image in widget"""
        # Recalculate base display size based on current widget dimensions
        self._calculate_base_display_size()
        # Reset zoom factor since base size now fits the widget
        self._zoom_factor = 1.0

    def _widget_to_image_coordinates(self, widget_x: int, widget_y: int) -> typing.Tuple[typing.Optional[int], typing.Optional[int]]:
        """Convert widget coordinates to image coordinates"""
        if not self.has_image():
            return None, None

        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 0 or height <= 0:
            return None, None

        img_width, img_height = self._original_image_size

        # Use base display size with zoom applied
        if self._base_display_width is None or self._base_display_height is None:
            self._calculate_base_display_size()

        display_width = self._base_display_width * self._zoom_factor
        display_height = self._base_display_height * self._zoom_factor

        # Calculate image position on screen (centered in the picture area + pan offset)
        image_left, image_top = self._placed_image_origin(display_width, display_height)
        image_right = image_left + display_width
        image_bottom = image_top + display_height

        # Check if click is within the displayed image bounds
        if (widget_x < image_left or widget_x > image_right or
            widget_y < image_top or widget_y > image_bottom):
            return None, None

        # Convert to image coordinates
        relative_x = (widget_x - image_left) / display_width
        relative_y = (widget_y - image_top) / display_height

        image_x = int(relative_x * img_width)
        image_y = int(relative_y * img_height)

        # Clamp to image bounds
        image_x = max(0, min(image_x, img_width - 1))
        image_y = max(0, min(image_y, img_height - 1))

        return image_x, image_y

    def _image_to_widget_coordinates(self, image_x: int, image_y: int) -> typing.Tuple[typing.Optional[int], typing.Optional[int]]:
        """Convert image coordinates to widget coordinates"""
        if not self.has_image():
            return None, None

        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 0 or height <= 0:
            return None, None

        img_width, img_height = self._original_image_size

        # Use base display size with zoom applied
        if self._base_display_width is None or self._base_display_height is None:
            self._calculate_base_display_size()

        display_width = self._base_display_width * self._zoom_factor
        display_height = self._base_display_height * self._zoom_factor

        # Calculate image position on screen (centered in the picture area + pan offset)
        image_left, image_top = self._placed_image_origin(display_width, display_height)

        # Convert image coordinates to relative coordinates (0.0 to 1.0)
        relative_x = image_x / img_width
        relative_y = image_y / img_height

        # Convert to widget coordinates
        widget_x = int(image_left + relative_x * display_width)
        widget_y = int(image_top + relative_y * display_height)

        return widget_x, widget_y

    def _zoom_by_factor(self, factor):
        """Zoom by a specific factor"""
        if not self.has_image():
            return

        old_zoom = self._zoom_factor
        new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom_factor * factor))

        if new_zoom != old_zoom:
            self._zoom_factor = new_zoom
            self.redraw()

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel zoom"""
        if not self.has_image():
            return

        if self._bbox_selection_mode:
            self._end_bbox_selection()

        # Get mouse position
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

        # Calculate new zoom
        old_zoom = self._zoom_factor
        if zoom_in:
            new_zoom = min(self._max_zoom, self._zoom_factor * self._zoom_step)
        else:
            new_zoom = max(self._min_zoom, self._zoom_factor / self._zoom_step)

        if new_zoom == old_zoom:
            return

        # Get image coordinates before zoom
        image_x, image_y = self._widget_to_image_coordinates(mouse_x, mouse_y)
        if image_x is None or image_y is None:
            return

        # Update zoom
        self._zoom_factor = new_zoom

        # Adjust pan to keep same image point under cursor
        new_widget_x, new_widget_y = self._image_to_widget_coordinates(image_x, image_y)
        if new_widget_x is not None and new_widget_y is not None:
            self._pan_x += mouse_x - new_widget_x
            self._pan_y += mouse_y - new_widget_y

        self.redraw()

    def _on_left_click(self, event):
        """Handle left mouse click"""
        if not self.has_image():
            return

        if self._animation is not None and not self._bbox_selection_mode and self._timeline_visible():
            hit = _animationplayback.control_hit(event.x, event.y, self.winfo_width(), self.winfo_height())
            if hit == 'play':
                self._animation.toggle()
                self._schedule_animation()
                self.redraw()
                return "break"
            if hit == 'loop':
                self._loop = not self._loop
                self._animation.loop = self._loop
                self.redraw()
                return "break"
            if hit == 'mute':
                self._preview_muted = not self._preview_muted
                self._push_preview_gain()
                _animationplayback.save_preview_audio(self._preview_volume, self._preview_muted)
                self.redraw()
                return "break"
            if hit == 'volume':
                self._volume_dragging = True
                self._set_volume_from_x(event.x)
                return "break"
            if hit == 'track':
                self._scrubbing = True
                self._resume_after_scrub = self._animation.playing
                if self._animation.playing:
                    self._animation.pause()
                self._scrub_to(event.x)
                return "break"

        # Check for modifier+drag panning
        if ((self._is_macos and event.state & 0x8) or
                (not self._is_macos and event.state & 0x4)):
            self._is_alt_panning = True
            self._last_pan_x = event.x
            self._last_pan_y = event.y
            self.configure(cursor="fleur")
            return "break"

        # Bounding box selection
        if not self._bbox_selection_mode:
            return

        image_x, image_y = self._widget_to_image_coordinates(event.x, event.y)
        if image_x is None or image_y is None:
            return

        self._bbox_start_coords = (image_x, image_y)
        self._bbox_end_coords = (image_x, image_y)
        self._bbox_widget_start = (event.x, event.y)
        self._bbox_widget_current = (event.x, event.y)

        return "break"

    def _on_left_drag(self, event):
        """Handle left mouse drag"""
        if self._volume_dragging:
            self._set_volume_from_x(event.x)
            return "break"

        if self._scrubbing:
            self._scrub_to(event.x)
            return "break"

        if not self.has_image():
            return

        if self._is_alt_panning:
            dx = event.x - self._last_pan_x
            dy = event.y - self._last_pan_y
            self._pan_x += dx
            self._pan_y += dy
            self._last_pan_x = event.x
            self._last_pan_y = event.y
            self.redraw()
            return "break"

        if not self._bbox_selection_mode or self._bbox_start_coords is None:
            return

        image_x, image_y = self._widget_to_image_coordinates(event.x, event.y)
        if image_x is None or image_y is None:
            return

        self._bbox_end_coords = (image_x, image_y)
        self._bbox_widget_current = (event.x, event.y)
        self.redraw()

        return "break"

    def _on_left_release(self, event):
        """Handle left mouse release"""
        if self._volume_dragging:
            self._set_volume_from_x(event.x)
            self._volume_dragging = False
            _animationplayback.save_preview_audio(self._preview_volume, self._preview_muted)
            self.redraw()
            return "break"

        if self._scrubbing:
            self._scrub_to(event.x)
            self._scrubbing = False
            if self._resume_after_scrub and self._animation is not None:
                self._animation.toggle()
            self._resume_after_scrub = False
            self._schedule_animation()
            self.redraw()
            return "break"

        if not self.has_image():
            return

        if self._is_alt_panning:
            self._is_alt_panning = False
            self.configure(cursor="")
            return "break"

        if not self._bbox_selection_mode or self._bbox_start_coords is None:
            return

        image_x, image_y = self._widget_to_image_coordinates(event.x, event.y)
        if image_x is None or image_y is None:
            return

        self._bbox_end_coords = (image_x, image_y)
        self._bbox_widget_current = (event.x, event.y)
        self._complete_bbox_selection()

        return "break"

    def _on_middle_click(self, event):
        """Handle middle mouse click for panning"""
        if not self.has_image():
            return

        self._is_panning = True
        self._last_pan_x = event.x
        self._last_pan_y = event.y
        self.configure(cursor="fleur")

    def _on_middle_drag(self, event):
        """Handle middle mouse drag for panning"""
        if not self._is_panning or not self.has_image():
            return

        dx = event.x - self._last_pan_x
        dy = event.y - self._last_pan_y
        self._pan_x += dx
        self._pan_y += dy
        self._last_pan_x = event.x
        self._last_pan_y = event.y
        self.redraw()

    def _on_middle_release(self, event):
        """Handle middle mouse release for panning"""
        self._is_panning = False
        self.configure(cursor="")

    def _on_key_press(self, event):
        """Handle key press events"""
        if event.keysym == 'Escape':
            if self._bbox_selection_mode:
                self._cancel_bbox_selection()
        elif event.keysym == 'space' and self._animation is not None:
            self._animation.toggle()
            self._schedule_animation()
            self.redraw()
            return "break"

    def bind_event(self, event: str, callback):
        """Bind an event to the image viewer widget"""
        self.bind(event, callback)

    def unbind_event(self, event: str):
        """Unbind an event from the image viewer widget"""
        self.unbind(event)

    def get_image_path(self) -> typing.Optional[str]:
        """Get the path of the currently loaded image"""
        return self._image_path

    def has_image(self) -> bool:
        """Check if an image is currently loaded"""
        return self._original_image_array is not None

    def get_coordinates_at_cursor(self, widget_x: int, widget_y: int) -> typing.Tuple[typing.Optional[int], typing.Optional[int]]:
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
            self.on_info("Bounding box selection mode started. Left-click and drag to select area. Press Escape to cancel.")

    def _complete_bbox_selection(self):
        """Complete bounding box selection and copy to clipboard"""
        if (self._bbox_start_coords is None or
                self._bbox_end_coords is None or
                self._bbox_selection_coords_seperator is None):
            return

        x1, y1 = self._bbox_start_coords
        x2, y2 = self._bbox_end_coords

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        bbox_text = self._bbox_selection_coords_seperator.join(str(c) for c in [x1, y1, x2, y2])

        try:
            self.clipboard_clear()
            self.clipboard_append(bbox_text)
            if self.on_info:
                self.on_info(f"Bounding box copied to clipboard: {bbox_text}")
        except Exception as e:
            if self.on_error:
                self.on_error(f"Image viewer: Failed to copy bounding box to clipboard: {e}")

        self._end_bbox_selection()

    def _end_bbox_selection(self):
        """End bounding box selection mode"""
        self._bbox_selection_mode = False
        self._bbox_selection_coords_seperator = None
        self._bbox_start_coords = None
        self._bbox_end_coords = None
        self._bbox_widget_start = None
        self._bbox_widget_current = None
        self.redraw()

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
                self.on_error(f"Image viewer: Failed to copy coordinates to clipboard: {e}")

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
                self.on_error(f"Image viewer: Failed to copy image path to clipboard: {e}")

    def reset_view(self):
        """Reset to show image at actual pixel size (1:1 zoom)"""
        if not self.has_image():
            return

        # Recalculate base display size for current widget dimensions
        self._calculate_base_display_size()

        # Calculate zoom factor to show image at actual pixel size
        img_width, img_height = self._original_image_size

        # Set zoom to show actual image size relative to base display size
        if self._base_display_width > 0:
            # Calculate zoom needed to show image at actual pixel size
            # Since base display size maintains aspect ratio, we can use either dimension
            self._zoom_factor = img_width / self._base_display_width
        else:
            self._zoom_factor = 1.0

        self._pan_x = 0
        self._pan_y = 0
        self.redraw()

    def zoom_to_fit(self):
        """Zoom to fit the image in the current view"""
        if not self.has_image():
            return

        # Recalculate base display size and reset view
        self._calculate_base_display_size()
        self._zoom_factor = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self.redraw()

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
        self.redraw()

    def request_help(self):
        """Show help information"""
        if self.on_info:
            if self._is_macos:
                help_text = [
                    "Image Viewer Controls:\n",
                    "• Mouse wheel: Zoom in/out",
                    "• Cmd+drag: Pan image in any direction",
                    "• Middle click+drag: Alternative panning",
                    "• Cmd+/Cmd-: Zoom in/out",
                    "• Escape: Cancel bounding box selection",
                    "• Animations: move the pointer over the picture for the timeline",
                    "• Space, or the play button: pause and resume",
                    "• Loop arrow: replay the clip, or stop at the end when slashed",
                    "• Speaker: mute (red X) or unmute; slider sets volume. Both are remembered"
                ]
            else:
                help_text = [
                    "Image Viewer Controls:\n",
                    "• Mouse wheel: Zoom in/out",
                    "• Ctrl+drag: Pan image in any direction",
                    "• Middle click+drag: Alternative panning",
                    "• Ctrl+/Ctrl-: Zoom in/out",
                    "• Escape: Cancel bounding box selection",
                    "• Animations: move the pointer over the picture for the timeline",
                    "• Space, or the play button: pause and resume",
                    "• Loop arrow: replay the clip, or stop at the end when slashed",
                    "• Speaker: mute (red X) or unmute; slider sets volume. Both are remembered"
                ]

            _helpdialog.show_help_dialog(
                title='Image viewer help',
                help_text='\n'.join(help_text),
                parent=self.master,
                size=(460, 360),
                position_widget=self.master,
                dock_to_right=False
            )

    def _timeline_visible(self) -> bool:
        if self._animation is None or self._bbox_selection_mode:
            return False
        if self._scrubbing or self._volume_dragging:
            return True
        if not self._pointer_inside:
            return False
        if not self._animation.playing:
            return True
        return (time.perf_counter() - self._pointer_moved) < 2.5

    def _on_pointer_enter(self, event):
        self._pointer_inside = True
        self._pointer_moved = time.perf_counter()
        if self._animation is not None:
            self._schedule_animation()
            self.redraw()

    def _on_pointer_leave(self, event):
        self._pointer_inside = False
        if not self._scrubbing:
            self.redraw()

    def _on_pointer_motion(self, event):
        self._pointer_inside = True
        self._pointer_moved = time.perf_counter()
        if self._animation is not None and not self._animation.playing:
            self._schedule_animation()
        if self._timeline_visible():
            self.redraw()

    def _preview_gain(self) -> float:
        if self._preview_muted:
            return 0.0
        return self._preview_volume

    def _push_preview_gain(self):
        if self._animation is not None:
            self._animation.set_gain(self._preview_gain())

    def _set_volume_from_x(self, widget_x: int):
        self._preview_volume = _animationplayback.volume_on_slider(
            widget_x, self.winfo_width(), self.winfo_height())
        if self._preview_volume > 0:
            self._preview_muted = False
        self._push_preview_gain()
        self.redraw()

    def _scrub_to(self, widget_x: int):
        clip = self._animation
        if clip is None:
            return
        seconds = _animationplayback.time_on_track(
            widget_x, self.winfo_width(), self.winfo_height(), clip.duration)
        clip.seek(seconds)
        frame = clip.frame()
        if frame is not None:
            self._replace_animation_frame(frame)
        self.redraw()

    def _schedule_animation(self):
        if self._animation is None:
            return
        if self._animation_after is not None:
            return
        delay = _animationplayback.playback_tick_delay_ms(
            self._animation,
            self._timeline_visible() or self._scrubbing or self._volume_dragging,
        )
        self._animation_after = self.after(delay, self._animation_tick)

    def _animation_tick(self):
        self._animation_after = None
        clip = self._animation
        if clip is None:
            return
        try:
            frame = clip.frame()
            if frame is not None and frame is not self._uploaded_frame:
                self._replace_animation_frame(frame)
            self.redraw()
        finally:
            if clip.playing or self._pointer_inside or self._scrubbing or self._volume_dragging:
                self._schedule_animation()

    def _replace_animation_frame(self, frame: np.ndarray):
        if not self._gl_initialized or self._gl_texture_id is None:
            self._original_image_array = frame
            self._uploaded_frame = frame
            return
        if not self._make_gl_current():
            self._original_image_array = frame
            self._uploaded_frame = frame
            return
        height, width = frame.shape[:2]
        current_height, current_width = self._original_image_array.shape[:2]
        if (width, height) != (current_width, current_height) or frame.shape[2] != self._original_image_array.shape[2]:
            self._original_image_array = frame
            self._original_image_size = (width, height)
            self._create_texture_from_array(frame)
        else:
            packed = np.ascontiguousarray(frame)
            try:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._gl_texture_id)
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, 0)
                pixel_format = gl.GL_RGBA if packed.shape[2] == 4 else gl.GL_RGB
                gl.glTexSubImage2D(
                    gl.GL_TEXTURE_2D, 0, 0, 0, width, height,
                    pixel_format, gl.GL_UNSIGNED_BYTE, packed)
            except gl.GLError:
                # Context/texture mismatch or a stale id: rebuild once.
                self._create_texture_from_array(frame)
            self._original_image_array = frame
        self._uploaded_frame = frame

    def _stop_animation(self):
        if self._animation_after is not None:
            try:
                self.after_cancel(self._animation_after)
            except Exception:
                pass
            self._animation_after = None
        clip = self._animation
        self._animation = None
        self._uploaded_frame = None
        self._scrubbing = False
        self._resume_after_scrub = False
        self._volume_dragging = False
        if clip is not None:
            clip.close()

    def _draw_timeline(self):
        clip = self._animation
        layout = _animationplayback.control_layout(self.winfo_width(), self.winfo_height())
        if clip is None or layout is None or self._overlay_shader_program is None:
            return
        width = self.winfo_width()
        height = self.winfo_height()
        bar = layout['bar']
        self._fill_rect(bar[0], bar[1], bar[2], bar[3], width, height, (0.0, 0.0, 0.0, 0.72))
        track = layout['track']
        self._fill_rect(track[0], track[1], track[2], track[3], width, height, (1.0, 1.0, 1.0, 0.35))
        duration = clip.duration if clip.duration > 0 else 1.0
        fraction = min(1.0, max(0.0, clip.time() / duration))
        played_right = track[0] + (track[2] - track[0]) * fraction
        self._fill_rect(track[0], track[1], played_right, track[3], width, height, (0.9, 0.15, 0.15, 1.0))
        knob = 7
        self._fill_rect(
            played_right - knob / 2, track[1] - 4, played_right + knob / 2, track[3] + 4,
            width, height, (1.0, 1.0, 1.0, 1.0))
        play = layout['play']
        if clip.playing:
            gap = 4
            bar_w = 4
            mid = (play[0] + play[2]) / 2
            self._fill_rect(mid - gap - bar_w, play[1], mid - gap, play[3], width, height, (1, 1, 1, 1))
            self._fill_rect(mid + gap, play[1], mid + gap + bar_w, play[3], width, height, (1, 1, 1, 1))
        else:
            self._fill_triangle(
                [(play[0] + 2, play[1]), (play[0] + 2, play[3]), (play[2], (play[1] + play[3]) / 2)],
                width, height, (1, 1, 1, 1))
        self._draw_clock(
            layout['elapsed'][0], layout['elapsed'][1],
            _animationplayback.format_clock(clip.time()), width, height)
        self._draw_clock(
            layout['duration'][0], layout['duration'][1],
            _animationplayback.format_clock(clip.duration), width, height)
        self._draw_icon_shapes(
            _animationplayback.loop_icon(layout['loop'], self._loop), width, height)
        self._draw_mute(layout['mute'], width, height)
        self._draw_volume(layout['volume'], width, height)

    def _draw_mute(self, rect, width, height):
        self._draw_icon_shapes(
            _animationplayback.speaker_icon(rect, self._preview_muted, self._preview_volume),
            width, height)

    def _draw_icon_shapes(self, shapes, width, height):
        for shape in shapes:
            kind = shape[0]
            if kind == 'rect':
                _kind, x1, y1, x2, y2, color = shape
                self._fill_rect(x1, y1, x2, y2, width, height, color)
            elif kind == 'poly':
                _kind, points, color = shape
                self._draw_overlay_fan(
                    [self._pixel_to_ndc(x, y, width, height) for x, y in points], color)
            elif kind == 'stroke':
                _kind, points, thickness, color = shape
                for start, end in zip(points, points[1:]):
                    self._fill_segment(*start, *end, thickness, width, height, color)

    def _fill_segment(self, x1, y1, x2, y2, thickness, width, height, color):
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return
        offset_x = -dy / length * (thickness / 2)
        offset_y = dx / length * (thickness / 2)
        self._draw_overlay_fan([
            self._pixel_to_ndc(x1 + offset_x, y1 + offset_y, width, height),
            self._pixel_to_ndc(x2 + offset_x, y2 + offset_y, width, height),
            self._pixel_to_ndc(x2 - offset_x, y2 - offset_y, width, height),
            self._pixel_to_ndc(x1 - offset_x, y1 - offset_y, width, height),
        ], color)

    def _draw_volume(self, rect, width, height):
        x1, y1, x2, y2 = rect
        self._fill_rect(x1, y1, x2, y2, width, height, (1, 1, 1, 0.35))
        shown = 0.0 if self._preview_muted else self._preview_volume
        filled = x1 + (x2 - x1) * shown
        self._fill_rect(x1, y1, filled, y2, width, height, (1, 1, 1, 1))
        knob = 5
        self._fill_rect(filled - knob / 2, y1 - 3, filled + knob / 2, y2 + 3, width, height, (1, 1, 1, 1))

    def _pixel_to_ndc(self, x: float, y: float, width: int, height: int) -> tuple[float, float]:
        return (x / width) * 2.0 - 1.0, -((y / height) * 2.0 - 1.0)

    def _fill_rect(self, x1, y1, x2, y2, width, height, color):
        corners = [
            self._pixel_to_ndc(x1, y1, width, height),
            self._pixel_to_ndc(x2, y1, width, height),
            self._pixel_to_ndc(x2, y2, width, height),
            self._pixel_to_ndc(x1, y2, width, height),
        ]
        self._draw_overlay_fan(corners, color)

    def _fill_triangle(self, points, width, height, color):
        self._draw_overlay_fan(
            [self._pixel_to_ndc(x, y, width, height) for x, y in points], color)

    def _draw_overlay_fan(self, ndc_points: list[tuple[float, float]], color):
        if len(ndc_points) < 3 or self._overlay_vao is None or self._overlay_vbo is None:
            return
        vertices = np.array([coord for point in ndc_points for coord in point], dtype=np.float32)
        gl.glUseProgram(self._overlay_shader_program)
        gl.glBindVertexArray(self._overlay_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._overlay_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
        gl.glUniform4f(self._overlay_color_uniform, *color)
        gl.glUniform2f(self._overlay_viewport_uniform, self.winfo_width(), self.winfo_height())
        gl.glUniform1i(self._overlay_is_dashed_uniform, 0)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, len(ndc_points))

    _CLOCK_FONT = {
        '0': ('111', '101', '101', '101', '111'),
        '1': ('010', '110', '010', '010', '111'),
        '2': ('111', '001', '111', '100', '111'),
        '3': ('111', '001', '111', '001', '111'),
        '4': ('101', '101', '111', '001', '001'),
        '5': ('111', '100', '111', '001', '111'),
        '6': ('111', '100', '111', '101', '111'),
        '7': ('111', '001', '001', '001', '001'),
        '8': ('111', '101', '111', '101', '111'),
        '9': ('111', '101', '111', '001', '111'),
        ':': ('0', '1', '0', '1', '0'),
    }

    def _draw_clock(self, x: float, y: float, text: str, width: int, height: int):
        scale = 2
        cursor = x
        for character in text:
            rows = self._CLOCK_FONT.get(character)
            if rows is None:
                continue
            glyph_width = len(rows[0])
            for row_index, row in enumerate(rows):
                for col_index, bit in enumerate(row):
                    if bit != '1':
                        continue
                    left = cursor + col_index * scale
                    top = y + row_index * scale
                    self._fill_rect(left, top, left + scale, top + scale, width, height, (1, 1, 1, 1))
            cursor += (glyph_width + 1) * scale

    def cleanup(self):
        """Clean up resources"""
        self._stop_animation()
        # Only attempt OpenGL cleanup if OpenGL was initialized
        gl_was_initialized = hasattr(self, '_gl_initialized') and self._gl_initialized

        # Clean up OpenGL resources
        if self._gl_texture_id is not None:
            if gl_was_initialized:
                try:
                    gl.glDeleteTextures([self._gl_texture_id])
                except:
                    pass  # Ignore errors during cleanup
            self._gl_texture_id = None

        if self._shader_program is not None:
            if gl_was_initialized:
                try:
                    gl.glDeleteProgram(self._shader_program)
                except:
                    pass  # Ignore errors during cleanup
            self._shader_program = None

        if self._overlay_shader_program is not None:
            if gl_was_initialized:
                try:
                    gl.glDeleteProgram(self._overlay_shader_program)
                except:
                    pass  # Ignore errors during cleanup
            self._overlay_shader_program = None

        if self._vao is not None:
            if gl_was_initialized:
                try:
                    gl.glDeleteVertexArrays(1, [self._vao])
                except:
                    pass  # Ignore errors during cleanup
            self._vao = None

        if self._overlay_vao is not None:
            if gl_was_initialized:
                try:
                    gl.glDeleteVertexArrays(1, [self._overlay_vao])
                except:
                    pass  # Ignore errors during cleanup
            self._overlay_vao = None

        if self._vbo is not None:
            if gl_was_initialized:
                try:
                    gl.glDeleteBuffers(1, [self._vbo])
                except:
                    pass  # Ignore errors during cleanup
            self._vbo = None

        if self._overlay_vbo is not None:
            if gl_was_initialized:
                try:
                    gl.glDeleteBuffers(1, [self._overlay_vbo])
                except:
                    pass  # Ignore errors during cleanup
            self._overlay_vbo = None

        if self._ebo is not None:
            if gl_was_initialized:
                try:
                    gl.glDeleteBuffers(1, [self._ebo])
                except:
                    pass  # Ignore errors during cleanup
            self._ebo = None

        # Mark OpenGL as no longer initialized to prevent further operations
        self._gl_initialized = False

        # Clear image data
        self._original_image_array = None
        self._sync_backdrop()
        self._original_image_size = None
        self._base_display_width = None
        self._base_display_height = None

        # Clear pending operations
        self._pending_load_operation = None
        self._pending_view_state = None

        # End any active selection
        if self._bbox_selection_mode:
            self._end_bbox_selection() 