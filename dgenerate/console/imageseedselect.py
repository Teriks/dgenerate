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

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
import typing

import dgenerate.console.appearance as _appearance
import dgenerate.console.filedialog as _filedialog
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.helpdialog as _helpdialog
import dgenerate.console.mousewheelbind as _mousewheelbind
import dgenerate.console.resources as _resources
import dgenerate.console.spinbox as _spinbox
import dgenerate.console.textentry as _t_entry
import dgenerate.console.util as _util
import dgenerate.textprocessing as _textprocessing

_dialog_state = _util.DialogState(save_position=True, save_size=False)

_CARD = '#f7f7f7'
_LINE = '#d0d0d0'


def _card(widget):
    if _appearance.is_aqua(widget):
        return 'systemTextBackgroundColor'
    return _CARD


def _line(widget):
    if _appearance.is_aqua(widget):
        return 'systemSeparatorColor'
    return _LINE


def _seed(**kwargs) -> str:
    return '    ' + _textprocessing.format_image_seed_uri(**kwargs)


_PICTURES_HELP = f"""\
Seed images, inpaint masks, and control images. Each file row is one
path. Several seed rows are written as images:. A still is repeated
for every frame of a longer clip. The shortest animation decides how
many frames are generated.

Img2img. The model changes this picture. Strength comes from
--image-seed-strengths, or from strength= in the LTX section.

{_seed(seed_images='seed.png')}

A batch. Each file is one generation.

{_seed(seed_images=['seed-1.png', 'seed-2.png'])}

Inpaint. White in the mask is where new content is painted. A seed
and a mask alone use the short form, with no mask= name.

{_seed(seed_images='seed.png', mask_images='mask.png')}

One mask covers every seed.

{_seed(seed_images=['a.png', 'b.png', 'c.png'], mask_images='mask.png')}

One mask per seed. The counts have to match. Two masks for three
seeds is rejected, and more masks than seeds is rejected.

{_seed(seed_images=['a.png', 'b.png'], mask_images=['mask-a.png', 'mask-b.png'])}

A control image alone is written as the seed by default. Turn on
"Keep control=" to retain its control role, for example with Wan VACE
or an LTX IC-LoRA reference clip when combining it with a first frame.

{_seed(control_images='control.png')}

With a preserved control role for Wan VACE.

{_seed(control_images='clip.mp4', preserve_control=True)}

One control image per ControlNet.

{_seed(control_images=['canny.png', 'depth.png'])}

Img2img guided by control images. The control images are named.

{_seed(seed_images='seed.png', control_images='control.png')}

{_seed(seed_images='seed.png', control_images=['canny.png', 'depth.png'])}

Inpaint guided by a control image. mask= is named once another
option is present.

{_seed(seed_images='seed.png', mask_images='mask.png', control_images='control.png')}

Wan VACE. One mask and a control clip, with no seed image. The clip
stays control= so it is not read as the opening frame. The mask marks
what the model may change. Several masks need seed images.

{_seed(mask_images='mask.png', control_images='clip.mp4')}
"""

_REFERENCE_HELP = f"""\
Extra images written as reference=. They need a seed image, or a
control clip. A reference alone is rejected.

A still is repeated across the animation. A video is read together
with the seed and the mask, and the shortest animation sets the length.

Beside a seed.

{_seed(seed_images='seed.png', reference_images='style.png')}

Several references.

{_seed(seed_images='seed.png', reference_images=['earth.jpg', 'mountain.png'])}

Inpaint with an extra reference. Flux.2 Klein inpaint passes these
as image_reference. Flux.2 text-to-image references are the seed
images themselves, not reference=.

{_seed(seed_images='seed.png', mask_images='mask.png', reference_images='style.png')}

The same reference for every seed in a batch.

{_seed(seed_images=['a.png', 'b.png'], reference_images='style.png')}

Wan VACE. A control clip and a reference, with no seed image.

{_seed(control_images='clip.mp4', reference_images='style.jpg')}

Wan VACE with a mask and a reference.

{_seed(mask_images='mask.png', control_images='clip.mp4', reference_images='style.jpg')}
"""

_LATENTS_HELP = f"""\
Tensor files from an earlier run (.pt, .pth, or .safetensors), saved
with --image-format pt, pth, or safetensors. They are used as-is.
Resize, aspect, and frame range are rejected when latents are the
only input.

Latents alone. Img2img from raw latents. The VAE decodes them back
to pixels, except --model-type upscaler-x2, which can take denoised
latents directly.

{_seed(latents='latents.pt')}

Several latent files.

{_seed(latents=['one.pt', 'two.pt'])}

One latent per seed. The counts have to match.

{_seed(seed_images=['a.png', 'b.png'], latents=['one.pt', 'two.pt'])}

{_seed(seed_images='seed.png', latents='latents.safetensors')}

Not for ControlNet, IP Adapter, or Stable Cascade img2img. Those
read pixels.
"""

_ADAPTER_HELP = f"""\
IP Adapter images. Each row is one image, and rows are joined with +.
An empty row is skipped. A row with options and no image is rejected.
Tensor files are not accepted.

Adapter images only, with no seed.

{_seed(adapter_images=['style.png', 'face.png'])}

With a seed.

{_seed(seed_images='seed.png', adapter_images=['style.png', 'face.png'])}

With an inpaint mask and a control image.

{_seed(seed_images='seed.png', mask_images='mask.png', adapter_images='style.png', control_images='control.png')}

Each row can resize that image, set a pixel alignment, or ignore its
aspect ratio. Alignment 1 is omitted. These options use | on that
image, separate from Resize in Size and frames.

{_seed(adapter_images='style.png|resize=512x512')}

{_seed(adapter_images='style.png|align=64')}

{_seed(adapter_images='style.png|resize=512|aspect=false')}

Two rows, each with its own options.

{_seed(adapter_images='style.png|resize=512x512|align=8 + face.png|aspect=false')}
"""

_SIZE_HELP = f"""\
Resize, aspect ratio, and the frame range for an animation.

Resize is WIDTHxHEIGHT, or a width alone. With only a seed, or a
seed and a mask, it is written in the short form.

{_seed(seed_images='seed.png', resize='800x600')}

{_seed(seed_images='seed.png', resize='800')}

{_seed(seed_images='seed.png', mask_images='mask.png', resize='512x512')}

With any other option, resize is named.

{_seed(seed_images='seed.png', control_images='control.png', resize='512x512')}

Keep aspect ratio is on by default and is omitted. Turning it off
stretches the image to the size.

{_seed(seed_images='seed.png', aspect=False)}

{_seed(seed_images='seed.png', resize='800x600', aspect=False)}

frame-start and frame-end trim a video, GIF, or animated WEBP.
Indexes start at 0, and frame-end is included. Either one can be
set alone. A still image is repeated. The shortest animation in
the seed sets how many frames are generated.

{_seed(seed_images='clip.mp4', frame_start=0, frame_end=10)}

{_seed(seed_images='clip.mp4', frame_start=16)}

{_seed(seed_images='clip.mp4', frame_end=40)}

All of these together.

{_seed(seed_images='seed.png', mask_images='mask.png', control_images='control.png', resize='800x600', aspect=False, frame_start=0, frame_end=10)}

These options are written with a seed image, a control image, or a
last frame. They are rejected when the only input is latents.
"""

_LTX_HELP = f"""\
Last frame, latent-frame index, strength, and extra conditions.
Only --model-type ltx reads ltx-index. A last frame can stand alone.
Index, strength, and extra conditions need a seed image. The last
frame stays at full strength. strength= does not apply to it, and
neither does --image-seed-strengths.

First and last frame.

{_seed(seed_images='first.png', end_image='last.png')}

Last frame only. The opening frame is left unset.

{_seed(end_image='last.png')}

IC-LoRA. The opening frame, a reference clip, and the last frame.
The clip stays in control=.

{_seed(seed_images='first.png', end_image='last.png', control_images='control.mp4')}

With a resize and a frame range on the clips.

{_seed(seed_images='clip.gif', end_image='last.png', control_images='control.gif', resize='512x512', frame_start=16, frame_end=40)}

ltx-index is a latent frame, not an output frame. 0 is the first.
-1 is the last latent frame, the same slot as last-frame=. A still
is held across the output frames that latent frame covers. A video
starts at that latent frame.

strength= is from 0 to 1. 1 keeps the conditioning frame. A lower
value lets the generated frames leave it. On img2img, strength=
overrides --image-seed-strengths for that seed. On LTX, an omitted
strength stays at 1 unless --image-seed-strengths supplies one.

{_seed(seed_images='pose.png', ltx_index=4, ltx_strength=0.5)}

{_seed(seed_images='last.png', ltx_index=-1, ltx_strength=1)}

The first frame can be loose while the last frame stays pinned.

{_seed(seed_images='first.png', end_image='last.png', ltx_index=0, ltx_strength=0)}

Extra conditions. Each row needs an image and an index. Strength
is optional. Rows are joined with " ++ ". last-frame=, control=,
masks, and latents stay on the first group. A path that contains
" ++ " is quoted.

{_seed(seed_images='a.png', ltx_index=0, ltx_strength=0.5, ltx_extra_conditions=[('b.png', 8, 1), ('c.png', -1)])}
"""

_WAN_HELP = f"""\
Wan VACE uses the control clip and mask from Pictures, and reference=
from Reference. wan-pose= and wan-face= are Wan-Animate. wan-driving=
is the source clip. wan-background= is replace mode. The Wan keywords
need a seed image, the character still.

Wan VACE. Edit a clip. The mask marks what may change. No seed image.

{_seed(mask_images='mask.png', control_images='clip.mp4')}

A control clip with a reference still, and no seed image.

{_seed(control_images='clip.mp4', reference_images='style.jpg')}

Mask, control clip, and reference.

{_seed(mask_images='mask.png', control_images='clip.mp4', reference_images='style.jpg')}

Wan-Animate. The seed is the character. Pose and face clips supply
the motion, and the prompt describes how the character looks.
--model-type wan-animate.

{_seed(seed_images='char.png', wan_pose_video='pose.mp4', wan_face_video='face.mp4')}

Driving clip. --wan-animate-preprocess derives pose and face from
this clip. Wan-Animate-2 (--model-type wan-animate-2) uses this same
form and does not take pose, face, or background.

{_seed(seed_images='char.png', wan_driving_video='drive.mp4')}

Replace mode. The character is painted into the background clip.
Add a mask from Pictures to mark where.

{_seed(seed_images='char.png', mask_images='mask.png', wan_background_video='bg.mp4', wan_driving_video='drive.mp4')}

Video-to-video. A video path as the seed. The length follows the
clip, so trim it with frame-end= instead of --video-lengths.

{_seed(seed_images='clip.mp4', frame_end=40)}

First and last frame, for FLF2V.

{_seed(seed_images='first.png', end_image='last.png')}

Image-to-video. A still in Seed image, with none of these clips.

{_seed(seed_images='still.png')}
"""


class _ImageSeedSelect(tk.Toplevel):
    def __init__(self, insert: typing.Callable[[str], None], master=None, position: tuple[int, int] = None):
        super().__init__(master)
        self.title('Insert Image Seed URI')
        self.configure(padx=10, pady=8)
        self.transient(master)
        self.resizable(True, True)

        base = tkfont.nametofont('TkDefaultFont')
        self._title_font = base.copy()
        self._title_font.configure(weight='bold')

        self.entries = []
        self._insert = insert
        self._row = 0

        outer = ttk.Frame(self)
        outer.grid(row=0, column=0, sticky=tk.NSEW)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        viewport = tk.Frame(outer)
        _appearance.configure_viewport(viewport)
        viewport.grid(row=0, column=0, sticky=tk.NSEW)

        self._canvas = tk.Canvas(viewport, width=740, height=560, highlightthickness=0, borderwidth=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._scroll = ttk.Scrollbar(outer, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scroll.set)
        self._inner = ttk.Frame(self._canvas, padding=(12, 10))
        self._inner.grid_columnconfigure(0, weight=1)
        self._window = self._canvas.create_window((0, 0), window=self._inner, anchor='nw')
        self._inner.bind('<Configure>', self._fit_scroll)
        self._canvas.bind('<Configure>', self._fit_inner_width)

        self._scroll.grid(row=0, column=1, sticky=tk.NS)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        self._insert_button = ttk.Button(self, text='Insert', command=self._insert_click)
        self._insert_button.grid(row=1, column=0, pady=(8, 0))

        self._build_form()
        _mousewheelbind.bind_mousewheel(self._canvas.bind, self._on_wheel)
        self._hook_wheel(self._inner)
        self.minsize(680, 420)
        _appearance.install(self)
        _util.position_toplevel(master, self, position=position)

    def _fit_scroll(self, _event):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def _fit_inner_width(self, event):
        self._canvas.itemconfigure(self._window, width=event.width)

    def _hook_wheel(self, widget):
        if isinstance(widget, (_spinbox.IntSpinbox, _spinbox.FloatSpinbox)):
            return
        _mousewheelbind.bind_mousewheel(widget.bind, self._on_wheel)
        for child in widget.winfo_children():
            self._hook_wheel(child)

    def _on_wheel(self, event):
        _mousewheelbind.handle_canvas_scroll(self._canvas, event)
        return 'break'

    def _show_if_hidden(self, widget):
        self.update_idletasks()
        canvas = self._canvas
        if not self.winfo_viewable() or canvas.winfo_height() <= 1:
            return
        margin = 8
        visible_top = canvas.winfo_rooty()
        visible_bottom = visible_top + canvas.winfo_height()
        widget_top = widget.winfo_rooty()
        widget_bottom = widget_top + max(widget.winfo_height(), 1)
        if widget_top >= visible_top + margin and widget_bottom <= visible_bottom - margin:
            return
        bbox = canvas.bbox('all')
        if not bbox:
            return
        span = max(bbox[3] - bbox[1], 1)
        view_height = canvas.winfo_height()
        if widget_bottom - widget_top >= view_height or widget_top < visible_top + margin:
            delta = widget_top - visible_top - margin
        else:
            delta = widget_bottom - (visible_bottom - margin)
        top = canvas.yview()[0] + delta / span
        canvas.yview_moveto(min(1, max(0, top)))

    def _track(self, widget):
        widget.bind('<Key>', self._valid)
        self.entries.append(widget)
        return widget

    def _section(self, title, help_text):
        header = ttk.Frame(self._inner)
        header.grid(row=self._row, column=0, sticky=tk.EW, pady=(12, 2))
        self._row += 1
        ttk.Label(header, text=title, font=self._title_font).pack(side=tk.LEFT)
        ttk.Button(
            header,
            text='Help',
            command=lambda heading=title, body=help_text: self._show_help(heading, body),
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Separator(header, orient='horizontal').pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        return header

    def _show_help(self, title, help_text):
        _helpdialog.show_help_dialog(
            parent=self,
            title=f'Image seed: {title}',
            help_text=help_text.strip(),
            size=(760, 560),
            dock_to_right=True,
            position_widget=self,
        )

    def _build_form(self):
        self._section('Pictures', _PICTURES_HELP)
        self._seeds = _FileRows(self, self._open_image, 'Seed image', start=1)
        self._masks = _FileRows(self, self._open_image, 'Inpaint mask')
        self._controls = _FileRows(self, self._open_image, 'Control image')
        self._preserve_control_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._inner,
            text='Keep control= when no seed image is supplied (VACE / IC-LoRA)',
            variable=self._preserve_control_var
        ).grid(row=self._row, column=0, sticky=tk.W, pady=(0, 4))
        self._row += 1

        self._section('Reference', _REFERENCE_HELP)
        self._references = _FileRows(self, self._open_image, 'Reference image')

        self._section('Latents', _LATENTS_HELP)
        self._latents = _FileRows(self, self._open_latents, 'Latents')

        self._section('IP Adapter', _ADAPTER_HELP)
        self._adapters = _AdapterRows(self)

        self._section('Size and frames', _SIZE_HELP)
        self._resize_entry = self._labeled_entry('Resize (WxH)')
        aspect_row = ttk.Frame(self._inner)
        aspect_row.grid(row=self._row, column=0, sticky=tk.W, pady=2)
        self._row += 1
        self._aspect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(aspect_row, text='Keep aspect ratio', variable=self._aspect_var).pack(side=tk.LEFT)
        self._frame_start = self._labeled_spin('Frame start', 0, None)
        self._frame_end = self._labeled_spin('Frame end', 0, None)

        self._section('LTX', _LTX_HELP)
        end_row = ttk.Frame(self._inner)
        end_row.grid(row=self._row, column=0, sticky=tk.EW, pady=2)
        self._row += 1
        end_row.grid_columnconfigure(1, weight=1)
        ttk.Label(end_row, text='Last frame').grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        self._end_entry = self._track(_t_entry.TextEntry(end_row))
        self._end_entry.grid(row=0, column=1, sticky=tk.EW)
        ttk.Button(end_row, text='File', command=lambda: self._open_image(self._end_entry)).grid(
            row=0, column=2, padx=(4, 0))
        self._ltx_index = self._labeled_spin('Latent frame index', -100000, 100000)
        self._ltx_strength = self._labeled_float('Strength')
        self._extras = _ExtraRows(self)

        self._section('Wan', _WAN_HELP)
        self._pose_entry = self._wan_file('Pose clip')
        self._face_entry = self._wan_file('Face clip')
        self._driving_entry = self._wan_file('Driving clip')
        self._background_entry = self._wan_file('Background clip')

    def _labeled_entry(self, label):
        row = ttk.Frame(self._inner)
        row.grid(row=self._row, column=0, sticky=tk.EW, pady=2)
        self._row += 1
        row.grid_columnconfigure(1, weight=1)
        ttk.Label(row, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        entry = self._track(_t_entry.TextEntry(row))
        entry.grid(row=0, column=1, sticky=tk.EW)
        return entry

    def _labeled_spin(self, label, low, high):
        row = ttk.Frame(self._inner)
        row.grid(row=self._row, column=0, sticky=tk.EW, pady=2)
        self._row += 1
        row.grid_columnconfigure(1, weight=1)
        ttk.Label(row, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        spin = _spinbox.IntSpinbox(row, from_=low, to=high, textvariable=tk.StringVar(value=''))
        spin.grid(row=0, column=1, sticky=tk.EW)
        spin.create_spin_buttons(row).grid(row=0, column=2, padx=(4, 0))
        self._track(spin)
        return spin

    def _labeled_float(self, label):
        row = ttk.Frame(self._inner)
        row.grid(row=self._row, column=0, sticky=tk.EW, pady=2)
        self._row += 1
        row.grid_columnconfigure(1, weight=1)
        ttk.Label(row, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        spin = _spinbox.FloatSpinbox(
            row, from_=0, to=1, increment=0.1, textvariable=tk.StringVar(value=''))
        spin.grid(row=0, column=1, sticky=tk.EW)
        spin.create_spin_buttons(row).grid(row=0, column=2, padx=(4, 0))
        self._track(spin)
        return spin

    def _wan_file(self, label):
        row = ttk.Frame(self._inner)
        row.grid(row=self._row, column=0, sticky=tk.EW, pady=2)
        self._row += 1
        row.grid_columnconfigure(1, weight=1)
        ttk.Label(row, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        entry = self._track(_t_entry.TextEntry(row))
        entry.grid(row=0, column=1, sticky=tk.EW)
        ttk.Button(row, text='File', command=lambda e=entry: self._open_image(e)).grid(
            row=0, column=2, padx=(4, 0))
        return entry

    def _place(self, widget):
        widget.grid(row=self._row, column=0, sticky=tk.EW, pady=(0, 4))
        self._row += 1

    @staticmethod
    def _fill(entry, file_path):
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def _open_image(self, entry):
        self._fill(entry, _filedialog.open_file_dialog(
            **_resources.get_file_dialog_args(['images-in', 'videos-in'])))

    def _open_latents(self, entry):
        self._fill(entry, _filedialog.open_file_dialog(
            title='Select Latents File',
            filetypes=[
                ('Tensor files', '*.pt *.pth *.safetensors'),
                ('PyTorch files', '*.pt *.pth'),
                ('SafeTensors files', '*.safetensors'),
                ('All files', '*.*')
            ]))

    def _paths_argument(self, paths):
        if not paths:
            return None
        if len(paths) == 1:
            return paths[0]
        return paths

    def _insert_click(self):
        if not self._ltx_index.is_valid():
            _entry.invalid_colors(self._ltx_index)
            return
        if not self._ltx_strength.is_valid():
            _entry.invalid_colors(self._ltx_strength)
            return
        if not self._frame_start.is_valid() or not self._frame_end.is_valid():
            _entry.invalid_colors(self._frame_start)
            _entry.invalid_colors(self._frame_end)
            return

        seeds = self._seeds.paths()
        masks = self._masks.paths()
        controls = self._controls.paths()
        references = self._references.paths()
        latents = self._latents.paths()
        end_image = _entry.shell_quote_if(self._end_entry.get().strip(), strict=True) or None

        if masks and not seeds and not controls:
            self._seeds.mark()
            self._controls.mark()
            return
        if len(masks) > 1 and not seeds:
            self._masks.mark()
            return
        if references and not seeds and not controls:
            self._seeds.mark()
            self._controls.mark()
            return
        if len(masks) > 1 and len(masks) != len(seeds):
            self._masks.mark()
            return
        if seeds and latents and len(seeds) != len(latents):
            self._latents.mark()
            self._seeds.mark()
            return

        resize_value = self._resize_entry.get().strip()
        if resize_value:
            try:
                _textprocessing.parse_image_size(resize_value)
            except ValueError:
                _entry.invalid_colors(self._resize_entry)
                return
        aspect_value = self._aspect_var.get()

        frame_start_value = self._frame_start.get().strip()
        frame_end_value = self._frame_end.get().strip()
        frame_start = int(frame_start_value) if frame_start_value else None
        frame_end = int(frame_end_value) if frame_end_value else None
        ltx_index_value = self._ltx_index.get().strip()
        ltx_strength_value = self._ltx_strength.get().strip()
        ltx_index = int(ltx_index_value) if ltx_index_value else None
        ltx_strength = float(ltx_strength_value) if ltx_strength_value else None
        pose = _entry.shell_quote_if(self._pose_entry.get().strip(), strict=True) or None
        face = _entry.shell_quote_if(self._face_entry.get().strip(), strict=True) or None
        driving = _entry.shell_quote_if(self._driving_entry.get().strip(), strict=True) or None
        background = _entry.shell_quote_if(self._background_entry.get().strip(), strict=True) or None
        wan_extras = pose or face or driving or background

        try:
            adapters = self._adapters.values()
            extras = self._extras.values()
        except _RowError as error:
            _entry.invalid_colors(error.widget)
            return

        latents_only = latents and not (seeds or controls or adapters or end_image or wan_extras)
        if latents_only and (resize_value or aspect_value is False or
                             frame_start is not None or frame_end is not None or
                             ltx_index is not None or ltx_strength is not None or extras):
            _entry.invalid_colors(self._resize_entry)
            return
        if (ltx_index is not None or ltx_strength is not None or extras or wan_extras) and not seeds:
            self._seeds.mark()
            return
        if (frame_start is not None or frame_end is not None or aspect_value is False or resize_value) and not (
                seeds or controls or latents or adapters or end_image):
            self._seeds.mark()
            return
        if not (seeds or controls or latents or adapters or end_image):
            self._seeds.mark()
            return

        try:
            value = _textprocessing.format_image_seed_uri(
                seed_images=self._paths_argument(seeds),
                mask_images=self._paths_argument(masks),
                control_images=self._paths_argument(controls),
                preserve_control=self._preserve_control_var.get(),
                reference_images=self._paths_argument(references),
                latents=self._paths_argument(latents),
                adapter_images=self._paths_argument(adapters),
                resize=resize_value or None,
                aspect=aspect_value,
                frame_start=frame_start,
                frame_end=frame_end,
                end_image=end_image,
                wan_pose_video=pose,
                wan_face_video=face,
                wan_background_video=background,
                wan_driving_video=driving,
                ltx_index=ltx_index,
                ltx_strength=ltx_strength,
                ltx_extra_conditions=extras or None
            )
        except ValueError:
            self._seeds.mark()
            return

        if value:
            self._insert(value)
        self.destroy()

    def _valid(self, _event):
        for entry in self.entries:
            _entry.valid_colors(entry)


class _RowError(Exception):
    def __init__(self, widget):
        self.widget = widget


class _FileRows:
    def __init__(self, dialog: _ImageSeedSelect, opener, title, start=0):
        self.dialog = dialog
        self.opener = opener
        self.entries = []
        self._frames = []

        shell = ttk.Frame(dialog._inner)
        dialog._place(shell)
        shell.grid_columnconfigure(0, weight=1)
        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky=tk.EW)
        ttk.Label(header, text=title).pack(side=tk.LEFT)
        ttk.Button(header, text='Add', command=self.add).pack(side=tk.RIGHT)
        self.body = ttk.Frame(shell)
        self.body.grid(row=1, column=0, sticky=tk.EW)
        self.body.grid_columnconfigure(0, weight=1)
        for _ in range(start):
            self.add()

    def add(self):
        row = ttk.Frame(self.body)
        row.grid(row=len(self._frames), column=0, sticky=tk.EW, pady=1)
        row.grid_columnconfigure(0, weight=1)
        entry = self.dialog._track(_t_entry.TextEntry(row))
        entry.grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(row, text='File', command=lambda e=entry: self.opener(e)).grid(row=0, column=1, padx=(4, 0))
        ttk.Button(row, text='Remove', command=lambda r=row, e=entry: self.remove(r, e)).grid(
            row=0, column=2, padx=(4, 0))
        self._frames.append(row)
        self.entries.append(entry)
        _appearance.apply(row)
        self.dialog._hook_wheel(row)
        self.dialog._show_if_hidden(row)

    def remove(self, row, entry):
        if entry in self.entries:
            self.entries.remove(entry)
        if entry in self.dialog.entries:
            self.dialog.entries.remove(entry)
        if row in self._frames:
            self._frames.remove(row)
        row.destroy()
        for index, frame in enumerate(self._frames):
            frame.grid(row=index, column=0, sticky=tk.EW, pady=1)
        self.body.configure(height=1)
        self.dialog.update_idletasks()

    def paths(self):
        values = []
        for entry in self.entries:
            text = entry.get().strip()
            if text:
                values.append(_entry.shell_quote_if(text, strict=True))
        return values

    def mark(self):
        if not self.entries:
            self.add()
        for entry in self.entries:
            _entry.invalid_colors(entry)


class _AdapterRows:
    def __init__(self, dialog: _ImageSeedSelect):
        self.dialog = dialog
        self.rows = []
        shell = ttk.Frame(dialog._inner)
        dialog._place(shell)
        shell.grid_columnconfigure(0, weight=1)
        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky=tk.EW)
        ttk.Label(header, text='Adapter images').pack(side=tk.LEFT)
        ttk.Button(header, text='Add', command=self.add).pack(side=tk.RIGHT)
        self.body = ttk.Frame(shell)
        self.body.grid(row=1, column=0, sticky=tk.EW)
        self.body.grid_columnconfigure(0, weight=1)

    def add(self):
        card = tk.Frame(self.body, bg=_card(self.dialog), highlightbackground=_line(self.dialog), highlightthickness=1,
                        padx=6, pady=4)
        card.grid(row=len(self.rows), column=0, sticky=tk.EW, pady=3)
        card.grid_columnconfigure(1, weight=1)
        tk.Label(card, text='Image', bg=_card(self.dialog)).grid(row=0, column=0, sticky=tk.W)
        path = self.dialog._track(_t_entry.TextEntry(card))
        path.grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Button(card, text='File', command=lambda e=path: self.dialog._open_image(e)).grid(row=0, column=2)
        ttk.Button(card, text='Remove', command=lambda c=card: self.remove(c)).grid(row=0, column=3, padx=(4, 0))

        tk.Label(card, text='Resize', bg=_card(self.dialog)).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        options = tk.Frame(card, bg=_card(self.dialog))
        options.grid(row=1, column=1, columnspan=3, sticky=tk.EW, pady=(4, 0))
        resize = self.dialog._track(_t_entry.TextEntry(options, width=12))
        resize.pack(side=tk.LEFT)
        tk.Label(options, text='Align', bg=_card(self.dialog)).pack(side=tk.LEFT, padx=(8, 4))
        align = _spinbox.IntSpinbox(options, from_=1, to=256, width=6, textvariable=tk.StringVar(value=''))
        align.pack(side=tk.LEFT)
        align.create_spin_buttons(options).pack(side=tk.LEFT, padx=(2, 0))
        self.dialog._track(align)
        aspect = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text='Aspect', variable=aspect).pack(side=tk.LEFT, padx=(8, 0))

        self.rows.append({'card': card, 'path': path, 'resize': resize, 'align': align, 'aspect': aspect})
        _appearance.apply(card)
        self.dialog._hook_wheel(card)
        self.dialog._show_if_hidden(card)

    def remove(self, card):
        kept = []
        for row in self.rows:
            if row['card'] is card:
                for widget in (row['path'], row['resize'], row['align']):
                    if widget in self.dialog.entries:
                        self.dialog.entries.remove(widget)
                card.destroy()
            else:
                kept.append(row)
        self.rows = kept
        for index, row in enumerate(self.rows):
            row['card'].grid(row=index, column=0, sticky=tk.EW, pady=3)
        self.body.configure(height=1)
        self.dialog.update_idletasks()

    def values(self):
        uris = []
        for row in self.rows:
            path = row['path'].get().strip()
            resize = row['resize'].get().strip()
            align = row['align'].get().strip()
            aspect = row['aspect'].get()
            if not path and not resize and not align and aspect:
                continue
            if not path:
                raise _RowError(row['path'])
            if not row['align'].is_valid():
                raise _RowError(row['align'])
            if resize:
                try:
                    _textprocessing.parse_image_size(resize)
                except ValueError:
                    raise _RowError(row['resize'])
            uri = _entry.shell_quote_if(path, strict=True)
            if resize:
                uri += f'|resize={resize}'
            if align and align != '1':
                uri += f'|align={align}'
            if not aspect:
                uri += '|aspect=false'
            uris.append(uri)
        return uris


class _ExtraRows:
    def __init__(self, dialog: _ImageSeedSelect):
        self.dialog = dialog
        self.rows = []
        shell = ttk.Frame(dialog._inner)
        dialog._place(shell)
        shell.grid_columnconfigure(0, weight=1)
        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky=tk.EW, pady=(6, 0))
        ttk.Label(header, text='Extra conditions').pack(side=tk.LEFT)
        ttk.Button(header, text='Add', command=self.add).pack(side=tk.RIGHT)
        self.body = ttk.Frame(shell)
        self.body.grid(row=1, column=0, sticky=tk.EW)
        self.body.grid_columnconfigure(0, weight=1)

    def add(self):
        card = tk.Frame(self.body, bg=_card(self.dialog), highlightbackground=_line(self.dialog), highlightthickness=1,
                        padx=6, pady=4)
        card.grid(row=len(self.rows), column=0, sticky=tk.EW, pady=3)
        card.grid_columnconfigure(1, weight=1)
        tk.Label(card, text='Image', bg=_card(self.dialog)).grid(row=0, column=0, sticky=tk.W)
        path = self.dialog._track(_t_entry.TextEntry(card))
        path.grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Button(card, text='File', command=lambda e=path: self.dialog._open_image(e)).grid(row=0, column=2)
        ttk.Button(card, text='Remove', command=lambda c=card: self.remove(c)).grid(row=0, column=3, padx=(4, 0))

        tk.Label(card, text='Index', bg=_card(self.dialog)).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        index = _spinbox.IntSpinbox(card, from_=-100000, to=100000, textvariable=tk.StringVar(value=''))
        index.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=(4, 0))
        index.create_spin_buttons(card).grid(row=1, column=2, pady=(4, 0))
        self.dialog._track(index)

        strength_row = tk.Frame(card, bg=_card(self.dialog))
        strength_row.grid(row=2, column=0, columnspan=4, sticky=tk.EW, pady=(4, 0))
        strength_row.grid_columnconfigure(1, weight=1)
        tk.Label(strength_row, text='Strength', bg=_card(self.dialog)).grid(row=0, column=0, sticky=tk.W)
        strength = _spinbox.FloatSpinbox(
            strength_row, from_=0, to=1, increment=0.1, textvariable=tk.StringVar(value=''))
        strength.grid(row=0, column=1, sticky=tk.EW, padx=4)
        strength.create_spin_buttons(strength_row).grid(row=0, column=2)
        self.dialog._track(strength)

        self.rows.append({'card': card, 'path': path, 'index': index, 'strength': strength})
        _appearance.apply(card)
        self.dialog._hook_wheel(card)
        self.dialog._show_if_hidden(card)

    def remove(self, card):
        kept = []
        for row in self.rows:
            if row['card'] is card:
                for widget in (row['path'], row['index'], row['strength']):
                    if widget in self.dialog.entries:
                        self.dialog.entries.remove(widget)
                card.destroy()
            else:
                kept.append(row)
        self.rows = kept
        for index, row in enumerate(self.rows):
            row['card'].grid(row=index, column=0, sticky=tk.EW, pady=3)
        self.body.configure(height=1)
        self.dialog.update_idletasks()

    def values(self):
        extras = []
        for row in self.rows:
            path = _entry.shell_quote_if(row['path'].get().strip(), strict=True)
            index_value = row['index'].get().strip()
            strength_value = row['strength'].get().strip()
            if not path and not index_value and not strength_value:
                continue
            if not row['index'].is_valid():
                raise _RowError(row['index'])
            if not row['strength'].is_valid():
                raise _RowError(row['strength'])
            if not path or index_value == '':
                raise _RowError(row['path'] if not path else row['index'])
            extras.append((
                path,
                int(index_value),
                float(strength_value) if strength_value else None
            ))
        return extras


def request_uri(master, insert: typing.Callable[[str], None], dialog_state: _util.DialogState | None = None):
    return _util.create_singleton_dialog(
        master=master,
        dialog_class=_ImageSeedSelect,
        state=_dialog_state if dialog_state is None else dialog_state,
        dialog_kwargs={'insert': insert}
    )
