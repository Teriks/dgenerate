import tkinter as tk

import dgenerate.console.imageseedselect as _imageseedselect
import dgenerate.console.recipesformentries.uriwithfloatargentry as _uriwithfloatargentry


class _ImageSeedEntry(_uriwithfloatargentry._UriWithFloatArgEntry):
    NAME = 'imageseed'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.build_uri_button = tk.Button(
            self.button_frame,
            text='Build URI',
            command=self._build_uri)

        self.build_uri_button.pack(side=tk.RIGHT)

    def _build_uri(self):
        r = _imageseedselect.request_uri(self.recipe_form)
        if r is not None and r.strip():
            self.uri_var.set(r)
