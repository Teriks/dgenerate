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

import os

import dgenerate.auto1111_metadata as _auto1111_metadata
import dgenerate.batchprocess.util as _b_util
import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import dgenerate.subcommands.subcommand as _subcommand
import dgenerate.textprocessing as _textprocessing


class Auto1111MetadataSubCommand(_subcommand.SubCommand):
    """
    Utility to add Automatic1111 style metadata to an image,
    converted from a dgenerate config produced by --output-configs, or from
    metadata on said image added by --output-metadata.

    Examples:

    dgenerate --sub-command auto1111-metadata --image generated_image.png

    dgenerate --sub-command auto1111-metadata --image generated_image.png --config generated_image.dgen

    See: dgenerate --sub-command auto1111-metadata --help
    """

    NAMES = ['auto1111-metadata']

    def __init__(self, program_name='auto1111-metadata', **kwargs):
        super().__init__(**kwargs)

        self._parser = parser = _b_util.DirectiveArgumentParser(
            prog=program_name,
            description=
            """Automatic1111 Metadata Tool.
            
            This adds Automatic1111 metadata to images generated with 
            dgenerate via metadata conversion.
            
            Accepts an input image and a dgenerate --output-configs file, or uses 
            the dgenerate --output-metadata data from the image.
            
            If models from HuggingFace repos are specified in the config, 
            only their slug / name will be included in the metadata and not their hashes.
            
            This tool is most applicable for generations involving single file checkpoints 
            and sub-models, such as VAEs, LoRAs, ControlNets, and Textual Inversions.
            
            If direct links to models are provided in the config (such as CivitAI links), 
            they will searched for in the dgenerate web cache, and if they are not found 
            there they will be downloaded to the web cache so they can be hashed.
            """
        )

        parser.add_argument(
            "image",
            type=str,
            help="Path to image file to process. If not providing a config file, "
                 "this image must contain dgenerate's metadata in the EXIF or PNG text metadata, "
                 "this is generated in the image by the dgenerate option --output-metadata."
        )
        parser.add_argument(
            "-o", "--output",
            type=str,
            help="Output path for processed image (defaults to overwriting input image)."
        )

        parser.add_argument(
            "-c", "--config",
            type=str,
            help="Path to dgenerate config file to extract generation parameters from, "
                 "this file is produced by --output-configs."
        )

        parser.add_argument(
            '-v', '--verbose', action='store_true',
            help='Enable debug output?'
        )

        parser.add_argument(
            '-ofm', '--offline-mode', action='store_true',
            help="""Prevent downloads of resources that do not exist on disk already.""")

    def __call__(self) -> int:
        """
        Main entry point for the subcommand. Parses arguments, fetches model data,
        extracts links, and logs the results.

        :return: Exit code.
        """
        args = self._parser.parse_args(self.args)

        if self._parser.return_code is not None:
            return self._parser.return_code

        if not os.path.exists(args.image):
            _messages.error(f"Input image file does not exist: {args.image}")
            return 1

        if args.config and not os.path.exists(args.config):
            _messages.error(f"Config file does not exist: {args.config}")
            return 1

        input_fmt = os.path.splitext(args.image)[1].lower().replace('.', '')

        supported_input_formats = _mediainput.get_supported_image_formats()

        if input_fmt not in supported_input_formats:
            _messages.error(
                f"Unsupported image input format: {input_fmt}. "
                f"Please use one of: {_textprocessing.oxford_comma(supported_input_formats, 'or')}")
            return 1

        output_path = args.image if not args.output else args.output

        img_format = os.path.splitext(output_path)[1].lower().replace('.', '')

        if img_format not in ('jpg', 'jpeg', 'png'):
            _messages.error(
                f"Unsupported output image format: {img_format}. Please use: .jpg, .jpeg, or .png")
            return 1

        # Validate output directory exists and is writable
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                _messages.error(f"Failed to create output directory {output_dir}: {e}")
                return 1

        if args.config is not None:
            if not os.path.exists(args.config):
                _messages.error(f"Config file does not exist: {args.config}")
                return 1

            with open(args.config, 'rt', encoding='utf-8') as f:
                config = f.read()
        else:
            config = None

        try:
            with _messages.with_level(_messages.DEBUG if args.verbose else _messages.INFO):
                _auto1111_metadata.convert_and_insert_metadata(
                    image_path=args.image,
                    output_path=args.output,
                    dgenerate_config=config,
                    local_files_only=self.local_files_only or args.offline_mode,
                )
        except _auto1111_metadata.Auto1111MetadataCreationError as e:
            _messages.error(e)
            return 1

        _messages.log(
            f"Automatic1111 metadata successfully added to: {output_path}"
        )
        return 0
