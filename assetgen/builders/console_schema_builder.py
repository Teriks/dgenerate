#!/usr/bin/env python3
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

import inspect
import itertools
import json
from pathlib import Path
from typing import Optional

import diffusers

import dgenerate.arguments as _arguments
import dgenerate.batchprocess as _batchprocess
import dgenerate.imageprocessors.imageprocessorloader as _imageprocessorloader
import dgenerate.latentsprocessors.latentsprocessorloader as _latentsprocessorloader
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.promptupscalers.promptupscalerloader as _promptupscalerloader
import dgenerate.promptweighters.promptweighterloader as _promptweighterloader
import dgenerate.textprocessing as _textprocessing


class ConsoleSchemaBuilder:
    """Builder for console UI schema files."""

    def __init__(self, project_dir: Optional[Path] = None):
        """
        Initialize the ConsoleSchemaBuilder.

        :param project_dir: Project directory (defaults to current working directory)
        :type project_dir: Optional[Path]
        """
        self.project_dir = project_dir or Path.cwd()
        self.schemas_dir = self.project_dir / 'dgenerate' / 'console' / 'schemas'

    def build(self):
        """Build all console schema files."""
        print("Building console schemas...")

        # Ensure schemas directory exists
        self.schemas_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._build_imageprocessors_schema()
            self._build_latentsprocessors_schema()
            self._build_promptupscalers_schema()
            self._build_promptweighters_schema()
            self._build_karrasschedulers_schema()
            self._build_quantizers_schema()
            self._build_directives_schema()
            self._build_functions_schema()
            self._build_mediaformats_schema()
            self._build_arguments_schema()
            self._build_sub_models_schema()

            print("✓ Console schemas built successfully")
        except Exception as e:
            print(f"✗ Error building console schemas: {e}")
            raise

    def _build_imageprocessors_schema(self):
        """Build image processors schema."""
        schema_file = self.schemas_dir / 'imageprocessors.json'
        plugin_loader = _imageprocessorloader.ImageProcessorLoader()
        schema = plugin_loader.get_accepted_args_schema(include_bases=True)

        # sort by processor name, this affects json output
        schema = dict(sorted(schema.items(), key=lambda x: x[0]))

        for plugin in schema.keys():
            schema[plugin].update({'PROCESSOR_HELP': plugin_loader.get_help(
                plugin, wrap_width=100, include_bases=True)})

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def _build_latentsprocessors_schema(self):
        """Build latents processors schema."""
        schema_file = self.schemas_dir / 'latentsprocessors.json'
        plugin_loader = _latentsprocessorloader.LatentsProcessorLoader()
        schema = plugin_loader.get_accepted_args_schema(include_bases=True)

        # sort by processor name, this affects json output
        schema = dict(sorted(schema.items(), key=lambda x: x[0]))

        for plugin in schema.keys():
            schema[plugin].update({'PROCESSOR_HELP': plugin_loader.get_help(
                plugin, wrap_width=100, include_bases=True)})

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def _build_promptupscalers_schema(self):
        """Build prompt upscalers schema."""
        schema_file = self.schemas_dir / 'promptupscalers.json'
        plugin_loader = _promptupscalerloader.PromptUpscalerLoader()
        schema = plugin_loader.get_accepted_args_schema(include_bases=True)

        # sort by processor name, this affects json output
        schema = dict(sorted(schema.items(), key=lambda x: x[0]))

        for plugin in schema.keys():
            schema[plugin].update({'PROMPT_UPSCALER_HELP': plugin_loader.get_help(
                plugin, wrap_width=100, include_bases=True)})

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def _build_promptweighters_schema(self):
        """Build prompt weighters schema."""
        schema_file = self.schemas_dir / 'promptweighters.json'
        plugin_loader = _promptweighterloader.PromptWeighterLoader()
        schema = plugin_loader.get_accepted_args_schema(include_bases=True)

        # sort by processor name, this affects json output
        schema = dict(sorted(schema.items(), key=lambda x: x[0]))

        for plugin in schema.keys():
            schema[plugin].update({'PROMPT_WEIGHTER_HELP': plugin_loader.get_help(
                plugin, wrap_width=100, include_bases=True)})

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def _build_karrasschedulers_schema(self):
        """Build Karras schedulers schema."""
        schema_file = self.schemas_dir / 'karrasschedulers.json'
        scheduler_names = sorted(
            [e.name for e in diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers] +
            ['LCMScheduler', 'FlowMatchEulerDiscreteScheduler', 'DDPMWuerstchenScheduler'])

        schema = _pipelinewrapper.get_scheduler_uri_schema([getattr(diffusers, n) for n in scheduler_names])

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def _build_quantizers_schema(self):
        """Build quantizers schema."""
        schema_file = self.schemas_dir / 'quantizers.json'

        uris = [
            _pipelinewrapper.uris.BNBQuantizerUri,
            _pipelinewrapper.uris.SDNQQuantizerUri
        ]

        schema = dict()

        for uri in uris:
            name = _pipelinewrapper.get_uri_names(uri)[0]
            schema[name] = _pipelinewrapper.get_uri_accepted_args_schema(uri)
            schema[name]['QUANTIZER_HELP'] = _pipelinewrapper.get_uri_help(uri, wrap_width=100)

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def _build_sub_models_schema(self):
        """Build sub-models schema."""
        schema_file = self.schemas_dir / 'submodels.json'

        uris = [
            _pipelinewrapper.uris.UNetUri,
            _pipelinewrapper.uris.TransformerUri,
            _pipelinewrapper.uris.TextEncoderUri,
            _pipelinewrapper.uris.VAEUri,
            _pipelinewrapper.uris.ImageEncoderUri,
            _pipelinewrapper.uris.LoRAUri,
            _pipelinewrapper.uris.IPAdapterUri,
            _pipelinewrapper.uris.ControlNetUri,
            _pipelinewrapper.uris.T2IAdapterUri,
            _pipelinewrapper.uris.TextualInversionUri,
            _pipelinewrapper.uris.AdetailerDetectorUri,
            _pipelinewrapper.uris.SCascadeDecoderUri,
            _pipelinewrapper.uris.SDXLRefinerUri
        ]

        schema = dict()
        for uri in uris:
            name = _pipelinewrapper.get_uri_names(uri)[0]
            schema[name] = _pipelinewrapper.get_uri_accepted_args_schema(uri)
            schema[name]['SUBMODEL_HELP'] = _pipelinewrapper.get_uri_help(uri, wrap_width=100)

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def _build_directives_schema(self):
        """Build directives schema."""
        schema_file = self.schemas_dir / 'directives.json'
        config_runner = _batchprocess.ConfigRunner()

        schema = dict()

        for directive in sorted(config_runner.directives.keys()):
            schema['\\' + directive] = config_runner.generate_directives_help(
                [directive], help_wrap_width=100
            )

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def _build_functions_schema(self):
        """Build functions schema."""
        schema_file = self.schemas_dir / 'functions.json'
        config_runner = _batchprocess.ConfigRunner()

        schema = dict()

        for function in sorted(itertools.chain(config_runner.template_functions.keys(), config_runner.builtins.keys())):
            schema[function] = config_runner.generate_functions_help(
                [function], help_wrap_width=100
            )

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def _build_mediaformats_schema(self):
        """Build media formats schema."""
        schema_file = self.schemas_dir / 'mediaformats.json'
        schema = dict()

        schema['images-in'] = sorted(_mediainput.get_supported_image_formats())
        schema['images-out'] = sorted(_mediaoutput.get_supported_static_image_formats())
        schema['videos-in'] = sorted(_mediainput.get_supported_animation_reader_formats())
        schema['videos-out'] = sorted(_mediaoutput.get_supported_animation_writer_formats())

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def _build_arguments_schema(self):
        """Build arguments schema."""
        schema_file = self.schemas_dir / 'arguments.json'
        schema = dict()

        def _opt_name(a):
            for opt in a.option_strings:
                if opt.startswith('--'):
                    return opt
            return a.option_strings[0]

        for action in (a for a in _arguments._actions if a.option_strings):
            schema[_opt_name(action)] = _textprocessing.wrap_paragraphs(
                inspect.cleandoc(action.help), width=100)

        with open(schema_file, 'w') as file:
            json.dump(schema, file)

    def get_output_directory(self) -> Path:
        """Get the path to the schemas output directory."""
        return self.schemas_dir
