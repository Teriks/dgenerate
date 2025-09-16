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

import hashlib
import json
import pathlib
import re
import urllib.parse

import tqdm
import os
import typing
import dgenerate.memory as _memory

import PIL.Image
import PIL.PngImagePlugin

import dgenerate.image as _image
import dgenerate.batchprocess.batchprocessor as _batchprocessor
import dgenerate.pipelinewrapper.enums
import dgenerate.webcache as _webcache
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.arguments as _arguments

import dgenerate.resources as _resources

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.filecache as _filecache
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

__doc__ = """This module provides functionality to embed Automatic1111 metadata
             into images. This metadata can be converted from dgenerate configs 
             produced by --output-configs, or images with dgenerate metadata 
             attached via --output-metadata."""


class Auto1111MetadataCreationError(Exception):
    """
    Exception raised when there is an error creating Automatic1111 metadata.
    """
    pass


# Mapping of dgenerate scheduler names to Automatic1111 sampler names
_SCHEDULER_TO_AUTOMATIC1111 = {
    "DDIMScheduler": "DDIM",
    "DDPMScheduler": "DDPM",
    "DEISMultistepScheduler": "DEIS",
    "DPMSolverMultistepScheduler": "DPM++ 2M",
    "DPMSolverSDEScheduler": "DPM++ SDE",
    "DPMSolverSinglestepScheduler": "DPM++ SDE Karras",
    "DPMSolverTwoStepScheduler": "DPM++ 2M SDE",
    "EDMEulerScheduler": "Euler",
    "EulerAncestralDiscreteScheduler": "Euler a",
    "EulerDiscreteScheduler": "Euler",
    "HeunDiscreteScheduler": "Heun",
    "KDPM2AncestralDiscreteScheduler": "DPM2 a",
    "KDPM2DiscreteScheduler": "DPM2",
    "LCMScheduler": "LCM",
    "LMSDiscreteScheduler": "LMS",
    "PNDMScheduler": "PNDM",
    "UniPCMultistepScheduler": "UniPC",
    "FlowMatchEulerDiscreteScheduler": "Euler FM",
    "DDPMWuerstchenScheduler": "DDPM Wuerstchen",
    "RASFlowMatchEulerDiscreteScheduler": "Euler FM RAS"
}

_DEFAULT_MODEL_TYPE_TO_SCHEDULER_MAP = {
    # Standard Stable Diffusion models
    _enums.ModelType.SD: "PNDMScheduler",
    _enums.ModelType.PIX2PIX: "PNDMScheduler",
    _enums.ModelType.SDXL: "DPMSolverMultistepScheduler",
    _enums.ModelType.SDXL_PIX2PIX: "DPMSolverMultistepScheduler",
    _enums.ModelType.KOLORS: "DPMSolverMultistepScheduler",

    # Upscaler models
    _enums.ModelType.UPSCALER_X2: "EulerDiscreteScheduler",
    _enums.ModelType.UPSCALER_X4: "PNDMScheduler",

    # Stable Diffusion 3
    _enums.ModelType.SD3: "FlowMatchEulerDiscreteScheduler",
    _enums.ModelType.SD3_PIX2PIX: "FlowMatchEulerDiscreteScheduler",

    # Stable Cascade models
    _enums.ModelType.S_CASCADE: "DDPMWuerstchenScheduler",
    _enums.ModelType.S_CASCADE_DECODER: "DDPMWuerstchenScheduler",

    # DeepFloyd IF models - all use DDPMScheduler
    _enums.ModelType.IF: "DDPMScheduler",
    _enums.ModelType.IFS: "DDPMScheduler",
    _enums.ModelType.IFS_IMG2IMG: "DDPMScheduler",

    # Flux models
    _enums.ModelType.FLUX: "FlowMatchEulerDiscreteScheduler",
    _enums.ModelType.FLUX_FILL: "FlowMatchEulerDiscreteScheduler"
}


def _extract_scheduler_name(scheduler_uri: str) -> str:
    """
    Extract the scheduler name from the dgenerate scheduler URI.
    """
    # Remove any URI parameters (everything after a semicolon)
    if ";" in scheduler_uri:
        scheduler_uri = scheduler_uri.split(";")[0].strip()

    return scheduler_uri


def _calculate_file_hash(file_path: str, length: int = 10) -> str:
    hasher = hashlib.sha256()
    file_size = os.path.getsize(file_path)
    chunk_size = _memory.calculate_chunk_size(file_size)
    with open(file_path, 'rb') as f:
        for chunk in tqdm.tqdm(
                iter(lambda: f.read(chunk_size), b''),
                desc=f'Hashing for Auto1111 Metadata: {file_path}',
                total=file_size // chunk_size,
                unit='chunk'
        ):
            hasher.update(chunk)
    return hasher.hexdigest()[:length]


class _ParseOnlyInvokerParseHalt(Exception):
    pass


class _ParseOnlyInvoker:
    """
    Intercepts dgenerate config invocation to extract arguments.
    """
    args: _arguments.DgenerateArguments | None

    def __init__(self):
        self.args = None

    def __call__(self, command_line, parsed_args):
        args_wrapped = \
            _textprocessing.wrap(
                command_line,
                width=_textprocessing.long_text_wrap_width()) + '\n'

        _messages.debug_log(
            f"performing Automatic1111 metadata "
            f"conversion on dgenerate config:\n\n{args_wrapped}",
            underline=True
        )

        self.args = _arguments.parse_args(parsed_args)
        # no invocations, just the first
        raise _ParseOnlyInvokerParseHalt()


def _get_auto1111_sampler(args: _arguments.DgenerateArguments) -> str | None:
    if args.scheduler_uri:
        # Extract the scheduler name from the URI
        scheduler_name = _extract_scheduler_name(args.scheduler_uri[0])

        # Look up the Automatic1111 sampler name from the mapping
        automatic1111_sampler = _SCHEDULER_TO_AUTOMATIC1111.get(scheduler_name)

    else:
        # If no scheduler URI is provided, use the default mapping
        scheduler_name = _DEFAULT_MODEL_TYPE_TO_SCHEDULER_MAP.get(args.model_type)

        model_type_string = dgenerate.pipelinewrapper.enums.get_model_type_string(args.model_type)

        if scheduler_name is not None:
            _messages.debug_log(
                f"Using default scheduler / sampler for --model-type {model_type_string}, {scheduler_name}.")

            # Look up the Automatic1111 sampler name from the mapping
            automatic1111_sampler = _SCHEDULER_TO_AUTOMATIC1111.get(scheduler_name)
        else:
            _messages.debug_log(
                f"Could not find default scheduler / sampler for --model-type {model_type_string}"
            )
            automatic1111_sampler = None

    if automatic1111_sampler is None:
        if scheduler_name is not None:
            # If not found, use original name
            _messages.debug_log(f"Using original scheduler name '{scheduler_name}' for metadata.")
            automatic1111_sampler = scheduler_name
        else:
            # Both automatic1111_sampler and scheduler_name are None
            _messages.debug_log("No scheduler / sampler name available for metadata.")
            return None
    else:
        _messages.debug_log(f"Using Automatic1111 sampler name '{automatic1111_sampler}' for metadata.")

    return automatic1111_sampler


def _config_to_automatic1111_dict(config: str, local_files_only: bool) -> typing.Dict[str, any]:
    """
    Parse a dgenerate config using dgenerate's shell and convert it to
    a dictionary of Automaticc1111 image metadata parameters.
    """
    parameters = {}

    # Create a parse only invoker to capture the arguments
    parse_only_invoker = _ParseOnlyInvoker()

    # Run the config with our parse only invoker
    # disable directives to avoid processing any
    # of the directives built into the base shell

    runner = _batchprocessor.BatchProcessor(
        invoker=parse_only_invoker,
        name='dgenerate',
        version=_resources.version(),
        disable_directives=True
    )

    try:
        runner.run_string(config)
    except _ParseOnlyInvokerParseHalt:
        # only accept one invocation from
        # the configuration
        pass

    if not parse_only_invoker.args:
        _messages.debug_log("No arguments found in dgenerate config.")
        return parameters

    args = parse_only_invoker.args

    # Extra data for CivitAI in particular,
    # appended at the end as "Civitai resources"
    civit_ai_resources = []

    def handle_civit_ai_resource(type, uri):
        civit_ai_id = _extract_civitai_id(uri)
        if civit_ai_id is not None:
            civit_ai_resources.append({"type": type, "modelVersionId": civit_ai_id})

    # Extract model information
    if args.model_path:
        handle_civit_ai_resource('checkpoint', args.model_path)

        model_name, model_hash = _process_model_path(
            model_title='Primary',
            model_path=args.model_path,
            local_files_only=local_files_only
        )
        if model_hash:
            parameters["Model hash"] = model_hash
        if model_name:
            parameters["Model"] = model_name

    # Extract prompts
    if args.prompts:
        # The first prompt's positive part
        parameters["Prompt"] = args.prompts[0].positive
        # The first prompt's negative part
        if args.prompts[0].negative:
            parameters["Negative prompt"] = args.prompts[0].negative

    # Extract generation parameters
    if args.guidance_scales:
        parameters["CFG scale"] = args.guidance_scales[0]
    if args.inference_steps:
        parameters["Steps"] = args.inference_steps[0]
    if args.seeds:
        parameters["Seed"] = args.seeds[0]

    sampler = _get_auto1111_sampler(args)

    if sampler is not None:
        parameters["Sampler"] = sampler

    # Extract VAE information
    if args.vae_uri:
        handle_civit_ai_resource('vae', args.vae_uri)

        vae_uri = _uris.VAEUri.parse(args.vae_uri)
        vae_name, vae_hash = _process_model_path(
            model_title='VAE',
            model_path=vae_uri.model,
            local_files_only=local_files_only
        )
        if vae_name:
            parameters["VAE"] = vae_name
        if vae_hash:
            parameters["VAE hash"] = vae_hash

    # Extract Text encoder information
    if args.text_encoder_uris:
        text_encoder_info = {}
        for text_encoder_uri_str in args.text_encoder_uris:
            handle_civit_ai_resource('encoder', text_encoder_uri_str)

            text_encoder_uri = _uris.TextEncoderUri.parse(text_encoder_uri_str)
            text_encoder_name, text_encoder_hash = _process_model_path(
                model_title='TextEncoder',
                model_path=text_encoder_uri.model,
                local_files_only=local_files_only
            )
            if text_encoder_name and text_encoder_hash:
                text_encoder_info[text_encoder_name] = text_encoder_hash

        if text_encoder_info:
            parameters["TextEncoder hashes"] = _textprocessing.quote(
                ", ".join([f"{name}: {hash_val}" for name, hash_val in text_encoder_info.items()])
            )

    # Extract LoRA information
    if args.lora_uris:
        lora_info = {}
        for lora_uri_str in args.lora_uris:
            handle_civit_ai_resource('lora', lora_uri_str)

            lora_uri = _uris.LoRAUri.parse(lora_uri_str)
            lora_name, lora_hash = _process_model_path(
                model_title='LoRA',
                model_path=lora_uri.model,
                local_files_only=local_files_only
            )
            if lora_name and lora_hash:
                lora_info[lora_name] = lora_hash

        if lora_info:
            parameters["Lora hashes"] = _textprocessing.quote(
                ", ".join([f"{name}: {hash_val}" for name, hash_val in lora_info.items()])
            )

    # Extract embeddings information
    if args.textual_inversion_uris:
        embedding_info = {}
        for embedding_uri_str in args.textual_inversion_uris:
            handle_civit_ai_resource('embed', embedding_uri_str)

            embedding_uri = _uris.TextualInversionUri.parse(embedding_uri_str)
            embedding_name, embedding_hash = _process_model_path(
                model_title='TextualInversion',
                model_path=embedding_uri.model,
                local_files_only=local_files_only
            )
            if embedding_name and embedding_hash:
                embedding_info[embedding_name] = embedding_hash

        if embedding_info:
            parameters["Embedding hashes"] = _textprocessing.quote(
                ", ".join([f"{name}: {hash_val}" for name, hash_val in embedding_info.items()])
            )

    # Extract ControlNet information
    if args.controlnet_uris:
        controlnet_info = {}
        for controlnet_uri_str in args.controlnet_uris:
            handle_civit_ai_resource('controlnet', controlnet_uri_str)

            controlnet_uri = _uris.ControlNetUri.parse(controlnet_uri_str)
            controlnet_name, controlnet_hash = _process_model_path(
                model_title='ControlNet',
                model_path=controlnet_uri.model,
                local_files_only=local_files_only
            )
            if controlnet_name and controlnet_hash:
                controlnet_info[controlnet_name] = controlnet_hash

        if controlnet_info:
            parameters["ControlNet hashes"] = _textprocessing.quote(
                ", ".join([f"{name}: {hash_val}" for name, hash_val in controlnet_info.items()])
            )

    # Check for clip skip
    if args.clip_skips:
        parameters["Clip skip"] = args.clip_skips[0]

    if civit_ai_resources:
        parameters['Civitai resources'] = json.dumps(civit_ai_resources)

    return parameters


def _get_dgenerate_metadata_from_image(img: PIL.Image.Image):
    """
    Retrieve any metadata that dgenerate has stored in an image.

    :param img: The open PIL image.

    :return: Decoded metadata string if found, otherwise None.
    """
    fmt = img.format.upper() if img.format else None

    if fmt == "JPEG":
        try:
            user_comment = _image.read_jpeg_exif_user_comment(img)

            if not user_comment:
                _messages.debug_log("No user comment found in JPEG EXIF data.")

            return user_comment

        except Exception as e:
            _messages.debug_log(f"Error reading EXIF data: {e}")
            return None

    elif fmt == "PNG":
        user_comment = img.info.get("DgenerateConfig", None)

        if not user_comment:
            _messages.debug_log("DgenerateConfig not found in PNG metadata.")

        return user_comment
    else:
        _messages.debug_log(f"Unsupported or unknown input image format: {fmt}")
        return None


def _add_exif_to_image(
        image_path: str,
        local_files_only: bool,
        output_path: typing.Optional[str] = None,
        dgenerate_config: typing.Optional[str] = None
):
    """
    Add Automatic1111 EXIF data to an image by parsing from a dgenerate config file.

    This config file is produced by dgenerate's --output-configs option, or stored
    in the image metadata by dgenerate's --output-metadata option.
    """

    if not output_path:
        output_path = image_path
    else:
        output_path_dir = os.path.dirname(output_path)
        if output_path_dir:
            os.makedirs(output_path_dir, exist_ok=True)

    # Start with existing metadata if requested
    parameters = {}

    # Open the image to get its dimensions
    try:
        img = PIL.Image.open(image_path)
        width, height = img.size
    except Exception as e:
        raise Auto1111MetadataCreationError(f'Failed to open image "{image_path}": {e}')

    try:
        # If dgenerate config is provided, parse it
        if dgenerate_config:
            config_params = _config_to_automatic1111_dict(
                dgenerate_config, local_files_only=local_files_only
            )
            if config_params:
                _messages.debug_log(f"Found {len(config_params)} applicable parameters in config text.")
                parameters.update(config_params)
            else:
                raise Auto1111MetadataCreationError("No parameters found in config text.")

        else:
            _messages.debug_log(f'Reading dgenerate config image metadata in: "{image_path}"')
            # Try to get the dgenerate config it from the image metadata
            image_metadata = _get_dgenerate_metadata_from_image(img)

            if image_metadata:
                config_params = _config_to_automatic1111_dict(
                    image_metadata, local_files_only=local_files_only
                )
                if config_params:
                    _messages.debug_log(f"Found {len(config_params)} applicable parameters in image metadata.")
                    parameters.update(config_params)
                else:
                    raise Auto1111MetadataCreationError("No parameters found in image metadata.")
            else:
                raise Auto1111MetadataCreationError("No dgenerate metadata found in image.")

        # Add size parameter if not already present
        if "Size" not in parameters:
            parameters["Size"] = f"{width}x{height}"

        # Build the Automatic1111 format string
        # First line: positive prompt
        positive_prompt = ""
        if "Prompt" in parameters:
            positive_prompt = parameters.pop("Prompt")

        # Second line: Negative prompt: [content]
        negative_prompt = ""
        if "Negative prompt" in parameters:
            negative_prompt = parameters.pop("Negative prompt")

        # Collect all other parameters in key: value format
        param_parts = []
        # Ensure Steps is first if it exists
        if "Steps" in parameters:
            steps_value = parameters.pop("Steps")
            param_parts.append(f"Steps: {steps_value}")

        # Then add all other parameters
        for key, value in parameters.items():
            param_parts.append(f"{key}: {value}")

        # Assemble the metadata as three distinct lines
        # This matches Automatic1111 format
        metadata_lines = [
            positive_prompt,
            f"Negative prompt: {negative_prompt}" if negative_prompt else "Negative prompt:",
            ", ".join(param_parts)
        ]

        # Join with actual newlines for proper formatting
        parameters_str = "\n".join(metadata_lines)

        # Print the metadata string for debugging
        _messages.debug_log("Generated metadata:")
        _messages.debug_log("---")
        _messages.debug_log(parameters_str)
        _messages.debug_log("---")

        # Determine image format
        img_format = os.path.splitext(output_path)[1].lower().replace('.', '')

        # Check for supported formats
        if img_format not in ('jpg', 'jpeg', 'png'):
            raise Auto1111MetadataCreationError(
                f"Unsupported output format: {img_format}. Only jpg, jpeg, and png are supported.")

        # If the image is a JPEG, add EXIF data
        if img_format in ('jpg', 'jpeg'):
            try:
                img.save(
                    output_path,
                    exif=_image.create_jpeg_exif_with_user_comment(parameters_str)
                )
                _messages.debug_log(f'Added EXIF metadata to "{output_path}" using Automatic1111 format.')
            except Exception as e:
                raise Auto1111MetadataCreationError(
                    f"Error writing EXIF data: {e}"
                )

        # If the image is a PNG, add metadata using PngInfo
        elif img_format == 'png':
            try:
                # Create PNG metadata
                metadata = PIL.PngImagePlugin.PngInfo()
                metadata.add_text("parameters", parameters_str)

                # Save image with metadata
                img.save(output_path, format='PNG', pnginfo=metadata)
                _messages.debug_log(f'Added PNG text metadata to "{output_path}" using Automatic1111 format.')
            except Exception as e:
                raise Auto1111MetadataCreationError(
                    f"Error writing PNG metadata: {e}"
                )

    except Exception as e:
        raise Auto1111MetadataCreationError(
            f"Unexpected error processing metadata: {e}"
        )
    finally:
        # Make sure to close the image
        if 'img' in locals():
            img.close()


def get_checkpoint_hash_cache() -> str:
    """
    Get the default model hash cache directory.

    Checkpoint hashes are used for Automatic1111 metadata to provide information about
    the models involved in a generation, this information is cached for performance.

    Or the value of the environmental variable ``DGENERATE_CACHE`` joined with ``auto1111_metadata/cache.db``.

    :return: string (directory path)
    """
    user_cache_path = os.environ.get('DGENERATE_CACHE')

    if user_cache_path is not None:
        path = os.path.join(user_cache_path, 'auto1111_metadata')
    else:
        path = os.path.expanduser(os.path.join('~', '.cache', 'dgenerate', 'auto1111_metadata'))

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return os.path.join(path, 'cache.db')


_checkpoint_hash_cache = _filecache.KeyValueStore(get_checkpoint_hash_cache())


class _ModelMimetypeException(Exception):
    pass


def _clean_url(url):
    # colons in model names break civitai
    # cannot actually get the model name,
    # so want to use the URL, need to make
    # sure it is clean

    parsed = urllib.parse.urlparse(url)

    # Remove username, password, port from netloc
    hostname = parsed.hostname or ''
    netloc = hostname

    # Remove colons in path, params, query, and fragment
    path = parsed.path.replace(':', '')
    params = parsed.params.replace(':', '')
    query = parsed.query.replace(':', '')
    fragment = parsed.fragment.replace(':', '')

    # Reconstruct without scheme and sensitive info
    clean_parts = ('', netloc, path, params, query, fragment)

    cleaned = urllib.parse.urlunparse(clean_parts)

    # Strip leading // if present
    if cleaned.startswith('//'):
        cleaned = cleaned[2:]

    return cleaned


def _extract_civitai_id(url):
    parsed = urllib.parse.urlparse(url)

    # Ensure it's a Civitai download URL
    if parsed.netloc not in ("civitai.com", "www.civitai.com"):
        return None

    match = re.match(r"^/api/download/models/(\d+)$", parsed.path)
    if match:
        return int(match.group(1))

    return None


def _process_model_path(model_title: str, model_path: str, local_files_only):
    """
    Process a model path, handling URLs using web cache if necessary.

    Returns the model name and its hash if available.

    If either value is unavailable, return None for the unavailable value.
    """
    if not model_path:
        return None, None

    # Check if model path is a URL
    if _webcache.is_downloadable_url(model_path):
        model_name = _clean_url(model_path)

        with _checkpoint_hash_cache:
            if model_path in _checkpoint_hash_cache:
                return model_name, _checkpoint_hash_cache[model_path]

        # Try to get a cached version of the model
        _messages.debug_log(
            f'Attempting to retrieve {model_title} model at '
            f'"{model_path}" from dgenerate web cache, a download may occur...'
        )

        try:
            with _hfhub.with_hf_errors_as_model_not_found():
                cached_path = _hfhub.webcache_or_hf_blob_download(
                    url=model_path,
                    mime_acceptable_desc='not text',
                    mimetype_is_supported=lambda m: m is not None and not m.startswith('text/'),
                    unknown_mimetype_exception=_ModelMimetypeException,
                    local_files_only=local_files_only
                )

            if cached_path and os.path.exists(cached_path):
                _messages.debug_log(f"{model_title} model cached at: {cached_path}")
                try:
                    # Calculate hash from the cached file content
                    model_hash = _calculate_file_hash(cached_path)
                    _messages.debug_log(f"Calculated hash from cached {model_title} model: {model_hash}")

                    with _checkpoint_hash_cache:
                        _checkpoint_hash_cache[model_path] = model_hash

                except Exception as e:

                    _messages.debug_log(
                        f"Failed to calculate hash for {model_title} model: {e}"
                    )

                    model_hash = None
            else:
                _messages.debug_log(
                    f"Could not retrieve {model_title} model from cache."
                )

                # No hash if we can't access the file
                model_hash = None
        except _ModelMimetypeException as e:
            model_hash = None
            _messages.debug_log(
                f"Could not retrieve {model_title} model, invalid download MIME type: {e}")

    else:
        # Regular file path
        if os.path.exists(model_path):
            model_name = os.path.basename(model_path)
            model_name_no_ext, _ = os.path.splitext(model_name)

            # colon messes up civitai parsing, the parser
            # is very simple
            model_name_no_ext.replace(':', '_')

            with _checkpoint_hash_cache:
                if model_name in _checkpoint_hash_cache:
                    return model_name_no_ext, _checkpoint_hash_cache[model_name]

            try:
                model_hash = _calculate_file_hash(model_path)
                _messages.debug_log(
                    f"Calculated hash from local {model_title} model file: {model_hash}")

                with _checkpoint_hash_cache:
                    _checkpoint_hash_cache[model_name] = model_hash

            except Exception as e:
                _messages.debug_log(f"Failed to calculate hash for {model_title} model file: {e}")
                model_hash = None

            model_name = model_name_no_ext
        else:
            model_name = model_path.replace(':', '_')

            _messages.debug_log(
                f'{model_title} model file not found: "{model_path}", '
                f'this may be a huggingface slug.')

            model_hash = None

    return model_name, model_hash


def convert_and_insert_metadata(
        image_path: str,
        output_path: typing.Optional[str] = None,
        dgenerate_config: typing.Optional[str] = None,
        local_files_only: bool = False
):
    """
    Convert a dgenerate config to Automatic1111 metadata and add it to an image.

    This function reads the dgenerate config file or existing dgenerate
    image metadata and converts it into Automatic1111 metadata format,
    then sets it to the image's EXIF data, or to a copy of that image
    at ``output_path``.

    This operation will destroy existing EXIF data on JPEGs, and PNGs will have
    their ``parameters`` metadata field set to the Automatic1111 metadata format,
    which will overwrite anything there. The ``DgenerateConfig`` field in PNGs will
    be preserved.

    :param image_path: input image path, this can be a JPEG or PNG file.
    :param output_path: output image path, this can be a JPEG or PNG file,
        if not provided the input image will be modified.
    :param dgenerate_config: dgenerate config text produced by ``--output-configs``, in the case that the
        image does not contain metadata produced by ``--output-metadata``. This is not a file path,
        it should be the config text itself as a string.
    :param local_files_only: if ``True``, do not download any files, only use local files and cache.

    :raise Auto1111MetadataCreationError: if there is an error creating or writing the metadata.
    """
    try:
        _add_exif_to_image(
            image_path,
            local_files_only,
            output_path,
            dgenerate_config
        )
    except _uris.InvalidVaeUriError as e:
        raise Auto1111MetadataCreationError(f"Invalid VAE URI found in the config: {e}")
    except _uris.InvalidLoRAUriError as e:
        raise Auto1111MetadataCreationError(f"Invalid LoRA URI found in the config: {e}")
    except _uris.InvalidTextualInversionUriError as e:
        raise Auto1111MetadataCreationError(f"Invalid Textual Inversion URI found in the config: {e}")
    except _uris.InvalidControlNetUriError as e:
        raise Auto1111MetadataCreationError(f"Invalid ControlNet URI found in the config: {e}")
    except Exception as e:
        raise Auto1111MetadataCreationError(f"An error occurred while adding metadata: {e}")


__all__ = _types.module_all()
