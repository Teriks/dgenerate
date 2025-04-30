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
from tqdm import tqdm
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
import dgenerate.messages as _messages
import dgenerate.arguments as _arguments
import dgenerate.subcommands.subcommand as _subcommand
import dgenerate.batchprocess.util as _b_util
import dgenerate.resources as _resources
import dgenerate.mediainput as _mediainput
import dgenerate.textprocessing as _textprocessing
import dgenerate.pipelinewrapper.enums as _enums

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
    _enums.ModelType.TORCH: "PNDMScheduler",
    _enums.ModelType.TORCH_PIX2PIX: "PNDMScheduler",
    _enums.ModelType.TORCH_SDXL: "DPMSolverMultistepScheduler",
    _enums.ModelType.TORCH_SDXL_PIX2PIX: "DPMSolverMultistepScheduler",
    _enums.ModelType.TORCH_KOLORS: "DPMSolverMultistepScheduler",

    # Upscaler models
    _enums.ModelType.TORCH_UPSCALER_X2: "EulerDiscreteScheduler",
    _enums.ModelType.TORCH_UPSCALER_X4: "PNDMScheduler",

    # Stable Diffusion 3
    _enums.ModelType.TORCH_SD3: "FlowMatchEulerDiscreteScheduler",

    # Stable Cascade models
    _enums.ModelType.TORCH_S_CASCADE: "DDPMWuerstchenScheduler",
    _enums.ModelType.TORCH_S_CASCADE_DECODER: "DDPMWuerstchenScheduler",

    # DeepFloyd IF models - all use DDPMScheduler
    _enums.ModelType.TORCH_IF: "DDPMScheduler",
    _enums.ModelType.TORCH_IFS: "DDPMScheduler",
    _enums.ModelType.TORCH_IFS_IMG2IMG: "DDPMScheduler",

    # Flux models
    _enums.ModelType.TORCH_FLUX: "FlowMatchEulerDiscreteScheduler",
    _enums.ModelType.TORCH_FLUX_FILL: "FlowMatchEulerDiscreteScheduler"
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
        for chunk in tqdm(
                iter(lambda: f.read(chunk_size), b''),
                desc=f'Hashing: {file_path}',
                total=file_size // chunk_size,
                unit='chunk'
        ):
            hasher.update(chunk)
    return hasher.hexdigest()[:length]


class _ParseOnlyInvoker:
    """
    Intercepts dgenerate config invocation to extract arguments.
    """
    args: _arguments.DgenerateArguments | None

    def __init__(self):
        self.args = None

    def __call__(self, args):
        self.args = _arguments.parse_args(args)
        return 0


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
            _messages.log(
                f"Using default scheduler / sampler for --model-type {model_type_string}, {scheduler_name}.")

            # Look up the Automatic1111 sampler name from the mapping
            automatic1111_sampler = _SCHEDULER_TO_AUTOMATIC1111.get(scheduler_name)
        else:
            _messages.log(
                f"Could not find default scheduler / sampler for --model-type {model_type_string}"
            )
            automatic1111_sampler = None

    if automatic1111_sampler is None:
        if scheduler_name is not None:
            # If not found, use original name
            _messages.log(f"Using original scheduler name '{scheduler_name}' for metadata.")
            automatic1111_sampler = scheduler_name
        else:
            # Both automatic1111_sampler and scheduler_name are None
            _messages.log("No scheduler / sampler name available for metadata.")
            return None
    else:
        _messages.log(f"Using Automatic1111 sampler name '{automatic1111_sampler}' for metadata.")

    return automatic1111_sampler


def _config_to_automatic1111_dict(config: str) -> typing.Dict[str, any]:
    """
    Parse a dgenerate config using dgenerate's shell and convert it to
    a dictionary of Automaticc1111 image metadata parameters.
    """
    parameters = {}

    # Create a parse only invoker to capture the arguments
    parse_only_invoker = _ParseOnlyInvoker()

    # Run the config with our parse only invoker
    runner = _batchprocessor.BatchProcessor(
        invoker=parse_only_invoker,
        name='dgenerate',
        version=_resources.version())

    runner.run_string(config)

    if not parse_only_invoker.args:
        _messages.log("No arguments found in dgenerate config.")
        return parameters

    args = parse_only_invoker.args

    # Extract model information
    if args.model_path:
        model_name, model_hash = _process_model_path(
            model_title='Primary',
            model_path=args.model_path
        )
        if model_name:
            parameters["Model"] = model_name
        if model_hash:
            parameters["Model hash"] = model_hash

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
        vae_uri = _uris.VAEUri.parse(args.vae_uri)
        vae_name, vae_hash = _process_model_path(
            model_title='VAE',
            model_path=vae_uri.model
        )
        if vae_name:
            parameters["VAE"] = vae_name
        if vae_hash:
            parameters["VAE hash"] = vae_hash

    # Extract LoRA information
    if args.lora_uris:
        lora_info = {}
        for lora_uri_str in args.lora_uris:
            lora_uri = _uris.LoRAUri.parse(lora_uri_str)
            lora_name, lora_hash = _process_model_path(
                model_title='LoRA',
                model_path=lora_uri.model
            )
            if lora_name and lora_hash:
                lora_info[lora_name] = lora_hash

        if lora_info:
            parameters["Lora hashes"] = ", ".join([f"{name}: {hash_val}" for name, hash_val in lora_info.items()])

    # Extract embeddings information
    if args.textual_inversion_uris:
        embedding_info = {}
        for embedding_uri_str in args.textual_inversion_uris:
            embedding_uri = _uris.TextualInversionUri.parse(embedding_uri_str)
            embedding_name, embedding_hash = _process_model_path(
                model_title='TextualInversion',
                model_path=embedding_uri.model
            )
            if embedding_name and embedding_hash:
                embedding_info[embedding_name] = embedding_hash

        if embedding_info:
            parameters["Embedding hashes"] = ", ".join(
                [f"{name}: {hash_val}" for name, hash_val in embedding_info.items()])

    # Extract ControlNet information
    if args.controlnet_uris:
        controlnet_info = {}
        for controlnet_uri_str in args.controlnet_uris:
            controlnet_uri = _uris.ControlNetUri.parse(controlnet_uri_str)
            controlnet_name, controlnet_hash = _process_model_path(
                model_title='ControlNet',
                model_path=controlnet_uri.model
            )
            if controlnet_name and controlnet_hash:
                controlnet_info[controlnet_name] = controlnet_hash

        if controlnet_info:
            parameters["ControlNet hashes"] = ", ".join(
                [f"{name}: {hash_val}" for name, hash_val in controlnet_info.items()])

    # Check for clip skip
    if args.clip_skips:
        parameters["Clip skip"] = args.clip_skips[0]

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
                _messages.log("No user comment found in JPEG EXIF data.")

            return user_comment

        except Exception as e:
            _messages.log(f"Error reading EXIF data: {e}")
            return None

    elif fmt == "PNG":
        user_comment = img.info.get("DgenerateConfig", None)

        if not user_comment:
            _messages.log("DgenerateConfig not found in PNG metadata.")

        return user_comment
    else:
        _messages.log(f"Unsupported or unknown input image format: {fmt}")
        return None


def _add_exif_to_image(
        image_path: str,
        output_path: typing.Optional[str] = None,
        dgenerate_config: typing.Optional[str] = None
) -> bool:
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
        _messages.error(f"Failed to open image {image_path}: {e}")
        return False

    try:
        # If dgenerate config is provided, parse it
        if dgenerate_config:
            if not os.path.exists(dgenerate_config):
                _messages.error(f"Config file not found: {dgenerate_config}")
                return False

            with open(dgenerate_config, 'rt', encoding='utf-8') as f:
                _messages.log(f"Reading dgenerate config from: {dgenerate_config}")
                config_params = _config_to_automatic1111_dict(f.read())
                if config_params:
                    _messages.log(f"Found {len(config_params)} applicable parameters in config file.")
                    parameters.update(config_params)
                else:
                    _messages.log("No parameters found in config file.")
                    return False

        else:
            _messages.log(f"Reading dgenerate config image metadata in: {image_path}")
            # Try to get the dgenerate config it from the image metadata
            image_metadata = _get_dgenerate_metadata_from_image(img)

            if image_metadata:
                config_params = _config_to_automatic1111_dict(image_metadata)
                if config_params:
                    _messages.log(f"Found {len(config_params)} applicable parameters in image metadata.")
                    parameters.update(config_params)
                else:
                    _messages.log("No parameters found in image metadata.")
                    return False
            else:
                _messages.log("No dgenerate metadata found in image.")
                return False

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
        _messages.log("Generated metadata:")
        _messages.log("---")
        _messages.log(parameters_str)
        _messages.log("---")

        # Determine image format
        img_format = os.path.splitext(output_path)[1].lower().replace('.', '')

        # Check for supported formats
        if img_format not in ('jpg', 'jpeg', 'png'):
            _messages.error(f"Unsupported output format: {img_format}. Only jpg, jpeg, and png are supported.")
            return False

        # If the image is a JPEG, add EXIF data
        if img_format in ('jpg', 'jpeg'):
            try:
                img.save(
                    output_path,
                    exif=_image.create_jpeg_exif_with_user_comment(parameters_str)
                )
                _messages.log(f"Added EXIF metadata to {output_path} using Automatic1111 format.")
            except Exception as e:
                _messages.error(f"Error writing EXIF data: {e}")
                return False

        # If the image is a PNG, add metadata using PngInfo
        elif img_format == 'png':
            try:
                # Create PNG metadata
                metadata = PIL.PngImagePlugin.PngInfo()
                metadata.add_text("parameters", parameters_str)

                # Save image with metadata
                img.save(output_path, format='PNG', pnginfo=metadata)
                _messages.log(f"Added PNG text metadata to {output_path} using Automatic1111 format.")
            except Exception as e:
                _messages.error(f"Error writing PNG metadata: {e}")
                return False

        return True

    except Exception as e:
        _messages.error(f"Unexpected error processing metadata: {e}")
        return False
    finally:
        # Make sure to close the image
        if 'img' in locals():
            img.close()


def _process_model_path(model_title: str, model_path: str):
    """
    Process a model path, handling URLs using web cache if necessary.

    Returns the model name and its hash if available.

    If either value is unavailable, return None for the unavailable value.
    """
    if not model_path:
        return None, None

    # Check if model path is a URL
    if _webcache.is_downloadable_url(model_path):
        model_name = model_path

        # Try to get a cached version of the model
        _messages.log(
            f'Attempting to retrieve {model_title} model at '
            f'"{model_path}" from dgenerate web cache, a download may occur...'
        )

        try:
            _, cached_path = _webcache.create_web_cache_file(
                url=model_path,
                mime_acceptable_desc='not text',
                mimetype_is_supported=lambda m: m is not None and not m.startswith('text/'),
                unknown_mimetype_exception=Exception("Failed to retrieve model.")
            )

            if cached_path and os.path.exists(cached_path):
                _messages.log(f"{model_title} model cached at: {cached_path}")
                try:
                    # Calculate hash from the cached file content
                    model_hash = _calculate_file_hash(cached_path)
                    _messages.log(f"Calculated hash from cached {model_title} model: {model_hash}")
                except Exception as e:
                    _messages.log(f"Failed to calculate hash for {model_title} model: {e}")
                    model_hash = None
            else:
                _messages.log(f"Could not retrieve {model_title} model from cache.")
                # No hash if we can't access the file
                model_hash = None
        except Exception as e:
            model_hash = None
            _messages.log(f"Could not retrieve {model_title} model: {e}")

    else:
        # Regular file path
        if os.path.exists(model_path):
            model_name = os.path.basename(model_path)

            try:
                model_hash = _calculate_file_hash(model_path)
                _messages.log(
                    f"Calculated hash from local {model_title} model file: {model_hash}")
            except Exception as e:
                _messages.log(f"Failed to calculate hash for {model_title} model file: {e}")
                model_hash = None
        else:
            model_name = model_path

            _messages.log(
                f"{model_title} model file not found: {model_path}, "
                f"this may be a huggingface slug.")

            model_hash = None

    return model_name, model_hash


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

        try:
            success = _add_exif_to_image(
                args.image,
                args.output,
                args.config
            )
        except _uris.InvalidVaeUriError as e:
            _messages.error(f"Invalid VAE URI found in the config: {e}")
            return 1
        except _uris.InvalidLoRAUriError as e:
            _messages.error(f"Invalid LoRA URI found in the config: {e}")
            return 1
        except _uris.InvalidTextualInversionUriError as e:
            _messages.error(f"Invalid Textual Inversion URI found in the config: {e}")
            return 1
        except _uris.InvalidControlNetUriError as e:
            _messages.error(f"Invalid ControlNet URI found in the config: {e}")
            return 1
        except Exception as e:
            _messages.error(f"An error occurred while adding metadata: {e}")
            return 1

        if success:
            _messages.log("Metadata successfully added.")
            return 0
        else:
            _messages.error("Failed to add metadata.")
            return 1
