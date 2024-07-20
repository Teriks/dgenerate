.. _Stable_Diffusion_Web_UI_1: https://github.com/AUTOMATIC1111/stable-diffusion-webui
.. _CivitAI_1: https://civitai.com/
.. _chaiNNer_1: https://github.com/chaiNNer-org/chaiNNer

.. |Documentation| image:: https://readthedocs.org/projects/dgenerate/badge/?version=v3.10.0
   :target: http://dgenerate.readthedocs.io/en/v3.10.0/

.. |Latest Release| image:: https://img.shields.io/github/v/release/Teriks/dgenerate
   :target: https://github.com/Teriks/dgenerate/releases/latest
   :alt: GitHub Latest Release

.. |Support Dgenerate| image:: https://img.shields.io/badge/Ko–fi-support%20dgenerate%20-hotpink?logo=kofi&logoColor=white
   :target: https://ko-fi.com/teriks
   :alt: ko-fi

Overview
========

|Documentation| |Latest Release| |Support Dgenerate|

``dgenerate`` is a command line tool and library for generating images and animation sequences
using Stable Diffusion and related techniques / models. Now Featuring a `Console UI`_ and
REPL shell mode for the dgenerate configuration / scripting language.

You can use dgenerate to generate multiple images or animated outputs using multiple combinations of
diffusion input parameters in batch, so that the differences in generated output can be compared / curated easily.

Simple txt2img generation without image inputs is supported, as well as img2img and inpainting, and ControlNets.

Animated output can be produced by processing every frame of a Video, GIF, WebP, or APNG through various implementations
of diffusion in img2img or inpainting mode, as well as with ControlNets and control guidance images, in any combination thereof.
MP4 (h264) video can be written without memory constraints related to frame count. GIF, WebP, and PNG/APNG can be
written WITH memory constraints, IE: all frames exist in memory at once before being written.

Video input of any runtime can be processed without memory constraints related to the video size.
Many video formats are supported through the use of PyAV (ffmpeg).

Animated image input such as GIF, APNG (extension must be .apng), and WebP, can also be processed WITH
memory constraints, IE: all frames exist in memory at once after an animated image is read.

PNG, JPEG, JPEG-2000, TGA (Targa), BMP, and PSD (Photoshop) are supported for static image inputs.

In addition to diffusion, dgenerate also supports the processing of any supported image, video, or
animated image using any of its built in image processors, which include various edge detectors,
depth detectors, segment generation, normal map generation, pose detection, non-diffusion based AI upscaling,
and more.

This software requires an Nvidia GPU supporting CUDA 12.1+, CPU rendering is possible for
some operations but extraordinarily slow.

For library documentation, and a better README reading experience which
includes proper syntax highlighting for examples, and side panel navigation,
please visit `readthedocs <http://dgenerate.readthedocs.io/en/v3.10.0/>`_.

----

* How to install
    * `Windows Install`_
    * `Linux or WSL Install`_
    * `Google Colab Install`_

* Usage Examples
    * `Basic Usage`_
    * `Negative Prompt`_
    * `Multiple Prompts`_
    * `Image Seed`_
    * `Inpainting`_
    * `Per Image Seed Resizing`_
    * `Animated Output`_
    * `Animation Slicing`_
    * `Inpainting Animations`_
    * `Deterministic Output`_
    * `Specifying a specific GPU for CUDA`_
    * `Specifying a Scheduler (sampler)`_
    * `Specifying a VAE`_
    * `VAE Tiling and Slicing`_
    * `Specifying a UNet`_
    * `Specifying an SDXL Refiner`_
    * `Specifying a Stable Cascade Decoder`_
    * `Specifying LoRAs`_
    * `Specifying Textual Inversions`_
    * `Specifying Control Nets`_
    * `Specifying Text Encoders`_
    * `Prompt Weighting and Enhancement`_
        * `The compel prompt weighter`_
        * `The sd-embed prompt weighter`_
    * `Utilizing CivitAI links and Other Hosted Models`_
    * `Specifying Generation Batch Size`_
    * `Image Processors`_
    * `Sub Commands`_
        * `--sub-command image-process`_
        * `--sub-command civitai-links`_
    * `Upscaling`_
        * `Upscaling with Diffusion Upscaler Models`_
        * `Upscaling with chaiNNer Compatible Upscaler Models`_
        * `Upscaling with NCNN Upscaler Models`_
    * `Writing and Running Configs`_
        * `Basic config syntax`_
        * `Built in template variables and functions`_
        * `Directives, and applying templating`_
        * `Setting template variables, in depth`_
        * `Setting environmental variables, in depth`_
        * `Globbing and path manipulation`_
        * `The \\print and \\echo directive`_
        * `The \\image_process directive`_
        * `The \\exec directive`_
        * `The \\download directive`_
        * `The download() template function`_
        * `The \\exit directive`_
        * `Running configs from the command line`_
        * `Config argument injection`_
    * `Writing Plugins`_
        * `Image processor plugins`_
        * `Config directive and template function plugins`_
        * `Sub-command plugins`_
        * `Prompt weighter plugins`_
    * `Console UI`_
    * `File Cache Control`_

Help Output
-----------

.. code-block:: text

    usage: dgenerate [-h] [-v] [--version] [--file | --shell | --no-stdin | --console]
                     [--plugin-modules PATH [PATH ...]] [--sub-command SUB_COMMAND]
                     [--sub-command-help [SUB_COMMAND ...]] [-ofm] [--templates-help [VARIABLE_NAME ...]]
                     [--directives-help [DIRECTIVE_NAME ...]] [--functions-help [FUNCTION_NAME ...]]
                     [-mt MODEL_TYPE] [-rev BRANCH] [-var VARIANT] [-sbf SUBFOLDER] [-atk TOKEN] [-bs INTEGER]
                     [-bgs SIZE] [-te TEXT_ENCODER_URIS [TEXT_ENCODER_URIS ...]]
                     [-te2 TEXT_ENCODER_URIS [TEXT_ENCODER_URIS ...]] [-un UNET_URI] [-un2 UNET_URI]
                     [-vae VAE_URI] [-vt] [-vs] [-lra LORA_URI [LORA_URI ...]] [-ti URI [URI ...]]
                     [-cn CONTROL_NET_URI [CONTROL_NET_URI ...]] [-sch SCHEDULER_URI] [-mqo | -mco]
                     [--s-cascade-decoder MODEL_URI] [-dqo] [-dco]
                     [--s-cascade-decoder-prompts PROMPT [PROMPT ...]]
                     [--s-cascade-decoder-inference-steps INTEGER [INTEGER ...]]
                     [--s-cascade-decoder-guidance-scales INTEGER [INTEGER ...]]
                     [--s-cascade-decoder-scheduler SCHEDULER_URI] [--sdxl-refiner MODEL_URI] [-rqo] [-rco]
                     [--sdxl-refiner-scheduler SCHEDULER_URI] [--sdxl-refiner-edit]
                     [--sdxl-second-prompts PROMPT [PROMPT ...]] [--sdxl-aesthetic-scores FLOAT [FLOAT ...]]
                     [--sdxl-crops-coords-top-left COORD [COORD ...]] [--sdxl-original-size SIZE [SIZE ...]]
                     [--sdxl-target-size SIZE [SIZE ...]] [--sdxl-negative-aesthetic-scores FLOAT [FLOAT ...]]
                     [--sdxl-negative-original-sizes SIZE [SIZE ...]]
                     [--sdxl-negative-target-sizes SIZE [SIZE ...]]
                     [--sdxl-negative-crops-coords-top-left COORD [COORD ...]]
                     [--sdxl-refiner-prompts PROMPT [PROMPT ...]]
                     [--sdxl-refiner-clip-skips INTEGER [INTEGER ...]]
                     [--sdxl-refiner-second-prompts PROMPT [PROMPT ...]]
                     [--sdxl-refiner-aesthetic-scores FLOAT [FLOAT ...]]
                     [--sdxl-refiner-crops-coords-top-left COORD [COORD ...]]
                     [--sdxl-refiner-original-sizes SIZE [SIZE ...]]
                     [--sdxl-refiner-target-sizes SIZE [SIZE ...]]
                     [--sdxl-refiner-negative-aesthetic-scores FLOAT [FLOAT ...]]
                     [--sdxl-refiner-negative-original-sizes SIZE [SIZE ...]]
                     [--sdxl-refiner-negative-target-sizes SIZE [SIZE ...]]
                     [--sdxl-refiner-negative-crops-coords-top-left COORD [COORD ...]] [-hnf FLOAT [FLOAT ...]]
                     [-ri INT [INT ...]] [-rg FLOAT [FLOAT ...]] [-rgr FLOAT [FLOAT ...]] [-sc] [-d DEVICE]
                     [-t DTYPE] [-s SIZE] [-na] [-o PATH] [-op PREFIX] [-ox] [-oc] [-om]
                     [-pw PROMPT_WEIGHTER_URI] [--prompt-weighter-help [PROMPT_WEIGHTER_NAMES ...]]
                     [-p PROMPT [PROMPT ...]] [--sd3-max-sequence-length INTEGER]
                     [--sd3-second-prompts PROMPT [PROMPT ...]] [--sd3-third-prompts PROMPT [PROMPT ...]]
                     [-cs INTEGER [INTEGER ...]] [-se SEED [SEED ...]] [-sei] [-gse COUNT] [-af FORMAT]
                     [-if FORMAT] [-nf] [-fs FRAME_NUMBER] [-fe FRAME_NUMBER] [-is SEED [SEED ...]]
                     [-sip PROCESSOR_URI [PROCESSOR_URI ...]] [-mip PROCESSOR_URI [PROCESSOR_URI ...]]
                     [-cip PROCESSOR_URI [PROCESSOR_URI ...]] [--image-processor-help [PROCESSOR_NAME ...]]
                     [-pp PROCESSOR_URI [PROCESSOR_URI ...]] [-iss FLOAT [FLOAT ...] | -uns INTEGER
                     [INTEGER ...]] [-gs FLOAT [FLOAT ...]] [-igs FLOAT [FLOAT ...]] [-gr FLOAT [FLOAT ...]]
                     [-ifs INTEGER [INTEGER ...]] [-mc EXPR [EXPR ...]] [-pmc EXPR [EXPR ...]]
                     [-umc EXPR [EXPR ...]] [-vmc EXPR [EXPR ...]] [-cmc EXPR [EXPR ...]] [-tmc EXPR [EXPR ...]]
                     model_path

    Batch image generation and manipulation tool supporting Stable Diffusion and related techniques /
    algorithms, with support for video and animated image processing.

    positional arguments:
      model_path            huggingface model repository slug, huggingface blob link to a model file, path to
                            folder on disk, or path to a .pt, .pth, .bin, .ckpt, or .safetensors file.

    options:
      -h, --help            show this help message and exit
      -v, --verbose         Output information useful for debugging, such as pipeline call and model load
                            parameters.
      --version             Show dgenerate's version and exit
      --file                Convenience argument for reading a configuration script from a file instead of using
                            a pipe. This is a meta argument which can not be used within a configuration script
                            and is only valid from the command line or during a popen invocation of dgenerate.
      --shell               When reading configuration from STDIN (a pipe), read forever, even when
                            configuration errors occur. This allows dgenerate to run in the background and be
                            communicated with by another process sending it commands. Launching dgenerate with
                            this option and not piping it input will attach it to the terminal like a shell.
                            Entering configuration into this shell will require two newlines to submit a command
                            due to parsing lookahead. IE: two presses of the enter key. This is a meta argument
                            which can not be used within a configuration script and is only valid from the
                            command line or during a popen invocation of dgenerate.
      --no-stdin            Can be used to indicate to dgenerate that it will not receive any piped in input.
                            This is useful for running dgenerate via popen from python or another application
                            using normal arguments, where it would otherwise try to read from STDIN and block
                            forever because it is not attached to a terminal. This is a meta argument which can
                            not be used within a configuration script and is only valid from the command line or
                            during a popen invocation of dgenerate.
      --console             Launch a terminal-like tkinter GUI that communicates with an instance of dgenerate
                            running in the background. This allows you to interactively write dgenerate config
                            scripts as if dgenerate were a shell / REPL. This is a meta argument which can not
                            be used within a configuration script and is only valid from the command line or
                            during a popen invocation of dgenerate.
      --plugin-modules PATH [PATH ...]
                            Specify one or more plugin module folder paths (folder containing __init__.py) or
                            python .py file paths, or python module names to load as plugins. Plugin modules can
                            currently implement image processors, config directives, config template functions,
                            prompt weighters, and sub-commands.
      --sub-command SUB_COMMAND
                            Specify the name a sub-command to invoke. dgenerate exposes some extra image
                            processing functionality through the use of sub-commands. Sub commands essentially
                            replace the entire set of accepted arguments with those of a sub-command which
                            implements additional functionality. See --sub-command-help for a list of sub-
                            commands and help.
      --sub-command-help [SUB_COMMAND ...]
                            Use this option alone (or with --plugin-modules) and no model specification in order
                            to list available sub-command names. Calling a subcommand with "--sub-command name
                            --help" will produce argument help output for that subcommand. When used with
                            --plugin-modules, sub-commands implemented by the specified plugins will also be
                            listed.
      -ofm, --offline-mode  Whether dgenerate should try to download huggingface models that do not exist in the
                            disk cache, or only use what is available in the cache. Referencing a model on
                            huggingface that has not been cached because it was not previously downloaded will
                            result in a failure when using this option.
      --templates-help [VARIABLE_NAME ...]
                            Print a list of template variables available in the interpreter environment used for
                            dgenerate config scripts, particularly the variables set after a dgenerate
                            invocation occurs. When used as a command line option, their values are not
                            presented, just their names and types. Specifying names will print type information
                            for those variable names.
      --directives-help [DIRECTIVE_NAME ...]
                            Use this option alone (or with --plugin-modules) and no model specification in order
                            to list available config directive names. Providing names will print documentation
                            for the specified directive names. When used with --plugin-modules, directives
                            implemented by the specified plugins will also be listed.
      --functions-help [FUNCTION_NAME ...]
                            Use this option alone (or with --plugin-modules) and no model specification in order
                            to list available config template function names. Providing names will print
                            documentation for the specified function names. When used with --plugin-modules,
                            functions implemented by the specified plugins will also be listed.
      -mt MODEL_TYPE, --model-type MODEL_TYPE
                            Use when loading different model types. Currently supported: torch, torch-pix2pix,
                            torch-sdxl, torch-sdxl-pix2pix, torch-upscaler-x2, torch-upscaler-x4, torch-if,
                            torch-ifs, torch-ifs-img2img, torch-s-cascade, or torch-sd3. (default: torch)
      -rev BRANCH, --revision BRANCH
                            The model revision to use when loading from a huggingface repository, (The git
                            branch / tag, default is "main")
      -var VARIANT, --variant VARIANT
                            If specified when loading from a huggingface repository or folder, load weights from
                            "variant" filename, e.g. "pytorch_model.<variant>.safetensors". Defaults to
                            automatic selection. This option is ignored if using flax.
      -sbf SUBFOLDER, --subfolder SUBFOLDER
                            Main model subfolder. If specified when loading from a huggingface repository or
                            folder, load weights from the specified subfolder.
      -atk TOKEN, --auth-token TOKEN
                            Huggingface auth token. Required to download restricted repositories that have
                            access permissions granted to your huggingface account.
      -bs INTEGER, --batch-size INTEGER
                            The number of image variations to produce per set of individual diffusion parameters
                            in one rendering step simultaneously on a single GPU. When using flax, batch size is
                            controlled by the environmental variable CUDA_VISIBLE_DEVICES which is a comma
                            separated list of GPU device numbers (as listed by nvidia-smi). Usage of this
                            argument with --model-type flax* will cause an error, diffusion with flax will
                            generate an image on every GPU that is visible to CUDA and this is currently
                            unchangeable. When generating animations with a --batch-size greater than one, a
                            separate animation (with the filename suffix "animation_N") will be written to for
                            each image in the batch. If --batch-grid-size is specified when producing an
                            animation then the image grid is used for the output frames. During animation
                            rendering each image in the batch will still be written to the output directory
                            along side the produced animation as either suffixed files or image grids depending
                            on the options you choose. (Torch Default: 1)
      -bgs SIZE, --batch-grid-size SIZE
                            Produce a single image containing a grid of images with the number of COLUMNSxROWS
                            given to this argument when --batch-size is greater than 1, or when using flax with
                            multiple GPUs visible (via the environmental variable CUDA_VISIBLE_DEVICES). If not
                            specified with a --batch-size greater than 1, images will be written individually
                            with an image number suffix (image_N) in the filename signifying which image in the
                            batch they are.
      -te TEXT_ENCODER_URIS [TEXT_ENCODER_URIS ...], --text-encoders TEXT_ENCODER_URIS [TEXT_ENCODER_URIS ...]
                            Specify Text Encoders for the main model using URIs, main models may use one or more
                            text encoders depending on the --model-type value and other dgenerate arguments.
                            See: --text-encoders help for information about what text encoders are needed for
                            your invocation. Examples: "CLIPTextModel;model=huggingface/text_encoder",
                            "CLIPTextModelWithProjection;model=huggingface/text_encoder;revision=main",
                            "T5TextModel;model=text_encoder_folder_on_disk". Or for Flax:
                            "FlaxCLIPTextModel;model=huggingface/text_encoder". For main models which require
                            multiple text encoders, the + symbol may be used to indicate that a default value
                            should be used for a particular text encoder, for example: --text-encoders + +
                            huggingface/encoder3. Any trailing text encoders which are not specified are given
                            their default value. The value "null" may be used to indicate that a specific text
                            encoder should not be loaded Blob links / single file loads are not supported for
                            Text Encoders. The "revision" argument specifies the model revision to use for the
                            Text Encoder when loading from huggingface repository, (The git branch / tag,
                            default is "main"). The "variant" argument specifies the Text Encoder model variant,
                            it is only supported for torch type models it is not supported for flax. If
                            "variant" is specified when loading from a huggingface repository or folder, weights
                            will be loaded from "variant" filename, e.g. "pytorch_model.<variant>.safetensors.
                            "variant" defaults to the value of --variant if it is not specified in the URI. The
                            "subfolder" argument specifies the UNet model subfolder, if specified when loading
                            from a huggingface repository or folder, weights from the specified subfolder. The
                            "dtype" argument specifies the Text Encoder model precision, it defaults to the
                            value of -t/--dtype and should be one of: auto, bfloat16, float16, or float32. If
                            you wish to load weights directly from a path on disk, you must point this argument
                            at the folder they exist in, which should also contain the config.json file for the
                            Text Encoder. For example, a downloaded repository folder from huggingface.
      -te2 TEXT_ENCODER_URIS [TEXT_ENCODER_URIS ...], --text-encoders2 TEXT_ENCODER_URIS [TEXT_ENCODER_URIS ...]
                            --text-encoders but for the SDXL refiner or Stable Cascade decoder model.
      -un UNET_URI, --unet UNET_URI
                            Specify a UNet using a URI. Examples: "huggingface/unet",
                            "huggingface/unet;revision=main", "unet_folder_on_disk". Blob links / single file
                            loads are not supported for UNets. The "revision" argument specifies the model
                            revision to use for the UNet when loading from huggingface repository or blob link,
                            (The git branch / tag, default is "main"). The "variant" argument specifies the UNet
                            model variant, it is only supported for torch type models it is not supported for
                            flax. If "variant" is specified when loading from a huggingface repository or
                            folder, weights will be loaded from "variant" filename, e.g.
                            "pytorch_model.<variant>.safetensors. "variant" defaults to the value of --variant
                            if it is not specified in the URI. The "subfolder" argument specifies the UNet model
                            subfolder, if specified when loading from a huggingface repository or folder,
                            weights from the specified subfolder. The "dtype" argument specifies the UNet model
                            precision, it defaults to the value of -t/--dtype and should be one of: auto,
                            bfloat16, float16, or float32. If you wish to load weights directly from a path on
                            disk, you must point this argument at the folder they exist in, which should also
                            contain the config.json file for the UNet. For example, a downloaded repository
                            folder from huggingface.
      -un2 UNET_URI, --unet2 UNET_URI
                            Specify a second UNet, this is only valid when using SDXL or Stable Cascade model
                            types. This UNet will be used for the SDXL refiner, or Stable Cascade decoder model.
      -vae VAE_URI, --vae VAE_URI
                            Specify a VAE using a URI. When using torch models the URI syntax is:
                            "AutoEncoderClass;model=(huggingface repository slug/blob link or file/folder
                            path)". Examples: "AutoencoderKL;model=vae.pt",
                            "AsymmetricAutoencoderKL;model=huggingface/vae",
                            "AutoencoderTiny;model=huggingface/vae",
                            "ConsistencyDecoderVAE;model=huggingface/vae". When using a Flax model, there is
                            currently only one available encoder class:
                            "FlaxAutoencoderKL;model=huggingface/vae". The AutoencoderKL encoder class accepts
                            huggingface repository slugs/blob links, .pt, .pth, .bin, .ckpt, and .safetensors
                            files. Other encoders can only accept huggingface repository slugs/blob links, or a
                            path to a folder on disk with the model configuration and model file(s). If an
                            AutoencoderKL VAE model file exists at a URL which serves the file as a raw
                            download, you may provide an http/https link to it and it will be downloaded to
                            dgenerates web cache. Aside from the "model" argument, there are four other optional
                            arguments that can be specified, these include "revision", "variant", "subfolder",
                            "dtype". They can be specified as so in any order, they are not positional: "Autoenc
                            oderKL;model=huggingface/vae;revision=main;variant=fp16;subfolder=sub_folder;dtype=f
                            loat16". The "revision" argument specifies the model revision to use for the VAE
                            when loading from huggingface repository or blob link, (The git branch / tag,
                            default is "main"). The "variant" argument specifies the VAE model variant, it is
                            only supported for torch type models it is not supported for flax. If "variant" is
                            specified when loading from a huggingface repository or folder, weights will be
                            loaded from "variant" filename, e.g. "pytorch_model.<variant>.safetensors. "variant"
                            in the case of --vae does not default to the value of --variant to prevent failures
                            during common use cases. The "subfolder" argument specifies the VAE model subfolder,
                            if specified when loading from a huggingface repository or folder, weights from the
                            specified subfolder. The "dtype" argument specifies the VAE model precision, it
                            defaults to the value of -t/--dtype and should be one of: auto, bfloat16, float16,
                            or float32. If you wish to load a weights file directly from disk, the simplest way
                            is: --vae "AutoencoderKL;my_vae.safetensors", or with a dtype
                            "AutoencoderKL;my_vae.safetensors;dtype=float16". All loading arguments except
                            "dtype" are unused in this case and may produce an error message if used. If you
                            wish to load a specific weight file from a huggingface repository, use the blob link
                            loading syntax: --vae "AutoencoderKL;https://huggingface.co/UserName/repository-
                            name/blob/main/vae_model.safetensors", the "revision" argument may be used with this
                            syntax.
      -vt, --vae-tiling     Enable VAE tiling (torch Stable Diffusion only). Assists in the generation of large
                            images with lower memory overhead. The VAE will split the input tensor into tiles to
                            compute decoding and encoding in several steps. This is useful for saving a large
                            amount of memory and to allow processing larger images. Note that if you are using
                            --control-nets you may still run into memory issues generating large images, or with
                            --batch-size greater than 1.
      -vs, --vae-slicing    Enable VAE slicing (torch Stable Diffusion models only). Assists in the generation
                            of large images with lower memory overhead. The VAE will split the input tensor in
                            slices to compute decoding in several steps. This is useful to save some memory,
                            especially when --batch-size is greater than 1. Note that if you are using
                            --control-nets you may still run into memory issues generating large images.
      -lra LORA_URI [LORA_URI ...], --loras LORA_URI [LORA_URI ...]
                            Specify one or more LoRA models using URIs (flax not supported). These should be a
                            huggingface repository slug, path to model file on disk (for example, a .pt, .pth,
                            .bin, .ckpt, or .safetensors file), or model folder containing model files. If a
                            LoRA model file exists at a URL which serves the file as a raw download, you may
                            provide an http/https link to it and it will be downloaded to dgenerates web cache.
                            huggingface blob links are not supported, see "subfolder" and "weight-name" below
                            instead. Optional arguments can be provided after a LoRA model specification, these
                            include: "scale", "revision", "subfolder", and "weight-name". They can be specified
                            as so in any order, they are not positional:
                            "huggingface/lora;scale=1.0;revision=main;subfolder=repo_subfolder;weight-
                            name=lora.safetensors". The "scale" argument indicates the scale factor of the LoRA.
                            The "revision" argument specifies the model revision to use for the LoRA when
                            loading from huggingface repository, (The git branch / tag, default is "main"). The
                            "subfolder" argument specifies the LoRA model subfolder, if specified when loading
                            from a huggingface repository or folder, weights from the specified subfolder. The
                            "weight-name" argument indicates the name of the weights file to be loaded when
                            loading from a huggingface repository or folder on disk. If you wish to load a
                            weights file directly from disk, the simplest way is: --loras "my_lora.safetensors",
                            or with a scale "my_lora.safetensors;scale=1.0", all other loading arguments are
                            unused in this case and may produce an error message if used.
      -ti URI [URI ...], --textual-inversions URI [URI ...]
                            Specify one or more Textual Inversion models using URIs (flax and SDXL not
                            supported). These should be a huggingface repository slug, path to model file on
                            disk (for example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model folder
                            containing model files. If a Textual Inversion model file exists at a URL which
                            serves the file as a raw download, you may provide an http/https link to it and it
                            will be downloaded to dgenerates web cache. huggingface blob links are not
                            supported, see "subfolder" and "weight-name" below instead. Optional arguments can
                            be provided after the Textual Inversion model specification, these include: "token",
                            "revision", "subfolder", and "weight-name". They can be specified as so in any
                            order, they are not positional:
                            "huggingface/ti_model;revision=main;subfolder=repo_subfolder;weight-
                            name=ti_model.safetensors". The "token" argument can be used to override the prompt
                            token used for the textual inversion prompt embedding. For normal Stable Diffusion
                            the default token value is provided by the model itself, but for Stable Diffusion XL
                            the default token value is equal to the model file name with no extension and all
                            spaces replaced by underscores. The "revision" argument specifies the model revision
                            to use for the Textual Inversion model when loading from huggingface repository,
                            (The git branch / tag, default is "main"). The "subfolder" argument specifies the
                            Textual Inversion model subfolder, if specified when loading from a huggingface
                            repository or folder, weights from the specified subfolder. The "weight-name"
                            argument indicates the name of the weights file to be loaded when loading from a
                            huggingface repository or folder on disk. If you wish to load a weights file
                            directly from disk, the simplest way is: --textual-inversions
                            "my_ti_model.safetensors", all other loading arguments are unused in this case and
                            may produce an error message if used.
      -cn CONTROL_NET_URI [CONTROL_NET_URI ...], --control-nets CONTROL_NET_URI [CONTROL_NET_URI ...]
                            Specify one or more ControlNet models using URIs. This should be a huggingface
                            repository slug / blob link, path to model file on disk (for example, a .pt, .pth,
                            .bin, .ckpt, or .safetensors file), or model folder containing model files. If a
                            ControlNet model file exists at a URL which serves the file as a raw download, you
                            may provide an http/https link to it and it will be downloaded to dgenerates web
                            cache. Optional arguments can be provided after the ControlNet model specification,
                            for torch these include: "scale", "start", "end", "revision", "variant",
                            "subfolder", and "dtype". For flax: "scale", "revision", "subfolder", "dtype",
                            "from_torch" (bool) They can be specified as so in any order, they are not
                            positional: "huggingface/controlnet;scale=1.0;start=0.0;end=1.0;revision=main;varian
                            t=fp16;subfolder=repo_subfolder;dtype=float16". The "scale" argument specifies the
                            scaling factor applied to the ControlNet model, the default value is 1.0. The
                            "start" (only for --model-type "torch*") argument specifies at what fraction of the
                            total inference steps to begin applying the ControlNet, defaults to 0.0, IE: the
                            very beginning. The "end" (only for --model-type "torch*") argument specifies at
                            what fraction of the total inference steps to stop applying the ControlNet, defaults
                            to 1.0, IE: the very end. The "revision" argument specifies the model revision to
                            use for the ControlNet model when loading from huggingface repository, (The git
                            branch / tag, default is "main"). The "variant" (only for --model-type "torch*")
                            argument specifies the ControlNet model variant, if "variant" is specified when
                            loading from a huggingface repository or folder, weights will be loaded from
                            "variant" filename, e.g. "pytorch_model.<variant>.safetensors. "variant" defaults to
                            automatic selection and is ignored if using flax. "variant" in the case of
                            --control-nets does not default to the value of --variant to prevent failures during
                            common use cases. The "subfolder" argument specifies the ControlNet model subfolder,
                            if specified when loading from a huggingface repository or folder, weights from the
                            specified subfolder. The "dtype" argument specifies the ControlNet model precision,
                            it defaults to the value of -t/--dtype and should be one of: auto, bfloat16,
                            float16, or float32. The "from_torch" (only for --model-type flax) this argument
                            specifies that the ControlNet is to be loaded and converted from a huggingface
                            repository or file that is designed for pytorch. (Defaults to false) If you wish to
                            load a weights file directly from disk, the simplest way is: --control-nets
                            "my_controlnet.safetensors" or --control-nets
                            "my_controlnet.safetensors;scale=1.0;dtype=float16", all other loading arguments
                            aside from "scale" and "dtype" are unused in this case and may produce an error
                            message if used ("from_torch" is available when using flax). If you wish to load a
                            specific weight file from a huggingface repository, use the blob link loading
                            syntax: --control-nets "https://huggingface.co/UserName/repository-
                            name/blob/main/controlnet.safetensors", the "revision" argument may be used with
                            this syntax.
      -sch SCHEDULER_URI, --scheduler SCHEDULER_URI
                            Specify a scheduler (sampler) by URI. Passing "help" to this argument will print the
                            compatible schedulers for a model without generating any images. Passing "helpargs"
                            will yield a help message with a list of overridable arguments for each scheduler
                            and their typical defaults. Arguments listed by "helpargs" can be overridden using
                            the URI syntax typical to other dgenerate URI arguments. Torch schedulers:
                            (DDIMScheduler, DDPMScheduler, PNDMScheduler, LMSDiscreteScheduler,
                            EulerDiscreteScheduler, HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                            DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler,
                            KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler, UniPCMultistepScheduler,
                            DPMSolverSDEScheduler, EDMEulerScheduler).
      -mqo, --model-sequential-offload
                            Force sequential model offloading for the main pipeline, this may drastically reduce
                            memory consumption and allow large models to run when they would otherwise not fit
                            in your GPUs VRAM. Inference will be much slower. Mutually exclusive with --model-
                            cpu-offload
      -mco, --model-cpu-offload
                            Force model cpu offloading for the main pipeline, this may reduce memory consumption
                            and allow large models to run when they would otherwise not fit in your GPUs VRAM.
                            Inference will be slower. Mutually exclusive with --model-sequential-offload
      --s-cascade-decoder MODEL_URI
                            Specify a Stable Cascade (torch-s-cascade) decoder model path using a URI. This
                            should be a huggingface repository slug / blob link, path to model file on disk (for
                            example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model folder containing
                            model files. Optional arguments can be provided after the decoder model
                            specification, these include: "revision", "variant", "subfolder", and "dtype". They
                            can be specified as so in any order, they are not positional: "huggingface/decoder_m
                            odel;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=float16". The
                            "revision" argument specifies the model revision to use for the decoder model when
                            loading from huggingface repository, (The git branch / tag, default is "main"). The
                            "variant" argument specifies the decoder model variant and defaults to the value of
                            --variant. When "variant" is specified when loading from a huggingface repository or
                            folder, weights will be loaded from "variant" filename, e.g.
                            "pytorch_model.<variant>.safetensors. The "subfolder" argument specifies the decoder
                            model subfolder, if specified when loading from a huggingface repository or folder,
                            weights from the specified subfolder. The "dtype" argument specifies the Stable
                            Cascade decoder model precision, it defaults to the value of -t/--dtype and should
                            be one of: auto, bfloat16, float16, or float32. If you wish to load a weights file
                            directly from disk, the simplest way is: --sdxl-refiner "my_decoder.safetensors" or
                            --sdxl-refiner "my_decoder.safetensors;dtype=float16", all other loading arguments
                            aside from "dtype" are unused in this case and may produce an error message if used.
                            If you wish to load a specific weight file from a huggingface repository, use the
                            blob link loading syntax: --s-cascade-decoder
                            "https://huggingface.co/UserName/repository-name/blob/main/decoder.safetensors", the
                            "revision" argument may be used with this syntax.
      -dqo, --s-cascade-decoder-sequential-offload
                            Force sequential model offloading for the Stable Cascade decoder pipeline, this may
                            drastically reduce memory consumption and allow large models to run when they would
                            otherwise not fit in your GPUs VRAM. Inference will be much slower. Mutually
                            exclusive with --s-cascade-decoder-cpu-offload
      -dco, --s-cascade-decoder-cpu-offload
                            Force model cpu offloading for the Stable Cascade decoder pipeline, this may reduce
                            memory consumption and allow large models to run when they would otherwise not fit
                            in your GPUs VRAM. Inference will be slower. Mutually exclusive with --s-cascade-
                            decoder-sequential-offload
      --s-cascade-decoder-prompts PROMPT [PROMPT ...]
                            One or more prompts to try with the Stable Cascade decoder model, by default the
                            decoder model gets the primary prompt, this argument overrides that with a prompt of
                            your choosing. The negative prompt component can be specified with the same syntax
                            as --prompts
      --s-cascade-decoder-inference-steps INTEGER [INTEGER ...]
                            One or more inference steps values to try with the Stable Cascade decoder. (default:
                            [10])
      --s-cascade-decoder-guidance-scales INTEGER [INTEGER ...]
                            One or more guidance scale values to try with the Stable Cascade decoder. (default:
                            [0])
      --s-cascade-decoder-scheduler SCHEDULER_URI
                            Specify a scheduler (sampler) by URI for the Stable Cascade decoder pass. Operates
                            the exact same way as --scheduler including the "help" option. Passing 'helpargs'
                            will yield a help message with a list of overridable arguments for each scheduler
                            and their typical defaults. Defaults to the value of --scheduler.
      --sdxl-refiner MODEL_URI
                            Specify a Stable Diffusion XL (torch-sdxl) refiner model path using a URI. This
                            should be a huggingface repository slug / blob link, path to model file on disk (for
                            example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model folder containing
                            model files. Optional arguments can be provided after the SDXL refiner model
                            specification, these include: "revision", "variant", "subfolder", and "dtype". They
                            can be specified as so in any order, they are not positional: "huggingface/refiner_m
                            odel_xl;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=float16". The
                            "revision" argument specifies the model revision to use for the refiner model when
                            loading from huggingface repository, (The git branch / tag, default is "main"). The
                            "variant" argument specifies the SDXL refiner model variant and defaults to the
                            value of --variant. When "variant" is specified when loading from a huggingface
                            repository or folder, weights will be loaded from "variant" filename, e.g.
                            "pytorch_model.<variant>.safetensors. The "subfolder" argument specifies the SDXL
                            refiner model subfolder, if specified when loading from a huggingface repository or
                            folder, weights from the specified subfolder. The "dtype" argument specifies the
                            SDXL refiner model precision, it defaults to the value of -t/--dtype and should be
                            one of: auto, bfloat16, float16, or float32. If you wish to load a weights file
                            directly from disk, the simplest way is: --sdxl-refiner
                            "my_sdxl_refiner.safetensors" or --sdxl-refiner
                            "my_sdxl_refiner.safetensors;dtype=float16", all other loading arguments aside from
                            "dtype" are unused in this case and may produce an error message if used. If you
                            wish to load a specific weight file from a huggingface repository, use the blob link
                            loading syntax: --sdxl-refiner "https://huggingface.co/UserName/repository-
                            name/blob/main/refiner_model.safetensors", the "revision" argument may be used with
                            this syntax.
      -rqo, --sdxl-refiner-sequential-offload
                            Force sequential model offloading for the SDXL refiner pipeline, this may
                            drastically reduce memory consumption and allow large models to run when they would
                            otherwise not fit in your GPUs VRAM. Inference will be much slower. Mutually
                            exclusive with --refiner-cpu-offload
      -rco, --sdxl-refiner-cpu-offload
                            Force model cpu offloading for the SDXL refiner pipeline, this may reduce memory
                            consumption and allow large models to run when they would otherwise not fit in your
                            GPUs VRAM. Inference will be slower. Mutually exclusive with --refiner-sequential-
                            offload
      --sdxl-refiner-scheduler SCHEDULER_URI
                            Specify a scheduler (sampler) by URI for the SDXL refiner pass. Operates the exact
                            same way as --scheduler including the "help" option. Passing 'helpargs' will yield a
                            help message with a list of overridable arguments for each scheduler and their
                            typical defaults. Defaults to the value of --scheduler.
      --sdxl-refiner-edit   Force the SDXL refiner to operate in edit mode instead of cooperative denoising mode
                            as it would normally do for inpainting and ControlNet usage. The main model will
                            preform the full amount of inference steps requested by --inference-steps. The
                            output of the main model will be passed to the refiner model and processed with an
                            image seed strength in img2img mode determined by (1.0 - high-noise-fraction)
      --sdxl-second-prompts PROMPT [PROMPT ...]
                            One or more secondary prompts to try using SDXL's secondary text encoder. By default
                            the model is passed the primary prompt for this value, this option allows you to
                            choose a different prompt. The negative prompt component can be specified with the
                            same syntax as --prompts
      --sdxl-aesthetic-scores FLOAT [FLOAT ...]
                            One or more Stable Diffusion XL (torch-sdxl) "aesthetic-score" micro-conditioning
                            parameters. Used to simulate an aesthetic score of the generated image by
                            influencing the positive text condition. Part of SDXL's micro-conditioning as
                            explained in section 2.2 of [https://huggingface.co/papers/2307.01952].
      --sdxl-crops-coords-top-left COORD [COORD ...]
                            One or more Stable Diffusion XL (torch-sdxl) "negative-crops-coords-top-left" micro-
                            conditioning parameters in the format "0,0". --sdxl-crops-coords-top-left can be
                            used to generate an image that appears to be "cropped" from the position --sdxl-
                            crops-coords-top-left downwards. Favorable, well-centered images are usually
                            achieved by setting --sdxl-crops-coords-top-left to "0,0". Part of SDXL's micro-
                            conditioning as explained in section 2.2 of
                            [https://huggingface.co/papers/2307.01952].
      --sdxl-original-size SIZE [SIZE ...], --sdxl-original-sizes SIZE [SIZE ...]
                            One or more Stable Diffusion XL (torch-sdxl) "original-size" micro-conditioning
                            parameters in the format (WIDTH)x(HEIGHT). If not the same as --sdxl-target-size the
                            image will appear to be down or up-sampled. --sdxl-original-size defaults to
                            --output-size or the size of any input images if not specified. Part of SDXL's
                            micro-conditioning as explained in section 2.2 of
                            [https://huggingface.co/papers/2307.01952]
      --sdxl-target-size SIZE [SIZE ...], --sdxl-target-sizes SIZE [SIZE ...]
                            One or more Stable Diffusion XL (torch-sdxl) "target-size" micro-conditioning
                            parameters in the format (WIDTH)x(HEIGHT). For most cases, --sdxl-target-size should
                            be set to the desired height and width of the generated image. If not specified it
                            will default to --output-size or the size of any input images. Part of SDXL's micro-
                            conditioning as explained in section 2.2 of
                            [https://huggingface.co/papers/2307.01952]
      --sdxl-negative-aesthetic-scores FLOAT [FLOAT ...]
                            One or more Stable Diffusion XL (torch-sdxl) "negative-aesthetic-score" micro-
                            conditioning parameters. Part of SDXL's micro-conditioning as explained in section
                            2.2 of [https://huggingface.co/papers/2307.01952]. Can be used to simulate an
                            aesthetic score of the generated image by influencing the negative text condition.
      --sdxl-negative-original-sizes SIZE [SIZE ...]
                            One or more Stable Diffusion XL (torch-sdxl) "negative-original-sizes" micro-
                            conditioning parameters. Negatively condition the generation process based on a
                            specific image resolution. Part of SDXL's micro-conditioning as explained in section
                            2.2 of [https://huggingface.co/papers/2307.01952]. For more information, refer to
                            this issue thread: https://github.com/huggingface/diffusers/issues/4208
      --sdxl-negative-target-sizes SIZE [SIZE ...]
                            One or more Stable Diffusion XL (torch-sdxl) "negative-original-sizes" micro-
                            conditioning parameters. To negatively condition the generation process based on a
                            target image resolution. It should be as same as the "--sdxl-target-size" for most
                            cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                            [https://huggingface.co/papers/2307.01952]. For more information, refer to this
                            issue thread: https://github.com/huggingface/diffusers/issues/4208.
      --sdxl-negative-crops-coords-top-left COORD [COORD ...]
                            One or more Stable Diffusion XL (torch-sdxl) "negative-crops-coords-top-left" micro-
                            conditioning parameters in the format "0,0". Negatively condition the generation
                            process based on a specific crop coordinates. Part of SDXL's micro-conditioning as
                            explained in section 2.2 of [https://huggingface.co/papers/2307.01952]. For more
                            information, refer to this issue thread:
                            https://github.com/huggingface/diffusers/issues/4208.
      --sdxl-refiner-prompts PROMPT [PROMPT ...]
                            One or more prompts to try with the SDXL refiner model, by default the refiner model
                            gets the primary prompt, this argument overrides that with a prompt of your
                            choosing. The negative prompt component can be specified with the same syntax as
                            --prompts
      --sdxl-refiner-clip-skips INTEGER [INTEGER ...]
                            One or more clip skip override values to try for the SDXL refiner, which normally
                            uses the clip skip value for the main model when it is defined by --clip-skips.
      --sdxl-refiner-second-prompts PROMPT [PROMPT ...]
                            One or more prompts to try with the SDXL refiner models secondary text encoder, by
                            default the refiner model gets the primary prompt passed to its second text encoder,
                            this argument overrides that with a prompt of your choosing. The negative prompt
                            component can be specified with the same syntax as --prompts
      --sdxl-refiner-aesthetic-scores FLOAT [FLOAT ...]
                            See: --sdxl-aesthetic-scores, applied to SDXL refiner pass.
      --sdxl-refiner-crops-coords-top-left COORD [COORD ...]
                            See: --sdxl-crops-coords-top-left, applied to SDXL refiner pass.
      --sdxl-refiner-original-sizes SIZE [SIZE ...]
                            See: --sdxl-refiner-original-sizes, applied to SDXL refiner pass.
      --sdxl-refiner-target-sizes SIZE [SIZE ...]
                            See: --sdxl-refiner-target-sizes, applied to SDXL refiner pass.
      --sdxl-refiner-negative-aesthetic-scores FLOAT [FLOAT ...]
                            See: --sdxl-negative-aesthetic-scores, applied to SDXL refiner pass.
      --sdxl-refiner-negative-original-sizes SIZE [SIZE ...]
                            See: --sdxl-negative-original-sizes, applied to SDXL refiner pass.
      --sdxl-refiner-negative-target-sizes SIZE [SIZE ...]
                            See: --sdxl-negative-target-sizes, applied to SDXL refiner pass.
      --sdxl-refiner-negative-crops-coords-top-left COORD [COORD ...]
                            See: --sdxl-negative-crops-coords-top-left, applied to SDXL refiner pass.
      -hnf FLOAT [FLOAT ...], --sdxl-high-noise-fractions FLOAT [FLOAT ...]
                            One or more high-noise-fraction values for Stable Diffusion XL (torch-sdxl), this
                            fraction of inference steps will be processed by the base model, while the rest will
                            be processed by the refiner model. Multiple values to this argument will result in
                            additional generation steps for each value. In certain situations when the mixture
                            of denoisers algorithm is not supported, such as when using --control-nets and
                            inpainting with SDXL, the inverse proportion of this value IE: (1.0 - high-noise-
                            fraction) becomes the --image-seed-strengths input to the SDXL refiner. (default:
                            [0.8])
      -ri INT [INT ...], --sdxl-refiner-inference-steps INT [INT ...]
                            One or more inference steps values for the SDXL refiner when in use. Override the
                            number of inference steps used by the SDXL refiner, which defaults to the value
                            taken from --inference-steps.
      -rg FLOAT [FLOAT ...], --sdxl-refiner-guidance-scales FLOAT [FLOAT ...]
                            One or more guidance scale values for the SDXL refiner when in use. Override the
                            guidance scale value used by the SDXL refiner, which defaults to the value taken
                            from --guidance-scales.
      -rgr FLOAT [FLOAT ...], --sdxl-refiner-guidance-rescales FLOAT [FLOAT ...]
                            One or more guidance rescale values for the SDXL refiner when in use. Override the
                            guidance rescale value used by the SDXL refiner, which defaults to the value taken
                            from --guidance-rescales.
      -sc, --safety-checker
                            Enable safety checker loading, this is off by default. When turned on images with
                            NSFW content detected may result in solid black output. Some pretrained models have
                            no safety checker model present, in that case this option has no effect.
      -d DEVICE, --device DEVICE
                            cuda / cpu. (default: cuda). Use: cuda:0, cuda:1, cuda:2, etc. to specify a specific
                            GPU. This argument is ignored when using flax, for flax use the environmental
                            variable CUDA_VISIBLE_DEVICES to specify which GPUs are visible to cuda, flax will
                            use every visible GPU.
      -t DTYPE, --dtype DTYPE
                            Model precision: auto, bfloat16, float16, or float32. (default: auto)
      -s SIZE, --output-size SIZE
                            Image output size, for txt2img generation, this is the exact output size. The
                            dimensions specified for this value must be aligned by 8 or you will receive an
                            error message. If an --image-seeds URI is used its Seed, Mask, and/or Control
                            component image sources will be resized to this dimension with aspect ratio
                            maintained before being used for generation by default. Unless --no-aspect is
                            specified, width will be fixed and a new height (aligned by 8) will be calculated
                            for the input images. In most cases resizing the image inputs will result in an
                            image output of an equal size to the inputs, except in the case of upscalers and
                            Deep Floyd --model-type values (torch-if*). If only one integer value is provided,
                            that is the value for both dimensions. X/Y dimension values should be separated by
                            "x". This value defaults to 512x512 for Stable Diffusion when no --image-seeds are
                            specified (IE txt2img mode), 1024x1024 for Stable Cascade and Stable Diffusion 3/XL
                            (SD3 or SDXL) model types, and 64x64 for --model-type torch-if (Deep Floyd stage 1).
                            Deep Floyd stage 1 images passed to superscaler models (--model-type torch-ifs*)
                            that are specified with the 'floyd' keyword argument in an --image-seeds definition
                            are never resized or processed in any way.
      -na, --no-aspect      This option disables aspect correct resizing of images provided to --image-seeds
                            globally. Seed, Mask, and Control guidance images will be resized to the closest
                            dimension specified by --output-size that is aligned by 8 pixels with no
                            consideration of the source aspect ratio. This can be overriden at the --image-seeds
                            level with the image seed keyword argument 'aspect=true/false'.
      -o PATH, --output-path PATH
                            Output path for generated images and files. This directory will be created if it
                            does not exist. (default: ./output)
      -op PREFIX, --output-prefix PREFIX
                            Name prefix for generated images and files. This prefix will be added to the
                            beginning of every generated file, followed by an underscore.
      -ox, --output-overwrite
                            Enable overwrites of files in the output directory that already exists. The default
                            behavior is not to do this, and instead append a filename suffix:
                            "_duplicate_(number)" when it is detected that the generated file name already
                            exists.
      -oc, --output-configs
                            Write a configuration text file for every output image or animation. The text file
                            can be used reproduce that particular output image or animation by piping it to
                            dgenerate STDIN or by using the --file option, for example "dgenerate < config.dgen"
                            or "dgenerate --file config.dgen". These files will be written to --output-path and
                            are affected by --output-prefix and --output-overwrite as well. The files will be
                            named after their corresponding image or animation file. Configuration files
                            produced for animation frame images will utilize --frame-start and --frame-end to
                            specify the frame number.
      -om, --output-metadata
                            Write the information produced by --output-configs to the PNG metadata of each
                            image. Metadata will not be written to animated files (yet). The data is written to
                            a PNG metadata property named DgenerateConfig and can be read using ImageMagick like
                            so: "magick identify -format "%[Property:DgenerateConfig] generated_file.png".
      -pw PROMPT_WEIGHTER_URI, --prompt-weighter PROMPT_WEIGHTER_URI
                            Specify a prompt weighter implementation by URI, for example: --prompt-weighter
                            compel, or --prompt-weighter sd-embed. By default, no prompt weighting syntax is
                            enabled, meaning that you cannot adjust token weights as you may be able to do in
                            software such as ComfyUI, Automatic1111, CivitAI etc. And in some cases the length
                            of your prompt is limited. Prompt weighters support these special token weighting
                            syntaxes and long prompts, currently there are two implementations "compel" and "sd-
                            embed". See: --prompt-weighter-help for a list of implementation names. You may also
                            use --prompt-weighter-help "name" to see comprehensive documentation for a specific
                            prompt weighter implementation.
      --prompt-weighter-help [PROMPT_WEIGHTER_NAMES ...]
                            Use this option alone (or with --plugin-modules) and no model specification in order
                            to list available prompt weighter names. Specifying one or more prompt weighter
                            names after this option will cause usage documentation for the specified prompt
                            weighters to be printed. When used with --plugin-modules, prompt weighters
                            implemented by the specified plugins will also be listed.
      -p PROMPT [PROMPT ...], --prompts PROMPT [PROMPT ...]
                            One or more prompts to try, an image group is generated for each prompt, prompt data
                            is split by ; (semi-colon). The first value is the positive text influence, things
                            you want to see. The Second value is negative influence IE. things you don't want to
                            see. Example: --prompts "shrek flying a tesla over detroit; clouds, rain, missiles".
                            (default: [(empty string)])
      --sd3-max-sequence-length INTEGER
                            The maximum amount of prompt tokens that the T5EncoderModel (third text encoder) of
                            Stable Diffusion 3 can handle. This should be an integer value between 1 and 512
                            inclusive. The higher the value the more resources and time are required for
                            processing. (default: 256)
      --sd3-second-prompts PROMPT [PROMPT ...]
                            One or more secondary prompts to try using the torch-sd3 (Stable Diffusion 3)
                            secondary text encoder. By default the model is passed the primary prompt for this
                            value, this option allows you to choose a different prompt. The negative prompt
                            component can be specified with the same syntax as --prompts
      --sd3-third-prompts PROMPT [PROMPT ...]
                            One or more tertiary prompts to try using the torch-sd3 (Stable Diffusion 3)
                            tertiary (T5) text encoder. By default the model is passed the primary prompt for
                            this value, this option allows you to choose a different prompt. The negative prompt
                            component can be specified with the same syntax as --prompts
      -cs INTEGER [INTEGER ...], --clip-skips INTEGER [INTEGER ...]
                            One or more clip skip values to try. Clip skip is the number of layers to be skipped
                            from CLIP while computing the prompt embeddings, it must be a value greater than or
                            equal to zero. A value of 1 means that the output of the pre-final layer will be
                            used for computing the prompt embeddings. This is only supported for --model-type
                            values "torch" and "torch-sdxl", including with --control-nets.
      -se SEED [SEED ...], --seeds SEED [SEED ...]
                            One or more seeds to try, define fixed seeds to achieve deterministic output. This
                            argument may not be used when --gse/--gen-seeds is used. (default: [randint(0,
                            99999999999999)])
      -sei, --seeds-to-images
                            When this option is enabled, each provided --seeds value or value generated by
                            --gen-seeds is used for the corresponding image input given by --image-seeds. If the
                            amount of --seeds given is not identical to that of the amount of --image-seeds
                            given, the seed is determined as: seed = seeds[image_seed_index % len(seeds)], IE:
                            it wraps around.
      -gse COUNT, --gen-seeds COUNT
                            Auto generate N random seeds to try. This argument may not be used when -se/--seeds
                            is used.
      -af FORMAT, --animation-format FORMAT
                            Output format when generating an animation from an input video / gif / webp etc.
                            Value must be one of: mp4, png, apng, gif, or webp. You may also specify "frames" to
                            indicate that only frames should be output and no coalesced animation file should be
                            rendered. (default: mp4)
      -if FORMAT, --image-format FORMAT
                            Output format when writing static images. Any selection other than "png" is not
                            compatible with --output-metadata. Value must be one of: png, apng, blp, bmp, dib,
                            bufr, pcx, dds, ps, eps, gif, grib, h5, hdf, jp2, j2k, jpc, jpf, jpx, j2c, icns,
                            ico, im, jfif, jpe, jpg, jpeg, tif, tiff, mpo, msp, palm, pdf, pbm, pgm, ppm, pnm,
                            pfm, bw, rgb, rgba, sgi, tga, icb, vda, vst, webp, wmf, emf, or xbm. (default: png)
      -nf, --no-frames      Do not write frame images individually when rendering an animation, only write the
                            animation file. This option is incompatible with --animation-format frames.
      -fs FRAME_NUMBER, --frame-start FRAME_NUMBER
                            Starting frame slice point for animated files (zero-indexed), the specified frame
                            will be included. (default: 0)
      -fe FRAME_NUMBER, --frame-end FRAME_NUMBER
                            Ending frame slice point for animated files (zero-indexed), the specified frame will
                            be included.
      -is SEED [SEED ...], --image-seeds SEED [SEED ...]
                            One or more image seed URIs to process, these may consist of URLs or file paths.
                            Videos / GIFs / WEBP files will result in frames being rendered as well as an
                            animated output file being generated if more than one frame is available in the
                            input file. Inpainting for static images can be achieved by specifying a black and
                            white mask image in each image seed string using a semicolon as the separating
                            character, like so: "my-seed-image.png;my-image-mask.png", white areas of the mask
                            indicate where generated content is to be placed in your seed image. Output
                            dimensions specific to the image seed can be specified by placing the dimension at
                            the end of the string following a semicolon like so: "my-seed-image.png;512x512" or
                            "my-seed-image.png;my-image-mask.png;512x512". When using --control-nets, a singular
                            image specification is interpreted as the control guidance image, and you can
                            specify multiple control image sources by separating them with commas in the case
                            where multiple ControlNets are specified, IE: (--image-seeds "control-image1.png,
                            control-image2.png") OR (--image-seeds "seed.png;control=control-image1.png,
                            control-image2.png"). Using --control-nets with img2img or inpainting can be
                            accomplished with the syntax: "my-seed-image.png;mask=my-image-mask.png;control=my-
                            control-image.png;resize=512x512". The "mask" and "resize" arguments are optional
                            when using --control-nets. Videos, GIFs, and WEBP are also supported as inputs when
                            using --control-nets, even for the "control" argument. --image-seeds is capable of
                            reading from multiple animated files at once or any combination of animated files
                            and images, the animated file with the least amount of frames dictates how many
                            frames are generated and static images are duplicated over the total amount of
                            frames. The keyword argument "aspect" can be used to determine resizing behavior
                            when the global argument --output-size or the local keyword argument "resize" is
                            specified, it is a boolean argument indicating whether aspect ratio of the input
                            image should be respected or ignored. The keyword argument "floyd" can be used to
                            specify images from a previous deep floyd stage when using --model-type torch-ifs*.
                            When keyword arguments are present, all applicable images such as "mask", "control",
                            etc. must also be defined with keyword arguments instead of with the short syntax.
      -sip PROCESSOR_URI [PROCESSOR_URI ...], --seed-image-processors PROCESSOR_URI [PROCESSOR_URI ...]
                            Specify one or more image processor actions to preform on the primary image
                            specified by --image-seeds. For example: --seed-image-processors "flip" "mirror"
                            "grayscale". To obtain more information about what image processors are available
                            and how to use them, see: --image-processor-help.
      -mip PROCESSOR_URI [PROCESSOR_URI ...], --mask-image-processors PROCESSOR_URI [PROCESSOR_URI ...]
                            Specify one or more image processor actions to preform on the inpaint mask image
                            specified by --image-seeds. For example: --mask-image-processors "invert". To obtain
                            more information about what image processors are available and how to use them, see:
                            --image-processor-help.
      -cip PROCESSOR_URI [PROCESSOR_URI ...], --control-image-processors PROCESSOR_URI [PROCESSOR_URI ...]
                            Specify one or more image processor actions to preform on the control image
                            specified by --image-seeds, this option is meant to be used with --control-nets.
                            Example: --control-image-processors "canny;lower=50;upper=100". The delimiter "+"
                            can be used to specify a different processor group for each image when using
                            multiple control images with --control-nets. For example if you have --image-seeds
                            "img1.png, img2.png" or --image-seeds "...;control=img1.png, img2.png" specified and
                            multiple ControlNet models specified with --control-nets, you can specify processors
                            for those control images with the syntax: (--control-image-processors "processes-
                            img1" + "processes-img2"), this syntax also supports chaining of processors, for
                            example: (--control-image-processors "first-process-img1" "second-process-img1" +
                            "process-img2"). The amount of specified processors must not exceed the amount of
                            specified control images, or you will receive a syntax error message. Images which
                            do not have a processor defined for them will not be processed, and the plus
                            character can be used to indicate an image is not to be processed and instead
                            skipped over when that image is a leading element, for example (--control-image-
                            processors + "process-second") would indicate that the first control guidance image
                            is not to be processed, only the second. To obtain more information about what image
                            processors are available and how to use them, see: --image-processor-help.
      --image-processor-help [PROCESSOR_NAME ...]
                            Use this option alone (or with --plugin-modules) and no model specification in order
                            to list available image processor names. Specifying one or more image processor
                            names after this option will cause usage documentation for the specified image
                            processors to be printed. When used with --plugin-modules, image processors
                            implemented by the specified plugins will also be listed.
      -pp PROCESSOR_URI [PROCESSOR_URI ...], --post-processors PROCESSOR_URI [PROCESSOR_URI ...]
                            Specify one or more image processor actions to preform on generated output before it
                            is saved. For example: --post-processors "upcaler;model=4x_ESRGAN.pth". To obtain
                            more information about what processors are available and how to use them, see:
                            --image-processor-help.
      -iss FLOAT [FLOAT ...], --image-seed-strengths FLOAT [FLOAT ...]
                            One or more image strength values to try when using --image-seeds for img2img or
                            inpaint mode. Closer to 0 means high usage of the seed image (less noise
                            convolution), 1 effectively means no usage (high noise convolution). Low values will
                            produce something closer or more relevant to the input image, high values will give
                            the AI more creative freedom. This value must be greater than 0 and less than or
                            equal to 1. (default: [0.8])
      -uns INTEGER [INTEGER ...], --upscaler-noise-levels INTEGER [INTEGER ...]
                            One or more upscaler noise level values to try when using the super resolution
                            upscaler --model-type torch-upscaler-x4 or torch-ifs. Specifying this option for
                            --model-type torch-upscaler-x2 will produce an error message. The higher this value
                            the more noise is added to the image before upscaling (similar to --image-seed-
                            strengths). (default: [20 for x4, 250 for torch-ifs/torch-ifs-img2img, 0 for torch-
                            ifs inpainting mode])
      -gs FLOAT [FLOAT ...], --guidance-scales FLOAT [FLOAT ...]
                            One or more guidance scale values to try. Guidance scale effects how much your text
                            prompt is considered. Low values draw more data from images unrelated to text
                            prompt. (default: [5])
      -igs FLOAT [FLOAT ...], --image-guidance-scales FLOAT [FLOAT ...]
                            One or more image guidance scale values to try. This can push the generated image
                            towards the initial image when using --model-type *-pix2pix models, it is
                            unsupported for other model types. Use in conjunction with --image-seeds, inpainting
                            (masks) and --control-nets are not supported. Image guidance scale is enabled by
                            setting image-guidance-scale > 1. Higher image guidance scale encourages generated
                            images that are closely linked to the source image, usually at the expense of lower
                            image quality. Requires a value of at least 1. (default: [1.5])
      -gr FLOAT [FLOAT ...], --guidance-rescales FLOAT [FLOAT ...]
                            One or more guidance rescale factors to try. Proposed by [Common Diffusion Noise
                            Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)
                            "guidance_scale" is defined as "φ" in equation 16. of [Common Diffusion Noise
                            Schedules and Sample Steps are Flawed] (https://arxiv.org/pdf/2305.08891.pdf).
                            Guidance rescale factor should fix overexposure when using zero terminal SNR. This
                            is supported for basic text to image generation when using --model-type "torch" but
                            not inpainting, img2img, or --control-nets. When using --model-type "torch-sdxl" it
                            is supported for basic generation, inpainting, and img2img, unless --control-nets is
                            specified in which case only inpainting is supported. It is supported for --model-
                            type "torch-sdxl-pix2pix" but not --model-type "torch-pix2pix". (default: [0.0])
      -ifs INTEGER [INTEGER ...], --inference-steps INTEGER [INTEGER ...]
                            One or more inference steps values to try. The amount of inference (de-noising)
                            steps effects image clarity to a degree, higher values bring the image closer to
                            what the AI is targeting for the content of the image. Values between 30-40 produce
                            good results, higher values may improve image quality and or change image content.
                            (default: [30])
      -mc EXPR [EXPR ...], --cache-memory-constraints EXPR [EXPR ...]
                            Cache constraint expressions describing when to clear all model caches automatically
                            (DiffusionPipeline, UNet, VAE, ControlNet, and Text Encoder) considering current
                            memory usage. If any of these constraint expressions are met all models cached in
                            memory will be cleared. Example, and default value: "used_percent > 70" For Syntax
                            See: [https://dgenerate.readthedocs.io/en/v3.10.0/dgenerate_submodules.html#dgenerate
                            .pipelinewrapper.CACHE_MEMORY_CONSTRAINTS]
      -pmc EXPR [EXPR ...], --pipeline-cache-memory-constraints EXPR [EXPR ...]
                            Cache constraint expressions describing when to automatically clear the in memory
                            DiffusionPipeline cache considering current memory usage, and estimated memory usage
                            of new models that are about to enter memory. If any of these constraint expressions
                            are met all DiffusionPipeline objects cached in memory will be cleared. Example, and
                            default value: "pipeline_size > (available * 0.75)" For Syntax See: [https://dgenera
                            te.readthedocs.io/en/v3.10.0/dgenerate_submodules.html#dgenerate.pipelinewrapper.PIPE
                            LINE_CACHE_MEMORY_CONSTRAINTS]
      -umc EXPR [EXPR ...], --unet-cache-memory-constraints EXPR [EXPR ...]
                            Cache constraint expressions describing when to automatically clear the in memory
                            UNet cache considering current memory usage, and estimated memory usage of new UNet
                            models that are about to enter memory. If any of these constraint expressions are
                            met all UNet models cached in memory will be cleared. Example, and default value:
                            "unet_size > (available * 0.75)" For Syntax See: [https://dgenerate.readthedocs.io/e
                            n/v3.10.0/dgenerate_submodules.html#dgenerate.pipelinewrapper.UNET_CACHE_MEMORY_CONST
                            RAINTS]
      -vmc EXPR [EXPR ...], --vae-cache-memory-constraints EXPR [EXPR ...]
                            Cache constraint expressions describing when to automatically clear the in memory
                            VAE cache considering current memory usage, and estimated memory usage of new VAE
                            models that are about to enter memory. If any of these constraint expressions are
                            met all VAE models cached in memory will be cleared. Example, and default value:
                            "vae_size > (available * 0.75)" For Syntax See: [https://dgenerate.readthedocs.io/en
                            /v3.10.0/dgenerate_submodules.html#dgenerate.pipelinewrapper.VAE_CACHE_MEMORY_CONSTRA
                            INTS]
      -cmc EXPR [EXPR ...], --control-net-cache-memory-constraints EXPR [EXPR ...]
                            Cache constraint expressions describing when to automatically clear the in memory
                            ControlNet cache considering current memory usage, and estimated memory usage of new
                            ControlNet models that are about to enter memory. If any of these constraint
                            expressions are met all ControlNet models cached in memory will be cleared. Example,
                            and default value: "control_net_size > (available * 0.75)" For Syntax See: [https://
                            dgenerate.readthedocs.io/en/v3.10.0/dgenerate_submodules.html#dgenerate.pipelinewrapp
                            er.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS]
      -tmc EXPR [EXPR ...], --text-encoder-cache-memory-constraints EXPR [EXPR ...]
                            Cache constraint expressions describing when to automatically clear the in memory
                            Text Encoder cache considering current memory usage, and estimated memory usage of
                            new Text Encoder models that are about to enter memory. If any of these constraint
                            expressions are met all Text Encoder models cached in memory will be cleared.
                            Example, and default value: "text_encoder_size > (available * 0.75)" For Syntax See:
                            [https://dgenerate.readthedocs.io/en/v3.10.0/dgenerate_submodules.html#dgenerate.pipe
                            linewrapper.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS]


Windows Install
===============

You can install using the Windows installer provided with each release on the
`Releases Page <https://github.com/Teriks/dgenerate/releases>`_, or you can manually
install with pipx, (or pip if you want) as described below.


Manual Install
--------------


Install Visual Studios (Community or other), make sure "Desktop development with C++" is selected, unselect anything you do not need.

https://visualstudio.microsoft.com/downloads/


Install rust compiler using rustup-init.exe (x64), use the default install options.

https://www.rust-lang.org/tools/install

Install Python:

https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe

Make sure you select the option "Add to PATH" in the python installer,
otherwise invoke python directly using it's full path while installing the tool.

Install GIT for Windows:

https://gitforwindows.org/


Install dgenerate
-----------------

Using Windows CMD

Install pipx:

.. code-block:: bash

    pip install pipx
    pipx ensurepath

    # Log out and log back in so PATH takes effect

Install dgenerate:

.. code-block:: bash

    pipx install dgenerate ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu121/"

    # with NCNN upscaler support

    pipx install dgenerate[ncnn] ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu121/"

    # If you want a specific version

    pipx install dgenerate==3.10.0 ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu121/"

    # with NCNN upscaler support and a specific version

    pipx install dgenerate[ncnn]==3.10.0 ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu121/"

    # You can install without pipx into your own environment like so

    pip install dgenerate==3.10.0 --extra-index-url https://download.pytorch.org/whl/cu121/

    # Or with NCNN

    pip install dgenerate[ncnn]==3.10.0 --extra-index-url https://download.pytorch.org/whl/cu121/


It is recommended to install dgenerate with pipx if you are just intending
to use it as a command line program, if you want to develop you can install it from
a cloned repository like this:

.. code-block:: bash

    # in the top of the repo make
    # an environment and activate it

    python -m venv venv
    venv\Scripts\activate

    # Install with pip into the environment

    pip install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu121/

    # Install with pip into the environment, include NCNN

    pip install --editable .[dev, ncnn] --extra-index-url https://download.pytorch.org/whl/cu121/


Run ``dgenerate`` to generate images:

.. code-block:: bash

    # Images are output to the "output" folder
    # in the current working directory by default

    dgenerate --help

    dgenerate stabilityai/stable-diffusion-2-1 ^
    --prompts "an astronaut riding a horse" ^
    --output-path output ^
    --inference-steps 40 ^
    --guidance-scales 10

Linux or WSL Install
====================

First update your system and install build-essential

.. code-block:: bash

    sudo apt update && sudo apt upgrade
    sudo apt install build-essential

Install CUDA Toolkit 12.*: https://developer.nvidia.com/cuda-downloads

I recommend using the runfile option:

.. code-block:: bash

    # CUDA Toolkit 12.2.1 For Ubuntu / Debian / WSL

    wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda_12.2.1_535.86.10_linux.run
    sudo sh cuda_12.2.1_535.86.10_linux.run

Do not attempt to install a driver from the prompts if using WSL.


.. code-block:: bash

    # On linux, if you intend to use flax, you may or may not need to create a symlink for libnvrtc
    # flax will look for libnvrtc.so, and may not be able to find it.

    ln -s /usr/local/cuda/lib64/libnvrtc.so.12 /usr/local/cuda/lib64/libnvrtc.so


Add libraries to linker path:

.. code-block:: bash

    # Add to ~/.bashrc

    # For Linux add the following
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    # For WSL add the following
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    # Add this in both cases as well
    export PATH=/usr/local/cuda/bin:$PATH


When done editing ``~/.bashrc`` do:

.. code-block:: bash

    source ~/.bashrc


Install Python 3.10+ (Debian / Ubuntu) and pipx
-----------------------------------------------

.. code-block:: bash

    sudo apt install python3.10 python3-pip pipx python3.10-venv python3-wheel
    pipx ensurepath

    source ~/.bashrc


Install dgenerate
-----------------

.. code-block:: bash

    # install with just support for torch

    pipx install dgenerate \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu121/"

    # With NCNN upscaler support (pip extras can be combined)
    # for instance [flax, ncnn] can be used, make sure to include
    # the -f argument mentioned here in --pip-args for flax
    # mentioned below

    # be aware that the ncnn python package depends on the non headless
    # version of python-opencv and it may cause issues
    # on headless systems without a window manager

    pipx install dgenerate[ncnn] \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu121/"

    # With flax/jax support

    pipx install dgenerate[flax] \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu121/ \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

    # If you want a specific version

    pipx install dgenerate==3.10.0 \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu121/"

    # Specific version with flax/jax support

    pipx install dgenerate[flax]==3.10.0 \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu121/ \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

    # You can install without pipx into your own environment like so

    pip3 install dgenerate==3.10.0 --extra-index-url https://download.pytorch.org/whl/cu121/

    # Or with NCNN

    pip3 install dgenerate[ncnn]==3.10.0 --extra-index-url https://download.pytorch.org/whl/cu121/

    # Or with flax

    pip3 install dgenerate[flax]==3.10.0 --extra-index-url https://download.pytorch.org/whl/cu121/ \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # With both

    pip3 install dgenerate[flax, ncnn]==3.10.0 --extra-index-url https://download.pytorch.org/whl/cu121/ \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


It is recommended to install dgenerate with pipx if you are just intending
to use it as a command line program, if you want to develop you can install it from
a cloned repository like this:

.. code-block:: bash

    # in the top of the repo make
    # an environment and activate it

    python3 -m venv venv
    source venv/bin/activate

    # Install with pip into the environment

    pip3 install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu121/

    # With flax if you want

    pip3 install --editable .[dev,flax] --extra-index-url https://download.pytorch.org/whl/cu121/ \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


Run ``dgenerate`` to generate images:

.. code-block:: bash

    # Images are output to the "output" folder
    # in the current working directory by default

    dgenerate --help

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --output-path output \
    --inference-steps 40 \
    --guidance-scales 10


Google Colab Install
====================

The following cell entries will get you started in a Google Collab environment.

Make sure you select a GPU runtime for your notebook, such as the T4 runtime.


1.) Install venv.

.. code-block:: bash

    !apt install python3-venv

2.) Create a virtual environment.

.. code-block:: bash

    !python3 -m venv venv

3.) Install dgenerate, you must activate the virtual environment in the same cell.

.. code-block:: bash

    !source /content/venv/bin/activate; pip install dgenerate==3.10.0 --extra-index-url https://download.pytorch.org/whl/cu121

4.) Finally you can run dgenerate, you must prefix all calls to dgenerate with an activation of the virtual environment, as
the virtual environment is not preserved between cells.  For brevity, and as an example, just print the help text here.

.. code-block:: bash

    !source /content/venv/bin/activate; dgenerate --help


Basic Usage
===========

The example below attempts to generate an astronaut riding a horse using 5 different
random seeds, 3 different inference steps values, and 3 different guidance scale values.

It utilizes the ``stabilityai/stable-diffusion-2-1`` model repo on `Hugging Face <https://huggingface.co/stabilityai/stable-diffusion-2-1>`_.

45 uniquely named images will be generated ``(5 x 3 x 3)``

Also Adjust output size to ``512x512`` and output generated images to the ``astronaut`` folder in the current working directory.

When ``--output-path`` is not specified, the default output location is the ``output`` folder
in the current working directory, if the path that is specified does not exist then it will be created.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 30 40 50 \
    --guidance-scales 5 7 10 \
    --output-size 512x512


Loading models from huggingface blob links is also supported:

.. code-block:: bash

    dgenerate https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors \
    --prompts "an astronaut riding a horse" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 30 40 50 \
    --guidance-scales 5 7 10 \
    --output-size 512x512


SDXL is supported and can be used to generate highly realistic images.

Prompt only generation, img2img, and inpainting is supported for SDXL.

Refiner models can be specified, ``fp16`` model variant and a datatype of ``float16`` is
recommended to prevent out of memory conditions on the average GPU :)

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --sdxl-high-noise-fractions 0.6 0.7 0.8 \
    --gen-seeds 5 \
    --inference-steps 50 \
    --guidance-scales 12 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --prompts "real photo of an astronaut riding a horse on the moon" \
    --variant fp16 --dtype float16 \
    --output-size 1024


Negative Prompt
===============

In order to specify a negative prompt, each prompt argument is split
into two parts separated by ``;``

The prompt text occurring after ``;`` is the negative influence prompt.

To attempt to avoid rendering of a saddle on the horse being ridden, you
could for example add the negative prompt ``saddle`` or ``wearing a saddle``
or ``horse wearing a saddle`` etc.


.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse; horse wearing a saddle" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


Multiple Prompts
================

Multiple prompts can be specified one after another in quotes in order
to generate images using multiple prompt variations.

The following command generates 10 uniquely named images using two
prompts and five random seeds ``(2x5)``

5 of them will be from the first prompt and 5 of them from the second prompt.

All using 50 inference steps, and 10 for guidance scale value.


.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" "an astronaut riding a donkey" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


Image Seed
==========

The ``--image-seeds`` argument can be used to specify one or more image input resource groups
for use in rendering, and allows for the specification of img2img source images, inpaint masks,
control net guidance images, deep floyd stage images, image group resizing, and frame slicing values
for animations. It possesses it's own URI syntax for defining different image inputs used for image generation,
the example described below is the simplest case for one image input (img2img).

This example uses a photo of Buzz Aldrin on the moon to generate a photo of an astronaut standing on mars
using img2img, this uses an image seed downloaded from wikipedia.

Disk file paths may also be used for image seeds and generally that is the standard use case,
multiple image seed definitions may be provided and images will be generated from each image
seed individually.

.. code-block:: bash

    # Generate this image using 5 different seeds, 3 different inference-step values, 3 different
    # guidance-scale values as above.

    # In addition this image will be generated using 3 different image seed strengths.

    # Adjust output size to 512x512 and output generated images to 'astronaut' folder, the image seed
    # will be resized to that dimension with aspect ratio respected by default, the width is fixed and
    # the height will be calculated, this behavior can be changed globally with the --no-aspect option
    # if desired or locally by specifying "img2img-seed.png;aspect=false" as your image seed

    # If you do not adjust the output size of the generated image, the size of the input image seed will be used.

    # 135 uniquely named images will be generated (5x3x3x3)

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut walking on mars" \
    --image-seeds https://upload.wikimedia.org/wikipedia/commons/9/98/Aldrin_Apollo_11_original.jpg \
    --image-seed-strengths 0.2 0.5 0.8 \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 30 40 50 \
    --guidance-scales 5 7 10 \
    --output-size 512x512


``--image-seeds`` serves as the entire mechanism for determining if img2img or inpainting is going to occur via
it's URI syntax described further in the section `Inpainting`_.

In addition to this it can be used to provide control guidance images in the case of txt2img, img2img, or inpainting
via the use of a URI syntax involving keyword arguments.

The syntax ``--image-seeds "my-image-seed.png;control=my-control-image.png"`` can be used with ``--control-nets`` to specify
img2img mode with a ControlNet for example, see: `Specifying Control Nets`_ for more information.


Inpainting
==========

Inpainting on an image can be preformed by providing a mask image with your image seed. This mask should be a black and white image
of identical size to your image seed.  White areas of the mask image will be used to tell the AI what areas of the seed image should be filled
in with generated content.

For using inpainting on animated image seeds, jump to: `Inpainting Animations`_

Some possible definitions for inpainting are:

    * ``--image-seeds "my-image-seed.png;my-mask-image.png"``
    * ``--image-seeds "my-image-seed.png;mask=my-mask-image.png"``

The format is your image seed and mask image separated by ``;``, optionally ``mask`` can be named argument.
The alternate syntax is for disambiguation when preforming img2img or inpainting operations while `Specifying Control Nets`_
or other operations where keyword arguments might be necessary for disambiguation such as per image seed `Animation Slicing`_,
and the specification of the image from a previous Deep Floyd stage using the ``floyd`` argument.

Mask images can be downloaded from URL's just like any other resource mentioned in an ``--image-seeds`` definition,
however for this example files on disk are used for brevity.

You can download them here:

 * `my-image-seed.png <https://raw.githubusercontent.com/Teriks/dgenerate/v3.10.0/examples/media/dog-on-bench.png>`_
 * `my-mask-image.png <https://raw.githubusercontent.com/Teriks/dgenerate/v3.10.0/examples/media/dog-on-bench-mask.png>`_

The command below generates a cat sitting on a bench with the images from the links above, the mask image masks out
areas over the dog in the original image, causing the dog to be replaced with an AI generated cat.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --image-seeds "my-image-seed.png;my-mask-image.png" \
    --prompts "Face of a yellow cat, high resolution, sitting on a park bench" \
    --image-seed-strengths 0.8 \
    --guidance-scales 10 \
    --inference-steps 100


Per Image Seed Resizing
=======================

If you want to specify multiple image seeds that will have different output sizes irrespective
of their input size or a globally defined output size defined with ``--output-size``,
You can specify their output size individually at the end of each provided image seed.

This will work when using a mask image for inpainting as well, including when using animated inputs.

This also works when `Specifying Control Nets`_ and guidance images for control nets.

Here are some possible definitions:

    * ``--image-seeds "my-image-seed.png;512x512"`` (img2img)
    * ``--image-seeds "my-image-seed.png;my-mask-image.png;512x512"`` (inpainting)
    * ``--image-seeds "my-image-seed.png;resize=512x512"`` (img2img)
    * ``--image-seeds "my-image-seed.png;mask=my-mask-image.png;resize=512x512"`` (inpainting)

The alternate syntax with named arguments is for disambiguation when `Specifying Control Nets`_, or
preforming per image seed `Animation Slicing`_, or specifying the previous Deep Floyd stage output
with the ``floyd`` keyword argument.

When one dimension is specified, that dimension is the width, and the height.

The height of an image is calculated to be aspect correct by default for all resizing
methods unless ``--no-aspect`` has been given as an argument on the command line or the
``aspect`` keyword argument is used in the ``--image-seeds`` definition.

The the aspect correct resize behavior can be controlled on a per image seed definition basis
using the ``aspect`` keyword argument.  Any value given to this argument overrides the presence
or absense of the ``--no-aspect`` command line argument.

the ``aspect`` keyword argument can only be used when all other components of the image seed
definition are defined using keyword arguments. ``aspect=false`` disables aspect correct resizing,
and ``aspect=true`` enables it.

Some possible definitions:

    * ``--image-seeds "my-image-seed.png;resize=512x512;aspect=false"`` (img2img)
    * ``--image-seeds "my-image-seed.png;mask=my-mask-image.png;resize=512x512;aspect=false"`` (inpainting)


The following example preforms img2img generation, followed by inpainting generation using 2 image seed definitions.
The involved images are resized using the basic syntax with no keyword arguments present in the image seeds.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --image-seeds "my-image-seed.png;1024" "my-image-seed.png;my-mask-image.png;512x512" \
    --prompts "Face of a yellow cat, high resolution, sitting on a park bench" \
    --image-seed-strengths 0.8 \
    --guidance-scales 10 \
    --inference-steps 100


Animated Output
===============

``dgenerate`` supports many video formats through the use of PyAV (ffmpeg), as well as GIF & WebP.

See ``--help`` for information about all formats supported for the ``--animation-format`` option.

When an animated image seed is given, animated output will be produced in the format of your choosing.

In addition, every frame will be written to the output folder as a uniquely named image.

By specifying ``--animation-format frames`` you can tell dgenerate that you just need
the frame images and not to produce any coalesced animation file for you. You may also
specify ``--no-frames`` to indicate that you only want an animation file to be produced
and no intermediate frames, though using this option with ``--animation-format frames``
is considered an error.

If the animation is not 1:1 aspect ratio, the width will be fixed to the width of the
requested output size, and the height calculated to match the aspect ratio of the animation.
Unless ``--no-aspect`` or the ``--image-seeds`` keyword argument ``aspect=false`` are specified,
in which case the video will be resized to the requested dimension exactly.

If you do not set an output size, the size of the input animation will be used.

.. code-block:: bash

    # Use a GIF of a man riding a horse to create an animation of an astronaut riding a horse.

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --image-seeds https://upload.wikimedia.org/wikipedia/commons/7/7b/Muybridge_race_horse_~_big_transp.gif \
    --image-seed-strengths 0.5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512 \
    --animation-format mp4


The above syntax is the same syntax used for generating an animation with a control
image when ``--control-nets`` is used.

Animations can also be generated using an alternate syntax for ``--image-seeds``
that allows the specification of a control image source when it is desired to use
``--control-nets`` with img2img or inpainting.

For more information about this see: `Specifying Control Nets`_

As well as the information about ``--image-seeds`` from dgenerates ``--help``
output.


Animation Slicing
=================

Animated inputs can be sliced by a frame range either globally using
``--frame-start`` and ``--frame-end`` or locally using the named argument
syntax for ``--image-seeds``, for example:

    * ``--image-seeds "animated.gif;frame-start=3;frame-end=10"``.

When using animation slicing at the ``--image-seed`` level, all image input definitions
other than the main image must be specified using keyword arguments.

For example here are some possible definitions:

    * ``--image-seeds "seed.gif;frame-start=3;frame-end=10"``
    * ``--image-seeds "seed.gif;mask=mask.gif;frame-start=3;frame-end=10``
    * ``--image-seeds "seed.gif;control=control-guidance.gif;frame-start=3;frame-end=10``
    * ``--image-seeds "seed.gif;mask=mask.gif;control=control-guidance.gif;frame-start=3;frame-end=10``
    * ``--image-seeds "seed.gif;floyd=stage1.gif;frame-start=3;frame-end=10"``
    * ``--image-seeds "seed.gif;mask=mask.gif;floyd=stage1.gif;frame-start=3;frame-end=10"``

Specifying a frame slice locally in an image seed overrides the global frame
slice setting defined by ``--frame-start`` or ``--frame-end``, and is specific only
to that image seed, other image seed definitions will not be affected.

Perhaps you only want to run diffusion on the first frame of an animated input in
order to save time in finding good parameters for generating every frame. You could
slice to only the first frame using ``--frame-start 0 --frame-end 0``, which will be much
faster than rendering the entire video/gif outright.

The slice range zero indexed and also inclusive, inclusive means that the starting and ending frames
specified by ``--frame-start`` and ``--frame-end`` will be included in the slice.  Both slice points
do not have to be specified at the same time. You can exclude the tail end of a video with
just ``--frame-end`` alone, or seek to a certain start frame in the video with ``--frame-start`` alone
and render from there onward, this applies for keyword arguments in the ``--image-seeds`` definition as well.

If your slice only results in the processing of a single frame, an animated file format will
not be generated, only a single image output will be generated for that image seed during the
generation step.


.. code-block:: bash

    # Generate using only the first frame

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --image-seeds https://upload.wikimedia.org/wikipedia/commons/7/7b/Muybridge_race_horse_~_big_transp.gif \
    --image-seed-strengths 0.5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512 \
    --animation-format mp4 \
    --frame-start 0 \
    --frame-end 0


Inpainting Animations
=====================

Image seeds can be supplied an animated or static image mask to define the areas for inpainting while generating an animated output.

Any possible combination of image/video parameters can be used. The animation with least amount of frames in the entire
specification determines the frame count, and any static images present are duplicated across the entire animation.
The first animation present in an image seed specification always determines the output FPS of the animation.

When an animated seed is used with an animated mask, the mask for every corresponding frame in the input is taken from the animated mask,
the runtime of the animated output will be equal to the shorter of the two animated inputs. IE: If the seed animation and the mask animation
have different length, the animated output is clipped to the length of the shorter of the two.

When a static image is used as a mask, that image is used as an inpaint mask for every frame of the animated seed.

When an animated mask is used with a static image seed, the animated output length is that of the animated mask. A video is
created by duplicating the image seed for every frame of the animated mask, the animated output being generated by masking
them together.


.. code-block:: bash

    # A video with a static inpaint mask over the entire video

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.mp4;my-static-mask.png" \
    --output-path inpaint \
    --animation-format mp4

    # Zip two videos together, masking the left video with corresponding frames
    # from the right video. The two animated inputs do not have to be the same file format
    # you can mask videos with gif/webp and vice versa

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.mp4;my-animation-mask.mp4" \
    --output-path inpaint \
    --animation-format mp4

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.mp4;my-animation-mask.gif" \
    --output-path inpaint \
    --animation-format mp4

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.gif;my-animation-mask.gif" \
    --output-path inpaint \
    --animation-format mp4

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.gif;my-animation-mask.webp" \
    --output-path inpaint \
    --animation-format mp4

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.webp;my-animation-mask.gif" \
    --output-path inpaint \
    --animation-format mp4

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-animation.gif;my-animation-mask.mp4" \
    --output-path inpaint \
    --animation-format mp4

    # etc...

    # Use a static image seed and mask it with every frame from an
    # Animated mask file

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-static-image-seed.png;my-animation-mask.mp4" \
    --output-path inpaint \
    --animation-format mp4

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-static-image-seed.png;my-animation-mask.gif" \
    --output-path inpaint \
    --animation-format mp4

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --prompts "an astronaut riding a horse" \
    --image-seeds "my-static-image-seed.png;my-animation-mask.webp" \
    --output-path inpaint \
    --animation-format mp4

    # etc...



Deterministic Output
====================

If you generate an image you like using a random seed, you can later reuse that seed in another generation.

Updates to the backing model may affect determinism in the generation.

Output images have a name format that starts with the seed, IE: ``s_(seed here)_ ...png``

Reusing a seed has the effect of perfectly reproducing the image in the case that all
other parameters are left alone, including the model version.

You can output a configuration file for each image / animation produced that will reproduce it
exactly using the option ``--output-configs``, that same information can be written to the
metadata of generated PNG files using the option ``--output-metadata`` and can be read back
with ImageMagick for example as so:

.. code-block:: bash

    magick identify -format "%[Property:DgenerateConfig]" generated_file.png

Generated configuration can be read back into dgenerate via a pipe or file redirection.

.. code-block:: bash

    # DO NOT DO THIS IF THE IMAGE IS UNTRUSTED, SUCH AS IF IT IS SOMEONE ELSE'S IMAGE!
    # VERIFY THAT THE METADATA CONTENT OF THE IMAGE IS NOT MALICIOUS FIRST,
    # USING THE IDENTIFY COMMAND ALONE

    magick identify -format "%[Property:DgenerateConfig]" generated_file.png | dgenerate

    dgenerate < generated-config.dgen

Specifying a seed directly and changing the prompt slightly, or parameters such as image seed strength
if using a seed image, guidance scale, or inference steps, will allow for generating variations close
to the original image which may possess all of the original qualities about the image that you liked as well as
additional qualities.  You can further manipulate the AI into producing results that you want with this method.

Changing output resolution will drastically affect image content when reusing a seed to the point where trying to
reuse a seed with a different output size is pointless.

The following command demonstrates manually specifying two different seeds to try: ``1234567890``, and ``9876543210``

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --seeds 1234567890 9876543210 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


Specifying a specific GPU for CUDA
==================================

The desired GPU to use for CUDA acceleration can be selected using ``--device cuda:N`` where ``N`` is
the device number of the GPU as reported by ``nvidia-smi``.

.. code-block:: bash

    # Console 1, run on GPU 0

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut_1 \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512 \
    --device cuda:0

    # Console 2, run on GPU 1 in parallel

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a cow" \
    --output-path astronaut_2 \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512 \
    --device cuda:1


Specifying a Scheduler (sampler)
================================

A scheduler (otherwise known as a sampler) for the main model can be selected via the use of ``--scheduler``.

And in the case of SDXL the refiner's scheduler can be selected independently with ``--sdxl-refiner-scheduler``.

For Stable Cascade the decoder scheduler can be specified via the argument ``-s-cascade-decoder-scheduler``
however only one scheduler type is supported for Stable Cascade (``DDPMWuerstchenScheduler``).

Both of these default to the value of ``--scheduler``, which in turn defaults to automatic selection.

Available schedulers for a specific combination of dgenerate arguments can be
queried using ``--scheduler help``, ``--sdxl-refiner-scheduler help``, or ``--s-cascade-decoder-scheduler help``
though they cannot be queried simultaneously.

In order to use the query feature it is ideal that you provide all the other arguments
that you plan on using while making the query, as different combinations of arguments
will result in different underlying pipeline implementations being created, each of which
may have different compatible scheduler names listed. The model needs to be loaded in order to
gather this information.

For example there is only one compatible scheduler for this upscaler configuration:

.. code-block:: bash

    dgenerate stabilityai/sd-x2-latent-upscaler --variant fp16 --dtype float16 \
    --model-type torch-upscaler-x2 \
    --prompts "none" \
    --image-seeds my-image.png \
    --output-size 256 \
    --scheduler help

    # Outputs:
    #
    # Compatible schedulers for "stabilityai/sd-x2-latent-upscaler" are:
    #
    #    "EulerDiscreteScheduler"

Typically however, there will be many compatible schedulers:

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --gen-seeds 2 \
    --prompts "none" \
    --scheduler help

    # Outputs:
    #
    # Compatible schedulers for "stabilityai/stable-diffusion-2" are:
    #
    #     "DDIMScheduler"
    #     "DDPMScheduler"
    #     "DEISMultistepScheduler"
    #     "DPMSolverMultistepScheduler"
    #     "DPMSolverSDEScheduler"
    #     "DPMSolverSinglestepScheduler"
    #     "EDMEulerScheduler"
    #     "EulerAncestralDiscreteScheduler"
    #     "EulerDiscreteScheduler"
    #     "HeunDiscreteScheduler"
    #     "KDPM2AncestralDiscreteScheduler"
    #     "KDPM2DiscreteScheduler"
    #     "LCMScheduler"
    #     "LMSDiscreteScheduler"
    #     "PNDMScheduler"
    #     "UniPCMultistepScheduler"


Passing ``helpargs`` to a ``--scheduler`` related option will reveal configuration arguments that
can be overridden via a URI syntax, for every possible scheduler.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --gen-seeds 2 \
    --prompts "none" \
    --scheduler helpargs


    # Outputs (shortened for brevity...):
    #
    # Compatible schedulers for "stabilityai/stable-diffusion-2" are:
    #    ...
    #
    #    PNDMScheduler:
    #        num-train-timesteps=1000
    #        beta-start=0.0001
    #        beta-end=0.02
    #        beta-schedule=linear
    #        trained-betas=None
    #        skip-prk-steps=False
    #        set-alpha-to-one=False
    #        prediction-type=epsilon
    #        timestep-spacing=leading
    #        steps-offset=0
    #
    #   ...


As an example, you may override the mentioned arguments for any scheduler in this manner:

.. code-block:: bash

    # Change prediction type of the scheduler to "v_prediction".
    # for some models this may be necessary, not for this model
    # this is just a syntax example

    dgenerate stabilityai/stable-diffusion-2 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --gen-seeds 2 \
    --prompts "none" \
    --scheduler PNDMScheduler;prediction-type=v_prediction


Specifying a VAE
================

To specify a VAE directly use ``--vae``.

VAEs are supported for these model types:

    * ``--model-type torch``
    * ``--model-type flax``
    * ``--model-type torch-pix2pix``
    * ``--model-type torch-upscaler-x2``
    * ``--model-type torch-upscaler-x4``
    * ``--model-type torch-sdxl``
    * ``--model-type torch-sdxl-pix2pix``
    * ``--model-type torch-sd3``

The URI syntax for ``--vae`` is ``AutoEncoderClass;model=(huggingface repository slug/blob link or file/folder path)``

Named arguments when loading a VAE are separated by the ``;`` character and are not positional,
meaning they can be defined in any order.

Loading arguments available when specifying a VAE for torch ``--model-type`` values
are: ``model``, ``revision``, ``variant``, ``subfolder``, and ``dtype``

Loading arguments available when specifying VAE for flax ``--model-type`` values
are: ``model``, ``revision``, ``subfolder``, ``dtype``

The only named arguments compatible with loading a .safetensors or other model file
directly off disk are ``model`` and ``dtype``

The other named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk.

Available encoder classes for torch models are:

* AutoencoderKL
* AsymmetricAutoencoderKL (Does not support ``--vae-slicing`` or ``--vae-tiling``)
* AutoencoderTiny
* ConsistencyDecoderVAE

Available encoder classes for flax models are:

* FlaxAutoencoderKL (Does not support ``--vae-slicing`` or ``--vae-tiling``)


The AutoencoderKL encoder class accepts huggingface repository slugs/blob links,
.pt, .pth, .bin, .ckpt, and .safetensors files. Other encoders can only accept huggingface
repository slugs/blob links, or a path to a folder on disk with the model
configuration and model file(s).


.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --vae "AutoencoderKL;model=stabilityai/sd-vae-ft-mse" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``,
``subfolder`` is required in this example as well because the VAE model file exists in a subfolder
of the specified huggingface repository.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --revision fp16 \
    --dtype float16 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;revision=fp16;subfolder=vae" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you wish to specify a weights variant IE: load ``pytorch_model.<variant>.safetensors``, from a huggingface
repository that has variants of the same model, use the named argument ``variant``.  This usage is only
valid when loading VAEs if ``--model-type`` is either ``torch`` or ``torch-sdxl``.  Attempting
to use it with FlaxAutoencoderKL with produce an error message. When not specified in the URI,
this value does NOT default to the value ``--variant`` to prevent errors during common use cases.
If you wish to select a variant you must specify it in the URI.


.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --variant fp16 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;subfolder=vae;variant=fp16" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;subfolder=vae" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you want to specify the model precision, use the named argument ``dtype``,
accepted values are the same as ``--dtype``, IE: ``float32``, ``float16``, ``auto``

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --revision fp16 \
    --dtype float16 \
    --vae "AutoencoderKL;model=stabilityai/stable-diffusion-2-1;revision=fp16;subfolder=vae;dtype=float16" \
    --prompts "an astronaut riding a horse" \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512


If you are loading a .safetensors or other file from a path on disk, only the ``model``, and ``dtype``
arguments are available.

.. code-block:: bash

    # These are only syntax examples

    dgenerate huggingface/diffusion_model \
    --vae "AutoencoderKL;model=my_vae.safetensors" \
    --prompts "Syntax example"

    dgenerate huggingface/diffusion_model \
    --vae "AutoencoderKL;model=my_vae.safetensors;dtype=float16" \
    --prompts "Syntax example"

VAE Tiling and Slicing
======================

You can use ``--vae-tiling`` and ``--vae-slicing`` to enable to generation of huge images
without running your GPU out of memory. Note that if you are using ``--control-nets`` you may
still be memory limited by the size of the image being processed by the ControlNet, and still
may run in to memory issues with large image inputs.

When ``--vae-tiling`` is used, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of
memory and to allow processing larger images.

When ``--vae-slicing`` is used, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory, especially
when ``--batch-size`` is greater than 1.

.. code-block:: bash

    # Here is an SDXL example of high resolution image generation utilizing VAE tiling/slicing

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --vae "AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix" \
    --vae-tiling \
    --vae-slicing \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --output-size 2048 \
    --sdxl-target-size 2048 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


Specifying a UNet
=================

An alternate UNet model can be specified via a URI with the ``--unet`` option, in a
similar fashion to ``--vae`` and other model arguments that accept URIs.

UNets are supported for these model types:

    * ``--model-type torch``
    * ``--model-type flax``
    * ``--model-type torch-if``
    * ``--model-type torch-ifs``
    * ``--model-type torch-ifs-img2img``
    * ``--model-type torch-pix2pix``
    * ``--model-type torch-upscaler-x2``
    * ``--model-type torch-upscaler-x4``
    * ``--model-type torch-sdxl``
    * ``--model-type torch-sdxl-pix2pix``
    * ``--model-type torch-s-cascade``

This is useful in particular for using the latent consistency scheduler as well as the
``lite`` variants of the unet models used with Stable Cascade.

The first component of the ``--unet`` URI is the model path itself.

You can provide a path to a huggingface repo, or a folder on disk (downloaded huggingface repository).

The latent consistency UNet for SDXL can be specified with the ``--unet`` argument.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --unet latent-consistency/lcm-sdxl \
    --scheduler LCMScheduler \
    --inference-steps 4 \
    --guidance-scales 8 \
    --gen-seeds 2 \
    --output-size 1024 \
    --prompts "a close-up picture of an old man standing in the rain"

Loading arguments available when specifying a UNet for torch ``--model-type`` values
are: ``revision``, ``variant``, ``subfolder``, and ``dtype``


In the case of ``--unet`` the ``variant`` loading argument defaults to the value
of ``--variant`` if you do not specify it in the URI.


Loading arguments available when specifying UNet for flax ``--model-type`` values
are: ``revision``, ``subfolder``, ``dtype``. variant is not used for flax.


The ``--unet2`` option can be used to specify a UNet for the
`SDXL Refiner <#specifying-an-sdxl-refiner>`_ or `Stable Cascade Decoder <#specifying-a-stable-cascade-decoder>`_,
and uses the same syntax as ``--unet``.

Here is an example of using the ``lite`` variants of Stable Cascade's
UNet models which have a smaller memory footprint using ``--unet`` and ``--unet2``.

.. code-block:: bash

    dgenerate stabilityai/stable-cascade-prior \
    --model-type torch-s-cascade \
    --variant bf16 \
    --dtype bfloat16 \
    --unet "stabilityai/stable-cascade-prior;subfolder=prior_lite" \
    --unet2 "stabilityai/stable-cascade;subfolder=decoder_lite" \
    --model-cpu-offload \
    --s-cascade-decoder-cpu-offload \
    --s-cascade-decoder "stabilityai/stable-cascade;dtype=float16" \
    --inference-steps 20 \
    --guidance-scales 4 \
    --s-cascade-decoder-inference-steps 10 \
    --s-cascade-decoder-guidance-scales 0 \
    --gen-seeds 2 \
    --prompts "an image of a shiba inu, donning a spacesuit and helmet"


Specifying an SDXL Refiner
==========================

When the main model is an SDXL model and ``--model-type torch-sdxl`` is specified,
you may specify a refiner model with ``--sdxl-refiner``.

You can provide a path to a huggingface repo/blob link, folder on disk, or a model file
on disk such as a .pt, .pth, .bin, .ckpt, or .safetensors file.

This argument is parsed in much the same way as the argument ``--vae``, except the
model is the first value specified.

Loading arguments available when specifying a refiner are: ``revision``, ``variant``, ``subfolder``, and ``dtype``

The only named argument compatible with loading a .safetensors or other file directly off disk is ``dtype``

The other named arguments are available when loading from a huggingface repo/blob link,
or folder that may or may not be a local git repository on disk.

.. code-block:: bash

    # Basic usage of SDXL with a refiner

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"



If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner "stabilityai/stable-diffusion-xl-refiner-1.0;revision=main" \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


If you wish to specify a weights variant IE: load ``pytorch_model.<variant>.safetensors``, from a huggingface
repository that has variants of the same model, use the named argument ``variant``. By default this
value is the same as ``--variant`` unless you override it.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner "stabilityai/stable-diffusion-xl-refiner-1.0;variant=fp16" \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    # This is a non working example as I do not know of a repo with an SDXL refiner
    # in a subfolder :) this is only a syntax example

    dgenerate huggingface/sdxl_model --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner "huggingface/sdxl_refiner;subfolder=repo_subfolder"


If you want to select the model precision, use the named argument ``dtype``. By
default this value is the same as ``--dtype`` unless you override it. Accepted
values are the same as ``--dtype``, IE: 'float32', 'float16', 'auto'

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner "stabilityai/stable-diffusion-xl-refiner-1.0;dtype=float16" \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


If you are loading a .safetensors or other file from a path on disk, simply do:

.. code-block:: bash

    # This is only a syntax example

    dgenerate huggingface/sdxl_model --model-type torch-sdxl \
    --sdxl-refiner my_refinermodel.safetensors


When preforming inpainting or when using `ControlNets <#specifying-control-nets>`_, the
refiner will automatically operate in edit mode instead of cooperative denoising mode.
Edit mode can be forced in other situations with the option ``--sdxl-refiner-edit``.

Edit mode means that the refiner model is accepting the fully (or mostly) denoised output
of the main model generated at the full number of inference steps specified, and acting
on it with an image strength (image seed strength) determined by (1.0 - high-noise-fraction).

The output latent from the main model is renoised with a certain amount of noise determined
by the strength, a lower number means less noise and less modification of the latent output
by the main model.

This is similar to what happens when using dgenerate in img2img with a standalone model,
technically it is just img2img, however refiner models are better at enhancing details
from the main model in this use case.

Specifying a Stable Cascade Decoder
===================================

When the main model is a Stable Cascade prior model and ``--model-type torch-s-cascade`` is specified,
you may specify a decoder model with ``--s-cascade-decoder``.

The syntax (and URI arguments) for specifying the decoder model is identical to specifying an SDXL refiner
model as mentioned above.

.. code-block:: bash

    dgenerate stabilityai/stable-cascade-prior \
    --model-type torch-s-cascade \
    --variant bf16 \
    --dtype bfloat16 \
    --model-cpu-offload \
    --s-cascade-decoder-cpu-offload \
    --s-cascade-decoder "stabilityai/stable-cascade;dtype=float16" \
    --inference-steps 20 \
    --guidance-scales 4 \
    --s-cascade-decoder-inference-steps 10 \
    --s-cascade-decoder-guidance-scales 0 \
    --gen-seeds 2 \
    --prompts "an image of a shiba inu, donning a spacesuit and helmet"


Specifying LoRAs
================

It is possible to specify one or more LoRA models using ``--loras``

LoRAs are supported for these model types:

    * ``--model-type torch``
    * ``--model-type torch-pix2pix``
    * ``--model-type torch-upscaler-x4``
    * ``--model-type torch-sdxl``
    * ``--model-type torch-sdxl-pix2pix``
    * ``--model-type torch-sd3`` (scale not supported yet)

When multiple specifications are given, all mentioned models will be fused into
the main model at a given scale.

You can provide a huggingface repository slug, .pt, .pth, .bin, .ckpt, or .safetensors files.
Blob links are not accepted, for that use ``subfolder`` and ``weight-name`` described below.

The LoRA scale can be specified after the model path by placing a ``;`` (semicolon) and
then using the named argument ``scale``

When a scale is not specified, 1.0 is assumed.

Named arguments when loading a LoRA are separated by the ``;`` character and are
not positional, meaning they can be defined in any order.

Loading arguments available when specifying a LoRA are: ``scale``, ``revision``, ``subfolder``, and ``weight-name``

The only named argument compatible with loading a .safetensors or other file directly off disk is ``scale``

The other named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk.

This example shows loading a LoRA using a huggingface repository slug and specifying scale for it.

.. code-block:: bash

    # Don't expect great results with this example,
    # Try models and LoRA's downloaded from CivitAI

    dgenerate runwayml/stable-diffusion-v1-5 \
    --loras "pcuenq/pokemon-lora;scale=0.5" \
    --prompts "Gengar standing in a field at night under a full moon, highquality, masterpiece, digital art" \
    --inference-steps 40 \
    --guidance-scales 10 \
    --gen-seeds 5 \
    --output-size 800


Specifying the file in a repository directly can be done with the named argument ``weight-name``

Shown below is an SDXL compatible LoRA being used with the SDXL base model and a refiner.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --inference-steps 30 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --prompts "sketch of a horse by Leonardo da Vinci" \
    --variant fp16 --dtype float16 \
    --loras "goofyai/SDXL-Lora-Collection;scale=1.0;weight-name=leonardo_illustration.safetensors" \
    --output-size 1024


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    dgenerate runwayml/stable-diffusion-v1-5 \
    --loras "pcuenq/pokemon-lora;scale=0.5;revision=main" \
    --prompts "Gengar standing in a field at night under a full moon, highquality, masterpiece, digital art" \
    --inference-steps 40 \
    --guidance-scales 10 \
    --gen-seeds 5 \
    --output-size 800


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    # This is a non working example as I do not know of a repo with a LoRA weight in a subfolder :)
    # This is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --loras "huggingface/lora_repo;scale=1.0;subfolder=repo_subfolder;weight-name=lora_weights.safetensors"


If you are loading a .safetensors or other file from a path on disk, only the ``scale`` argument is available.

.. code-block:: bash

    # This is only a syntax example

    dgenerate runwayml/stable-diffusion-v1-5 \
    --prompts "Syntax example" \
    --loras "my_lora.safetensors;scale=1.0"


Specifying Textual Inversions
=============================

One or more Textual Inversion models (otherwise known as embeddings) may be specified with ``--textual-inversions``

Textual inversions are supported for these model types:

    * ``--model-type torch``
    * ``--model-type torch-pix2pix``
    * ``--model-type torch-upscaler-x4``
    * ``--model-type torch-sdxl``
    * ``--model-type torch-sdxl-pix2pix``

You can provide a huggingface repository slug, .pt, .pth, .bin, .ckpt, or .safetensors files.
Blob links are not accepted, for that use ``subfolder`` and ``weight-name`` described below.

Arguments pertaining to the loading of each textual inversion model may be specified in the same
way as when using ``--loras`` minus the scale argument.

Available arguments are: ``token``,  ``revision``, ``subfolder``, and ``weight-name``

Named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk, when loading directly from a .safetensors file
or other file from a path on disk they should not be used.

The ``token`` argument may be used to override the prompt token value, which is the text token
in the prompt that triggers the inversion, textual inversions for stable diffusion usually
include this token value in the model itself, for instance in the example below the token
for ``Isometric_Dreams-1000.pt`` is ``Isometric_Dreams``.

The token value used for SDXL (Stable Diffusion XL) models is a bit different, a default
value is not provided in the model file. If you do not provide a token value, dgenerate
will assign the tokens default value to the filename of the model with any spaces converted to
underscores, and with the file extension removed.


.. code-block:: bash

    # Load a textual inversion from a huggingface repository specifying it's name in the repository
    # as an argument

    dgenerate Duskfallcrew/isometric-dreams-sd-1-5  \
    --textual-inversions "Duskfallcrew/IsometricDreams_TextualInversions;weight-name=Isometric_Dreams-1000.pt" \
    --scheduler KDPM2DiscreteScheduler \
    --inference-steps 30 \
    --guidance-scales 7 \
    --prompts "a bright photo of the Isometric_Dreams, a tv and a stereo in it and a book shelf, a table, a couch,a room with a bed"


You can change the ``token`` value to affect the prompt token used to trigger the embedding

.. code-block:: bash

    # Load a textual inversion from a huggingface repository specifying it's name in the repository
    # as an argument

    dgenerate Duskfallcrew/isometric-dreams-sd-1-5  \
    --textual-inversions "Duskfallcrew/IsometricDreams_TextualInversions;weight-name=Isometric_Dreams-1000.pt;token=<MY_TOKEN>" \
    --scheduler KDPM2DiscreteScheduler \
    --inference-steps 30 \
    --guidance-scales 7 \
    --prompts "a bright photo of the <MY_TOKEN>, a tv and a stereo in it and a book shelf, a table, a couch,a room with a bed"


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    # This is a non working example as I do not know of a repo that utilizes revisions with
    # textual inversion weights :) this is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --textual-inversions "huggingface/ti_repo;revision=main"


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    # This is a non working example as I do not know of a repo with a textual
    # inversion weight in a subfolder :) this is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --textual-inversions "huggingface/ti_repo;subfolder=repo_subfolder;weight-name=ti_model.safetensors"


If you are loading a .safetensors or other file from a path on disk, simply do:

.. code-block:: bash

    # This is only a syntax example

    dgenerate runwayml/stable-diffusion-v1-5 \
    --prompts "Syntax example" \
    --textual-inversions "my_ti_model.safetensors"



Specifying Control Nets
=======================

One or more ControlNet models may be specified with ``--control-nets``, and multiple control
net guidance images can be specified via ``--image-seeds`` in the case that you specify
multiple control net models.

ControlNet models are supported for these model types:

    * ``--model-type torch``
    * ``--model-type flax``
    * ``--model-type torch-sdxl``
    * ``--model-type torch-sd3`` (img2img and inpainting not supported)

You can provide a huggingface repository slug / blob link, .pt, .pth, .bin, .ckpt, or .safetensors files.

Control images for the Control Nets can be provided using ``--image-seeds``

When using ``--control-nets`` specifying control images via ``--image-seeds`` can be accomplished in these ways:

    * ``--image-seeds "control-image.png"`` (txt2img)
    * ``--image-seeds "img2img-seed.png;control=control-image.png"`` (img2img)
    * ``--image-seeds "img2img-seed.png;mask=mask.png;control=control-image.png"`` (inpainting)

Multiple control image sources can be specified in these ways when using multiple control nets:

    * ``--image-seeds "control-1.png, control-2.png"`` (txt2img)
    * ``--image-seeds "img2img-seed.png;control=control-1.png, control-2.png"`` (img2img)
    * ``--image-seeds "img2img-seed.png;mask=mask.png;control=control-1.png, control-2.png"`` (inpainting)


It is considered a syntax error if you specify a non-equal amount of control guidance
images and ``--control-nets`` URIs and you will receive an error message if you do so.

``resize=WIDTHxHEIGHT`` can be used to select a per ``--image-seeds`` resize dimension for all image
sources involved in that particular specification, as well as ``aspect=true/false`` and the frame
slicing arguments ``frame-start`` and ``frame-end``.

ControlNet guidance images may actually be animations such as MP4s, GIFs etc. Frames can be
taken from multiple videos simultaneously. Any possible combination of image/video parameters can be used.
The animation with least amount of frames in the entire specification determines the frame count, and
any static images present are duplicated across the entire animation. The first animation present
in an image seed specification always determines the output FPS of the animation.

Arguments pertaining to the loading of each ControlNet model specified with ``--control-nets`` may be
declared in the same way as when using ``--vae`` with the addition of a ``scale`` argument and ``from_torch``
argument when using flax ``--model-type`` values.

Available arguments when using torch ``--model-type`` values are: ``scale``, ``start``, ``end``, ``revision``, ``variant``, ``subfolder``, ``dtype``

Available arguments when using flax ``--model-type`` values are: ``scale``, ``revision``, ``subfolder``, ``dtype``, ``from_torch``

Most named arguments apply to loading from a huggingface repository or folder
that may or may not be a local git repository on disk, when loading directly from a .safetensors file
or other file from a path on disk the available arguments are ``scale``, ``start``, ``end``, and ``from_torch``.
``from_torch`` can be used with flax for loading pytorch models from .pt or other files designed for torch from a repo or file/folder on disk.

The ``scale`` argument indicates the affect scale of the control net model.

For torch, the ``start`` argument indicates at what fraction of the total inference steps
at which the control net model starts to apply guidance. If you have multiple
control net models specified, they can apply guidance over different segments
of the inference steps using this option, it defaults to 0.0, meaning start at the
first inference step.

for torch, the ``end`` argument indicates at what fraction of the total inference steps
at which the control net model stops applying guidance. It defaults to 1.0, meaning
stop at the last inference step.


These examples use: `vermeer_canny_edged.png <https://raw.githubusercontent.com/Teriks/dgenerate/v3.10.0/examples/media/vermeer_canny_edged.png>`_


.. code-block:: bash

    # Torch example, use "vermeer_canny_edged.png" as a control guidance image

    dgenerate runwayml/stable-diffusion-v1-5 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earring by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets "lllyasviel/sd-controlnet-canny;scale=0.5" \
    --image-seeds "vermeer_canny_edged.png"


    # If you have an img2img image seed, use this syntax

    dgenerate runwayml/stable-diffusion-v1-5 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earring by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets "lllyasviel/sd-controlnet-canny;scale=0.5" \
    --image-seeds "my-image-seed.png;control=vermeer_canny_edged.png"


    # If you have an img2img image seed and an inpainting mask, use this syntax

    dgenerate runwayml/stable-diffusion-v1-5 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earring by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets "lllyasviel/sd-controlnet-canny;scale=0.5" \
    --image-seeds "my-image-seed.png;mask=my-inpaint-mask.png;control=vermeer_canny_edged.png"

    # Flax example

    dgenerate runwayml/stable-diffusion-v1-5 --model-type flax \
    --revision bf16 \
    --dtype float16 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earring by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets "lllyasviel/sd-controlnet-canny;scale=0.5;from_torch=true" \
    --image-seeds "vermeer_canny_edged.png"

    # SDXL example

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --vae "AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix" \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --prompts "Taylor Swift, high quality, masterpiece, high resolution; low quality, bad quality, sketches" \
    --control-nets "diffusers/controlnet-canny-sdxl-1.0;scale=0.5" \
    --image-seeds "vermeer_canny_edged.png" \
    --output-size 1024


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    # This is a non working example as I do not know of a repo that utilizes revisions with
    # ControlNet weights :) this is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --control-nets "huggingface/cn_repo;revision=main"


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    # This is a non working example as I do not know of a repo with a textual
    # inversion weight in a subfolder :) this is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --control-nets "huggingface/cn_repo;subfolder=repo_subfolder"


If you are loading a .safetensors or other file from a path on disk, simply do:

.. code-block:: bash

    # This is only a syntax example

    dgenerate runwayml/stable-diffusion-v1-5 \
    --prompts "Syntax example" \
    --control-nets "my_cn_model.safetensors"


Specifying Text Encoders
========================

Diffusion pipelines supported by dgenerate may use a varying number of
text encoder sub models, currently up to 3. ``--model-type torch-sd3``
for instance uses 3 text encoder sub models, all of which can be
individually specified from the command line if desired.

To specify a Text Encoder models directly use ``--text-encoders`` for
the primary model and ``--text-encoders2`` for the SDXL Refiner or
Stable Cascade decoder.

Text Encoder URIs do not support loading from blob links or a single file,
text encoders must be loaded from a huggingface slug or a folder on disk
containing the models and configuration.

The syntax for specifying text encoders is similar to that of ``--vae``

The URI syntax for ``--text-encoders`` is ``TextEncoderClass;model=(huggingface repository slug or folder path)``

Loading arguments available when specifying a Text Encoder for torch ``--model-type`` values
are: ``model``, ``revision``, ``variant``, ``subfolder``, and ``dtype``

The ``variant`` argument defaults to the value of ``--variant``

Loading arguments available when specifying a Text Encoder for flax ``--model-type`` values
are: ``model``, ``revision``, ``subfolder``, ``dtype``

In both cases, the ``dtype`` argument defaults to the value of ``--dtype``

The other named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk.

Available encoder classes for torch models are:

* CLIPTextModel
* CLIPTextModelWithProjection
* T5EncoderModel

Available encoder classes for flax models are:

* FlaxCLIPTextModel

You can query the text encoder types and position for a model by passing ``help``
as an argument to ``--text-encoders`` or ``--text-encoders2``. This feature
may not be used for both arguments simultaneously, and also may not be used
when passing ``help`` or ``helpargs`` to any ``--scheduler`` type argument.

.. code-block:: bash

    # ask for text encoder help on the main model that is mentioned

    dgenerate https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium_incl_clips.safetensors \
    --model-type torch-sd3 \
    --variant fp16 \
    --dtype float16 \
    --text-encoders help

    # outputs:

    # Text encoder type help:
    #
    #     0 = CLIPTextModelWithProjection
    #     1 = CLIPTextModelWithProjection
    #     2 = T5EncoderModel

    # this means that there are 3 text encoders that we
    # could potentially specify manually in the order
    # displayed for this model

When specifying multiple text encoders, a special syntax is allowed to indicate that
a text encoder should be loaded from defaults, this syntax involves the plus
symbol. When a plus symbol is encountered it is regarded as "use default".

For instance in the example below, only the last of the three text encoders
involved in the Stable Diffusion 3 pipeline is specified, as it is the only
one not included with the main model file.

This text encoder is loaded from a subfolder of the Stable Diffusion 3
repository on huggingface.

.. code-block:: bash

    # This is an example of individually specifying text encoders
    # specifically for stable diffusion 3, this model from the blob
    # link includes the clip encoders, so we only need to specify
    # the T5 encoder, which is encoder number 3, the + symbols indicate
    # the first 2 encoders are assigned their default value, they are
    # loaded from the checkpoint file for the main model

    dgenerate https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium_incl_clips.safetensors \
    --model-type torch-sd3 \
    --variant fp16 \
    --dtype float16 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --text-encoders + + \
        T5EncoderModel;model=stabilityai/stable-diffusion-3-medium-diffusers;subfolder=text_encoder_3 \
    --clip-skips 0 \
    --gen-seeds 2 \
    --output-path output \
    --model-sequential-offload \
    --prompts "a horse outside a barn"


You may also use the URI value ``null``, to indicate that you do not want to ever load a specific text encoder at all.

For instance, you can prevent Stable Diffusion 3 from loading and using the T5 encoder all together.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-3-medium-diffusers \
    --model-type torch-sd3 \
    --variant fp16 \
    --dtype float16 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --text-encoders + + null \
    --clip-skips 0 \
    --gen-seeds 2 \
    --output-path output \
    --model-sequential-offload \
    --prompts "a horse outside a barn"


Any text encoder shared via the ``\use_modules`` directive in a config files is considered a default
value for the text encoder in the next pipeline that runs, using ``+`` will maintain this value
and using ``null`` will override it.

.. code-block:: jinja

    #! dgenerate --file
    #! dgenerate 3.10.0

    # this model will load all three text encoders,
    # they are not cached individually as we did not explicitly
    # specify any of them, they are cached with the pipeline
    # as a whole

    stabilityai/stable-diffusion-3-medium-diffusers
    --model-type torch-sd3
    --variant fp16
    --dtype float16
    --inference-steps 30
    --guidance-scales 5.00
    --clip-skips 0
    --gen-seeds 2
    --output-path output
    --model-sequential-offload
    --prompts "a horse outside a barn"

    # store all the text encoders from the last pipeline
    # into the variable "encoders"

    \save_modules encoders text_encoder text_encoder_2 text_encoder_3

    # share them with the next pipeline

    \use_modules encoders

    # use all of the encoders except the T5 encoder (third encoder)
    # sharing modules this way saves a significant amount
    # of memory

    stabilityai/stable-diffusion-3-medium-diffusers
    --model-type torch-sd3
    --variant fp16
    --dtype float16
    --inference-steps 30
    --guidance-scales 5.00
    --clip-skips 0
    --text-encoders + + null
    --gen-seeds 2
    --output-path output
    --model-sequential-offload
    --prompts "a horse outside a barn"


Prompt Weighting and Enhancement
================================

By default, the prompt token weighting syntax that you may be familiar with from other software such as
`ComfyUI <https://github.com/comfyanonymous/ComfyUI>`_, `Stable Diffusion Web UI <Stable_Diffusion_Web_UI_1>`_,
and `CivitAI <CivitAI_1>`_ etc. is not enabled, and prompts over ``77`` tokens in length are not supported.

However! dgenerate implements prompt weighting and prompt enhancements through internal plugins
called prompt weighters, which can be selectively enabled to process your prompts. They support
special token weighting syntaxes, and overcome limitations on prompt length.

The names of all prompt weighter implementations can be seen by using the argument ``--prompt-weighter-help``,
and specific documentation for a prompt weighter can be printed py passing its name to this argument.

You may also use the config directive ``\prompt_weighter_help`` inside of a config, or
more likely when you are working inside the `Console UI`_ shell.

There are currently two prompt weighter implementations, the ``compel`` prompt weighter, and
the ``sd-embed`` prompt weighter.


The compel prompt weighter
--------------------------

The ``compel`` prompt weighter uses the `compel <https://github.com/damian0815/compel>`_ library to
support `InvokeAI <https://github.com/invoke-ai/InvokeAI>`_ style prompt token weighting syntax for
Stable Diffusion 1/2, and Stable Diffusion XL.

You can read about InvokeAI prompt syntax here: `Invoke AI prompting documentation <https://invoke-ai.github.io/InvokeAI/features/PROMPTS/>`_

It is a bit different than `Stable Diffusion Web UI <Stable_Diffusion_Web_UI_1>`_ syntax,
which is a syntax used by the majority of other image generation software. It possesses some neat
features not mentioned in this documentation, that are worth reading about in the links provided above.


.. code-block:: bash

    # print out the documentation for the compel prompt weighter

    dgenerate --prompt-weighter-help compel


.. code-block:: text

    compel:
        arguments:
            syntax: str = "compel"

        Implements prompt weighting syntax for Stable Diffusion 1/2 and Stable Diffusion XL using
        compel. The default syntax is "compel" which is analogous to the syntax used by InvokeAI.

        Specifying the syntax "sdwui" will translate your prompt from Stable Diffusion Web UI syntax
        into compel / InvokeAI syntax before generating the prompt embeddings.

        If you wish to use prompt syntax for weighting tokens that is similar to ComfyUI, Automatic1111,
        or CivitAI for example, use: 'compel;syntax=sdwui'

        The underlying weighting behavior for tokens is not exactly the same as other software that uses
        the more common "sdwui" syntax, so your prompt may need adjusting if you are reusing a prompt
        from those other pieces of software.

        You can read about compel here: https://github.com/damian0815/compel

        And InvokeAI here: https://github.com/invoke-ai/InvokeAI

        This prompt weighter supports the model types:

        --model-type torch
        --model-type torch-pix2pix
        --model-type torch-upscaler-x4
        --model-type torch-sdxl
        --model-type torch-sdxl-pix2pix

        The secondary prompt option for SDXL --sdxl-second-prompts is supported by this prompt weighter
        implementation. However, --sdxl-refiner-second-prompts is not supported and will be ignored
        with a warning message.

    ====================================================================================================


You can enable the ``compel`` prompt weighter by specifying it with the ``--prompt-weighter`` argument.

.. code-block:: bash

    # Some very simple examples

    # Increase the weight of (picking apricots)

    dgenerate stabilityai/stable-diffusion-2-1 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024 \
    --prompt-weighter compel \
    --prompts "a tall man (picking apricots)++"

    # Specify a weight

    dgenerate stabilityai/stable-diffusion-2-1 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024 \
    --prompt-weighter compel \
    --prompts "a tall man (picking apricots)1.3"


If you prefer the prompt weighting syntax used by Stable Diffusion Web UI, you can specify
the plugin argument ``syntax=sdwui`` which will translate your prompt from that syntax into
compel / InvokeAI syntax for you.


.. code-block:: bash

    # Some very simple examples

    # Increase the weight of (picking apricots)

    dgenerate stabilityai/stable-diffusion-2-1 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024 \
    --prompt-weighter "compel;syntax=sdwui" \
    --prompts "a tall man ((picking apricots))"

    # Specify a weight

    dgenerate stabilityai/stable-diffusion-2-1 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024 \
    --prompt-weighter "compel;syntax=sdwui" \
    --prompts "a tall man (picking apricots:1.3)"


The weighting algorithm is not entirely identical to other pieces of software, so if
you are migrating prompts they will likely require some adjustment.


The sd-embed prompt weighter
----------------------------

The ``sd-embed`` prompt weighter uses the `sd_embed <https://github.com/xhinker/sd_embed>`_ library to support
`Stable Diffusion Web UI <Stable_Diffusion_Web_UI_1>`_ style prompt token
weighting syntax for Stable Diffusion 1/2, Stable Diffusion XL, and Stable Diffusion 3.


The syntax that ``sd-embed`` uses is the more wide spread prompt syntax used by software such as
`Stable Diffusion Web UI <Stable_Diffusion_Web_UI_1>`_ and `CivitAI <CivitAI_1>`_


Quite notably, the ``sd-embed`` prompt weighter supports Stable Diffusion 3, where
as the ``compel`` prompt weighter currently does not.


.. code-block:: bash

    # print out the documentation for the sd-embed prompt weighter

    dgenerate --prompt-weighter-help sd-embed


.. code-block:: text

    sd-embed:

        Implements prompt weighting syntax for Stable Diffusion 1/2, Stable Diffusion XL, and Stable
        Diffusion 3 using sd_embed.

        sd_embed uses a Stable Diffusion Web UI compatible prompt syntax.

        See: https://github.com/xhinker/sd_embed

        @misc{sd_embed_2024,
          author       = {Shudong Zhu(Andrew Zhu)},
          title        = {Long Prompt Weighted Stable Diffusion Embedding},
          howpublished = {\url{https://github.com/xhinker/sd_embed}},
          year         = {2024},
        }

        --model-type torch
        --model-type torch-pix2pix
        --model-type torch-upscaler-x4
        --model-type torch-sdxl
        --model-type torch-sdxl-pix2pix
        --model-type torch-sd3

        The secondary prompt option for SDXL --sdxl-second-prompts is supported by this prompt weighter
        implementation. However, --sdxl-refiner-second-prompts is not supported and will be ignored with
        a warning message.

        The secondary prompt option for SD3 --sd3-second-prompts is not supported by this prompt weighter
        implementation.  Neither is --sd3-third-prompts. The prompts from these arguments will be ignored.

    ====================================================================================================


You can enable the ``sd-embed`` prompt weighter by specifying it with the ``--prompt-weighter`` argument.


.. code-block:: bash

    # You need a huggingface API token to run this example

    dgenerate stabilityai/stable-diffusion-3-medium-diffusers \
    --model-type torch-sd3 \
    --variant fp16 \
    --dtype float16 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024x1024 \
    --model-sequential-offload \
    --prompt-weighter sd-embed \
    --auth-token $HF_TOKEN \
    --prompts "a (man:1.2) standing on the (beach:1.2) looking out in to the water during a (sunset)"


Utilizing CivitAI links and Other Hosted Models
===============================================

Any model accepted by dgenerate that can be specified as a single file
inside of a URI or otherwise can be specified by a URL link to a model
file itself. dgenerate will attempt to download the file from the link,
store it in the web cache, and then use it.

You may also use the ``\download`` config directive to assist in pre
downloading other resources from the internet. The directive has the ability
to specify arbitrary storage locations. See: `The \\download directive`_

You can also use the ``download()`` template function for similar
purposes. See: `The download() template function`_

In the case of CivitAI you can use this to bake models into your script
that will be automatically downloaded for you, you just need a CivitAI
account and API token to download models.

Your API token can be created on this page: https://civitai.com/user/account

Near the bottom of the page in the section: ``API Keys``

You can use the `civitai-links`_ sub-command to fetch the necessary model links from a CivitAI model page.

If you plan to download many large models to the web cache in this manner you may wish
to adjust the global cache expiry time so that they exist in the cache longer than the default of 12 hours.

You can see how to change the cache expiry time in this section `File Cache Control`_

.. code-block:: bash

    #!/usr/bin/env bash

    # Download the main model from civitai using an api token

    # https://civitai.com/models/122822?modelVersionId=133832

    TOKEN=your_api_token_here

    MODEL="https://civitai.com/api/download/models/133832?type=Model&format=SafeTensor&size=full&fp=fp16&token=$TOKEN"

    dgenerate $MODEL \
    --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --sdxl-high-noise-fractions 0.8 \
    --guidance-scales 8 \
    --inference-steps 40 \
    --prompts "a fluffy cat playing in the grass"


This method can be used for VAEs, LoRAs, ControlNets, and Textual Inversions
as well whenever single file loads are supported by the argument.


Specifying Generation Batch Size
================================

Multiple image variations from the same seed can be produce on a GPU simultaneously
using the ``--batch-size`` option of dgenerate. This can be used in combination with
``--batch-grid-size`` to output image grids if desired.

When not writing to image grids the files in the batch will be written to disk
with the suffix ``_image_N`` where N is index of the image in the batch of images
that were generated.

When producing an animation, you can either write ``N`` animation output files
with the filename suffixes ``_animation_N`` where ``N`` is the index of the image
in the batch which makes up the frames.  Or you can use ``--batch-grid-size`` to
write frames to a single animated output where the frames are all image grids
produced from the images in the batch.

With larger ``--batch-size`` values, the use of ``--vae-slicing`` can make the difference
between an out of memory condition and success, so it is recommended that you
try this option if you experience an out of memory condition due to the use of
``--batch-size``.


Image Processors
================

Images provided through ``--image-seeds`` can be processed before being used for image generation
through the use of the arguments ``--seed-image-processors``, ``--mask-image-processors``, and
``--control-image-processors``. In addition, dgenerates output can be post processed with the
used of the ``--post-processors`` argument, which is useful for using the ``upscaler`` processor.
An important note about ``--post-processors`` is that post processing occurs before any image grid
rendering is preformed when ``--batch-grid-size`` is specified with a ``--batch-size`` greater than one,
meaning that the output images are processed with your processor before being put into a grid.

Each of these options can receive one or more specifications for image processing actions,
multiple processing actions will be chained together one after another.

Using the option ``--image-processor-help`` with no arguments will yield a list of available image processor names.

.. code-block:: bash

    dgenerate --image-processor-help

Output:

.. code-block:: text

    Available image processors:

        "anyline"
        "canny"
        "flip"
        "grayscale"
        "hed"
        "invert"
        "leres"
        "letterbox"
        "lineart"
        "lineart-anime"
        "lineart-standard"
        "midas"
        "mirror"
        "mlsd"
        "normal-bae"
        "openpose"
        "pidi"
        "posterize"
        "sam"
        "solarize"
        "teed"
        "upscaler"
        "upscaler-ncnn"
        "zoe"


Specifying one or more specific processors for example: ``--image-processor-help canny openpose`` will yield
documentation pertaining to those processor modules. This includes accepted arguments and their types for the
processor module and a description of what the module does.

Custom image processor modules can also be loaded through the ``--plugin-modules`` option as discussed
in the `Writing Plugins`_ section.

All processors posses the arguments: ``output-file`` and  ``output-overwrite``.

The ``output-file`` argument can be used to write the processed image to a specific file, if multiple
processing steps occur such as when rendering an animation or multiple generation steps, a numbered suffix
will be appended to this filename. Note that an output file will only be produced in the case that the
processor actually modifies an input image in some way. This can be useful for debugging an image that
is being fed into diffusion or a ControlNet.

The ``output-overwrite`` is a boolean argument can be used to tell the processor that you do not want numbered
suffixes to be generated for ``output-file`` and to simply overwrite it.

Some processors inherit the arguments: ``device``, and ``model-offload``.

The ``device`` argument can be used to override what device any hardware accelerated image processing
occurs on if any. It defaults to the value of ``--device`` and has the same syntax for specifying device
ordinals, for instance if you have multiple GPUs you may specify ``device=cuda:1`` to run image processing
on your second GPU, etc. Not all image processors respect this argument as some image processing is only
ever CPU based.

The ``model-offload`` argument is a boolean argument that can be used to force any torch modules / tensors
associated with an image processor to immediately evacuate the GPU or other non CPU processing device
as soon as the processor finishes processing an image.  Usually, any modules / tensors will be
brought on to the desired device right before processing an image, and left on the device until
the image processor object leaves scope and is garbage collected.

``model-offload`` can be useful for achieving certain GPU or processing device memory constraints, however
it is slower when processing multiple images in a row, as the modules / tensors must be brought on to the
desired device repeatedly for each image. In the context of dgenerate invocations where processors can
be used as preprocessors or postprocessors, the image processor object is garbage collected when the
invocation completes, this is also true for the ``\image_process`` directive.  Using this argument
with a preprocess specification, such as ``--control-image-processors`` may yield a noticeable memory
overhead reduction when using a single GPU, as any models from the image processor will be moved to the
CPU immediately when it is done, clearing up VRAM space before the diffusion models enter GPU VRAM.

For an example, images can be processed with the canny edge detection algorithm or OpenPose (rigging generation)
before being used for generation with a model + a ControlNet.

This image of a `horse <https://raw.githubusercontent.com/Teriks/dgenerate/v3.10.0/examples/media/horse2.jpeg>`_
is used in the example below with a ControlNet that is trained to generate images from canny edge detected input.

.. code-block:: bash

    # --control-image-processors is only used for control images
    # in this case the single image seed is considered a control image
    # because --control-nets is being used

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --vae "AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix" \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --prompts "Majestic unicorn, high quality, masterpiece, high resolution; low quality, bad quality, sketches" \
    --control-nets "diffusers/controlnet-canny-sdxl-1.0;scale=0.5" \
    --image-seeds "horse.jpeg" \
    --control-image-processors "canny;lower=50;upper=100" \
    --gen-seeds 2 \
    --output-size 1024 \
    --output-path unicorn


The ``--control-image-processors`` has a special additional syntax that the other processor specification
options do not, which is used to describe which processor group is affecting which control guidance image
source in an ``--image-seeds`` specification.

For instance if you have multiple control guidance images, and multiple control nets which are going
to use those images, or frames etc. and you want to process each guidance image with a separate
processor OR processor chain. You can specify how each image is processed by delimiting the
processor specification groups with + (the plus symbol)

Like this:

    * ``--control-nets "huggingface/controlnet1" "huggingface/controlnet2"``
    * ``--image-seeds "image1.png, image2.png"``
    * ``--control-image-processors "affect-image1" + "affect-image2"``


Specifying a non-equal amount of control guidance images and ``--control-nets`` URIs is
considered a syntax error and you will receive an error message if you do so.

You can use processor chaining as well:

    * ``--control-nets "huggingface/controlnet1" "huggingface/controlnet2"``
    * ``--image-seeds "image1.png, image2.png"``
    * ``--control-image-processors "affect-image1" "affect-image1-again" + "affect-image2"``

In the case that you would only like the second image affected:

    * ``--control-nets "huggingface/controlnet1" "huggingface/controlnet2"``
    * ``--image-seeds "image1.png, image2.png"``
    * ``--control-image-processors + "affect-image2"``


The plus symbol effectively creates a NULL processor as the first entry in the example above.

When multiple guidance images are present, it is a syntax error to specify more processor chains
than control guidance images.  Specifying less processor chains simply means that the trailing
guidance images will not be processed, you can avoid processing leading guidance images
with the mechanism described above.

This can be used with an arbitrary amount of control image sources and control nets, take
for example the specification:

    * ``--control-nets "huggingface/controlnet1" "huggingface/controlnet2" "huggingface/controlnet3"``
    * ``--image-seeds "image1.png, image2.png, image3.png"``
    * ``--control-image-processors + + "affect-image3"``


The two + (plus symbol) arguments indicate that the first two images mentioned in the control image
specification in ``--image-seeds`` are not to be processed by any processor.


Sub Commands
============

dgenerate implements additional functionality through the option ``--sub-command``.

For a list of available sub-commands use ``--sub-command-help``, which by default
will list available sub-command names.

For additional information on a specific sub-command use ``--sub-command-help NAME`` multiple
sub-command names can be specified here if desired however currently there is only one available.

All sub-commands respect the ``--plugin-modules`` and ``--verbose`` arguments
even if their help output does not specify them, these arguments are handled
by dgenerate and not the sub-command.

--sub-command image-process
---------------------------

The ``image-process`` sub-command can be used to run image processors implemented
by dgenerate on any file of your choosing including animated images and videos.

It has a similar but slightly different design/usage to the main dgenerate
command itself.

It can be used to run canny edge detection, openpose, etc. on any image or
video/animated file that you want.

The help output of ``image-process`` is as follows:


.. code-block:: text

    usage: \image_process [-h] [-p PROCESSORS [PROCESSORS ...]] [--plugin-modules PATH [PATH ...]]
                          [-o OUTPUT [OUTPUT ...]] [-ff FRAME_FORMAT] [-ox] [-r RESIZE] [-na]
                          [-al ALIGN] [-d DEVICE] [-fs FRAME_NUMBER] [-fe FRAME_NUMBER] [-nf | -naf]
                          input [input ...]

    This command allows you to use dgenerate image processors directly on files of your choosing.

    positional arguments:
      input                 Input file paths, may be a static images or animated files supported by
                            dgenerate. URLs will be downloaded.

    options:
      -h, --help            show this help message and exit
      -p PROCESSORS [PROCESSORS ...], --processors PROCESSORS [PROCESSORS ...]
                            One or more image processor URIs, specifying multiple will chain them
                            together. See: dgenerate --image-processor-help
      --plugin-modules PATH [PATH ...]
                            Specify one or more plugin module folder paths (folder containing
                            __init__.py) or python .py file paths to load as plugins. Plugin modules
                            can implement image processors.
      -o OUTPUT [OUTPUT ...], --output OUTPUT [OUTPUT ...]
                            Output files, parent directories mentioned in output paths will be created
                            for you if they do not exist. If you do not specify output files, the
                            output file will be placed next to the input file with the added suffix
                            '_processed_N' unless --output-overwrite is specified, in that case it
                            will be overwritten. If you specify multiple input files and output files,
                            you must specify an output file for every input file, or a directory
                            (indicated with a trailing directory seperator character, for example
                            "my_dir/" or "my_dir\" if the directory does not exist yet). Failure to
                            specify an output file with a URL as an input is considered an error.
                            Supported file extensions for image output are equal to those listed under
                            --frame-format.
      -ff FRAME_FORMAT, --frame-format FRAME_FORMAT
                            Image format for animation frames. Must be one of: png, apng, blp, bmp,
                            dib, bufr, pcx, dds, ps, eps, gif, grib, h5, hdf, jp2, j2k, jpc, jpf, jpx,
                            j2c, icns, ico, im, jfif, jpe, jpg, jpeg, tif, tiff, mpo, msp, palm, pdf,
                            pbm, pgm, ppm, pnm, pfm, bw, rgb, rgba, sgi, tga, icb, vda, vst, webp,
                            wmf, emf, or xbm.
      -ox, --output-overwrite
                            Indicate that it is okay to overwrite files, instead of appending a
                            duplicate suffix.
      -r RESIZE, --resize RESIZE
                            Preform naive image resizing (LANCZOS).
      -na, --no-aspect      Make --resize ignore aspect ratio.
      -al ALIGN, --align ALIGN
                            Align images / videos to this value in pixels, default is 8. Specifying 1
                            will disable resolution alignment.
      -d DEVICE, --device DEVICE
                            Processing device, for example "cuda", "cuda:1".
      -fs FRAME_NUMBER, --frame-start FRAME_NUMBER
                            Starting frame slice point for animated files (zero-indexed), the
                            specified frame will be included. (default: 0)
      -fe FRAME_NUMBER, --frame-end FRAME_NUMBER
                            Ending frame slice point for animated files (zero-indexed), the specified
                            frame will be included.
      -nf, --no-frames      Do not write frames, only an animation file. Cannot be used with --no-
                            animation-file.
      -naf, --no-animation-file
                            Do not write an animation file, only frames. Cannot be used with --no-
                            frames.


Overview of specifying ``image-process`` inputs and outputs

.. code-block:: bash

    # Overview of specifying outputs, image-process can do simple operations
    # like resizing images and forcing image alignment with --align, without the
    # need to specify any other processing operations with --processors. Running
    # image-process on an image with no other arguments simply aligns it to 8 pixels,
    # given the defaults for its command line arguments

    # More file formats than .png are supported for static image output, all
    # extensions mentioned in the image-process --help documentation for --frame-format
    # are supported, the supported formats are identical to that mentioned in the --image-format
    # option help section of dgenerates --help output

    # my_file.png -> my_file_processed_1.png

    dgenerate --sub-command image-process my_file.png --resize 512x512

    # my_file.png -> my_file.png (overwrite)

    dgenerate --sub-command image-process my_file.png --resize 512x512 --output-overwrite

    # my_file.png -> my_file.png (overwrite)

    dgenerate --sub-command image-process my_file.png -o my_file.png --resize 512x512 --output-overwrite

    # my_file.png -> my_dir/my_file_processed_1.png

    dgenerate --sub-command image-process my_file.png -o my_dir/ --resize 512x512 --no-aspect

    # my_file_1.png -> my_dir/my_file_1_processed_1.png
    # my_file_2.png -> my_dir/my_file_2_processed_2.png

    dgenerate --sub-command image-process my_file_1.png my_file_2.png -o my_dir/ --resize 512x512

    # my_file_1.png -> my_dir_1/my_file_1_processed_1.png
    # my_file_2.png -> my_dir_2/my_file_2_processed_2.png

    dgenerate --sub-command image-process my_file_1.png my_file_2.png \
    -o my_dir_1/ my_dir_2/ --resize 512x512

    # my_file_1.png -> my_dir_1/renamed.png
    # my_file_2.png -> my_dir_2/my_file_2_processed_2.png

    dgenerate --sub-command image-process my_file_1.png my_file_2.png \
    -o my_dir_1/renamed.png my_dir_2/ --resize 512x512


A few usage examples with processors:

.. code-block:: bash

    # image-process can support any input format that dgenerate itself supports
    # including videos and animated files. It also supports all output formats
    # supported by dgenerate for writing videos/animated files, and images.

    # create a video rigged with OpenPose, frames will be rendered to the directory "output" as well.

    dgenerate --sub-command image-process my-video.mp4 \
    -o output/rigged-video.mp4 --processors "openpose;include-hand=true;include-face=true"

    # Canny edge detected video, also using processor chaining to mirror the frames
    # before they are edge detected

    dgenerate --sub-command image-process my-video.mp4 \
    -o output/canny-video.mp4 --processors mirror "canny;blur=true;threshold-algo=otsu"


--sub-command civitai-links
---------------------------

The ``civitai-links`` subcommand can be used to list all of the hard links for models available on a CivitAI model page.

These links can be used directly with dgenerate, it will automatically download the model for you.

You only need to select which models you wish to use from the links listed by this command.

See: `Utilizing CivitAI links and Other Hosted Models`_ for more information about how to use these links.

To get direct links to civit AI models you can use the ``civitai-links`` subcommand
or the ``\civitai_links`` directive inside of a config to list all available models
on a CivitAI model page.

For example:

.. code-block:: bash

    # get links for the Crystal Clear XL model on CivitAI

    dgenerate --sub-command civitai-links https://civitai.com/models/122822?modelVersionId=133832

    # you can also automatically append your API token to the end of the URLs with --token
    # some models will require that you authenticate to download, this will add your token
    # to the URL for you

    dgenerate --sub-command civitai-links https://civitai.com/models/122822?modelVersionId=133832 --token $MY_API_TOKEN


This will list every model link on the page, with title, there may be many model links
depending on what the page has available for download.

Output from the above example:

.. code-block:: txt

    Models at: https://civitai.com/models/122822?modelVersionId=133832
    ==================================================================

    CCXL (Model): https://civitai.com/api/download/models/133832?format=SafeTensor&size=full&fp=fp16


Upscaling
=========

dgenerate implements four different methods of upscaling images, animated images, or video.

Upscaling with the Stable Diffusion based x2 and x4 upscalers.

With the `upscale` image processor which is compatible with models implemented in the `spandrel <https://github.com/chaiNNer-org/spandrel>`_ module.

And with the `upscaler-ncnn` image processor, which implements upscaling with NCNN upscaling models
compatible with `upscayl <https://github.com/upscayl/upscayl>`_ , or `chaiNNer <chaiNNer_1>`_ and similar software.


Upscaling with Diffusion Upscaler Models
----------------------------------------

Stable diffusion image upscaling models can be used via the model types ``torch-upscaler-x2`` and ``torch-upscaler-x4``.

The image used in the example below is this `low resolution cat <https://raw.githubusercontent.com/Teriks/dgenerate/v3.10.0/examples/media/low_res_cat.png>`_

.. code-block:: bash

    # The image produced with this model will be
    # two times the --output-size dimension IE: 512x512 in this case
    # The image is being resized to 256x256, and then upscaled by 2x

    dgenerate stabilityai/sd-x2-latent-upscaler --variant fp16 --dtype float16 \
    --model-type torch-upscaler-x2 \
    --prompts "a picture of a white cat" \
    --image-seeds low_res_cat.png \
    --output-size 256


    # The image produced with this model will be
    # four times the --output-size dimension IE: 1024x1024 in this case
    # The image is being resized to 256x256, and then upscaled by 4x

    dgenerate stabilityai/stable-diffusion-x4-upscaler --variant fp16 --dtype float16 \
     --model-type torch-upscaler-x4 \
    --prompts "a picture of a white cat" \
    --image-seeds low_res_cat.png \
    --output-size 256 \
    --upscaler-noise-levels 20


Upscaling with chaiNNer Compatible Upscaler Models
--------------------------------------------------

`chaiNNer <chaiNNer_1>`_ compatible upscaler models from https://openmodeldb.info/
and elsewhere can be utilized for tiled upscaling using dgenerates ``upscaler`` image processor and the
``--post-processors`` option.  The ``upscaler`` image processor can also be used for processing
input images via the other options mentioned in `Image Processors`_ such as ``--seed-image-processors``

The ``upscaler`` image processor can make use of URLs or files on disk.

In this example we reference a link to the SwinIR x4 upscaler from the creators github release.

This uses the upscaler to upscale the output image by x4 producing an image that is 4096x4096

The ``upscaler`` image processor respects the ``--device`` option of dgenerate, and is CUDA accelerated by default.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork" \
    --post-processors "upscaler;model=https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"


In addition to this the ``\image_process`` config directive, or ``--sub-command image-process`` can be used to upscale
any file that you want including animated images and videos. It is worth noting that the sub-command and directive
will work with any named image processor implemented by dgenerate.


.. code-block:: bash

    # print the help output of the sub command "image-process"
    # the image-process subcommand can process multiple files and do
    # and several other things, it is worth reading :)

    dgenerate --sub-command image-process --help

    # any directory mentioned in the output spec is created automatically

    dgenerate --sub-command image-process my-file.png \
    --output output/my-file-upscaled.png \
    --processors "upscaler;model=https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"


Upscaling with NCNN Upscaler Models
-----------------------------------

The ``upscaler-ncnn`` image processor will be available if you have manually installed dgenerate
with the ``[ncnn]`` extra, or if you are using dgenerate from the packaged windows installer or portable
windows install zip from the releases page.

NCNN can use Vulkan for hardware accelerated inference and is also heavily optimized for CPU use
if needed.

It is not recommended to use this upscaler as a post-process or pre-process step on the GPU for a dgenerate
invocation involving diffusion, as the Vulkan allocator in NCNN does not play very nice with the
torch allocator used for diffusion with dgenerate. It will likely hard crash your system, unless
you have another GPU available to run the ncnn upscaler on in parallel.

When using the ``upscaler-ncnn`` processor, you must specify both the ``model`` and ``param`` arguments,
these refer to the ``model.bin`` and ``model.param`` file associated with the model.

These arguments may be a path to a file on disk or a hard link to a downloadable model in raw form.

By default the ``upscaler-ncnn`` processor does not run on the GPU, you must enable this with the ``use-gpu``
argument shown below, just be careful not to crash your system by using it along side diffusion on
the same GPU.

You can set the GPU index that you wish the processor to run on using the ``gpu-index`` argument,
since the ncnn upscaler can run on GPUs other than Nvidia GPUs, figuring out what index
you need to use is platform specific, but for Nvidia users just use the ``nvidia-smi`` command
from a terminal to get this value.

If you do not specify a ``gpu-index``, index 0 is used, which is most likely your main GPU.

The ``--device`` argument to dgenerate and the ``image-process`` subcommand / ``\image_process`` directive
is ignored by this image processor.

 .. code-block:: bash

     #! /usr/bin/env bash

     # this auto downloads x2 upscaler models from the upscayl repository into
     # dgenerates web cache, and then use them

     MODEL=https://github.com/upscayl/upscayl/raw/main/models/realesr-animevideov3-x2.bin
     PARAM=https://github.com/upscayl/upscayl/raw/main/models/realesr-animevideov3-x2.param

     dgenerate --sub-command image-process my-file.png \
     --output output/my-file-upscaled.png \
     --processors "upscaler-ncnn;model=${MODEL};param=${PARAM};use-gpu=true"

If you are upscaling using the CPU, you can specify a thread count using the ``threads`` argument.

This argument can be an integer quantity of threads, the keyword ``auto``
(max logical processors, max threads) or the keyword ``half`` (half your logical processors).

 .. code-block:: bash

     #! /usr/bin/env bash

     # this auto downloads x2 upscaler models from the upscayl repository into
     # dgenerates web cache, and then use them

     MODEL=https://github.com/upscayl/upscayl/raw/main/models/realesr-animevideov3-x2.bin
     PARAM=https://github.com/upscayl/upscayl/raw/main/models/realesr-animevideov3-x2.param

     dgenerate --sub-command image-process my-file.png \
     --output output/my-file-upscaled.png \
     --processors "upscaler-ncnn;model=${MODEL};param=${PARAM};threads=half"


The argument ``winograd=true`` can be used to enable the winograd convolution when running on CPU,
similarly the ``sgemm=true`` argument can be used to enable the sgemm convolution optimization.

In addition, you can control OpenMP blocktime using the ``blocktime`` argument, which should be
an integer value between 0 and 400 inclusive, representing milliseconds.

These arguments can only be used when running on the CPU and will throw an argument error otherwise.

When they are not specified, optimal defaults from ncnn for your platform are used.


Writing and Running Configs
===========================

Config scripts can be read from ``stdin`` using a shell pipe or file redirection, or by
using the ``--file`` argument to specify a file to interpret.

Config scripts are processed with model caching and other optimizations, in order
to increase speed when many dgenerate invocations with different arguments are desired.

Loading the necessary libraries and bringing models into memory is quite slow, so using dgenerate
this way allows for multiple invocations using different arguments, without needing to load the
machine learning libraries and models multiple times in a row.

When a model is loaded dgenerate caches it in memory with it's creation parameters, which includes
among other things the pipeline mode (basic, img2img, inpaint), user specified UNets, VAEs, LoRAs,
Textual Inversions, and ControlNets.

If another invocation of the model occurs with creation parameters that are identical, it will be
loaded out of an in memory cache, which greatly increases the speed of the invocation.

Diffusion Pipelines, user specified UNets, VAEs, Text Encoders, and ControlNet models are cached individually.

UNets, VAEs, Text Encoder, and ControlNet model objects can be reused by diffusion pipelines in certain
situations when specified explicitly and this is taken advantage of by using an in
memory cache of these objects.

In effect, the creation of a diffusion pipeline is memoized, as well as the creation of
any pipeline subcomponents when you have specified them explicitly with a URI.

A number of things effect cache hit or miss upon a dgenerate invocation, extensive information
regarding runtime caching behavior of a pipelines and other models can be observed using ``-v/--verbose``

When loading multiple different models be aware that they will all be retained in memory for
the duration of program execution, unless all models are flushed using the ``\clear_model_cache`` directive or
individually using one of: ``\clear_pipeline_cache``, ``\clear_unet_cache``, ``\clear_vae_cache``,
``\clear_text_encoder_cache``, or ``\clear_control_net_cache``.

dgenerate uses heuristics to clear the in memory cache automatically when needed, including a size estimation
of models before they enter system memory, however by default it will use system memory very aggressively
and it is not entirely impossible to run your system out of memory if you are not careful.

Basic config syntax
-------------------

The basic idea of the dgenerate config syntax is that it is a pseudo Unix shell mixed with Jinja2 templating.

The config language provides many niceties for batch processing large amounts of images
and image output in a Unix shell like environment with Jinja2 control constructs.

Shell builtins, known as directives, are prefixed with ``\``, for example: ``\print``

Environmental variables will be expanded in config scripts using both Unix and Windows CMD syntax

.. code-block:: jinja

    # these all expand from your system environment
    # if the variable is not set, they expand to nothing

    \print $VARIABLE
    \print ${VARIABLE}
    \print %VARIABLE%

Empty lines and comments starting with ``#`` will be ignored, comments that occur at the end of lines will also be ignored.

You can create a multiline continuation using ``\`` to indicate that a line continues similar to bash.

Unlike bash, if the next line starts with ``-`` it is considered part of a continuation as well
even if ``\`` had not been used previously. This allows you to list out many Posix style shell
options starting with ``-`` without having to end every line with ``\``.

Comments can be interspersed with invocation or directive arguments
on their own line with the use of ``\`` on the last line before
comments and whitespace begin. This can be used to add documentation
above individual arguments instead of at the tail end of them.

The following is a config file example that covers the most basic syntax concepts.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # If a hash-bang version is provided in the format above
    # a warning will be produced if the version you are running
    # is not compatible (SemVer), this can be used anywhere in the
    # config file, a line number will be mentioned in the warning when the
    # version check fails

    # Comments in the file will be ignored

    # Each dgenerate invocation in the config begins with the path to a model,
    # IE. the first argument when using dgenerate from the command line, the
    # rest of the options that follow are the options to dgenerate that you
    # would use on the command line

    # Guarantee unique file names are generated under the output directory by specifying unique seeds

    stabilityai/stable-diffusion-2-1 --prompts "an astronaut riding a horse" --seeds 41509644783027 --output-path output --inference-steps 30 --guidance-scales 10
    stabilityai/stable-diffusion-2-1 --prompts "a cowboy riding a horse" --seeds 78553317097366 --output-path output --inference-steps 30 --guidance-scales 10
    stabilityai/stable-diffusion-2-1 --prompts "a martian riding a horse" --seeds 22797399276707 --output-path output --inference-steps 30 --guidance-scales 10

    # Guarantee that no file name collisions happen by specifying different output paths for each invocation

    stabilityai/stable-diffusion-2-1 --prompts "an astronaut riding a horse" --output-path unique_output_1  --inference-steps 30 --guidance-scales 10
    stabilityai/stable-diffusion-2-1 --prompts "a cowboy riding a horse" --output-path unique_output_2 --inference-steps 30 --guidance-scales 10

    # Multiline continuations are possible implicitly for argument
    # switches IE lines starting with '-'

    stabilityai/stable-diffusion-2-1 --prompts "a martian riding a horse"
    --output-path unique_output_3  # there can be comments at the end of lines
    --inference-steps 30 \         # this comment is also ignored

    # There can be comments or newlines within the continuation
    # but you must provide \ on the previous line to indicate that
    # it is going to happen

    --guidance-scales 10

    # The continuation ends (on the next line) when the last line does
    # not end in \ or start with -

    # the ability to use tail comments means that escaping of the # is sometimes
    # necessary when you want it to appear literally, see: examples/config_syntax/tail-comments-config.dgen
    # for examples.


    # Configuration directives provide extra functionality in a config, a directive
    # invocation always starts with a backslash

    # A clear model cache directive can be used inbetween invocations if cached models that
    # are no longer needed in your generation pipeline start causing out of memory issues

    \clear_model_cache

    # Additionally these other directives exist to clear user loaded models
    # out of dgenerates in memory cache individually

    # Clear specifically diffusion pipelines

    \clear_pipeline_cache

    # Clear specifically user specified UNet models

    \clear_unet_cache

    # Clear specifically user specified VAE models

    \clear_vae_cache

    # Clear specifically user specified Text Encoder models

    \clear_text_encoder_cache

    # Clear specifically ControlNet models

    \clear_control_net_cache


    # This model was used before but will have to be fully instantiated from scratch again
    # after a cache flush which may take some time

    stabilityai/stable-diffusion-2-1 --prompts "a martian riding a horse"
    --output-path unique_output_4


Built in template variables and functions
-----------------------------------------

There is valuable information about the previous invocation of dgenerate that
is set in the environment and available to use via Jinja2 templating or in
the ``\setp`` directive, some of these include:

* ``{{ last_images }}`` (An iterable of un-quoted filenames which were generated)
* ``{{ last_animations }}`` (An iterable of un-quoted filenames which were generated)

There are template variables for prompts, containing the previous prompt values:

* ``{{ last_prompts }}`` (List of prompt objects with the un-quoted attributes 'positive' and 'negative')
* ``{{ last_sdxl_second_prompts }}``
* ``{{ last_sdxl_refiner_prompts }}``
* ``{{ last_sdxl_refiner_second_prompts }}``

Some available custom jinja2 functions/filters are:

* ``{{ first(list_of_items) }}`` (First element in a list)
* ``{{ last(list_of_items) }}`` (Last element in a list)
* ``{{ unquote('"unescape-me"') }}`` (shell unquote / split, works on strings and lists)
* ``{{ quote('escape-me') }}`` (shell quote, works on strings and lists)
* ``{{ format_prompt(prompt_object) }}`` (Format and quote one or more prompt objects with their delimiter, works on single prompts and lists)
* ``{{ format_size(size_tuple) }}`` (Format a size tuple / iterable, join with "x" character)
* ``{{ align_size('700x700', 8) }}`` (Align a size string or tuple to a specific alignment, return a formatted string by default)
* ``{{ pow2_size('700x700', 8) }}`` (Round a size string or tuple to the nearest power of 2, return a formatted string by default)
* ``{{ size_is_aligned('700x700', 8) }}`` (Check if a size string or tuple is aligned to a specific alignment, return ``True`` or ``False``)
* ``{{ size_is_pow2('700x700') }}`` (Check if a size string or tuple is a power of 2 dimension, return ``True`` or ``False``)
* ``{{ format_model_type(last_model_type) }}`` (Format a ``ModelType`` enum to a value to ``--model-type``)
* ``{{ format_dtype(last_dtype) }}`` (Format a ``DataType`` enum to a value to ``--dtype``)
* ``{{ gen_seeds(n) }}`` (Return a list of random integer seeds in the form of strings)
* ``{{ cwd() }}`` (Return the current working directory as a string)
* ``{{ download(url) }}`` (Download from a url to the web cache and return the file path)
* ``{{ have_feature(feature_name) }}`` (Check for feature and return bool, value examples: ``flax``, or ``ncnn``)

The above functions which possess arguments can be used as either a function or filter IE: ``{{ "quote_me" | quote }}``

The option ``--functions-help`` and the directive ``\functions_help`` can be used to print
documentation for template functions. When the option or directive is used alone all built
in functions will be printed with their signature, specifying function names as arguments
will print documentation for those specific functions.

To receive information about Jinja2 template variables that are set after a dgenerate invocation.
You can use the ``\templates_help`` directive which is similar to the ``--templates-help`` option
except it will print out all of the template variables assigned values instead of just their
names and types. This is useful for figuring out the values of template variables set after
a dgenerate invocation in a config file for debugging purposes. You can specify one or
more template variable names as arguments to ``\templates_help`` to receive help for only
the mentioned variable names.

Template variables set with the ``\set``, ``\setp``, and ``\sete`` directive will
also be mentioned in this output.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # Invocation will proceed as normal

    stabilityai/stable-diffusion-2-1 --prompts "a man walking on the moon without a space suit"

    # Print all set template variables

    \templates_help


The ``\templates_help`` output from the above example is:

.. code-block:: text

    Config template variables are:
    
        Name: "glob"
            Type: <class 'module'>
        Name: "injected_args"
            Type: collections.abc.Sequence[str]
        Name: "injected_device"
            Type: typing.Optional[str]
        Name: "injected_plugin_modules"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "injected_verbose"
            Type: typing.Optional[bool]
        Name: "last_animation_format"
            Type: <class 'str'>
        Name: "last_animations"
            Type: collections.abc.Iterable[str]
        Name: "last_auth_token"
            Type: typing.Optional[str]
        Name: "last_batch_grid_size"
            Type: typing.Optional[tuple[int, int]]
        Name: "last_batch_size"
            Type: typing.Optional[int]
        Name: "last_clip_skips"
            Type: typing.Optional[collections.abc.Sequence[int]]
        Name: "last_control_image_processors"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "last_control_net_uris"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "last_device"
            Type: <class 'str'>
        Name: "last_dtype"
            Type: <enum 'DataType'>
        Name: "last_frame_end"
            Type: typing.Optional[int]
        Name: "last_frame_start"
            Type: <class 'int'>
        Name: "last_guidance_rescales"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_guidance_scales"
            Type: collections.abc.Sequence[float]
        Name: "last_image_format"
            Type: <class 'str'>
        Name: "last_image_guidance_scales"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_image_seed_strengths"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_image_seeds"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "last_images"
            Type: collections.abc.Iterable[str]
        Name: "last_inference_steps"
            Type: collections.abc.Sequence[int]
        Name: "last_lora_uris"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "last_mask_image_processors"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "last_model_cpu_offload"
            Type: <class 'bool'>
        Name: "last_model_path"
            Type: typing.Optional[str]
        Name: "last_model_sequential_offload"
            Type: <class 'bool'>
        Name: "last_model_type"
            Type: <enum 'ModelType'>
        Name: "last_no_aspect"
            Type: <class 'bool'>
        Name: "last_no_frames"
            Type: <class 'bool'>
        Name: "last_offline_mode"
            Type: <class 'bool'>
        Name: "last_output_configs"
            Type: <class 'bool'>
        Name: "last_output_metadata"
            Type: <class 'bool'>
        Name: "last_output_overwrite"
            Type: <class 'bool'>
        Name: "last_output_path"
            Type: <class 'str'>
        Name: "last_output_prefix"
            Type: typing.Optional[str]
        Name: "last_output_size"
            Type: typing.Optional[tuple[int, int]]
        Name: "last_parsed_image_seeds"
            Type: typing.Optional[collections.abc.Sequence[dgenerate.mediainput.ImageSeedParseResult]]
        Name: "last_post_processors"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "last_prompt_weighter_uri"
            Type: typing.Optional[str]
        Name: "last_prompts"
            Type: collections.abc.Sequence[dgenerate.prompt.Prompt]
        Name: "last_revision"
            Type: <class 'str'>
        Name: "last_s_cascade_decoder_cpu_offload"
            Type: typing.Optional[bool]
        Name: "last_s_cascade_decoder_guidance_scales"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_s_cascade_decoder_inference_steps"
            Type: typing.Optional[collections.abc.Sequence[int]]
        Name: "last_s_cascade_decoder_prompts"
            Type: typing.Optional[collections.abc.Sequence[dgenerate.prompt.Prompt]]
        Name: "last_s_cascade_decoder_scheduler"
            Type: typing.Optional[str]
        Name: "last_s_cascade_decoder_sequential_offload"
            Type: typing.Optional[bool]
        Name: "last_s_cascade_decoder_uri"
            Type: typing.Optional[str]
        Name: "last_safety_checker"
            Type: <class 'bool'>
        Name: "last_scheduler"
            Type: typing.Optional[str]
        Name: "last_sd3_max_sequence_length"
            Type: typing.Optional[int]
        Name: "last_sd3_second_prompts"
            Type: typing.Optional[collections.abc.Sequence[dgenerate.prompt.Prompt]]
        Name: "last_sd3_third_prompts"
            Type: typing.Optional[collections.abc.Sequence[dgenerate.prompt.Prompt]]
        Name: "last_sdxl_aesthetic_scores"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_sdxl_crops_coords_top_left"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_high_noise_fractions"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_sdxl_negative_aesthetic_scores"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_sdxl_negative_crops_coords_top_left"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_negative_original_sizes"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_negative_target_sizes"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_original_sizes"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_refiner_aesthetic_scores"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_sdxl_refiner_clip_skips"
            Type: typing.Optional[collections.abc.Sequence[int]]
        Name: "last_sdxl_refiner_cpu_offload"
            Type: typing.Optional[bool]
        Name: "last_sdxl_refiner_crops_coords_top_left"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_refiner_edit"
            Type: typing.Optional[bool]
        Name: "last_sdxl_refiner_guidance_rescales"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_sdxl_refiner_guidance_scales"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_sdxl_refiner_inference_steps"
            Type: typing.Optional[collections.abc.Sequence[int]]
        Name: "last_sdxl_refiner_negative_aesthetic_scores"
            Type: typing.Optional[collections.abc.Sequence[float]]
        Name: "last_sdxl_refiner_negative_crops_coords_top_left"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_refiner_negative_original_sizes"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_refiner_negative_target_sizes"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_refiner_original_sizes"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_refiner_prompts"
            Type: typing.Optional[collections.abc.Sequence[dgenerate.prompt.Prompt]]
        Name: "last_sdxl_refiner_scheduler"
            Type: typing.Optional[str]
        Name: "last_sdxl_refiner_second_prompts"
            Type: typing.Optional[collections.abc.Sequence[dgenerate.prompt.Prompt]]
        Name: "last_sdxl_refiner_sequential_offload"
            Type: typing.Optional[bool]
        Name: "last_sdxl_refiner_target_sizes"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_sdxl_refiner_uri"
            Type: typing.Optional[str]
        Name: "last_sdxl_second_prompts"
            Type: typing.Optional[collections.abc.Sequence[dgenerate.prompt.Prompt]]
        Name: "last_sdxl_target_sizes"
            Type: typing.Optional[collections.abc.Sequence[tuple[int, int]]]
        Name: "last_second_text_encoder_uris"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "last_second_unet_uri"
            Type: typing.Optional[str]
        Name: "last_seed_image_processors"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "last_seeds"
            Type: collections.abc.Sequence[int]
        Name: "last_seeds_to_images"
            Type: <class 'bool'>
        Name: "last_subfolder"
            Type: typing.Optional[str]
        Name: "last_text_encoder_uris"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "last_textual_inversion_uris"
            Type: typing.Optional[collections.abc.Sequence[str]]
        Name: "last_unet_uri"
            Type: typing.Optional[str]
        Name: "last_upscaler_noise_levels"
            Type: typing.Optional[collections.abc.Sequence[int]]
        Name: "last_vae_slicing"
            Type: <class 'bool'>
        Name: "last_vae_tiling"
            Type: <class 'bool'>
        Name: "last_vae_uri"
            Type: typing.Optional[str]
        Name: "last_variant"
            Type: typing.Optional[str]
        Name: "path"
            Type: <class 'module'>
        Name: "saved_modules"
            Type: dict[str, dict[str, typing.Any]]

The following is output from ``\functions_help`` showing every implemented template function signature.

.. code-block:: text

    Available config template functions:

        abs(args, kwargs)
        align_size(size: str | tuple, align: int, format_size: bool = True) -> str | tuple
        all(args, kwargs)
        any(args, kwargs)
        ascii(args, kwargs)
        bin(args, kwargs)
        bool(args, kwargs)
        bytearray(args, kwargs)
        bytes(args, kwargs)
        callable(args, kwargs)
        chr(args, kwargs)
        complex(args, kwargs)
        cwd() -> str
        dict(args, kwargs)
        divmod(args, kwargs)
        download(url: str, output: str | None = None, overwrite: bool = False, text: bool = False) -> str
        enumerate(args, kwargs)
        filter(args, kwargs)
        first(iterable: collections.abc.Iterable[typing.Any]) -> typing.Any
        float(args, kwargs)
        format(args, kwargs)
        format_dtype(dtype: <enum 'DataType'>) -> str
        format_model_type(model_type: <enum 'ModelType'>) -> str
        format_prompt(prompts: dgenerate.prompt.Prompt | collections.abc.Iterable[dgenerate.prompt.Prompt]) -> str
        format_size(size: collections.abc.Iterable[int]) -> str
        frozenset(args, kwargs)
        gen_seeds(n: int) -> list[str]
        getattr(args, kwargs)
        hasattr(args, kwargs)
        hash(args, kwargs)
        have_feature(feature_name: str) -> bool
        hex(args, kwargs)
        int(args, kwargs)
        iter(args, kwargs)
        last(iterable: list | collections.abc.Iterable[typing.Any]) -> typing.Any
        len(args, kwargs)
        list(args, kwargs)
        map(args, kwargs)
        max(args, kwargs)
        min(args, kwargs)
        next(args, kwargs)
        object(args, kwargs)
        oct(args, kwargs)
        ord(args, kwargs)
        pow(args, kwargs)
        pow2_size(size: str | tuple, format_size: bool = True) -> str | tuple
        quote(strings: str | collections.abc.Iterable[typing.Any]) -> str
        range(args, kwargs)
        repr(args, kwargs)
        reversed(args, kwargs)
        round(args, kwargs)
        set(args, kwargs)
        size_is_aligned(size: str | tuple, align: int) -> bool
        size_is_pow2(size: str | tuple) -> bool
        slice(args, kwargs)
        sorted(args, kwargs)
        str(args, kwargs)
        sum(args, kwargs)
        tuple(args, kwargs)
        type(args, kwargs)
        unquote(strings: str | collections.abc.Iterable[typing.Any], expand: bool = False) -> list
        zip(args, kwargs)


Directives, and applying templating
-----------------------------------

You can see all available config directives with the command
``dgenerate --directives-help``, providing this option with a name, or multiple
names such as: ``dgenerate --directives-help save_modules use_modules`` will print
the documentation for the specified directives. The backslash may be omitted.
This option is also available as the config directive ``\directives_help``.

Example output:

.. code-block:: text

    Available config directives:

        "\cd"
        "\clear_control_net_cache"
        "\clear_model_cache"
        "\clear_modules"
        "\clear_pipeline_cache"
        "\clear_text_encoder_cache"
        "\clear_unet_cache"
        "\clear_vae_cache"
        "\cp"
        "\directives_help"
        "\download"
        "\echo"
        "\env"
        "\exec"
        "\exit"
        "\functions_help"
        "\gen_seeds"
        "\help"
        "\image_process"
        "\image_processor_help"
        "\import_plugins"
        "\ls"
        "\mkdir"
        "\mv"
        "\popd"
        "\print"
        "\prompt_weighter_help"
        "\pushd"
        "\pwd"
        "\rm"
        "\rmdir"
        "\save_modules"
        "\set"
        "\sete"
        "\setp"
        "\templates_help"
        "\unset"
        "\unset_env"
        "\use_modules"


Here are examples of other available directives such as ``\set``, ``\setp``, and
``\print`` as well as some basic Jinja2 templating usage. This example also covers
the usage and purpose of ``\save_modules`` for saving and reusing pipeline modules
such as VAEs etc. outside of relying on the caching system.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # You can define your own template variables with the \set directive
    # the \set directive does not do any shell args parsing on its value
    # operand, meaning the quotes will be in the string that is assigned
    # to the variable my_prompt

    \set my_prompt "an astronaut riding a horse; bad quality"

    # If your variable is long you can use continuation, note that
    # continuation replaces newlines and surrounding whitespace
    # with a single space

    \set my_prompt "my very very very very very very very \
                    very very very very very very very very \
                    long long long long long prompt"

    # You can print to the console with templating using the \print directive
    # for debugging purposes

    \print {{ my_prompt }}


    # The \setp directive can be used to define python literal template variables

    \setp my_list [1, 2, 3, 4]

    \print {{ my_list | join(' ') }}


    # Literals defined by \setp can reference other template variables by name.
    # the following creates a nested list

    \setp my_list [1, 2, my_list, 4]

    \print {{ my_list }}


    # \setp can evaluate template functions

    \setp directory_content glob.glob('*')

    \setp current_directory cwd()


    # the \gen_seeds directive can be used to store a list of
    # random seed integers into a template variable.
    # (they are strings for convenience)

    \gen_seeds my_seeds 10

    \print {{ my_seeds | join(' ') }}


    # An invocation sets various template variables related to its
    # execution once it is finished running

    stabilityai/stable-diffusion-2-1 --prompts {{ my_prompt }} --gen-seeds 5


    # Print a quoted filename of the last image produced by the last invocation
    # This could potentially be passed to --image-seeds of the next invocation
    # If you wanted to run another pass over the last image that was produced

    \print {{ quote(last(last_images)) }}

    # you can also get the first image easily with the function "first"

    \print {{ quote(first(last_images)) }}


    # if you want to append a mask image file name

    \print "{{ last(last_images) }};my-mask.png"


    # Print a list of properly quoted filenames produced by the last
    # invocation separated by spaces if there is multiple, this could
    # also be passed to --image-seeds

    # in the case that you have generated an animated output with frame
    # output enabled, this will contain paths to the frames

    \print {{ quote(last_images) }}


    # For loops are possible

    \print {% for image in last_images %}{{ quote(image) }} {% endfor %}


    # For loops are possible with normal continuation
    # when not using a heredoc template continuation (mentioned below),
    # such as when the loop occurs in the body of a directive or a
    # dgenerate invocation, however this sort of continuation usage will
    # replace newlines and whitespace with a single space.

    # IE this template will be: "{% for image in last_images %} {{ quote(image) }} {% endfor %}"

    \print {% for image in last_images %} \
            {{ quote(image) }} \
           {% endfor %}


    # Access to the last prompt is available in a parsed representation
    # via "last_prompt", which can be formatted properly for reuse
    # by using the function "format_prompt"

    stabilityai/stable-diffusion-2-1 --prompts {{ format_prompt(last(last_prompts)) }}

    # You can get only the positive or negative part if you want via the "positive"
    # and "negative" properties on a prompt object, these attributes are not
    # quoted so you need to quote them one way or another, preferably using the
    # dgenerate template function "quote" which will shell quote any special
    # characters that the argument parser is not going to understand

    stabilityai/stable-diffusion-2-1 --prompts {{ quote(last(last_prompts).positive) }}

    # "last_prompts" returns all the prompts used in the last invocation as a list
    # the "format_prompt" function can also work on a list

    stabilityai/stable-diffusion-2-1 --prompts "prompt 1" "prompt 2" "prompt 3"

    stabilityai/stable-diffusion-2-1 --prompts {{ format_prompt(last_prompts) }}


    # Execute additional config with full templating.
    # The sequence !END is interpreted as the end of a
    # template continuation, a template continuation is
    # started when a line begins with the character {
    # and is effectively a heredoc, in that all whitespace
    # within is preserved including newlines

    {% for image in last_images %}
        stabilityai/stable-diffusion-2-1 --image-seeds {{ quote(image) }} --prompts {{ my_prompt }}
    {% endfor %} !END


    # Multiple lines can be used with a template continuation
    # the inside of the template will be expanded to raw config
    # and then be ran, so make sure to use line continuations within
    # where they are necessary as you would do in the top level of
    # a config file. The whole of the template continuation is
    # processed by Jinja, from { to !END, so only one !END is
    # ever necessary after the external template
    # when dealing with nested templates

    {% for image in last_images %}
        stabilityai/stable-diffusion-2-1
        --image-seeds {{ quote(image) }}
        --prompts {{ my_prompt }}
    {% endfor %} !END


    # The above are both basically equivalent to this

    stabilityai/stable-diffusion-2-1 --image-seeds {{ quote(last_images) }} --prompts {{ my_prompt }}


    # You can save modules from the main pipeline used in the last invocation
    # for later reuse using the \save_modules directive, the first argument
    # is a variable name and the rest of the arguments are diffusers pipeline
    # module names to save to the variable name, this is an advanced usage
    # and requires some understanding of the diffusers library to utilize correctly

    stabilityai/stable-diffusion-2-1
    --variant fp16
    --dtype float16
    --prompts "an astronaut walking on the moon"
    --safety-checker
    --output-size 512


    \save_modules stage_1_modules feature_extractor safety_checker

    # that saves the feature_extractor module object in the pipeline above,
    # you can specify multiple module names to save if desired

    # Possible Module Names:

    # unet
    # vae
    # transformer
    # text_encoder
    # text_encoder_2
    # text_encoder_3
    # tokenizer
    # tokenizer_2
    # tokenizer_3
    # safety_checker
    # feature_extractor
    # controlnet
    # scheduler


    # To use the saved modules in the next invocation use  \use_modules

    \use_modules stage_1_modules

    # now the next invocation will use those modules instead of loading them from internal
    # in memory cache, disk, or huggingface

    stabilityai/stable-diffusion-x4-upscaler
    --variant fp16
    --dtype float16
    --model-type torch-upscaler-x4
    --prompts {{ format_prompt(last_prompts) }}
    --image-seeds {{ quote(last_images) }}
    --vae-tiling


    # you should clear out the saved modules if you no longer need them
    # and your config file is going to continue, or if the dgenerate
    # process is going to be kept alive for some reason such as in
    # some library usage scenarios, or perhaps if you are using it
    # like a server that reads from stdin :)

    \clear_modules stage_1_modules


Setting template variables, in depth
------------------------------------

The directives ``\set``, ``\sete``, and ``\setp`` can be used to set the value
of template variables within a configuration.  The directive ``\unset`` can be
used to undefine template variables.

All three of the assignment directives have unique behavior.

The ``\set`` directive sets a value with templating and environmental variable expansion applied to it,
and nothing else aside from the value being striped of leading and trailing whitespace. The value that is
set to the template variables is essentially the text that you supply as the value, as is. Or the text that
the templates or environment variables in the value expand to, unmodified or parsed in any way.

This is for assigning literal text values to a template variable.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    \set my_variable "I am an incomplete string and this is completely fine because I am a raw value

    # prints exactly what is above, including the quote at the beginning

    \print {{ my_variable }}

    # add a quote to the end of the string using templates

    \set my_variable {{ my_variable }}"

    # prints a fully quoted string

    \print {{ my_variable }}

    # indirect expansion is allowed

    \set var_name template_variable
    \env ENV_VAR_NAMED=env_var_named

    \set {{ var_name }} Hello!
    \set $ENV_VAR_NAMED Hi!

    # prints Hello!, Hi!

    \print {{ template_variable }}
    \print {{ env_var_named }}


The ``\sete`` directive can be used to assign the result of shell parsing and expansion to a
template variable, the value provided will be shell parsed into tokens as if it were a line of
dgenerate config. This is useful because you can use the config languages built in shell globbing
feature to assign template variables.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # lets pretend the directory "my_files" is full of files

    \sete my_variable --argument my_files/*

    # prints the python array ['--argument', 'my_files/file1', 'my_files/file2', ...]

    \print {{ my_variable }}

    # Templates and environmental variable references
    # are also parsed in the \sete directive, just as they are with \set

    \set directory my_files

    \sete my_variable --argument {{ directory }}/*

    # indirect expansion is allowed

    \set var_name template_variable
    \env ENV_VAR_NAMED=env_var_named

    \sete {{ var_name }} my_files/*
    \sete $ENV_VAR_NAMED my_files/*

    # both print ['my_files/file1', 'my_files/file2', ...]

    \print {{ template_variable }}
    \print {{ env_var_named }}


The ``\setp`` directive can be used to assign the result of evaluating a limited subset of python
expressions to a template variable.  This can be used to set a template variable to the result
of a mathematical expression, python literal value such as a list, dictionary, set, etc...
python comprehension, or python ternary statement.  In addition, all template functions
implemented by dgenerate are available for use in the evaluated expressions.


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    \setp my_variable 10*10

    # prints 100

    \print {{ my_variable }}

    # you can reference variables defined in the environment

    \setp my_variable [my_variable, my_variable*2]

    # prints [100, 200]

    \print {{ my_variable }}

    # all forms of python comprehensions are supported
    # such as list, dict, and set comprehensions

    \setp my_variable [i for i in range(0,5)]

    # prints [0, 1, 2, 3, 4]

    \print {{ my_variable }}

    # declare a literal string value

    \setp my_variable "my string value"

    # prints the string without quotes included, the string was parsed

    \print {{ my_variable }}

    # templates and environmental variable references
    # are also expanded in \setp values

    \setp my_variable [my_variable, "{{ my_variable }}"]

    # prints ["my string value", "my string value"]

    \print {{ my_variable }}

    # my_variable is a literal list so it can be
    # looped over with a jinja template continuation

    {% for value in my_variable %}
        \print {{ value }}
    {% endfor %} !END

    # indirect expansion is allowed

    \set var_name template_variable
    \env ENV_VAR_NAMED=env_var_named

    \setp {{ var_name }} "Hello!"
    \setp $ENV_VAR_NAMED [template_variable]

    # prints "Hello!", ["Hello!"]

    \print {{ template_variable }}
    \print {{ env_var_named }}


Setting environmental variables, in depth
-----------------------------------------

The directives ``\env`` and ``\unset_env`` can be used to
manipulate multiple environmental variables at once.

The directive ``\env`` can also be used without arguments to print out
the values of all environment variables that exist in your environment
for debugging purposes.

Indirect expansion is allowed just like with ``\set``, ``\sete``, and ``\setp``.


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    \env MY_ENV_VAR=1 MY_ENV_VAR2=1

    # prints 1 2

    \print $MY_ENV_VAR $MY_ENV_VAR2

    # indirect expansion is allowed

    \set name env_var_name
    \set value Hello!

    \set name_holder {{ name }}

    \env {{ name_holder }}={{ value }}

    # this treats the expansion of {{ name }} as a an environmental variable name

    \set output ${{ name }}

    # prints Hello!

    \print {{ output }}

    # unset an environmental variable, the names
    # undergo expansion, and are undefined in order

    \env NAME_HOLDER=MY_ENV_VAR2

    \unset_env MY_ENV_VAR $NAME_HOLDER {{ name }} NAME_HOLDER


    # prints every defined environmental variable
    # we have undefined everything that we defined
    # above so the names from this script will not
    # be present

    \env


Globbing and path manipulation
------------------------------

The entirety of pythons builtin ``glob`` and ``os.path`` module are also accessible during templating, you
can glob directories using functions from the glob module, you can also glob directory's using shell
globbing.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # globbing can be preformed via shell expansion or using
    # the glob module inside jinja templates

    # note that shell globbing and home directory expansion
    # does not occur inside quoted strings

    # \echo can be use to show the results of globbing that
    # occurs during shell expansion. \print does not preform shell
    # expansion nor does \set or \setp, all other directives do, as well
    # as dgenerate invocations

    # shell globs which produce 0 files are considered an error

    \echo ../media/*.png

    \echo ~

    # \sete can be used to set a template variable to the result
    # of one or more shell globs

    \sete myfiles ../media/*.png


    # with Jinja2:


    # The most basic usage is full expansion of every file

    \set myfiles {{ quote(glob.glob('../media/*.png')) }}

    \print {{ myfiles }}

    # If you have a LOT of files, you may want to
    # process them using an iterator like so

    {% for file in glob.iglob('../media/*.png') %}
        \print {{ quote(file) }}
    {% endfor %} !END

    # usage of os.path via path

    \print {{ path.abspath('.') }}

    # Simple inline usage

    stabilityai/stable-diffusion-2-1
    --variant fp16
    --dtype float16
    --prompts "In the style of picaso"
    --image-seeds {{ quote(glob.glob('../media/*.png')) }}
    --output-path {{ quote(path.join(path.abspath('.'), 'output')) }}

    # equivalent

    stabilityai/stable-diffusion-2-1
    --variant fp16
    --dtype float16
    --prompts "In the style of picaso"
    --image-seeds ../media/*.png
    --output-path ./output


The \\print and \\echo directive
--------------------------------

The ``\print`` and ``\echo`` directive can both be used to output text to the console.

The difference between the two directives is that ``\print`` only ever prints
the raw value with templating and environmental variable expansion applied,
similar to the behavior of ``\set``

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # the text after \print(space) will be printed verbatim

    \print I am a raw value, I have no ability to * glob

    # Print the PATH environmental variable

    \set header Path Elements:

    \print {{ header }} $PATH
    \print {{ header }} ${PATH}
    \print {{ header }} %PATH%

The ``\echo`` directive preforms shell expansion into tokens before printing, like ``\sete``,
This can be useful for debugging / displaying the results of a shell expansion.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # lets pretend "directory" is full of files

    # this prints: directory/file1 directory/file2 ...

    \echo directory/*

    # Templates and environmental variables are expanded

    # this prints: Files: directory/file1 directory/file2 ...

    \set header Files:

    \echo {{ header }} directory/*


The \\image_process directive
-----------------------------

The dgenerate sub-command ``image-process`` has a config directive implementation.


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # print the help message of --sub-command image-process, this does
    # not cause the config to exit

    \image_process --help

    \set myfiles {{ quote(glob.glob('my_images/*.png')) }}

    # this will create the directory "upscaled"
    # the files will be named "upscaled/FILENAME_processed_1.png" "upscaled/FILENAME_processed_2.png" ...

    \image_process {{ myfiles }} \
    --output upscaled/
    --processors upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth


    # the last_images template variable will be set, last_animations is also usable if
    # animations were written. In the case that you have generated an animated output with frame
    # output enabled, this will contain paths to the frames

    \print {{ quote(last_images) }}

The \\exec directive
--------------------

The ``\exec`` directive can be used to run native system commands and supports bash
pipe and file redirection syntax in a platform independent manner. All file
redirection operators supported by bash are supported. This can be useful
for running other image processing utilities as subprocesses from within a
config script.


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # run dgenerate as a subprocess, read a config
    # and send stdout and stderr to a file

    \exec dgenerate < my_config.dgen &> log.txt

    # chaining processes together with pipes is supported
    # this example emulates 'cat' on Windows using cmd

    \exec cmd /c "type my_config.dgen" | dgenerate &> log.txt

    # on a Unix platform you could simply use cat

    \exec cat my_config.dgen | dgenerate &> log.txt


The \\download directive
------------------------

Arbitrary files can be downloaded via the ``\download`` directive.

This directive can be used to download a file and assign its
downloaded path to a template variable.

Files can either be inserted into dgenerates web cache or
downloaded to a specific directory or absolute path.

This directive is designed with using cached files in mind,
so it will reuse existing files by default when downloading
to an explicit path.

See the directives help output for more details: ``\download --help``

If you plan to download many large models to the web cache in
this manner you may wish to adjust the global cache expiry time
so that they exist in the cache longer than the default of 12 hours.

You can see how to do this in the section `File Cache Control`_

This directive is primarily intended to download models and or other
binary file formats such as images and will raise an error if it encounters
a text mimetype. This  behavior can be overridden with the ``-t/--text`` argument.

Be weary that if you have a long-running loop in your config using
a top level jinja template, which refers to your template variable,
cache expiry may invalidate the file stored in your variable.

You can rectify this by putting the download directive inside of
your processing loop so that the file is simply re-downloaded if
it expires in the cache.

Or you may be better off using the ``download``
template function which provides this functionality
as a template function. See: `The download() template function`_


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # download a model into the web cache,
    # assign its path to the variable "path"

    \download path https://modelhost.com/somemodel.safetensors

    # download to the models folder in the current directory
    # the models folder will be created if it does not exist
    # if somemodel.safetensors already exists it will be reused
    # instead of being downloaded again

    \download path https://modelhost.com/somemodel.safetensors -o models/somemodel.safetensors

    # download into the folder without specifying a name
    # the name will be derived from the URL or content disposition
    # header from the http request, if you are not careful you may
    # end up with a file named in a way you were not expecting.
    # only use this if you know how the service you are downloading
    # from behaves in this regard

    \download path https://modelhost.com/somemodel.safetensors -o models/


    # download a model into the web cache an overwrite any cached model using -x

    \download path https://modelhost.com/somemodel.safetensors -x

    # Download to an explicit path without any cached file reuse
    # using the -x/--overwrite argument. In effect, always freshly
    # download the file

    \download path https://modelhost.com/somemodel.safetensors -o models/somemodel.safetensors -x

    \download path https://modelhost.com/somemodel.safetensors -o models/ -x


The download() template function
--------------------------------

The template function ``download`` is analogous to the ``\download`` directive

And can be used to download a file with the same behaviour and return its
path as a string, this may be easier to use inside of certain jinja flow
control constructs.


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    \set my_variable {{ download('https://modelhost.com/model.safetensors') }}

    \set my_variable {{ download('https://modelhost.com/model.safetensors', output='model.safetensors') }}

    \set my_variable {{ download('https://modelhost.com/model.safetensors', output='directory/') }}

    # you can also use any template function with \setp (python expression evaluation)

    \setp my_variable download('https://modelhost.com/model.safetensors')


The signature for this template function is: ``download(url: str, output: str | None = None, overwrite: bool = False, text: bool = False) -> str``


The \\exit directive
--------------------

You can exit a config early if need be using the ``\exit`` directive

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # exit the process with return code 0, which indicates success

    \print "success"
    \exit


An explicit return code can be provided as well


.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate 3.10.0

    # exit the process with return code 1, which indicates an error

    \print "some error occurred"
    \exit 1


Running configs from the command line
-------------------------------------

To utilize configuration files use the ``--file`` option,
or pipe them into the command, or use file redirection:


Use the ``--file`` option

.. code-block:: bash

    dgenerate --file my-config.dgen


Piping or redirection in Bash:

.. code-block:: bash

    # Pipe
    cat my-config.dgen | dgenerate

    # Redirection
    dgenerate < my-config.dgen


Redirection in Windows CMD:

.. code-block:: bash

    dgenerate < my-config.dgen


Piping Windows Powershell:

.. code-block:: powershell

    Get-Content my-config.dgen | dgenerate


Config argument injection
-------------------------

You can inject arguments into every dgenerate invocation of a batch processing
configuration by simply specifying them. The arguments will added to the end
of the argument specification of every call.

.. code-block:: bash

    # Pipe
    cat my-animations-config.dgen | dgenerate --frame-start 0 --frame-end 10

    # Redirection
    dgenerate --frame-start 0 --frame-end 10 < my-animations-config.dgen


On Windows CMD:

.. code-block:: bash

    dgenerate  --frame-start 0 --frame-end 10 < my-animations-config.dgen


On Windows Powershell:

.. code-block:: powershell

    Get-Content my-animations-config.dgen | dgenerate --frame-start 0 --frame-end 10


If you need arguments injected from the command line within the config for
some other purpose such as for using with the ``\image_process`` directive
which does not automatically recieve injected arguments, use the
``injected_args``  and related ``injected_*`` template variables.

.. code-block:: jinja

    # all injected args

    \print {{ quote(injected_args) }}

    # just the injected device

    \print {{ '--device '+injected_device if injected_device else '' }}

    # was -v/--verbose injected?

    \print {{ '-v' if injected_verbose else '' }}

    # plugin module paths injected with --plugin-modules

    \print {{ quote(injected_plugin_modules) if injected_plugin_modules else '' }}

Console UI
==========

.. image:: https://raw.githubusercontent.com/Teriks/dgenerate-readme-embeds/master/ui.gif
   :alt: console ui

You can launch a cross platform Tkinter GUI for interacting with a
live dgenerate process using ``dgenerate --console`` or via the optionally
installed desktop shortcut on Windows.

This provides a basic REPL for the dgenerate config language utilizing
a ``dgenerate --shell`` subprocess to act as the live interpreter, it
also features full context aware syntax highlighting for the dgenerate
config language.

It can be used to work with dgenerate without encountering the startup
overhead of loading large python modules for every command line invocation.

The GUI console supports command history via the up and down arrow keys as a
normal terminal would, optional multiline input for sending multiline commands / configuration
to the shell. And various editing niceties such as GUI file / directory path insertion,
the ability to insert templated command recipes for quickly getting started and getting results,
and a selection menu for inserting karras schedulers by name.

Also supported is the ability to view the latest image as it is produced by ``dgenerate`` or
``\image_process`` via an image pane or standalone window.

The console UI always starts in single line entry mode (terminal mode), multiline input mode
is activated via the insert key and indicated by the presence of line numbers, you must deactivate this mode
to submit commands via the enter key, however you can use the run button from the run menu (or ``Ctrl+Space``)
to run code in this mode. You cannot page through command history in this mode, and code will remain in the
console input pane upon running it making the UI function more like a code editor than a terminal.

The console can be opened with a file loaded in multiline input mode
by using the command: ``dgenerate --console filename.dgen``

``Ctrl+Q`` can be used in input pane for killing and then restarting the background interpreter process.

``Ctrl+F`` (find) and ``Ctrl+R`` (find/replace) is supported for both the input and output panes.

All common text editing features that you would expect to find in a basic text editor are present,
as well as python regex support for find / replace, with group substitution supporting the syntax
``\n`` or ``\{n}`` where ``n`` is the match group number.

Scroll back history in the output window is currently limited to 10000 lines however the console
app itself echos all ``stdout`` and ``stderr`` of the interpreter, so you can save all output to a log
file via file redirection if desired when launching the console from the terminal.

This can be configured by setting the environmental variable ``DGENERATE_CONSOLE_MAX_SCROLLBACK=10000``

Command history is currently limited to 500 commands, multiline commands are also
saved to command history.  The command history file is stored at ``-/.dgenerate_console_history``,
on Windows this equates to ``%USERPROFILE%\.dgenerate_console_history``

This can be configured by setting the environmental variable ``DGENERATE_CONSOLE_MAX_HISTORY=500``

Any UI settings that persist on startup are stored in ``-/.dgenerate_console_settings`` or
on Windows ``%USERPROFILE%\.dgenerate_console_settings``

Writing Plugins
===============

dgenerate has the capability of loading in additional functionality through the use of
the ``--plugin-modules`` option and ``\import_plugins`` config directive.

You simply specify one or more module directories on disk, paths to python files, or references
to modules installed in the python environment using the argument or import directive.

dgenerate supports implementing image processors, config directives, config template functions,
prompt weighters, and sub-commands through plugins.

~~~~

Image processor plugins
-----------------------

A code example as well as a usage example for image processor plugins can be found
in the `"writing_plugins/image_processor" <https://github.com/Teriks/dgenerate/tree/v3.10.0/examples/writing_plugins/image_processor>`_
folder of the examples folder.

The source code for the built in `canny <https://github.com/Teriks/dgenerate/blob/v3.10.0/dgenerate/imageprocessors/canny.py>`_ processor,
the `openpose <https://github.com/Teriks/dgenerate/blob/v3.10.0/dgenerate/imageprocessors/openpose.py>`_ processor, and the simple
`pillow image operations <https://github.com/Teriks/dgenerate/blob/v3.10.0/dgenerate/imageprocessors/imageops.py>`_ processors can also
be of reference as they are written as internal image processor plugins.

~~~~


Config directive and template function plugins
----------------------------------------------

An example for writing config directives can be found in the `"writing_plugins/config_directive" <https://github.com/Teriks/dgenerate/tree/v3.10.0/examples/writing_plugins/config_directive>`_  example folder.

Config template functions can also be implemented by plugins, see: `"writing_plugins/template_function" <https://github.com/Teriks/dgenerate/tree/v3.10.0/examples/writing_plugins/template_function>`_

Currently the only internal directive that is implemented as a plugin is the ``\image_process`` directive, who's source file 
`can be located here <https://github.com/Teriks/dgenerate/blob/v3.10.0/dgenerate/batchprocess/image_process_directive.py>`_.

The source file for the ``\image_process`` directive is terse as most of it is implemented as reusable code.

The behavior of ``\image_process`` which is also used for ``--sub-command image-process`` is
`is implemented here <https://github.com/Teriks/dgenerate/blob/v3.10.0/dgenerate/image_process>`_.

~~~~


Sub-command plugins
-------------------

Reference for writing sub-commands can be found in the `image-process <https://github.com/Teriks/dgenerate/blob/v3.10.0/dgenerate/subcommands/image_process.py>`_
sub-command implementation, and a plugin skeleton file for sub-commands can be found in the 
`"writing_plugins/sub_command" <https://github.com/Teriks/dgenerate/tree/v3.10.0/examples/writing_plugins/sub_command>`_ example folder.

~~~~


Prompt weighter plugins
-----------------------

Reference for writing prompt weighters can be found in the `CompelPromptWeighter <https://github.com/Teriks/dgenerate/blob/v3.10.0/dgenerate/promptweighters/compelpromptweighter.py>`_
and `SdEmbedPromptWeighter <https://github.com/Teriks/dgenerate/blob/v3.10.0/dgenerate/promptweighters/sdembedpromptweighter.py>`_ internal prompt weighter implementations.
 
A plugin skeleton file for prompt weighters can be found in the 
`"writing_plugins/prompt_weighter" <https://github.com/Teriks/dgenerate/tree/v3.10.0/examples/writing_plugins/prompt_weighter>`_
example folder.

~~~~


File Cache Control
==================

dgenerate will cache downloaded non hugging face models, downloaded ``--image-seeds`` files,
files downloaded by the ``\download`` directive, ``download`` template function, and downloaded
files used by image processors in the directory ``~/.cache/dgenerate/web``

On Windows this equates to: ``%USERPROFILE%\.cache\dgenerate\web``

You can control where these files are cached with the environmental variable ``DGENERATE_WEB_CACHE``.

Files are cleared from the web cache automatically after an expiry time upon running dgenerate or
when downloading additional files, the default value is after 12 hours.

This can be controlled with the environmental variable ``DGENERATE_WEB_CACHE_EXPIRY_DELTA``.

The value of ``DGENERATE_WEB_CACHE_EXPIRY_DELTA`` is that of the named arguments of pythons
`datetime.timedelta <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_ class
seperated by semicolons.

For example: ``DGENERATE_WEB_CACHE_EXPIRY_DELTA="days=5;hours=6"``

Specifying ``"forever"`` or an empty string will disable cache expiration for every downloaded file.

Files downloaded from huggingface by the diffusers/huggingface_hub library will be cached under
``~/.cache/huggingface/``, on Windows this equates to ``%USERPROFILE%\.cache\huggingface\``.

This is controlled by the environmental variable ``HF_HOME``

In order to specify that all large model files be stored in another location,
for example on another disk, simply set ``HF_HOME`` to a new path in your environment.

You can read more about environmental variables that affect huggingface libraries on this
`huggingface documentation page <https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables>`_.
