Overview
========

**dgenerate** is a command line tool for generating images and animation sequences using stable diffusion.

**dgenerate** can generate multiple images or animated outputs using multiple combinations of input parameters
for stable diffusion in batch, so that the differences in generated output can be compared / curated easily.

Animated output can be produced by processing every frame of a video, gif, webp through stable diffusion as
an image seed with a given prompt and generation parameters.

Video of infinite runtime can be processed without memory constraints.

GIF's and WebP can also be processed, with memory constraints.

This software requires an Nvidia GPU supporting CUDA 11.8+, CPU rendering is possible but extraordinarily slow.

----


* How to install
    * `Windows Install </#windows-install>`_
    * `Linux or WSL Install </#linux-or-wsl-install>`_

* Usage Examples
    * `Basic Usage </#basic-usage>`_
    * `Negative Prompt </#negative-prompt>`_
    * `Multiple Prompts </#multiple-prompts>`_
    * `Image Seed </#image-seed>`_
    * `Inpainting </#inpainting>`_
    * `Per Image Seed Resizing </#per-image-seed-resizing>`_
    * `Animated Output </#animated-output>`_
    * `Animation Slicing </#animation-slicing>`_
    * `Inpainting Animations </#inpainting-animations>`_
    * `Deterministic Output </#deterministic-output>`_
    * `Specifying a specific GPU for CUDA </#specifying-a-specific-gpu-for-cuda>`_
    * `Specifying a Scheduler (sampler) </#specifying-a-scheduler-sampler>`_
    * `Specifying a VAE </#specifying-a-vae>`_
    * `VAE Tiling and Slicing </#vae-tiling-and-slicing>`_
    * `Specifying a LoRA Finetune </#specifying-a-lora-finetune>`_
    * `Specifying Textual Inversions </#specifying-textual-inversions>`_
    * `Specifying Control Nets </#specifying-control-nets>`_
    * `Specifying Generation Batch Size </#specifying-generation-batch-size>`_
    * `Image Preprocessors </#image-preprocessors>`_
    * `Writing Plugins </#writing-plugins>`_
    * `Batch Processing From STDIN </#batch-processing-from-stdin>`_
    * `Batch Processing Argument Injection </#batch-processing-argument-injection>`_
    * `File Cache Control </#file-cache-control>`_

dgenerate help output
---------------------

.. code-block::

    usage: dgenerate [-h] [-v] [--version] [--plugin-modules PATH [PATH ...]] [--offline-mode]
                     [--templates-help] [--model-type MODEL_TYPE] [--revision BRANCH] [--variant VARIANT]
                     [--subfolder MODEL_SUBFOLDER] [--auth-token TOKEN] [--batch-size INTEGER]
                     [--batch-grid-size SIZE] [--vae VAE_URI] [--vae-tiling] [--vae-slicing] [--loras LORA_URI]
                     [--textual-inversions TEXTUAL_INVERSION_URI [TEXTUAL_INVERSION_URI ...]]
                     [--control-nets CONTROL_NET_URI [CONTROL_NET_URI ...]] [--scheduler SCHEDULER_NAME]
                     [--sdxl-refiner MODEL_URI] [--sdxl-refiner-scheduler SCHEDULER_NAME]
                     [--sdxl-second-prompts PROMPT [PROMPT ...]] [--sdxl-aesthetic-scores FLOAT [FLOAT ...]]
                     [--sdxl-crops-coords-top-left COORD [COORD ...]] [--sdxl-original-size SIZE [SIZE ...]]
                     [--sdxl-target-size SIZE [SIZE ...]] [--sdxl-negative-aesthetic-scores FLOAT [FLOAT ...]]
                     [--sdxl-negative-original-sizes SIZE [SIZE ...]]
                     [--sdxl-negative-target-sizes SIZE [SIZE ...]]
                     [--sdxl-negative-crops-coords-top-left COORD [COORD ...]]
                     [--sdxl-refiner-prompts PROMPT [PROMPT ...]]
                     [--sdxl-refiner-second-prompts PROMPT [PROMPT ...]]
                     [--sdxl-refiner-aesthetic-scores FLOAT [FLOAT ...]]
                     [--sdxl-refiner-crops-coords-top-left COORD [COORD ...]]
                     [--sdxl-refiner-original-sizes SIZE [SIZE ...]]
                     [--sdxl-refiner-target-sizes SIZE [SIZE ...]]
                     [--sdxl-refiner-negative-aesthetic-scores FLOAT [FLOAT ...]]
                     [--sdxl-refiner-negative-original-sizes SIZE [SIZE ...]]
                     [--sdxl-refiner-negative-target-sizes SIZE [SIZE ...]]
                     [--sdxl-refiner-negative-crops-coords-top-left COORD [COORD ...]] [-hnf FLOAT [FLOAT ...]]
                     [-ri INT [INT ...]] [-rg FLOAT [FLOAT ...]] [-rgr FLOAT [FLOAT ...]] [--safety-checker]
                     [-d DEVICE] [-t DTYPE] [-s SIZE] [-o PATH] [-op PREFIX] [-ox] [-oc] [-om]
                     [-p PROMPT [PROMPT ...]] [-se SEED [SEED ...] | -gse COUNT] [-af FORMAT] [-fs FRAME_NUMBER]
                     [-fe FRAME_NUMBER] [-is SEED [SEED ...]]
                     [--seed-image-preprocessors PREPROCESSOR [PREPROCESSOR ...]]
                     [--mask-image-preprocessors PREPROCESSOR [PREPROCESSOR ...]]
                     [--control-image-preprocessors PREPROCESSOR [PREPROCESSOR ...]]
                     [--image-preprocessor-help [PREPROCESSOR ...]] [-iss FLOAT [FLOAT ...] | -uns INTEGER
                     [INTEGER ...]] [-gs FLOAT [FLOAT ...]] [-igs FLOAT [FLOAT ...]] [-gr FLOAT [FLOAT ...]]
                     [-ifs INTEGER [INTEGER ...]]
                     model_path

    Stable diffusion batch image generation tool with support for video / gif / webp animation transcoding.

    positional arguments:
      model_path            huggingface model repository slug, huggingface blob link to a model file, path to
                            folder on disk, or path to a .pt, .pth, .bin, .ckpt, or .safetensors file.

    options:
      -h, --help            show this help message and exit
      -v, --verbose         Output information useful for debugging, such as pipeline call and model load
                            parameters.
      --version             Show dgenerate's version and exit
      --plugin-modules PATH [PATH ...]
                            Specify one or more plugin module folder paths (folder containing __init__.py) or
                            python .py file paths to load as plugins. Plugin modules can currently only
                            implement image preprocessors.
      --offline-mode        Whether dgenerate should try to download huggingface models that do not exist in the
                            disk cache, or only use what is available in the cache. Referencing a model on
                            huggingface that has not been cached because it was not previously downloaded will
                            result in a failure when using this option.
      --templates-help      Print a list of template variables available after a dgenerate invocation during
                            batch processing from STDIN.
      --model-type MODEL_TYPE
                            Use when loading different model types. Currently supported: torch, torch-pix2pix,
                            torch-sdxl, torch-sdxl-pix2pix, torch-upscaler-x2, or torch-upscaler-x4. (default:
                            torch)
      --revision BRANCH     The model revision to use when loading from a huggingface repository, (The git
                            branch / tag, default is "main")
      --variant VARIANT     If specified when loading from a huggingface repository or folder, load weights from
                            "variant" filename, e.g. "pytorch_model.<variant>.safetensors". Defaults to
                            automatic selection. This option is ignored if using flax.
      --subfolder MODEL_SUBFOLDER
                            Main model subfolder. If specified when loading from a huggingface repository or
                            folder, load weights from the specified subfolder.
      --auth-token TOKEN    Huggingface auth token. Required to download restricted repositories that have
                            access permissions granted to your huggingface account.
      --batch-size INTEGER  The number of image variations to produce per set of individual diffusion parameters
                            in one rendering step simultaneously on a single GPU. When using flax, batch size is
                            controlled by the environmental variable CUDA_VISIBLE_DEVICES which is a comma
                            seperated list of GPU device numbers (as listed by nvidia-smi). Usage of this
                            argument with --model-type flax* will cause an error, diffusion with flax will
                            generate an image on every GPU that is visible to CUDA and this is currently
                            unchangeable. When generating animations with a --batch-size greater than one, a
                            separate animation (with the filename suffix "animation_N") will be written to for
                            each image in the batch. If --batch-grid-size is specified when producing an
                            animation then the image grid is used for the output frames. During animation
                            rendering each image in the batch will still be written to the output directory
                            along side the produced animation as either suffixed files or image grids depending
                            on the options you choose. (Torch Default: 1)
      --batch-grid-size SIZE
                            Produce a single image containing a grid of images with the number of COLUMNSxROWS
                            given to this argument when --batch-size is greater than 1, or when using flax with
                            multiple GPUs visible (via the environmental variable CUDA_VISIBLE_DEVICES). If not
                            specified with a --batch-size greater than 1, images will be written individually
                            with an image number suffix (image_N) in the filename signifying which image in the
                            batch they are.
      --vae VAE_URI         Specify a VAE using a URI. When using torch models the URI syntax is:
                            "AutoEncoderClass;model=(huggingface repository slug/blob link or file/folder
                            path)". Examples: "AutoencoderKL;model=vae.pt",
                            "AsymmetricAutoencoderKL;model=huggingface/vae",
                            "AutoencoderTiny;model=huggingface/vae". When using a Flax model, there is currently
                            only one available encoder class: "FlaxAutoencoderKL;model=huggingface/vae". The
                            AutoencoderKL encoder class accepts huggingface repository slugs/blob links, .pt,
                            .pth, .bin, .ckpt, and .safetensors files. Other encoders can only accept
                            huggingface repository slugs/blob links, or a path to a folder on disk with the
                            model configuration and model file(s). Aside from the "model" argument, there are
                            four other optional arguments that can be specified, these include "revision",
                            "variant", "subfolder", "dtype". They can be specified as so in any order, they are
                            not positional: "AutoencoderKL;model=huggingface/vae;revision=main;variant=fp16;subf
                            older=sub_folder;dtype=float16". The "revision" argument specifies the model
                            revision to use for the VAE when loading from huggingface repository or blob link,
                            (The git branch / tag, default is "main"). The "variant" argument specifies the VAE
                            model variant, if "variant" is specified when loading from a huggingface repository
                            or folder, weights will be loaded from "variant" filename, e.g.
                            "pytorch_model.<variant>.safetensors. "variant" defaults to automatic selection and
                            is ignored if using flax. "variant" in the case of --vae does not default to the
                            value of --variant to prevent failures during common use cases. The "subfolder"
                            argument specifies the VAE model subfolder, if specified when loading from a
                            huggingface repository or folder, weights from the specified subfolder. The "dtype"
                            argument specifies the VAE model precision, it defaults to the value of -t/--dtype
                            and should be one of: auto, float16, or float32. If you wish to load a weights file
                            directly from disk, the simplest way is: --vae "AutoencoderKL;my_vae.safetensors",
                            or with a dtype "AutoencoderKL;my_vae.safetensors;dtype=float16", all other loading
                            arguments are unused in this case and may produce an error message if used. If you
                            wish to load a specific weight file from a huggingface repository, use the blob link
                            loading syntax: --vae "AutoencoderKL;https://huggingface.co/UserName/repository-
                            name/blob/main/vae_model.safetensors", the revision argument may be used with this
                            syntax.
      --vae-tiling          Enable VAE tiling (torch models only). Assists in the generation of large images
                            with lower memory overhead. The VAE will split the input tensor into tiles to
                            compute decoding and encoding in several steps. This is useful for saving a large
                            amount of memory and to allow processing larger images. Note that if you are using
                            --control-nets you may still run into memory issues generating large images, or with
                            --batch-size greater than 1.
      --vae-slicing         Enable VAE slicing (torch* models only). Assists in the generation of large images
                            with lower memory overhead. The VAE will split the input tensor in slices to compute
                            decoding in several steps. This is useful to save some memory, especially when
                            --batch-size is greater than 1. Note that if you are using --control-nets you may
                            still run into memory issues generating large images, or with --batch-size greater
                            than 1.
      --loras LORA_URI, --lora LORA_URI
                            Specify a LoRA model using a URI (flax not supported). This should be a huggingface
                            repository slug, path to model file on disk (for example, a .pt, .pth, .bin, .ckpt,
                            or .safetensors file), or model folder containing model files. huggingface blob
                            links are not supported, see "subfolder" and "weight-name" below instead. Optional
                            arguments can be provided after the LoRA model specification, these include:
                            "scale", "revision", "subfolder", and "weight-name". They can be specified as so in
                            any order, they are not positional:
                            "huggingface/lora;scale=1.0;revision=main;subfolder=repo_subfolder;weight-
                            name=lora.safetensors". The "scale" argument indicates the scale factor of the LoRA.
                            The "revision" argument specifies the model revision to use for the VAE when loading
                            from huggingface repository, (The git branch / tag, default is "main"). The
                            "subfolder" argument specifies the VAE model subfolder, if specified when loading
                            from a huggingface repository or folder, weights from the specified subfolder. The
                            "weight-name" argument indicates the name of the weights file to be loaded when
                            loading from a huggingface repository or folder on disk. If you wish to load a
                            weights file directly from disk, the simplest way is: --lora "my_lora.safetensors",
                            or with a scale "my_lora.safetensors;scale=1.0", all other loading arguments are
                            unused in this case and may produce an error message if used.
      --textual-inversions TEXTUAL_INVERSION_URI [TEXTUAL_INVERSION_URI ...]
                            Specify one or more Textual Inversion models using URIs (flax and SDXL not
                            supported). This should be a huggingface repository slug, path to model file on disk
                            (for example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model folder
                            containing model files. huggingface blob links are not supported, see "subfolder"
                            and "weight-name" below instead. Optional arguments can be provided after the
                            Textual Inversion model specification, these include: "revision", "subfolder", and
                            "weight-name". They can be specified as so in any order, they are not positional:
                            "huggingface/ti_model;revision=main;subfolder=repo_subfolder;weight-
                            name=lora.safetensors". The "revision" argument specifies the model revision to use
                            for the Textual Inversion model when loading from huggingface repository, (The git
                            branch / tag, default is "main"). The "subfolder" argument specifies the Textual
                            Inversion model subfolder, if specified when loading from a huggingface repository
                            or folder, weights from the specified subfolder. The "weight-name" argument
                            indicates the name of the weights file to be loaded when loading from a huggingface
                            repository or folder on disk. If you wish to load a weights file directly from disk,
                            the simplest way is: --textual-inversions "my_ti_model.safetensors", all other
                            loading arguments are unused in this case and may produce an error message if used.
      --control-nets CONTROL_NET_URI [CONTROL_NET_URI ...]
                            Specify one or more ControlNet models using URIs. This should be a huggingface
                            repository slug / blob link, path to model file on disk (for example, a .pt, .pth,
                            .bin, .ckpt, or .safetensors file), or model folder containing model files. Optional
                            arguments can be provided after the ControlNet model specification, for torch these
                            include: "scale", "start", "end", "revision", "variant", "subfolder", and "dtype".
                            For flax: "scale", "revision", "subfolder", "dtype", "from_torch" (bool) They can be
                            specified as so in any order, they are not positional:cd "huggingface/controlnet;sca
                            le=1.0;start=0.0;end=1.0;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=f
                            loat16". The "scale" argument specifies the scaling factor applied to the ControlNet
                            model, the default value is 1.0. The "start" (only for --model-type "torch*")
                            argument specifies at what fraction of the total inference steps to begin applying
                            the ControlNet, defaults to 0.0, IE: the very beginning. The "end" (only for
                            --model-type "torch*") argument specifies at what fraction of the total inference
                            steps to stop applying the ControlNet, defaults to 1.0, IE: the very end. The
                            "revision" argument specifies the model revision to use for the ControlNet model
                            when loading from huggingface repository, (The git branch / tag, default is "main").
                            The "variant" (only for --model-type "torch*") argument specifies the ControlNet
                            model variant, if "variant" is specified when loading from a huggingface repository
                            or folder, weights will be loaded from "variant" filename, e.g.
                            "pytorch_model.<variant>.safetensors. "variant" defaults to automatic selection and
                            is ignored if using flax. "variant" in the case of --control-nets does not default
                            to the value of --variant to prevent failures during common use cases. The
                            "subfolder" argument specifies the ControlNet model subfolder, if specified when
                            loading from a huggingface repository or folder, weights from the specified
                            subfolder. The "dtype" argument specifies the ControlNet model precision, it
                            defaults to the value of -t/--dtype and should be one of: auto, float16, or float32.
                            The "from_torch" (only for --model-type flax) this argument specifies that the
                            ControlNet is to be loaded and converted from a huggingface repository or file that
                            is designed for pytorch. (Defaults to false) If you wish to load a weights file
                            directly from disk, the simplest way is: --control-nets "my_controlnet.safetensors"
                            or --control-nets "my_controlnet.safetensors;scale=1.0;dtype=float16", all other
                            loading arguments aside from "scale" and "dtype" are unused in this case and may
                            produce an error message if used ("from_torch" is available when using flax). If you
                            wish to load a specific weight file from a huggingface repository, use the blob link
                            loading syntax: --control-nets "https://huggingface.co/UserName/repository-
                            name/blob/main/controlnet.safetensors", the revision argument may be used with this
                            syntax.
      --scheduler SCHEDULER_NAME
                            Specify a scheduler (sampler) by name. Passing "help" to this argument will print
                            the compatible schedulers for a model without generating any images. Torch
                            schedulers: (DDIMScheduler, DDPMScheduler, PNDMScheduler, LMSDiscreteScheduler,
                            EulerDiscreteScheduler, HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                            DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler,
                            KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler, UniPCMultistepScheduler,
                            DPMSolverSDEScheduler).
      --sdxl-refiner MODEL_URI
                            Stable Diffusion XL (torch-sdxl) refiner model path using a URI. This should be a
                            huggingface repository slug / blob link, path to model file on disk (for example, a
                            .pt, .pth, .bin, .ckpt, or .safetensors file), or model folder containing model
                            files. Optional arguments can be provided after the SDXL refiner model
                            specification, these include: "revision", "variant", "subfolder", and "dtype". They
                            can be specified as so in any order, they are not positional: "huggingface/refiner_m
                            odel_xl;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=float16". The
                            "revision" argument specifies the model revision to use for the Textual Inversion
                            model when loading from huggingface repository, (The git branch / tag, default is
                            "main"). The "variant" argument specifies the SDXL refiner model variant and
                            defaults to the value of --variant, when "variant" is specified when loading from a
                            huggingface repository or folder, weights will be loaded from "variant" filename,
                            e.g. "pytorch_model.<variant>.safetensors. "variant" defaults to automatic
                            selection. The "subfolder" argument specifies the SDXL refiner model subfolder, if
                            specified when loading from a huggingface repository or folder, weights from the
                            specified subfolder. The "dtype" argument specifies the SDXL refiner model
                            precision, it defaults to the value of -t/--dtype and should be one of: auto,
                            float16, or float32. If you wish to load a weights file directly from disk, the
                            simplest way is: --sdxl-refiner "my_sdxl_refiner.safetensors" or --sdxl-refiner
                            "my_sdxl_refiner.safetensors;dtype=float16", all other loading arguments aside from
                            "dtype" are unused in this case and may produce an error message if used. If you
                            wish to load a specific weight file from a huggingface repository, use the blob link
                            loading syntax: --sdxl-refiner "https://huggingface.co/UserName/repository-
                            name/blob/main/refiner_model.safetensors", the revision argument may be used with
                            this syntax.
      --sdxl-refiner-scheduler SCHEDULER_NAME
                            Specify a scheduler (sampler) by name for the SDXL refiner pass. Operates the
                            exactsame way as --scheduler including the "help" option. Defaults to the value of
                            --scheduler.
      --sdxl-second-prompts PROMPT [PROMPT ...]
                            List of secondary prompts to try using SDXL's secondary text encoder. By default the
                            model is passed the primary prompt for this value, this option allows you to choose
                            a different prompt. The negative prompt component can be specified with the same
                            syntax as --prompts
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
                            parameters in the format (WIDTHxHEIGHT). If not the same as --sdxl-target-size the
                            image will appear to be down or upsampled. --sdxl-original-size defaults to
                            --output-size if not specified. Part of SDXL's micro-conditioning as explained in
                            section 2.2 of [https://huggingface.co/papers/2307.01952]
      --sdxl-target-size SIZE [SIZE ...], --sdxl-target-sizes SIZE [SIZE ...]
                            One or more Stable Diffusion XL (torch-sdxl) "target-size" micro-conditioning
                            parameters in the format (WIDTHxHEIGHT). For most cases, --sdxl-target-size should
                            be set to the desired height and width of the generated image. If not specified it
                            will default to --output-size. Part of SDXL's micro-conditioning as explained in
                            section 2.2 of [https://huggingface.co/papers/2307.01952]
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
                            target image resolution. It should be as same as the "target_size" for most cases.
                            Part of SDXL's micro-conditioning as explained in section 2.2 of
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
                            List of prompts to try with the SDXL refiner model, by default the refiner model
                            gets the primary prompt, this argument overrides that with a prompt of your
                            choosing. The negative prompt component can be specified with the same syntax as
                            --prompts
      --sdxl-refiner-second-prompts PROMPT [PROMPT ...]
                            List of prompts to try with the SDXL refiner models secondary text encoder, by
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
                            High noise fraction for Stable Diffusion XL (torch-sdxl), this fraction of inference
                            steps will be processed by the base model, while the rest will be processed by the
                            refiner model. Multiple values to this argument will result in additional generation
                            steps for each value. In certain situations when the mixture of denoisers algorithm
                            is not supported, such as when using --control-nets and inpainting with SDXL, the
                            inverse proportion of this value IE: (1.0 - high-noise-fraction) becomes the
                            --image-seed-strength input to the SDXL refiner. (default: [0.8])
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
      --safety-checker      Enable safety checker loading, this is off by default. When turned on images with
                            NSFW content detected may result in solid black output. Some pretrained models have
                            settings indicating a safety checker is not to be loaded, in that case this option
                            has no effect.
      -d DEVICE, --device DEVICE
                            cuda / cpu. (default: cuda). Use: cuda:0, cuda:1, cuda:2, etc. to specify a specific
                            GPU.
      -t DTYPE, --dtype DTYPE
                            Model precision: auto, float16, or float32. (default: auto)
      -s SIZE, --output-size SIZE
                            Image output size. If an image seed is used it will be resized to this dimension
                            with aspect ratio maintained, width will be fixed and a new height will be
                            calculated. If only one integer value is provided, that is the value for both
                            dimensions. X/Y dimension values should be separated by "x". (default: 512x512 when
                            no image seeds are specified)
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
                            dgenerate STDIN, for example "dgenerate < config.txt". These files will be written
                            to --output-directory and are affected by --output-prefix and --output-overwrite as
                            well. The files will be named after their corresponding image or animation file.
                            Configuration files produced for animation frame images will utilize --frame-start
                            and --frame-end to specify the frame number.
      -om, --output-metadata
                            Write the information produced by --output-configs to the PNG metadata of each
                            image. Metadata will not be written to animated files (yet). The data is written to
                            a PNG metadata property named DgenerateConfig and can be read using ImageMagick like
                            so: "magick identify -format "%[Property:DgenerateConfig] generated_file.png".
      -p PROMPT [PROMPT ...], --prompts PROMPT [PROMPT ...]
                            List of prompts to try, an image group is generated for each prompt, prompt data is
                            split by ; (semi-colon). The first value is the positive text influence, things you
                            want to see. The Second value is negative influence IE. things you don't want to
                            see. Example: --prompts "shrek flying a tesla over detroit; clouds, rain, missiles".
                            (default: [(empty string)])
      -se SEED [SEED ...], --seeds SEED [SEED ...]
                            List of seeds to try, define fixed seeds to achieve deterministic output. This
                            argument may not be used when --gse/--gen-seeds is used. (default: [randint(0,
                            99999999999999)])
      -gse COUNT, --gen-seeds COUNT
                            Auto generate N random seeds to try. This argument may not be used when -se/--seeds
                            is used.
      -af FORMAT, --animation-format FORMAT
                            Output format when generating an animation from an input video / gif / webp etc.
                            Value must be one of: mp4, gif, or webp. (default: mp4)
      -fs FRAME_NUMBER, --frame-start FRAME_NUMBER
                            Starting frame slice point for animated files, the specified frame will be included.
      -fe FRAME_NUMBER, --frame-end FRAME_NUMBER
                            Ending frame slice point for animated files, the specified frame will be included.
      -is SEED [SEED ...], --image-seeds SEED [SEED ...]
                            List of image seeds to try when processing image seeds, these may be URLs or file
                            paths. Videos / GIFs / WEBP files will result in frames being rendered as well as an
                            animated output file being generated if more than one frame is available in the
                            input file. Inpainting for static images can be achieved by specifying a black and
                            white mask image in each image seed string using a semicolon as the separating
                            character, like so: "my-seed-image.png;my-image-mask.png", white areas of the mask
                            indicate where generated content is to be placed in your seed image. Output
                            dimensions specific to the image seed can be specified by placing the dimension at
                            the end of the string following a semicolon like so: "my-seed-image.png;512x512" or
                            "my-seed-image.png;my-image-mask.png;512x512". Inpainting masks can be downloaded
                            for you from a URL or be a path to a file on disk. When using --control-nets, a
                            singular image specification is interpreted as the control guidance image. Using
                            --control-nets with img2img or inpainting can be accomplished with the syntax: "my-
                            seed-image.png;mask=my-image-mask.png;control=my-control-image.png;resize=512x512".
                            The "mask" and "resize" arguments are optional when using --control-nets. Videos,
                            GIFs, and WEBP are also supported as inputs when using --control-nets, even for the
                            "control" argument. --image-seeds is capable of reading from 3 animated files at
                            once or any combination of animated files and images, the animated file with the
                            least amount of frames dictates how many frames are generated.
      --seed-image-preprocessors PREPROCESSOR [PREPROCESSOR ...]
                            Specify one or more image preprocessor actions to preform on the primary image
                            specified by --image-seeds. For example: --seed-image-preprocessors "flip" "mirror"
                            "grayscale". To obtain more information about what image preprocessors are available
                            and how to use them, see: --image-preprocessor-help.
      --mask-image-preprocessors PREPROCESSOR [PREPROCESSOR ...]
                            Specify one or more image preprocessor actions to preform on the inpaint mask image
                            specified by --image-seeds. For example: --mask-image-preprocessors "invert". To
                            obtain more information about what image preprocessors are available and how to use
                            them, see: --image-preprocessor-help.
      --control-image-preprocessors PREPROCESSOR [PREPROCESSOR ...]
                            Specify one or more image preprocessor actions to preform on the control image
                            specified by --image-seeds. For example: --control-image-preprocessors
                            "canny;lower=50;upper=100". This option is ment to be used in combination with
                            --control-nets. To obtain more information about what image preprocessors are
                            available and how to use them, see: --image-preprocessor-help.
      --image-preprocessor-help [PREPROCESSOR ...]
                            Use this option alone (or with --plugin-modules) and no model specification in order
                            to list available image preprocessor module names. Specifying one or more module
                            names after this option will cause usage documentation for the specified modules to
                            be printed.
      -iss FLOAT [FLOAT ...], --image-seed-strengths FLOAT [FLOAT ...]
                            List of image seed strengths to try. Closer to 0 means high usage of the seed image
                            (less noise convolution), 1 effectively means no usage (high noise convolution). Low
                            values will produce something closer or more relevant to the input image, high
                            values will give the AI more creative freedom. (default: [0.8])
      -uns INTEGER [INTEGER ...], --upscaler-noise-levels INTEGER [INTEGER ...]
                            List of upscaler noise levels to try when using the super resolution upscaler
                            (torch-upscaler-x4). These values will be ignored when using (torch-upscaler-x2).
                            The higher this value the more noise is added to the image before upscaling (similar
                            to --image-seed-strength). (default: [20])
      -gs FLOAT [FLOAT ...], --guidance-scales FLOAT [FLOAT ...]
                            List of guidance scales to try. Guidance scale effects how much your text prompt is
                            considered. Low values draw more data from images unrelated to text prompt.
                            (default: [5])
      -igs FLOAT [FLOAT ...], --image-guidance-scales FLOAT [FLOAT ...]
                            Push the generated image towards the inital image when using --model-type *-pix2pix
                            models. Use in conjunction with --image-seeds, inpainting (masks) and --control-nets
                            are not supported. Image guidance scale is enabled by setting image-guidance-scale >
                            1. Higher image guidance scale encourages generated images that are closely linked
                            to the source image, usually at the expense of lower image quality. Requires a value
                            of at least 1. (default: [1.5])
      -gr FLOAT [FLOAT ...], --guidance-rescales FLOAT [FLOAT ...]
                            List of guidance rescale factors to try. Proposed by [Common Diffusion Noise
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
                            Lists of inference steps values to try. The amount of inference (de-noising) steps
                            effects image clarity to a degree, higher values bring the image closer to what the
                            AI is targeting for the content of the image. Values between 30-40 produce good
                            results, higher values may improve image quality and or change image content.
                            (default: [30])



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

https://www.python.org/ftp/python/3.11.3/python-3.11.3-amd64.exe

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

    pipx install git+https://github.com/Teriks/dgenerate.git ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/"

    # If you want a specific version

    pipx install git+https://github.com/Teriks/dgenerate.git@v2.0.0 ^
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/"


It is recommended to install dgenerate with pipx if you are just intending
to use it, if you want to develop you can install it from a cloned repository
like this:

.. code-block:: bash

    # in the top of the repo make
    # an environment and activate it

    python -m venv venv
    venv\Scripts\activate

    # Install with pip into the environment

    pip install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu118/


Run **dgenerate** to generate images:

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

First update your system and install build-essential and native dependencies

.. code-block:: bash

    sudo apt update && sudo apt upgrade
    sudo apt install build-essential

    # Install libgl1 dependency for OpenCV.
    # Needed on WSL, not sure about normal Ubuntu/Debian?
    # I don't have a linux machine with a GPU :)
    # You'll probably need to install this
    # if your install is headless, you will
    # know because a relevant exception will
    # be produced when running dgenerate if you need it

    sudo apt install libgl1


Install CUDA Toolkit 12.*: https://developer.nvidia.com/cuda-downloads

I recommend using the runfile option:

.. code-block:: bash

    # CUDA Toolkit 12.2.1 For Ubuntu / Debian / WSL

    wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda_12.2.1_535.86.10_linux.run
    sudo sh cuda_12.2.1_535.86.10_linux.run

Do not attempt to install a driver from the prompts if using WSL.

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

    pipx install git+https://github.com/Teriks/dgenerate.git \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/"

    # With flax/jax support

    pipx install "dgenerate[flax] @ git+https://github.com/Teriks/dgenerate.git" \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/ \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

    # If you want a specific version

    pipx install git+https://github.com/Teriks/dgenerate.git@v2.0.0 \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/"

    # Specific version with flax/jax support

    pipx install "dgenerate[flax] @ git+https://github.com/Teriks/dgenerate.git@v2.0.0" \
    --pip-args "--extra-index-url https://download.pytorch.org/whl/cu118/ \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"


It is recommended to install dgenerate with pipx if you are just intending
to use it, if you want to develop you can install it from a cloned repository
like this:

.. code-block:: bash

    # in the top of the repo make
    # an environment and activate it

    python3 -m venv venv
    source venv/bin/activate

    # Install with pip into the environment

    pip3 install --editable .[dev] --extra-index-url https://download.pytorch.org/whl/cu118/

    # With flax if you want

    pip3 install --editable .[dev,flax] --extra-index-url https://download.pytorch.org/whl/cu118/ \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


Run **dgenerate** to generate images:

.. code-block:: bash

    # Images are output to the "output" folder
    # in the current working directory by default

    dgenerate --help

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --output-path output \
    --inference-steps 40 \
    --guidance-scales 10

Basic Usage
===========

The example below attempts to generate an astronaut riding a horse using 5 different
random seeds, 3 different inference steps values, and 3 different guidance scale values.

It utilizes the "stabilityai/stable-diffusion-2-1" model repo on `Hugging Face <https://huggingface.co/stabilityai/stable-diffusion-2-1>`_.

45 uniquely named images will be generated (5 x 3 x 3)

Also Adjust output size to 512x512 and output generated images to the "astronaut" folder in the current working directory.

When ``--output-path`` is not specified, the default output location is the "output" folder in the current working directory,
if the path that is specified does not exist then it will be created.

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

Refiner models can be specified, fp16 model variant and a datatype of float16 is
recommended to prevent out of memory conditions on the average GPU :)

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --sdxl-high-noise-fractions 0.6 0.7 0.8 \
    --gen-seeds 5 \
    --inference-steps 50 \
    --guidance-scale 12 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --prompts "real photo of an astronaut riding a horse on the moon" \
    --variant fp16 --dtype float16 \
    --output-size 1024
    
    
Negative Prompt
===============

In order to specify a negative prompt, each prompt argument is split
into two parts separated by ``;``

The prompt text occuring after ``;`` is the negative influence prompt.

To attempt to avoid rendering of a saddle on the horse being ridden, you
could for example add the negative prompt "saddle" or "wearing a saddle"
or "horse wearing a saddle" etc.


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
prompts and five random seeds (2x5)
 
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

Use a photo of Buzz Aldrin on the moon to generate a photo of an astronaut standing on mars using Img2Img,
this uses an image seed downloaded from wikipedia.

Disk file paths may also be used for image seeds, multiple image seeds may be provided, images will be
generated from each image seed individually.

Generate this image using 5 different seeds, 3 different inference-step values, 3 different
guidance-scale values as above.

In addition this image will be generated using 3 different image seed strengths.

Adjust output size to 512x512 and output generated images to 'astronaut' folder, if the image seed
is not a 1:1 aspect ratio the width will be fixed to the requested width and the height of the output image
calculated to maintain aspect ratio.

If you do not adjust the output size of the generated image, the size of the input image seed will be used.

135 uniquely named images will be generated (5x3x3x3)

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut walking on mars" \
    --image-seeds https://upload.wikimedia.org/wikipedia/commons/9/98/Aldrin_Apollo_11_original.jpg \
    --image-seed-strengths 0.2 0.5 0.8 \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 30 40 50 \
    --guidance-scales 5 7 10 \
    --output-size 512x512


The syntax "my-image-seed.png;control=my-control-image.png" can be used with ``--control-nets`` to specify
img2img mode with a Control Net, see: `Specifying Control Nets </#specifying-control-nets>`_ for more information.


Inpainting
==========

Inpainting on an image can be preformed by providing a mask image with your image seed. This mask should be a black and white image
of identical size to your image seed.  White areas of the mask image will be used to tell the AI what areas of the seed image should be filled
in with generated content.

.. _Inpainting Animations: /#inpainting-animations

For using inpainting on animated image seeds, jump to: `Inpainting Animations`_

In order to use inpainting, specify your image seed like so:
``--image-seeds "my-image-seed.png;my-mask-image.png"`` or ``--image-seeds "my-image-seed.png;mask=my-mask-image.png"``

The format is your image seed and mask image seperated by ``;``, optionally mask can be named argument.
The alternate syntax is for disambiguation when using `Control Nets </#specifying-control-nets>`_.

Mask images can be downloaded from URL's just like image seeds, however for this example the syntax specifies a file on disk for brevity.

`my-image-seed.png <https://raw.githubusercontent.com/Teriks/dgenerate/master/examples/media/dog-on-bench.png>`_

`my-mask-image.png <https://raw.githubusercontent.com/Teriks/dgenerate/master/examples/media/dog-on-bench-mask.png>`_

The command below generates a cat sitting on a bench with the images from the links above, the mask image masks out
areas over the dog in the original image, causing the dog to be replaced with an AI generated cat.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --image-seeds "my-image-seed.png;my-mask-image.png" \
    --prompts "Face of a yellow cat, high resolution, sitting on a park bench" \
    --image-seed-strengths 0.8 \
    --guidance-scale 10 \
    --inference-steps 100


Per Image Seed Resizing
=======================

If you want to specify multiple image seeds that will have different output sizes irrespective
of their input size or a globally defined output size defined with ``--output-size``,
You can specify their output size individually at the end of each provided image seed.

This will work when using a mask image for inpainting as well, including when using animated inputs.

The syntax is: ``--image-seeds "my-image-seed.png;512x512"`` or ``--image-seeds "my-image-seed.png;my-mask-image.png;512x512"``
or ``--image-seeds "my-image-seed.png;mask=my-mask-image.png;resize=512x512"``

The alternate syntax with named arguments is for disambiguation when using `Control Nets </#specifying-control-nets>`_.

When one dimension is specified, that dimension is the width, and the height is
calculated from the aspect ratio of the input image.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --image-seeds "my-image-seed.png;1024" "my-image-seed.png;my-mask-image.png;512x512" \
    --prompts "Face of a yellow cat, high resolution, sitting on a park bench" \
    --image-seed-strengths 0.8 \
    --guidance-scale 10 \
    --inference-steps 100


Animated Output
===============

**dgenerate** supports many video formats through the use of PyAV, as well as GIF & WebP.

When an animated image seed is given, animated output will be produced in the format of your choosing.

In addition, every frame will be written to the output folder as a uniquely named image.

Use a GIF of a man riding a horse to create an animation of an astronaut riding a horse.

Output to an MP4.  See ``--help`` for information about formats supported by ``--animation-format``

If the animation is not 1:1 aspect ratio, the width will be fixed to the width of the
requested output size, and the height calculated to match the aspect ratio of the animation.

If you do not set an output size, the size of the input animation will be used.

.. code-block:: bash

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

For more information about this see: `Specifying Control Nets </#specifying-control-nets>`_

As well as the information about ``--image-seeds`` from dgenerates ``--help``
output.


Animation Slicing
=================

Animated inputs can be sliced by a frame range, currently this only works globally so
if you provide multiple animated inputs they will all be sliced in an identical manner 
using the provided slice setting. Individual slice settings per image seed will probably 
be added in the future.

Perhaps you only want to run diffusion on the first frame of an animated input in
order to save time in finding good parameters for generating every frame. You could
do something like this in order to test different parameters on only the first frame,
which will be much faster than rendering the entire video/gif outright.

The slice range is inclusive, meaning that the frames pecified by ``--frame-start`` and ``--frame-end``
will be included in the slice.  Both slice points do not have to be specified at the same time, IE, you can slice
the tail end of a video out, or seek to a certain frame in the video and start from there if you wanted, by only
specifying a start, or an end parameter instead of both simultaneously.

If your slice only results in the processing of a single frame, it will be treated as a normal image seed and only
image output will be produced instead of an animation.


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

All combinations of animated seed and animated / or static mask can be handled.

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

    # Zip two videos together, masking the left video with corrisponding frames
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

Output images have a name format that starts with the seed, IE: `s_(seed here)_ ...png`

Reusing a seed has the effect of perfectly reproducing the image in the case that all
other parameters are left alone, including the model version.

You can output a configuration file for each image / animation produced that will reproduce it
exactly using the option ``--output-configs``, that same information can be written to the
metadata of generated PNG files using the option ``--output-metadata`` and can be read back
with ImageMagick for example as so:

.. code-block:: bash

    magick identify -format "%[Property:DgenerateConfig] generated_file.png

Generated configuration files can be read back into dgenerate using `Batch Processing From STDIN </#batch-processing-from-stdin>`_.

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

The refiner scheduler defaults to the value of ``--scheduler``, which in turn defaults to automatic selection.


Available schedulers for a specific combination of dgenerate arguments can be
queried using ``--scheduler help``, or ``--sdxl-refiner-scheduler help``, though both cannot
be queried simultaneously.

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
    #    "EulerDiscreteScheduler"
    #    "DPMSolverSinglestepScheduler"
    #    "DDIMScheduler"
    #    "KDPM2DiscreteScheduler"
    #    "KDPM2AncestralDiscreteScheduler"
    #    "HeunDiscreteScheduler"
    #    "DEISMultistepScheduler"
    #    "DPMSolverSDEScheduler"
    #    "DDPMScheduler"
    #    "PNDMScheduler"
    #    "UniPCMultistepScheduler"
    #    "EulerAncestralDiscreteScheduler"
    #    "DPMSolverMultistepScheduler"
    #    "LMSDiscreteScheduler"


Specifying a VAE
================

To specify a VAE directly use ``--vae``.

The syntax for ``--vae`` is ``AutoEncoderClass;model=(huggingface repository slug/blob link or file/folder path)``

Named arguments when loading a VAE are seperated by the ``;`` character and are not positional,
meaning they can be defined in any order.

Loading arguments available when specifying
a Torch VAE are: ``model``, ``revision``, ``variant``, ``subfolder``, and ``dtype``

Loading arguments available when specifying
a Flax VAE are ``model``, ``revision``, ``subfolder``, ``dtype``

The only named arguments compatible with loading a .safetensors or other model file
directly off disk is ``model``, ``dtype``, and ``revision``

The other named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk.

Available encoder classes for torch models are:

* AutoencoderKL
* AsymmetricAutoencoderKL (Does not support ``--vae-slicing`` or ``--vae-tiling``)
* AutoencoderTiny

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
valid when loading VAE's if ``--model-type`` is either ``torch`` or ``torch-sdxl``.  Attempting
to use it with FlaxAutoencoderKL with produce an error message. By default this value is the same as
``--variant`` when that option is specified for the main model.


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
accepted values are the same as ``--dtype``, IE: 'float32', 'float16', 'auto'

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
still be memory limited by the size of the image being processed by the Control Net, and still
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
    --vae AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix \
    --vae-tiling \
    --vae-slicing \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --output-size 2048 \
    --sdxl-target-size 2048 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


Specifying a LoRA Finetune
==========================

To specify a LoRA finetune model use ``--lora``

You can provide a huggingface repository slug, .pt, .pth, .bin, .ckpt, or .safetensors files.
Blob links are not accepted, for that use ``subfolder`` and ``weight-name`` described below.

The LoRA scale can be specified after the model path by placing a ``;`` (semicolon) and
then using the named argument ``scale``

When a scale is not specified, 1.0 is assumed.

Named arguments when loading a LoRA are seperated by the ``;`` character and are
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
    --lora "pcuenq/pokemon-lora;scale=0.5" \
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
    --lora "goofyai/SDXL-Lora-Collection;scale=1.0;weight-name=leonardo_illustration.safetensors" \
    --output-size 1024


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    dgenerate runwayml/stable-diffusion-v1-5 \
    --lora "pcuenq/pokemon-lora;scale=0.5;revision=main" \
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
    --lora "huggingface/lora_repo;scale=1.0;subfolder=repo_subfolder;weight-name=lora_weights.safetensors"


If you are loading a .safetensors or other file from a path on disk, only the ``scale`` argument is available.

.. code-block:: bash

    # This is only a syntax example

    dgenerate runwayml/stable-diffusion-v1-5 \
    --prompts "Syntax example" \
    --lora "my_lora.safetensors;scale=1.0"


Specifying Textual Inversions
=============================

One or more Textual Inversion models may be specified with ``--textual-inversions``

You can provide a huggingface repository slug, .pt, .pth, .bin, .ckpt, or .safetensors files.
Blob links are not accepted, for that use ``subfolder`` and ``weight-name`` described below.

Arguments pertaining to the loading of each textual inversion model may be specified in the same
way as when using ``--lora`` minus the scale argument.

Available arguments are: ``revision``, ``subfolder``, and ``weight-name``

Named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk, when loading directly from a .safetensors file
or other file from a path on disk they should not be used.


.. code-block:: bash

    # Load a textual inversion from a huggingface repository specifying it's name in the repository
    # as an argument

    Duskfallcrew/isometric-dreams-sd-1-5  \
    --textual-inversions Duskfallcrew/IsometricDreams_TextualInversions;weight-name=Isometric_Dreams-1000.pt \
    --scheduler KDPM2DiscreteScheduler \
    --inference-steps 30 \
    --guidance-scales 7 \
    --prompts "a bright photo of the Isometric_Dreams, a tv and a stereo in it and a book shelf, a table, a couch,a room with a bed"


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

One or more Control Net models may be specified with ``--control-nets``

You can provide a huggingface repository slug / blob link, .pt, .pth, .bin, .ckpt, or .safetensors files.

Control images for the Control Nets can be provided using ``--image-seeds``

When using ``--control-nets`` the syntax for specifying control images via ``--image-seeds`` is:

``--image-seeds "my-control-image.png"`` or ``--image-seeds "my-img2img-seed.png;mask=my-inpaint-mask.png;control=my-control-image.png"``

The "mask" argument is optional and used to request inpainting, ``resize=WIDTHxHEIGHT`` can be used to
select a per ``--image-size`` resize dimension for all image sources involved in that particular
specification.

Control Net guidance images may actually be animations such as MP4's, GIF's etc.
In fact when using ``--image-seeds``, frames can be taken from 3 videos simultaneously
(or any possible combination of image/video parameters) in order to generate a frame when
using an img2img image seed and/or inpaint mask with a Control Net guidance image.

Arguments pertaining to the loading of each Control Net model may be specified in the same
way as when using ``--vae`` with the addition of a ``scale`` argument and ``from_torch`` argument
when using flax.

Available arguments when using torch are: ``scale``, ``start``, ``end``, ``revision``, ``variant``, ``subfolder``, ``dtype``

Available arguments when using flax are: ``scale``, ``revision``, ``subfolder``, ``dtype``, ``from_torch``

Most named arguments apply to loading from a huggingface repository or folder
that may or may not be a local git repository on disk, when loading directly from a .safetensors file
or other file from a path on disk the available arguments are ``scale``, ``start``, ``end``, and ``from_torch``.
``from_torch`` can be used with flax for loading pytorch models from .pt or other files designed for torch from a repo or file/folder on disk.


The ``scale`` argument indicates the effect scale of the control net model.


For torch, the ``start`` argument indicates at what fraction of the total inference steps
at which the control net model starts to apply guidance. If you have multiple
control net models specified, they can apply guidance over different segments
of the inference steps using this option, it defaults to 0.0, meaning start at the
first inference step.


for torch, the ``end`` argument indicates at what fraction of the total inference steps
at which the control net model stops applying guidance. It defaults to 1.0, meaning
stop at the last inference step.


These examples use: `vermeer_canny_edged.png <https://raw.githubusercontent.com/Teriks/dgenerate/master/examples/media/vermeer_canny_edged.png>`_


.. code-block:: bash

    # Torch example, use "vermeer_canny_edged.png" as a control guidance image

    runwayml/stable-diffusion-v1-5 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earing by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets lllyasviel/sd-controlnet-canny;scale=0.5 \
    --image-seeds "vermeer_canny_edged.png"


    # If you have an img2img image seed, use this syntax

    runwayml/stable-diffusion-v1-5 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earing by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets lllyasviel/sd-controlnet-canny;scale=0.5 \
    --image-seeds "my-image-seed.png;control=vermeer_canny_edged.png"


    # If you have an img2img image seed and an inpainting mask, use this syntax

    runwayml/stable-diffusion-v1-5 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earing by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets lllyasviel/sd-controlnet-canny;scale=0.5 \
    --image-seeds "my-image-seed.png;mask=my-inpaint-mask.png;control=vermeer_canny_edged.png"

    # Flax example

    runwayml/stable-diffusion-v1-5 --model-type flax \
    --revision bf16 \
    --dtype float16 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earing by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets lllyasviel/sd-controlnet-canny;scale=0.5;from_torch=true \
    --image-seeds "vermeer_canny_edged.png"

    # SDXL example

    stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --vae AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --prompts "Taylor Swift, high quality, masterpiece, high resolution; low quality, bad quality, sketches" \
    --control-nets diffusers/controlnet-canny-sdxl-1.0;scale=0.5 \
    --image-seeds "vermeer_canny_edged.png" \
    --output-size 1024


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    # This is a non working example as I do not know of a repo that utilizes revisions with
    # Control Net weights :) this is only a syntax example

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



Specifying Generation Batch Size
================================

Multiple image variations from the same seed can be produce on a GPU simultaneously
using the ``--batch-size`` option of dgenerate. This can be used in combination with
``--batch-grid-size`` to output image grids if desired.

When not writing to image grids the files in the batch will be written to disk
with the suffix ``_image_N`` where N is index of the image in the batch of images
that were generated.

When producing an animation, you can either write **N** animation output files
with the filename suffixes ``_animation_N`` where **N** is the index of the image
in the batch which makes up the frames.  Or you can use ```--batch-grid-size`` to
write frames to a single animated output where the frames are all image grids
produced from the images in the batch.

With larger ``--batch-size`` values, the use of ``--vae-slicing`` can make the difference
between an out of memory condition and success, so it is recommended that you
try this option if you experience an out of memory condition due to the use of
``--batch-size``.


Image Preprocessors
===================

Images provided through ``--image-seeds`` can be preprocessed before being used for image generation
through the use of the arguments ``--seed-image-preprocessors``, ``--mask-image-preprocessors``, and
``--control-image-preprocessors``.

Each of these options can receive one or more specifications for image preprocessing actions.

For example images can be preprocessed with the canny edge detection algorithm or OpenPose (rigging generation)
before being used for generation with a model + a Control Net.

This image of a `horse <https://raw.githubusercontent.com/Teriks/dgenerate/master/examples/media/horse2.jpeg>`_
is used in the example below with a Control Net that is trained to generate images from canny edge detected input.

.. code-block:: bash

    # --control-image-preprocessors is only used for control images
    # in this case the single image seed is considered a control image
    # because --control-nets is being used

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --vae AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --prompts "Majestic unicorn, high quality, masterpiece, high resolution; low quality, bad quality, sketches" \
    --control-nets diffusers/controlnet-canny-sdxl-1.0;scale=0.5 \
    --image-seeds "horse.jpeg" \
    --control-image-preprocessors "canny;lower=50;upper=100" \
    --gen-seeds 2 \
    --output-size 1024 \
    --output-path unicorn


Using the option ``--image-preprocessor-help`` with no arguments will yield a list of available image preprocessor names.

You can also use ``--plugin-modules`` with his argument to include plugin modules into the preprocessor module search path.

Specifying one or more specific preprocessors for example: ``--image-preprocessor-help canny openpose`` will yield
documentation pertaining to those preprocessor modules. This includes accepted arguments for the preprocessor module
and a description of the module.

All preprocessors posses the arguments: ``output-file``, ``output-overwrite``, and ``device``

The ``output-file`` argument can be used to write the preprocessed image to a specific file, if multiple
processing steps occur such as when rendering an animation or multiple generation steps, a numbered suffix
will be appended to this filename. Note that an output file will only be produced in the case that the
preprocessor actually modifies an input image in some way.

The ``output-overwrite`` is a boolean argument can be used to tell the preprocessor that you do not want numbered
suffixes to be generated for ``output-file`` and to simply overwrite it.

The ``device`` argument can be used to override what device any hardware accelerated image processing
occurs on if any. It defaults to the value of ``--device`` and has the same syntax for specifying device
ordinals, for instance if you have multiple GPUs you may specify ``device=cuda:1`` to run image processing
on your second GPU, etc.

Custom image preprocessor modules can also be loaded through the ``--plugin-modules`` option as discussed in the next section.

Writing Plugins
===============

dgenerate has the capability of loading in additional functionality through the use of the ``--plugin-modules`` option.

You simply specify one or more module directories on disk, or paths to python files, using this argument.

Currently the only supported functionality of plugin modules is to add image preprocessors.

A code example as well as a command line usage example for image preprocessor plugins can be found
in the `"plugins/image_preprocessor" <https://github.com/Teriks/dgenerate/tree/master/examples/plugins/image_preprocessor>`_
folder of the examples folder.

The source code for the built in `canny <https://github.com/Teriks/dgenerate/blob/master/dgenerate/preprocessors/canny.py>`_ preprocessor,
the `openpose <https://github.com/Teriks/dgenerate/blob/master/dgenerate/preprocessors/openpose.py>`_ preprocessor, and the simple
`pillow image operations <https://github.com/Teriks/dgenerate/blob/master/dgenerate/preprocessors/pil_imageops.py>`_ preprocessors can also
be of reference as they are written as internal image preprocessor plugins.



Upscaling with Upscaler Models
==============================

Stable diffusion image upscaling models can be used via the model types ``torch-upscaler-x2`` and ``torch-upscaler-x4``.

The image used in the example below is this `low resolution cat <https://raw.githubusercontent.com/Teriks/dgenerate/master/examples/media/low_res_cat.png>`_

.. code-block:: bash

    # The image produced with this model will be
    # two times the --output-size dimension IE: 512x512 in this case
    # The image is being resized to 256x256, and then upscaled by 2x

    stabilityai/sd-x2-latent-upscaler --variant fp16 --dtype float16 \
    --model-type torch-upscaler-x2 \
    --prompts "a picture of a white cat" \
    --image-seeds low_res_cat.png \
    --output-size 256


    # The image produced with this model will be
    # four times the --output-size dimension IE: 1024x1024 in this case
    # The image is being resized to 256x256, and then upscaled by 4x

    stabilityai/stable-diffusion-x4-upscaler --variant fp16 --dtype float16 \
     --model-type torch-upscaler-x4 \
    --prompts "a picture of a white cat" \
    --image-seeds low_res_cat.png \
    --output-size 256 \
    --upscaler-noise-levels 20


Specifying an SDXL Refiner
==========================

When the main model is an SDXL model and ``--model-type torch-sdxl`` is specified,
you may specify a refiner model with ``--sdxl-refiner-path``.

You can provide paths to a huggingface repo/blob link, folder on disk, or a model file
on disk such as a .pt, .pth, .bin, .ckpt, or .safetensors file.

This argument is parsed in much the same way as the argument ``--vae``, except the
model is the first value specified.

Loading arguments available when specifying a refiner are: ``revision``, ``variant``, ``subfolder``, and ``dtype``

The only named argument compatible with loading a .safetensors or other file directly off disk is ``dtype``

The other named arguments are available when loading from a huggingface repo/blob link,
or folder that may or may not be a local git repository on disk.

.. code-block:: bash

    # Basic usage of SDXL with a refiner

    stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"



If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0;revision=main \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


If you wish to specify a weights variant IE: load ``pytorch_model.<variant>.safetensors``, from a huggingface
repository that has variants of the same model, use the named argument ``variant``. By default this
value is the same as ``--variant`` unless you override it.

.. code-block:: bash

    stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0;variant=fp16 \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    # This is a non working example as I do not know of a repo with an SDXL refiner
    # in a subfolder :) this is only a syntax example

    huggingface/sdxl_model --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner huggingface/sdxl_refiner;subfolder=repo_subfolder


If you want to select the model precision, use the named argument ``dtype``. By
default this value is the same as ``--dtype`` unless you override it. Accepted
values are the same as ``--dtype``, IE: 'float32', 'float16', 'auto'

.. code-block:: bash

    stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0;dtype=float16 \
    --sdxl-high-noise-fractions 0.8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --output-size 1024 \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"


If you are loading a .safetensors or other file from a path on disk, simply do:

.. code-block:: bash

    # This is only a syntax example

    huggingface/sdxl_model --model-type torch-sdxl \
    --sdxl-refiner my_refinermodel.safetensors



Batch Processing From STDIN
===========================

Program configuration can be read from STDIN and processed in batch with model caching,
in order to increase speed when many invocations with different arguments are desired.

Loading the necessary libraries and bringing models into memory is quite slow, so using the program this
way allows for multiple invocations using different arguments, without needing to load the libraries and
models multiple times, only the first time, or in the case of models the first time the model is encountered.

When a model is loaded dgenerate caches it in memory with it's creation parameters, which includes among other things
the pipeline mode (basic, img2img, inpaint), attached control nets, vae's, lora's and textual inversions.
If another invocation of the model occurs with creation parameters that are identical, it will be loaded out of cache.

Main/Refiner Models, VAE's, and Control Net models are cached individually.

VAE's and ControlNet models objects can be reused by main / refiner models in certain situations
and this is taken advantage of.

A number of things affect cache hit or miss upon model invocation, information regarding runtime
caching behavior of a pipeline can be observed using ``-v/--verbose``

When loading multiple different models be aware that they will all be retained in memory for
the duration of program execution, unless all models are flushed using the ``\clear_pipeline_caches`` directive.
Memory consumption may become and issue if you are not careful.

Environmental variables will be expanded in the provided input to **STDIN** when using this feature,
you may use Unix style notation for environmental variables even on Windows.

There is also information about the previous execution of dgenerate that is available to use
via Jinja2 templating which can be passed to ``--image-seeds``, these include:

* ``{{ last_images }}`` (A list of un-quoted filenames)
* ``{{ last_animations }}`` (A list of un-quoted filenames)

There are templates for prompts, containing the previous prompt values:

* ``{{ last_prompts }}`` (List of prompt objects with the un-quoted attributes 'positive' and 'negative')
* ``{{ last_sdxl_second_prompts }}``
* ``{{ last_sdxl_refiner_prompts }}``
* ``{{ last_sdxl_refiner_second_prompts }}``

Available custom jinja2 functions/filters are:

* ``{{ last(list_of_items) }}`` (Last element in a list)
* ``{{ unquote('"quotes_will_be_removed"') }}``
* ``{{ quote('quotes_will_be_added') }}``
* ``{{ quote_spaces(str_or_list) }}`` (Quote any string that contains spaces, works on lists)
* ``{{ format_prompt(prompt_object) }}`` (Format and quote a prompt object with its delimiter, works on lists)

The above can be used as either a function or filter IE: ``{{ "quote_me" | quote }}``

Empty lines and comments starting with ``#`` will be ignored.

You can create a multiline continuation using ``\`` to indicate that a line continues.

The following is a config file example that covers very basic syntax concepts:

.. code-block::

    #! dgenerate 2.0.0

    # If a hash-bang version is provided in the format above
    # a warning will be produced if the version you are running
    # is not compatible, this can be used anywhere in the config
    # file, a line number will be mentioned in the warning when the
    # version check fails

    # Comments in the file will be ignored

    # Guarantee unique file names are generated under the output directory by specifying unique seeds

    stabilityai/stable-diffusion-2-1 --prompts "an astronaut riding a horse" --seeds 41509644783027 --output-path output --inference-steps 30 --guidance-scales 10
    stabilityai/stable-diffusion-2-1 --prompts "a cowboy riding a horse" --seeds 78553317097366 --output-path output --inference-steps 30 --guidance-scales 10
    stabilityai/stable-diffusion-2-1 --prompts "a martian riding a horse" --seeds 22797399276707 --output-path output --inference-steps 30 --guidance-scales 10

    # Guarantee that no file name collisions happen by specifying different output paths for each invocation

    stabilityai/stable-diffusion-2-1 --prompts "an astronaut riding a horse" --output-path unique_output_1  --inference-steps 30 --guidance-scales 10
    stabilityai/stable-diffusion-2-1 --prompts "a cowboy riding a horse" --output-path unique_output_2 --inference-steps 30 --guidance-scales 10

    # Multiline continuations are possible by using \

    stabilityai/stable-diffusion-2-1 --prompts "a martian riding a horse" \
    --output-path unique_output_3  \

    # There can be comments or newlines within the continuation
    # The continuation ends when a line does not end in \

    --inference-steps 30 \
    --guidance-scales 10


    # A clear pipeline caches directive can be used inbetween invocations if cached models that
    # are no longer needed in your generation pipeline start causing out of memory issues

    \clear_pipeline_caches


    # This model was used before but will have to be fully instantiated from scratch again
    # after a cache flush which may take some time

    stabilityai/stable-diffusion-2-1 --prompts "a martian riding a horse" \
    --output-path unique_output_4


Here are examples of other available directives and templating:


.. code-block:: bash

    #! dgenerate 2.0.0

    # You can define your own template variables with the \set directive

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


    # An invocation sets various template variables related to its
    # execution once it is finished running

    stabilityai/stable-diffusion-2-1 --prompts {{ my_prompt }} --gen-seeds 5


    # Print a quoted filename of the last image produced by the last invocation
    # This could potentially be passed to --image-seeds of the next invocation
    # If you wanted to run another pass over the last image that was produced

    \print {{ quote(last(last_images)) }}


    # if you want to append a mask image file name

    \print "{{ last(last_images) }};my-mask.png"


    # Print a list of properly quoted filenames produced by the last
    # invocation seperated by spaces if there is multiple, this could
    # also be passed to --image-seeds

    \print {{ quote_spaces(last_images) | join(' ') }}


    # For loops are possible

    \print {% for image in last_images %}{{ quote(image) }} {% endfor %}


    # For loops are possible with continuation
    # however continuations will replace newlines
    # and whitespace with a single space.

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
    # quoted so you need to quote them one way or another

    stabilityai/stable-diffusion-2-1 --prompts "{{ last(last_prompts).positive }}"

    # "last_prompts" returns all the prompts used in the last invocation as a list
    # the "format_prompt" function can also work on a list

    stabilityai/stable-diffusion-2-1 --prompts "prompt 1" "prompt 2" "prompt 3"

    stabilityai/stable-diffusion-2-1 --prompts {{ format_prompt(last_prompts) }}


    # Execute additional config with full templating.
    # The sequence !END is interpreted as a newline within
    # the config file generated by the template and is required
    # when the template generates multiple lines of configuration.
    # You really should not need to use this feature.

    {% for image in last_images %} \
        stabilityai/stable-diffusion-2-1 --image-seeds {{ quote(image) }} --prompt {{ my_prompt }} !END \
    {% endfor %}


    # Multiple lines with continuations inside the config template.
    # Probably try to avoid this :)

    {% for image in last_images %} \
        stabilityai/stable-diffusion-2-1 \ !END \
        --image-seeds {{ quote(image) }} \ !END \
        --prompt {{ my_prompt }} !END \
    {% endfor %}


    # The above are both basically equivalent to this

    stabilityai/stable-diffusion-2-1 --image-seeds {{ quote_spaces(last_images) | join(' ') }} --prompt {{ my_prompt }}


To utilize configuration files on Linux, pipe them into the command or use redirection:


.. code-block:: bash

    # Pipe
    cat my-config.txt | dgenerate

    # Redirection
    dgenerate < my-config.txt


On Windows CMD:

.. code-block:: bash

    dgenerate < my-arguments.txt


On Windows Powershell:

.. code-block:: powershell

    Get-Content my-arguments.txt | dgenerate



Batch Processing Argument Injection
===================================


You can inject arguments into every generation call of a batch processing
configuration by simply specifying them. The arguments will added to the end
of the argument specification of every call.

.. code-block:: bash

    # Pipe
    cat my-animations-config.txt | dgenerate --frame-start 0 --frame-end 10

    # Redirection
    dgenerate --frame-start 0 --frame-end 10 < my-animations-config.txt


On Windows CMD:

.. code-block:: bash

    dgenerate  --frame-start 0 --frame-end 10 < my-animations-config.txt


On Windows Powershell:

.. code-block:: powershell

    Get-Content my-animations-config.txt | dgenerate --frame-start 0 --frame-end 10



File Cache Control
==================

dgenerate will cache ``--image-seeds`` files downloaded from the web while it is running in the
directory ``~/.cache/dgenerate/web``, on Windows this equates to ``%HOME%\.cache\dgenerate\web``

You can control where image seed files are cached with the environmental variable ``DGENERATE_WEB_CACHE``.

This directory is automatically cleared when dgenerate exits under any circumstance aside from a
complete interpreter crash.

Files downloaded from huggingface by the diffusers/huggingface_hub library will be cached under
``~/.cache/huggingface/``, on Windows this equates to ``%HOME%\.cache\huggingface\``.

This is controlled by the environmental variable ``HF_HOME``

In order to specify that all large model files be stored in another location,
for example on another disk, simply set ``HF_HOME`` to a new path in your environment.

You can read more about environmental variables that affect huggingface libraries on this
`huggingface documentation page <https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables>`_.











