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

RECIPES = {
    "Stable Diffusion":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-2-1", "optional":false, "file-types":["models"]}]
        @dropdown[{"label":"Model dtype", "arg":"--dtype", "options":["float16", "float32"]}]
        @dropdown[{"label":"Model variant", "arg":"--variant", "options":["fp16"]}]
        @karrasscheduler[{}]
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "divider-after":true}]
        @torchvae[{"label":"VAE File / URI"}]
        @switch[{"label":"VAE Tiling", "arg":"--vae-tiling"}]
        @switch[{"label":"VAE Slicing", "arg":"--vae-slicing", "divider-after":true}]
        @uriwithscale[{"label":"LoRa File / URI", "scale_label":"LoRA Scale", "arg":"--loras", "file":true, "file-types":["models"]}]
        @uriwithscale[{"label":"ControlNet Directory / URI", "scale_label":"ControlNet Scale", "arg":"--control-nets", "dir":true}]
        @dir[{"label":"UNet Directory / URI", "arg":"--unet", "divider-after":true}]
        @uriwithargscale[{"label":"Image Seed", "arg":"--image-seeds", "file":true, "file-types":["images-in", "videos-in"], "scale_label":"Image Seed Strength", "scale_arg":"--image-seed-strengths", "min":0.01, "max":1, "default":""}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Seed Image Processor"}]
        @imageprocessor[{"arg":"--control-image-processors", "label":"Control Image Processor"}]
        @imageprocessor[{"arg":"--mask-image-processors", "label":"Inpaint Mask Processor", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imagesize[{"label":"Output Size (WxH)", "arg":"--output-size", "default":"512x512"}]
        @dropdown[{"label":"Prompt Weighter", "arg":"--prompt-weighter", "options":["compel", "compel;syntax=sdwui", "sd-embed"]}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion 3":
        r"""
        # Stable Diffusion 3 requires a huggingface auth token to access
        # you must request access to the repository

        \setp auth_token @string[{"label": "Hugging Face Auth Token", "default":"$HF_TOKEN", "optional":false}]

        \set auth_token {{ '--auth-token ' + quote(auth_token) if auth_token else '' }}

        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-3-medium-diffusers", "optional":false, "file-types":["models"]}]
        --model-type torch-sd3 {{ auth_token }}
        @dropdown[{"label":"Model dtype", "arg":"--dtype", "options":["float16", "float32"], "default":"float16"}]
        @dropdown[{"label":"Model variant", "arg":"--variant", "options":["fp16"], "default":"fp16"}]
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "default":1, "divider-after":true}]
        @torchvae[{"label":"VAE File / URI"}]
        @switch[{"label":"VAE Tiling", "arg":"--vae-tiling"}]
        @switch[{"label":"VAE Slicing", "arg":"--vae-slicing", "divider-after":true}]
        @uriwithscale[{"label":"LoRa File / URI", "scale_label":"LoRA Scale", "arg":"--loras", "file":true, "file-types":["models"]}]
        @uriwithscale[{"label":"ControlNet Directory / URI", "scale_label":"ControlNet Scale", "arg":"--control-nets", "dir":true, "divider-after":true}]
        @uriwithargscale[{"label":"Image Seed", "arg":"--image-seeds", "file":true, "file-types":["images-in", "videos-in"], "scale_label":"Image Seed Strength", "scale_arg":"--image-seed-strengths", "min":0.01, "max":1, "default":""}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Seed Image Processor"}]
        @imageprocessor[{"arg":"--control-image-processors", "label":"Control Image Processor", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imagesize[{"label":"Output Size (WxH)", "arg":"--output-size", "default":"1024x1024"}]
        @dropdown[{"label":"Prompt Weighter", "arg":"--prompt-weighter", "options":["sd-embed"]}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":["models"]}]
        --model-type torch-sdxl
        @dropdown[{"label":"Model dtype", "arg":"--dtype", "options":["float16", "float32"], "default":"float16"}]
        @dropdown[{"label":"Model variant", "arg":"--variant", "options":["fp16"], "default":"fp16"}]
        @karrasscheduler[{}]
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "divider-after":true}]
        @file[{"label":"Refiner File / URI", "arg":"--sdxl-refiner", "default":"stabilityai/stable-diffusion-xl-refiner-1.0", "file-types":["models"]}]
        @karrasscheduler[{"label":"Refiner Scheduler", "arg":"--sdxl-refiner-scheduler"}]
        @switchradio[{"labels":["Refiner CPU Offload", "Refiner Sequential Offload"], "args":["--sdxl-refiner-cpu-offload", "--sdxl-refiner-sequential-offload"]}]
        @switch[{"label":"Refiner Edit Mode", "arg":"--sdxl-refiner-edit", "divider-after":true}]
        @torchvae[{"label":"VAE File / URI"}]
        @switch[{"label":"VAE Tiling", "arg":"--vae-tiling"}]
        @switch[{"label":"VAE Slicing", "arg":"--vae-slicing", "divider-after":true}]
        @uriwithscale[{"label":"LoRa File / URI", "scale_label":"LoRA Scale", "arg":"--loras", "file":true, "file-types":["models"]}]
        @uriwithscale[{"label":"ControlNet Directory / URI", "scale_label":"ControlNet Scale", "arg":"--control-nets", "dir":true}]
        @dir[{"label":"UNet Directory / URI", "arg":"--unet"}]
        @dir[{"label":"Refiner UNet Directory / URI", "arg":"--unet2", "divider-after":true}]
        @uriwithargscale[{"label":"Image Seed", "arg":"--image-seeds", "file":true, "file-types":["images-in", "videos-in"], "scale_label":"Image Seed Strength", "scale_arg":"--image-seed-strengths", "min":0.01, "max":1, "default":""}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Seed Image Processor"}]
        @imageprocessor[{"arg":"--control-image-processors", "label":"Control Image Processor"}]
        @imageprocessor[{"arg":"--mask-image-processors", "label":"Inpaint Mask Processor", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @float[{"label":"SDXL High Noise Fraction", "arg":"--sdxl-high-noise-fractions", "default":0.8, "min":0.001, "max":0.99}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imagesize[{"label":"Output Size (WxH)", "arg":"--output-size", "default":"1024x1024"}]
        @dropdown[{"label":"Prompt Weighter", "arg":"--prompt-weighter", "options":["compel", "compel;syntax=sdwui", "sd-embed"]}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (no refiner)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":["models"]}]
        --model-type torch-sdxl
        @dropdown[{"label":"Model dtype", "arg":"--dtype", "options":["float16", "float32"], "default":"float16"}]
        @dropdown[{"label":"Model variant", "arg":"--variant", "options":["fp16"], "default":"fp16"}]
        @karrasscheduler[{}]
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "divider-after":true}]
        @torchvae[{"label":"VAE File / URI"}]
        @switch[{"label":"VAE Tiling", "arg":"--vae-tiling"}]
        @switch[{"label":"VAE Slicing", "arg":"--vae-slicing", "divider-after":true}]
        @uriwithscale[{"label":"LoRa File / URI", "scale_label":"LoRA Scale", "arg":"--loras", "file":true, "file-types":["models"]}]
        @uriwithscale[{"label":"ControlNet Directory / URI", "scale_label":"ControlNet Scale", "arg":"--control-nets", "dir":true}]
        @dir[{"label":"UNet Directory / URI", "arg":"--unet", "divider-after":true}]
        @uriwithargscale[{"label":"Image Seed", "arg":"--image-seeds", "file":true, "file-types":["images-in", "videos-in"], "scale_label":"Image Seed Strength", "scale_arg":"--image-seed-strengths", "min":0.01, "max":1, "default":""}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Seed Image Processor"}]
        @imageprocessor[{"arg":"--control-image-processors", "label":"Control Image Processor"}]
        @imageprocessor[{"arg":"--mask-image-processors", "label":"Inpaint Mask Processor", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imagesize[{"label":"Output Size (WxH)", "arg":"--output-size", "default":"1024x1024"}]
        @dropdown[{"label":"Prompt Weighter", "arg":"--prompt-weighter", "options":["compel", "compel;syntax=sdwui", "sd-embed"]}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM UNet)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":["models"]}]
        --model-type torch-sdxl
        @dropdown[{"label":"Model dtype", "arg":"--dtype", "options":["float16", "float32"], "default":"float16"}]
        @dropdown[{"label":"Model variant", "arg":"--variant", "options":["fp16"], "default":"fp16"}]
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "default":0, "divider-after":true}]
        --scheduler LCMScheduler
        @file[{"label":"Refiner File / URI", "arg":"--sdxl-refiner", "default":"stabilityai/stable-diffusion-xl-refiner-1.0", "file-types":["models"]}]
        @karrasscheduler[{"label":"Refiner Scheduler", "arg":"--sdxl-refiner-scheduler", "default":"UniPCMultistepScheduler"}]
        @switchradio[{"labels":["Refiner CPU Offload", "Refiner Sequential Offload"], "args":["--sdxl-refiner-cpu-offload", "--sdxl-refiner-sequential-offload"]}]
        @switch[{"label":"Refiner Edit Mode", "arg":"--sdxl-refiner-edit", "divider-after":true}]
        @torchvae[{"label":"VAE File / URI"}]
        @switch[{"label":"VAE Tiling", "arg":"--vae-tiling"}]
        @switch[{"label":"VAE Slicing", "arg":"--vae-slicing", "divider-after":true}]
        @uriwithscale[{"label":"LoRa File / URI", "scale_label":"LoRA Scale", "arg":"--loras", "file":true, "file-types":["models"]}]
        @uriwithscale[{"label":"ControlNet Directory / URI", "scale_label":"ControlNet Scale", "arg":"--control-nets", "dir":true}]
        @dir[{"label":"UNet Directory / URI", "arg":"--unet", "default":"latent-consistency/lcm-sdxl"}]
        @dir[{"label":"Refiner UNet Directory / URI", "arg":"--unet2", "divider-after":true}]
        @uriwithargscale[{"label":"Image Seed", "arg":"--image-seeds", "file":true, "file-types":["images-in", "videos-in"], "scale_label":"Image Seed Strength", "scale_arg":"--image-seed-strengths", "min":0.01, "max":1, "default":""}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Seed Image Processor"}]
        @imageprocessor[{"arg":"--control-image-processors", "label":"Control Image Processor"}]
        @imageprocessor[{"arg":"--mask-image-processors", "label":"Inpaint Mask Processor", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":4, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":8, "min":0}]
        @float[{"label":"SDXL High Noise Fraction", "arg":"--sdxl-high-noise-fractions", "default":0.8, "min":0.001, "max":0.99}]
        @int[{"label":"Refiner Inference Steps", "arg":"--sdxl-refiner-inference-steps", "default":100, "min":1}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imagesize[{"label":"Output Size (WxH)", "arg":"--output-size", "default":"1024x1024"}]
        @dropdown[{"label":"Prompt Weighter", "arg":"--prompt-weighter", "options":["compel", "compel;syntax=sdwui", "sd-embed"]}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM UNet no refiner)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":["models"]}]
        --model-type torch-sdxl
        @dropdown[{"label":"Model dtype", "arg":"--dtype", "options":["float16", "float32"], "default":"float16"}]
        @dropdown[{"label":"Model variant", "arg":"--variant", "options":["fp16"], "default":"fp16"}]
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "default":0, "divider-after":true}]
        --scheduler LCMScheduler
        @torchvae[{"label":"VAE File / URI"}]
        @switch[{"label":"VAE Tiling", "arg":"--vae-tiling"}]
        @switch[{"label":"VAE Slicing", "arg":"--vae-slicing", "divider-after":true}]
        @uriwithscale[{"label":"LoRa File / URI", "scale_label":"LoRA Scale", "arg":"--loras", "file":true, "file-types":["models"]}]
        @uriwithscale[{"label":"ControlNet Directory / URI", "scale_label":"ControlNet Scale", "arg":"--control-nets", "dir":true}]
        @dir[{"label":"UNet Directory / URI", "arg":"--unet", "default":"latent-consistency/lcm-sdxl", "divider-after":true}]
        @uriwithargscale[{"label":"Image Seed", "arg":"--image-seeds", "file":true, "file-types":["images-in", "videos-in"], "scale_label":"Image Seed Strength", "scale_arg":"--image-seed-strengths", "min":0.01, "max":1, "default":""}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Seed Image Processor"}]
        @imageprocessor[{"arg":"--control-image-processors", "label":"Control Image Processor"}]
        @imageprocessor[{"arg":"--mask-image-processors", "label":"Inpaint Mask Processor", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":4, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":8, "min":0}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imagesize[{"label":"Output Size (WxH)", "arg":"--output-size", "default":"1024x1024"}]
        @dropdown[{"label":"Prompt Weighter", "arg":"--prompt-weighter", "options":["compel", "compel;syntax=sdwui", "sd-embed"]}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM LoRA)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":["models"]}]
        --model-type torch-sdxl
        @dropdown[{"label":"Model dtype", "arg":"--dtype", "options":["float16", "float32"], "default":"float16"}]
        @dropdown[{"label":"Model variant", "arg":"--variant", "options":["fp16"], "default":"fp16"}]
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "default":0, "divider-after":true}]
        --scheduler LCMScheduler
        @file[{"label":"Refiner File / URI", "arg":"--sdxl-refiner", "default":"stabilityai/stable-diffusion-xl-refiner-1.0", "file-types":["models"]}]
        @karrasscheduler[{"label":"Refiner Scheduler", "arg":"--sdxl-refiner-scheduler", "default":"UniPCMultistepScheduler"}]
        @switchradio[{"labels":["Refiner CPU Offload", "Refiner Sequential Offload"], "args":["--sdxl-refiner-cpu-offload", "--sdxl-refiner-sequential-offload"]}]
        @switch[{"label":"Refiner Edit Mode", "arg":"--sdxl-refiner-edit", "divider-after":true}]
        @torchvae[{"label":"VAE File / URI"}]
        @switch[{"label":"VAE Tiling", "arg":"--vae-tiling"}]
        @switch[{"label":"VAE Slicing", "arg":"--vae-slicing", "divider-after":true}]
        @uriwithscale[{"label":"LoRa File / URI", "scale_label":"LoRA Scale", "arg":"--loras", "default":"latent-consistency/lcm-lora-sdxl", "file":true, "file-types":["models"]}]
        @uriwithscale[{"label":"ControlNet Directory / URI", "scale_label":"ControlNet Scale", "arg":"--control-nets", "dir":true}]
        @dir[{"label":"UNet Directory / URI", "arg":"--unet"}]
        @dir[{"label":"Refiner UNet Directory / URI", "arg":"--unet2", "divider-after":true}]
        @uriwithargscale[{"label":"Image Seed", "arg":"--image-seeds", "file":true, "file-types":["images-in", "videos-in"], "scale_label":"Image Seed Strength", "scale_arg":"--image-seed-strengths", "min":0.01, "max":1, "default":""}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Seed Image Processor"}]
        @imageprocessor[{"arg":"--control-image-processors", "label":"Control Image Processor"}]
        @imageprocessor[{"arg":"--mask-image-processors", "label":"Inpaint Mask Processor", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":8, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":1, "min":0}]
        @float[{"label":"SDXL High Noise Fraction", "arg":"--sdxl-high-noise-fractions", "default":0.8, "min":0.001, "max":0.99}]
        @int[{"label":"Refiner Inference Steps", "arg":"--sdxl-refiner-inference-steps", "default":100, "min":1}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imagesize[{"label":"Output Size (WxH)", "arg":"--output-size", "default":"1024x1024"}]
        @dropdown[{"label":"Prompt Weighter", "arg":"--prompt-weighter", "options":["compel", "compel;syntax=sdwui", "sd-embed"]}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM LoRA no refiner)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":["models"]}]
        --model-type torch-sdxl
        @dropdown[{"label":"Model dtype", "arg":"--dtype", "options":["float16", "float32"], "default":"float16"}]
        @dropdown[{"label":"Model variant", "arg":"--variant", "options":["fp16"], "default":"fp16"}]
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "default":0, "divider-after":true}]
        --scheduler LCMScheduler
        @torchvae[{"label":"VAE File / URI"}]
        @switch[{"label":"VAE Tiling", "arg":"--vae-tiling"}]
        @switch[{"label":"VAE Slicing", "arg":"--vae-slicing", "divider-after":true}]
        @uriwithscale[{"label":"LoRa File / URI", "scale_label":"LoRA Scale", "arg":"--loras", "default":"latent-consistency/lcm-lora-sdxl", "file":true, "file-types":["models"]}]
        @uriwithscale[{"label":"ControlNet Directory / URI", "scale_label":"ControlNet Scale", "arg":"--control-nets", "dir":true}]
        @dir[{"label":"UNet Directory / URI", "arg":"--unet", "divider-after":true}]
        @uriwithargscale[{"label":"Image Seed", "arg":"--image-seeds", "file":true, "file-types":["images-in", "videos-in"], "scale_label":"Image Seed Strength", "scale_arg":"--image-seed-strengths", "min":0.01, "max":1, "default":""}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Seed Image Processor"}]
        @imageprocessor[{"arg":"--control-image-processors", "label":"Control Image Processor"}]
        @imageprocessor[{"arg":"--mask-image-processors", "label":"Inpaint Mask Processor", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":4, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":1, "min":0}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imagesize[{"label":"Output Size (WxH)", "arg":"--output-size", "default":"1024x1024"}]
        @dropdown[{"label":"Prompt Weighter", "arg":"--prompt-weighter", "options":["compel", "compel;syntax=sdwui", "sd-embed"]}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Deep Floyd": r"""
        # DeepFloyd requires a multistage generation process involving 
        # multiple models and more advanced use of dgenerate
        
        # You need a huggingface account (http://huggingface.co) and to 
        # request access to the models at (https://huggingface.co/DeepFloyd) 
        # in order for dgenerate to be able to download the required models
        
        # once you have done this, provide your access token 
        # from (https://huggingface.co/settings/tokens)
        
        # Or set the environmental variable HF_TOKEN on your system
    
        \set prompt "add your prompt here"
        
        \setp auth_token @string[{"label": "Hugging Face Auth Token", "default":"$HF_TOKEN", "optional":false}]
        
        \set device @device[{"optional":false}]
        
        \set output_dir @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output", "optional":false, "divider-after":true}]
        
        \set auth_token {{ '--auth-token ' + quote(auth_token) if auth_token else '' }}
        
        @file[{"label":"Stage 1 Model File / HF Slug", "default": "DeepFloyd/IF-I-M-v1.0", "optional":false, "file-types":["models"]}]
        --variant fp16
        --dtype float16
        --model-type torch-if
        --model-sequential-offload
        @int[{"label":"Stage 1 Inference Steps", "arg":"--inference-steps", "default":60, "min":1}]
        @float[{"label":"Stage 1 Guidance Scale", "arg":"--guidance-scales", "default":7, "min":0}]
        --output-size 64
        @seeds[{"label":"Seeds", "divider-after":true}]
        --prompts {{ prompt }}
        --output-prefix stage1 {{ device }} {{ output_dir }} {{ auth_token }}
        
        \save_modules stage_1_modules feature_extractor
        
        @file[{"label":"Stage 2 Model File / HF Slug", "default": "DeepFloyd/IF-II-M-v1.0", "optional":false, "file-types":["models"]}]
        --variant fp16
        --dtype float16
        --model-type torch-ifs
        --model-sequential-offload
        @int[{"label":"Stage 2 Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Stage 2 Guidance Scale", "arg":"--guidance-scales", "default":4, "min":0}]
        @int[{"label":"Stage 2 Upscaler Noise Level", "arg":"--upscaler-noise-levels", "default":250, "min":1, "divider-after":true}]
        --prompts {{ format_prompt(last_prompts) }}
        --seeds {{ last_seeds | join(' ') }}
        --seeds-to-images
        --image-seeds {{ quote(last_images) }}
        --output-prefix stage2 {{ device }} {{ output_dir }} {{ auth_token }}
        
        \use_modules stage_1_modules
        
        @file[{"label":"Stage 3 - x4 Upscaler Model File / HF Slug", "default": "stabilityai/stable-diffusion-x4-upscaler", "optional":false, "file-types":["models"]}]
        --variant fp16
        --dtype float16
        --model-type torch-upscaler-x4
        @karrasscheduler[{"label":"Stage 3 Scheduler"}]
        @torchvae[{"label":"Stage 3 VAE File / URI"}]
        @int[{"label":"Stage 3 Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Stage 3 Guidance Scale", "arg":"--guidance-scales", "default":9, "min":0}]
        --prompts {{ format_prompt(last_prompts) }}
        --seeds {{ last_seeds | join(' ') }}
        --seeds-to-images
        --image-seeds {{ quote(last_images) }}
        @int[{"label":"Stage 3 Upscaler Noise Level", "arg":"--upscaler-noise-levels", "default":20, "min":1, "divider-after":true}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        --output-prefix stage3 {{ device }} {{ output_dir }} {{ auth_token }}
        
        \clear_modules stage_1_modules
    """,

    "Stable Cascade":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-cascade-prior", "optional":false, "file-types":["models"]}]
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "default":0, "divider-after":true}]
        --model-type torch-s-cascade
        --variant bf16
        --dtype bfloat16
        @file[{"label":"Decoder File / URI", "arg":"--s-cascade-decoder", "default":"stabilityai/stable-cascade;dtype=float16", "file-types":["models"]}]
        @switchradio[{"labels":["Decoder CPU Offload", "Decoder Sequential Offload"], "args":["--s-cascade-decoder-cpu-offload", "--s-cascade-decoder-sequential-offload"], "default":0, "divider-after":true}]
        @dir[{"label":"UNet Directory / URI", "arg":"--unet"}]
        @dir[{"label":"Decoder UNet Directory / URI", "arg":"--unet2", "divider-after":true}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "file-types":["images-in", "videos-in"]}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Seed Image Processor", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":20, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":4, "min":0}]
        @int[{"label":"Decoder Inference Steps", "arg":"--s-cascade-decoder-inference-steps", "default":10, "min":1}]
        @float[{"label":"Decoder Guidance Scale", "arg":"--s-cascade-decoder-guidance-scales", "default":0, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imagesize[{"label":"Output Size (WxH)", "arg":"--output-size", "default":"1024x1024"}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Cascade (UNet lite)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-cascade-prior", "optional":false, "file-types":["models"]}]
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "default":0, "divider-after":true}]
        --model-type torch-s-cascade
        --variant bf16
        --dtype bfloat16
        @file[{"label":"Decoder File / URI", "arg":"--s-cascade-decoder", "default":"stabilityai/stable-cascade;dtype=float16", "file-types":["models"]}]
        @switchradio[{"labels":["Decoder CPU Offload", "Decoder Sequential Offload"], "args":["--s-cascade-decoder-cpu-offload", "--s-cascade-decoder-sequential-offload"], "default":0, "divider-after":true}]
        @dir[{"label":"UNet Directory / URI", "arg":"--unet", "default":"stabilityai/stable-cascade-prior;subfolder=prior_lite"}]
        @dir[{"label":"Decoder UNet Directory / URI", "arg":"--unet2", "default":"stabilityai/stable-cascade;subfolder=decoder_lite", "divider-after":true}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "file-types":["images-in", "videos-in"]}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Seed Image Processor", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":20, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":4, "min":0}]
        @int[{"label":"Decoder Inference Steps", "arg":"--s-cascade-decoder-inference-steps", "default":10, "min":1}]
        @float[{"label":"Decoder Guidance Scale", "arg":"--s-cascade-decoder-guidance-scales", "default":0, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imagesize[{"label":"Output Size (WxH)", "arg":"--output-size", "default":"1024x1024"}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Upscaling (Stable Diffusion x2)":
        """
        stabilityai/sd-x2-latent-upscaler
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "divider-after":true}]
        --model-type torch-upscaler-x2
        --dtype float16
        @file[{"label":"Input Image File", "arg":"--image-seeds", "optional":false, "mode":"input", "file-types":["images-in", "videos-in"]}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Input Image Processor", "divider-after":true}]
        @switch[{"label":"VAE Tiling", "arg":"--vae-tiling"}]
        @switch[{"label":"VAE Slicing", "arg":"--vae-slicing", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "your prompt here"
        """,
    "Upscaling (Stable Diffusion x4)":
        """
        stabilityai/stable-diffusion-x4-upscaler
        @switchradio[{"labels":["Model CPU Offload", "Model Sequential Offload"], "args":["--model-cpu-offload", "--model-sequential-offload"], "divider-after":true}]
        --model-type torch-upscaler-x4
        --dtype float16
        --variant fp16
        @file[{"label":"Input Image File", "arg":"--image-seeds", "optional":false, "mode":"input", "file-types":["images-in", "videos-in"]}]
        @imageprocessor[{"arg":"--seed-image-processors", "label":"Input Image Processor", "divider-after":true}]
        @switch[{"label":"VAE Tiling", "arg":"--vae-tiling"}]
        @switch[{"label":"VAE Slicing", "arg":"--vae-slicing", "divider-after":true}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @int[{"label":"Upscaler Noise Level", "arg":"--upscaler-noise-levels", "default":20, "min":0}]
        @seeds[{"label":"Seeds"}]
        @int[{"label":"Batch Size", "arg":"--batch-size", "default":"", "min":1}]
        @imagesize[{"label":"Batch Grid Size (CxR)", "arg":"--batch-grid-size", "default":"", "divider-after":true}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @dropdown[{"label":"Prompt Weighter", "arg":"--prompt-weighter", "options":["compel", "compel;syntax=sdwui", "sd-embed"]}]
        @imageprocessor[{"arg":"--post-processors", "label":"Post Processor"}]
        @device[{}]
        --prompts "your prompt here"
        """,
    "Upscaling [openmodeldb.info] (Spandrel / chaiNNer)":
        """
        \\image_process @file[{"label":"Input Image File", "optional":false, "mode":"input", "file-types":["images-in", "videos-in"]}]
        @file[{"label":"Output Image File", "default":"output.png", "arg":"--output", "mode":"output", "file-types":["images-out", "videos-out"]}]
        @int[{"label":"Image Alignment", "arg":"--align", "default":1, "min":1}]
        @file[{"label":"Upscaler Model / URL", "arg":"--processors", "before":"upscaler;model=", "file-types":["models"], "default": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", "optional":false}]
        @device[{}]
        """,
    "Upscaling (to directory) [openmodeldb.info] (Spandrel / chaiNNer)":
        """
        \\image_process @file[{"label":"Input Image File", "optional":false, "mode":"input", "file-types":["images-in", "videos-in"]}]
        @dir[{"label":"Output Directory", "default":"output", "arg":"--output", "after":"/", "mode":"output"}]
        @int[{"label":"Image Alignment", "arg":"--align", "default":1, "min":1}]
        @file[{"label":"Upscaler Model / URL", "arg":"--processors", "before":"upscaler;model=", "file-types":["models"], "default": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", "optional":false}]
        @device[{}]
        """,
    "Generic Image Process":
        """
        \\image_process @file[{"label":"Input File", "optional":false, "mode":"input", "file-types":["images-in", "videos-in"]}]
        @file[{"label":"Output File", "default":"output.png", "arg":"--output", "mode":"output", "file-types":["images-out", "videos-out"]}]
        @switch[{"label":"Output Overwrite", "arg":"--output-overwrite", "divider-after":true}]
        @imagesize[{"label":"Resize Dimension (WxH)", "arg":"--resize"}]
        @switch[{"label":"Resize Ignores Aspect", "arg":"--no-aspect"}]
        @int[{"label":"Image Alignment", "arg":"--align", "min":1, "default":8, "divider-after":true}]
        @int[{"label":"Frame Start (inclusive)", "arg":"--frame-start", "min":0, "default":""}]
        @int[{"label":"Frame End (inclusive)", "arg":"--frame-end", "min":0, "default":""}]
        @switch[{"label":"Don't Output Frames", "arg":"--no-frames"}]
        @switch[{"label":"Don't Output Animation File", "arg":"--no-animation-file", "divider-after":true}]
        @imageprocessor[{"arg":"--processors", "label":"Image Processor"}]
        @device[{}]
        """,
    "Generic Image Process (to directory)":
        """
        \\image_process @file[{"label":"Input File", "optional":false, "mode":"input", "file-types":["images-in", "videos-in"]}]
        @dir[{"label":"Output Directory", "default":"output", "arg":"--output", "after":"/", "mode":"output"}]
        @switch[{"label":"Output Overwrite", "arg":"--output-overwrite", "divider-after":true}]
        @imagesize[{"label":"Resize Dimension (WxH)", "arg":"--resize"}]
        @switch[{"label":"Resize Ignores Aspect", "arg":"--no-aspect"}]
        @int[{"label":"Image Alignment", "arg":"--align", "min":1, "default":8, "divider-after":true}]
        @int[{"label":"Frame Start (inclusive)", "arg":"--frame-start", "min":0, "default":""}]
        @int[{"label":"Frame End (inclusive)", "arg":"--frame-end", "min":0, "default":""}]
        @switch[{"label":"Don't Output Frames", "arg":"--no-frames"}]
        @switch[{"label":"Don't Output Animation File", "arg":"--no-animation-file", "divider-after":true}]
        @imageprocessor[{"arg":"--processors", "label":"Image Processor"}]
        @device[{}]
        """
}
