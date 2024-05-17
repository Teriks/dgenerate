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
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-2-1", "optional":false, "file-types":"models"}]
        @karrasscheduler[{}]
        @torchvae[{"label":"VAE File / URI"}]
        @file[{"label":"LoRa File / URI", "arg":"--loras", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"ControlNet File / URI", "arg":"--control-nets", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "after":"\\n--image-seed-strengths 0.8"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"512x512"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":"models"}]
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        @karrasscheduler[{}]
        @file[{"label":"Refiner File / URI", "arg":"--sdxl-refiner", "default":"stabilityai/stable-diffusion-xl-refiner-1.0", "file-types":"models"}]
        @karrasscheduler[{"label":"Refiner Scheduler", "arg":"--sdxl-refiner-scheduler"}]
        @torchvae[{"label":"VAE File / URI"}]
        @file[{"label":"LoRa File / URI", "arg":"--loras", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"ControlNet File / URI", "arg":"--control-nets", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "file-types":"models"}]
        @file[{"label":"Refiner UNet File / URI", "arg":"--unet2", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "after":"\\n--image-seed-strengths 0.8"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @float[{"label":"SDXL High Noise Fraction", "arg":"--sdxl-high-noise-fractions", "default":0.8, "min":0.001, "max":0.99}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":0}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"1024x1024"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (no refiner)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":"models"}]
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        @karrasscheduler[{}]
        @torchvae[{"label":"VAE File / URI"}]
        @file[{"label":"LoRa File / URI", "arg":"--loras", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"ControlNet File / URI", "arg":"--control-nets", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "after":"\\n--image-seed-strengths 0.8"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"1024x1024"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM UNet no refiner)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":"models"}]
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --scheduler LCMScheduler
        @torchvae[{"label":"VAE File / URI"}]
        @file[{"label":"LoRa File / URI", "arg":"--loras", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"ControlNet File / URI", "arg":"--control-nets", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "default":"latent-consistency/lcm-sdxl", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "after":"\\n--image-seed-strengths 0.8"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":4, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":8, "min":0}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"1024x1024"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM UNet cooperative refiner)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":"models"}]
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --scheduler LCMScheduler
        @file[{"label":"Refiner File / URI", "arg":"--sdxl-refiner", "default":"stabilityai/stable-diffusion-xl-refiner-1.0", "file-types":"models"}]
        @karrasscheduler[{"label":"Refiner Scheduler", "arg":"--sdxl-refiner-scheduler", "default":"UniPCMultistepScheduler"}]
        @torchvae[{"label":"VAE File / URI"}]
        @file[{"label":"LoRa File / URI", "arg":"--loras", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"ControlNet File / URI", "arg":"--control-nets", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "default":"latent-consistency/lcm-sdxl", "file-types":"models"}]
        @file[{"label":"Refiner UNet File / URI", "arg":"--unet2", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "after":"\\n--image-seed-strengths 0.8"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":4, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":8, "min":0}]
        @float[{"label":"SDXL High Noise Fraction", "arg":"--sdxl-high-noise-fractions", "default":0.8, "min":0.001, "max":0.99}]
        @int[{"label":"Refiner Inference Steps", "arg":"--sdxl-refiner-inference-steps", "default":50, "min":1}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"1024x1024"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM UNet refiner edit mode)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":"models"}]
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --sdxl-refiner-edit
        --scheduler LCMScheduler
        @file[{"label":"Refiner File / URI", "arg":"--sdxl-refiner", "default":"stabilityai/stable-diffusion-xl-refiner-1.0", "file-types":"models"}]
        @karrasscheduler[{"label":"Refiner Scheduler", "arg":"--sdxl-refiner-scheduler", "default":"UniPCMultistepScheduler"}]
        @torchvae[{"label":"VAE File / URI"}]
        @file[{"label":"LoRa File / URI", "arg":"--loras", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"ControlNet File / URI", "arg":"--control-nets", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "default":"latent-consistency/lcm-sdxl", "file-types":"models"}]
        @file[{"label":"Refiner UNet File / URI", "arg":"--unet2", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "after":"\\n--image-seed-strengths 0.8"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":4, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":8, "min":0}]
        @float[{"label":"SDXL High Noise Fraction", "arg":"--sdxl-high-noise-fractions", "default":0.8, "min":0.001, "max":0.99}]
        @int[{"label":"Refiner Inference Steps", "arg":"--sdxl-refiner-inference-steps", "default":100, "min":1}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"1024x1024"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM LoRA no refiner)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":"models"}]
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --scheduler LCMScheduler
        @torchvae[{"label":"VAE File / URI"}]
        @file[{"label":"LoRa File / URI", "arg":"--loras", "default":"latent-consistency/lcm-lora-sdxl", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"ControlNet File / URI", "arg":"--control-nets", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "after":"\\n--image-seed-strengths 0.8"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":4, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":1, "min":0}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"1024x1024"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM LoRA cooperative refiner)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":"models"}]
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --scheduler LCMScheduler
        @file[{"label":"Refiner File / URI", "arg":"--sdxl-refiner", "default":"stabilityai/stable-diffusion-xl-refiner-1.0", "file-types":"models"}]
        @karrasscheduler[{"label":"Refiner Scheduler", "arg":"--sdxl-refiner-scheduler", "default":"UniPCMultistepScheduler"}]
        @torchvae[{"label":"VAE File / URI"}]
        @file[{"label":"LoRa File / URI", "arg":"--loras", "default":"latent-consistency/lcm-lora-sdxl", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"ControlNet File / URI", "arg":"--control-nets", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "file-types":"models"}]
        @file[{"label":"Refiner UNet File / URI", "arg":"--unet2", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "after":"\\n--image-seed-strengths 0.8"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":8, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":1, "min":0}]
        @float[{"label":"SDXL High Noise Fraction", "arg":"--sdxl-high-noise-fractions", "default":0.8, "min":0.001, "max":0.99}]
        @int[{"label":"Refiner Inference Steps", "arg":"--sdxl-refiner-inference-steps", "default":100, "min":1}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"1024x1024"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM LoRA refiner edit mode)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-diffusion-xl-base-1.0", "optional":false, "file-types":"models"}]
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --sdxl-refiner-edit
        --scheduler LCMScheduler
        @file[{"label":"Refiner File / URI", "arg":"--sdxl-refiner", "default":"stabilityai/stable-diffusion-xl-refiner-1.0", "file-types":"models"}]
        @karrasscheduler[{"label":"Refiner Scheduler", "arg":"--sdxl-refiner-scheduler", "default":"UniPCMultistepScheduler"}]
        @torchvae[{"label":"VAE File / URI"}]
        @file[{"label":"LoRa File / URI", "arg":"--loras", "default":"latent-consistency/lcm-lora-sdxl", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"ControlNet File / URI", "arg":"--control-nets", "after":";scale=1.0", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "file-types":"models"}]
        @file[{"label":"Refiner UNet File / URI", "arg":"--unet2", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds", "after":"\\n--image-seed-strengths 0.8"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":8, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":1, "min":0}]
        @float[{"label":"SDXL High Noise Fraction", "arg":"--sdxl-high-noise-fractions", "default":0.8, "min":0.001, "max":0.99}]
        @int[{"label":"Refiner Inference Steps", "arg":"--sdxl-refiner-inference-steps", "default":100, "min":1}]
        @int[{"label":"Clip Skip", "arg":"--clip-skips", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"1024x1024"}]
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
        
        \set auth_token @string[{"label": "Hugging Face Auth Token", "arg":"--auth-token", "default":"%HF_TOKEN%", "optional":false}]
        
        \set device @device[{"optional":false}]
        
        \set output_dir @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output", "optional":false}]
        
        @file[{"label":"Stage 1 Model File / HF Slug", "default": "DeepFloyd/IF-I-M-v1.0", "optional":false, "file-types":"models"}]
        --variant fp16
        --dtype float16
        --model-type torch-if
        --model-sequential-offload
        @int[{"label":"Stage 1 Inference Steps", "arg":"--inference-steps", "default":60, "min":1}]
        @float[{"label":"Stage 1 Guidance Scale", "arg":"--guidance-scales", "default":7, "min":0}]
        --output-size 64
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        --prompts {{ prompt }}
        --output-prefix stage1 {{ device }} {{ output_dir }} {{ auth_token }}
        
        \save_modules stage_1_modules feature_extractor
        
        @file[{"label":"Stage 2 Model File / HF Slug", "default": "DeepFloyd/IF-II-M-v1.0", "optional":false, "file-types":"models"}]
        --variant fp16
        --dtype float16
        --model-type torch-ifs
        --model-sequential-offload
        @int[{"label":"Stage 2 Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Stage 2 Guidance Scale", "arg":"--guidance-scales", "default":4, "min":0}]
        @int[{"label":"Stage 2 Upscaler Noise Level", "arg":"--upscaler-noise-levels", "default":250, "min":1}]
        --prompts {{ format_prompt(last_prompts) }}
        --seeds {{ last_seeds | join(' ') }}
        --seeds-to-images
        --image-seeds {{ quote(last_images) }}
        --output-prefix stage2 {{ device }} {{ output_dir }} {{ auth_token }}
        
        \use_modules stage_1_modules
        
        @file[{"label":"x4 Upscaler Model File / HF Slug", "default": "stabilityai/stable-diffusion-x4-upscaler", "optional":false, "file-types":"models"}]
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
        @int[{"label":"Stage 3 Upscaler Noise Level", "arg":"--upscaler-noise-levels", "default":20, "min":1}]
        --output-prefix stage3 {{ device }} {{ output_dir }} {{ auth_token }}
        
        \clear_modules stage_1_modules
    """,

    "Stable Cascade":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-cascade-prior", "optional":false, "file-types":"models"}]
        --model-type torch-s-cascade
        --variant bf16
        --dtype bfloat16
        --model-cpu-offload
        --s-cascade-decoder-cpu-offload
        @file[{"label":"Decoder File / URI", "arg":"--s-cascade-decoder", "default":"stabilityai/stable-cascade;dtype=float16", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "file-types":"models"}]
        @file[{"label":"Decoder UNet / URI", "arg":"--unet2", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":20, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":4, "min":0}]
        @int[{"label":"Decoder Inference Steps", "arg":"--s-cascade-decoder-inference-steps", "default":10, "min":1}]
        @float[{"label":"Decoder Guidance Scale", "arg":"--s-cascade-decoder-guidance-scales", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"1024x1024"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Stable Cascade (UNet lite)":
        """
        @file[{"label":"Model File / HF Slug", "default": "stabilityai/stable-cascade-prior", "optional":false, "file-types":"models"}]
        --model-type torch-s-cascade
        --variant bf16
        --dtype bfloat16
        --model-cpu-offload
        --s-cascade-decoder-cpu-offload
        @file[{"label":"Decoder File / URI", "arg":"--s-cascade-decoder", "default":"stabilityai/stable-cascade;dtype=float16", "file-types":"models"}]
        @file[{"label":"UNet File / URI", "arg":"--unet", "default":"stabilityai/stable-cascade-prior;subfolder=prior_lite", "file-types":"models"}]
        @file[{"label":"Decoder UNet / URI", "arg":"--unet2", "default":"stabilityai/stable-cascade;subfolder=decoder_lite", "file-types":"models"}]
        @file[{"label":"Image Seed", "arg":"--image-seeds"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":20, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":4, "min":0}]
        @int[{"label":"Decoder Inference Steps", "arg":"--s-cascade-decoder-inference-steps", "default":10, "min":1}]
        @float[{"label":"Decoder Guidance Scale", "arg":"--s-cascade-decoder-guidance-scales", "default":0, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @string[{"label":"Output Size", "arg":"--output-size", "default":"1024x1024"}]
        @device[{}]
        --prompts "add your prompt here"
        """,
    "Upscaling (Stable Diffusion x2)":
        """
        stabilityai/sd-x2-latent-upscaler 
        --dtype float16
        --model-type torch-upscaler-x2
        @file[{"label":"Input Image File", "arg":"--image-seeds", "optional":false, "mode":"input"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @device[{}]
        --prompts "your prompt here"
        """,
    "Upscaling (Stable Diffusion x4)":
        """
        stabilityai/stable-diffusion-x4-upscaler 
        --variant fp16 --dtype float16
        --model-type torch-upscaler-x4
        @file[{"label":"Input Image File", "arg":"--image-seeds", "optional":false, "mode":"input"}]
        @int[{"label":"Inference Steps", "arg":"--inference-steps", "default":30, "min":1}]
        @float[{"label":"Guidance Scale", "arg":"--guidance-scales", "default":5, "min":0}]
        @int[{"label":"Upscaler Noise Level", "arg":"--upscaler-noise-levels", "default":20, "min":0}]
        @int[{"label":"Number Of Seeds", "arg":"--gen-seeds", "default":1, "min":1}]
        @dir[{"label":"Output Directory", "arg":"--output-path", "default":"output"}]
        @device[{}]
        --prompts "your prompt here"
        """,
    "Upscaling [openmodeldb.info] (Spandrel / chaiNNer)":
        """
        \\image_process @file[{"label":"Input Image File", "optional":false, "mode":"input"}]
        @file[{"label":"Output Image File", "default":"output.png", "arg":"--output", "mode":"output"}]
        @int[{"label":"Image Alignment", "arg":"--align", "default":1, "min":1}]
        @file[{"label":"Upscaler Model / URL", "arg":"--processors", "before":"upscaler;model=", "file-types":"models", "default": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", "optional":false}]
        @device[{}]
        """
}
