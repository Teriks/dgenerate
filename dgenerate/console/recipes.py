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
        @file[Model File / Slug]:[{}]:"stabilityai/stable-diffusion-2-1"
        @optionalscheduler[Scheduler]:[--scheduler {}]:""@optionalpredictiontype[Scheduler Prediction Type]:[;prediction-type={}]:""
        @optionalfile[VAE File / Slug]:[--vae AutoencoderKL;model={}]:""
        @optionalfile[LoRa File / URI]:[--loras {};scale=1.0]:""
        @optionalfile[ControlNet File / URI]:[--control-nets {};scale=1.0]:""
        @optionalfile[UNet File / URI]:[--unet {}]:""
        @optionalfile[Image Seed]:[--image-seeds {}\\n--image-seed-strengths 0.8]:""
        @int[Inference Steps]:[--inference-steps {}]:"30"
        @int[Guidance Scale]:[--guidance-scales {}]:"5"
        @int[Clip Skip]:[--clip-skips {}]:"0"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @dir[Output Directory]:[--output-path {}]:"output"
        @optionalstring[Output Size]:[--output-size {}]:"512x512"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL":
        """
        @file[Model File / HF Slug]:[{}]:"stabilityai/stable-diffusion-xl-base-1.0"
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        @optionalscheduler[Scheduler]:[--scheduler {}]:""@optionalpredictiontype[Scheduler Prediction Type]:[;prediction-type={}]:""
        @optionalscheduler[Refiner Scheduler]:[--sdxl-refiner-scheduler {}]:""@optionalpredictiontype[Refiner Scheduler Prediction Type]:[;prediction-type={}]:""
        @optionalfile[Refiner File / URI]:[--sdxl-refiner {}]:"stabilityai/stable-diffusion-xl-refiner-1.0"
        @optionalfile[VAE File / Slug]:[--vae AutoencoderKL;model={}]:""
        @optionalfile[LoRa File / URI]:[--loras {};scale=1.0]:""
        @optionalfile[ControlNet File / URI]:[--control-nets {};scale=1.0]:""
        @optionalfile[UNet File / URI]:[--unet {}]:""
        @optionalfile[Refiner UNet File / URI]:[--unet2 {}]:""
        @optionalfile[Image Seed]:[--image-seeds {}\\n--image-seed-strengths 0.8]:""
        @int[Inference Steps]:[--inference-steps {}]:"30"
        @int[Guidance Scale]:[--guidance-scales {}]:"5"
        @float[SDXL High Noise Fraction]:[--sdxl-high-noise-fractions {}]:"0.8"
        @int[Clip Skip]:[--clip-skips {}]:"0"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @dir[Output Directory]:[--output-path {}]:"output"
        @optionalstring[Output Size]:[--output-size {}]:"1024x1024"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (no refiner)":
        """
        @file[Model File / HF Slug]:[{}]:"stabilityai/stable-diffusion-xl-base-1.0"
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        @optionalscheduler[Scheduler]:[--scheduler {}]:""@optionalpredictiontype[Scheduler Prediction Type]:[;prediction-type={}]:""
        @optionalfile[VAE File / Slug]:[--vae AutoencoderKL;model={}]:""
        @optionalfile[LoRa File / URI]:[--loras {};scale=1.0]:""
        @optionalfile[ControlNet File / URI]:[--control-nets {};scale=1.0]:""
        @optionalfile[UNet File / URI]:[--unet {}]:""
        @optionalfile[Refiner UNet File / URI]:[--unet2 {}]:""
        @optionalfile[Image Seed]:[--image-seeds {}\\n--image-seed-strengths 0.8]:""
        @int[Inference Steps]:[--inference-steps {}]:"30"
        @int[Guidance Scale]:[--guidance-scales {}]:"5"
        @int[Clip Skip]:[--clip-skips {}]:"0"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @dir[Output Directory]:[--output-path {}]:"output"
        @optionalstring[Output Size]:[--output-size {}]:"1024x1024"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM UNet no refiner)":
        """
        @file[Model File / HF Slug]:[{}]:"stabilityai/stable-diffusion-xl-base-1.0"
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --scheduler LCMScheduler
        @optionalfile[VAE File / Slug]:[--vae AutoencoderKL;model={}]:""
        @optionalfile[LoRa File / URI]:[--loras {};scale=1.0]:""
        @optionalfile[ControlNet File / URI]:[--control-nets {};scale=1.0]:""
        @optionalfile[UNet File / URI]:[--unet {}]:"latent-consistency/lcm-sdxl"
        @optionalfile[Refiner UNet File / URI]:[--unet2 {}]:""
        @optionalfile[Image Seed]:[--image-seeds {}\\n--image-seed-strengths 0.8]:""
        @int[Inference Steps]:[--inference-steps {}]:"4"
        @int[Guidance Scale]:[--guidance-scales {}]:"8"
        @int[Clip Skip]:[--clip-skips {}]:"0"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @dir[Output Directory]:[--output-path {}]:"output"
        @optionalstring[Output Size]:[--output-size {}]:"1024x1024"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM UNet cooperative refiner)":
        """
        @file[Model File / HF Slug]:[{}]:"stabilityai/stable-diffusion-xl-base-1.0"
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --scheduler LCMScheduler
        @optionalscheduler[Refiner Scheduler]:[--sdxl-refiner-scheduler {}]:"UniPCMultistepScheduler"@optionalpredictiontype[Refiner Scheduler Prediction Type]:[;prediction-type={}]:""
        @optionalfile[Refiner File / URI]:[--sdxl-refiner {}]:"stabilityai/stable-diffusion-xl-refiner-1.0"
        @optionalfile[VAE File / Slug]:[--vae AutoencoderKL;model={}]:""
        @optionalfile[LoRa File / URI]:[--loras {};scale=1.0]:""
        @optionalfile[ControlNet File / URI]:[--control-nets {};scale=1.0]:""
        @optionalfile[UNet File / URI]:[--unet {}]:"latent-consistency/lcm-sdxl"
        @optionalfile[Refiner UNet File / URI]:[--unet2 {}]:""
        @optionalfile[Image Seed]:[--image-seeds {}\\n--image-seed-strengths 0.8]:""
        @int[Inference Steps]:[--inference-steps {}]:"4"
        @int[Guidance Scale]:[--guidance-scales {}]:"8"
        @float[SDXL High Noise Fraction]:[--sdxl-high-noise-fractions {}]:"0.8"
        @int[Refiner Inference Steps]:[--sdxl-refiner-inference-steps {}]:"50"
        @int[Clip Skip]:[--clip-skips {}]:"0"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @dir[Output Directory]:[--output-path {}]:"output"
        @optionalstring[Output Size]:[--output-size {}]:"1024x1024"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM UNet refiner edit mode)":
        """
        @file[Model File / HF Slug]:[{}]:"stabilityai/stable-diffusion-xl-base-1.0"
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --sdxl-refiner-edit
        --scheduler LCMScheduler
        @optionalscheduler[Refiner Scheduler]:[--sdxl-refiner-scheduler {}]:"UniPCMultistepScheduler"@optionalpredictiontype[Refiner Scheduler Prediction Type]:[;prediction-type={}]:""
        @optionalfile[Refiner File / URI]:[--sdxl-refiner {}]:"stabilityai/stable-diffusion-xl-refiner-1.0"
        @optionalfile[VAE File / Slug]:[--vae AutoencoderKL;model={}]:""
        @optionalfile[LoRa File / URI]:[--loras {};scale=1.0]:""
        @optionalfile[ControlNet File / URI]:[--control-nets {};scale=1.0]:""
        @optionalfile[UNet File / URI]:[--unet {}]:"latent-consistency/lcm-sdxl"
        @optionalfile[Refiner UNet File / URI]:[--unet2 {}]:""
        @optionalfile[Image Seed]:[--image-seeds {}\\n--image-seed-strengths 0.8]:""
        @int[Inference Steps]:[--inference-steps {}]:"4"
        @int[Guidance Scale]:[--guidance-scales {}]:"8"
        @float[SDXL High Noise Fraction]:[--sdxl-high-noise-fractions {}]:"0.8"
        @int[Refiner Inference Steps]:[--sdxl-refiner-inference-steps {}]:"100"
        @int[Clip Skip]:[--clip-skips {}]:"0"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @dir[Output Directory]:[--output-path {}]:"output"
        @optionalstring[Output Size]:[--output-size {}]:"1024x1024"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM LoRA no refiner)":
        """
        @file[Model File / HF Slug]:[{}]:"stabilityai/stable-diffusion-xl-base-1.0"
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --scheduler LCMScheduler
        @optionalfile[VAE File / Slug]:[--vae AutoencoderKL;model={}]:""
        @optionalfile[LoRa File / URI]:[--loras {};scale=1.0]:"latent-consistency/lcm-lora-sdxl"
        @optionalfile[ControlNet File / URI]:[--control-nets {};scale=1.0]:""
        @optionalfile[UNet File / URI]:[--unet {}]:""
        @optionalfile[Refiner UNet File / URI]:[--unet2 {}]:""
        @optionalfile[Image Seed]:[--image-seeds {}\\n--image-seed-strengths 0.8]:""
        @int[Inference Steps]:[--inference-steps {}]:"4"
        @int[Guidance Scale]:[--guidance-scales {}]:"1"
        @int[Clip Skip]:[--clip-skips {}]:"0"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @dir[Output Directory]:[--output-path {}]:"output"
        @optionalstring[Output Size]:[--output-size {}]:"1024x1024"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM LoRA cooperative refiner)":
        """
        @file[Model File / HF Slug]:[{}]:"stabilityai/stable-diffusion-xl-base-1.0"
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --scheduler LCMScheduler
        @optionalscheduler[Refiner Scheduler]:[--sdxl-refiner-scheduler {}]:"UniPCMultistepScheduler"@optionalpredictiontype[Refiner Scheduler Prediction Type]:[;prediction-type={}]:""
        @optionalfile[Refiner File / URI]:[--sdxl-refiner {}]:"stabilityai/stable-diffusion-xl-refiner-1.0"
        @optionalfile[VAE File / Slug]:[--vae AutoencoderKL;model={}]:""
        @optionalfile[LoRa File / URI]:[--loras {};scale=1.0]:"latent-consistency/lcm-lora-sdxl"
        @optionalfile[ControlNet File / URI]:[--control-nets {};scale=1.0]:""
        @optionalfile[UNet File / URI]:[--unet {}]:""
        @optionalfile[Refiner UNet File / URI]:[--unet2 {}]:""
        @optionalfile[Image Seed]:[--image-seeds {}\\n--image-seed-strengths 0.8]:""
        @int[Inference Steps]:[--inference-steps {}]:"8"
        @int[Guidance Scale]:[--guidance-scales {}]:"1"
        @float[SDXL High Noise Fraction]:[--sdxl-high-noise-fractions {}]:"0.8"
        @int[Refiner Inference Steps]:[--sdxl-refiner-inference-steps {}]:"100"
        @int[Clip Skip]:[--clip-skips {}]:"0"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @dir[Output Directory]:[--output-path {}]:"output"
        @optionalstring[Output Size]:[--output-size {}]:"1024x1024"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Stable Diffusion XL (LCM LoRA refiner edit mode)":
        """
        @file[Model File / HF Slug]:[{}]:"stabilityai/stable-diffusion-xl-base-1.0"
        --model-type torch-sdxl
        --dtype float16
        --variant fp16
        --model-cpu-offload
        --sdxl-refiner-edit
        --scheduler LCMScheduler
        @optionalscheduler[Refiner Scheduler]:[--sdxl-refiner-scheduler {}]:"UniPCMultistepScheduler"@optionalpredictiontype[Refiner Scheduler Prediction Type]:[;prediction-type={}]:""
        @optionalfile[Refiner File / URI]:[--sdxl-refiner {}]:"stabilityai/stable-diffusion-xl-refiner-1.0"
        @optionalfile[VAE File / Slug]:[--vae AutoencoderKL;model={}]:""
        @optionalfile[LoRa File / URI]:[--loras {};scale=1.0]:"latent-consistency/lcm-lora-sdxl"
        @optionalfile[ControlNet File / URI]:[--control-nets {};scale=1.0]:""
        @optionalfile[UNet File / URI]:[--unet {}]:""
        @optionalfile[Refiner UNet File / URI]:[--unet2 {}]:""
        @optionalfile[Image Seed]:[--image-seeds {}\\n--image-seed-strengths 0.8]:""
        @int[Inference Steps]:[--inference-steps {}]:"8"
        @int[Guidance Scale]:[--guidance-scales {}]:"1"
        @float[SDXL High Noise Fraction]:[--sdxl-high-noise-fractions {}]:"0.8"
        @int[Refiner Inference Steps]:[--sdxl-refiner-inference-steps {}]:"100"
        @int[Clip Skip]:[--clip-skips {}]:"0"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @dir[Output Directory]:[--output-path {}]:"output"
        @optionalstring[Output Size]:[--output-size {}]:"1024x1024"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Stable Cascade":
        """
        @file[Model File / HF Slug]:[{}]:"stabilityai/stable-cascade-prior"
        --model-type torch-s-cascade
        --variant bf16
        --dtype bfloat16
        --model-cpu-offload
        --s-cascade-decoder-cpu-offload
        @file[Decoder File / URI]:[--s-cascade-decoder {}]:"stabilityai/stable-cascade;dtype=float16"
        @optionalfile[UNet / URI]:[--unet {}]:""
        @optionalfile[Decoder UNet / URI]:[--unet2 {}]:""
        @optionalfile[Image Seed]:[--image-seeds {}]:""
        @int[Inference Steps]:[--inference-steps {}]:"20"
        @int[Guidance Scale]:[--guidance-scales {}]:"4"
        @int[Decoder Inference Steps]:[--s-cascade-decoder-inference-steps {}]:"10"
        @int[Decoder Guidance Scale]:[--s-cascade-decoder-guidance-scales {}]:"0"
        @dir[Output Directory]:[--output-path {}]:"output"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @optionalstring[Output Size]:[--output-size {}]:"1024x1024"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Stable Cascade (UNet lite)":
        """
        @file[Model File / HF Slug]:[{}]:"stabilityai/stable-cascade-prior"
        --model-type torch-s-cascade
        --variant bf16
        --dtype bfloat16
        --model-cpu-offload
        --s-cascade-decoder-cpu-offload
        @file[Decoder File / URI]:[--s-cascade-decoder {}]:"stabilityai/stable-cascade;dtype=float16"
        @optionalfile[UNet / URI]:[--unet {}]:"stabilityai/stable-cascade-prior;subfolder=prior_lite"
        @optionalfile[Decoder UNet / URI]:[--unet2 {}]:"stabilityai/stable-cascade;subfolder=decoder_lite"
        @optionalfile[Image Seed]:[--image-seeds {}]:""
        @int[Inference Steps]:[--inference-steps {}]:"20"
        @int[Guidance Scale]:[--guidance-scales {}]:"4"
        @int[Decoder Inference Steps]:[--s-cascade-decoder-inference-steps {}]:"10"
        @int[Decoder Guidance Scale]:[--s-cascade-decoder-guidance-scales {}]:"0"
        @dir[Output Directory]:[--output-path {}]:"output"
        @int[Number Of Seeds]:[--gen-seeds {}]:"1"
        @optionalstring[Output Size]:[--output-size {}]:"1024x1024"
        @device[Device]:[--device {}]:"device"
        --prompts "add your prompt here"
        """,
    "Image Upscaling (Spandrel / chaiNNer)":
        """
        \\image_process @file[Image File]:[{}]:""
        @outputfile[Output File]:[--output {}]:"upscaled.png"
        @int[Image Alignment]:[--align {}]:"1"
        @file[Upscaler Model / URL]:[--processors upscaler;model={}]:"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
        @device[Device]:[--device {}]:"device"
        """
}
