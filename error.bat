dgenerate runwayml/stable-diffusion-v1-5 ^
--prompts "an astronaut riding a horse" ^
--control-nets lllyasviel/sd-controlnet-canny ^
--image-seeds "my-static-image-seed.png;mask=my-animation-mask.webp;control=my-animation-mask.webp" ^
--output-path inpaint2 ^
--animation-format mp4 ^
--output-configs --frame-end 1 ^
--inference-steps 5

dgenerate runwayml/stable-diffusion-v1-5 ^
--prompts "an astronaut riding a horse" ^
--control-nets lllyasviel/sd-controlnet-canny ^
--image-seeds "my-static-image-seed.png;mask=my-animation-mask.gif;control=my-animation-mask.gif" ^
--output-path inpaint ^
--animation-format mp4 ^
--output-configs --frame-end 1 ^
--inference-steps 5

dgenerate runwayml/stable-diffusion-v1-5 ^
--prompts "an astronaut riding a horse" ^
--control-nets lllyasviel/sd-controlnet-canny ^
--image-seeds "my-static-image-seed.png;mask=my-animation-mask.mp4;control=my-animation-mask.mp4" ^
--output-path inpaint ^
--animation-format mp4 ^
--output-configs --frame-end 1 ^
--inference-steps 5