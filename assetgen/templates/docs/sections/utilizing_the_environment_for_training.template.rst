Utilizing the Python Environment for Training
=============================================

The environment that dgenerate ships with is suitable for training and fine-tuning models
using the dreambooth LoRA training example scripts from the diffusers repository.

The ``\\exec`` directive can be used to invoke python from dgenerate's environment to run
these scripts if you desire.

You may also use tools like ``accelerate`` from the dgenerate environment.

This allows you to preform LoRA training and other training tasks without needing to
spend time making sure your python environment is correct for it.

This works even when you install dgenerate using a network install wizard into an
isolated python environment.

For example, you can setup a training configuration entirely inside of a dgenerate script like so:


```
#! /usr/bin/env dgenerate --file
#! dgenerate @VERSION

# you can download the dreambooth LoRA training script from
# https://github.com/huggingface/diffusers/blob/v0.35.1/examples/dreambooth/train_dreambooth_lora_sdxl.py

# This would be appropriate for a single character fine-tuning task with about 30 curated images

\exec python train_dreambooth_lora_sdxl.py
--pretrained_model_name_or_path "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
--dataset_name "dataset/instance_images"
--instance_prompt "My fancy character"
--caption_column "text"
--output_dir "output_lora"
--resolution 1024
--train_batch_size 1
--gradient_accumulation_steps 4
--learning_rate 1.5e-4
--text_encoder_lr 7.5e-5
--max_train_steps 1000
--rank 16
--lora_dropout 0.1
--train_text_encoder
--mixed_precision "fp16"
--enable_xformers_memory_efficient_attention
--gradient_checkpointing
--checkpointing_steps 100
--seed 42
```

``xformers`` and ``datasets`` are included in the dgenerate environment, allowing for memory efficient training
and also the use of Hugging Face datasets, or folders in Hugging Face dataset format with advanced captioning.

The nice thing about this is that you can use dgenerates scripting features to automate training variations,
for instance different learning rates, etc. if you just want to run a series of training jobs with different parameters.


```
#! /usr/bin/env dgenerate --file
#! dgenerate @VERSION

# UNet and Text Encoder learning rates to try

\setp learn_rates {"mild": ("5e-5", "2.5e-5"), "moderate": ("1.5e-4", "7.5e-5"), "aggressive": ("1e-4", "5e-5")}
\setp ranks [8, 16, 32]

{% for rank in ranks %}
    {% for rate in learn_rates.items() %}
        \exec dpython train_dreambooth_lora_sdxl.py
        --pretrained_model_name_or_path "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
        --dataset_name "dataset/instance_images"
        --instance_prompt "My fancy character"
        --caption_column "text"
        --output_dir "output_lora_{{rank}}_{{rate[0]}}"
        --resolution 1024
        --train_batch_size 1
        --gradient_accumulation_steps 4
        --learning_rate rate[1][0]
        --text_encoder_lr rate[1][1]
        --max_train_steps 1000
        --rank {{ rank }}
        --lora_dropout 0.1
        --train_text_encoder
        --mixed_precision "fp16"
        --enable_xformers_memory_efficient_attention
        --gradient_checkpointing
        --checkpointing_steps 100
        --seed 42
    {% endfor %}
{% endfor %}

```

