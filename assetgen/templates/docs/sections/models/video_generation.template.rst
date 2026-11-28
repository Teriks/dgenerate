.. _video-generation:

Video Generation
================

Each model below writes one clip per pipeline call. Conditioning stays in
``--image-seeds``. ``--video-lengths`` is seconds and ``--video-fps`` is the
frame rate. One combination of prompt, seed, guidance, steps, and those video
arguments writes one animation file.

* HunyuanVideo 1.5 (``hyvideo-15``)
* Motif-Video (``motif-video``)
* MiniMax-H3 (``minimax-h3-video``)
* Kandinsky 5 (``kandinsky5-video``)
* Kandinsky 6 (``kandinsky6-video``)
* LTX-2.5 (``ltx2``) and LTX-Video (``ltx-video``)
* Wan 2.1 / 2.2 (``wan``, ``wan-animate``, and ``wan-animate-2``)

HunyuanVideo 1.5, Motif-Video, and Kandinsky 5 write frames only.
MiniMax-H3, Kandinsky 6, and LTX-2.5 also write a soundtrack. That audio is
muxed into an mp4. GIF, WebP, and ``--animation-format frames`` drop it.

Library
-------

A video pipeline yields :class:`dgenerate.VideoGeneratedEvent` with the frames,
frame rate, and soundtrack in memory. ``RenderLoop(..., disable_writes=True)``
does not write a file. ``event.write`` muxes the clip to a path or a binary
stream. A path's extension selects the container. A ``.frames`` path writes
one PNG per frame. A stream is mp4 unless
``animation_format`` is given. A seekable stream, such as ``io.BytesIO``, is
rewound when the mux finishes. A non-seekable stream receives a fragmented mp4.
The soundtrack is muxed into mp4. Call ``write`` while handling the event; the
frames are closed when the loop moves on.

.. code-block:: python

    from dgenerate import RenderLoop, VideoGeneratedEvent

    render_loop = RenderLoop(config=config, disable_writes=True)
    for event in render_loop.events():
        if isinstance(event, VideoGeneratedEvent):
            event.write(event.suggested_filename)
            event.write(stream)

``PipelineWrapper`` returns the same clip on
:class:`dgenerate.pipelinewrapper.GenerationResult`. ``result.write`` is that
mux.

.. code-block:: python

    result = wrapper(args)
    result.write('clip.mp4')
    result.write(stream)

``dgenerate.mediaoutput.write_animation`` is the same mux for frames you
already have. Examples are under
`examples/library_usage <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/library_usage>`_.

HunyuanVideo 1.5
----------------

Use ``--model-type hyvideo-15`` with a Diffusers HunyuanVideo 1.5
checkpoint. The published 480p T2V and I2V checkpoints are separate; the
checkpoint determines which pipeline is loaded. The defaults are 15 fps,
guidance 6, and 50 inference steps. ``FlowMatchEulerDiscreteScheduler`` is the
supported scheduler. Hunyuan also accepts ``--sigmas`` when specifying a custom
FlowMatch schedule; the number of sigmas determines the number of denoising
steps.

Hunyuan opening still
~~~~~~~~~~~~~~~~~~~~~

With no image seed, a prompt generates a clip from noise. With one still in
``--image-seeds``, that image is the opening frame. A video seed, more than
one reference, ``last-frame=``, and the Wan VACE keywords are rejected.
``--video-lengths`` is seconds. The duration becomes a frame count from
``--video-fps`` and the checkpoint VAE's temporal stride. Width and height
are aligned down to a multiple of 16. When ``--output-size`` is omitted, the
still supplies the aspect ratio, and the image is resized to the canvas.
The video VAE is tiled, so ``--vae-tiling`` is rejected. ``--vae-slicing``
is available. There is no soundtrack. ``--batch-size`` greater than 1 is
rejected.

``--transformer`` replaces the denoising transformer. The parent checkpoint
still supplies the VAE and text encoders. ``--quantizer-map`` can name the
transformer and the Hunyuan text components.

LoRA and offload
~~~~~~~~~~~~~~~~

``--loras`` loads a HunyuanVideo 1.5 LoRA onto the transformer, including
original-layout checkpoints and Musubi ``lora_unet_`` files. A ComfyUI quantized
transformer keeps those adapters active instead of fusing them into the packed
weights.

The examples use ``bfloat16``, sequential CPU offload, 50 steps, and an
``848x480`` canvas. Sequential offload reduces peak GPU memory by moving
components between CPU and GPU. ``--model-cpu-offload`` and
``--model-group-offload`` are also available. ``--torch-compile`` compiles
the repeated transformer blocks. The first run can take longer while
compilation happens. The Console UI has separate text-to-video and
image-to-video recipes.

* `Text-to-video <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/hyvideo_15/text_to_video/config.dgen>`_
* `4-step LoRA <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/hyvideo_15/lora/lightx2v-4step-config.dgen>`_
* `Image-to-video <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/hyvideo_15/image_to_video/config.dgen>`_
* `T2V GGUF transformer <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/hyvideo_15/gguf/text-to-video-config.dgen>`_

Motif-Video
-----------

Use ``--model-type motif-video`` with ``Motif-Technologies/Motif-Video-2B``.
The same repository supports text-to-video and first-frame image-to-video; the
mode selects the corresponding pipeline. Defaults are 24 fps, guidance 5, and
50 inference steps. Motif supports ``FlowMatchEulerDiscreteScheduler`` and
``DPMSolverMultistepScheduler``. ``--max-sequence-length`` controls the prompt
token limit (default 512).

Motif opening still
~~~~~~~~~~~~~~~~~~~

With no image seed, a prompt generates a clip from noise. With one still in
``--image-seeds``, that image is the opening frame. A video seed, more than
one reference, ``last-frame=``, and the Wan VACE keywords are rejected.
``--video-lengths`` is seconds. The duration becomes a frame count from
``--video-fps`` and the checkpoint VAE's temporal stride. Width and height
are aligned down to a multiple of 16. When ``--output-size`` is omitted, the
still supplies the aspect ratio, and the image is resized to the canvas.
The video VAE is tiled, so ``--vae-tiling`` is rejected. ``--vae-slicing``
is available. There is no soundtrack. ``--batch-size`` greater than 1 is
rejected.

``--transformer`` replaces the denoising transformer. The parent checkpoint
still supplies the VAE and text encoders. ``--quantizer-map`` can name the
transformer and the Motif text components. A GGUF transformer can be used
for either mode while the model path stays the parent Diffusers repository.

Timestep schedule
~~~~~~~~~~~~~~~~~

``--timesteps`` sets an explicit timestep schedule. Give one or more
comma-separated integer schedules, such as ``1000,700,300,0``, or ``expr:``
expressions evaluated with the scheduler's default array named ``timesteps``
(for example, ``expr: timesteps[::2]``). Multiple schedules are tried in turn.
The number of selected timesteps replaces ``--inference-steps`` for that run.
``--timesteps`` cannot be combined with ``--sigmas``. Both work with any model
type whose scheduler takes them, including pipelines that do not accept them
as call arguments; the schedule is applied at the scheduler. Motif's default
``DPMSolverMultistepScheduler`` takes timesteps only; select
``FlowMatchEulerDiscreteScheduler`` for ``--sigmas``.

LoRA and GGUF
~~~~~~~~~~~~~

``--loras`` loads Diffusers-format and Musubi Motif-Video LoRAs onto the
transformer. The standard examples use ``bfloat16``, sequential CPU offload,
50 steps, and ``832x480``. ``--model-cpu-offload``, ``--model-group-offload``,
and ``--torch-compile`` are available. The Console UI has separate
text-to-video and image-to-video recipes.

* `Text-to-video <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/motif_video/text_to_video/config.dgen>`_
* `Image-to-video <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/motif_video/image_to_video/config.dgen>`_
* `T2V GGUF transformer <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/motif_video/gguf/text-to-video-config.dgen>`_
* `GGUF as the model <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/motif_video/gguf/as-model-config.dgen>`_

MiniMax-H3
----------

Use ``--model-type minimax-h3-video`` with ``MiniMaxAI/MiniMax-H3``. One denoising
loop writes the video and a stereo soundtrack. There is no separate vocoder.
The checkpoint is guidance-distilled, so omit ``--guidance-scales`` and do not
pass a negative prompt. The only scheduler name is ``MiniMaxH3Scheduler``.
``--scheduler`` overlays the video scheduler, which starts at shift 12.
``--second-model-scheduler`` overlays the audio scheduler, which starts at
shift 3, and does not inherit ``--scheduler``. Omitting either keeps that
checkpoint scheduler. ``--inference-steps`` defaults to
30. That count includes the terminal sigma, so the model runs 29 evaluations.
``--sigmas`` is the grid before the shift, between 1 and 0. A terminal 0 is
appended when missing. The video and audio schedulers each shift that one
list by their own shift, so both keep the same step count. The expression
symbol ``sigmas`` is ``linspace(1, 0, steps)`` without the terminal 0, which
reproduces the default schedule. ``--timesteps`` is rejected.
``--spectrum`` forecasts the packed hidden state and skips the transformer
blocks on later steps. The video and audio output heads still run.
``ref2va`` forecasts ``transformer_ref``. A forecast step leaves a Fun control
branch idle, because the last real pass already included its hint.
See `examples/minimax_h3_video/spectrum/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/minimax_h3_video/spectrum/config.dgen>`_.

Frame snap
~~~~~~~~~~

The frame rate is 24. ``--video-lengths`` is seconds and snaps up to the next
``17n+5`` frames. A request can be 5 to 15 seconds. The clip comes out between
5.17 seconds (124 frames) and 14.38 seconds (345 frames). Omit the
option and the clip is 124 frames, 5.17 seconds. Height and width are multiples of 32. Omit
``--output-size`` and the canvas follows the first keyframe, or 16:9 when
there is no keyframe.

Text and keyframes
~~~~~~~~~~~~~~~~~~

* No image seed is text to video and audio. This loads ``transformer/``.
* One still image is the opening keyframe. ``last-frame=`` is the closing
  keyframe, alone or with the opening image. This also loads ``transformer/``.

See `examples/minimax_h3_video/text_to_video/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/minimax_h3_video/text_to_video/config.dgen>`_
and `examples/minimax_h3_video/keyframes/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/minimax_h3_video/keyframes/config.dgen>`_.

.. code-block:: shell

    dgenerate MiniMaxAI/MiniMax-H3 --model-type minimax-h3-video --dtype bfloat16 \\
        --model-sequential-offload --inference-steps 30 --video-lengths 5 \\
        --animation-format mp4 --output-size 1344x768 \\
        --prompts "A red fox trotting through a snowy pine forest, snow crunching underfoot"

    dgenerate MiniMaxAI/MiniMax-H3 --model-type minimax-h3-video --dtype bfloat16 \\
        --model-sequential-offload --video-lengths 5 --animation-format mp4 \\
        --output-size 1024x576 \\
        --image-seeds "examples/media/man-fighting-pose.jpg;last-frame=examples/media/beach.jpg" \\
        --prompts "The man in the opening frame walks onto the beach in the closing frame"

References
~~~~~~~~~~

``reference=`` is an ordered list of images, videos, and audio clips and
loads ``transformer_ref/``. At most 9 images, 3 videos, 3 audio clips, and
12 references in total. An audio reference has to be paired with an image
or a video. A keyframe and ``reference=`` cannot be combined. A video
reference keeps the frame rate and soundtrack reported by the file.
With neither ``--reference-image-processors`` nor ``resize=``, an image
reference is scaled to a 2048 short edge and a video reference is placed on
the 768 canvas. With either one, that frame size, snapped to a multiple of
32, is what the video VAE encodes. An audio reference is left unchanged.

See `examples/minimax_h3_video/references/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/minimax_h3_video/references/config.dgen>`_.

.. code-block:: shell

    dgenerate MiniMaxAI/MiniMax-H3 --model-type minimax-h3-video --dtype bfloat16 \\
        --model-sequential-offload --video-lengths 5 --animation-format mp4 \\
        --output-size 1024x576 \\
        --image-seeds "reference=examples/media/man-fighting-pose.jpg, examples/media/rickroll-roll.gif" \\
        --prompts "The man in the image dances with the motion in the video"

Fun control
~~~~~~~~~~~

``--control-nets`` loads one MiniMax-H3 Fun control branch on top of the text
to video transformer. This is not a Diffusers ControlNet. The published weights
are ``alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0`` with
``weight-name=MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors``. ``scale`` is
the control strength. A bare ``--image-seeds`` path is the control video, for
example canny, depth, or pose. ``"source.mp4;mask=mask.mp4"`` inpaints, and
``control=`` can be added. White mask pixels are regenerated. Omit
``--output-size`` and the canvas follows that video, snapped down to a multiple
of 32. The branch was trained on text to video, so a keyframe or ``reference=``
cannot be combined with it.

``latents=`` replaces the video noise. ``audio-latents=`` replaces the audio noise.

See `examples/minimax_h3_video/controlnet/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/minimax_h3_video/controlnet/config.dgen>`_.

.. code-block:: shell

    dgenerate MiniMaxAI/MiniMax-H3 --model-type minimax-h3-video --dtype bfloat16 \\
        --model-sequential-offload --video-lengths 5 --animation-format mp4 \\
        --output-size 1024x576 \\
        --control-nets "alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0;weight-name=MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors" \\
        --control-image-processors "openpose;include-hand=true;include-face=true" \\
        --image-seeds examples/media/rickroll-roll.gif \\
        --prompts "A person dancing, following the pose in the control video"

Weights
~~~~~~~

``--transformer`` replaces the denoiser and leaves Qwen3-VL and the VAEs on
the model path. The file can be Diffusers weights, a ComfyUI safetensors file,
or a GGUF. A name containing ``fl2va`` is the text and keyframe transformer.
A name containing ``ref2va`` is the reference transformer.

``--loras`` loads a LoRA onto the transformer for the workflow above. On a
full-precision checkpoint it is fused into the weights. With ``--quantizer``,
pass ``--lora-disk-fuse`` so that fuse is cached before quantization. A GGUF
or Comfy ``--transformer`` keeps the LoRA on the packed weights.

The examples use sequential CPU offload. ``--model-cpu-offload``,
``--model-sequential-offload``, and ``--model-group-offload`` place the text
encoder and transformer the same way: each visits the GPU for its forward
and returns to CPU.

Prompt rewrite
~~~~~~~~~~~~~~

``--pipeline-prompt-enhance`` rewrites the prompt with that Qwen3-VL encoder
before the prompt is encoded. The rewrite stays in the language of the prompt.
The opening frame is shown as Picture 1 and the last frame as Picture 2.
A last frame alone is Picture 1. A reference list is shown in order:
an image is Picture N, a video is Video N shown as its opening frame, and
audio is Audio N with no picture. The rewrite keeps those labels. An argument value
can replace the built-in system prompt, see ``--help`` for usage details.

Kandinsky 5
------------

Use ``--model-type kandinsky5-video`` with a Kandinsky 5.0 video checkpoint.
Text-to-video checkpoints include Lite and Pro variants; image-to-video
checkpoints are published separately. Choose a T2V or I2V repository for the
intended mode. Defaults are 24 fps, guidance 5, and 50 inference steps, with
``FlowMatchEulerDiscreteScheduler``. The common 5-second checkpoints generate
121 frames at 24 fps. T2V Lite also has 10-second checkpoints; match the
requested duration to the duration for which the checkpoint was trained.

Kandinsky opening still
~~~~~~~~~~~~~~~~~~~~~~~

With no image seed, a prompt generates a clip from noise. With one still in
``--image-seeds``, that image is the opening frame. A video seed, more than
one reference, ``last-frame=``, and the Wan VACE keywords are rejected.
``--video-lengths`` is seconds. The duration becomes a frame count from
``--video-fps`` and the checkpoint VAE's temporal stride. Width and height
are aligned down to a multiple of 16. When ``--output-size`` is omitted, the
still supplies the aspect ratio, and the image is resized to the canvas.
The video VAE is tiled, so ``--vae-tiling`` is rejected. ``--vae-slicing``
is available. There is no soundtrack. ``--batch-size`` greater than 1 is
rejected.

Distilled checkpoints
~~~~~~~~~~~~~~~~~~~~~

Checkpoint variants can have different guidance and step expectations. The
16-step distilled T2V Lite checkpoint should be run with 16 inference steps and
guidance 1, as should the distilled Pro checkpoints; no-CFG checkpoints should
also use guidance 1. The GGUF example uses a Q4_K_S quantization of
the T2V Pro sft transformer. ``--torch-compile`` and Cache-DiT can be
used to reduce denoising cost. The examples use sequential CPU offload; the
other model offload modes are available as well.

``--transformer`` replaces the denoising transformer. The parent checkpoint
still supplies the VAE and text encoders. ``--quantizer-map`` can name the
transformer and the Kandinsky 5 text components. GGUF is demonstrated for
text-to-video. The Console UI has separate text-to-video and image-to-video
recipes.

* `Text-to-video <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky5/text_to_video/config.dgen>`_
* `Image-to-video <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky5/image_to_video/config.dgen>`_
* `T2V GGUF transformer <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky5/text_to_video_gguf/config.dgen>`_

Camera LoRAs
~~~~~~~~~~~~

LoRAs are supported through ``--loras``. The examples include camera-orbit and
object-rotation LoRAs for T2V Lite/Pro and I2V Lite/Pro; preserve each LoRA's
trigger phrase in the prompt.

* `Kandinsky 5 video LoRA examples <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/kandinsky5/lora>`_

.. _kandinsky-6:

Kandinsky 6
-----------

Use ``--model-type kandinsky6-video`` with a Kandinsky 6.0 checkpoint. One
pipeline writes the picture and a 44.1 kHz soundtrack. Text-to-video and
first-frame image-to-video use that same repository. There is no separate
image-to-video checkpoint. A Kandinsky 6 super-resolution repository is
rejected. ``--loras`` is rejected: no Kandinsky 6 LoRA loader or checkpoint
is published.

Text and first frame
~~~~~~~~~~~~~~~~~~~~

No image seed is text to video. One still image is the first frame. A mask,
a second image, a video seed, ``last-frame=``, ``control=``, and
``reference=`` are rejected.

``--video-fps`` defaults to 24. ``--video-lengths`` is seconds and snaps to
``temporal*k+1`` frames. The video VAE's temporal compression is 4, so five
seconds is 121 frames, the length these checkpoints were trained for. Width
and height are aligned down to a multiple of 16. The examples use
``864x480``. Omit ``--video-lengths`` and the pipeline chooses the length.
The soundtrack is muxed into an mp4. GIF, WebP, and
``--animation-format frames`` drop that audio. There is no separate audio
prompt or audio guidance scale. Describe the sound in the same prompt.

.. code-block:: shell

    dgenerate kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers \\
        --model-type kandinsky6-video --dtype bfloat16 \\
        --model-sequential-offload --inference-steps 10 --guidance-scales 1 \\
        --video-lengths 5 --video-fps 24 --animation-format mp4 \\
        --output-size 864x480 \\
        --prompts "A red kite glides above a windswept coastal meadow at golden hour. Audio: wind and distant surf."

    dgenerate kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers \\
        --model-type kandinsky6-video --dtype bfloat16 \\
        --model-sequential-offload --inference-steps 10 --guidance-scales 1 \\
        --image-seeds examples/media/man-fighting-pose.jpg \\
        --video-lengths 5 --video-fps 24 --animation-format mp4 \\
        --output-size 864x480 \\
        --prompts "The person slowly lowers their arms and turns toward the camera. Audio: cloth movement, a quiet room."

* `Text-to-video, Lite distilled <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/text_to_video/config.dgen>`_
* `Text-to-video, Lite <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/text_to_video/full-config.dgen>`_
* `Text-to-video, Pro distilled <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/text_to_video_pro/config.dgen>`_
* `Image-to-video <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/image_to_video/config.dgen>`_

Scheduler and checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~

``--scheduler`` accepts ``FlowMatchEulerDiscreteScheduler`` and
``PiflowScheduler``. Omitting it keeps the checkpoint scheduler. A distilled
checkpoint ships ``PiflowScheduler``. When that scheduler is loaded, omitted
guidance is 1 and omitted steps are 10. Any other loaded scheduler uses
guidance 5 and 50 steps. Values you set are used as written. Guidance 1 is
unguided.

* Lite distilled, public: ``kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers``. 10 steps, guidance 1.
* Lite, public: ``kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers``. 50 steps, guidance 5.
* Pro distilled, public: ``kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers``. 10 steps, guidance 1. The transformer is about 60 GB.
* Pro, gated: ``kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers``. 50 steps, guidance 5.

Replacement modules
~~~~~~~~~~~~~~~~~~~

``--transformer`` replaces the denoiser. The repository still supplies
Qwen2.5-VL, CLIP, the video VAE, the audio VAE, and the vocoder. A GGUF or a
Comfy safetensors file loads that way. The Pro Q4_K_S GGUF stays on the gated
Pro repository, so that config needs a Hugging Face token.
``--text-encoders`` replaces Qwen2.5-VL in the first slot. ``+`` keeps CLIP
from the repository. A packed quant stays packed: do not put a quantizer on
that URI. ``--quantizer-map`` may name ``transformer``, ``text_encoder``, or
``text_encoder_2``.

* `Pro GGUF transformer <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/gguf/pro-config.dgen>`_
* `Pro GGUF text encoder <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/gguf/text-encoder-config.dgen>`_
* `Lite distilled int8 transformer <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/comfy_quant/lite-distill-config.dgen>`_

Spectrum and DyPE
~~~~~~~~~~~~~~~~~

``--spectrum`` forecasts the fused video and audio blocks. A skipped step
forecasts the audio stream from the cache-point block along with the video
feature. ``--dype`` scales the height and width position axes and leaves the
time axis on the checkpoint table. The reference long side defaults to 864.
``--cache-dit`` and ``--torch-compile`` are available. The examples use
sequential CPU offload. ``--model-cpu-offload`` and ``--model-group-offload``
are available as well. The video VAE is tiled already, so ``--vae-tiling`` is
rejected. ``--vae-slicing`` is available. ``--batch-size`` greater than 1 is
rejected. ``--timesteps`` sets an explicit timestep schedule.

* `Spectrum <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/spectrum/lite-config.dgen>`_
* `DyPE <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/dype/lite-config.dgen>`_

Prompt expansion
~~~~~~~~~~~~~~~~

``--pipeline-prompt-enhance`` turns on the pipeline's ``expand_prompts``
rewrite. With no value, text-to-video and image-to-video each keep their own
expansion instruction. A value replaces that instruction for the clip. The
prompt is still appended as ``Rewrite Prompt:``. Braces in the value stay
literal. A first frame is shown to the rewrite. The
``--pipeline-prompt-enhance-*`` options apply to that rewrite. The video still
uses ``--seeds`` or ``--gen-seeds``. ``--prompt-upsampler`` is rejected: the
rewrite uses the loaded Qwen2.5-VL encoder. When ``--max-sequence-length`` is
omitted, the rewrite uses 1024 tokens. The same system-prompt value is
described under `Decode, prompts, and duration`_.

* `Prompt rewrite <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/pipeline_prompt_enhance/text-to-video-config.dgen>`_
* `Library usage <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/library_usage/kandinsky6_video/main.py>`_

The Console UI recipes cover Lite, Pro, the GGUF transformer, and the int8
transformer, each with and without an image seed.

LTX
---

``--model-type ltx2`` and ``ltx-video`` each write one clip per pipeline call.
One combination of prompt, seed, guidance, steps, image seed, ``--video-lengths``,
``--video-fps``, ``--ltx-audio-guidance-scales``, and ``--ltx-audio-guidance-rescales``
writes one animation file. LTX does not run once per input frame.

``--ltx-audio-guidance-scales`` and ``--ltx-audio-guidance-rescales`` are the audio
CFG and audio rescale. Both are combinatorial, the same way ``--prompts`` and
``--seeds`` are. The frame count inside a clip is not a separate product factor.

Audio from LTX-2.5 is muxed into an mp4. GIF, WebP, and ``--animation-format frames``
drop that audio. LTX-Video writes frames only. Conditioning stays in ``--image-seeds``.

LTX-2.5
~~~~~~~

``--model-type ltx2`` loads ``Lightricks/LTX-2.5-Diffusers``.

The same model type also runs the original 19B checkpoint,
``Lightricks/LTX-2``. Use that repository for a named 19B, LTX-2.0, or
Gemma 3 checkpoint. The 19B repository supplies Gemma 3, the 2-layer text
connectors, and the video and audio VAEs. It has no duration head. Its
scheduler already uses dynamic shifting. A Comfy GGUF of the 19B dev or
distilled transformer is ``--transformer``, or the model path. The file is
told apart from LTX-2.5 by the absence of ``keyframes_abs_pos_embedding``
and ``prompt_adaln_single``. The distilled 19B file needs
``--scheduler FlowMatchEulerDiscreteScheduler;use-dynamic-shifting=false``
so the 8-value sigma table is used. See
`examples/ltx2/gguf/ltx2-dev-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/gguf/ltx2-dev-config.dgen>`_
and
`examples/ltx2/gguf/ltx2-distilled-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/gguf/ltx2-distilled-config.dgen>`_.

* No image seed is text to video. Omitting ``--video-lengths`` lets the model's duration head choose the length.
* One image is the first frame.
* ``last-frame=`` is the last frame. A first frame and ``last-frame=`` can be used together.
* ``ltx-index`` and ``strength`` place a condition on a chosen latent frame.
  See `Condition placement`_.
* Either slot can be a video or animated image instead of a still. See `Video conditioning`_.
* With ``--ltx-ic-lora``, a plain path is instead the reference clip for an IC-LoRA, such as canny, depth, or pose control. See `IC-LoRA control`_.
* ``images:`` is rejected.

Width and height are rounded down to a multiple of 32, with a warning. ``--video-fps`` defaults to 24.
``--video-lengths`` snaps to ``8k+1`` frames at that rate, the same rule as a
conditioning clip.

``--model-sequential-offload``, ``--model-cpu-offload``, and ``--model-group-offload``
work the same way they do for image models. ``--torch-compile`` compiles the
repeated transformer blocks. The first denoising step waits while the graph
builds. A GGUF, SDNQ, or bitsandbytes
transformer follows the placement described under `Precision, GGUF, and offload`_. The examples under `examples/ltx2 <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx2>`_
use the published
repository as-is. ``--ltx-latent-upscale`` runs the two-stage sampler in that
same generation: a half-resolution pass, the checkpoint latent upsampler, then
a short refine at ``--output-size``. ``--transformer`` is
described under `Submodels`_. ``model_index.json`` selects the pipeline: ``LTX2Pipeline``
is LTX-2.5, and ``LTXPipeline`` is :ref:`ltx-video-legacy`.

.. _ltx-video-legacy:

LTX-Video
~~~~~~~~~

``Lightricks/LTX-Video`` is the legacy 2B checkpoint. ``model_index.json`` selects
``LTXPipeline``. It writes a picture only. There is no soundtrack, no duration
head, and no IC-LoRA.

``--inference-steps`` and ``--guidance-scales`` are sent as written. The distilled
sigma table and the guidance rewrites under `Guidance, steps, and sigmas`_ apply
to LTX-2.5 only. The published configs use 50 steps, guidance ``3``,
``--video-fps 25``, and ``--video-lengths 4.84``, which is 121 frames,
at ``768x512``. Omitting ``--video-lengths`` uses the pipeline default of
161 frames. dgenerate warns when the clip is shorter than 121 frames or smaller
than about 704 by 480, because the model follows the prompt at the published size.

A prompt can name what to avoid after one ``;``. The examples use
``worst quality, inconsistent motion, blurry, jittery, distorted``.

The noise schedule shifts with the number of latent tokens. A size or length
past what that shift can represent is rejected. Use a smaller ``--output-size``
or a shorter clip.

A first frame, ``last-frame=``, an opening or closing clip, ``ltx-index``, and
``strength`` work as described under `Video conditioning`_ and `Condition placement`_.
``--loras`` loads a diffusers LoRA onto the transformer. ``--transformer`` replaces
that transformer, including a city96 ``.gguf`` file. The ``Lightricks/LTX-Video``
repository still supplies the VAE and text encoders. ``--scheduler`` accepts
``FlowMatchEulerDiscreteScheduler`` and its URI arguments.

These options are rejected: ``--ltx-audio-guidance-scales``,
``--ltx-audio-guidance-rescales``, ``--sigmas``, ``--ltx-ic-lora``,
``--ltx-latent-upscale``, ``--ltx-stg-scales``, ``--ltx-modality-scales``,
``--ltx-video-decoder diffusion``, ``--ltx-diffusion-decode-tile``, ``--pipeline-prompt-enhance``,
``--ltx-image-crfs``, ``--ltx-no-cross-timestep``,
``--ltx-video-min-seconds``, and ``--ltx-video-max-seconds``.

The Console UI recipes are ``LTX-Video`` and ``LTX-Video (GGUF)``.

* `examples/ltx_video/basic/text-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx_video/basic/text-to-video-config.dgen>`_
* `examples/ltx_video/image_conditioning/first-frame-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx_video/image_conditioning/first-frame-config.dgen>`_
* `examples/ltx_video/video_conditioning/video-extension-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx_video/video_conditioning/video-extension-config.dgen>`_
* `examples/ltx_video/lora/crush-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx_video/lora/crush-config.dgen>`_
* `examples/ltx_video/gguf/text-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx_video/gguf/text-to-video-config.dgen>`_
* `examples/ltx_video/gguf/first-frame-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx_video/gguf/first-frame-config.dgen>`_
* `examples/ltx_video/gguf/text-encoder-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx_video/gguf/text-encoder-config.dgen>`_

Video conditioning
~~~~~~~~~~~~~~~~~~

The main ``--image-seeds`` path and ``last-frame=`` each accept a video or an animated image
as well as a still. Both LTX-2.5 and the earlier LTX-Video accept this. A file with a
single frame is treated as a still.

* A video in the main path is the opening clip. The generated clip continues from it.
* A video in ``last-frame=`` is the closing clip. The generated clip leads into it.
* A still and a video can be mixed, for example a video first and a still ``last-frame=``.

.. code-block:: bash

    # continue an existing clip
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx2 \
    --video-fps 25 --video-lengths 4 --output-size 512x512 \
    --image-seeds "input.gif" \
    --prompts "The singer keeps swaying, then points at the camera."

    # lead into an existing clip
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx2 \
    --video-fps 25 --video-lengths 4 --output-size 512x512 \
    --image-seeds ";last-frame=input.gif" \
    --prompts "A singer walks in and starts to sway at the microphone."

    # frames 16 through 40 of a clip, then a still last frame
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx2 \
    --video-fps 25 --video-lengths 4 --output-size 512x512 \
    --image-seeds "input.gif;frame-start=16;frame-end=40;last-frame=last.png" \
    --prompts "The scene fades into a pencil sketch of mountains."

``--frame-start`` and ``--frame-end``, or ``frame-start=`` and ``frame-end=`` in the seed,
choose which frames of the video are used. The slice applies to both the main path and
``last-frame=``. A slice in the seed overrides the global options, as described under
`Animation Slicing`_.

Frame counts:

* Each conditioning clip is cut to a frame count of ``8k+1``, the same rule as the
  output length. A 54 frame gif gives 49 conditioning frames.
* A clip longer than the output is cut to fit. With ``--video-lengths`` set, only the
  frames that can be used are decoded. Without it the whole slice is decoded, so use
  ``--frame-end`` on long files.
* When there is an opening and a closing condition, the opening clip is shortened so
  the two do not overlap.
* A closing video needs a fixed output length. LTX-2.5 picks its own length when
  ``--video-lengths`` is omitted, so set ``--video-lengths`` when ``last-frame=`` is a video.
  An opening video works either way.

Frames are used as they are and are not resampled. When the file's frame rate differs
from ``--video-fps`` dgenerate prints a warning, because motion will play faster or
slower. Set ``--video-fps`` to the file's rate to keep the speed.

``resize=``, ``aspect=``, and ``align=`` in the seed apply to every conditioning frame.

``--seed-image-processors`` runs on every frame of the main path and of ``last-frame=``. Give it two
chains separated by ``+`` to process them differently. The first chain runs on the main path and
the second on ``last-frame=``. A leading or trailing ``+`` leaves one side unprocessed.
``--last-frame-image-processors`` is a dedicated chain for ``last-frame=`` and overrides the second
seed chain when both are set.

.. code-block:: bash

    # grayscale first frame, original last frame
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx2 \
    --video-lengths 3 --output-size 512x512 \
    --image-seeds "painting.png;last-frame=painting.png" \
    --seed-image-processors grayscale + \
    --prompts "A black and white painting slowly fills with warm color."

    # process only last-frame=
    --seed-image-processors + "canny;lower=50;upper=100"

An opening clip is held at full strength unless that path sets ``strength`` or
``--image-seed-strengths``. ``last-frame=`` stays at strength 1. The model keeps
those frames and generates around them. It does not restyle the whole input video.
Partial strength on a chosen latent frame is described under `Condition placement`_.

See `examples/ltx2/video_conditioning <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx2/video_conditioning>`_,
`examples/ltx2/image_conditioning <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx2/image_conditioning>`_,
and `examples/ltx_video/video_conditioning/video-extension-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx_video/video_conditioning/video-extension-config.dgen>`_.

IC-LoRA control
~~~~~~~~~~~~~~~

An IC-LoRA (in-context LoRA) guides LTX-2.5 with a reference video, for example canny
edges, a depth map, or a pose skeleton. Load it with ``--ltx-ic-lora``. The reference clip
comes from ``--image-seeds``, and ``--control-image-processors`` turns it into the signal
the IC-LoRA expects. The reference frames guide the output and are not placed in it.

With ``--ltx-ic-lora``, a plain seed path is the reference clip, the same way a plain path is the
control image when ``--control-nets`` is given to an image model. To add a first or last
frame, put the reference in ``control=``:

.. code-block:: bash

    # reference clip only
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx2 \
    --ltx-ic-lora "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control;weight-name=ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
    --video-fps 25 --video-lengths 4 --output-size 512x512 \
    --image-seeds "input.gif" \
    --control-image-processors "canny;lower=50;upper=100" \
    --prompts "A man in a shiny silver suit sings at a vintage microphone."

    # first frame plus reference clip
    --image-seeds "first.png;control=input.gif"

* ``--ltx-ic-lora`` takes the same URI as ``--loras``, plus ``attention`` and ``downscale``.
  ``scale`` is the LoRA weight. ``attention``, from 0 to 1, is how strongly the generated
  video attends to the reference, 1 by default.
* ``--loras`` can be used at the same time, for example a style LoRA. All of them are fused
  into the transformer together, and ``--lora-fuse-scale`` applies to all of them.
* Lightricks publishes IC-LoRAs for the distilled checkpoint, which is what
  ``Lightricks/LTX-2.5-Diffusers`` loads by default.
* The reference is one video, animated image, or still. It uses the same frame slicing,
  ``resize=``, and frame count rules as `Video conditioning`_.
* Without ``--video-lengths`` the output is as long as the reference clip, cut to ``8k+1``
  frames. With ``--video-lengths`` a longer reference is cut to the output length.
* Some IC-LoRAs read the reference at a reduced size. dgenerate reads
  ``reference_downscale_factor`` from the IC-LoRA's safetensors metadata, and ``downscale``
  in the URI overrides it. The union control LoRA uses 2, so the output width and height
  are rounded down to a multiple of 64.
* ``--ltx-ic-lora`` needs an LTX-2 checkpoint. The earlier LTX-Video pipeline rejects it.

See the examples in `examples/ltx2/ic_lora <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx2/ic_lora>`_.
`canny-anime-lora-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/ic_lora/canny-anime-lora-config.dgen>`_
and `depth-realism-lora-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/ic_lora/depth-realism-lora-config.dgen>`_
add a style LoRA from ``--loras`` to the IC-LoRA.

In the Console UI, the LTX-2.5 recipes under ``Edit -> Insert Code -> Recipe`` have an IC-LoRA
field, an IC-LoRA control clip, and a control clip processor. ``Edit -> Insert URI -> Sub Model URI``
builds an ``--ltx-ic-lora`` URI, and ``Edit -> Insert URI -> Image Seed URI`` accepts a last frame or
closing clip for ``last-frame=``. The LTX recipes also take a separate processor for the first and last frame.

Guidance, steps, and sigmas
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loaded scheduler decides which path runs. dgenerate does not pick a path from
the repository name or a ``--transformer`` subfolder. ``--transformer`` replaces
weights only. ``--scheduler`` accepts ``FlowMatchEulerDiscreteScheduler`` and URI
arguments on that class, such as ``shift`` and ``use-dynamic-shifting``. Those
arguments overlay the checkpoint scheduler config. Other scheduler names are
rejected. Omitting ``--scheduler`` keeps the checkpoint scheduler.
``--scheduler help`` and ``--scheduler helpargs`` still print help.

When you omit ``--guidance-scales`` or ``--inference-steps``, generation-step text
shows **auto**. At run time the video pipeline reads the loaded pipeline's
``__call__`` defaults, then the model-specific rules below, then global fallbacks
(not during config ``check`` or ``fill_defaults``).
For ``--model-type ltx-video``, omitted values follow ``LTXPipeline`` (published
Diffusers checkpoints typically use guidance ``3`` and ``50`` steps). Values you
set are passed as written.

For ``--model-type ltx2``, omitted guidance becomes ``1`` on a distilled checkpoint,
or video ``3`` with audio ``7`` on a full checkpoint. Values you set are used as
written. An omitted step count uses the pipeline default (``30`` on full
``LTX2Pipeline`` when neither ``--timesteps`` nor ``--sigmas`` is set). A
distilled checkpoint with an omitted step count runs its 8-value sigma table.

**Scheduler without dynamic shifting (the published LTX-2.5 scheduler)**

* The clip uses Diffusers' distilled 8-value sigma table. Another ``--inference-steps``
  count linearly resamples that table to that many steps, keeping its shape.
* Omitted guidance is ``1`` (unguided). Any explicit guidance value is used as written.
* Video guidance and audio guidance receive that same number unless
  ``--ltx-audio-guidance-scales`` is set.
* After the clip runs, the written config records 8 steps and the guidance that
  was actually used.

**Scheduler with dynamic shifting**

* ``--inference-steps`` is sent as ``num_inference_steps`` when neither
  ``--timesteps`` nor ``--sigmas`` is set. Omitted steps use ``30`` on a full
  checkpoint.
* Omitted guidance becomes video ``3`` and audio ``7``, unless
  ``--ltx-audio-guidance-scales`` is set. Explicit values are used for video, and
  for audio unless ``--ltx-audio-guidance-scales`` is set.

``--sigmas`` (CSV list or ``expr:``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This path wins over both of the above.

* A CSV list is the schedule. An ``expr:`` expression is evaluated against the
  distilled table when the scheduler has no dynamic shifting, or against
  ``scheduler.set_timesteps(--inference-steps)`` when it does.
* The step count becomes the length of the resulting list. ``--inference-steps``
  is overwritten to match, including in the written config.
* ``--guidance-scales`` is used as written. Video and audio guidance get the same
  value unless ``--ltx-audio-guidance-scales`` is set.

``--sigmas`` is combinatorial with ``--guidance-scales``, ``--inference-steps``,
``--guidance-rescales``, ``--ltx-audio-guidance-scales``,
``--ltx-audio-guidance-rescales``, ``--video-lengths``, and ``--video-fps``. See
:ref:`specifying-sigmas` and `examples/ltx2/sigmas/sigmas-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/sigmas/sigmas-config.dgen>`_.

``--guidance-rescales``
^^^^^^^^^^^^^^^^^^^^^^^

LTX accepts this. A value you set is sent as video guidance rescale, and as
audio rescale unless ``--ltx-audio-guidance-rescales`` is set. If you omit it,
the pipeline keeps its own default (``0.7``). The rescale only applies while
classifier-free guidance is on (guidance greater than 1).

``--max-sequence-length``
^^^^^^^^^^^^^^^^^^^^^^^^^

LTX accepts this as Gemma's prompt token budget, from 1 to 1024. If you omit
it, the pipeline keeps 1024.

``--vae-slicing``
^^^^^^^^^^^^^^^^^

LTX accepts this on the video VAE and the audio VAE. ``--vae-tiling`` is
rejected: the video VAE is always tiled.

Spatio-temporal guidance
^^^^^^^^^^^^^^^^^^^^^^^^

``--ltx-stg-scales`` is spatio-temporal guidance. ``0`` turns it off. When it is on
and ``--ltx-stg-blocks`` is omitted, block ``28`` is used.
``--ltx-modality-scales`` is modality-isolation guidance. ``1`` turns it off.
``--ltx-audio-stg-scales`` and ``--ltx-audio-modality-scales`` copy the video values
when omitted. ``--ltx-no-cross-timestep`` selects the LTX-2.0 cross-modality timestep.
LTX-2.3 and LTX-2.5 leave that timestep on.

A full-transformer one-pass call uses video guidance ``3``, audio guidance ``7``,
STG ``1``, and modality guidance ``3``. The distilled checkpoint stays at guidance ``1``
and does not use those extra terms.

Soundtrack
~~~~~~~~~~

One prompt drives both the picture and the soundtrack. Gemma encodes that
string once; the text connectors split the packed hidden states into video
tokens and audio tokens. There is no audio-only prompt argument, so
``--second-prompts`` cannot be a soundtrack prompt.

Audio CFG is a separate pipeline scale. ``--ltx-audio-guidance-scales`` sets it
and is combinatorial. Omit that option and audio copies the video guidance
that will be sent. On a full checkpoint with omitted video guidance, video uses
``3`` and audio ``7``.

``--ltx-audio-guidance-rescales`` is the matching combinatorial audio rescale.
Omit it and audio copies ``--guidance-rescales`` when that is set, otherwise
the pipeline default ``0.7`` is left in place.

Diffusers suggests keeping audio guidance higher than video guidance when
you set them yourself. See `examples/ltx2/audio/audio-guidance-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/audio/audio-guidance-config.dgen>`_.

Clip chaining
~~~~~~~~~~~~~

``last_images`` and ``last_animations`` work the same way they do for image models. A still written by an
earlier invocation can be the first frame of an LTX clip, and an animation written by an earlier
invocation can be its opening clip.
The pipeline cache counts the video checkpoint and moves the previous pipeline back to CPU before the clip runs.

Submodels
~~~~~~~~~

``--unet`` is rejected. ``--vae`` and ``--text-encoders`` work like they do
for image models, including ``+`` to keep a slot's checkpoint default.
``--vae`` accepts ``AutoencoderKLLTX2Video`` (and ``AutoencoderKLLTXVideo``
for LTX-Video). ``--transformer`` and ``--loras`` do.

``--quantizer`` quantizes the diffusion transformer, the text encoder, and the text
connectors. ``--quantizer-map`` can limit that to ``transformer``, ``text_encoder``, or
``connectors``. The unused prompt-enhancer Gemma is not loaded. ``--transformer`` replaces
the diffusion transformer and accepts the same URI as it does for Flux, including
``subfolder``, ``dtype``, and ``quantizer``. A quantizer on that URI wins over ``--quantizer``.
Replacing the transformer does not change the scheduler.

The repository's default transformer is distilled. The full (non-distilled) transformer is
in its ``transformer_full`` subfolder. Load it with
``--transformer "Lightricks/LTX-2.5-Diffusers;subfolder=transformer_full"`` and give it a
normal schedule with
``--scheduler "FlowMatchEulerDiscreteScheduler;use-dynamic-shifting=true;shift-terminal=0.1"``,
otherwise it runs the distilled 8-value sigma table. Set ``--inference-steps`` and the guidance
stack for a guided model, for example 30 steps, ``--guidance-scales 3``,
``--ltx-audio-guidance-scales 7``, ``--ltx-stg-scales 1``, ``--ltx-modality-scales 3``,
and ``--ltx-stg-blocks 28``.
Lightricks IC-LoRAs are trained on the distilled transformer.
See `examples/ltx2/full_transformer/text-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/full_transformer/text-to-video-config.dgen>`_.

``--loras`` loads diffusers-format adapters onto that transformer and fuses them, including
``--lora-fuse-scale`` and each URI ``scale``. IC-LoRAs load with ``--ltx-ic-lora`` instead, see `IC-LoRA control`_.
See `examples/ltx2/lora/cinemagraph-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/lora/cinemagraph-config.dgen>`_
and `examples/ltx_video/lora/crush-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx_video/lora/crush-config.dgen>`_.

Two-stage generation
^^^^^^^^^^^^^^^^^^^^

``--ltx-latent-upscale`` runs both stages in the same generation, the way
``--sdxl-refiner`` follows a base pass. ``--output-size`` is the finished clip,
and both sides are rounded down to a multiple of 64. Stage 1 denoises at half of that size
with the main guidance, steps, and ``--sigmas``. The checkpoint ``latent_upsampler``
doubles the video latents. Stage 2 refines at the full size with the published
3-value sigma table. ``--ltx-noise-scales`` defaults to the first of those sigmas.
Stage 2 guidance defaults to 1. ``--ltx-stage-sigmas``, ``--ltx-stage-guidance-scales``,
and ``--ltx-stage-audio-guidance-scales`` override the refine pass.
Passing a URI to ``--ltx-latent-upscale`` selects a separate upsampler; its URI
accepts ``revision``, ``variant``, and ``subfolder``. For example,
``Lightricks/LTX-2.5-Diffusers;subfolder=latent_upsampler``. The upsampler uses
the main model's dtype.

The distilled checkpoint needs no stage LoRA. See
`examples/ltx2/two_stage/image-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/two_stage/image-to-video-config.dgen>`_.

The full transformer does. Load ``transformer_full``, give stage 1 dynamic shifting
and ``shift-terminal=0.1``, and put the distilled LoRA on the refine pass only:

``--ltx-stage-loras "Lightricks/LTX-2.5-Diffusers;weight-name=ltx-2.5-22b-distilled-lora-450-bf16.safetensors"``

See `examples/ltx2/two_stage/full-transformer-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/two_stage/full-transformer-config.dgen>`_.

Decode, prompts, and duration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--ltx-video-decoder diffusion`` decodes with the checkpoint diffusion decoder
instead of the convolutional VAE, in the same generation. ``conv`` is the default.
``--ltx-diffusion-decoder`` can select a separate decoder repository or local folder;
its URI accepts ``revision``, ``subfolder``, and ``dtype``. For example,
``Lightricks/LTX-2.5-Diffusers;subfolder=diffusion_decoder;dtype=bfloat16``.
When omitted, the decoder is loaded from the main checkpoint's ``diffusion_decoder`` folder.
Linux uses NATTEN through the ``kernels`` package. The first decode downloads
that kernel from the Hub, so leave ``DIFFUSERS_DISABLE_REMOTE_CODE`` unset.
Windows has no NATTEN build. That window runs with Triton when it is
installed, including the ``triton-windows`` extra. Without Triton the same
window is gathered in PyTorch. The FlexAttention path
builds a neighborhood mask that does not fit in GPU memory at video resolution.
``--ltx-diffusion-decode-tile`` sets the diffusion-decoder tile, for example ``384x384x24``.
Omit it and the missing-NATTEN path uses that size. NATTEN stays untiled unless the option is set.
``--ltx-decode-timesteps`` and ``--ltx-decode-noise-scales`` are the decode arguments.
See `examples/ltx2/decode/diffusion-decoder-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/decode/diffusion-decoder-config.dgen>`_.

``--pipeline-prompt-enhance`` turns on the pipeline's prompt enhancer.
LTX-2.5 uses ``google/gemma-4-E2B-it``. With no value, it selects the
text-to-video or image-to-video system prompt. A value replaces that prompt.
Any other ``--pipeline-prompt-enhance-*`` option turns enhancement on with
that built-in prompt when ``--pipeline-prompt-enhance`` itself is omitted.

FLUX.3 and Kandinsky 6 take the same system-prompt value and the same
sampling options. With no value, FLUX.3 keeps the upsampler's system prompt
and decodes greedily. Kandinsky 6 keeps its expansion instruction. Kandinsky 6
rejects ``--prompt-upsampler`` and rewrites with the loaded Qwen2.5-VL
encoder. See `Kandinsky 6`_.

``--prompt-upsampler`` names a different upsampler and turns enhancement on
by itself. The model is a Transformers causal or image-to-text model. The
denoiser stays on ``--transformer``. The URI arguments match
``--transformer``: ``revision``, ``variant``, ``subfolder``, ``dtype``, and
``quantizer``. FLUX.3 loads its Qwen3.5 prompt upsampler with this option.

``weight-name`` selects one ``.gguf`` or ``.safetensors`` file. ``mmproj`` is
the llama.cpp vision-tower GGUF loaded beside a language GGUF. Give that
file's path or its URL. A bare file name is not searched for beside the
language file or in its repository.

A repository load uses that repository's config and processor. A GGUF or
safetensors quant uses a configuration dgenerate ships for that shape. A
``config.json`` next to the quant is not read. ``mmproj`` is only the file
named on the URI.

A quant of this model loads when the file keeps the original names, when it
was written with ``--llama``, or when it is a Comfy safetensors file of those
names. ComfyUI-GGUF has no Gemma 4 architecture, so a Gemma enhancer GGUF is
for dgenerate and llama.cpp. FLUX.3's diffusion text encoder is Qwen3-VL 4B.
The upsampler here is the Qwen3.5 model.

LTX-2.5's built-in enhancer, including text-to-video, image-to-video, and a
quantized upsampler:

`examples/ltx2/basic/prompt-enhancer-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/basic/prompt-enhancer-config.dgen>`_,
`examples/ltx2/pipeline_prompt_enhance/text-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/pipeline_prompt_enhance/text-to-video-config.dgen>`_,
`examples/ltx2/pipeline_prompt_enhance/i2v-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/pipeline_prompt_enhance/i2v-config.dgen>`_,
and `examples/ltx2/pipeline_prompt_enhance/quantized-upsampler-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/pipeline_prompt_enhance/quantized-upsampler-config.dgen>`_.

A GGUF of that Gemma enhancer, as two blob links for the language file and ``mmproj``:

`examples/ltx2/pipeline_prompt_enhance/gguf-upsampler-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/pipeline_prompt_enhance/gguf-upsampler-config.dgen>`_.

FLUX.3's published Qwen3.5 upsampler, quantized the same way:

`examples/flux3/pipeline_prompt_enhance/quantized-upsampler-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux3/pipeline_prompt_enhance/quantized-upsampler-config.dgen>`_.

A custom system prompt and the sampling options:

`examples/flux3/pipeline_prompt_enhance/sampling-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux3/pipeline_prompt_enhance/sampling-config.dgen>`_
and
`examples/kandinsky6/pipeline_prompt_enhance/sampling-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/kandinsky6/pipeline_prompt_enhance/sampling-config.dgen>`_.

``--ltx-image-crfs`` recompresses a conditioning still before the VAE encode. Omit it
and the pipeline default is used (``18`` on LTX-2.5). ``0`` skips recompression.
See `examples/ltx2/image_conditioning/image-crf-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/image_conditioning/image-crf-config.dgen>`_.

``--ltx-video-min-seconds`` and ``--ltx-video-max-seconds`` clamp the duration head.
They apply only when ``--video-lengths`` is omitted. Give the same number of
values to each. The value in each position is used together:
``--ltx-video-min-seconds 2 4 --ltx-video-max-seconds 6 8`` writes two clips, one
from 2 to 6 seconds and one from 4 to 8. A single value with the other option
omitted uses that option's pipeline default (``1`` and ``20``). Each bound pair
is then tried in turn with the other arguments. See
`examples/ltx2/basic/duration-bounds-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/basic/duration-bounds-config.dgen>`_
and `examples/ltx2/basic/duration-head-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/basic/duration-head-config.dgen>`_.

Condition placement
~~~~~~~~~~~~~~~~~~~

``ltx-index`` is an LTX image-seed keyword. ``strength`` is the weight of that
condition, from 0 to 1, and the same keyword sets img2img strength on image
models that accept it. A path with neither keyword is still the first frame at
full strength, which is the same as ``ltx-index=0`` and ``strength=1``.
``--image-seed-strengths`` fills any LTX group that omits ``strength``. Several
values are tried in turn. ``last-frame=`` stays at strength 1.

``ltx-index``
^^^^^^^^^^^^^

``ltx-index`` is which latent frame the file is written into. It is not an output
frame number and not ``frame-start``.

LTX's video VAE compresses time by 8. A clip of ``8k+1`` output frames is
``k+1`` latent frames. The first latent frame is output frame 0. Each latent
frame after that covers the next 8 output frames. At 24 fps that chunk is about
a third of a second.

A still placed on a latent frame is held across the output frames that latent
frame covers, so the picture stays still there instead of moving through it.
``ltx-index=0`` holds the opening. ``ltx-index=-1`` holds the last latent frame,
the same slot ``last-frame=`` uses. ``ltx-index=4`` on a long clip is the latent frame
that begins at output frame 33 (``1 + 4*8``).

A video or animated image placed at an index starts at that latent frame. Its
frame count is still cut to ``8k+1`` before it is placed, using the same rule as
an opening clip. The clip is trimmed so it fits in the frames that remain after
that index.

``strength``
^^^^^^^^^^^^

``strength`` is how strongly that latent frame must match the file, from 0
to 1. ``1`` keeps the conditioning frames. A lower value lets the generated
frames leave them. Omitting it is ``1``, unless ``--image-seed-strengths``
is set. On an img2img seed the same keyword overrides ``--image-seed-strengths``
for that seed.

``last-frame=``
^^^^^^^^^^^^^^^

``last-frame=`` is the last frame at strength 1. It does not take ``strength``.
To hold the end more loosely, drop ``last-frame=`` and place that file with
``ltx-index=-1``:

.. code-block:: bash

    # last frame, full strength
    --image-seeds "start.jpg;last-frame=end.jpg"

    # last frame, partial strength
    --image-seeds "start.jpg ++ end.jpg;ltx-index=-1;strength=0.4"

Several conditions
^^^^^^^^^^^^^^^^^^

One ``--image-seeds`` value can carry several conditions. Separate them with
a space, ``++``, and a space. The first group is the primary
path and may omit ``ltx-index``. Each later group is one file and must include
``ltx-index``. ``strength`` is optional on every group. ``last-frame=``, ``control=``,
masks, and latents belong on the primary group only.

.. code-block:: bash

    --image-seeds "open.mp4;ltx-index=0 ++ mid.jpg;ltx-index=4;strength=0.6 ++ close.jpg;ltx-index=-1"

``open.mp4`` starts at the first latent frame. ``mid.jpg`` is held at latent
frame 4, loosely. ``close.jpg`` is the last latent frame at full strength.

``control=`` is not placed in the clip. With ``--ltx-ic-lora`` it is the
reference the IC-LoRA reads. ``ltx-index`` does not move that reference.

See `examples/ltx2/image_conditioning/indexed-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx2/image_conditioning/indexed-config.dgen>`_.

What LTX rejects
~~~~~~~~~~~~~~~~

ControlNets, T2I adapters, IP adapters, textual inversions, a replacement UNet,
an image encoder, the SDXL refiner, Stable Cascade, Adetailer, PAG and PAG scales,
any scheduler other than ``FlowMatchEulerDiscreteScheduler``, prompt weighters, second or third prompts,
clip skip, inpaint crop, HiDiffusion, TeaCache, DeepCache, SADA, RAS, Spectrum,
mask processors, raw latents and latents processors, ``--denoising-start`` /
``--denoising-end``, ``--batch-size`` greater than 1, ``--batch-grid-size``, latent output
formats, the safety checker, ``--vae-tiling``, and ``--original-config``.
Seed processors are limited to two chains, and
control processors to one.
``--quantizer-map`` may only name ``transformer``, ``text_encoder``, or ``connectors``.

LTX-2 and LTX-2.5 configs are in `examples/ltx2 <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx2>`_,
including Comfy GGUFs under `examples/ltx2/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx2/gguf>`_
and pre-quantized Gemma 4 text encoders under
`examples/ltx2/comfy_quant <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx2/comfy_quant>`_.
LTX-Video configs are in `examples/ltx_video <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx_video>`_.

Wan (2.1 / 2.2)
---------------

``--model-type wan``, ``wan-animate``, and ``wan-animate-2`` each write one clip
per pipeline call. Wan does not run once per input frame. Clip length and frame
rate are ``--video-lengths`` and ``--video-fps``. Other Wan options use the
``--wan-`` prefix. Conditioning stays in ``--image-seeds``.

``model_index.json`` selects the checkpoint family. A Wan-Animate repository
must be loaded with ``--model-type wan-animate``, and a Wan-Animate-2 repository
with ``--model-type wan-animate-2``.

A prompt can name what to avoid after one ``;``. The published examples use a
long negative that starts ``Bright tones, overexposed, static, blurred details``.
Omitting ``--guidance-scales`` or ``--inference-steps`` uses the loaded Wan
pipeline defaults (typically guidance ``5`` and ``50`` steps on ``WanPipeline``).
The examples explicitly set sampling: ``5`` for Wan 2.1 and ``4`` with ``--wan-low-noise-guidance-scales 3``
for Wan 2.2 MoE.

``--video-fps`` defaults to 16 for wan, 30 for wan-animate, and 24 for
wan-animate-2. ``--video-lengths`` is seconds. Wan snaps that duration to
``temporal*k+1`` frames, where ``temporal`` comes from the loaded VAE (4 on Wan
2.1). Five seconds at 16 fps is 81 frames. Width and height must be divisible
by the checkpoint's spatial multiple, which is 16 on the published Diffusers
checkpoints. ``832x480`` and ``1280x720`` are the sizes used in the examples.

``--scheduler`` for ``--model-type wan`` is ``FlowMatchEulerDiscreteScheduler``.
Wan-Animate uses ``UniPCMultistepScheduler``. Wan-Animate-2 also accepts
``EulerDiscreteScheduler``, ``DPMSolverMultistepScheduler``, and
``FlowMatchEulerDiscreteScheduler``.

Text to video
~~~~~~~~~~~~~

No image seed is text to video. ``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` is the
Wan 2.2 mixture-of-experts checkpoint. The examples use guidance 4 and
``--wan-low-noise-guidance-scales 3``. ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers`` is
the small Wan 2.1 checkpoint.

.. code-block:: bash

    Wan-AI/Wan2.2-T2V-A14B-Diffusers --model-type wan \
        --guidance-scales 4 --wan-low-noise-guidance-scales 3 \
        --video-lengths 5 --video-fps 16 --output-size 832x480 \
        --prompts "a red fox riding a horse down a highway;static, blurry"

See `examples/wan/basic/text-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/basic/text-to-video-config.dgen>`_
for Wan 2.1 and
`examples/wan/basic/text-to-video-wan22-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/basic/text-to-video-wan22-config.dgen>`_
for Wan 2.2.

Image to video
~~~~~~~~~~~~~~

One still is the first frame. The checkpoint needs an image encoder.
``Wan-AI/Wan2.2-I2V-A14B-Diffusers`` is the Wan 2.2 image-to-video model.
It only embeds the opening still. ``last-frame=`` still needs a first-last-frame
checkpoint such as ``Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers``.

.. code-block:: bash

    --image-seeds first.jpg

See `examples/wan/image_conditioning/image-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/image_conditioning/image-to-video-config.dgen>`_
for Wan 2.1 and
`examples/wan/image_conditioning/image-to-video-wan22-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/image_conditioning/image-to-video-wan22-config.dgen>`_
for Wan 2.2.

First and last frame
~~~~~~~~~~~~~~~~~~~~

``last-frame=`` is first-last-frame (FLF2V). Load a checkpoint such as
``Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers``. Combining ``last-frame=`` with a
Wan 2.1 I2V transformer is rejected, because that transformer cannot embed
the first and last images together. The published FLF2V size is ``1280x720``.

.. code-block:: bash

    --image-seeds "first.png;last-frame=last.png"

See `examples/wan/flf2v/first-last-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/flf2v/first-last-config.dgen>`_.

Video to video
~~~~~~~~~~~~~~

A video or animated image in the seed path is video-to-video. ``--video-lengths``
is rejected. The output follows the input clip. Trim it with ``frame-start=``
and ``frame-end=`` on the seed, or with ``--frame-start`` and ``--frame-end``.
``--image-seed-strengths``, or ``strength=`` on the seed, is how strongly the
result stays with the input. ``--timesteps`` takes an explicit timestep schedule
in this mode only. Each value is a comma-separated integer list or an ``expr:``
expression, tried in turn.

.. code-block:: bash

    --image-seeds "clip.mp4;frame-end=48" --image-seed-strengths 0.7

See `examples/wan/video_to_video/video-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/video_to_video/video-to-video-config.dgen>`_.

VACE
~~~~

``control=``, ``mask=``, and ``reference=`` are VACE inputs. They require a VACE
checkpoint such as ``Wan-AI/Wan2.1-VACE-1.3B-diffusers``. ``control=`` is the
control clip. ``reference=`` is an appearance still. White ``mask=`` pixels are
generated and black pixels keep the source clip. A video in the seed path is
that source clip when ``control=`` is omitted. A still in the seed path is not
used. ``reference=`` alone is reference-to-video. ``--control-image-processors``
runs on ``control=``. The same kind of processor on a plain video seed is
``--seed-image-processors``. Leave the clip unprocessed when a mask should
preserve it. ``--wan-conditioning-scales`` is the VACE scale: one float, or a
comma-separated list with one scale per VACE layer. Several values are tried
in turn.

.. code-block:: bash

    --image-seeds "control=hiker.mp4"
    --control-image-processors canny

    --image-seeds "hiker.mp4"

    --image-seeds "hiker.mp4;mask=person.gif;reference=astronaut.jpg"

    --image-seeds "reference=astronaut.jpg"

See `examples/wan/vace <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/wan/vace>`_.

Wan 2.2 MoE
~~~~~~~~~~~

A Wan 2.2 repository whose ``model_index.json`` lists ``transformer_2`` is a
mixture of experts. The high-noise expert uses ``--guidance-scales``. The
low-noise expert uses ``--wan-low-noise-guidance-scales``, and copies
``--guidance-scales`` when that option is omitted. ``--wan-boundary-ratios``
is the switch point from 0 to 1. Omit it and the checkpoint value is kept.
``--wan-expand-timesteps`` turns on per-token timestep expansion.
``--wan-second-transformer`` replaces the low-noise expert. It uses the same
URI grammar as ``--transformer``, including a GGUF file.

``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` is the text-to-video MoE checkpoint.
See `examples/wan/moe/moe-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/moe/moe-config.dgen>`_.

Wan-Animate
~~~~~~~~~~~

``--model-type wan-animate`` takes one character still plus motion.
``Wan-AI/Wan2.2-Animate-14B-Diffusers`` is the checkpoint. ``--video-lengths``
is rejected. The output follows the pose clip. ``--video-fps`` defaults to 30.
Omitting ``--inference-steps`` uses ``20`` and omitting ``--guidance-scales`` uses ``1``,
matching ``WanAnimatePipeline``. Wan-Animate runs unguided. Values you set are kept.

Pass ``wan-pose=`` and ``wan-face=`` clips, or pass ``wan-driving=`` and let
dgenerate derive them. ``--wan-animate-preprocess`` runs ``openpose`` for the
pose and ``yolo`` in crop mode for the face
(``Bingsu/adetailer;weight-name=face_yolov8n.pt;crops=True;crop-square=True;crop-scale=1.4``).
The derived frames stay in memory.
``--wan-animate-preprocess-output`` writes the pose and face clips to the
directory given by ``--output-path``. A name that already exists receives a
``_duplicate_`` suffix.
Custom plugins can replace that pair: ``--wan-pose-image-processors`` and
``--wan-face-image-processors`` run on ``wan-driving=`` when the matching clip
is omitted, and on ``wan-pose=`` or ``wan-face=`` when those clips are given.
``--wan-driving-image-processors`` runs on the driving clip first.
``--wan-background-image-processors`` runs on ``wan-background=``.

``--wan-animate-mode animate`` is character animation. ``replace`` needs
``wan-background=`` and ``mask=`` clips of the scene the character is placed
into. ``--wan-segment-frame-lengths`` must be ``4N+1``. The default is 77.
``--wan-prev-segment-frames`` is the overlap from the previous segment. 1 or 5
is recommended. The default is 1. ``--wan-motion-encode-batch-size`` trades
memory for speed in the Wan-Animate motion encoder. Wan-Animate-2 does not
use this option.

.. code-block:: bash

    --image-seeds "character.jpg;wan-driving=motion.mp4;frame-end=48" \
        --wan-animate-preprocess

    --wan-animate-mode replace \
        --image-seeds "character.jpg;wan-driving=motion.mp4;wan-background=scene.mp4;mask=person.gif"

``last-frame=``, ``control=``, and ``reference=`` are rejected. Those keywords
belong to FLF2V and VACE. ``wan-driving=`` is rejected on ``--model-type wan``.

See `examples/wan_animate <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/wan_animate>`_.

Wan-Animate-2
~~~~~~~~~~~~~

``--model-type wan-animate-2`` takes one character still and ``wan-driving=``.
The driving clip is the motion source. There is no pose or face clip.
``--video-lengths`` is rejected. The output follows the driving clip.
``--video-fps`` defaults to 24. ``wan-pose=``, ``wan-face=``,
``wan-background=``, ``last-frame=``, ``control=``, ``mask=``, and
``reference=`` are rejected.

``Wan-AI/Wan2.2-Animate-2-14B-Diffusers`` uses ``40`` steps when
``--inference-steps`` is omitted. ``Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers``
uses ``10``. Values you set are kept. The published demo uses ``--wan-segment-frame-lengths 81``,
``--wan-prev-segment-frames 1``, and ``--max-sequence-length 512``.
``--wan-driving-image-processors`` can preprocess the driving clip.
The transformer blocks are compiled on every load, with or without
``--torch-compile``. That compile keeps flex attention block-sparse.

.. code-block:: bash

    --image-seeds "character.png;wan-driving=template.mp4;frame-end=48"

See `examples/wan_animate_2 <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/wan_animate_2>`_.

Precision, GGUF, and offload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--dtype bfloat16`` is the dtype used by the examples. The VAE still loads in
``float32``, including when ``--vae`` omits ``dtype=``. ``AutoencoderKLWan`` is
fragile in ``bfloat16``. Set ``dtype=`` on an ``AutoencoderKLWan`` URI only
when you want another precision. ``--quantizer-map`` may name ``transformer``,
``transformer_2``, ``text_encoder``, ``text_encoder_2``, or ``image_encoder``.
It may not name ``vae``.

``--transformer`` replaces the diffusion transformer, including a city96
``.gguf`` file. The Diffusers repository still supplies the VAE and the UMT5
text encoder. ``--wan-second-transformer`` does the same for the Wan 2.2
low-noise expert. ``--text-encoders`` replaces UMT5. Use ``+`` to keep the
checkpoint default, or ``null`` to skip a slot. Wan 2.2 has that same one
UMT5 encoder. See
`examples/wan/gguf/text-encoder-wan22-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/gguf/text-encoder-wan22-config.dgen>`_
and
`examples/wan/comfy_quant/text-encoder-wan22-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/comfy_quant/text-encoder-wan22-config.dgen>`_.
``--loras`` loads a Diffusers
LoRA onto the transformer. ``component=transformer_2`` loads one LoRA onto the
Wan 2.2 low-noise expert instead, so a high-noise and low-noise pair can be
given together.

LightX2V's 4-step Wan LoRAs use the undistilled checkpoint, guidance 1, 4
steps, and ``UniPCMultistepScheduler`` with ``flow-shift=5``. Wan 2.2 uses one
LoRA per expert. See
`examples/wan/lora/lightx2v-4step-t2v-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/lora/lightx2v-4step-t2v-config.dgen>`_,
`examples/wan/lora/lightx2v-4step-i2v-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/lora/lightx2v-4step-i2v-config.dgen>`_,
and
`examples/wan/lora/lightx2v-4step-moe-t2v-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/lora/lightx2v-4step-moe-t2v-config.dgen>`_.

``--model-sequential-offload``, ``--model-cpu-offload``, and
``--model-group-offload`` are available. ``--torch-compile`` compiles repeated
transformer blocks on Wan and Wan-Animate. Wan-Animate-2 compiles those blocks
on every load. Quantized weights stay eager and dgenerate warns. Sequential offload and model CPU
offload move a GGUF or SDNQ transformer. Bitsandbytes 8-bit stays on the GPU
where it was loaded, and bitsandbytes 4-bit stays on the GPU under sequential
offload. Group offload streams the full-precision modules, such as the VAE and text
encoder, and places a GGUF, SDNQ, or bitsandbytes transformer on the run
device instead of streaming its packed weights.

See `examples/wan/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/wan/gguf>`_
and `examples/wan/lora/lora-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/wan/lora/lora-config.dgen>`_.

Configs are in `examples/wan <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/wan>`_,
`examples/wan_animate <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/wan_animate>`_,
and `examples/wan_animate_2 <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/wan_animate_2>`_.
