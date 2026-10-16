# DyPE adapter

This package adapts the positional-frequency interpolation from [guyyariv/DyPE](https://github.com/guyyariv/DyPE)
to dgenerate's Diffusers pipelines. The upstream project is MIT licensed; its
license is included here. The adapter uses dgenerate's existing PyTorch install
and adds no package dependency. DyPE is opt-in with `--dype`.

`--dype-methods` accepts `yarn` (default), `ntk`, and `base`; the render loop
tries each supplied method in turn. `base` keeps the original RoPE frequencies
and is useful as a comparison when swept with the extrapolation methods.

`--dype-mscales` controls YARN's attention magnitude scale, which multiplies
the rotated query and key vectors by `0.1 * ln(s) + 1` for an extrapolation
factor `s`. `constant` (default) is upstream DyPE's fixed scale. `decay` fades
it toward 1 as `t^2`, so the final denoising steps run unscaled; this is the
`vision_yarn` schedule from [ComfyUI-DyPE](https://github.com/wildminder/ComfyUI-DyPE).
`off` disables it. Only the `yarn` method applies this scale.

The constant scale suits FLUX.1-Krea-dev, which upstream DyPE was tuned on.
FLUX.1-dev at the same settings produces a fine diagonal texture, with a period
of one to two latent tokens, over flat regions; disabling the scale removes it.
The texture forms late in denoising, which is why `decay` is offered as a
middle ground that keeps the scale during the structural steps; in testing
it removes the texture on FLUX.1-dev while leaving Krea output close to
`constant`.

`--dype-axis-scalings` controls non-square images. Upstream DyPE scales each
spatial axis by its own ratio to the base length, so at 2048x1024 the height
axis is interpolated and magnitude-scaled while the 1024-pixel width axis is
left untouched. `independent` keeps that behavior. `shared` (default) follows
ComfyUI-DyPE's `vision_yarn`: the NTK frequency base, NTK factor, and YARN
magnitude scale come from the larger axis ratio and apply to both axes, while
YARN's linear interpolation stays per-axis. Square images and the long axis are
identical in both modes. On portrait and landscape tests with FLUX.1-dev and
Krea, `shared` gave more coherent scene structure where the short side sat at
the base resolution and somewhat richer detail elsewhere. Wan's adapter already
scales both axes by the long side, so it ignores this option.

## Pipeline coverage

| Pipeline family | Adapter | Status |
| --- | --- | --- |
| Flux.1 | Flux positional embedder | Upstream architecture |
| Flux Fill and Kontext | Flux positional embedder | Experimental sibling adapters |
| Qwen-Image | Qwen shape-based RoPE | Upstream architecture |
| Qwen-Image Edit and Layered | Qwen shape-based RoPE | Experimental sibling adapters |
| Flux.2 and Klein KV | Flux.2 positional embedder | Experimental candidate |
| Z-Image and Omni | Callable spatial RoPE | Experimental candidate |
| Wan, Wan-Animate, Wan-Animate-2 | Spatial axes of 3D RoPE | Experimental video candidate |

The adapters modify only spatial axes. Qwen text frequencies and Wan's temporal
axis remain unchanged. `--dype-base-resolutions` overrides the reference long
side used to decide when spatial scaling begins; defaults are 1024 pixels for
Flux, Flux.2, and Z-Image, 1328 for Qwen-Image, and 832 for Wan.

The upstream DyPE Flux pipeline also adjusts the scheduler shift calculation:
it passes the current image sequence length as `max_seq_len`, rather than using
the scheduler config's fixed maximum. The extra's `dype_context` mirrors that
behavior for Diffusers `FluxPipeline` calls and leaves other pipeline types'
scheduler behavior unchanged.

DyPE cannot be combined with Cache-DiT, TeaCache, Spectrum, or SADA. Each of
those features reuses or skips denoiser work computed at an earlier step, so
the reused blocks would keep that step's positional frequencies. Torch compile
is allowed: RoPE is built in the eager transformer forward and passed into the
compiled blocks. ADetailer refinement passes use the same DyPE patch.

## SEGA compatibility

SEGA can refer to two separate methods. The original
[semantic-guidance implementation](https://github.com/ml-research/semantic-image-editing)
changes the denoising update through classifier-free guidance and does not
modify RoPE. It is not integrated here. The newer
[Spectral-Energy Guided Attention implementation](https://github.com/rajabi2001/sega)
derives per-frequency RoPE attention scaling from the current latent spectrum.
Its Flux and Qwen implementations modify the same positional embedding calls
replaced by these DyPE adapters. They cannot currently be enabled as independent
patches; combining them needs a fused RoPE adapter that passes SEGA's per-step
spectral profile into the selected DyPE method. That fusion is a follow-up
candidate, not an implemented or quality-validated mode.

## Follow-up candidates

LTX and LTX-2 need separate adapters. LTX uses normalized spatiotemporal
coordinates, while LTX-2 has distinct video, audio, and cross-attention RoPE
modules. Their frequency mapping should be evaluated independently before
exposing them through `--dype`.

The spectral SEGA fusion described above is another follow-up candidate. The
official implementation currently targets Flux, Qwen-Image, and Ideogram4;
adding it would require per-step latent spectrum capture and a shared RoPE
frequency/magnitude path.

The upstream README says the work is patent pending and asks users to contact
the authors about commercial use or licensing. This note is retained here
because the MIT source license does not resolve that separate notice.
