# DyPE adapter

This package adapts the positional-frequency interpolation from [guyyariv/DyPE](https://github.com/guyyariv/DyPE)
to dgenerate's Diffusers pipelines. The upstream project is MIT licensed; its
license is included here. The adapter uses dgenerate's existing PyTorch install
and adds no package dependency. DyPE is opt-in with `--dype`.

`--dype-methods` accepts `yarn` (default), `ntk`, and `base`; the render loop
tries each supplied method in turn. `base` keeps the original RoPE frequencies
and is useful as a comparison when swept with the extrapolation methods.

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
