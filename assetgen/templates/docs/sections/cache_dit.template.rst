Cache-DiT Configuration
========================

Cache-DiT reuses transformer block results across denoising steps. It is most
useful with larger DiT models and longer generation schedules. It can improve
throughput, with a configurable speed and output-quality tradeoff. Quantization
and layerwise offload can reduce memory pressure; parallelism can spread work
across GPUs. These options depend on the model, backend, and hardware, so they
do not all apply to every pipeline.

The Cache-DiT 1.5.2 cache adapters cover these dgenerate ``--model-type`` values:
``sd3``, ``sd3-pix2pix``, ``flux``, ``flux-fill``, ``flux-kontext``, ``flux2``,
``flux2-klein-kv``, ``z-image``, ``z-image-omni``, ``qwen-image``,
``qwen-image-edit``, ``qwen-image-layered``, ``ltx``, and ``wan``. dgenerate
also supplies custom Cache-DiT block adapters for ``wan-animate`` and
``wan-animate-2`` (including the distilled Wan-Animate-2 family). These use the
transformers' ``blocks`` lists with Cache-DiT's ``Pattern_2`` and ``Pattern_3``
contracts, respectively. Wan-Animate-2 keeps its own reference K/V pass; its
interaction with Cache-DiT caching should be checked with the target checkpoint
and cache settings. Cache-DiT Ray mode is not supported for these custom
adapters. Standard UNet model types such as ``sd`` and ``sdxl`` do not use this
DiT cache. Check Cache-DiT's feature documentation for model-specific
parallelism, attention, quantization, and offload support.

``--cache-dit`` enables Cache-DiT with its defaults. Use ``--cache-dit-config``
to supply Cache-DiT's YAML configuration; that option also enables Cache-DiT.
The native ``cache_config``, ``calibrator_config``, ``parallelism_config``,
``quantize_config``, and ``attention_backend`` sections pass through Cache-DiT's
config loader. Advanced per-block cache overrides can be supplied through the
``params_modifiers`` list. Each entry may contain ``cache_config`` and/or
``calibrator_config`` settings, and entries correspond in order to the block
groups selected by Cache-DiT's adapter:

.. code-block:: yaml

    params_modifiers:
      - cache_config:
          residual_diff_threshold: 0.08
      - cache_config:
          residual_diff_threshold: 0.14

Layerwise offload is configured by pipeline component name. Each component's
settings are forwarded to Cache-DiT's ``layerwise_offload`` API. The onload
device defaults to ``--device`` and the offload device defaults to CPU. For
example:

.. code-block:: yaml

    cache_config:
      max_warmup_steps: 8
      residual_diff_threshold: 0.08
    parallelism_config:
      ulysses_size: 2
      use_ray: true
      attention_backend: native
      ulysses_float8: true
    quantize_config:
      quant_type: float8_per_row
      exclude_layers:
        - embedder
    layerwise_offload:
      transformer:
        async_transfer: true
        transfer_buckets: 2
        persistent_buckets: 16
        persistent_bins: 4
        prefetch_limit: true
        max_copy_streams: 2
        max_inflight_prefetch_bytes: 4294967296

Callable extension points can use ``python:module:attribute`` references. For
example, set ``parallelism_config.ray_transfer_fn`` or a component's
``layerwise_offload.module_filter`` to a module-level callable reference. The
module must be importable by the main process and by Ray workers when Ray is
used; Cache-DiT's ``parallelism_config.ray_runtime_env`` can configure worker
imports.

For standalone CLI use, set ``parallelism_config.use_ray: true`` and install
Cache-DiT's ``ray`` extra. Native parallelism requires an initialized
``torch.distributed`` process group; dgenerate does not initialize process
groups, so native mode is available when dgenerate is embedded in a host
application that does. Install Cache-DiT's ``parallelism`` and
``quantization`` extras when the selected backend requires them. Cache-DiT
quantization cannot be combined with dgenerate's ``--quantizer`` settings.
Layerwise offload cannot be combined with dgenerate's ``--model-cpu-offload``,
``--model-sequential-offload``, or ``--model-group-offload`` settings.
Cache-DiT Ray parallelism also cannot be combined with those dgenerate offload
settings. Parallelism, attention backends, and quantization modify the loaded
pipeline, so keep their settings unchanged while reusing that pipeline.
Configure Cache-DiT parallelism, attention, quantization, and layerwise offload
before compilation; these settings cannot be combined with dgenerate's
``--torch-compile``. For Ray parallelism, use Cache-DiT's
``parallelism_config.ray_use_compile`` with dgenerate compilation disabled.

See the Cache-DiT `YAML configuration <https://cache-dit.readthedocs.io/en/latest/user_guide/LOAD_CONFIGS/>`_,
`API <https://cache-dit.readthedocs.io/en/latest/user_guide/API_DOCS/>`_, and
`layerwise offload <https://cache-dit.readthedocs.io/en/latest/user_guide/OFFLOAD/>`_
documentation for available settings and hardware requirements.
