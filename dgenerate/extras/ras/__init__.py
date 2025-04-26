import importlib.util
from contextlib import contextmanager
from diffusers import StableDiffusion3Pipeline

__version__ = "0.1"


class RASArgs:
    def __init__(
        self,
        num_inference_steps: int,
        patch_size: int = 2,
        starvation_scale: float = 0.1,
        scheduler_start_step: int = 4,
        scheduler_end_step: int | None = None,
        metric: str = "std",
        error_reset_steps: list[int] = None,
        replace_with_flash_attn: bool = False,
        sample_ratio: float = 0.5,
        skip_num_step: int = 0,
        skip_num_step_length: int = 0,
        height: int = 1024,
        width: int = 1024,
        high_ratio: float = 1,
        enable_index_fusion: bool = False
    ):
        self.patch_size = patch_size
        self.scheduler_start_step = scheduler_start_step
        self.scheduler_end_step = num_inference_steps if scheduler_end_step is None else scheduler_end_step
        self.metric = metric
        self.error_reset_steps = [12, 22] if error_reset_steps is None else error_reset_steps
        self.replace_with_flash_attn = replace_with_flash_attn
        self.sample_ratio = sample_ratio
        self.num_inference_steps = num_inference_steps
        self.skip_num_step = skip_num_step
        self.skip_num_step_length = skip_num_step_length
        self.height = height
        self.width = width
        self.high_ratio = high_ratio
        self.enable_index_fusion = enable_index_fusion
        self.starvation_scale = starvation_scale


@contextmanager
def sd3_ras_context(pipeline: StableDiffusion3Pipeline, args: RASArgs, enabled: bool = True):
    """
    Context manager to enable/disable RAS (Reinforcement Attention System) on a Stable Diffusion 3 pipeline.

    This context manager temporarily modifies the SD3 pipeline to use RAS components
    and ensures proper cleanup of the pipeline state when exiting the context.

    :param pipeline: The :py:class:`diffusers.StableDiffusion3Pipeline` to modify
    :param enabled: Whether to enable RAS (True) or disable it (False)
    :param args: Optional RASArgs object containing RAS configuration
    :yields: The modified pipeline
    """
    if not enabled:
        yield pipeline
        return

    if importlib.util.find_spec('triton') is None:
        raise RuntimeError(
            "RAS requires the 'triton' package to be installed. "
            "Please install it using 'pip install triton' or "
            "'pip install triton-windows' for Windows users.")

    from .utils.ras_manager import MANAGER
    from .utils.stable_diffusion_3.update_pipeline_sd3 import update_sd3_pipeline

    # Store original state
    original_scheduler = pipeline.scheduler
    original_transformer_forward = pipeline.transformer.forward
    original_attn_processors = [block.attn.processor for block in pipeline.transformer.transformer_blocks]

    if args.enable_index_fusion and pipeline.transformer.config.qk_norm == 'rms_norm':
        raise ValueError("Index fusion is not supported with SD3.5.")

    try:
        MANAGER.set_parameters(args)
        
        # Enable RAS
        update_sd3_pipeline(pipeline)
        MANAGER.reset_cache()
        yield pipeline

    finally:
        # Restore original state
        pipeline.scheduler = original_scheduler
        pipeline.transformer.forward = original_transformer_forward
        for block, processor in zip(pipeline.transformer.transformer_blocks, original_attn_processors):
            block.attn.set_processor(processor)
        MANAGER.reset_cache()


__all__ = ['sd3_ras_context', 'RASArgs']
