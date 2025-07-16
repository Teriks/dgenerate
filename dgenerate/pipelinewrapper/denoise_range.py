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
import inspect
import itertools

import diffusers.schedulers
import torch
from contextlib import contextmanager

import dgenerate.extras.kolors


def _is_sdxl_pipeline(pipeline):
    """Check if a pipeline is an SDXL pipeline that supports native denoising_start/denoising_end."""
    return pipeline.__class__.__name__.startswith('StableDiffusionXL')


def _is_sd3_pipeline(pipeline):
    """Check if a pipeline is an SD3 pipeline that uses flow matching."""
    return pipeline.__class__.__name__.startswith('StableDiffusion3')


def _is_flux_pipeline(pipeline):
    """Check if a pipeline is a Flux pipeline that uses flow matching."""
    return pipeline.__class__.__name__.startswith('Flux')


def _is_flow_matching_scheduler(scheduler):
    """Check if a scheduler uses flow matching."""
    return isinstance(scheduler, diffusers.schedulers.FlowMatchEulerDiscreteScheduler)


def _is_sigma_based_scheduler(scheduler):
    """Check if a scheduler primarily uses sigma-based stepping."""
    sigma_based_schedulers = (
        diffusers.schedulers.EulerDiscreteScheduler,
        diffusers.schedulers.HeunDiscreteScheduler,
        diffusers.schedulers.KDPM2DiscreteScheduler,
        diffusers.schedulers.LMSDiscreteScheduler,
        diffusers.schedulers.EDMEulerScheduler,
    )
    return isinstance(scheduler, sigma_based_schedulers)


def _is_dpm_solver_scheduler(scheduler):
    """Check if a scheduler is a DPM solver variant."""
    dpm_schedulers = (
        diffusers.schedulers.DPMSolverMultistepScheduler,
        diffusers.schedulers.DPMSolverSDEScheduler,
        diffusers.schedulers.DPMSolverSinglestepScheduler,
    )
    return isinstance(scheduler, dpm_schedulers)


def _is_multistep_scheduler(scheduler):
    """Check if a scheduler uses multistep methods."""
    multistep_schedulers = (
        diffusers.schedulers.DEISMultistepScheduler,
        diffusers.schedulers.UniPCMultistepScheduler,
        diffusers.schedulers.PNDMScheduler,
    )
    return isinstance(scheduler, multistep_schedulers)


def _is_lcm_scheduler(scheduler):
    """Check if a scheduler is a Latent Consistency Model scheduler."""
    return isinstance(scheduler, diffusers.schedulers.LCMScheduler)


def _is_wuerstchen_scheduler(scheduler):
    """Check if a scheduler is specific to Wuerstchen models."""
    return isinstance(scheduler, diffusers.schedulers.DDPMWuerstchenScheduler)


def _is_problematic_scheduler(scheduler):
    """Check if a scheduler has state management issues with denoise_range.
    
    Only considers schedulers that are actually supported by dgenerate (listed in pygments.py).
    """
    problematic_schedulers = (
        # SDE schedulers have stochastic state that doesn't split well
        diffusers.schedulers.DPMSolverSDEScheduler,
        
        # Ancestral schedulers add noise stochastically at each step
        # Breaking the sequence disrupts the noise pattern
        diffusers.schedulers.EulerAncestralDiscreteScheduler,
        diffusers.schedulers.KDPM2AncestralDiscreteScheduler,
        
        # These schedulers produce black outputs or incorrect progress bars with denoise range
        diffusers.schedulers.HeunDiscreteScheduler,
        diffusers.schedulers.KDPM2DiscreteScheduler,
        diffusers.schedulers.DPMSolverSinglestepScheduler,
        
        # Multistep schedulers have complex state that causes runtime errors
        diffusers.schedulers.DEISMultistepScheduler,
        diffusers.schedulers.UniPCMultistepScheduler,

        # LCM scheduler is stateful and cannot be reliably split into ranges
        diffusers.schedulers.LCMScheduler,

        # Wuerstchen scheduler not compatible with denoise range
        diffusers.schedulers.DDPMWuerstchenScheduler,
    )

    return isinstance(scheduler, problematic_schedulers)


def _supports_denoise_range_flow_matching(pipeline):
    """Check if a pipeline supports our denoise range flow matching modifications."""
    # Currently SD3 and Flux pipelines are known to work with our modifications
    return _is_sd3_pipeline(pipeline) or _is_flux_pipeline(pipeline)


def _create_progress_bar_modifier(original_scheduler, original_progress_bar, inference_steps_ref):
    """Create a progress bar modifier function that handles denoise range correctly."""
    def modified_progress_bar(total=None, **kwargs):
        # Get current inference_steps value (may not be set yet)
        current_inference_steps = inference_steps_ref[0] if inference_steps_ref[0] > 0 else None

        # If inference_steps isn't available yet, fall back to original behavior
        if current_inference_steps is None or total is None:
            if original_progress_bar:
                return original_progress_bar(total=total, **kwargs)
            else:
                try:
                    from tqdm.auto import tqdm
                    return tqdm(total=total, **kwargs)
                except ImportError:
                    class DummyProgressBar:
                        def __enter__(self):
                            return self
                        def __exit__(self, *args):
                            pass
                        def update(self, *args, **kwargs):
                            pass
                    return DummyProgressBar()

        # we can determine if strength is applied with this factor
        # if it is < 1.0, an img2img/inpaint pipeline has applied strength
        strength = total / current_inference_steps

        # Use the stored actual timesteps count, fallback to scheduler timesteps if available
        if hasattr(original_scheduler, 'timesteps'):
            actual_total = len(original_scheduler.timesteps)
        else:
            actual_total = total

        # calculate the actual total based on a possible strength factor
        # with strength = 1.0, it will be the same scheduler timesteps
        # which is already divided based on the start and end range
        actual_total = int(actual_total * strength) - int((current_inference_steps - actual_total) * (1.0 - strength))

        if original_progress_bar:
            return original_progress_bar(total=actual_total, **kwargs)
        else:
            # Fallback to tqdm if no original progress_bar method
            try:
                from tqdm.auto import tqdm
                return tqdm(total=actual_total, **kwargs)
            except ImportError:
                # Return a dummy context manager if tqdm is not available
                class DummyProgressBar:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass

                    def update(self, *args, **kwargs):
                        pass

                return DummyProgressBar()

    return modified_progress_bar


def _apply_flow_matching_denoise_range(pipeline, start, end, inference_steps_ref):
    """Apply denoise range for flow matching schedulers."""
    original_scheduler = pipeline.scheduler
    original_set_timesteps = original_scheduler.set_timesteps
    original_progress_bar = getattr(pipeline, 'progress_bar', None)

    def modified_set_timesteps(num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None, **kwargs):
        # Update the reference so progress bar can access it
        if num_inference_steps is not None:
            inference_steps_ref[0] = num_inference_steps

        # Call original to set up the full schedule
        original_set_timesteps(num_inference_steps=num_inference_steps, device=device, sigmas=sigmas, mu=mu, timesteps=timesteps, **kwargs)

        # Get the full timesteps and sigmas
        full_timesteps = original_scheduler.timesteps.clone()
        full_sigmas = original_scheduler.sigmas.clone()

        # Calculate the range indices
        num_steps = len(full_timesteps)
        start_idx = int(start * num_steps)
        end_idx = int(end * num_steps)
        end_idx = min(end_idx, num_steps)

        if start_idx >= end_idx:
            end_idx = start_idx + 1

        # Select timesteps for this range
        selected_timesteps = full_timesteps[start_idx:end_idx]

        # For sigmas, include one extra element to handle scheduler boundary access
        # The flow matching scheduler needs to access sigma[i+1]
        sigma_end_idx = min(end_idx + 1, len(full_sigmas))
        selected_sigmas = full_sigmas[start_idx:sigma_end_idx]

        # Update scheduler state
        original_scheduler.timesteps = selected_timesteps
        original_scheduler.sigmas = selected_sigmas
        original_scheduler._step_index = None  # Reset step index

        # For Flux pipelines, ensure the number of inference steps matches the selected range
        # This helps maintain proper quality by ensuring the scheduler knows the actual step count
        if _is_flux_pipeline(pipeline):
            original_scheduler.num_inference_steps = len(selected_timesteps)

    modified_progress_bar = _create_progress_bar_modifier(original_scheduler, original_progress_bar, inference_steps_ref)

    # Store original num_inference_steps for Flux pipelines
    original_num_inference_steps = getattr(original_scheduler, 'num_inference_steps', None)

    try:
        original_scheduler.set_timesteps = modified_set_timesteps

        # Override progress_bar method if it exists
        if hasattr(pipeline, 'progress_bar'):
            pipeline.progress_bar = modified_progress_bar

        yield
    finally:
        original_scheduler.set_timesteps = original_set_timesteps

        # Restore original num_inference_steps for Flux pipelines
        if _is_flux_pipeline(pipeline) and original_num_inference_steps is not None:
            original_scheduler.num_inference_steps = original_num_inference_steps

        # Restore original progress_bar if it was modified
        if original_progress_bar is not None:
            pipeline.progress_bar = original_progress_bar


def _apply_sigma_based_denoise_range(pipeline, start, end, inference_steps_ref):
    """Apply denoise range for sigma-based schedulers (Euler, Heun, K-DPM, LMS, EDM)."""
    original_scheduler = pipeline.scheduler
    original_set_timesteps = original_scheduler.set_timesteps
    original_progress_bar = getattr(pipeline, 'progress_bar', None)

    def modified_set_timesteps(num_inference_steps=None, device=None, **kwargs):
        # Update the reference so progress bar can access it
        if num_inference_steps is not None:
            inference_steps_ref[0] = num_inference_steps

        # Filter kwargs to only include parameters the scheduler accepts
        sig = inspect.signature(original_set_timesteps)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # Call original to set up the full schedule
        original_set_timesteps(num_inference_steps=num_inference_steps, device=device, **filtered_kwargs)

        # Get the full timesteps and sigmas
        full_timesteps = original_scheduler.timesteps.clone()

        # Calculate the range indices
        num_steps = len(full_timesteps)
        start_idx = int(start * num_steps)
        end_idx = int(end * num_steps)
        end_idx = min(end_idx, num_steps)

        if start_idx >= end_idx:
            end_idx = start_idx + 1

        # Select timesteps for this range
        selected_timesteps = full_timesteps[start_idx:end_idx]
        original_scheduler.timesteps = selected_timesteps

        # For schedulers with sigmas, update them as well
        if hasattr(original_scheduler, 'sigmas') and original_scheduler.sigmas is not None:
            full_sigmas = original_scheduler.sigmas.clone()
            # Include one extra element for boundary access
            sigma_end_idx = min(end_idx + 1, len(full_sigmas))
            selected_sigmas = full_sigmas[start_idx:sigma_end_idx]
            original_scheduler.sigmas = selected_sigmas

    modified_progress_bar = _create_progress_bar_modifier(original_scheduler, original_progress_bar, inference_steps_ref)

    try:
        original_scheduler.set_timesteps = modified_set_timesteps

        # Override progress_bar method if it exists
        if hasattr(pipeline, 'progress_bar'):
            pipeline.progress_bar = modified_progress_bar

        yield
    finally:
        original_scheduler.set_timesteps = original_set_timesteps

        # Restore original progress_bar if it was modified
        if original_progress_bar is not None:
            pipeline.progress_bar = original_progress_bar


def _apply_dpm_solver_denoise_range(pipeline, start, end, inference_steps_ref):
    """Apply denoise range for DPM Solver schedulers."""
    original_scheduler = pipeline.scheduler
    original_set_timesteps = original_scheduler.set_timesteps
    original_progress_bar = getattr(pipeline, 'progress_bar', None)

    def modified_set_timesteps(num_inference_steps=None, device=None, **kwargs):
        # Update the reference so progress bar can access it
        if num_inference_steps is not None:
            inference_steps_ref[0] = num_inference_steps

        # Filter kwargs to only include parameters the scheduler accepts
        sig = inspect.signature(original_set_timesteps)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # Call original to set up the full schedule
        original_set_timesteps(num_inference_steps=num_inference_steps, device=device, **filtered_kwargs)

        # Get the full timesteps
        full_timesteps = original_scheduler.timesteps.clone()

        # Calculate the range indices
        num_steps = len(full_timesteps)
        start_idx = int(start * num_steps)
        end_idx = int(end * num_steps)
        end_idx = min(end_idx, num_steps)

        if start_idx >= end_idx:
            end_idx = start_idx + 1

        # Select timesteps for this range
        selected_timesteps = full_timesteps[start_idx:end_idx]
        original_scheduler.timesteps = selected_timesteps

        # Reset DPM solver state
        if hasattr(original_scheduler, 'model_outputs'):
            original_scheduler.model_outputs = [None] * original_scheduler.config.solver_order

        # For DPM solvers with sigmas
        if hasattr(original_scheduler, 'sigmas') and original_scheduler.sigmas is not None:
            full_sigmas = original_scheduler.sigmas.clone()
            # Include one extra element for boundary access
            sigma_end_idx = min(end_idx + 1, len(full_sigmas))
            selected_sigmas = full_sigmas[start_idx:sigma_end_idx]
            original_scheduler.sigmas = selected_sigmas

    modified_progress_bar = _create_progress_bar_modifier(original_scheduler, original_progress_bar, inference_steps_ref)

    try:
        original_scheduler.set_timesteps = modified_set_timesteps

        # Override progress_bar method if it exists
        if hasattr(pipeline, 'progress_bar'):
            pipeline.progress_bar = modified_progress_bar

        yield
    finally:
        original_scheduler.set_timesteps = original_set_timesteps

        # Restore original progress_bar if it was modified
        if original_progress_bar is not None:
            pipeline.progress_bar = original_progress_bar


def _apply_multistep_denoise_range(pipeline, start, end, inference_steps_ref):
    """Apply denoise range for multistep schedulers (DEIS, UniPC, PNDM)."""
    original_scheduler = pipeline.scheduler
    original_set_timesteps = original_scheduler.set_timesteps
    original_progress_bar = getattr(pipeline, 'progress_bar', None)

    def modified_set_timesteps(num_inference_steps=None, device=None, **kwargs):
        # Update the reference so progress bar can access it
        if num_inference_steps is not None:
            inference_steps_ref[0] = num_inference_steps

        # Filter kwargs to only include parameters the scheduler accepts
        sig = inspect.signature(original_set_timesteps)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # Call original to set up the full schedule
        original_set_timesteps(num_inference_steps=num_inference_steps, device=device, **filtered_kwargs)

        # Get the full timesteps
        full_timesteps = original_scheduler.timesteps.clone()

        # Calculate the range indices
        num_steps = len(full_timesteps)
        start_idx = int(start * num_steps)
        end_idx = int(end * num_steps)
        end_idx = min(end_idx, num_steps)

        if start_idx >= end_idx:
            end_idx = start_idx + 1

        # Select timesteps for this range
        selected_timesteps = full_timesteps[start_idx:end_idx]
        original_scheduler.timesteps = selected_timesteps

        # Reset multistep state
        if hasattr(original_scheduler, 'model_outputs'):
            original_scheduler.model_outputs = []
        if hasattr(original_scheduler, 'timestep_list'):
            original_scheduler.timestep_list = []
        if hasattr(original_scheduler, 'cur_sample'):
            original_scheduler.cur_sample = None
        if hasattr(original_scheduler, 'counter'):
            original_scheduler.counter = 0
        if hasattr(original_scheduler, 'ets'):
            original_scheduler.ets = []

    modified_progress_bar = _create_progress_bar_modifier(original_scheduler, original_progress_bar, inference_steps_ref)

    try:
        original_scheduler.set_timesteps = modified_set_timesteps

        # Override progress_bar method if it exists
        if hasattr(pipeline, 'progress_bar'):
            pipeline.progress_bar = modified_progress_bar

        yield
    finally:
        original_scheduler.set_timesteps = original_set_timesteps

        # Restore original progress_bar if it was modified
        if original_progress_bar is not None:
            pipeline.progress_bar = original_progress_bar





def _apply_standard_denoise_range(pipeline, start, end, inference_steps_ref):
    """Apply denoise range for standard timestep-based schedulers (DDIM, DDPM, etc.)."""
    original_scheduler = pipeline.scheduler
    original_step = original_scheduler.step
    original_set_timesteps = original_scheduler.set_timesteps
    original_progress_bar = getattr(pipeline, 'progress_bar', None)

    def modified_set_timesteps(num_inference_steps, device=None, **kwargs):
        # Update the reference so progress bar can access it
        inference_steps_ref[0] = num_inference_steps
        original_set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = original_scheduler.timesteps

        # Calculate the subset of timesteps for this range
        num_steps = len(timesteps)

        # For exact half splits, use integer division to avoid rounding issues
        if start == 0.0 and end == 0.5:
            # First half gets the smaller portion for odd numbers
            start_idx = 0
            end_idx = num_steps // 2
        elif start == 0.5 and end == 1.0:
            # Second half gets the larger portion for odd numbers
            start_idx = num_steps // 2
            end_idx = num_steps
        else:
            # For other ranges, use precise calculation
            start_idx = int(start * num_steps)
            end_idx = int(end * num_steps)

            # Ensure valid range
            if start_idx >= end_idx:
                end_idx = start_idx + 1
            end_idx = min(end_idx, num_steps)

        # Select the timesteps for this range
        original_scheduler.timesteps = timesteps[start_idx:end_idx]

    def modified_step(*args, **kwargs):
        # The timesteps are now pre-filtered, so we can just call the original step
        return original_step(*args, **kwargs)

    modified_progress_bar = _create_progress_bar_modifier(original_scheduler, original_progress_bar, inference_steps_ref)

    try:
        original_scheduler.set_timesteps = modified_set_timesteps
        original_scheduler.step = modified_step

        # Override progress_bar method if it exists
        if hasattr(pipeline, 'progress_bar'):
            pipeline.progress_bar = modified_progress_bar

        yield
    finally:
        original_scheduler.set_timesteps = original_set_timesteps
        original_scheduler.step = original_step

        # Restore original progress_bar if it was modified
        if original_progress_bar is not None:
            pipeline.progress_bar = original_progress_bar


class DenoiseRangeError(Exception):
    """Exception for denoise_range context manager errors."""
    pass


def _class_predicate(o):
    return (inspect.isclass(o) and
            (o.__name__.startswith('StableDiffusionXL') or o.__name__.startswith('Kolors')))

_classes_to_check = list(
    itertools.chain(
        inspect.getmembers(diffusers, _class_predicate),
        inspect.getmembers(dgenerate.extras.kolors, _class_predicate)
    )
)

_supports_native_denoising_end = set()

for name, cls in _classes_to_check:
    if 'denoising_end' in inspect.signature(cls.__call__).parameters:
        _supports_native_denoising_end.add(cls)

_supports_native_denoising_start = set()

for name, cls in _classes_to_check:
    if 'denoising_start' in inspect.signature(cls.__call__).parameters:
        _supports_native_denoising_start.add(cls)

_classes_to_check = None


def supports_native_denoising_end(cls: type):
    """
    Does a pipeline class natively support ``denoising_end``

    :param cls: The pipeline class
    :return: ``True`` or ``False``
    """
    return cls in _supports_native_denoising_end

def supports_native_denoising_start(cls: type):
    """
    Does a pipeline class natively support ``denoising_start``

    :param cls: The pipeline class
    :return: ``True`` or ``False``
    """
    return cls in _supports_native_denoising_start

@contextmanager
def denoise_range(pipeline, start: float | None = 0.0, end: float | None = 1.0):
    """Context manager to temporarily set denoising range for a pipeline.

    This allows you to split the denoising process into a specific range, allowing
    for cooperative denoising with multiple pipelines.

    For SDXL pipelines, this transparently uses the native denoising_start/denoising_end
    parameters and supports ALL schedulers. For other pipelines (SD 1.5, etc.), it falls
    back to scheduler-specific manipulation that only works with stateless schedulers.

    SD 1.5 Supported schedulers (stateless):

    * ``EulerDiscreteScheduler``
    * ``LMSDiscreteScheduler``
    * ``EDMEulerScheduler``
    * ``DPMSolverMultistepScheduler``
    * ``DDIMScheduler``
    * ``DDPMScheduler``
    * ``PNDMScheduler``

    SD3/Flux Supported schedulers:

    * ``FlowMatchEulerDiscreteScheduler`` (flow matching with dedicated support)

    :param pipeline: The diffusion pipeline to modify
    :param start: Start point for denoising (0.0 to 1.0)
    :param end: End point for denoising (0.0 to 1.0)
    """
    if start is None:
        start = 0.0

    if end is None:
        end = 1.0

    if start == 0.0 and end == 1.0:
        # No need to modify if the full range is used
        yield
        return

    # For SDXL pipelines, use native denoising_start/denoising_end support
    if _is_sdxl_pipeline(pipeline):
        pipeline_class = pipeline.__class__
        original_call = pipeline_class.__call__

        if start is not None and start != 0.0:
            if not supports_native_denoising_start(pipeline_class):
                raise DenoiseRangeError(
                    f"{pipeline_class} not support a denoising_start parameter > 0.0, "
                    f"use an img2img or inpaint pipeline to refine and pass the "
                    f"latents in the image parameter.")

        if end is not None and end != 1.0:
            if not supports_native_denoising_end(pipeline_class):
                raise DenoiseRangeError(
                    f"{pipeline_class} not support a denoising_end parameter < 1.0, "
                    f"use a txt2img pipeline to create the initial latents.")

        def modified_call(self, *args, **kwargs):
            # Inject denoising_start/denoising_end into the call arguments
            if start > 0.0:
                kwargs['denoising_start'] = start
            if end < 1.0:
                kwargs['denoising_end'] = end

            # Ensure we're using torch.no_grad for inference
            with torch.no_grad():
                return original_call(self, *args, **kwargs)

        try:
            # Modify the class method, not the instance method
            pipeline_class.__call__ = modified_call
            yield
        finally:
            # Restore the original class method
            pipeline_class.__call__ = original_call
        return

    # Store the actual number of timesteps that will be processed (using list for reference)
    inference_steps_ref = [0]

    # Determine scheduler type and apply appropriate denoise range strategy
    scheduler = pipeline.scheduler

    # Check for problematic schedulers first
    if _is_problematic_scheduler(scheduler):
        scheduler_name = scheduler.__class__.__name__

        # Provide specific error messages based on scheduler type using isinstance
        if isinstance(scheduler, (diffusers.schedulers.EulerAncestralDiscreteScheduler,
                                 diffusers.schedulers.KDPM2AncestralDiscreteScheduler)):
            error_msg = (f"Scheduler {scheduler_name} is an ancestral sampler that adds noise "
                        f"stochastically at each step. Denoise range splitting disrupts this noise "
                        f"pattern. Consider using the non-ancestral version (e.g., EulerDiscreteScheduler "
                        f"instead of EulerAncestralDiscreteScheduler, or KDPM2DiscreteScheduler "
                        f"instead of KDPM2AncestralDiscreteScheduler).")
        elif isinstance(scheduler, diffusers.schedulers.DPMSolverSDEScheduler):
            error_msg = (f"Scheduler {scheduler_name} is a Stochastic Differential Equation (SDE) "
                        f"scheduler with complex stochastic state that doesn't split reliably. "
                        f"Consider using DPMSolverMultistepScheduler instead.")
        elif isinstance(scheduler, (diffusers.schedulers.HeunDiscreteScheduler,
                                   diffusers.schedulers.KDPM2DiscreteScheduler,
                                   diffusers.schedulers.DPMSolverSinglestepScheduler)):
            error_msg = (f"Scheduler {scheduler_name} produces black outputs and incorrect progress "
                        f"bars when used with denoise range splitting. Consider using EulerDiscreteScheduler, "
                        f"LMSDiscreteScheduler, or DPMSolverMultistepScheduler instead.")
        elif isinstance(scheduler, (diffusers.schedulers.DEISMultistepScheduler,
                                    diffusers.schedulers.UniPCMultistepScheduler)):
            error_msg = (f"Scheduler {scheduler_name} has complex multistep state that causes runtime "
                        f"errors with denoise range splitting. Consider using DPMSolverMultistepScheduler, "
                        f"EulerDiscreteScheduler, PNDMScheduler, or DDIMScheduler instead.")
        elif isinstance(scheduler, diffusers.schedulers.LCMScheduler):
            error_msg = (f"Scheduler {scheduler_name} uses timestep-dependent boundary conditions and consistency "
                        f"distillation that require the complete original timestep sequence from training. "
                        f"Splitting the timestep schedule breaks the learned consistency mapping and produces "
                        f"corrupted outputs. Consider using EulerDiscreteScheduler, DPMSolverMultistepScheduler, "
                        f"or DDIMScheduler instead.")
        elif isinstance(scheduler, diffusers.schedulers.DDPMWuerstchenScheduler):
            error_msg = (f"Scheduler {scheduler_name} is specific to Wuerstchen models and not "
                        f"compatible with denoise range splitting. Consider using DDIMScheduler "
                        f"or EulerDiscreteScheduler instead.")
        else:
            error_msg = (f"Scheduler {scheduler_name} has stochastic state that "
                        f"doesn't work reliably with denoise range splitting. Consider using "
                        f"DPMSolverMultistepScheduler, EulerDiscreteScheduler, or DDIMScheduler.")
        
        raise DenoiseRangeError(error_msg)
    
    if _is_flow_matching_scheduler(scheduler):
        # Flow matching schedulers (FlowMatchEulerDiscreteScheduler)
        yield from _apply_flow_matching_denoise_range(pipeline, start, end, inference_steps_ref)
    elif _is_sigma_based_scheduler(scheduler):
        # Sigma-based schedulers (Euler, Heun, K-DPM, LMS, EDM)
        yield from _apply_sigma_based_denoise_range(pipeline, start, end, inference_steps_ref)
    elif _is_dpm_solver_scheduler(scheduler):
        # DPM solver schedulers (excluding problematic ones already filtered out)
        yield from _apply_dpm_solver_denoise_range(pipeline, start, end, inference_steps_ref)
    elif _is_multistep_scheduler(scheduler):
        # Multistep schedulers (PNDM)
        yield from _apply_multistep_denoise_range(pipeline, start, end, inference_steps_ref)
    else:
        # Standard timestep-based schedulers (DDIM, DDPM, Wuerstchen, etc.)
        yield from _apply_standard_denoise_range(pipeline, start, end, inference_steps_ref)
