import torch

from dataclasses import dataclass
from typing import Optional, Union, Tuple

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from . import exceptions


@dataclass
class SchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


@dataclass
class EulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


def lagrange_skip(t_points, x_values, t_eval):
    P_of_t = torch.zeros_like(x_values[0])
    n = len(t_points)

    for i in range(n):
        term = x_values[i].clone()
        for j in range(n):
            if j != i:
                factor = (t_eval - t_points[j]) / (t_points[i] - t_points[j])
                term = term * factor
        P_of_t += term

    return P_of_t


def patch_solver(solver_class):
    class PatchedEulerDiscreteScheduler(solver_class):
        def step(
                self,
                model_output: torch.FloatTensor,
                timestep: Union[float, torch.FloatTensor],
                sample: torch.FloatTensor,
                s_churn: float = 0.0,
                s_tmin: float = 0.0,
                s_tmax: float = float("inf"),
                s_noise: float = 1.0,
                generator: Optional[torch.Generator] = None,
                return_dict: bool = True,
        ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:

            if self.step_index is None:
                self._init_step_index(timestep)

            sigma = self.sigmas[self.step_index]
            gamma = min(s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
            noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device,
                                 generator=generator)

            eps = noise * s_noise
            sigma_hat = sigma * (gamma + 1)

            if gamma > 0:
                sample = sample + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5

            # == Conduct the step skipping identified from last step == #
            for i in range(1):
                self._cache_bus.prev_epsilon_guided[i] = self._cache_bus.prev_epsilon_guided[i + 1]
            self._cache_bus.prev_epsilon_guided[-1] = model_output.clone()

            if self._cache_bus.skip_this_step and self._cache_bus.pred_m_m_1 is not None:
                pred_original_sample = self._cache_bus.pred_m_m_1
                self._cache_bus.skip_this_step = False
            else:
                if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
                    pred_original_sample = model_output
                elif self.config.prediction_type == "epsilon":
                    pred_original_sample = sample - sigma_hat * model_output
                elif self.config.prediction_type == "v_prediction":
                    pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample / (sigma ** 2 + 1))
                else:
                    raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`")

            # == Patch begin
            N = self.timesteps.shape[0]
            delta = 1 / N

            sigma_t, sigma_s0, sigma_s1 = (
                self.sigmas[self.step_index + 1] / (torch.sqrt(1 + self.sigmas[self.step_index + 1] ** 2)),
                self.sigmas[self.step_index] / (torch.sqrt(1 + self.sigmas[self.step_index] ** 2)),
                self.sigmas[self.step_index - 1] / (torch.sqrt(1 + self.sigmas[self.step_index - 1] ** 2)),
            )

            self.betas = self.betas.to('cuda')

            beta_n = self.betas[int(self.timesteps[self.step_index])]
            s_alpha_cumprod_n = sigma_s0 / self.sigmas[self.step_index]
            s_alpha_cumprod_n_m1 = sigma_t / self.sigmas[self.step_index + 1]

            epsilon_0, epsilon_1 = self._cache_bus.prev_epsilon_guided[-1], self._cache_bus.prev_epsilon_guided[-2]
            m0 = pred_original_sample / s_alpha_cumprod_n

            if self._cache_bus._tome_info['args']['lagrange_term'] != 0:
                use_lagrange = True

                lagrange_term = self._cache_bus._tome_info['args']['lagrange_term']
                lagrange_step = self._cache_bus._tome_info['args']['lagrange_step']
                lagrange_int = self._cache_bus._tome_info['args']['lagrange_int']

                if self._step_index % lagrange_int == 1:
                    for i in range(lagrange_term - 1):
                        self._cache_bus.lagrange_x0[i] = self._cache_bus.lagrange_x0[i + 1]
                        self._cache_bus.lagrange_step[i] = self._cache_bus.lagrange_step[i + 1]
                    self._cache_bus.lagrange_x0[-1] = m0
                    self._cache_bus.lagrange_step[-1] = self._step_index

            else: use_lagrange = False

            # 2. Convert to an ODE derivative
            derivative = (sample - pred_original_sample) / sigma_hat
            dt = self.sigmas[self.step_index + 1] - sigma_hat
            prev_sample = sample + derivative * dt

            # == Criteria == #
            if self.config.prediction_type == "epsilon":
                f = (- 0.5 * beta_n * N * sample * s_alpha_cumprod_n) + (0.5 * beta_n * N / sigma_s0) * epsilon_0
            elif self.config.prediction_type == "v_prediction":
                f = (0.5 * beta_n * N / sigma_s0) * s_alpha_cumprod_n * epsilon_0
            else:
                raise RuntimeError

            if self._cache_bus.prev_f[0] is not None:
                max_interval = self._cache_bus._tome_info['args']['max_interval']
                acc_range = self._cache_bus._tome_info['args']['acc_range']

                pred_prev_sample = (sample * s_alpha_cumprod_n - 0.625 * delta * f - 0.75 * delta * self._cache_bus.prev_f[-1] + 0.375 * delta * self._cache_bus.prev_f[-2]) / s_alpha_cumprod_n_m1
                residual_factor = (prev_sample - pred_prev_sample)
                momentum = residual_factor * ((f - self._cache_bus.prev_f[-1]) - (self._cache_bus.prev_f[-1] - self._cache_bus.prev_f[-2]))

                momentum_mean = momentum.mean()
                self._cache_bus.rel_momentum_list.append((momentum_mean.item()))
                lagrange_this_step = False

                if self._cache_bus.cons_skip >= max_interval:
                    self._cache_bus.skip_this_step = False
                    self._cache_bus.cons_skip = 0

                elif not use_lagrange and (momentum_mean <= 0 and self._step_index in range(acc_range[0], acc_range[1])):
                    # == Here we skip step with psi / x0 interpolation == #
                    self._cache_bus.skip_this_step = True
                    self._cache_bus.cons_skip += 1
                    self._cache_bus.skipping_path.append(self._step_index)

                elif use_lagrange and (momentum_mean <= 0 and self._step_index in range(acc_range[0], lagrange_step)):
                    # == Here we skip step with psi / x0 interpolation == #
                    self._cache_bus.skip_this_step = True
                    self._cache_bus.cons_skip += 1
                    self._cache_bus.skipping_path.append(self._step_index)

                elif use_lagrange and (self._step_index % lagrange_int != 0 and self._step_index in range(lagrange_step, acc_range[1])):
                    # == Here we skip step using lagrange == #
                    self._cache_bus.skip_this_step = True
                    self._cache_bus.cons_skip += 1
                    self._cache_bus.skipping_path.append(self._step_index)
                    lagrange_this_step = True

                else:
                    self._cache_bus.skip_this_step = False
                    self._cache_bus.cons_skip = 0

                # Here we conduct token-wise pruning:
                if not self._cache_bus.skip_this_step:
                    max_fix = self._cache_bus._tome_info['args']['max_fix']

                    momentum_mean = momentum.mean(dim=1)
                    momentum_flat = momentum_mean.view(-1)
                    mask = momentum_flat >= 0
                    num_mask = mask.sum().item()
                    score = torch.zeros_like(momentum_flat)

                    if num_mask > max_fix:
                        _, topk_indices = torch.topk(momentum_flat, max_fix, largest=True, sorted=False)
                        score[topk_indices] = 1.0
                    else:
                        score[mask] = 1.0

                    self._cache_bus.temporal_score = score.view_as(momentum_mean)

            # upon completion increase step index by one
            self._step_index += 1

            if self._cache_bus.skip_this_step:
                sigma = self.sigmas[self.step_index]
                gamma = min(s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
                sigma_hat = sigma * (gamma + 1)

                if not lagrange_this_step:
                    epsilon_m1 = epsilon_0

                    if self.config.prediction_type == "epsilon":
                        pred_m_m_1 = pred_prev_sample - sigma_hat * epsilon_m1
                        self._cache_bus.pred_m_m_1 = pred_m_m_1.clone()

                    elif self.config.prediction_type == "v_prediction":
                        pred_m_m_1 = epsilon_m1 * (- sigma / (sigma ** 2 + 1) ** 0.5) + (prev_sample / (sigma ** 2 + 1))
                        self._cache_bus.pred_m_m_1 = pred_m_m_1.clone()

                    else:
                        raise exceptions.SADAUnsupportedError(
                            "Unsupported prediction type for SADA acceleration. "
                            "Only 'epsilon' and 'v_prediction' are supported."
                        )

                else:
                    pred_m_m_1 = lagrange_skip(self._cache_bus.lagrange_step, self._cache_bus.lagrange_x0, self._step_index)
                    self._cache_bus.pred_m_m_1 = (pred_m_m_1 * s_alpha_cumprod_n_m1).clone()

            for i in range(1):
                self._cache_bus.prev_f[i] = self._cache_bus.prev_f[i + 1]
            self._cache_bus.prev_f[-1] = f

            if not return_dict:
                return (prev_sample,)

            return EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


    class PatchedDPMSolverMultistepScheduler(solver_class):
        def step(
                self,
                model_output: torch.Tensor,
                timestep: Union[int, torch.Tensor],
                sample: torch.Tensor,
                generator=None,
                variance_noise: Optional[torch.Tensor] = None,
                return_dict: bool = True,
        ) -> Union[SchedulerOutput, Tuple]:
            if self.step_index is None:
                self._init_step_index(timestep)

            # Improve numerical stability for small number of steps
            lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
                    self.config.euler_at_final
                    or (self.config.lower_order_final and len(self.timesteps) < 15)
                    or self.config.final_sigmas_type == "zero"
            )
            lower_order_second = (
                    (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(
                self.timesteps) < 15
            )

            # == Cache epsilon before Data Reconstruction == #
            for i in range(1):
                self._cache_bus.prev_epsilon_guided[i] = self._cache_bus.prev_epsilon_guided[i + 1]
            self._cache_bus.prev_epsilon_guided[-1] = model_output.clone()

            # == Conduct the step skipping identified from last step == #
            if self._cache_bus.skip_this_step and self._cache_bus.pred_m_m_1 is not None:
                model_output = self._cache_bus.pred_m_m_1
                self._cache_bus.skip_this_step = False
            else:
                model_output = self.convert_model_output(model_output, sample=sample)

            if self._cache_bus._tome_info['args']['lagrange_term'] != 0:
                use_lagrange = True

                lagrange_term = self._cache_bus._tome_info['args']['lagrange_term']
                lagrange_step = self._cache_bus._tome_info['args']['lagrange_step']
                lagrange_int = self._cache_bus._tome_info['args']['lagrange_int']

                if self._step_index % lagrange_int == 1:
                    for i in range(lagrange_term - 1):
                        self._cache_bus.lagrange_x0[i] = self._cache_bus.lagrange_x0[i + 1]
                        self._cache_bus.lagrange_step[i] = self._cache_bus.lagrange_step[i + 1]
                    self._cache_bus.lagrange_x0[-1] = model_output
                    self._cache_bus.lagrange_step[-1] = self._step_index

            else: use_lagrange = False

            sigma_t, sigma_s0, sigma_s1 = (
                self.sigmas[self.step_index + 1],
                self.sigmas[self.step_index],
                self.sigmas[self.step_index - 1],
            )

            N = self.timesteps.shape[0]
            delta = 1 / N  # step size correlates with number of inference step

            alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
            beta_n = self.betas[self.timesteps[self.step_index]]
            s_alpha_cumprod_n = self.alpha_t[self.timesteps[self.step_index]]
            epsilon_0, epsilon_1 = self._cache_bus.prev_epsilon_guided[-1], self._cache_bus.prev_epsilon_guided[-2]

            for i in range(self.config.solver_order - 1):
                self.model_outputs[i] = self.model_outputs[i + 1]
            self.model_outputs[-1] = model_output

            # Upcast to avoid precision issues when computing prev_sample
            sample = sample.to(torch.float32)
            if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
                noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=torch.float32
                )
            elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
                noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
            else:
                noise = None

            if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
                prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
            elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
                prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
            else:
                prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample, noise=noise)

            if self.lower_order_nums < self.config.solver_order:
                self.lower_order_nums += 1

            # Cast sample back to expected dtype
            prev_sample = prev_sample.to(model_output.dtype)

            # == Criteria == #
            if self.config.prediction_type == "epsilon":
                f = (- 0.5 * beta_n * N * sample) + (0.5 * beta_n * N / sigma_s0) * epsilon_0
            elif self.config.prediction_type == "v_prediction":
                f = (0.5 * beta_n * N / sigma_s0) * s_alpha_cumprod_n * epsilon_0
            else: raise RuntimeError

            if self._cache_bus.prev_f[0] is not None:
                max_interval = self._cache_bus._tome_info['args']['max_interval']
                acc_range = self._cache_bus._tome_info['args']['acc_range']

                pred_prev_sample = sample - 0.625 * delta * f - 0.75 * delta * self._cache_bus.prev_f[-1] + 0.375 * delta * self._cache_bus.prev_f[-2]
                residual_factor = (prev_sample - pred_prev_sample)
                momentum = residual_factor * ((f - self._cache_bus.prev_f[-1]) - (self._cache_bus.prev_f[-1] - self._cache_bus.prev_f[-2]))

                momentum_mean = momentum.mean()
                self._cache_bus.rel_momentum_list.append((momentum_mean.item()))
                lagrange_this_step = False

                if self._cache_bus.cons_skip >= max_interval:
                    self._cache_bus.skip_this_step = False
                    self._cache_bus.cons_skip = 0

                elif not use_lagrange and (momentum_mean <= 0 and self._step_index in range(acc_range[0], acc_range[1])):
                    # == Here we skip step with psi / x0 interpolation == #
                    self._cache_bus.skip_this_step = True
                    self._cache_bus.cons_skip += 1
                    self._cache_bus.skipping_path.append(self._step_index)

                elif use_lagrange and (momentum_mean <= 0 and self._step_index in range(acc_range[0], lagrange_step)):
                    # == Here we skip step with psi / x0 interpolation == #
                    self._cache_bus.skip_this_step = True
                    self._cache_bus.cons_skip += 1
                    self._cache_bus.skipping_path.append(self._step_index)

                elif use_lagrange and (self._step_index % lagrange_int != 0 and self._step_index in range(lagrange_step, acc_range[1])):
                    # == Here we skip step using lagrange == #
                    self._cache_bus.skip_this_step = True
                    self._cache_bus.cons_skip += 1
                    self._cache_bus.skipping_path.append(self._step_index)
                    lagrange_this_step = True

                else:
                    self._cache_bus.skip_this_step = False
                    self._cache_bus.cons_skip = 0

                # Here we conduct token-wise pruning:
                if not self._cache_bus.skip_this_step:
                    # == Define the masking on pruning
                    max_fix = self._cache_bus._tome_info['args']['max_fix']

                    momentum_mean = momentum.mean(dim=1)
                    momentum_flat = momentum_mean.view(-1)
                    mask = momentum_flat >= 0
                    num_mask = mask.sum().item()
                    score = torch.zeros_like(momentum_flat)

                    if num_mask > max_fix:
                        _, topk_indices = torch.topk(momentum_flat, max_fix, largest=True, sorted=False)
                        score[topk_indices] = 1.0
                    else:
                        score[mask] = 1.0

                    self._cache_bus.temporal_score = score.view_as(momentum_mean)

            # upon completion increase step index by one
            self._step_index += 1

            # == approximate next step data reconstruction == #
            if self._cache_bus.skip_this_step:
                if not lagrange_this_step:
                    # interpolate on trajectory (x_0)
                    pred_prev_sample = pred_prev_sample.to(model_output.dtype)
                    pred_m_m_1 = self.convert_model_output(epsilon_0, sample=pred_prev_sample)
                    self._cache_bus.pred_m_m_1 = pred_m_m_1.clone()

                else:
                    pred_m_m_1 = lagrange_skip(self._cache_bus.lagrange_step, self._cache_bus.lagrange_x0, self._step_index)
                    self._cache_bus.pred_m_m_1 = pred_m_m_1.clone()

            # update on y (f)
            for j in range(1):
                self._cache_bus.prev_f[j] = self._cache_bus.prev_f[j + 1]
            self._cache_bus.prev_f[-1] = f

            if not return_dict:
                return (prev_sample,)

            return SchedulerOutput(prev_sample=prev_sample)


    class PatchedFlowMatchEulerDiscreteScheduler(solver_class):
        def step(
                self,
                model_output: torch.FloatTensor,
                timestep: Union[float, torch.FloatTensor],
                sample: torch.FloatTensor,
                s_churn: float = 0.0,
                s_tmin: float = 0.0,
                s_tmax: float = float("inf"),
                s_noise: float = 1.0,
                generator: Optional[torch.Generator] = None,
                return_dict: bool = True,
        ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
            if self.step_index is None:
                self._init_step_index(timestep)

            # Upcast to avoid precision issues when computing prev_sample
            sample = sample.to(torch.float32)

            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]

            # Euler method step
            if self._cache_bus.skip_this_step and self._cache_bus.prev_f[-3] is not None:
                f_1 = self._cache_bus.prev_f[-2] + (self._cache_bus.prev_f[-2] - self._cache_bus.prev_f[-3])  # interpolation

                delta = (self.timesteps[self.step_index] - self.timesteps[self.step_index + 1]) / 1000.0
                prev_sample = sample - delta * f_1  # same idea with above, but a little different in terms of implementation

                self._cache_bus.skip_this_step = False

            else:
                f_1 = self._cache_bus.prev_f[-1]
                prev_sample = sample + (sigma_next - sigma) * model_output


            if self._cache_bus._tome_info['args']['lagrange_term'] != 0:
                use_lagrange = True

                lagrange_term = self._cache_bus._tome_info['args']['lagrange_term']
                lagrange_step = self._cache_bus._tome_info['args']['lagrange_step']
                lagrange_int = self._cache_bus._tome_info['args']['lagrange_int']

            else: use_lagrange = False

            if self._cache_bus.prev_f[0] is not None:
                max_interval = self._cache_bus._tome_info['args']['max_interval']
                acc_range = self._cache_bus._tome_info['args']['acc_range']

                momentum =  - (f_1 - self._cache_bus.prev_f[-2]) + (self._cache_bus.prev_f[-2] - self._cache_bus.prev_f[-3])
                momentum_mean = momentum.mean()

                if self._cache_bus.cons_skip >= max_interval:
                    self._cache_bus.skip_this_step = False
                    self._cache_bus.cons_skip = 0

                elif not use_lagrange and (
                        momentum_mean <= 0 and self._step_index in range(acc_range[0], acc_range[1])):
                    # == Here we skip step with psi / x0 interpolation == #
                    self._cache_bus.skip_this_step = True
                    self._cache_bus.cons_skip += 1
                    self._cache_bus.skipping_path.append(self._step_index)

                elif use_lagrange and (momentum_mean <= 0 and self._step_index in range(acc_range[0], lagrange_step)):
                    # == Here we skip step with psi / x0 interpolation == #
                    self._cache_bus.skip_this_step = True
                    self._cache_bus.cons_skip += 1
                    self._cache_bus.skipping_path.append(self._step_index)

                elif use_lagrange and (self._step_index % lagrange_int != 0 and self._step_index in range(lagrange_step, acc_range[1])):
                    # == Here we skip step using lagrange == #
                    self._cache_bus.skip_this_step = True
                    self._cache_bus.cons_skip += 1
                    self._cache_bus.skipping_path.append(self._step_index)

                else:
                    self._cache_bus.skip_this_step = False
                    self._cache_bus.cons_skip = 0

                # Here we conduct token-wise pruning:
                if not self._cache_bus.skip_this_step:
                    # == Define the masking on pruning
                    max_fix = self._cache_bus._tome_info['args']['max_fix']

                    momentum_mean = momentum.mean(dim=-1)
                    momentum_flat = momentum_mean.view(-1)
                    mask = momentum_flat >= 0
                    num_mask = mask.sum().item()
                    score = torch.zeros_like(momentum_flat)

                    if num_mask > max_fix:
                        _, topk_indices = torch.topk(momentum_flat, max_fix, largest=True, sorted=False)
                        score[topk_indices] = 1.0
                    else:
                        score[mask] = 1.0

                    self._cache_bus.temporal_score = score.view_as(momentum_mean)

            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)

            # upon completion increase step index by one
            self._step_index += 1

            if not return_dict:
                return (prev_sample,)

            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    if solver_class.__name__ == "EulerDiscreteScheduler":
        return PatchedEulerDiscreteScheduler
    elif solver_class.__name__ == "DPMSolverMultistepScheduler":
        return PatchedDPMSolverMultistepScheduler
    elif solver_class.__name__ == "FlowMatchEulerDiscreteScheduler":
        return PatchedFlowMatchEulerDiscreteScheduler
    else:
        raise exceptions.SADAUnsupportedError(
            f"Unsupported scheduler class '{solver_class.__name__}' for SADA acceleration. "
            f"Only EulerDiscreteScheduler, DPMSolverMultistepScheduler, and FlowMatchEulerDiscreteScheduler are supported."
        )
