"""
Chebyshev feature forecaster used by Spectrum.

Adapted from https://github.com/hanjq17/Spectrum (MIT). Each cached denoiser
feature is treated as a function of normalized diffusion time and fit with
ridge regression on Chebyshev polynomials. A one-step Newton extrapolation is
blended in with weight ``w``.
"""

import torch
import torch.nn as nn


def _flatten(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    shape = x.shape
    return x.reshape(1, -1), shape


def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return x_flat.reshape(shape)


class ChebyshevForecaster(nn.Module):
    """
    Ridge regression onto Chebyshev polynomials ``T_0 .. T_M``.

    :param M: Polynomial degree. The design matrix has ``M + 1`` columns.
    :param K: How many recent ``(time, feature)`` pairs to keep.
    :param lam: Ridge penalty.
    """

    def __init__(self, M: int = 4, K: int = 100, lam: float = 0.1):
        super().__init__()
        if K < M + 2:
            raise ValueError('Spectrum history length must be at least degree + 2.')
        self.M = M
        self.K = K
        self.lam = lam
        self.register_buffer('t_buf', torch.empty(0))
        self._H_buf: torch.Tensor | None = None
        self._shape: torch.Size | None = None
        self._coef: torch.Tensor | None = None

    @property
    def P(self) -> int:
        return self.M + 1

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        # Callers pass time already scaled into [0, 1].
        return 2.0 * t - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)
        count = taus.shape[0]
        columns = [torch.ones((count, 1), device=taus.device, dtype=taus.dtype)]
        if self.M == 0:
            return columns[0]
        columns.append(taus)
        for _ in range(2, self.M + 1):
            columns.append(2 * taus * columns[-1] - columns[-2])
        return torch.cat(columns[:self.M + 1], dim=1)

    def update(self, t: float, h: torch.Tensor) -> None:
        """Append one observation and drop the fitted coefficients."""
        h_flat, shape = _flatten(h.detach())
        h_flat = h_flat.to(dtype=torch.float32)
        device = h_flat.device
        t_value = torch.as_tensor(t, dtype=torch.float32, device=device)

        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            raise ValueError('Spectrum feature shape changed during a sample.')

        if self.t_buf.numel() and self.t_buf.device != device:
            self.t_buf = self.t_buf.to(device)
            self._H_buf = None if self._H_buf is None else self._H_buf.to(device)

        if self.t_buf.numel() == 0 or self._H_buf is None:
            self.t_buf = t_value.reshape(1)
            self._H_buf = h_flat
        else:
            self.t_buf = torch.cat([self.t_buf, t_value.reshape(1)], dim=0)
            self._H_buf = torch.cat([self._H_buf, h_flat], dim=0)
            if self.t_buf.numel() > self.K:
                self.t_buf = self.t_buf[-self.K:]
                self._H_buf = self._H_buf[-self.K:]
        self._coef = None

    def ready(self) -> bool:
        return self.t_buf.numel() >= 2 and self._H_buf is not None

    def _fit(self) -> None:
        if self._coef is not None:
            return
        taus = self._taus(self.t_buf)
        design = self._build_design(taus).to(torch.float32)
        history = self._H_buf.to(torch.float32)
        width = design.shape[1]
        gram = design.transpose(0, 1) @ design
        gram = gram + self.lam * torch.eye(width, device=gram.device, dtype=gram.dtype)
        right_hand = design.transpose(0, 1) @ history
        try:
            factor = torch.linalg.cholesky(gram)
        except RuntimeError:
            jitter = 1e-6 * gram.diag().mean()
            factor = torch.linalg.cholesky(
                gram + jitter * torch.eye(width, device=gram.device, dtype=gram.dtype))
        self._coef = torch.cholesky_solve(right_hand, factor)

    @torch.no_grad()
    def predict(self, t_star: float) -> torch.Tensor:
        if self._shape is None:
            raise RuntimeError('Spectrum has no cached features to predict from.')
        self._fit()
        device = self.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=torch.float32, device=device)
        row = self._build_design(self._taus(t_star).reshape(1))
        flat = row.to(torch.float32) @ self._coef
        return _unflatten(flat, self._shape)

    def clear(self) -> None:
        self.t_buf = torch.empty(0, device=self.t_buf.device)
        self._H_buf = None
        self._shape = None
        self._coef = None


class SpectrumPredictor(nn.Module):
    """
    ``(1 - w) * taylor + w * chebyshev``.

    ``w`` near 1 trusts the global fit. ``w`` near 0 trusts the last step.
    """

    def __init__(self, chebyshev: ChebyshevForecaster, w: float = 0.5):
        super().__init__()
        self.cheb = chebyshev
        self.w = w

    def update_w(self, w: float) -> None:
        self.w = w

    def update(self, t: float, h: torch.Tensor) -> None:
        self.cheb.update(t, h)

    def ready(self) -> bool:
        return self.cheb.ready()

    def clear(self) -> None:
        self.cheb.clear()

    @torch.no_grad()
    def _taylor(self, t_star: torch.Tensor) -> torch.Tensor:
        history = self.cheb._H_buf
        times = self.cheb.t_buf
        if history is None or times.numel() < 2:
            return _unflatten(history[-1:], self.cheb._shape)
        latest = history[-1]
        previous = history[-2]
        step = (times[-1] - times[-2]).clamp_min(1e-8)
        fraction = ((t_star - times[-1]) / step).to(latest.dtype)
        extrapolated = latest + fraction * (latest - previous)
        return _unflatten(extrapolated.unsqueeze(0), self.cheb._shape)

    @torch.no_grad()
    def predict(self, t_star: float) -> torch.Tensor:
        spectral = self.cheb.predict(t_star)
        device = self.cheb.t_buf.device
        t_value = torch.as_tensor(t_star, dtype=torch.float32, device=device)
        local = self._taylor(t_value)
        return (1.0 - self.w) * local + self.w * spectral
