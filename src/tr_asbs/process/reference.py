"""Analytic Brownian and scalar linear Gaussian reference processes.

The backward base drift is -f in elapsed reverse time. h is an absolute
score-unit control; h=0 is NOT the stationary reverse OU process.
"""

import torch

from .sde import FwdBwdSDE, ZeroDriftSDE, _is_graph_data


def positions(data):
    return data.pos if _is_graph_data(data) else data


def coefficient(value, data):
    pos = positions(data)
    value = torch.as_tensor(value).to(pos)
    if _is_graph_data(data) and value.ndim:
        value = value.reshape(-1)[data.batch]
    while value.ndim < pos.ndim:
        value = value.unsqueeze(-1)
    return value


class BrownianReference(ZeroDriftSDE):
    def __init__(self, noise_schedule, is_mean_free=False):
        super().__init__(is_mean_free)
        self.noise_schedule = noise_schedule

    def diffusion(self, time):
        return self.noise_schedule.diffusion(time)

    def diffsquare_integral(self, time):
        return self.noise_schedule.variance(time)

    def inverse_variance(self, v):
        return self.noise_schedule.inverse_variance(v)

    def backward_step_diffusion(self, time, dt):
        process = getattr(self.noise_schedule, "process", None)
        if process is not None:
            return process.backward_step_diffusion(
                time, dt
            ) * self.noise_schedule.scale.to(time)
        return self.diffusion(time)


class LinearGaussianReference(FwdBwdSDE):
    """f(t,x)=-a(t)x. Supports constant OU or constant/linear-beta VP.

    OU: a=rate, g=diffusion_scale. VP: a=beta(t)/2,
    g=stationary_scale*sqrt(beta(t)). The VP source must have that scale.
    """

    def __init__(
        self,
        kind="ou",
        rate=1.0,
        diffusion_scale=1.0,
        beta_min=1.0,
        beta_max=1.0,
        stationary_scale=1.0,
        is_mean_free=False,
    ):
        super().__init__(is_mean_free)
        if (
            kind not in ("ou", "vp")
            or rate < 0
            or min(diffusion_scale, beta_min, beta_max, stationary_scale) <= 0
        ):
            raise ValueError("Invalid linear Gaussian reference parameters")
        self.kind = kind
        for key, value in dict(
            rate=rate,
            diffusion_scale=diffusion_scale,
            beta_min=beta_min,
            beta_max=beta_max,
            stationary_scale=stationary_scale,
        ).items():
            self.register_buffer(key, torch.tensor(value, dtype=torch.double))

    def _time(self, t):
        return torch.as_tensor(t, device=self.rate.device, dtype=torch.double)

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * self._time(t)

    def integrated_rate(self, s, t):
        s, t = self._time(s), self._time(t)
        if self.kind == "ou":
            return self.rate * (t - s)
        return (
            self.beta_min * (t - s)
            + (self.beta_max - self.beta_min) * (t * t - s * s) / 2
        ) / 2

    def transition_coefficients(self, s, t, backward=False):
        """(Phi, K, Q) for frozen score-unit control over the interval."""
        integral = self.integrated_rate(s, t)
        signed = integral if backward else -integral
        phi = torch.exp(signed)
        if self.kind == "vp":
            k = 2 * self.stationary_scale.square() * torch.abs(torch.expm1(signed))
            q = self.stationary_scale.square() * torch.abs(torch.expm1(2 * signed))
        elif self.rate.item() == 0:
            k = q = self.diffusion_scale.square() * (self._time(t) - self._time(s))
        else:
            k = (
                self.diffusion_scale.square()
                * torch.abs(torch.expm1(signed))
                / self.rate
            )
            q = (
                self.diffusion_scale.square()
                * torch.abs(torch.expm1(2 * signed))
                / (2 * self.rate)
            )
        return phi, k, q

    def diffusion(self, time):
        g = (
            self.stationary_scale * self.beta(time).sqrt()
            if self.kind == "vp"
            else self.diffusion_scale + self._time(time) * 0
        )
        return g.to(time)

    def drift(self, time, data):
        a = self.beta(time) / 2 if self.kind == "vp" else self.rate
        return -coefficient(a, data) * positions(data)

    def bwd_drift(self, time, data):
        return -self.drift(time, data)

    def diffsquare_integral(self, time):
        t = self._time(time)
        if self.kind == "vp":
            return self.stationary_scale.square() * 2 * self.integrated_rate(0, t)
        return self.diffusion_scale.square() * t

    @property
    def total_variance(self):
        """Integrated diffusion, distinct from the OU transition covariance."""
        return self.diffsquare_integral(1)

    def reparametrize_time(self, time):
        return self.diffsquare_integral(time) / self.total_variance

    def inverse_variance(self, v):
        lo, hi = torch.zeros_like(v), torch.ones_like(v)
        for _ in range(56):
            mid = (lo + hi) / 2
            left = self.diffsquare_integral(mid) < v
            lo, hi = torch.where(left, mid, lo), torch.where(left, hi, mid)
        return (lo + hi) / 2

    def adjoint_transport(self, time):
        return torch.exp(-self.integrated_rate(time, 1))

    def reverse_conditional(self, data_t, s, t, initial_variance):
        """True uncontrolled Gaussian reverse conditional for a centered source.

        This differs from the learned absolute-h elapsed-time solver kernel.
        Under a stationary source, its mean is Phi(s,t)*x_t, and Q equals the
        forward reference Q. Used as an analytic oracle, not as q_h.
        """
        from .transition import GaussianTransition, with_positions

        phi_0s, _, q_0s = self.transition_coefficients(0, s)
        phi_st, _, q_st = self.transition_coefficients(s, t)
        variance_s = phi_0s.square() * initial_variance + q_0s
        variance_t = phi_st.square() * variance_s + q_st
        gain = variance_s * phi_st / variance_t
        mean = with_positions(data_t, coefficient(gain, data_t) * positions(data_t))
        variance = variance_s * q_st / variance_t
        return GaussianTransition(mean, variance, self.is_mean_free)

    @torch.no_grad()
    def sample_posterior(self, time, data_0, data_1):
        phi_0t, _, q_0t = self.transition_coefficients(0, time)
        phi_t1, _, q_t1 = self.transition_coefficients(time, 1)
        q_01 = phi_t1.square() * q_0t + q_t1
        gain = q_0t * phi_t1 / q_01
        mean = coefficient(phi_0t, data_0) * positions(data_0)
        mean = mean + coefficient(gain, data_0) * (
            positions(data_1) - coefficient(phi_t1, data_0) * mean
        )
        var = (q_0t - gain.square() * q_01).clamp_min(0)
        sample = mean + coefficient(var.sqrt(), data_0) * self.randn_like(data_0)
        if _is_graph_data(data_0):
            result = data_0.clone()
            result.pos = sample
            return result
        return sample

    @torch.no_grad()
    def cond_score_t0(self, time, data_0, data_t):
        phi, _, q = self.transition_coefficients(0, time)
        return (
            coefficient(phi, data_t) * positions(data_0) - positions(data_t)
        ) / coefficient(q, data_t)

    @torch.no_grad()
    def cond_score_1t(self, time, data_1, data_t):
        phi, _, q = self.transition_coefficients(time, 1)
        return (
            coefficient(phi, data_t)
            * (positions(data_1) - coefficient(phi, data_t) * positions(data_t))
            / coefficient(q, data_t)
        )
