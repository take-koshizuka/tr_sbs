"""Opt-in Hydra presets; replace whole configs to avoid stale VE arguments."""

import math
from omegaconf import open_dict


def configure_reference(cfg):
    preset = getattr(cfg, "reference_preset", None)
    if preset is None:
        return
    scale = float(cfg.source.get("scale", 1.0))
    variance = getattr(cfg, "reference_variance", None)
    if variance is not None and (not math.isfinite(variance) or variance <= 0):
        raise ValueError("reference_variance must be finite and positive")
    steps = int(getattr(cfg, "reference_steps", 128))
    solver = getattr(cfg, "reference_solver", None)
    noise = "tr_asbs.process.noise_schedule."
    ref = "tr_asbs.process.reference."
    if preset.startswith("brownian_"):
        shape = preset.removeprefix("brownian_")
        if shape == "cosine":
            schedule = dict(
                _target_=noise + "LegacyNoiseSchedule",
                total_variance=variance,
                process=dict(
                    _target_="tr_asbs.process.sde.PISCosineSDE",
                    sigma_min=cfg.reference_g_min,
                    sigma_max=cfg.reference_g_max,
                    num_steps=cfg.reference_noise_knots,
                ),
            )
        elif shape == "edm":
            schedule = dict(
                _target_=noise + "LegacyNoiseSchedule",
                total_variance=variance,
                process=dict(
                    _target_="tr_asbs.process.sde.EDMSDE",
                    sigma_min=cfg.reference_g_min,
                    sigma_max=cfg.reference_g_max,
                    rho=cfg.reference_edm_rho,
                ),
            )
        elif shape in ("constant", "linear", "geometric"):
            schedule = dict(
                _target_=noise + "AnalyticNoiseSchedule",
                kind=shape,
                g_min=cfg.reference_g_min,
                g_max=cfg.reference_g_max,
                total_variance=variance,
            )
        else:
            raise ValueError(f"Unknown Brownian noise preset: {shape}")
        base = dict(_target_=ref + "BrownianReference", noise_schedule=schedule)
        solver = solver or "diffusion_euler"
    elif preset in ("ou", "vp"):
        base = dict(_target_=ref + "LinearGaussianReference", kind=preset)
        if preset == "ou":
            base.update(
                rate=cfg.reference_rate,
                diffusion_scale=(variance or cfg.reference_g_max**2) ** 0.5,
            )
        else:
            beta_min, beta_max = cfg.vp_beta_min, cfg.vp_beta_max
            if variance is not None:
                factor = variance / (scale**2 * (beta_min + beta_max) / 2)
                beta_min, beta_max = beta_min * factor, beta_max * factor
            base.update(beta_min=beta_min, beta_max=beta_max, stationary_scale=scale)
        solver = solver or "linear_exponential"
    else:
        raise ValueError(f"Unknown reference_preset: {preset}")
    solvers = dict(
        euler="EulerMaruyamaIntegrator",
        legacy="ExponentialIntegrator",
        diffusion_euler="DiffusionTimeEulerIntegrator",
        linear_exponential="LinearExponentialIntegrator",
    )
    if solver not in solvers:
        raise ValueError(f"Unknown reference_solver: {solver}")
    integrator = dict(
        _target_="tr_asbs.process.integrator." + solvers[solver],
        steps=steps,
        grid=cfg.reference_grid,
    )
    if solver == "linear_exponential":
        integrator["control_evaluation"] = cfg.reference_control_evaluation
    with open_dict(cfg):
        cfg.base_sde, cfg.integrator = base, integrator
