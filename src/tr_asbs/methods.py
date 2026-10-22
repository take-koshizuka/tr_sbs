"""Paper-facing sampler names and their compatible training entrypoints.

This registry intentionally uses only the standard library. Existing module,
class and Hydra config paths remain the implementation and checkpoint API.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class Method:
    """A public method name with the settings that define its identity."""

    key: str
    name: str
    module: str
    description: str
    gamma_schedule: str | None = None
    algorithm: str | None = None
    variant_of: str | None = None
    atomic_module: str | None = None


_METHODS = (
    Method(
        "asbs",
        "ASBS",
        "tr_asbs.experiment.train_asbs",
        "Original adjoint Schrödinger bridge sampler.",
    ),
    Method(
        "gamma-asbs",
        "Gamma-ASBS",
        "tr_asbs.experiment.train_gamma_asbs",
        "Gamma-ASBS with a fixed gamma from the experiment or an override.",
        gamma_schedule="fixed",
    ),
    Method(
        "tr-asbs",
        "TR-ASBS",
        "tr_asbs.experiment.train_gamma_asbs",
        "Trust-region variant of Gamma-ASBS, without importance weighting.",
        gamma_schedule="trust_region",
        variant_of="Gamma-ASBS",
    ),
    Method(
        "asbs-iw",
        "ASBS-IW",
        "tr_asbs.experiment.train_gamma_asbs_iw",
        "Importance-weighted ASBS with a fixed gamma.",
        gamma_schedule="fixed",
        atomic_module="tr_asbs.experiment.train_gamma_asbs_iw_ala2",
    ),
    Method(
        "tr-sbs",
        "TR-SBS",
        "tr_asbs.experiment.train_gamma_asbs_iw",
        "Importance-weighted sampler with trust-region gamma selection.",
        gamma_schedule="trust_region",
        atomic_module="tr_asbs.experiment.train_gamma_asbs_iw_ala2",
    ),
    Method(
        "pis",
        "PIS",
        "tr_asbs.experiment.train_variational",
        "Path integral sampler for flat vector targets.",
        algorithm="pis",
    ),
    Method(
        "dis",
        "DIS",
        "tr_asbs.experiment.train_variational",
        "Time-Reversed Diffusion Sampler for flat vector targets.",
        algorithm="dis",
    ),
)

METHODS: Mapping[str, Method] = MappingProxyType(
    {method.key: method for method in _METHODS}
)


def get_method(name: str) -> Method:
    """Resolve case-insensitive paper names, also accepting underscores."""

    key = name.strip().lower().replace("_", "-")
    try:
        return METHODS[key]
    except KeyError as error:
        raise ValueError(
            f"Unknown method {name!r}; choose one of {', '.join(METHODS)}."
        ) from error
