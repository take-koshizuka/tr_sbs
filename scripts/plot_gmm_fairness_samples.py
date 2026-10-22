#!/usr/bin/env python3
"""Plot like-for-like 2D views of retained GMM SD-best samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tr_asbs.potential.gmm import GMM


ASBS_PATTERN = re.compile(
    r"gmm_(mode_k40|dim_d100)_"
    r"(asbs|gamma_asbs|gamma_asbs_iw_fixed|"
    r"gamma_asbs_iw_trust_region_0p[13])_seed([012])"
)
DIS_PATTERN = re.compile(r"gmm_(mode_k40|dim_d100)_dis_seed([012])")
PIS_PATTERN = re.compile(r"tune2_gmm_(mode_k40|dim_d100)_pis_smax(128|32)_seed0")
METHODS = ["target", "asbs", "gamma", "iw_fixed", "iw_tr", "dis", "pis"]
METHOD_LABELS = {
    "target": "Ground truth",
    "asbs": "ASBS",
    "gamma": "Gamma-ASBS",
    "iw_fixed": "IW fixed",
    "iw_tr": "IW TR",
    "dis": "DIS",
    "pis": "PIS tuned",
}
ASBS_METHODS = {
    "asbs": "asbs",
    "gamma_asbs": "gamma",
    "gamma_asbs_iw_fixed": "iw_fixed",
    "gamma_asbs_iw_trust_region_0p1": "iw_tr",
    "gamma_asbs_iw_trust_region_0p3": "iw_tr",
}
SUITES = {
    "mode_k40": {
        "title": "GMM mode sweep hardest setting: K=40, d=10",
        "num_components": 40,
        "dim": 10,
        "loc_scaling": 40.0,
        "bound": 50.0,
        "output_stem": "gmm_mode_k40",
    },
    "dim_d100": {
        "title": "GMM dimension sweep hardest setting: K=10, d=100",
        "num_components": 10,
        "dim": 100,
        "loc_scaling": 7.0,
        "bound": 15.0,
        "output_stem": "gmm_dim_d100",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_pt_samples(path: Path) -> np.ndarray:
    artifact: dict[str, Any] = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    return artifact["samples"].detach().cpu().numpy()


def _collect_samples(
    input_root: Path,
) -> dict[tuple[str, str, int], tuple[np.ndarray, Path]]:
    collected: dict[tuple[str, str, int], tuple[np.ndarray, Path]] = {}
    for path in sorted(input_root.rglob("best_loss_SD_data_1.npy")):
        match = next(
            (
                ASBS_PATTERN.fullmatch(part)
                for part in path.parts
                if ASBS_PATTERN.fullmatch(part) is not None
            ),
            None,
        )
        if match is None:
            continue
        suite, raw_method, seed_text = match.groups()
        key = (suite, ASBS_METHODS[raw_method], int(seed_text))
        collected[key] = (np.load(path), path)

    for path in sorted(input_root.rglob("best_loss_SD.pt")):
        dis_match = next(
            (
                DIS_PATTERN.fullmatch(part)
                for part in path.parts
                if DIS_PATTERN.fullmatch(part) is not None
            ),
            None,
        )
        if dis_match is not None:
            suite, seed_text = dis_match.groups()
            collected[(suite, "dis", int(seed_text))] = (
                _load_pt_samples(path),
                path,
            )
            continue
        pis_match = next(
            (
                PIS_PATTERN.fullmatch(part)
                for part in path.parts
                if PIS_PATTERN.fullmatch(part) is not None
            ),
            None,
        )
        if pis_match is not None:
            suite, sigma_text = pis_match.groups()
            expected_sigma = "128" if suite == "mode_k40" else "32"
            if sigma_text != expected_sigma:
                raise RuntimeError(f"Unexpected PIS sigma in {path}")
            collected[(suite, "pis", 0)] = (_load_pt_samples(path), path)

    expected = {
        (suite, method, seed)
        for suite in SUITES
        for method in METHODS[1:-1]
        for seed in range(3)
    }
    expected.update((suite, "pis", 0) for suite in SUITES)
    actual = set(collected)
    if actual != expected:
        raise RuntimeError(
            f"Sample artifact mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return collected


def _target_samples(
    *,
    num_components: int,
    dim: int,
    loc_scaling: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(12345)
    target = GMM(
        dim=dim,
        num_components=num_components,
        loc_scaling=loc_scaling,
        scale_scaling=1.0,
        max_weight_ratio=3.0,
        uniform_mixing_weights=False,
        softplus=True,
        device="cpu",
    )
    samples = target.sample((4000,)).cpu().numpy()
    return samples, target.locs[:, :2].cpu().numpy(), target.cat_probs.cpu().numpy()


def _validate_samples(
    samples: dict[tuple[str, str, int], tuple[np.ndarray, Path]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (suite, method, seed), (values, path) in sorted(samples.items()):
        if values.shape != (4000, int(SUITES[suite]["dim"])):
            raise RuntimeError(f"Unexpected sample shape {values.shape}: {path}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite sample values: {path}")
        bound = float(SUITES[suite]["bound"])
        first_two = values[:, :2]
        in_bounds = np.logical_and(
            np.abs(first_two[:, 0]) <= bound,
            np.abs(first_two[:, 1]) <= bound,
        )
        rows.append(
            {
                "suite": suite,
                "method": method,
                "seed": seed,
                "num_samples": len(values),
                "dimension": values.shape[1],
                "finite": True,
                "plot_in_bounds_fraction": float(in_bounds.mean()),
                "source_path": str(path),
            }
        )
    return rows


def _plot_suite(
    suite: str,
    samples: dict[tuple[str, str, int], tuple[np.ndarray, Path]],
    output_dir: Path,
) -> Path:
    config = SUITES[suite]
    target, centers, probabilities = _target_samples(
        num_components=int(config["num_components"]),
        dim=int(config["dim"]),
        loc_scaling=float(config["loc_scaling"]),
    )
    bound = float(config["bound"])
    figure, axes = plt.subplots(
        3,
        len(METHODS),
        figsize=(19.5, 8.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    center_sizes = 18.0 + 42.0 * probabilities / probabilities.max()
    for seed, row_axes in enumerate(axes):
        for method, axis in zip(METHODS, row_axes, strict=True):
            if method == "target":
                values = target
                color = "#4D4D4D"
            elif method == "pis" and seed > 0:
                axis.text(
                    0.5,
                    0.5,
                    "not run\n(tuning is seed 0 only)",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    fontsize=9,
                    color="#666666",
                )
                values = None
                color = "#0072B2"
            else:
                values = samples[(suite, method, seed)][0]
                color = "#0072B2" if method != "pis" else "#CC79A7"

            if values is not None:
                axis.scatter(
                    values[:, 0],
                    values[:, 1],
                    s=2.2,
                    alpha=0.20,
                    color=color,
                    edgecolors="none",
                    rasterized=True,
                )
            axis.scatter(
                centers[:, 0],
                centers[:, 1],
                s=center_sizes,
                marker="x",
                linewidths=0.8,
                color="#D55E00",
                alpha=0.9,
                zorder=3,
            )
            axis.set_xlim(-bound, bound)
            axis.set_ylim(-bound, bound)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.12, linewidth=0.4)
            if seed == 0:
                axis.set_title(METHOD_LABELS[method], fontsize=11)
        row_axes[0].set_ylabel(f"seed {seed}\n$x_2$", fontsize=10)
    for axis in axes[-1]:
        axis.set_xlabel("$x_1$", fontsize=10)

    figure.suptitle(
        f"{config['title']} — SD-best retained checkpoint\n"
        "4,000 saved samples per panel; orange crosses are target component means. "
        "ASBS-family evaluation RNG was not fixed; DIS/PIS used report_seed=12345.",
        fontsize=13,
    )
    output_path = output_dir / f"{config['output_stem']}_sd_best_samples.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = _parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _collect_samples(input_root)
    manifest_rows = _validate_samples(samples)
    manifest_path = output_dir / "sample_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    for suite in SUITES:
        print(_plot_suite(suite, samples, output_dir))
    print(manifest_path)


if __name__ == "__main__":
    main()
