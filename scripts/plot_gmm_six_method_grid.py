#!/usr/bin/env python3
"""Plot the six-method, score-free GMM mode/dimension comparisons.

Inputs are the frozen three-seed aggregate CSVs in
``reports/gmm_scorefree_six_methods/data``.  The ASBS-family and PIS/DIS
aggregators use different column names, so this script checks and joins their
reported means and *sample* standard deviations without reselecting epochs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "gmm_scorefree_six_methods"
GRIDS = {"mode": (5, 10, 20, 30, 40), "dim": (10, 25, 50, 100)}
SELECTIONS = ("min_online_sd", "max_online_raw_ess", "final")

# Match tr_asbs_wandb/make_plot.py: ASBS blue, IW fixed orange/square,
# IW TR red/diamond.  The remaining colors and markers are distinct tab10.
METHODS = {
    "asbs": ("ASBS", "tab:blue", "o"),
    "gamma_asbs": ("TR-ASBS", "tab:green", "^"),
    "iw_fixed": ("ASBS-IW", "tab:orange", "s"),
    "iw_tr": ("TR-SBS", "tab:red", "D"),
    "pis": ("PIS", "tab:purple", "P"),
    "dis": ("DIS", "tab:brown", "X"),
}

METRICS = {
    "SD": ("SD", "sd_mean", "sd_sample_sd", "loss/SD_mean", "loss/SD_sd", True),
    "TVD": ("Soft TVD", "tvd_soft_mean", "tvd_soft_sample_sd", "loss/tvd_soft_mean", "loss/tvd_soft_sd", False),
    "W2": (r"$W_2$", "w2_mean", "w2_sample_sd", "loss/W_2_mean", "loss/W_2_sd", False),
    "Sliced_W2": (r"Sliced $W_2$", "sliced_w2_mean", "sliced_w2_sample_sd", "loss/Sliced_W_2_mean", "loss/Sliced_W_2_sd", False),
    "raw_path_ESS": ("Raw path ESS", "raw_path_ess_mean", "raw_path_ess_sample_sd", "eval/raw_path_ess_mean", "eval/raw_path_ess_sd", True),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_finite(row: dict[str, str], column: str, source: Path) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {column} in {source}: {row}")
    return value


def canonical_method(raw: str, suite: str) -> str:
    if raw == "gamma_asbs_iw_fixed":
        return "iw_fixed"
    if raw == f"gamma_asbs_iw_trust_region_{'0p3' if suite == 'mode' else '0p1'}":
        return "iw_tr"
    if raw in METHODS:
        return raw
    raise ValueError(f"Unexpected {suite} method: {raw}")


def load_grid(data_dir: Path, suite: str, selection: str) -> dict[tuple[str, int], dict[str, tuple[float, float]]]:
    asbs_path = data_dir / f"asbs_{suite}_aggregate.csv"
    variational_path = data_dir / "pis_dis_aggregate.csv"
    result: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}

    for source, rows in (
        (asbs_path, read_rows(asbs_path)),
        (variational_path, read_rows(variational_path)),
    ):
        for row in rows:
            if row["selection"] != selection:
                continue
            is_variational = source == variational_path
            if is_variational and row["suite"] != suite:
                continue
            method = canonical_method(row["method"], suite)
            if is_variational and method not in {"pis", "dis"}:
                raise ValueError(f"Unexpected PIS/DIS row: {row}")
            if not is_variational and method in {"pis", "dis"}:
                raise ValueError(f"Unexpected ASBS-family row: {row}")
            x = int(row["grid_value"] if is_variational else row["num_components" if suite == "mode" else "dimension"])
            if x not in GRIDS[suite]:
                raise ValueError(f"Unexpected {suite}={x} in {source}")
            if int(row["n_seeds"]) != 3:
                raise ValueError(f"Expected three seeds for {method}, {suite}={x}")
            key = (method, x)
            if key in result:
                raise ValueError(f"Duplicate {key} for selection={selection}")
            metrics = {}
            for metric, (_, a_mean, a_sd, v_mean, v_sd, _) in METRICS.items():
                mean_col, sd_col = (v_mean, v_sd) if is_variational else (a_mean, a_sd)
                mean = as_finite(row, mean_col, source)
                sd = as_finite(row, sd_col, source)
                if sd < 0:
                    raise ValueError(f"Negative sample SD for {key}, {metric}")
                metrics[metric] = (mean, sd)
            result[key] = metrics

    expected = {(method, x) for method in METHODS for x in GRIDS[suite]}
    if set(result) != expected:
        raise ValueError(f"{suite} {selection} grid mismatch: missing={sorted(expected - set(result))}, extra={sorted(set(result) - expected)}")
    return result


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.color": "#D3D3D3",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "mathtext.fontset": "stixsans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_one(
    result: dict[tuple[str, int], dict[str, tuple[float, float]]],
    suite: str,
    metric: str,
    output_dir: Path,
) -> None:
    ylabel, *_, logarithmic = METRICS[metric]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for method, (label, color, marker) in METHODS.items():
        x = list(GRIDS[suite])
        values = [result[(method, point)][metric] for point in x]
        y = [mean for mean, _ in values]
        if logarithmic:
            if any(mean <= 0 for mean in y):
                raise ValueError(f"Log-scale {metric} has nonpositive mean for {method}")
            # Matplotlib needs a positive lower endpoint on logarithmic axes.
            yerr = [[min(sd, mean * (1 - 1e-6)) for mean, sd in values],
                    [sd for _, sd in values]]
        else:
            yerr = [sd for _, sd in values]
        ax.errorbar(
            x, y, yerr=yerr, label=label, color=color, ecolor=color,
            marker=marker, linestyle="-", linewidth=2, markersize=7,
            capsize=4, elinewidth=1.5, markeredgecolor="white",
            markeredgewidth=0.5, alpha=0.9,
        )
    ax.set_xlabel(r"$K$" if suite == "mode" else r"$d$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(GRIDS[suite])
    if logarithmic:
        ax.set_yscale("log")
    if metric == "TVD":
        ax.set_ylim(bottom=0, top=1.02)
    # The main SD/TVD panels have an open band between the methods. Use a
    # shorter three-column legend there so it does not cover the low TR-SBS
    # curve; use Matplotlib's data-aware placement for the other panels.
    interior_positions = {
        ("mode", "SD"): (0.5, 0.72),
        ("mode", "TVD"): (0.5, 0.76),
        ("mode", "W2"): (0.55, 0.69),
        ("dim", "TVD"): (0.28, 0.77),
    }
    interior = interior_positions.get((suite, metric))
    legend_options = dict(frameon=True, framealpha=0.9, edgecolor="gray",
                          fancybox=False, ncol=3 if interior else 2,
                          loc="center" if interior else "best",
                          bbox_to_anchor=interior)
    if (suite, metric) == ("dim", "TVD"):
        # The gap is narrower here, so make the legend taller but less wide.
        legend_options.update(ncol=2, fontsize=12, labelspacing=0.25,
                              borderpad=0.3, handletextpad=0.3,
                              columnspacing=0.8)
    legend = ax.legend(**legend_options)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"GMM_{suite}_{metric}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    legend.set_frame_on(False)
    frameless_dir = output_dir / "legend_no_box"
    frameless_dir.mkdir(parents=True, exist_ok=True)
    frameless_stem = frameless_dir / stem.name
    fig.savefig(frameless_stem.with_suffix(".pdf"))
    fig.savefig(frameless_stem.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPORT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPORT_ROOT / "figures")
    parser.add_argument("--selection", choices=SELECTIONS, default="min_online_sd")
    args = parser.parse_args()
    setup_style()
    for suite in GRIDS:
        result = load_grid(args.data_dir, suite, args.selection)
        for metric in METRICS:
            plot_one(result, suite, metric, args.output_dir / args.selection)
    print(f"Saved {len(GRIDS) * len(METRICS)} boxed and frameless PDF/PNG figure pairs "
          f"in {args.output_dir / args.selection}")


if __name__ == "__main__":
    main()
