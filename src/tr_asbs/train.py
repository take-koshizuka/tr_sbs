"""One public launcher for the repository's existing training runtimes.

Examples (submit training commands through the project's remote scheduler)::

    python -m tr_asbs.train --method tr-sbs --experiment gmm_mode --dry-run
    python -m tr_asbs.train --method asbs-iw --experiment mw5 gamma=0.5

Listing and dry runs import no training, Torch, Hydra or molecular packages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import shlex
import sys
from typing import Sequence

from tr_asbs.methods import METHODS, Method, get_method


# These existing experiment configs use ThermoAtomicData graph batches. Native
# internal-coordinate and FlattenedAtomicPotential configs use the flat trainer.
ATOMIC_EXPERIMENTS = frozenset({"ala2", "ala2_gamma_asbs_iw", "ala4"})
SUPPORTED_ATOMIC_EXPERIMENTS = frozenset({"ala2_gamma_asbs_iw"})


@dataclass(frozen=True)
class Dispatch:
    method: Method
    backend: str
    module: str
    overrides: tuple[str, ...]

    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.overrides]

    def as_dict(self) -> dict:
        return {
            "method": self.method.name,
            "variant_of": self.method.variant_of,
            "backend": self.backend,
            "module": self.module,
            "gamma_schedule": self.method.gamma_schedule,
            "algorithm": self.method.algorithm,
            "overrides": list(self.overrides),
            "command": shlex.join(self.command()),
        }


def _override_assignment(override: str) -> tuple[str, str] | None:
    """Recognize Hydra's normal, append and deletion assignment forms."""

    if "=" not in override:
        return None
    key, value = override.split("=", 1)
    return key.lstrip("+~"), value


def build_dispatch(
    method_name: str,
    experiment: str,
    overrides: Sequence[str] = (),
    backend: str = "auto",
) -> Dispatch:
    """Choose an unchanged runtime and enforce its public method identity."""

    method = get_method(method_name)
    if backend not in {"auto", "flat", "atomic"}:
        raise ValueError("Backend must be auto, flat or atomic.")
    if not experiment or experiment.startswith(("/", ".")):
        raise ValueError("Experiment must be a Hydra experiment config name.")

    config_file = Path(__file__).parent / "config" / "experiment" / f"{experiment}.yaml"
    if not config_file.is_file():
        raise ValueError(f"Unknown experiment {experiment!r}: {config_file}")

    is_atomic = experiment in ATOMIC_EXPERIMENTS
    if is_atomic and experiment not in SUPPORTED_ATOMIC_EXPERIMENTS:
        raise ValueError(
            f"Experiment {experiment!r} is a legacy atomic curriculum preset "
            "and cannot run through this public launcher. Use --experiment "
            "ala2_gamma_asbs_iw with --method asbs-iw or --method tr-sbs for "
            "atomic alanine dipeptide; retain the original entrypoint and "
            "configuration for legacy reproductions."
        )
    selected_backend = ("atomic" if is_atomic else "flat") if backend == "auto" else backend
    if is_atomic and selected_backend != "atomic":
        raise ValueError(
            f"Experiment {experiment!r} uses graph batches; select --backend atomic "
            "with ASBS-IW or TR-SBS. For flat alanine internal coordinates, use "
            "--experiment ad_ic_gamma_asbs."
        )
    if selected_backend == "atomic":
        if method.atomic_module is None:
            raise ValueError(
                f"{method.name} has no public atomic backend. ASBS-IW and TR-SBS "
                "support atomic alanine; historical atomic ASBS commands remain "
                "available through tr_asbs.experiment.train_wt_asbs."
            )
        if not is_atomic:
            raise ValueError(
                "The atomic backend requires a supported graph-batch experiment: "
                + ", ".join(sorted(SUPPORTED_ATOMIC_EXPERIMENTS))
                + "."
            )
        module = method.atomic_module
    else:
        module = method.module

    required = {
        "experiment": experiment,
        "paper_method": method.name,
        "method_id": method.key,
        "method_variant_of": method.variant_of or "null",
    }
    if method.gamma_schedule is not None:
        required["gamma_schedule"] = method.gamma_schedule
    if method.algorithm is not None:
        required["algorithm"] = method.algorithm
        required["algorithm.name"] = method.algorithm
        required["algorithm.method._target_"] = (
            f"tr_asbs.variational.{method.algorithm.upper()}Method"
        )

    # Reject contradictory method controls before importing a runtime or touching
    # output directories. Repeated identical controls are harmless and omitted.
    retained = []
    for override in overrides:
        assignment = _override_assignment(override)
        override_key = (
            assignment[0] if assignment is not None else override.lstrip("+~")
        )
        group_key = override_key.split("@", 1)[0]
        if method.algorithm is not None and group_key == "algorithm.method":
            raise ValueError(
                f"{method.name} controls algorithm.method; override individual "
                "non-identity algorithm settings instead."
            )
        if group_key == "defaults" or (
            "@" in override_key
            and group_key in {"experiment", "algorithm", *required}
        ):
            raise ValueError(
                f"Cannot replace public method defaults through {override!r}; "
                "select --method and --experiment instead."
            )
        if assignment is not None:
            key, value = assignment
            if key in required:
                if override.startswith("~") or value != required[key]:
                    raise ValueError(
                        f"{method.name} requires {key}={required[key]}; "
                        f"conflicting override {override!r}."
                    )
                continue
        elif override.lstrip("+~") in required:
            raise ValueError(f"Cannot remove method control {override!r}.")
        retained.append(override)

    metadata = {"paper_method", "method_id", "method_variant_of"}
    controls = [
        f"{'++' if key in metadata else ''}{key}={value}"
        for key, value in required.items()
    ]
    return Dispatch(method, selected_backend, module, tuple(controls + retained))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a paper-named method using the existing Hydra trainers.",
        epilog=(
            "Remaining arguments are passed to Hydra unchanged. Use --dry-run "
            "to inspect the command; submit expensive training to remote compute."
        ),
    )
    parser.add_argument("--method", help="Paper method name; see --list-methods.")
    parser.add_argument("--experiment", help="Existing config/experiment name, without .yaml.")
    parser.add_argument("--backend", choices=("auto", "flat", "atomic"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON dispatch without running training.")
    parser.add_argument("--list-methods", action="store_true", help="List canonical paper names and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args, overrides = parser.parse_known_args(argv)
    if args.list_methods:
        for method in METHODS.values():
            print(f"{method.key:12} {method.name:12} {method.description}")
        return
    if args.method is None or args.experiment is None:
        parser.error("--method and --experiment are required unless --list-methods is used")
    try:
        dispatch = build_dispatch(args.method, args.experiment, overrides, args.backend)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(dispatch.as_dict(), indent=2))
        return

    # Hydra parses sys.argv inside its decorated main. Do not compose or mutate
    # its configs here: existing defaults, saved config schemas and checkpoints
    # retain their original meaning.
    previous_argv = sys.argv
    try:
        sys.argv = [dispatch.module, *dispatch.overrides]
        importlib.import_module(dispatch.module).main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
