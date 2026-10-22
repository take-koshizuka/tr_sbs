"""Compose a reproduction command and check its data without running a model."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Sequence

from tr_asbs.train import build_dispatch


DATA_KEYS = frozenset(
    {"ref_samples_path", "transform_data_path", "calibration_bundle_path",
     "topology_pdb_file", "md_eval_pdb_file", "atoms_file"}
)


def configured_data(config) -> list[tuple[str, str]]:
    """Read only external input fields, resolving their normal interpolations."""
    from omegaconf import DictConfig, ListConfig

    found = []

    def walk(node, prefix=""):
        if isinstance(node, DictConfig):
            for key in node:
                location = f"{prefix}.{key}" if prefix else key
                if key in DATA_KEYS and node[key] is not None:
                    found.append((location, str(node[key])))
                # Do not evaluate unrelated Hydra runtime interpolations.
                elif isinstance(node._get_node(key), (DictConfig, ListConfig)):
                    walk(node[key], location)
        elif isinstance(node, ListConfig):
            for index, child in enumerate(node):
                if isinstance(child, (DictConfig, ListConfig)):
                    walk(child, f"{prefix}.{index}")

    walk(config)
    return found


def inspect_configuration(dispatch, *, data_root: Path, skip_data=False) -> dict:
    from hydra import compose, initialize_config_dir

    config_directory = Path(__file__).parent / "config"
    config_name = "variational_train" if dispatch.method.algorithm else "train"
    with initialize_config_dir(version_base=None, config_dir=str(config_directory)):
        config = compose(config_name=config_name, overrides=list(dispatch.overrides))

    files = []
    if not skip_data:
        for field, value in configured_data(config):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = data_root / path
            path = path.resolve()
            files.append({"field": field, "path": str(path), "exists": path.is_file()})

    # A config check does not instantiate targets, evaluate energies or train.
    potential_target = str(config.potential.get("_target_", ""))
    modules = ["torch", "lightning", "scipy", "ot"]
    extras = []
    if any(term in potential_target for term in
           ("alanine_dipeptide", "flattened_atomic", ".openmm.")) or dispatch.backend == "atomic":
        extras.append("molecular")
        modules.extend(["openmm", "openmmtools", "mdtraj", "normflows", "tables"])
    if dispatch.backend == "atomic" or "flattened_atomic" in potential_target:
        extras.append("chem")
        modules.extend(["fairchem", "e3nn"])
    missing_modules = [name for name in modules if importlib.util.find_spec(name) is None]
    unsupported = bool(dispatch.method.algorithm) and any(
        term in potential_target for term in ("lennardjones", "morse_energy", "flattened_atomic")
    )
    return {
        **dispatch.as_dict(),
        "configuration": "composed",
        "data": "not_checked" if skip_data else files,
        "optional_extras": extras,
        "missing_modules": missing_modules,
        "support_error": (
            "PIS/DIS do not support the existing mean-free particle targets."
            if unsupported else None
        ),
        "training_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="tr-sbs")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--backend", choices=("auto", "flat", "atomic"), default="auto")
    parser.add_argument("--data-root", type=Path, default=Path.cwd(),
                        help="Resolve input paths relative to this directory (default: current directory).")
    parser.add_argument("--skip-data", action="store_true",
                        help="Only compose the configuration and check installed modules.")
    args, overrides = parser.parse_known_args(argv)
    try:
        dispatch = build_dispatch(args.method, args.experiment, overrides, args.backend)
        result = inspect_configuration(dispatch, data_root=args.data_root,
                                       skip_data=args.skip_data)
    except Exception as error:
        parser.exit(2, f"Configuration check failed: {error}\n")
    print(json.dumps(result, indent=2))
    missing_data = not args.skip_data and any(not item["exists"] for item in result["data"])
    if missing_data or result["missing_modules"] or result["support_error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
