"""Tensor targets must remain usable without optional molecular/JAX stacks."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_tensor_target_imports_without_optional_dependencies():
    source = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            roots = {
                "e3nn", "equinox", "fairchem", "jax", "mace", "mdtraj",
                "netCDF4", "normflows", "openmm", "openmmtools", "ott",
                "pymbar", "simtk", "tables", "torch_geometric",
            }

            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in self.roots:
                    raise ModuleNotFoundError(
                        f"Optional dependency blocked by test: {fullname}",
                        name=fullname,
                    )

        sys.meta_path.insert(0, BlockOptionalImports())
        modules = [
            "tr_asbs.process.sde",
            "tr_asbs.process.source",
            "tr_asbs.process.integrator",
            "tr_asbs.data.buffer",
            "tr_asbs.experiment.train_asbs",
            "tr_asbs.experiment.train_gamma_asbs",
            "tr_asbs.experiment.train_gamma_asbs_iw",
            "tr_asbs.experiment.train_variational",
            "tr_asbs.potential.gmm",
            "tr_asbs.potential.many_well_energy",
            "tr_asbs.potential.lennardjones_energy",
            "tr_asbs.potential.morse_energy",
            "tr_asbs.model.egnn",
        ]
        for module in modules:
            importlib.import_module(module)

        from pathlib import Path
        from typing import get_type_hints
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        import tr_asbs
        from tr_asbs.process.sde import BaseSDE

        get_type_hints(BaseSDE.drift)
        torch.set_num_threads(1)
        config_dir = Path(tr_asbs.__file__).parent / "config" / "experiment"
        for name in ["lj13", "morse13_ce_fast"]:
            cfg = OmegaConf.load(config_dir / f"{name}.yaml")
            controller = instantiate(cfg.controller)
            corrector = instantiate(cfg.corrector)
            positions = torch.randn(
                2, cfg.potential.n_particles, cfg.potential.spatial_dim
            )
            positions -= positions.mean(dim=1, keepdim=True)
            with torch.no_grad():
                for model in [controller, corrector]:
                    result = model(torch.full((2,), 0.5), positions)
                    assert result.shape == positions.shape
                    assert torch.isfinite(result).all()
        assert not (BlockOptionalImports.roots & sys.modules.keys())
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
