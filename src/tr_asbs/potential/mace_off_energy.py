# Substitute for the gated/broken FairChem eSEN path (see fairchem_energy.py):
# facebook/adjoint_sampling's esen_spice.pt checkpoint requires approved
# HuggingFace access we don't have, and the installed fairchem-core (2.13.0)
# has renamed OCPCalculator -> FAIRChemCalculator/MLIPPredictUnit since
# fairchem_energy.py was written, so that class no longer even imports.
# MACE-OFF23 is a publicly downloadable, comparable equivariant-GNN organic-
# molecule potential (same purpose/category as eSEN), used here instead.
import os
import sys

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from .base_energy import BaseEnergy
from ..utils.eval_logweight import compute_log_weights_metrics


def _import_real_mace_calculators():
    """The project vendors a partial `mace` package under src/mace -- just
    enough for sde.py's `from mace.tools.torch_geometric import Batch` -- which
    shadows a full, properly pip-installed mace-torch (needed here for
    mace.calculators/mace.data) since both share the `mace` top-level name and
    the project's own src/ directory sits ahead of site-packages on sys.path.
    Evict any already-imported vendored `mace*` modules and deprioritize the
    vendored src/ path (moved later in sys.path, not removed -- tr_asbs itself
    still needs it) so the real package resolves instead.
    """
    for mod_name in [m for m in list(sys.modules) if m == "mace" or m.startswith("mace.")]:
        mod_file = getattr(sys.modules[mod_name], "__file__", "") or ""
        if "site-packages" not in mod_file:
            del sys.modules[mod_name]
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if src_dir in sys.path:
        sys.path.remove(src_dir)
        sys.path.append(src_dir)
    from mace.calculators import mace_off  # noqa: F401 -- import validates resolution
    import mace

    assert "site-packages" in (mace.__file__ or ""), (
        f"mace still resolved to the vendored copy: {mace.__file__}"
    )


class MaceOffEnergy(BaseEnergy):
    """
    Ground-truth-free molecular energy using a pretrained MACE-OFF23 potential.

    Unlike the LJ potentials' fixed/fully-connected graph, MACE uses a cutoff-
    radius neighbor list, so the graph structure must be rebuilt from the
    *current* positions on every call, not fixed once at construction time
    (verified: even trans-butane's own equilibrium geometry has atom pairs
    beyond the 4.5A cutoff, so a fixed graph is not just an approximation
    but wrong). The neighbor-list search itself is a non-differentiable,
    numpy/CPU-side indexing operation (standard practice for MLIPs -- only the
    energy computed *given* a fixed edge set needs to be differentiable w.r.t.
    positions, not the edge selection itself), done in a per-sample Python
    loop since MACE's own neighborhood search is not batched.

    No reference samples are available (no equivalent of the LJ ground-truth
    split), so this follows the same NoGT pattern as LennardJonesEnergyNoGT:
    energy statistics + the trainer's own importance-weight/ESS diagnostics
    only.
    """

    @classmethod
    def from_ase_molecule(
        cls,
        molecule_name: str,
        model_size: str = "small",
        device: str = "cpu",
        energy_clip_range: float | None = 2000.0,
    ) -> "MaceOffEnergy":
        """Convenience constructor: build atomic_numbers/reference_positions
        from ASE's small-molecule database (ase.collections.g2 -- e.g.
        'trans-butane', 'C6H6') instead of hardcoding coordinate arrays in a
        config file."""
        from ase.build import molecule

        atoms = molecule(molecule_name)
        return cls(
            n_particles=len(atoms),
            atomic_numbers=atoms.get_atomic_numbers().tolist(),
            reference_positions=atoms.get_positions(),
            model_size=model_size,
            device=str(device),  # Hydra passes fabric.device as a torch.device
            # object, but MACE's calculator loader expects a plain string
            # internally (does a `"cuda" in device` membership check).
            energy_clip_range=energy_clip_range,
        )

    def __init__(
        self,
        n_particles: int,
        atomic_numbers: list[int],
        reference_positions,
        model_size: str = "small",
        device: str = "cpu",
        energy_clip_range: float | None = 2000.0,
    ):
        spatial_dim = 3
        super().__init__(n_particles * spatial_dim)
        device = str(device)  # tolerate a torch.device object too
        assert len(atomic_numbers) == n_particles
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.device = device
        self.atomic_numbers_np = np.asarray(atomic_numbers, dtype=np.int64)
        self.energy_clip_range = energy_clip_range

        _import_real_mace_calculators()
        from mace.calculators import mace_off

        calc = mace_off(model=model_size, device=device, default_dtype="float32")
        self.model = calc.models[0].to(device)
        self.z_table = calc.z_table
        r_max = self.model.r_max
        self.cutoff = float(r_max.item() if torch.is_tensor(r_max) else r_max)

        # Real molecular potentials (this one included) have a physically huge
        # -- but finite, not inf/nan -- energy for compressed/near-overlapping
        # geometries, e.g. a modest 0.3A random perturbation around butane's
        # equilibrium structure pushed energy from ~-4285 eV to ~8e11 eV in
        # testing. Being finite, this would NOT trip training_step()'s
        # skip-on-non-finite-loss safety net (the same mechanism that, before
        # energy_clip was added to LennardJonesEnergy, silently discarded
        # ~93% of LJ-13's batches -- see lj13_logvar_fast.yaml), so it would
        # instead silently poison the loss/gradients while looking "valid".
        # Clip around this reference structure's own energy, generous enough
        # (2000 eV default) to cover any physically reasonable conformer/strain
        # difference (typically << 100 eV) while still bounding pathological
        # blowups.
        ref = torch.as_tensor(reference_positions, dtype=torch.float32, device=device)
        with torch.no_grad():
            self.energy_ref = self.eval(ref.unsqueeze(0), _clip=False).item()

    @property
    def is_mean_free(self) -> bool:
        return True

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def _build_batch_dict(self, positions: torch.Tensor) -> dict:
        """positions: (B, n_particles, 3), requires_grad tracked upstream."""
        from mace.data import Configuration, AtomicData
        from mace.tools.torch_geometric import Batch

        positions_np = positions.detach().cpu().numpy()
        data_list = []
        for b in range(positions_np.shape[0]):
            config = Configuration(
                atomic_numbers=self.atomic_numbers_np,
                positions=positions_np[b],
                properties={},
                property_weights={},
            )
            data_list.append(
                AtomicData.from_config(config, z_table=self.z_table, cutoff=self.cutoff)
            )
        # The caller's positions (e.g. samples from log_details()'s initial
        # eval, generated via the source/integrator) are not guaranteed to be
        # on the same device as this model's own (fixed-at-construction-time)
        # weights -- observed a device-mismatch RuntimeError deep in the
        # model's first linear layer when they weren't. The model's device is
        # the authoritative target, not positions.device or Batch.to()'s own
        # (unreliable -- some fields stayed on CPU) placement.
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        batch = Batch.from_data_list(data_list).to(model_device)
        batch_dict = batch.to_dict()
        for key, value in batch_dict.items():
            if torch.is_tensor(value):
                target_dtype = model_dtype if torch.is_floating_point(value) else value.dtype
                batch_dict[key] = value.to(device=model_device, dtype=target_dtype)
        # Substitute the differentiable positions tensor back in, in the same
        # per-atom order from_config produced (flattened over the batch dim).
        batch_dict["positions"] = positions.reshape(-1, 3).to(device=model_device, dtype=model_dtype)
        return batch_dict

    def eval(self, samples: torch.Tensor, _clip: bool = True) -> torch.Tensor:
        """E(x). samples: (B, n_particles*3) or (B, n_particles, 3)."""
        batch_size = samples.shape[0]
        positions = samples.reshape(batch_size, self.n_particles, self.spatial_dim)
        batch_dict = self._build_batch_dict(positions)
        out = self.model(batch_dict, training=False, compute_force=False)
        energy = out["energy"]
        if _clip and self.energy_clip_range is not None:
            energy = torch.clamp(
                energy,
                min=self.energy_ref - self.energy_clip_range,
                max=self.energy_ref + self.energy_clip_range,
            )
        return energy

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -self.eval(x)

    def sample(self, size):
        raise NotImplementedError(
            "No ground-truth samples available for MaceOffEnergy -- energy/ESS "
            "diagnostics only (see class docstring)."
        )

    @torch.no_grad()
    def visualize(self, samples, show=True, title=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        energy_samples = self.eval(samples).detach().cpu()
        ax.hist(energy_samples, bins=100, density=True, alpha=0.6, histtype="step", linewidth=4, label="generated data")
        ax.set_xlabel("Energy (eV)")
        ax.legend()
        wb_key = f"figures/vis_{title}" if title is not None else "figures/vis"
        wb = {wb_key: [wandb.Image(fig)]}
        plt.close(fig)
        return wb

    @torch.no_grad()
    def visualize_raw(self, samples, title="raw"):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5)
        wb = {f"figures/vis_{title}": [wandb.Image(fig)]}
        plt.close(fig)
        return wb

    def log_details(self, results: dict, prefix: str = "", visualize: bool = True) -> dict:
        data_1 = results["data_1"]
        energy = self.eval(data_1)

        logs = {
            f"{prefix}loss/neg_avg_energy": energy.mean().detach().cpu(),
            f"{prefix}energy/std": energy.std(unbiased=False).detach().cpu(),
            f"{prefix}energy/min": energy.min().detach().cpu(),
            f"{prefix}energy/max": energy.max().detach().cpu(),
        }

        for name, value in results.items():
            if not name.startswith("log_RN"):
                continue
            base_name = name[len("log_"):] if name.startswith("log_") else name
            stabilize = not name.endswith("det")
            metrics = compute_log_weights_metrics(value, stabilize=stabilize, normalize=stabilize)
            logs.update({f"{prefix}{base_name}/{key}": val for key, val in metrics.items()})

        if visualize:
            logs.update(self.visualize(data_1, title=f"{prefix}data_1"))

        if prefix == "":
            if "lm" in results:
                logs["lag_multiplier/lambda"] = results["lm"]
            if "gamma" in results:
                logs["lag_multiplier/gamma"] = results["gamma"]

        return logs

    def __repr__(self):
        return f"MaceOff-{self.n_particles}atoms"
