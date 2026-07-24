from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Sequence, Dict, Any
from collections.abc import Sequence as _Sequence
import warnings
import os

import torch
import numpy as np

import wandb
import matplotlib.pyplot as plt

from pathlib import Path

from ..utils.distribution_distance import (
    compute_distribution_distances,
    compute_eot,
    w2_1d_samples,
)
from ..utils.eval_logweight import compute_log_weights_metrics
from ..utils.geometry import subtract_com_vector


LogProbFn = Callable[[torch.Tensor], torch.Tensor]

class BaseEnergy:
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
   
    @property
    def is_mean_free(self) -> bool:
        return False

    # E(x)
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -self.eval(x)

    # ∇E(x)
    def grad_E(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)

            E = self.eval(x) # -log_prob
            grad_E = torch.autograd.grad(
                E.sum(), x, create_graph=False
            )[0]

        return grad_E
    
    def _clip_score(self, grad_E, max_grad_E_norm=None):
        if max_grad_E_norm is not None:
            norm = torch.linalg.vector_norm(grad_E, dim=-1).detach()
            clip_coefficient = torch.clamp(max_grad_E_norm / (norm + 1e-6), max=1)
            clip_coefficient = clip_coefficient.unsqueeze(-1)
        else:
            clip_coefficient = torch.ones_like(grad_E)

        return grad_E * clip_coefficient

    # score := - ∇E(x)
    def score(self, x: torch.Tensor) -> torch.Tensor:
        return - self.grad_E(x)
    
    def clipped_score(self, x: torch.Tensor, max_norm: float) -> torch.Tensor:
        grad_E = self.grad_E(x)
        return -self._clip_score(grad_E, max_norm)

    def __call__(self, x: torch.Tensor) -> Dict:
        assert x.ndim == 2 and x.shape[-1] == self.dim

        # forces: ∇E = - ∇ log p, as p(x) = C exp(-E(x))
        output_dict = {}
        output_dict["forces"] = self.grad_E(x)
        return output_dict 
    
    @abstractmethod
    def visualize(self, samples, title):
        pass
    
    @abstractmethod
    def sample(self, size: tuple | int) -> torch.Tensor:
        pass
    
    @torch.no_grad()
    def sample_loss(self, samples: torch.Tensor) -> torch.Tensor:
        gt_samples = self.sample(samples.shape[0])
        samples = samples.view(samples.shape[0], -1).detach().cpu()
        gt_samples = gt_samples.view(gt_samples.shape[0], -1).detach().cpu()
        res = compute_distribution_distances(samples, gt_samples.to(samples.device))
        return res
    @torch.no_grad()
    def energy_dist_error(self, energies: torch.Tensor) -> torch.Tensor:
        gt_samples = self.sample(energies.shape[0])
        gt_samples = gt_samples.view(gt_samples.shape[0], -1).detach()
        gt_energies = -self.eval(gt_samples)
        energy_w2 = w2_1d_samples(energies, gt_energies).detach().cpu()
        return {"energy_W_2": energy_w2}
    
    def log_details(self, results: dict, prefix="", visualize=True) -> dict:
        data_1 = results["data_1"]
        loss_metrics = self.sample_loss(data_1)
        energy = -self.eval(data_1)
        loss_metrics.update({'neg_avg_energy': -energy.mean().item()})
        loss_metrics.update(self.energy_dist_error(energy))
        log = {f'{prefix}loss/{key}': loss_metrics[key] for key in loss_metrics }
        del loss_metrics, energy
        
        for name, value in results.items():
            # log_RN で始まるログだけ処理
            if not name.startswith('log_RN'):
                continue
            # ex.)
            #   name = "log_RN[u|u_next]"      -> base_name = "RN[u|u_next]"
            #   name = "log_RN[u|u_next]_pred" -> base_name = "RN[u|u_next]_pred"
            #   name = "log_RN[u|h]"           -> base_name = "RN[u|h]"
            #   name = "log_RN[u_next|u]"      -> base_name = "RN[u_next|u]"
            if name.startswith('log_'):
                base_name = name[len('log_'):]
            else:
                # fallback just in case (normally should not come here) 
                base_name = name
            
            is_det = name.endswith('det')

            if is_det:
                metrics = compute_log_weights_metrics(value, stabilize=False, normalize=False)
            else:
                metrics = compute_log_weights_metrics(value)
                
            rnd_metrics = {
                f'{prefix}{base_name}/{key}': v
                for key, v in metrics.items()
            }
            log.update(rnd_metrics)
                
        if visualize:
            log_fig = self.visualize(data_1, title=f"{prefix}data_1")
            log.update(log_fig)
        
        if prefix == "":
            if 'lm' in results:
                log['lag_multiplier/lambda'] = results['lm']
            if 'gamma' in results:
                log['lag_multiplier/gamma'] = results['gamma']
            
        return log
    
    @torch.no_grad()
    def visualize_raw(self, samples, title="raw"):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5)
        plt.show()
        wb = {f"figures/vis_{title}": [wandb.Image(fig)]}
        plt.close(fig)
        return wb
    
class BaseSymtheticEnergy(BaseEnergy):
    def __init__(self, n_particles, spatial_dim, ref_samples_path):
        dim = n_particles * spatial_dim
        super().__init__(dim)
        self.spatial_dim = spatial_dim
        self.n_particles = n_particles
        
        # root = Path(os.path.abspath(__file__)).parent.parent.parent
        ref_samples_np = np.load(Path(ref_samples_path), allow_pickle=True)
        self.ref_samples = subtract_com_vector(
            torch.tensor(ref_samples_np),
            self.n_particles,
            self.spatial_dim,
        )
        
    @property
    def is_mean_free(self) -> bool:
        return True
        
    def clipped_score(self, x: torch.Tensor, max_norm: float) -> torch.Tensor:
        grad_E = self.grad_E(x)
        clipped_score = -self._clip_score(grad_E, max_norm)
        return subtract_com_vector(clipped_score, self.n_particles, self.spatial_dim)
        
    def sample(self, size: int | tuple) -> torch.Tensor:
        num_samples = size if isinstance(size, int) else sum(list(size))
        idxs = torch.randperm(len(self.ref_samples))[:num_samples]
        gt_samples = self.ref_samples[idxs]
        sample_size = size if isinstance(size, tuple) else (size,)
        return gt_samples.reshape(sample_size + (self.n_particles, self.spatial_dim))
    
    def eot_loss(self, samples: torch.Tensor) -> Dict[str, float]:
        """
        Compute EOT-based distance between generated samples and reference data.
        """
        gt_samples = self.sample(samples.shape[0])
        samples_flat = samples.detach().cpu().reshape(samples.shape[0], -1)
        gt_flat = gt_samples.detach().cpu().reshape(gt_samples.shape[0], -1)
        eot_res = compute_eot(samples_flat, gt_flat, self.n_particles, self.spatial_dim)["Eq-EMD2"]
        return {"Eq-EMD2": float(eot_res)}
    
    def log_details(self, results: dict, prefix="", visualize=True) -> dict:
        logs = super().log_details(results, prefix=prefix, visualize=visualize)
        pred_samples = results["data_1"]
        
        eot_metrics = self.eot_loss(pred_samples)
        logs[f'{prefix}loss/Eq-EMD2'] = eot_metrics['Eq-EMD2']
            
        return logs

    @torch.no_grad()
    def visualize_raw(self, samples, title="raw"):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5)
        plt.show()
        wb = {f"figures/vis_{title}": [wandb.Image(fig)]}
        plt.close(fig)
        return wb
    
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x is None or not self._can_normalize:
            return x

        mins = self.normalization_min
        maxs = self.normalization_max

        # [ 0, 1 ]
        x = (x - mins) / (maxs - mins + 1e-5)
        # [ -1, 1 ]
        return x * 2 - 1

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        if x is None or not self._can_normalize:
            return x

        mins = self.normalization_min
        maxs = self.normalization_max

        x = (x + 1) / 2
        return x * (maxs - mins) + mins
