"""Hydra entrypoint for reusable PyTorch variational samplers.

Examples:
    uv run python -m tr_asbs.experiment.train_variational experiment=mw5 algorithm=pis
    uv run python -m tr_asbs.experiment.train_variational experiment=mw5 algorithm=dis
"""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig
import torch

from tr_asbs.variational.trainer import VariationalTrainer


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="variational_train",
)
def main(cfg: DictConfig) -> None:
    logger_cfg = cfg.fabric.get("loggers")
    logger_target = (
        "" if logger_cfg is None else str(logger_cfg.get("_target_", "")).lower()
    )
    if "wandb" in logger_target:
        if cfg.get("wandb_mode") is not None:
            os.environ["WANDB_MODE"] = str(cfg.wandb_mode)
        if cfg.get("wandb_host") is not None:
            os.environ["WANDB_BASE_URL"] = str(cfg.wandb_host)

    torch.set_float32_matmul_precision("high")
    trainer = VariationalTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
