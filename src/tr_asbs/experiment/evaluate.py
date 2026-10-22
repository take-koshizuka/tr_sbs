from pathlib import Path
from copy import deepcopy
import json
import hashlib

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from lightning.fabric import seed_everything
from lightning.pytorch.utilities import rank_zero_warn

from tr_asbs.process.sde import ControlledFBSDE
from tr_asbs.utils.training import EMA


def stable_group_id(cfg: dict, exclude=("seed", "name")):
    if not isinstance(cfg, dict):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    d = {k: cfg[k] for k in sorted(cfg.keys()) if k not in exclude}
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


class GammaASBSEvaluator:
    """
    - group_id で決まる checkpoint_directory から
      (A) latest epoch_XXXX.pt
      (B) loss_energy_W_2_best.pt (など任意の *_best.pt)
      をロードして評価し、W&B に記録する。
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # W&B logger（Fabric logger と同じ instantiate を使う想定）
        wandb_logger = hydra.utils.instantiate(cfg.fabric.loggers)
        full_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_logger.log_hyperparams(full_cfg)

        # 学習コードと同じ group_id 生成（exclude=("name")）
        self.group_id = stable_group_id(cfg, exclude=("name"))
        cfg.fabric.loggers.group = stable_group_id(cfg)

        # Fabric setup
        self.fabric = hydra.utils.instantiate(cfg.fabric, loggers=wandb_logger)
        self.fabric.launch()

        self.rank_seed = cfg.seed ^ self.fabric.global_rank
        seed_everything(self.rank_seed, workers=False)

        self.checkpoint_directory = Path(self.cfg.checkpoint_directory) / self.group_id

        with self.fabric.init_module():
            # Potential / models
            self.potential = hydra.utils.instantiate(cfg.potential, device=self.fabric.device)

            controller = hydra.utils.instantiate(cfg.controller)
            corrector = hydra.utils.instantiate(cfg.corrector)

            # EMA 設定は学習と合わせる
            if cfg.ema_decay is not None:
                self.controller = EMA(controller, cfg.ema_decay, cfg.ema_start_step)
                self.corrector = EMA(corrector, cfg.ema_decay, cfg.ema_start_step)
            else:
                self.controller = controller
                self.corrector = corrector

            # Process / integrator
            self.source = hydra.utils.instantiate(cfg.source, device=self.fabric.device)
            self.base_sde = hydra.utils.instantiate(cfg.base_sde, is_mean_free=self.potential.is_mean_free)

            self.sde = ControlledFBSDE(
                base_sde=self.base_sde,
                u_controller=self.controller,
                h_controller=self.corrector,
            )

        # setup (optimizer 不要)
        self.controller = self.fabric.setup(self.controller)
        self.corrector = self.fabric.setup(self.corrector)

        self.controller.eval()
        self.corrector.eval()
        self.sde.eval()

    # -------------------------
    # checkpoint load helpers
    # -------------------------

    def _find_latest_epoch_ckpt(self) -> Path:
        if not self.checkpoint_directory.exists():
            raise FileNotFoundError(f"checkpoint_directory does not exist: {self.checkpoint_directory}")
        epoch_files = list(self.checkpoint_directory.glob("epoch_*.pt"))
        if len(epoch_files) == 0:
            raise FileNotFoundError(f"No epoch_*.pt found in {self.checkpoint_directory}")
        return max(epoch_files, key=lambda p: p.stat().st_mtime)

    def load_epoch_ckpt(self, ckpt_path: Path) -> dict:
        """
        save_checkpoint() が保存する epoch_XXXX.pt を読む。
        state に controller/corrector を渡して fabric.load する。
        """
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Epoch checkpoint not found: {ckpt_path}")

        state = {
            "controller": self.controller,
            "corrector": self.corrector,
            # optimizer は評価では不要。state に含めないと remainder に回る。
        }
        remainder = self.fabric.load(ckpt_path, state)

        self.controller.eval()
        self.corrector.eval()
        self.sde.eval()

        return {
            "loaded_from": str(ckpt_path),
            "current_epoch": remainder.get("current_epoch", None),
            "global_step": remainder.get("global_step", None),
            "remainder_keys": list(remainder.keys()),
        }

    def load_best_ckpt(self, best_filename: str) -> dict:
        """
        Checkpointer が保存する *_best.pt を読む。
        payload: {"fbsde": self.sde.state_dict(), "_checkpoint": ...} を前提。
        """
        ckpt_path = self.checkpoint_directory / best_filename
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "fbsde" not in ckpt:
            raise KeyError(f"{best_filename} does not contain 'fbsde'. Keys={list(ckpt.keys())}")

        # fbsde をロード（controller/corrector の重みもここに含まれている前提）
        self.sde.load_state_dict(ckpt["fbsde"])
        self.sde.eval()
        self.controller.eval()
        self.corrector.eval()

        meta = {"loaded_from": str(ckpt_path)}
        if "_checkpoint" in ckpt and "status" in ckpt["_checkpoint"]:
            meta["best_status"] = ckpt["_checkpoint"]["status"]
        return meta

    # -------------------------
    # evaluation
    # -------------------------

    @torch.no_grad()
    def evaluate_current_model(self, tag_prefix: str) -> dict:
        """
        現在 self.sde が参照している controller/corrector の状態で評価し、
        W&B summary に tag_prefix/xxx を記録する。
        """
        integrator = hydra.utils.instantiate(self.cfg.integrator, sde=self.sde)

        data_0 = self.source.sample((self.cfg.eval_samples,))
        data_1 = integrator.run(
            initial_data=data_0,
            center_every_step=self.potential.is_mean_free,
            zero_last_step_noise=False,
            return_trajectory=False,
            progress_bar=(self.fabric.global_rank == 0) and self.cfg.progress_bar,
        )
        logs = self.potential.log_details({"data_0": data_0, "data_1": data_1}, visualize=True)

        if self.fabric.global_rank == 0 and wandb.run is not None:
            for k, v in logs.items():
                wandb.run.summary[f"{tag_prefix}/{k}"] = v

        return logs

    # -------------------------
    # main
    # -------------------------

    def run(self):
        if self.fabric.global_rank == 0:
            print(f"[Eval] checkpoint_directory = {self.checkpoint_directory}")

        # W&B login
        if self.fabric.global_rank == 0:
            wandb.login(host=self.cfg.wandb_host)

        # summary に基本情報
        if self.fabric.global_rank == 0 and wandb.run is not None:
            wandb.run.summary["eval_only"] = True
            wandb.run.summary["seed"] = int(self.cfg.seed)
            wandb.run.summary["group_id"] = self.group_id
            wandb.run.summary["checkpoint_directory"] = str(self.checkpoint_directory)

        # ---- (A) latest epoch_XXXX.pt を評価 ----
        try:
            latest_epoch_path = self._find_latest_epoch_ckpt()
            meta_epoch = self.load_epoch_ckpt(latest_epoch_path)

            if self.fabric.global_rank == 0 and wandb.run is not None:
                wandb.run.summary["epoch_ckpt/loaded_from"] = meta_epoch["loaded_from"]
                if meta_epoch["current_epoch"] is not None:
                    wandb.run.summary["epoch_ckpt/current_epoch"] = int(meta_epoch["current_epoch"])
                if meta_epoch["global_step"] is not None:
                    wandb.run.summary["epoch_ckpt/global_step"] = int(meta_epoch["global_step"])

            self.evaluate_current_model(tag_prefix="final")

        except Exception as e:
            if self.fabric.global_rank == 0:
                print(f"[Eval] Skipped latest epoch evaluation due to error: {e}")
            if self.fabric.global_rank == 0 and wandb.run is not None:
                wandb.run.summary["eval_epoch_latest/error"] = str(e)

        # ---- (B) best: loss_energy_W_2_best.pt を評価 ----
        # ここがあなたの要件の「特に loss_energy_W_2_best.pt を読む」
        best_filename = "loss_energy_W_2_best.pt"
        try:
            meta_best = self.load_best_ckpt(best_filename)

            if self.fabric.global_rank == 0 and wandb.run is not None:
                wandb.run.summary["best_ckpt/loaded_from"] = meta_best["loaded_from"]
                if "best_status" in meta_best:
                    st = meta_best["best_status"]
                    if "step" in st:
                        wandb.run.summary["best_ckpt/step"] = int(st["step"])
                    if "value" in st:
                        try:
                            wandb.run.summary["best_ckpt/value"] = float(st["value"])
                        except Exception:
                            pass

            # best の重みで評価
            self.evaluate_current_model(tag_prefix="eval_best_loss_energy_W_2")

        except Exception as e:
            if self.fabric.global_rank == 0:
                print(f"[Eval] Skipped best evaluation due to error: {e}")
            if self.fabric.global_rank == 0 and wandb.run is not None:
                wandb.run.summary["eval_best_loss_energy_W_2/error"] = str(e)

        # finish
        if self.fabric.global_rank == 0:
            wandb.finish()


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    evaluator = GammaASBSEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
