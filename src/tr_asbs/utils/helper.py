import torch
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Literal
from datetime import datetime
import math


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


class JointMetricDivergenceStopper:
    """Stop only after two quality metrics jointly show sustained collapse.

    This deliberately does not stop on a plateau: a run may recover late.  It is
    intended as an opt-in compute guard for expensive sweeps, not as model
    selection (best checkpoints remain responsible for that).
    """

    def __init__(self, config):
        self.enabled = bool(config.get("enabled", False))
        self.primary_metric = str(config.get("primary_metric", "loss/SD"))
        self.secondary_metric = str(config.get("secondary_metric", "loss/tvd_soft"))
        self.min_evals = int(config.get("min_evals", 20))
        self.primary_factor = float(config.get("primary_factor", 2.0))
        self.secondary_delta = float(config.get("secondary_delta", 0.15))
        self.patience = int(config.get("patience", 8))
        self.nonfinite_patience = int(config.get("nonfinite_patience", 2))
        self.best_primary = math.inf
        self.best_secondary = math.inf
        self.bad_count = 0
        self.nonfinite_count = 0
        self.reason = None

    def update(self, metrics: Dict, eval_index: int) -> bool:
        if not self.enabled:
            return False
        primary = float(metrics[self.primary_metric])
        secondary = float(metrics[self.secondary_metric])
        if not math.isfinite(primary) or not math.isfinite(secondary):
            self.nonfinite_count += 1
            if self.nonfinite_count >= self.nonfinite_patience:
                self.reason = f"non-finite metrics for {self.nonfinite_count} evaluations"
                return True
            return False
        self.nonfinite_count = 0
        self.best_primary = min(self.best_primary, primary)
        self.best_secondary = min(self.best_secondary, secondary)
        if eval_index < self.min_evals:
            self.bad_count = 0
            return False
        collapsed = (
            primary > self.primary_factor * self.best_primary
            and secondary > self.best_secondary + self.secondary_delta
        )
        self.bad_count = self.bad_count + 1 if collapsed else 0
        if self.bad_count >= self.patience:
            self.reason = (
                f"sustained collapse for {self.bad_count} evaluations: "
                f"{self.primary_metric}={primary:.6g} (best={self.best_primary:.6g}), "
                f"{self.secondary_metric}={secondary:.6g} (best={self.best_secondary:.6g})"
            )
            return True
        return False

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            
def project_path(*args) -> str:
    """
    Abstraction from os.path.join()
    Builds absolute paths from relative path strings with ilfvrm package as root.
    If args already contains an absolute path, it is used as root for the subsequent joins
    Args:
        *args:

    Returns:
        absolute path

    """
    return os.path.abspath(os.path.join(PACKAGE_DIR, *args))


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary into a flat dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The parent key for the current level of the dictionary.
        sep (str): The separator to use between keys.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def extract_last_entry(dictionary):
    last_entries = {}
    for key, value in dictionary.items():
        try:
            last_entries[key] = value[-min(len(value), 1)]
        except:
            pass
    return last_entries

Mode = Literal["min", "max"]
TrackOn = Literal["ema", "raw"]
CheckpointOn = Literal["improve", "stop", "both"]  # 保存トリガー種別

@dataclass
class EarlyStopperEMAConfig:
    patience: int = 5
    mode: Mode = "min"
    min_delta: float = 0.0
    ema_alpha: float = 0.1
    bias_correction: bool = True
    track_on: TrackOn = "ema"

    # ▼ 追加: 保存のオン/オフと保存条件
    enable_checkpoint: bool = False
    checkpoint_on: CheckpointOn = "improve"     # 改善時 / 停止時 / 両方
    checkpoint_dir: Optional[str] = None        # 例: "./ckpts"
    filename_fmt: str = "step{step}_{tag}.pt"   # 出力ファイル名

class EarlyStopperEMA:
    def __init__(
        self,
        config: EarlyStopperEMAConfig,
        on_stop: Optional[Callable[[Dict], None]] = None,
        make_payload: Optional[Callable[[], Dict]] = None,          # 省略可（デフォルト実装あり）
        save_fn: Optional[Callable[[str, Dict], None]] = None       # 省略可（torch.save を使用）
    ) -> None:
        self.cfg = config
        self.on_stop = on_stop
        self.make_payload = make_payload
        self.save_fn = save_fn

        self._ema: Optional[float] = None
        self._t: int = 0
        self._best: Optional[float] = None
        self._wait: int = 0
        self._stopped: bool = False
        self._last_step: Optional[int] = None
        self._last_raw: Optional[float] = None

        if self.cfg.mode == "min":
            self._is_better = lambda current, best: (best - current) > self.cfg.min_delta
        elif self.cfg.mode == "max":
            self._is_better = lambda current, best: (current - best) > self.cfg.min_delta
        else:
            self._is_better = lambda current, best: True

        # 保存先準備（保存が有効なときのみ）
        if self.cfg.enable_checkpoint and self.cfg.checkpoint_dir:
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

    @property
    def ema(self) -> Optional[float]:
        return self._ema

    @property
    def best(self) -> Optional[float]:
        return self._best

    @property
    def wait(self) -> int:
        return self._wait

    @property
    def stopped(self) -> bool:
        return self._stopped

    def reset(self) -> None:
        self.__init__(self.cfg, self.on_stop, self.make_payload, self.save_fn)

    def _update_ema(self, x: float) -> float:
        self._t += 1
        if self._ema is None:
            self._ema = x
        else:
            a = self.cfg.ema_alpha
            self._ema = a * x + (1 - a) * self._ema
        if self.cfg.bias_correction:
            denom = 1.0 - (1.0 - self.cfg.ema_alpha) ** self._t
            return self._ema / denom
        return self._ema

    def _default_make_payload(self) -> Dict:
        # デフォルトでは最低限のメタだけ。ユーザー側で model/optimizer を渡すのを推奨
        return {}

    def _default_save_fn(self, path: str, payload: Dict) -> None:
        torch.save(payload, path)

    def _maybe_checkpoint(self, *, tag: str, status: Dict, extra_payload: Optional[Dict] = None) -> None:
        # ① 保存OFFなら即リターン
        if not self.cfg.enable_checkpoint:
            return
        # ② 保存先未指定なら安全にリターン
        if not self.cfg.checkpoint_dir:
            return

        # ③ ファイル名生成
        step = status.get("step")
        fname = self.cfg.filename_fmt.format(step=step, tag=tag)
        path = os.path.join(self.cfg.checkpoint_dir, fname)

        # ④ ペイロード組み立て（ユーザー提供がなければデフォルト）
        mk = self.make_payload or self._default_make_payload
        payload = dict(mk())
        if extra_payload:
            payload.update(extra_payload)
        payload["_early_stopper"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "config": {
                "mode": self.cfg.mode,
                "patience": self.cfg.patience,
                "min_delta": self.cfg.min_delta,
                "ema_alpha": self.cfg.ema_alpha,
                "bias_correction": self.cfg.bias_correction,
                "track_on": self.cfg.track_on,
            },
        }

        # ⑤ 保存実体（ユーザー提供がなければ torch.save）
        saver = self.save_fn or self._default_save_fn
        saver(path, payload)

    def update(self, value: float, *, step: Optional[int] = None) -> Dict:
        if self._stopped:
            return self._status_dict(should_stop=True)

        self._last_step = step
        self._last_raw = value

        ema_now = self._update_ema(value)
        target = ema_now if self.cfg.track_on == "ema" else value

        if self._best is None:
            self._best = target
            improved = True
            self._wait = 0
        else:
            if self._is_better(target, self._best):
                self._best = target
                improved = True
                self._wait = 0
            else:
                improved = False
                self._wait += 1

        should_stop = self._wait >= self.cfg.patience
        status = self._status_dict(should_stop=should_stop, improved=improved)

        # 改善時保存
        if improved and self.cfg.enable_checkpoint and self.cfg.checkpoint_on in ("improve", "both"):
            self._maybe_checkpoint(tag="best", status=status)

        # 停止確定時保存
        if should_stop:
            self._stopped = True
            if self.cfg.enable_checkpoint and self.cfg.checkpoint_on in ("stop", "both"):
                self._maybe_checkpoint(tag="stop", status=status)
            if self.on_stop is not None:
                self.on_stop(status)

        return status

    def _status_dict(self, *, should_stop: bool, improved: Optional[bool] = None) -> Dict:
        return {
            "step": self._last_step,
            "raw": self._last_raw,
            "ema": self.ema,
            "best": self.best,
            "wait": self.wait,
            "should_stop": should_stop,
            "improved": improved,
            "mode": self.cfg.mode,
            "track_on": self.cfg.track_on,
            "patience": self.cfg.patience,
            "min_delta": self.cfg.min_delta,
        }
        

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Literal
from datetime import datetime
import os
import torch

Mode = Literal["min", "max", "off"]  # off = 毎回改善扱いにする


@dataclass
class CheckpointConfig:
    enable: bool = True
    checkpoint_dir: str = "./ckpts"
    filename_fmt: str = "step{step}_{tag}.pt"  # {step}, {tag} を使える

    # 改善判定用
    mode: Mode = "min"      # "min": 小さいほど良い, "max": 大きいほど良い, "off": 常に改善扱い
    min_delta: float = 0.0  # どれだけ良くなったら「改善」とするか


class Checkpointer:
    def __init__(
        self,
        config: CheckpointConfig,
        make_payload: Optional[Callable[[], Dict]] = None,
        save_fn: Optional[Callable[[str, Dict], None]] = None,
    ) -> None:
        self.cfg = config
        self.make_payload = make_payload
        self.save_fn = save_fn

        self.best: Optional[float] = None
        self.best_step: Optional[int] = None

        if self.cfg.enable and self.cfg.checkpoint_dir:
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        # 改善判定関数
        if self.cfg.mode == "min":
            self._is_better = lambda current, best: (best - current) > self.cfg.min_delta
        elif self.cfg.mode == "max":
            self._is_better = lambda current, best: (current - best) > self.cfg.min_delta
        else:  # "off"
            self._is_better = lambda current, best: True

    # -------- 内部ユーティリティ --------
    def _default_make_payload(self) -> Dict:
        return {}

    def _default_save_fn(self, path: str, payload: Dict) -> None:
        torch.save(payload, path)

    def _build_payload(self, *, status: Dict, extra_payload: Optional[Dict]) -> Dict:
        mk = self.make_payload or self._default_make_payload
        payload = dict(mk())
        if extra_payload:
            payload.update(extra_payload)

        payload["_checkpoint"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
        }
        return payload

    def _save(self, *, tag: str, status: Dict, extra_payload: Optional[Dict]) -> Optional[str]:
        if not self.cfg.enable or not self.cfg.checkpoint_dir:
            return None

        step = status.get("step", 0)
        fname = self.cfg.filename_fmt.format(step=step, tag=tag)
        path = os.path.join(self.cfg.checkpoint_dir, fname)

        payload = self._build_payload(status=status, extra_payload=extra_payload)
        saver = self.save_fn or self._default_save_fn
        saver(path, payload)
        return path

    # -------- 公開 API --------
    def update(
        self,
        value: float,
        *,
        step: Optional[int] = None,
        tag_best: str = "best",
        extra_status: Optional[Dict] = None,
        extra_payload: Optional[Dict] = None,
    ) -> Dict:
        status = {"step": step, "value": value}
        if extra_status:
            status.update(extra_status)

        if self.best is None:
            improved = True
            self.best = value
            self.best_step = step
        else:
            improved = self._is_better(value, self.best)
            if improved:
                self.best = value
                self.best_step = step

        path: Optional[str] = None
        if improved and self.cfg.enable:
            path = self._save(tag=tag_best, status=status, extra_payload=extra_payload)

        return {
            "value": value,
            "best": self.best,
            "improved": improved,
            "step": step,
            "best_step": self.best_step,
            "path": path,
        }

    def save_last(
        self,
        *,
        step: Optional[int] = None,
        tag_last: str = "last",
        extra_status: Optional[Dict] = None,
        extra_payload: Optional[Dict] = None,
    ) -> Optional[str]:
        if not self.cfg.enable:
            return None

        status = {"step": step, "best": self.best, "best_step": self.best_step}
        if extra_status:
            status.update(extra_status)

        return self._save(tag=tag_last, status=status, extra_payload=extra_payload)
