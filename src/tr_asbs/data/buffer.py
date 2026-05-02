# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import Dataset


class BatchBuffer(Dataset):
    """Generic tensor buffer that stores batched samples on CPU."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.storage: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def extend(self, **kwargs: torch.Tensor) -> None:
        """Append batch tensors and split them into per-sample entries."""
        if not kwargs:
            return

        first_key = next(iter(kwargs))
        batch_size = kwargs[first_key].shape[0]

        for key, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)

            if value.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch for key {key}. "
                    f"Expected {batch_size}, got {value.shape[0]}"
                )

            value = value.detach().cpu()
            value_list = list(torch.unbind(value, dim=0))
            self.storage[key].extend(value_list)

        if len(self) > self.max_size:
            current_len = len(next(iter(self.storage.values())))
            trim_size = current_len - self.max_size
            if trim_size > 0:
                for key in self.storage:
                    self.storage[key] = self.storage[key][trim_size:]

    def add(self, batch: dict) -> None:
        """Backward-compatible alias."""
        self.extend(**batch)

    def __len__(self) -> int:
        try:
            key = next(iter(self.storage.keys()))
            return len(self.storage[key])
        except StopIteration:
            return 0

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.storage.items()}

    def state_dict(self) -> Dict[str, Any]:
        return {"storage": self.storage}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.storage = state_dict["storage"]

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}

        elem = batch[0]
        return {key: torch.stack([d[key] for d in batch]) for key in elem.keys()}
