# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import Dataset


@lru_cache(maxsize=1)
def _thermo_atomic_data_type():
    """Import the optional atomistic data type only when that buffer is used."""
    from tr_asbs.data.atomic_data import ThermoAtomicData

    return ThermoAtomicData


class BatchBuffer(Dataset):
    """Generic tensor buffer that stores batched samples on CPU."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.storage: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.metadata = None

    def bind_metadata(self, metadata):
        if self.metadata is None and len(self):
            raise ValueError("Nonempty buffer has no kernel metadata; regenerate it")
        if self.metadata is not None and self.metadata != metadata:
            raise ValueError("Buffer kernel metadata differs; clear and regenerate it")
        self.metadata = metadata

    def clear(self) -> None:
        """Drop all stored entries. Used to guarantee a buffer never mixes
        samples generated from two different frozen policy snapshots (e.g.
        u_{i-1}/h_{i-1} vs. u_i/h_i across outer loops) -- without this, FIFO
        eviction alone leaves stale cross-vintage entries in the buffer until
        enough fresh samples accumulate to evict them, and self-normalizing
        their importance weights together (see normalize_log_weights) is not
        valid since they're Radon-Nikodym derivatives against different
        reference measures."""
        self.storage = defaultdict(list)
        self.metadata = None

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
        return {"storage": self.storage, "metadata": self.metadata}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.storage = state_dict["storage"]
        self.metadata = state_dict.get("metadata")

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}

        elem = batch[0]
        return {key: torch.stack([d[key] for d in batch]) for key in elem.keys()}


class ThermoAtomicDataBuffer(Dataset):
    """A buffer to store ThermoAtomicData and tensors for training. Currently
    intended to be instantiated for every rank in a distributed setting.
    Counterpart of BatchBuffer above for the Ala2/OpenMM+PaiNN atomic
    pipeline, where entries are graph-batches (ThermoAtomicData) rather than
    flat tensors.
    """

    def __init__(self, max_size: int, filter_nan: bool = True):
        self.max_size = max_size
        self.storage = defaultdict(list)
        self.filter_nan = filter_nan

    def extend(self, **kwargs: Any) -> None:
        """Extend the buffer with a dictionary of ThermoAtomicData and tensors."""
        thermo_atomic_data_type = _thermo_atomic_data_type()

        # Extract the batch index
        for value in kwargs.values():
            if isinstance(value, thermo_atomic_data_type):
                num_graphs, batch_index = value.num_graphs, value.batch
                num_atoms = batch_index.shape[0]
                batch_index = batch_index.cpu()
                break
        else:
            raise ValueError(
                "No ThermoAtomicData found in the input dictionary, cannot determine "
                "batch index."
            )

        # Add data to the buffer
        for key, value in kwargs.items():
            if not isinstance(value, (thermo_atomic_data_type, torch.Tensor)):
                raise TypeError(
                    f"Expected ThermoAtomicData or torch.Tensor, got {type(value)} for "
                    f"key '{key}'"
                )
            # Always move to cpu for storage
            value = value.to("cpu")
            if isinstance(value, thermo_atomic_data_type):
                value_list = value.batch_to_atomicdata_list()
            else:  # torch.Tensor
                if value.shape[0] == num_atoms:  # per-atom property
                    value_list = [value[batch_index == i] for i in range(num_graphs)]
                elif value.shape[0] == num_graphs:  # per-graph property
                    value_list = [value[i].unsqueeze(0) for i in range(num_graphs)]
                else:
                    raise ValueError(
                        f"Tensor shape {value.shape} does not match expected "
                        f"per-atom ({num_atoms},) or per-graph ({num_graphs},) shape."
                    )
            self.storage[key].extend(value_list)

        # If nan values are present in tensors, remove them
        if self.filter_nan:
            nan_indices = []
            for key, value in self.storage.items():
                if isinstance(value[0], torch.Tensor):
                    for i, tensor in enumerate(value):
                        if torch.isnan(tensor).any():
                            nan_indices.append(i)
            if nan_indices:
                for key, value in self.storage.items():
                    self.storage[key] = [
                        value[i] for i in range(len(value)) if i not in nan_indices
                    ]

        # Ensure the buffer does not exceed the maximum size
        if len(self) > self.max_size:
            for key in self.storage:
                self.storage[key] = self.storage[key][-self.max_size :]

    def __len__(self) -> int:
        try:
            key = next(iter(self.storage.keys()))
            return len(self.storage[key])
        except StopIteration:
            return 0

    def __getitem__(self, idx) -> dict[str, Any]:
        return {key: value[idx] for key, value in self.storage.items()}

    @staticmethod
    def collate_fn(batch: Sequence[Any]) -> Any:
        """Collate function to combine a batch of data into a single dictionary."""
        thermo_atomic_data_type = _thermo_atomic_data_type()
        element = batch[0]
        if isinstance(element, thermo_atomic_data_type):
            return thermo_atomic_data_type.from_data_list(batch)
        elif isinstance(element, torch.Tensor):
            return torch.cat(batch, dim=0)
        elif isinstance(element, float):
            return torch.as_tensor(batch, dtype=torch.float)
        elif isinstance(element, int):
            return torch.as_tensor(batch, dtype=torch.long)
        elif isinstance(element, str):
            return batch
        elif isinstance(element, Mapping):
            return {
                key: ThermoAtomicDataBuffer.collate_fn([item[key] for item in batch])
                for key in element.keys()
            }
        elif isinstance(element, Sequence):  # already not str
            return [ThermoAtomicDataBuffer.collate_fn(item) for item in zip(*batch)]
        else:
            raise TypeError(f"Unsupported type {type(element)} for collate_fn.")
