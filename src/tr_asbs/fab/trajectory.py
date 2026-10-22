"""Bounded trajectory loading for transform calibration and separate references."""

from pathlib import Path

import mdtraj


def load_trajectory_frames(path, max_frames=None, stride=1, offset=0):
    path = str(Path(path).expanduser().resolve())
    if stride < 1 or offset < 0 or (max_frames is not None and max_frames < 1):
        raise ValueError("Invalid trajectory frame selection")
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    if max_frames is None:
        return mdtraj.load(path, stride=stride)[offset:]
    chunks, remaining = [], max_frames
    for chunk in mdtraj.iterload(
        path, chunk=min(max_frames, 1024), stride=stride, skip=offset
    ):
        selected = chunk[:remaining]
        chunks.append(selected)
        remaining -= len(selected)
        if remaining == 0:
            break
    if not chunks:
        raise ValueError("Trajectory frame selection is empty")
    return mdtraj.join(chunks)


def trajectory_metadata(path, max_frames, stride, offset):
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return dict(
        path=str(resolved),
        bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        max_frames=max_frames,
        stride=stride,
        offset=offset,
    )
