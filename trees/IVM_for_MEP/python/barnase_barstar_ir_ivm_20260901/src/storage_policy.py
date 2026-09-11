"""Pure path-history retention policy; never participates in IR-IVM numerics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


LEGACY = "legacy"
COMMITTED = "committed"
MODES = (LEGACY, COMMITTED)
LEGACY_MILESTONES = frozenset({1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500})


def should_store_path(
    *, mode, save_all_paths, call, is_initial, reparam_committed
):
    """Return a storage decision using metadata already computed online."""
    if mode not in MODES:
        raise ValueError(f"unknown path storage mode: {mode}")
    if mode == COMMITTED:
        if save_all_paths:
            raise ValueError("committed storage cannot be combined with --save-all-paths")
        return bool(is_initial or reparam_committed)
    return bool(save_all_paths or int(call) in LEGACY_MILESTONES)


@dataclass
class SyntheticPrefixRecorder:
    """No-Force harness used to prove full/thin storage is output-only."""

    mode: str
    save_all_paths: bool = False
    stored: list = field(default_factory=list)
    online_digest: list = field(default_factory=list)
    current_path: np.ndarray | None = None

    def observe(self, call, path, *, reparam_committed=False):
        path = np.asarray(path, dtype=np.float64)
        self.current_path = path.copy()
        self.online_digest.append((int(call), path.tobytes(), bool(reparam_committed)))
        if should_store_path(
            mode=self.mode,
            save_all_paths=self.save_all_paths,
            call=call,
            is_initial=len(self.online_digest) == 1,
            reparam_committed=reparam_committed,
        ):
            self.stored.append((int(call), path.copy()))
        return self.current_path.copy()

    def finalize(self, call, path):
        path = np.asarray(path, dtype=np.float64)
        if not self.stored or self.stored[-1][0] != int(call):
            self.stored.append((int(call), path.copy()))
        return path.copy()
