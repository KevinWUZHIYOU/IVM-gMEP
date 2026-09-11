#!/usr/bin/env python3
"""Rebuild the LEPS article-method figure with the audited IR-IVM winner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType


HERE = Path(__file__).resolve()
CAMPAIGN_ROOT = HERE.parents[1]
LEPS_ROOT = HERE.parents[2]
PLOT_PRIMITIVES = (
    LEPS_ROOT
    / "ir_ivm_qm10_v2p20_microrefine_20260829/src/remake_leps_all_methods.py"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source = PLOT_PRIMITIVES.read_text(encoding="utf-8")
old_marker_rule = "markevery = max(len(nof) // 8, 1)"
new_marker_rule = (
    'markevery = 1 if name == "IR-IVM" else max(len(nof) // 8, 1)'
)
if source.count(old_marker_rule) != 1:
    raise RuntimeError("frozen plot marker rule drift")
source = source.replace(old_marker_rule, new_marker_rule)
plot = ModuleType("_frozen_leps_all_methods_plot")
plot.__file__ = str(PLOT_PRIMITIVES)
sys.modules[plot.__name__] = plot
exec(compile(source, str(PLOT_PRIMITIVES), "exec"), plot.__dict__)

# Reuse the established article figure's geometry, typography, method order,
# baseline loaders, and independently evaluated metrics.  Only the gated
# campaign, winning IR-IVM result, and provenance owner change.
plot.HERE = HERE
plot.CAMPAIGN_ROOT = CAMPAIGN_ROOT
plot.LEPS_ROOT = LEPS_ROOT
plot.WINNER = CAMPAIGN_ROOT / "runs/756bf4584a55e959.npz"
plot.SELECTION = CAMPAIGN_ROOT / "reports/acceptance/selection_manifest.json"


def main():
    plot.main()
    manifest_path = (
        CAMPAIGN_ROOT / "reports/figure/leps_all_methods_metrics_provenance.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frozen_plot_primitives"] = {
        "path": str(PLOT_PRIMITIVES.relative_to(LEPS_ROOT)),
        "sha256": sha256(PLOT_PRIMITIVES),
    }
    manifest["ir_ivm_marker_contract"] = {
        "initial_state_count": 1,
        "completed_outer_iteration_count": 28,
        "plotted_ir_ivm_state_count": 29,
        "marker_every": 1,
        "meaning": "one marker for the initial state and every completed outer iteration",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
