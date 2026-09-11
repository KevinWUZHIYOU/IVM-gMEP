#!/usr/bin/env python3
"""Make a non-formal LEPS all-methods preview with the sync-one candidate."""

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
CANDIDATE_RUN_ID = "356307bb5a618535"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source = PLOT_PRIMITIVES.read_text(encoding="utf-8")
old_marker_rule = "markevery = max(len(nof) // 8, 1)"
new_marker_rule = 'markevery = 1 if name == "IR-IVM" else max(len(nof) // 8, 1)'
if source.count(old_marker_rule) != 1:
    raise RuntimeError("frozen plot marker rule drift")
source = source.replace(old_marker_rule, new_marker_rule)

old_gate = '''if not selection["campaign_pass"] or not winner["strict_pass"] or winner["run_id"] != WINNER.stem:
        raise RuntimeError("strict acceptance gate is not satisfied by the plotted result")'''
new_gate = '''if winner["run_id"] != WINNER.stem:
        raise RuntimeError("selected preview result does not match the registered candidate")'''
if source.count(old_gate) != 1:
    raise RuntimeError("frozen strict-gate source drift")
source = source.replace(old_gate, new_gate)

plot = ModuleType("_frozen_leps_all_methods_sync1_preview")
plot.__file__ = str(PLOT_PRIMITIVES)
sys.modules[plot.__name__] = plot
exec(compile(source, str(PLOT_PRIMITIVES), "exec"), plot.__dict__)

# Reuse the formal figure's method inputs, metric definitions, colors,
# typography, axes, and layout.  Only the candidate IR-IVM trajectory and
# provenance owner change.  The output is deliberately a preview because the
# candidate passes 19/21 rather than 21/21 registered comparisons.
plot.HERE = HERE
plot.CAMPAIGN_ROOT = CAMPAIGN_ROOT
plot.LEPS_ROOT = LEPS_ROOT
plot.WINNER = CAMPAIGN_ROOT / "runs" / f"{CANDIDATE_RUN_ID}.npz"
plot.SELECTION = CAMPAIGN_ROOT / "reports/acceptance/selection_manifest.json"


def main():
    plot.main()
    manifest_path = (
        CAMPAIGN_ROOT / "reports/figure/leps_all_methods_metrics_provenance.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = manifest.pop("strict_winner")
    manifest["candidate_preview"] = candidate
    manifest["formal_replacement_authorized"] = False
    manifest["formal_replacement_reason"] = (
        "single synchronized recovery passes 19 of 21 registered same-NoF "
        "checks and misses the densified barrier criterion at NoF 150 and 278"
    )
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
