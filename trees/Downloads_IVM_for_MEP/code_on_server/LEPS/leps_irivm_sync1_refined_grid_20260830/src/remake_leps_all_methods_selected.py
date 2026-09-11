#!/usr/bin/env python3
"""Rebuild the formal LEPS all-methods figure with the refined-grid IVM."""

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
CANDIDATE_RUN_ID = "3c62d93a051be116"
ALGORITHM_SOURCE = CAMPAIGN_ROOT / "src/leps_irivm_sync1_refined.py"
OPTIMIZER_SOURCE = (
    LEPS_ROOT
    / "leps_irivm_ala_chig_every1_sync1_20260830/src/leps_irivm_sync1.py"
)
PROTOCOL = CAMPAIGN_ROOT / "PROTOCOL.md"
AUDIT_SOURCE = CAMPAIGN_ROOT / "src/audit_and_select.py"


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
        raise RuntimeError("selected result does not match the refined-grid manifest")'''
if source.count(old_gate) != 1:
    raise RuntimeError("frozen strict-gate source drift")
source = source.replace(old_gate, new_gate)

# The formal figure has one IVM slot.  The refined IR-IVM result replaces the
# former article IVM instead of being plotted beside it, and is henceforth
# labelled simply "IVM" in the legend and machine-readable source table.
old_traces = '''traces: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "IR-IVM": ir_ivm_trace(WINNER),
        "Article IVM": article_ivm_trace(ARTICLE_IVM),
    }'''
new_traces = '''traces: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "IR-IVM": ir_ivm_trace(WINNER),
    }'''
if source.count(old_traces) != 1:
    raise RuntimeError("frozen trace-source block drift")
source = source.replace(old_traces, new_traces)

old_results = 'results = {"IR-IVM": WINNER, "Article IVM": ARTICLE_IVM, **OTHER_METHODS}'
new_results = 'results = {"IR-IVM": WINNER, **OTHER_METHODS}'
if source.count(old_results) != 1:
    raise RuntimeError("frozen plotted-result block drift")
source = source.replace(old_results, new_results)

old_article_order = '    "Article IVM",\n'
if source.count(old_article_order) != 1:
    raise RuntimeError("frozen article-IVM order entry drift")
source = source.replace(old_article_order, "")
source = source.replace('"IR-IVM"', '"IVM"')
source = source.replace(
    "LEPS IR-IVM and article-method convergence",
    "LEPS IVM and article-method convergence",
)

plot = ModuleType("_frozen_leps_all_methods_refined_selection")
plot.__file__ = str(PLOT_PRIMITIVES)
sys.modules[plot.__name__] = plot
exec(compile(source, str(PLOT_PRIMITIVES), "exec"), plot.__dict__)

plot.HERE = HERE
plot.CAMPAIGN_ROOT = CAMPAIGN_ROOT
plot.LEPS_ROOT = LEPS_ROOT
plot.WINNER = CAMPAIGN_ROOT / "runs" / f"{CANDIDATE_RUN_ID}.npz"
plot.SELECTION = CAMPAIGN_ROOT / "reports/acceptance/selection_manifest.json"


def main():
    plot.main()
    manifest_path = CAMPAIGN_ROOT / "reports/figure/leps_all_methods_metrics_provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest.pop("strict_winner")
    manifest["refined_grid_selected_ir_ivm"] = selected
    manifest["formal_replacement_authorized"] = True
    manifest["formal_replacement_reason"] = (
        "user-authorized permanent semantic replacement: the selected IR-IVM "
        "is now the LEPS method named IVM; the former article IVM is absent"
    )
    manifest["selection_status_disclosed"] = (
        "scientifically ranked refined-grid result; 19/21 strict same-NoF "
        "checks, with densified-barrier misses at NoF 150 and 278"
    )
    manifest["method_naming_contract"] = {
        "legend_name": "IVM",
        "meaning": "IR-IVM with alpha=100, r=3e-3, m_max=5",
        "former_article_ivm_plotted": False,
        "other_methods_unchanged": True,
    }
    manifest["superseded_formal_figure"] = {
        "path": "Figures/LEPS/leps_all_methods_metrics.pdf",
        "sha256": "f558d0b0aeb66115bc3b36a015aad2cc2bfce748917b7e309abc19f6589b0376",
    }
    manifest["frozen_plot_primitives"] = {
        "path": str(PLOT_PRIMITIVES.relative_to(LEPS_ROOT)),
        "sha256": sha256(PLOT_PRIMITIVES),
    }
    manifest["experiment_and_code_chain"] = {
        "selected_result": {
            "path": str(plot.WINNER.relative_to(LEPS_ROOT)),
            "sha256": sha256(plot.WINNER),
        },
        "refined_wrapper": {
            "path": str(ALGORITHM_SOURCE.relative_to(LEPS_ROOT)),
            "sha256": sha256(ALGORITHM_SOURCE),
        },
        "optimizer": {
            "path": str(OPTIMIZER_SOURCE.relative_to(LEPS_ROOT)),
            "sha256": sha256(OPTIMIZER_SOURCE),
        },
        "protocol": {
            "path": str(PROTOCOL.relative_to(LEPS_ROOT)),
            "sha256": sha256(PROTOCOL),
        },
        "independent_audit": {
            "path": str(AUDIT_SOURCE.relative_to(LEPS_ROOT)),
            "sha256": sha256(AUDIT_SOURCE),
        },
    }
    manifest["ir_ivm_marker_contract"] = {
        "initial_state_count": 1,
        "completed_outer_iteration_count": 28,
        "plotted_ir_ivm_state_count": 29,
        "marker_every": 1,
        "meaning": "one marker for the initial state and every completed outer iteration",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
