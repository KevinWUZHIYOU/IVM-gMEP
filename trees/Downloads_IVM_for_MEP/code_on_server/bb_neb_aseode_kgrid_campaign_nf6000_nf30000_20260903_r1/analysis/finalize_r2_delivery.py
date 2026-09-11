#!/usr/bin/env python3
"""Close R2 figures, machine attachments, PDF QA, and Markdown provenance.

This finalizer is deliberately zero-physical.  It replays the existing R2
readers, validates every stored metric and hash, and publishes only the two
documentation artifacts.  It never evaluates an energy or Force.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bb_discrete32_analysis_r2_delivery", HERE / "bb_discrete32_analysis_r2.py"
)
R2 = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("cannot load R2 analysis module")
SPEC.loader.exec_module(R2)

SCHEMA = "bb-discrete32-ode-nf30000-r2-completion-audit-v1"
STATUS = "PASS_FINAL_R2_FIGURES_QA_AND_PROVENANCE_CLOSED"
QA_SCHEMA = "bb-discrete32-ode-nf30000-r2-pdf-qa-v2"
QA_STATUS = "PASS_FINAL_R2_PDF_QA"
EXPECTED_QA_CHECKS = {
    "page_count_and_dimensions", "fonts_embedded",
    "required_text_and_finite_labels",
    "cost_to_precision_excludes_previous_IVM_and_uncommitted_trial",
    "all_pages_rendered_at_240_dpi", "rendered_pixels_finite_and_nonblank",
    "axes_legends_arrows_and_clipping_visually_inspected",
    "cost_to_precision_formal_curves_are_only_IVM_and_ODE_NEB",
    "call515_is_labeled_diagnostic_only",
    "energy_profiles_use_stored_32_point_paths",
    "energy_profile_includes_previous_IVM_NF9990",
    "unrelaxed_R2B_surface_high_energy_clipping_is_display_only",
    "main_pdf_untouched",
}
MAIN_PDF = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/main.pdf")
MAIN_PDF_SHA256 = "b36e2825cedcb2e712b6a318671f3a693ef12dbe33e15059d2e19c66cbf07c2d"
MAIN_PDF_BYTES = 5256405
FIGURE_DIR = (
    HERE.parents[2] / "Figures" / "barnase_barstar" / "active_candidates"
)
PREFIX = "barnase_barstar_discrete32_ode_nf30000_r2"
EXPECTED_OUTPUTS = {
    "comparison_pack_json": FIGURE_DIR / f"{PREFIX}_comparison.json",
    "cost_to_precision_pdf": FIGURE_DIR / f"{PREFIX}_cost_to_precision.pdf",
    "energy_vs_arc_length_pdf": FIGURE_DIR / f"{PREFIX}_energy_vs_arc_length.pdf",
    "evolution_pdf": FIGURE_DIR / f"{PREFIX}_evolution.pdf",
    "same_N_F_table_csv": FIGURE_DIR / f"{PREFIX}_same_nf.csv",
    "pdf_qa_json": FIGURE_DIR / f"{PREFIX}_pdf_qa.json",
    "provenance_md": FIGURE_DIR / f"{PREFIX}_provenance.md",
    "completion_audit_json": FIGURE_DIR / f"{PREFIX}_completion_audit.json",
}
PDF_KEYS = (
    "cost_to_precision_pdf", "energy_vs_arc_length_pdf", "evolution_pdf"
)
TOKENS = {
    "@CREATED_HKT@", "@OUTPUT_TABLE@", "@MANIFEST_BINDING@",
    "@COMPLETION_AUDIT_PATH@", "@SELECTED_K@", "@METRIC_SUMMARY@",
    "@ODE_TERMINAL_BINDING@", "@CACHE_BINDING@",
    "@CACHE_TRANSPORT_BINDINGS@", "@IR_BINDINGS@", "@SURFACE_BINDINGS@",
    "@CODE_TABLE@", "@PDF_QA_SUMMARY@",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly_file(path: Path | str, role: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    status = resolved.lstat()
    require(
        stat.S_ISREG(status.st_mode)
        and not resolved.is_symlink()
        and stat.S_IMODE(status.st_mode) & 0o222 == 0,
        f"{role} is not a read-only regular nonsymlink file",
    )
    return resolved


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main_pdf_binding() -> dict[str, Any]:
    resolved = MAIN_PDF.expanduser().resolve(strict=True)
    status = resolved.lstat()
    require(
        resolved == MAIN_PDF
        and stat.S_ISREG(status.st_mode)
        and not resolved.is_symlink(),
        "formal root main.pdf path/type differs",
    )
    record = binding(resolved)
    require(
        record == {
            "path": str(MAIN_PDF),
            "sha256": MAIN_PDF_SHA256,
            "bytes": MAIN_PDF_BYTES,
        },
        "formal root main.pdf differs from the pre-authoring binding",
    )
    return record


def resolve_binding(value: Any, role: str) -> Path:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"},
            f"{role} is not an exact path/SHA binding")
    path = readonly_file(value["path"], role)
    require(value["sha256"] == sha256_file(path), f"{role} SHA differs")
    return path


def resolve_terminal_authorization(terminal: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the one mode-normalized authorization already audited remotely.

    The canonical terminal audit can only be produced on the remote host after
    ``verify_execution_authorization`` has accepted the remote mode-0444 file.
    Mutagen preserves its bytes but presents the exact campaign copy locally as
    mode 0600.  This narrow reader neither changes that mode nor extends the
    cache adapter's transport exception to any other path.
    """

    record = terminal.get("authorization", {})
    expected = (
        R2.CACHE_REPLICA_RUN.parent.parent / "authorizations" /
        f"{R2.CACHE_REPLICA_RUN.name}.execution-authorization.json"
    )
    require(
        isinstance(record, Mapping)
        and set(record) == {"path", "sha256"}
        and Path(str(record["path"])).is_absolute()
        and Path(str(record["path"])) == expected
        and len(str(record["sha256"])) == 64,
        "terminal-audited authorization binding/path differs",
    )
    path = Path(str(record["path"])).resolve(strict=True)
    status = path.lstat()
    mode = stat.S_IMODE(status.st_mode)
    require(
        stat.S_ISREG(status.st_mode)
        and not path.is_symlink()
        and mode in {0o444, 0o600}
        and sha256_file(path) == record["sha256"],
        "terminal-audited authorization local replica differs",
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    require(
        document.get("execution_authorized") is True
        and document.get("candidate", {}).get("run_label")
        == R2.CACHE_REPLICA_RUN.name
        and document.get("candidate", {}).get("paid_batches") == 1000
        and document.get("candidate", {}).get("terminal_optimization_N_F") == 30000
        and document.get("authorization_file", {}).get("mode") == "0444"
        and document.get("authorization_file", {}).get("path") == str(expected),
        "terminal-audited authorization identity/budget differs",
    )
    return {
        **binding(path),
        "local_replica_mode_octal": f"{mode:04o}",
        "remote_mode_verified_by_canonical_terminal_audit": "0444",
        "local_mode_was_changed_by_analysis": False,
    }


def load_json(path: Path | str, role: str) -> tuple[Path, dict[str, Any]]:
    resolved = readonly_file(path, role)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{role} is not a JSON object")
    return resolved, value


def multiplier(histories: Mapping[str, Sequence[Mapping[str, Any]]], metric: str) -> dict[str, Any]:
    ivm, ode = histories["IVM"], histories["ODE-NEB"]
    threshold = max(min(float(row[metric]) for row in ivm),
                    min(float(row[metric]) for row in ode))
    ivm_nf = R2.BASE_MODULE.first_passage(ivm, metric, threshold)
    ode_nf = R2.BASE_MODULE.first_passage(ode, metric, threshold)
    require(ivm_nf and ode_nf, f"common first passage is absent for {metric}")
    ratio = max(ivm_nf, ode_nf) / min(ivm_nf, ode_nf)
    return {
        "metric": metric, "threshold": threshold,
        "ivm_first_passage_N_F": ivm_nf,
        "ode_neb_first_passage_N_F": ode_nf,
        "ratio": ratio, "display": f"{ratio:.1f}x",
    }


def validate_csv(path: Path, pack: Mapping[str, Any]) -> None:
    with path.open("r", newline="", encoding="utf-8") as stream:
        observed = list(csv.DictReader(stream))
    fields = (
        "latest_observed_N_F", "best_i_perp", "best_delta_e32_eV",
        "best_delta_e32_kcal_per_mol",
    )
    expected = {}
    for envelope in pack["same_N_F_budget_envelopes"]:
        for method, values in envelope["methods"].items():
            expected[(int(envelope["budget_N_F"]), method)] = values
    require(len(observed) == len(expected), "same-NoF CSV row count differs")
    for row in observed:
        require(set(row) == {"budget_N_F", "method", *fields},
                "same-NoF CSV columns differ")
        key = (int(row["budget_N_F"]), row["method"])
        require(key in expected, f"unexpected same-NoF CSV row {key}")
        values = expected[key]
        if values is None:
            require(all(row[field] == "" for field in fields),
                    f"absent same-NoF row differs at {key}")
        else:
            require(int(row[fields[0]]) == int(values[fields[0]]),
                    f"same-NoF budget source differs at {key}")
            require(all(float(row[field]) == float(values[field]) for field in fields[1:]),
                    f"same-NoF metric differs at {key}")


def validate_qa(path: Path, manifest_path: Path, comparison_path: Path,
                outputs: Mapping[str, Path]) -> dict[str, Any]:
    qa_path, qa = load_json(path, "R2 PDF QA")
    checks = qa.get("checks")
    formal_main_pdf = main_pdf_binding()
    analysis_source = readonly_file(
        HERE / "bb_discrete32_analysis_r2.py", "R2 analysis source for PDF QA"
    )
    require(
        qa.get("schema") == QA_SCHEMA and qa.get("status") == QA_STATUS
        and qa.get("physical_calls") == 0
        and qa.get("openmm_contexts_constructed") == 0
        and qa.get("bb_locks_acquired") == 0
        and qa.get("manifest") == binding(manifest_path)
        and qa.get("comparison") == binding(comparison_path)
        and qa.get("analysis_source") == binding(analysis_source)
        and qa.get("main_pdf") == formal_main_pdf
        and isinstance(checks, Mapping)
        and set(checks) == EXPECTED_QA_CHECKS
        and all(value is True for value in checks.values())
        and qa.get("plot_contract") == R2.FIGURE_CONTRACT_R2,
        "R2 PDF QA schema/bindings/checks differ",
    )
    records = qa.get("pdfs", {})
    require(set(records) == {"cost_to_precision", "energy_vs_arc_length", "evolution"},
            "R2 PDF QA PDF roles differ")
    for role, key in zip(records, PDF_KEYS):
        require({field: records[role][field] for field in ("path", "sha256", "bytes")}
                == binding(outputs[key]), f"R2 PDF QA binding differs for {role}")
        require(records[role].get("pages") == 1
                and records[role].get("render_dpi") == 240
                and records[role].get("visual_status")
                == "PASS_EXPLICIT_HUMAN_CONFIRMATION",
                f"R2 PDF QA structural/visual status differs for {role}")
        render = records[role].get("render", {})
        render_path = readonly_file(render.get("path", ""), f"{role} QA render")
        require({field: render[field] for field in ("path", "sha256", "bytes")}
                == binding(render_path) and render.get("finite_pixels") is True,
                f"R2 PDF raster binding differs for {role}")
    return {"binding": binding(qa_path), "document": qa,
            "main_pdf": formal_main_pdf}


def collect_evidence(manifest_path: Path, comparison_path: Path,
                     csv_path: Path, qa_path: Path) -> dict[str, Any]:
    manifest_path, manifest = load_json(manifest_path, "final R2 manifest")
    require(manifest.get("schema") == R2.MANIFEST_SCHEMA
            and manifest.get("status") == "FINAL_INPUTS_FROZEN",
            "R2 manifest is not final")
    raw_outputs = manifest.get("outputs", {}).get("final", {})
    require(set(raw_outputs) == set(EXPECTED_OUTPUTS), "R2 final output map differs")
    outputs = {}
    for role, expected in EXPECTED_OUTPUTS.items():
        observed = Path(str(raw_outputs[role])).expanduser().resolve()
        require(observed == expected.resolve(), f"R2 final output path differs: {role}")
        if role not in {"provenance_md", "completion_audit_json"}:
            outputs[role] = readonly_file(observed, f"R2 output {role}")
    require(outputs["comparison_pack_json"] == comparison_path.resolve(strict=True)
            and outputs["same_N_F_table_csv"] == csv_path.resolve(strict=True)
            and outputs["pdf_qa_json"] == qa_path.resolve(strict=True),
            "R2 CLI machine-output paths differ from manifest")
    comparison_path, pack = load_json(comparison_path, "R2 comparison pack")
    require(
        pack.get("schema") == R2.PACK_SCHEMA
        and pack.get("status")
        == "COMPLETE_ZERO_FORCE_DISCRETE32_NF30000_R2_ANALYSIS"
        and pack.get("manifest")
        == {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
        and pack.get("metric_protocol", {}).get("path_images") == 32
        and pack.get("metric_protocol", {}).get("densification") is False,
        "R2 comparison identity/protocol differs",
    )
    methods, meta = R2.load_complete_data_r2(manifest)
    formal = {name: [R2.BASE_MODULE.serializable_row(row) for row in methods[name]]
              for name in ("IVM", "Previous IVM", "ODE-NEB")}
    diagnostic = [R2.BASE_MODULE.serializable_row(row)
                  for row in methods[R2.IR_DIAGNOSTIC_KEY]]
    require(pack.get("histories") == formal
            and pack.get("diagnostic_uncommitted_states") == diagnostic
            and pack.get("diagnostic_exclusion", {}).get(
                "excluded_from_cost_to_precision") is True
            and max(row["optimization_N_F"] for row in formal["IVM"]) == 8982
            and max(row["optimization_N_F"] for row in formal["ODE-NEB"]) == 30000
            and diagnostic[0]["optimization_N_F"] == 14983,
            "R2 comparison formal/diagnostic histories differ from source replay")
    evolution_registration, legacy_layout = R2.validate_evolution_registration(
        manifest
    )
    require(
        pack.get("evolution_surface_envelope_registration")
        == evolution_registration
        and pack.get("legacy_evolution_layout_reference") == legacy_layout,
        "R2 comparison evolution registration/provenance differs",
    )
    surface = R2.load_surface_r2(manifest)
    require(surface["surface_energy_eV"].shape == (59, 28), "R2 surface shape differs")
    validate_csv(outputs["same_N_F_table_csv"], pack)
    qa = validate_qa(outputs["pdf_qa_json"], manifest_path, comparison_path, outputs)
    formal_input = manifest["inputs"]["formal_ode_neb"]
    terminal_record = formal_input["terminal_audit"]
    require(
        isinstance(terminal_record, Mapping)
        and set(terminal_record) == {"path", "sha256"},
        "ODE terminal-audit manifest binding differs",
    )
    terminal_path = R2.resolve_replica_terminal_audit_file(
        terminal_record["path"], str(terminal_record["sha256"])
    )
    cache_path = resolve_binding(formal_input["incremental_cache"], "ODE final cache")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    authorization = resolve_terminal_authorization(terminal)
    require(
        terminal.get("status") == "PASS"
        and terminal.get("terminal_eligible") is True
        and terminal.get("formal_figure_eligible") is True
        and terminal.get("paid_batches") == 1000
        and terminal.get("paid_optimization_N_F") == 30000
        and terminal.get("physical_force_calls_by_external_audit") == 0
        and terminal.get("OpenMM_contexts_constructed_by_external_audit") == 0
        and cache.get("status") == "FINAL_BOUND_TO_CANONICAL_TERMINAL_AUDIT"
        and cache.get("processed_batches") == 1000
        and cache.get("processed_optimization_N_F") == 30000
        and cache.get("local_mutagen_replica_transport")
        == R2.expected_cache_replica_transport(),
        "ODE terminal/cache accounting or replica provenance differs",
    )
    surface_result = resolve_binding(
        manifest["inputs"]["ir_centered_surface_bundle"], "R2B surface result"
    )
    surface_audit = resolve_binding(
        manifest["inputs"]["ir_centered_surface_audit"], "R2B surface audit"
    )
    surface_adapter = resolve_binding(
        manifest["inputs"]["surface_center_state_audit"], "R2B center adapter"
    )
    state_binding = resolve_binding(
        manifest["inputs"]["ir_ivm_full_state"]["state_binding"], "IR state binding"
    )
    verification = resolve_binding(
        manifest["inputs"]["ir_ivm_full_state"]["verification"], "IR verification"
    )
    source_manifest = readonly_file(
        Path(manifest["inputs"]["ode_source_root"]) / "SOURCE_MANIFEST.json",
        "ODE source manifest",
    )
    code_paths = {
        "analysis": HERE / "bb_discrete32_analysis_r2.py",
        "cache_wrapper": R2.CACHE_REPLICA_WRAPPER,
        "input_finalizer": HERE / "finalize_analysis_inputs_r2.py",
        "pdf_qa": HERE / "audit_final_pdfs_r2.py",
        "delivery_finalizer": Path(__file__),
        "partial_preview": HERE / "render_partial_nf15000_preview_r2.py",
        "regression_tests": HERE / "test_bb_discrete32_analysis_r2.py",
        "delivery_tests": HERE / "test_r2_delivery_pipeline.py",
        "provenance_template": HERE / "FINAL_PROVENANCE_R2.template.md",
    }
    codes = {role: binding(readonly_file(path, f"R2 code {role}"))
             for role, path in code_paths.items()}
    histories = pack["histories"]
    evidence = {
        "manifest": binding(manifest_path),
        "comparison": binding(comparison_path),
        "machine_outputs": {
            role: binding(path) for role, path in outputs.items()
        },
        "pdf_qa": qa["binding"],
        "main_pdf": qa["main_pdf"],
        "ode": {
            "run_dir": formal_input["run_dir"],
            "terminal_audit": binding(terminal_path),
            "incremental_cache": binding(cache_path),
            "authorization": authorization,
            "paid_batches": 1000,
            "optimization_N_F": 30000,
            "selected_spring_k_eV_per_A2": float(meta["selected_spring_k_eV_per_A2"]),
            "replica_transport": cache["local_mutagen_replica_transport"],
        },
        "ir": {
            "formal": {"call_per_image": 318, "optimization_N_F": 8982,
                       "formally_reportable": True},
            "diagnostic": {"call_per_image": 515, "optimization_N_F": 14983,
                           "formally_reportable": False,
                           "state_role": "terminal_uncommitted_charged_tail"},
            "state_binding": binding(state_binding),
            "verification": binding(verification),
        },
        "surface": {
            "result": binding(surface_result), "audit": binding(surface_audit),
            "center_adapter": binding(surface_adapter),
            "shape": [59, 28], "force_calls": 0,
            "optimization_N_F_contribution": 0,
            "role": "unrelaxed Cartesian 0 K diagnostic surface centered on call515",
            "display_cap_kcal_per_mol": R2.EVOLUTION_DISPLAY_CAP_KCAL_PER_MOL,
            "stored_values_unchanged": True,
        },
        "source_manifest": binding(source_manifest),
        "codes": codes,
        "derived": {
            "i_perp": multiplier(histories, "i_perp"),
            "delta_e32_kcal_per_mol": multiplier(
                histories, "delta_e32_kcal_per_mol"
            ),
        },
        "formal_vs_diagnostic": {
            "cost_and_same_N_F_use_formal_only": True,
            "call515_appears_only_as_labeled_diagnostic_in_energy_and_evolution": True,
        },
        "evolution_budget_registration": evolution_registration,
        "legacy_evolution_layout_reference": legacy_layout,
    }
    require("openmm" not in sys.modules and "simtk.openmm" not in sys.modules,
            "R2 delivery finalizer imported OpenMM")
    return evidence


def format_binding(value: Mapping[str, Any]) -> str:
    return f"`{value['path']}` (SHA-256 `{value['sha256']}`, {value['bytes']} bytes)"


def render_provenance(template_path: Path, evidence: Mapping[str, Any],
                      audit_path: Path) -> str:
    template = readonly_file(template_path, "R2 provenance template").read_text(
        encoding="utf-8"
    )
    found = {token for token in TOKENS if token in template}
    require(found == TOKENS, "R2 provenance template token inventory differs")
    machine = evidence["machine_outputs"]
    output_order = [
        "cost_to_precision_pdf", "energy_vs_arc_length_pdf", "evolution_pdf",
        "comparison_pack_json", "same_N_F_table_csv", "pdf_qa_json",
    ]
    output_table = "\n".join(
        ["| 角色 | 路径 | SHA-256 | 字节 |", "|---|---|---|---:|"]
        + [f"| {role} | `{machine[role]['path']}` | `{machine[role]['sha256']}` | "
           f"{machine[role]['bytes']} |" for role in output_order]
    )
    derived = evidence["derived"]
    metric_summary = (
        f"同精度首次达到成本：$I_\\perp$ 为 {derived['i_perp']['display']} "
        f"(IVM {derived['i_perp']['ivm_first_passage_N_F']} vs ODE-NEB "
        f"{derived['i_perp']['ode_neb_first_passage_N_F']} NoF)；"
        f"$\\Delta E_{{32}}$ 为 {derived['delta_e32_kcal_per_mol']['display']} "
        f"(IVM {derived['delta_e32_kcal_per_mol']['ivm_first_passage_N_F']} vs "
        f"ODE-NEB {derived['delta_e32_kcal_per_mol']['ode_neb_first_passage_N_F']} NoF)。"
    )
    transport = evidence["ode"]["replica_transport"]
    transport_text = (
        f"wrapper {format_binding(transport['wrapper'])}；receipt "
        f"{format_binding(transport['transport_receipt'])}"
    )
    ir = evidence["ir"]
    ir_text = f"state {format_binding(ir['state_binding'])}；verification {format_binding(ir['verification'])}"
    surface = evidence["surface"]
    surface_text = (
        f"result {format_binding(surface['result'])}；audit {format_binding(surface['audit'])}；"
        f"center adapter {format_binding(surface['center_adapter'])}"
    )
    code_table = "\n".join(
        ["| 角色 | 路径 | SHA-256 | 字节 |", "|---|---|---|---:|"]
        + [f"| {role} | `{record['path']}` | `{record['sha256']}` | {record['bytes']} |"
           for role, record in evidence["codes"].items()]
    )
    replacements = {
        "@CREATED_HKT@": datetime.now(timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        ),
        "@OUTPUT_TABLE@": output_table,
        "@MANIFEST_BINDING@": format_binding(evidence["manifest"]),
        "@COMPLETION_AUDIT_PATH@": str(audit_path),
        "@SELECTED_K@": f"{evidence['ode']['selected_spring_k_eV_per_A2']:g}",
        "@METRIC_SUMMARY@": metric_summary,
        "@ODE_TERMINAL_BINDING@": format_binding(evidence["ode"]["terminal_audit"]),
        "@CACHE_BINDING@": format_binding(evidence["ode"]["incremental_cache"]),
        "@CACHE_TRANSPORT_BINDINGS@": transport_text,
        "@IR_BINDINGS@": ir_text,
        "@SURFACE_BINDINGS@": surface_text,
        "@CODE_TABLE@": code_table,
        "@PDF_QA_SUMMARY@": (
            f"{QA_STATUS}；三页均由 Poppler 在 240 dpi 完整渲染并完成显式视觉检查；"
            f"receipt {format_binding(evidence['pdf_qa'])}。"
        ),
    }
    require(set(replacements) == TOKENS, "R2 provenance replacement set differs")
    rendered = template
    for token, replacement in replacements.items():
        rendered = rendered.replace(token, replacement)
    require(not any(token in rendered for token in TOKENS)
            and "PENDING" not in rendered and "NOT FINAL" not in rendered,
            "R2 final provenance retains a nonfinal marker")
    return rendered.rstrip() + "\n"


def completion_audit(evidence: Mapping[str, Any], provenance_path: Path,
                     provenance_bytes: bytes, audit_path: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **dict(evidence),
        "provenance": {
            "path": str(provenance_path),
            "sha256": sha256_bytes(provenance_bytes),
            "bytes": len(provenance_bytes),
            "pending_markers_remaining": 0,
        },
        "completion_audit_path": str(audit_path),
        "publication": {
            "mode": "O_EXCL_stage_fsync_link_pair",
            "overwrites": False,
            "formal_outputs_created_by_this_finalizer": [
                str(provenance_path), str(audit_path)
            ],
        },
        "physical_calls": 0,
        "openmm_contexts_constructed": 0,
        "bb_locks_acquired": 0,
        "main_pdf_untouched": True,
    }


def publish_pair(provenance_path: Path, audit_path: Path,
                 provenance: bytes, audit: bytes) -> None:
    require(not provenance_path.exists() and not audit_path.exists(),
            "refusing to overwrite R2 provenance/completion audit")
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    require(provenance_path.parent == audit_path.parent,
            "R2 documentation outputs must share one directory")
    staged = []
    published = []
    try:
        for target, data in ((provenance_path, provenance), (audit_path, audit)):
            temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.chmod(temporary, 0o444)
            staged.append((temporary, target))
        for temporary, target in staged:
            os.link(temporary, target); published.append(target)
        directory = os.open(provenance_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _target in staged:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--same-nf-csv", type=Path, required=True)
    parser.add_argument("--pdf-qa", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    evidence = collect_evidence(
        args.manifest, args.comparison, args.same_nf_csv, args.pdf_qa
    )
    provenance_path = EXPECTED_OUTPUTS["provenance_md"].resolve()
    audit_path = EXPECTED_OUTPUTS["completion_audit_json"].resolve()
    text = render_provenance(args.template, evidence, audit_path)
    text_bytes = text.encode("utf-8")
    document = completion_audit(evidence, provenance_path, text_bytes, audit_path)
    audit_bytes = json.dumps(
        document, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8") + b"\n"
    require(document["provenance"]["sha256"] == sha256_bytes(text_bytes)
            and document["physical_calls"] == 0,
            "R2 completion audit closure differs before publication")
    publish_pair(provenance_path, audit_path, text_bytes, audit_bytes)
    print(json.dumps({
        "status": STATUS,
        "provenance": binding(provenance_path),
        "completion_audit": binding(audit_path),
        "physical_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
