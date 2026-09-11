"""Finalize the figure manifest after an explicit Poppler-PNG visual review.

The plotting command deliberately stops at ``awaiting_poppler_visual_qa``.
This command is the only transition to a complete figure manifest.  It does
not alter the pending manifest, the PDF, either PNG, the provenance note, the
audit manifest, or any scientific result.  It creates one new canonical JSON
file and refuses to overwrite it.

Typical use, after rendering the PDF with ``pdftoppm`` and inspecting the
rendered PNG at full resolution::

    python -m src.finalize_figure_qa \
      --qa-png figures/qa/q10_unified_cycle_irivm_vs_article_2x2-1.png \
      --reviewer "Codex view_image full-resolution inspection" \
      --confirmation POPPLER_PNG_VISUALLY_APPROVED
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import struct
from typing import Any, Mapping
import zlib

from .formal_io import (
    FormalIOError,
    atomic_write_json,
    derive_campaign_paths,
    load_canonical_json,
    resolve_registered_path,
    sha256_file,
)


VISUAL_QA_CONFIRMATION = "POPPLER_PNG_VISUALLY_APPROVED"


class FigureQAFinalizationError(RuntimeError):
    """Raised when any pre-QA identity or explicit review gate fails."""


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FigureQAFinalizationError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise FigureQAFinalizationError(f"{label} is not hexadecimal") from exc
    if value != value.lower():
        raise FigureQAFinalizationError(f"{label} must use lowercase hexadecimal")
    return value


def _registered_existing_path(campaign_root: Path, path: Path, label: str) -> tuple[Path, str]:
    root = campaign_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (FileNotFoundError, ValueError) as exc:
        raise FigureQAFinalizationError(
            f"{label} must be an existing file inside the campaign"
        ) from exc
    # Re-resolve through the formal registered-path gate to reject symlink
    # components and non-files, not merely lexical escapes.
    checked = resolve_registered_path(root, relative)
    return checked, relative


def _registered_new_path(campaign_root: Path, path: Path, label: str) -> tuple[Path, str]:
    root = campaign_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise FigureQAFinalizationError(
            f"{label} must remain inside the campaign"
        ) from exc
    if resolved.exists():
        raise FigureQAFinalizationError(f"refusing to overwrite {label}: {resolved}")
    portable = PurePosixPath(relative)
    if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
        raise FigureQAFinalizationError(f"non-portable {label} path: {relative!r}")
    cursor = root
    for part in portable.parts[:-1]:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise FigureQAFinalizationError(f"symlink component in {label}: {cursor}")
    return resolved, relative


def _checked_output_specs(
    pending: Mapping[str, Any], pending_path: Path, campaign_root: Path
) -> dict[str, dict[str, str]]:
    raw_outputs = pending.get("outputs")
    if not isinstance(raw_outputs, dict) or len(raw_outputs) != 3:
        raise FigureQAFinalizationError("pending manifest must declare exactly three outputs")
    classified: dict[str, dict[str, str]] = {}
    suffix_to_role = {".pdf": "pdf", ".png": "png", ".md": "provenance"}
    for name, expected_raw in raw_outputs.items():
        if not isinstance(name, str) or PurePosixPath(name).name != name:
            raise FigureQAFinalizationError("pending output names must be safe basenames")
        role = suffix_to_role.get(Path(name).suffix.lower())
        if role is None or role in classified:
            raise FigureQAFinalizationError("pending outputs must be one PDF, one PNG, and one MD")
        expected = _require_sha256(expected_raw, f"pending output {name}")
        path, relative = _registered_existing_path(
            campaign_root, pending_path.parent / name, f"pending {role}"
        )
        actual = sha256_file(path)
        if actual != expected:
            raise FigureQAFinalizationError(
                f"{role} hash drift: expected {expected}, got {actual}"
            )
        classified[role] = {"relative_path": relative, "sha256": actual}
    if set(classified) != {"pdf", "png", "provenance"}:
        raise FigureQAFinalizationError("pending output roles are incomplete")
    pdf_prefix = resolve_registered_path(
        campaign_root, classified["pdf"]["relative_path"]
    ).read_bytes()[:8]
    if not pdf_prefix.startswith(b"%PDF-"):
        raise FigureQAFinalizationError("declared PDF lacks a PDF header")
    return classified


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Validate the complete PNG chunk/CRC envelope and return IHDR dimensions."""

    raw = path.read_bytes()
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise FigureQAFinalizationError("QA artifact is not a PNG")
    offset = 8
    dimensions: tuple[int, int] | None = None
    saw_idat = False
    saw_iend = False
    first = True
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise FigureQAFinalizationError("truncated PNG chunk envelope")
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        kind = raw[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(raw):
            raise FigureQAFinalizationError("truncated PNG chunk data")
        expected_crc = struct.unpack(">I", raw[data_end:crc_end])[0]
        actual_crc = zlib.crc32(kind)
        actual_crc = zlib.crc32(raw[data_start:data_end], actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise FigureQAFinalizationError("PNG chunk CRC mismatch")
        if first:
            if kind != b"IHDR" or length != 13:
                raise FigureQAFinalizationError("PNG does not begin with a valid IHDR")
            width, height = struct.unpack(">II", raw[data_start : data_start + 8])
            if width <= 0 or height <= 0:
                raise FigureQAFinalizationError("PNG dimensions must be positive")
            dimensions = (width, height)
            first = False
        elif kind == b"IHDR":
            raise FigureQAFinalizationError("PNG contains multiple IHDR chunks")
        if kind == b"IDAT":
            saw_idat = True
        if kind == b"IEND":
            if length != 0 or crc_end != len(raw):
                raise FigureQAFinalizationError("invalid or non-terminal PNG IEND")
            saw_iend = True
            offset = crc_end
            break
        offset = crc_end
    if dimensions is None or not saw_idat or not saw_iend or offset != len(raw):
        raise FigureQAFinalizationError("PNG is missing required chunks")
    return dimensions


def _verify_authenticated_inputs(
    pending: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    campaign_root: Path,
    audit_manifest_path: Path,
) -> None:
    authenticated = pending.get("authenticated_inputs")
    if not isinstance(authenticated, dict):
        raise FigureQAFinalizationError("pending authenticated_inputs is absent")
    audit_artifacts = audit.get("artifacts")
    if not isinstance(audit_artifacts, dict):
        raise FigureQAFinalizationError("audit artifact manifest is absent")
    # Re-hash every audit artifact, not only the two tables read by the plotter.
    # This makes completion refuse any post-audit drift in the scientific
    # report set while still keeping the finalizer read-only with respect to it.
    for name, raw_spec in audit_artifacts.items():
        if not isinstance(name, str) or not isinstance(raw_spec, dict):
            raise FigureQAFinalizationError("malformed audit artifact declaration")
        relative = raw_spec.get("relative_path")
        if not isinstance(relative, str):
            raise FigureQAFinalizationError(f"audit artifact {name} lacks a path")
        artifact_path = resolve_registered_path(audit_manifest_path.parent, relative)
        expected = _require_sha256(raw_spec.get("sha256"), f"audit artifact {name}")
        if sha256_file(artifact_path) != expected:
            raise FigureQAFinalizationError(f"audit artifact hash drift: {name}")
    for name in ("checkpoint_comparison.csv", "public_trajectory_metrics.csv"):
        spec = audit_artifacts.get(name)
        if not isinstance(spec, dict):
            raise FigureQAFinalizationError(f"audit does not declare {name}")
        expected = _require_sha256(spec.get("sha256"), f"audit artifact {name}")
        if authenticated.get(name) != expected:
            raise FigureQAFinalizationError(f"pending/audit identity mismatch for {name}")
    registration = audit.get("registration")
    if not isinstance(registration, dict):
        raise FigureQAFinalizationError("audit registration identity is absent")
    registration_sha = _require_sha256(
        registration.get("sha256"), "audit registration"
    )
    registration_relative = registration.get("relative_path")
    seal_relative = registration.get("seal_relative_path")
    if not isinstance(registration_relative, str) or not isinstance(seal_relative, str):
        raise FigureQAFinalizationError("audit registration paths are absent")
    registration_path = resolve_registered_path(campaign_root, registration_relative)
    seal_path = resolve_registered_path(campaign_root, seal_relative)
    if sha256_file(registration_path) != registration_sha:
        raise FigureQAFinalizationError("registered preregistration hash drift")
    seal_sha = _require_sha256(registration.get("seal_sha256"), "audit registration seal")
    if sha256_file(seal_path) != seal_sha:
        raise FigureQAFinalizationError("registered preregistration seal hash drift")
    if authenticated.get("registration_sha256") != registration_sha:
        raise FigureQAFinalizationError("pending/audit registration identity mismatch")
    runs = audit.get("runs")
    if not isinstance(runs, list) or not runs:
        raise FigureQAFinalizationError("audit run identities are absent")
    expected_results: dict[str, str] = {}
    expected_receipts: dict[str, str] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise FigureQAFinalizationError("malformed audit run identity")
        key = str(int(run["n_internal"]))
        if key in expected_results:
            raise FigureQAFinalizationError("duplicate audit run resolution")
        expected_results[key] = _require_sha256(
            run.get("result_sha256"), f"N={key} result"
        )
        expected_receipts[key] = _require_sha256(
            run.get("receipt_sha256"), f"N={key} receipt"
        )
        result_relative = run.get("result_path")
        receipt_relative = run.get("receipt_path")
        if not isinstance(result_relative, str) or not isinstance(receipt_relative, str):
            raise FigureQAFinalizationError(f"N={key} run paths are absent")
        result_path = resolve_registered_path(campaign_root, result_relative)
        receipt_path = resolve_registered_path(campaign_root, receipt_relative)
        if sha256_file(result_path) != expected_results[key]:
            raise FigureQAFinalizationError(f"N={key} result hash drift")
        if sha256_file(receipt_path) != expected_receipts[key]:
            raise FigureQAFinalizationError(f"N={key} receipt hash drift")
    if authenticated.get("result_sha256") != expected_results:
        raise FigureQAFinalizationError("pending/audit result identities mismatch")
    if authenticated.get("receipt_sha256") != expected_receipts:
        raise FigureQAFinalizationError("pending/audit receipt identities mismatch")


def finalize_visual_qa(
    pending_manifest_path: Path,
    audit_manifest_path: Path,
    qa_png_path: Path,
    final_manifest_path: Path,
    *,
    campaign_root: Path,
    confirmation: str,
    reviewer: str,
) -> Path:
    """Create an immutable complete manifest after every identity gate passes."""

    if confirmation != VISUAL_QA_CONFIRMATION:
        raise FigureQAFinalizationError(
            "explicit Poppler visual-QA confirmation token is required"
        )
    normalized_reviewer = reviewer.strip()
    if not normalized_reviewer or len(normalized_reviewer) > 240:
        raise FigureQAFinalizationError("reviewer must be a non-empty label of at most 240 chars")

    campaign = campaign_root.resolve(strict=True)
    pending_path, pending_relative = _registered_existing_path(
        campaign, pending_manifest_path, "pending figure manifest"
    )
    audit_path, audit_relative = _registered_existing_path(
        campaign, audit_manifest_path, "audit manifest"
    )
    qa_path, qa_relative = _registered_existing_path(
        campaign, qa_png_path, "Poppler QA PNG"
    )
    final_path, final_relative = _registered_new_path(
        campaign, final_manifest_path, "final figure manifest"
    )
    if final_path.suffix.lower() != ".json":
        raise FigureQAFinalizationError("final figure manifest must be JSON")

    pending = load_canonical_json(pending_path)
    if not isinstance(pending, dict) or pending.get("schema_version") != 1:
        raise FigureQAFinalizationError("unsupported pending figure manifest schema")
    if pending.get("status") != "awaiting_poppler_visual_qa":
        raise FigureQAFinalizationError("pending figure is not awaiting Poppler visual QA")
    if pending.get("qa_png_sha256") is not None:
        raise FigureQAFinalizationError("pending figure already contains a QA identity")
    if pending.get("cryptographic_provenance_complete") is not True:
        raise FigureQAFinalizationError("pending cryptographic provenance is incomplete")
    if pending.get("formal_provenance_complete") is not False:
        raise FigureQAFinalizationError("pending formal-provenance state is not fail-closed")

    pending_sha = sha256_file(pending_path)
    expected_audit_sha = _require_sha256(
        pending.get("audit_manifest_sha256"), "pending audit manifest"
    )
    actual_audit_sha = sha256_file(audit_path)
    if actual_audit_sha != expected_audit_sha:
        raise FigureQAFinalizationError("audit manifest hash drift")
    if pending.get("audit_manifest_relative_path") != audit_path.name:
        raise FigureQAFinalizationError("pending audit-manifest basename mismatch")
    audit = load_canonical_json(audit_path)
    if not isinstance(audit, dict):
        raise FigureQAFinalizationError("audit manifest root is not an object")
    if audit.get("outcome") != "complete_scientific_pass":
        raise FigureQAFinalizationError("audit outcome is not complete_scientific_pass")
    if audit.get("integrity_complete") is not True:
        raise FigureQAFinalizationError("audit integrity is incomplete")
    _verify_authenticated_inputs(
        pending,
        audit,
        campaign_root=campaign,
        audit_manifest_path=audit_path,
    )

    outputs = _checked_output_specs(pending, pending_path, campaign)
    plot_source_path = resolve_registered_path(campaign, "src/plot_comparison.py")
    plot_source_sha = sha256_file(plot_source_path)
    if plot_source_sha != _require_sha256(
        pending.get("plot_source_sha256"), "pending plot source"
    ):
        raise FigureQAFinalizationError("plot source hash drift")
    finalizer_relative = Path(__file__).resolve(strict=True).relative_to(campaign).as_posix()
    finalizer_sha = sha256_file(Path(__file__))
    registered_sources = audit.get("source_sha256")
    if not isinstance(registered_sources, dict):
        raise FigureQAFinalizationError("audit registered-source identities are absent")
    for relative, actual in (
        ("src/plot_comparison.py", plot_source_sha),
        (finalizer_relative, finalizer_sha),
    ):
        expected = _require_sha256(
            registered_sources.get(relative), f"registered source {relative}"
        )
        if actual != expected:
            raise FigureQAFinalizationError(
                f"current source is not the registered source: {relative}"
            )

    qa_dimensions = _png_dimensions(qa_path)
    qa_sha = sha256_file(qa_path)
    if qa_path == resolve_registered_path(campaign, outputs["png"]["relative_path"]):
        raise FigureQAFinalizationError("QA PNG must be a separate Poppler-rendered file")
    if qa_sha == outputs["png"]["sha256"]:
        raise FigureQAFinalizationError(
            "QA PNG is byte-identical to the direct Matplotlib PNG; independent rendering is required"
        )

    provenance_path = resolve_registered_path(
        campaign, outputs["provenance"]["relative_path"]
    )
    try:
        provenance_text = provenance_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FigureQAFinalizationError("provenance note is not UTF-8") from exc
    for label, digest in (
        ("audit", actual_audit_sha),
        ("plot source", plot_source_sha),
        ("PDF", outputs["pdf"]["sha256"]),
        ("PNG", outputs["png"]["sha256"]),
    ):
        if digest not in provenance_text:
            raise FigureQAFinalizationError(
                f"provenance note does not bind the current {label} identity"
            )

    final_manifest = {
        "schema_version": 1,
        "status": "complete",
        "outcome": "complete_visual_qa_pass",
        "completion_gate": "explicit_post_poppler_png_visual_review",
        "pre_qa_manifest": {
            "relative_path": pending_relative,
            "sha256": pending_sha,
        },
        "audit_manifest": {
            "relative_path": audit_relative,
            "sha256": actual_audit_sha,
        },
        "plot_source": {
            "relative_path": "src/plot_comparison.py",
            "sha256": plot_source_sha,
        },
        "finalizer_source": {
            "relative_path": finalizer_relative,
            "sha256": finalizer_sha,
        },
        "artifacts": {
            **outputs,
            "poppler_rendered_qa_png": {
                "relative_path": qa_relative,
                "sha256": qa_sha,
                "width_pixels": qa_dimensions[0],
                "height_pixels": qa_dimensions[1],
            },
        },
        "authenticated_inputs": pending["authenticated_inputs"],
        "checkpoint_counts": pending.get("checkpoint_counts"),
        "methods": pending.get("methods"),
        "densification_count": pending.get("densification_count"),
        "reference_barrier": pending.get("reference_barrier"),
        "visual_qa": {
            "status": "passed",
            "renderer": "Poppler pdftoppm",
            "confirmation": VISUAL_QA_CONFIRMATION,
            "reviewer": normalized_reviewer,
            "rendered_from_pdf_sha256": outputs["pdf"]["sha256"],
        },
        "cryptographic_provenance_complete": True,
        "formal_provenance_complete": True,
        "scientific_results_modified": False,
        "final_manifest_relative_path": final_relative,
    }
    # The only write in this function is a new, refuse-overwrite canonical
    # manifest after all reads and gates above have completed.
    atomic_write_json(final_path, final_manifest)
    # Reopen and byte-canonicalize before reporting success.
    if load_canonical_json(final_path) != final_manifest:
        raise FigureQAFinalizationError("final figure manifest reopen mismatch")
    return final_path


def main() -> int:
    paths = derive_campaign_paths(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pending-manifest",
        type=Path,
        default=paths.campaign_root / "figures/figure_manifest.json",
    )
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=paths.campaign_root / "reports/audit/audit_manifest.json",
    )
    parser.add_argument("--qa-png", type=Path, required=True)
    parser.add_argument(
        "--final-manifest",
        type=Path,
        default=paths.campaign_root / "figures/final_figure_manifest.json",
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--confirmation", required=True)
    arguments = parser.parse_args()
    try:
        final_path = finalize_visual_qa(
            arguments.pending_manifest,
            arguments.audit_manifest,
            arguments.qa_png,
            arguments.final_manifest,
            campaign_root=paths.campaign_root,
            confirmation=arguments.confirmation,
            reviewer=arguments.reviewer,
        )
    except (
        FigureQAFinalizationError,
        FormalIOError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FIGURE QA FINALIZATION REFUSED: {exc}")
        return 1
    print(f"FIGURE QA FINALIZATION COMPLETE: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
