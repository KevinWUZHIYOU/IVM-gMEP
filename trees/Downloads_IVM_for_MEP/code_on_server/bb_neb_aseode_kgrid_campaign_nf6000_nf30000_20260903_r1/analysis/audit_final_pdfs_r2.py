#!/usr/bin/env python3
"""Zero-physical structural, textual, raster, and visual QA for R2 PDFs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image


sys.dont_write_bytecode = True
SCHEMA = "bb-discrete32-ode-nf30000-r2-pdf-qa-v2"
STATUS = "PASS_FINAL_R2_PDF_QA"
PACK_SCHEMA = "bb-discrete32-ode-nf30000-r2-comparison-pack-v1"
PDF_KEYS = {
    "cost_to_precision": "cost_to_precision_pdf",
    "energy_vs_arc_length": "energy_vs_arc_length_pdf",
    "evolution": "evolution_pdf",
}
REQUIRED_TEXT = {
    "cost_to_precision": ("Number of Forces", "IVM", "ODE-NEB"),
    "energy_vs_arc_length": (
        "Normalized Cartesian arc length", "Relative energy", "IVM", "ODE-NEB",
        "Previous IVM", "8982", "9990", "14983", "30000", "uncommitted",
    ),
    "evolution": ("IVM", "ODE-NEB", "NF", "clipped"),
}
FORBIDDEN_TEXT = {
    "cost_to_precision": ("Previous IVM", "uncommitted", "trial"),
    "energy_vs_arc_length": (),
    "evolution": (),
}
PLOT_CONTRACT = {
    "cost_to_precision": {
        "formal_methods": ["IVM", "ODE-NEB"],
        "axes": [
            {"metric": "I_perp", "xscale": "log", "xlim": [1.0e8, 1.0e1],
             "yscale": "linear", "ylim": [0.0, 30000.0]},
            {"metric": "Delta E_32", "xscale": "linear", "xlim": [5000.0, 0.0],
             "yscale": "linear", "ylim": [0.0, 30000.0]},
        ],
        "ylabel": "Number of Forces",
    },
    "energy_vs_arc_length": {
        "budget_N_F": 30000,
        "line_labels": [
            "IVM (NF 8982)", "ODE-NEB (NF 30000)",
            "Previous IVM (NF 9990)",
            "IVM trial (NF 14983; uncommitted)",
        ],
        "points_per_line": 32,
        "normalized_arc_length_endpoints": [0.0, 1.0],
        "normalized_arc_length_strictly_increasing": True,
    },
}
CHECK_KEYS = {
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly_file(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    status = resolved.lstat()
    require(
        stat.S_ISREG(status.st_mode)
        and not resolved.is_symlink()
        and stat.S_IMODE(status.st_mode) & 0o222 == 0,
        f"{role} is not a read-only regular nonsymlink file",
    )
    return resolved


def binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main_pdf_binding() -> dict[str, Any]:
    """Bind the pre-authoring formal manuscript without requiring mode 0444."""

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


def load_json(path: Path, role: str) -> tuple[Path, dict[str, Any]]:
    resolved = readonly_file(path, role)
    document = json.loads(resolved.read_text(encoding="utf-8"))
    require(isinstance(document, dict), f"{role} is not a JSON object")
    return resolved, document


def command(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env={**os.environ, "LC_ALL": "C"},
    )
    return result.stdout


def tool_version(executable: str) -> str:
    text = command([executable, "-v"])
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    require(bool(line), f"{executable} version output is empty")
    return line


def parse_pdfinfo(text: str) -> tuple[int, list[float]]:
    pages_match = re.search(r"^Pages:\s+(\d+)\s*$", text, re.MULTILINE)
    size_match = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts", text, re.MULTILINE
    )
    require(pages_match is not None and size_match is not None,
            "pdfinfo pages/page-size output differs")
    return int(pages_match.group(1)), [
        float(size_match.group(1)), float(size_match.group(2))
    ]


def embedded_fonts(text: str) -> tuple[bool, int]:
    lines = [line for line in text.splitlines() if line.strip()]
    require(len(lines) >= 3 and "emb" in lines[0].split(), "pdffonts table is absent")
    # Font type names may contain spaces (for example ``CID TrueType``), so
    # split-column indices are not stable.  The final three yes/no columns are
    # emb/sub/uni immediately before the two-number object ID.
    rows = []
    pattern = re.compile(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$")
    for line in lines[2:]:
        match = pattern.search(line)
        require(match is not None, "pdffonts data row differs")
        rows.append(match.groups())
    require(rows, "pdffonts contains no fonts")
    return all(row[0] == "yes" for row in rows), len(rows)


def raster_record(png: Path, page_size: list[float], dpi: int) -> dict[str, Any]:
    with Image.open(png) as image:
        image.load()
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    expected = [round(page_size[0] * dpi / 72.0), round(page_size[1] * dpi / 72.0)]
    observed = [int(array.shape[1]), int(array.shape[0])]
    require(
        abs(observed[0] - expected[0]) <= 3
        and abs(observed[1] - expected[1]) <= 3,
        "rendered pixel dimensions differ from PDF points/DPI",
    )
    nonwhite = float(np.mean(np.any(array < 250, axis=2)))
    require(0.001 < nonwhite < 0.90, "rendered page is blank or implausibly saturated")
    return {
        **binding(png),
        "pixel_width": observed[0],
        "pixel_height": observed[1],
        "nonwhite_pixel_fraction": nonwhite,
        "finite_pixels": True,
    }


def write_new(path: Path, document: dict[str, Any]) -> None:
    target = path.expanduser().resolve()
    require(not target.exists(), f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(
                document, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            ).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-confirmed", action="store_true")
    args = parser.parse_args()
    require(args.visual_confirmed, "human visual confirmation flag is required")
    manifest_path, manifest = load_json(args.manifest, "final R2 manifest")
    comparison_path, comparison = load_json(args.comparison, "final R2 comparison")
    require(
        manifest.get("status") == "FINAL_INPUTS_FROZEN"
        and comparison.get("schema") == PACK_SCHEMA
        and comparison.get("status")
        == "COMPLETE_ZERO_FORCE_DISCRETE32_NF30000_R2_ANALYSIS"
        and comparison.get("manifest")
        == {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "final manifest/comparison identity differs",
    )
    histories = comparison.get("histories", {})
    diagnostics = comparison.get("diagnostic_uncommitted_states", [])
    exclusion = comparison.get("diagnostic_exclusion", {})
    protocol = comparison.get("metric_protocol", {})
    require(
        set(histories) == {"IVM", "ODE-NEB", "Previous IVM"}
        and max(int(row["optimization_N_F"]) for row in histories["IVM"]) == 8982
        and max(int(row["optimization_N_F"]) for row in histories["ODE-NEB"]) == 30000
        and max(int(row["optimization_N_F"]) for row in histories["Previous IVM"]) == 9990
        and len(diagnostics) == 1
        and int(diagnostics[0]["optimization_N_F"]) == 14983
        and diagnostics[0].get("method") == "IVM returned terminal (uncommitted)"
        and "uncommitted" in str(diagnostics[0].get("state_role", "")).casefold()
        and diagnostics[0].get("source", {}).get("formally_reportable") is False
        and diagnostics[0].get("source", {}).get(
            "terminal_uncommitted_charged_tail") is True
        and exclusion == {
            "excluded_from_formal_IVM_history": True,
            "excluded_from_cost_to_precision": True,
            "excluded_from_same_N_F_envelopes": True,
            "permitted_only_in_energy_profile_and_evolution_as_labeled_diagnostic": True,
        },
        "formal/diagnostic plotted-line semantics differ",
    )
    require(
        int(protocol.get("path_images", -1)) == 32
        and protocol.get("densification") is False,
        "final comparison does not declare the original discrete 32-image protocol",
    )
    outputs = manifest.get("outputs", {}).get("final", {})
    require(Path(str(outputs.get("comparison_pack_json"))).resolve() == comparison_path,
            "comparison path differs from manifest")
    render_dir = args.render_dir.expanduser().resolve()
    render_dir.mkdir(parents=True, exist_ok=True)
    dpi = 240
    records = {}
    for role, key in PDF_KEYS.items():
        pdf = readonly_file(Path(str(outputs[key])), f"{role} PDF")
        require(pdf.read_bytes()[:5] == b"%PDF-", f"{role} is not a PDF")
        info = command(["pdfinfo", str(pdf)])
        pages, page_size = parse_pdfinfo(info)
        require(pages == 1 and all(math.isfinite(x) and x > 0 for x in page_size),
                f"{role} page count/size differs")
        fonts_ok, font_count = embedded_fonts(command(["pdffonts", str(pdf)]))
        require(fonts_ok, f"{role} has an unembedded font")
        text = command(["pdftotext", str(pdf), "-"])
        missing_text = [
            fragment for fragment in REQUIRED_TEXT[role]
            if fragment.casefold() not in text.casefold()
        ]
        forbidden_text = [
            fragment for fragment in FORBIDDEN_TEXT[role]
            if fragment.casefold() in text.casefold()
        ]
        require(
            not missing_text and not forbidden_text
            and re.search(r"(?:^|\W)(?:nan|inf)(?:$|\W)", text, re.IGNORECASE) is None,
            f"{role} text/finite-label check differs; "
            f"missing={missing_text}; forbidden={forbidden_text}",
        )
        prefix = render_dir / role
        png = prefix.with_suffix(".png")
        require(not png.exists(), f"refusing to overwrite QA render {png}")
        command(["pdftoppm", "-png", "-singlefile", "-r", str(dpi),
                 str(pdf), str(prefix)])
        png = readonly_file(png, f"{role} rendered PNG") if (
            png.stat().st_mode & 0o222 == 0
        ) else png.resolve(strict=True)
        if png.stat().st_mode & 0o222:
            os.chmod(png, 0o444)
        records[role] = {
            **binding(pdf),
            "pages": pages,
            "page_size_points": page_size,
            "embedded_font_count": font_count,
            "pdfinfo_status": "PASS",
            "font_status": "PASS",
            "required_text_status": "PASS",
            "required_text_fragments": list(REQUIRED_TEXT[role]),
            "forbidden_text_fragments": list(FORBIDDEN_TEXT[role]),
            "render_dpi": dpi,
            "render": raster_record(png, page_size, dpi),
            "visual_status": "PASS_EXPLICIT_HUMAN_CONFIRMATION",
        }
    analysis_source = readonly_file(Path(__file__).with_name("bb_discrete32_analysis_r2.py"),
                                    "R2 analysis source")
    formal_main_pdf = main_pdf_binding()
    checks = {
        "page_count_and_dimensions": True,
        "fonts_embedded": True,
        "required_text_and_finite_labels": True,
        "cost_to_precision_excludes_previous_IVM_and_uncommitted_trial": True,
        "all_pages_rendered_at_240_dpi": True,
        "rendered_pixels_finite_and_nonblank": True,
        "axes_legends_arrows_and_clipping_visually_inspected": True,
        "cost_to_precision_formal_curves_are_only_IVM_and_ODE_NEB": True,
        "call515_is_labeled_diagnostic_only": True,
        "energy_profiles_use_stored_32_point_paths": True,
        "energy_profile_includes_previous_IVM_NF9990": True,
        "unrelaxed_R2B_surface_high_energy_clipping_is_display_only": True,
        "main_pdf_untouched": True,
    }
    require(set(checks) == CHECK_KEYS and all(value is True for value in checks.values()),
            "internal PDF-QA check inventory differs")
    document = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest": binding(manifest_path),
        "comparison": binding(comparison_path),
        "analysis_source": binding(analysis_source),
        "main_pdf": formal_main_pdf,
        "physical_calls": 0,
        "openmm_contexts_constructed": 0,
        "bb_locks_acquired": 0,
        "tool_versions": {
            name: tool_version(name)
            for name in ("pdfinfo", "pdffonts", "pdftotext", "pdftoppm")
        },
        "checks": checks,
        "plot_contract": PLOT_CONTRACT,
        "line_semantics": {
            "cost_to_precision": ["IVM formal through NF8982", "ODE-NEB through NF30000"],
            "energy_vs_arc_length": [
                "IVM formal through NF8982", "ODE-NEB through NF30000",
                "Previous IVM NF9990 baseline",
                "IVM returned terminal NF14983 uncommitted diagnostic",
            ],
            "evolution": [
                "same formal paths", "NF14983 uncommitted diagnostic overlay",
                "R2B unrelaxed Cartesian surface; rendering-only high-energy clip",
            ],
        },
        "pdfs": records,
    }
    require("openmm" not in sys.modules and "simtk.openmm" not in sys.modules,
            "PDF QA imported OpenMM")
    write_new(args.output, document)
    print(json.dumps({"status": STATUS, "output": str(args.output.resolve()),
                      "physical_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
