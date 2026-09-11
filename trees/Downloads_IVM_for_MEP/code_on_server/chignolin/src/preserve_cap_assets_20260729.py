#!/usr/bin/env python3
"""Capture or verify immutable cap=15/18/20/25/30 Chignolin assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


RELATIVE_ROOTS = (
    "reports/reviewer_subproblem_cap15_dense_strict_20260729a",
    "reports/reviewer_fixed_alpha_inner18_20260729a",
    "reports/reviewer_fixed_alpha_inner20_20260729a",
    "reports/reviewer_fixed_alpha_inner25_20260729a",
    "reports/reviewer_fixed_alpha_inner30_20260729a",
    "reports/reviewer_subproblem_caps18_20_dense_20260729a",
    "reports/finalized_subproblem_caps18_20_20260729a",
    "reports/finalized_subproblem_caps18_20_common17_20260729a",
    "reports/reviewer_subproblem_cap_score_20260729a",
    "figures/reviewer_subproblem_cap_alpha_well_20260729a",
    "figures/reviewer_subproblem_cap_comparison_20260729a",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict]:
    records = []
    for relative_root in RELATIVE_ROOTS:
        directory = root / relative_root
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            stat = path.stat()
            records.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": int(stat.st_size),
                    "sha256": sha256(path),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("capture", "verify"), required=True
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    current = inventory(root)
    if args.mode == "capture":
        if manifest.exists():
            raise FileExistsError(
                f"Refusing to overwrite preservation manifest: {manifest}"
            )
        payload = {
            "purpose": (
                "Immutable pre-transactional inventory of all retained "
                "Chignolin cap=15/18/20/25/30 reports and figures."
            ),
            "root": str(root),
            "relative_roots": list(RELATIVE_ROOTS),
            "records": current,
        }
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, manifest)
        print(manifest)
        return
    expected = json.loads(manifest.read_text())
    if expected["records"] != current:
        expected_by_path = {
            record["relative_path"]: record
            for record in expected["records"]
        }
        current_by_path = {
            record["relative_path"]: record for record in current
        }
        changed = sorted(
            path
            for path in set(expected_by_path) | set(current_by_path)
            if expected_by_path.get(path) != current_by_path.get(path)
        )
        raise RuntimeError(
            "Preserved cap assets changed: " + ", ".join(changed)
        )
    print(
        json.dumps(
            {
                "verified": True,
                "manifest": str(manifest),
                "manifest_sha256": sha256(manifest),
                "file_count": len(current),
            }
        )
    )


if __name__ == "__main__":
    main()
