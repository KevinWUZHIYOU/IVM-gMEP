#!/usr/bin/env python3
"""Write a deterministic GNU-compatible SHA256SUMS file for one archive."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    output = root / "SHA256SUMS"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != output and not path.is_symlink()
    )
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} with {len(paths)} entries")


if __name__ == "__main__":
    main()
