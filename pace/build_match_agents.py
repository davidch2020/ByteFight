#!/usr/bin/env python3
"""Prepare head-to-head agent folders for PACE uploads.

This script copies selected agent folders into `pace/out/` so they can be
uploaded directly for large-scale comparisons.
"""

from __future__ import annotations

import pathlib
import shutil


ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT / "3600-agents"
OUT_DIR = ROOT / "pace" / "out"

# Source folder -> output folder
AGENT_COPIES = {
    "Yolanda": "YolandaCurrent",
    "BobExperiment": "BobExperiment",
}


def copy_agent(src_name: str, dst_name: str) -> None:
    src = AGENTS_DIR / src_name
    dst = OUT_DIR / dst_name
    if not src.exists():
        raise FileNotFoundError(f"Missing agent folder: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in AGENT_COPIES.items():
        copy_agent(src_name, dst_name)
    print(f"Prepared {len(AGENT_COPIES)} head-to-head agent folders in {OUT_DIR}")


if __name__ == "__main__":
    main()
