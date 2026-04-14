#!/usr/bin/env python3
"""Generate PACE-ready eval variants from YolandaV8Eval.

This script copies `3600-agents/YolandaV8Eval/agent.py` into named output
folders and rewrites the tunable eval constants near the top of the file.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil


ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_AGENT = ROOT / "3600-agents" / "YolandaV8Eval" / "agent.py"
CONFIG = ROOT / "pace" / "eval_variants.json"
OUT_DIR = ROOT / "pace" / "out"

TUNABLES = [
    "EVAL_MARGIN_BASE",
    "EVAL_MARGIN_TURNS",
    "EVAL_SELF_CARPET",
    "EVAL_OPP_CARPET",
    "EVAL_PRIMED_COUNT",
    "EVAL_CAN_PRIME",
    "EVAL_ADJ_PRIMED",
]


def apply_weights(source: str, weights: dict[str, float]) -> str:
    updated = source
    for key in TUNABLES:
        if key not in weights:
            continue
        pattern = rf"^{key}\s*=\s*[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?\s*$"
        replacement = f"{key} = {weights[key]}"
        updated, count = re.subn(pattern, replacement, updated, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Could not update {key} cleanly")
    return updated


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    source = BASE_AGENT.read_text()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for variant in cfg["variants"]:
        name = variant["name"]
        weights = variant["weights"]
        target_dir = OUT_DIR / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_agent = target_dir / "agent.py"
        target_agent.write_text(apply_weights(source, weights))

    print(f"Generated {len(cfg['variants'])} variants in {OUT_DIR}")


if __name__ == "__main__":
    main()
