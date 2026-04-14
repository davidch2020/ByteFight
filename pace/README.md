PACE tuning workflow for `YolandaV8Eval`

What this gives you:
- a single stable base agent: `3600-agents/YolandaV8Eval/agent.py`
- a tiny set of named eval weights at the top of that file
- a build script that stamps different weight sets into copied agent folders

Typical workflow:
1. Edit `pace/eval_variants.json` to add candidate weight sets.
2. Run:
   `.venv/bin/python pace/build_eval_variants.py`
3. Upload the generated folders from `pace/out/` to PACE.

Notes:
- The generated agents only change eval weights. Search/time/order behavior stays
  identical to `YolandaV8Eval`.
- This is meant for controlled regression / PACE tuning. Keep the feature set
  fixed and only tune the weights.

Current tunable weights:
- `EVAL_MARGIN_BASE`
- `EVAL_MARGIN_TURNS`
- `EVAL_SELF_CARPET`
- `EVAL_OPP_CARPET`
- `EVAL_PRIMED_COUNT`
- `EVAL_CAN_PRIME`
- `EVAL_ADJ_PRIMED`

Suggested starting philosophy:
- vary 4-7 weights only
- use a stable baseline opponent first
- avoid mixing search/time changes into the same tuning run

Head-to-head uploads:
- To prepare the current `Yolanda` and `GarryImported` folders for direct PACE
  comparison, run:
  `.venv/bin/python pace/build_match_agents.py`
- This will copy them into:
  - `pace/out/YolandaCurrent`
  - `pace/out/GarryImported`
