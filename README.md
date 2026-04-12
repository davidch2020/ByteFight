# ByteFight

## Status

Last updated: 2026-04-12 EDT

Current goal: keep the repo easy to reason about while testing one serious idea at a time.

## Current Layout

- `3600-agents/Yolanda/agent.py`
  - Main active working bot.
  - Current direction:
    - iterative deepening with alpha-beta
    - rat belief tracking
    - simplified phase-aware heuristic
    - root and interior search handled with EV shortcuts, not full chance-node expectiminimax
- `3600-agents/YolandaApurbo/agent.py`
  - Current partner branch copy.
  - This now matches the former `YolandaAgent4` code.
- `3600-agents/YolandaV7/agent.py`
  - Preserved checkpoint from the stronger earlier Albert run.
  - Useful “safe fallback” snapshot.
- `3600-agents/YolandaV8/agent.py`
  - Unified live cell-potential experiment.
  - Recomputes run arrays from the current tree node inside `_eval()`.
- `3600-agents/experiments/YolandaAgent2`
  - Archived experiment branch.
- `3600-agents/experiments/YolandaAgent3`
  - Archived experiment branch.
- `3600-agents/YolandaV1` through `3600-agents/YolandaV6`
  - Historical checkpoints kept for reference and regression testing.
- `3600-agents/RandomAgent/agent.py`
  - Random baseline for smoke testing.
- `3600-agents/matches/`
  - Local batch match reports written by `run_many.py`.

## Repo Notes

- `run_many.py` supports parallel local batches and writes reports into `3600-agents/matches`.
- `engine/run_local_agents.py` supports explicit output paths for batch workers.
- Zip artifacts, depth sweep files, and older clutter have been cleaned out.

## Current Heuristic Direction

For `Yolanda`, the current simplified active heuristic is trying to focus on a smaller core:

- point margin
- immediate carpet value
- current-square prime lane value
- opponent immediate carpet threat
- simple phase scaling

The more speculative helper terms are still available in code where useful, but the active path has been trimmed back on purpose.

For `YolandaV8`, the experiment is different:

- compute fresh run-length arrays from the live board at each evaluated node
- assign each `SPACE` or `PRIMED` cell a lane-based potential
- score the board as:
  - current score margin
  - plus my discounted access to valuable cells
  - minus the opponent’s discounted access to those same cells

## Useful Commands

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run one local match:

```bash
.venv/bin/python engine/run_local_agents.py Yolanda Yolanda
```

Run a local batch:

```bash
.venv/bin/python run_many.py Yolanda RandomAgent -n 10 -j 4
.venv/bin/python run_many.py Yolanda YolandaV7 -n 10 -j 4
.venv/bin/python run_many.py YolandaV8 Yolanda -n 10 -j 4
```

Compile-check an agent:

```bash
.venv/bin/python -m py_compile 3600-agents/Yolanda/agent.py
.venv/bin/python -m py_compile 3600-agents/YolandaV8/agent.py
```

## Working Style Going Forward

- keep `Yolanda` as the main branch
- preserve known-good or interesting branches as separate agent folders
- make one clear change at a time
- show exact old-vs-new code before editing when tuning the main bot
- only promote an experiment branch after it beats the current baseline cleanly
