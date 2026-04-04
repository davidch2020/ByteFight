# ByteFight

## Status

Last updated: 2026-04-04 12:45 EDT

Current goal: get a scrimmage-ready AI tournament bot by April 5.

## Current Bot Setup

- `3600-agents/Yolanda/agent.py`
  - Main working bot.
  - Current main line matches `YolandaV3`.
  - Uses:
    - rat belief tracking with transition + sensor updates
    - conservative search with a fixed probability threshold
    - 1-ply movement heuristic with opponent-aware mobility penalty
- `3600-agents/YolandaV1/agent.py`
  - Snapshot baseline of the earlier rule-based version.
  - Preferred non-`roll_length == 1` carpets, then prime, then fallback random.
- `3600-agents/YolandaV2/agent.py`
  - Strong movement-only heuristic baseline before rat-search logic.
- `3600-agents/YolandaV3/agent.py`
  - Frozen checkpoint of the stronger threshold-based rat-search version.
- `3600-agents/RandomAgent/agent.py`
  - Pure random-move baseline used for comparison.

## Progress So Far

- Set up a local Python 3.12 virtual environment to match the assignment environment.
- Installed project requirements into `.venv`.
- Added `.gitignore` entries for:
  - `3600-agents/matches/`
  - `.venv/`
  - `__pycache__/`
- Added `run_many.py` to batch local matches and print game summaries.
- Created `RandomAgent` as a baseline comparison bot.
- Created `YolandaV1` as a saved baseline before moving to the heuristic evaluator.
- Created `YolandaV2` and `YolandaV3` as stronger checkpoints during iterative tuning.
- Added rat belief tracking and a conservative search rule to `Yolanda`.
- Added a lightweight opponent-aware movement heuristic.

## Recent Test Results

- `Yolanda` strongly beats `RandomAgent` in both seat orders.
- `Yolanda` beats `YolandaV2` on larger batch samples.
- The EV-based search experiment underperformed and was reverted.
- Current conclusion:
  - `YolandaV3` behavior is the strongest local version so far.
  - Current `Yolanda` is set back to that stable checkpoint.

## Useful Commands

Activate environment:

```bash
source .venv/bin/activate
```

Run one local match:

```bash
python engine/run_local_agents.py Yolanda Yolanda
```

Run a batch of matches:

```bash
python run_many.py Yolanda RandomAgent -n 10
python run_many.py Yolanda YolandaV3 -n 10
python run_many.py Yolanda YolandaV2 -n 10
```

## Next Steps

- Test against George when available.
- Tune heuristics only when batch data shows a clear benefit.
- Keep `YolandaV3` as the fallback checkpoint while experimenting.
