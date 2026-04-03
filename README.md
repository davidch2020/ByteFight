# ByteFight

## Status

Last updated: 2026-04-03 19:02 EDT

Current goal: get a scrimmage-ready AI tournament bot by April 5.

## Current Bot Setup

- `3600-agents/Yolanda/agent.py`
  - Main working bot.
  - Uses a simple 1-ply heuristic:
    - forecast each legal move
    - score immediate point gain
    - reward future carpet opportunities
    - reward future mobility through prime/plain follow-up moves
    - penalize weak future `CARPET` moves with `roll_length == 1`
- `3600-agents/YolandaV1/agent.py`
  - Snapshot baseline of the earlier rule-based version.
  - Preferred non-`roll_length == 1` carpets, then prime, then fallback random.
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
- Replaced the original random `Yolanda` policy with a heuristic move evaluator.

## Recent Test Results

- `Yolanda` strongly beats `RandomAgent` in both seat orders.
- Updated heuristic `Yolanda` also beats `YolandaV1` decisively in batch tests.
- Current conclusion:
  - `Yolanda` is now clearly stronger than the previous versions tested so far.
  - The current movement heuristic is a solid scrimmage-ready foundation.

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
python run_many.py Yolanda YolandaV1 -n 10
```

## Next Steps

- Continue tuning the heuristic evaluator only if tests show a clear benefit.
- Start implementing rat belief tracking.
- After rat belief is stable, decide when search moves become worthwhile.
