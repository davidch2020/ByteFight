# ByteFight

## Status

Last updated: 2026-04-06 EDT

Current goal: keep `Yolanda` as the active tournament bot while preserving stable checkpoints for safe rollback and teammate experiments.

## Current Bot Setup

- `3600-agents/Yolanda/agent.py`
  - Main working bot.
  - Uses:
    - rat belief tracking with transition + sensor updates
    - conservative search with a fixed probability threshold
    - depth-5 minimax for movement selection
    - alpha-beta pruning with lightweight move ordering
    - heuristic leaf evaluation over score gain, carpet potential, and mobility
- `3600-agents/YolandaV1/agent.py`
  - Snapshot baseline of the earlier rule-based version.
  - Preferred non-`roll_length == 1` carpets, then prime, then fallback random.
- `3600-agents/YolandaV2/agent.py`
  - Strong movement-only heuristic baseline before rat-search logic.
- `3600-agents/YolandaV3/agent.py`
  - Frozen checkpoint of the stronger threshold-based rat-search version.
- `3600-agents/YolandaV4/agent.py`
  - Frozen checkpoint of the depth-4 minimax version.
  - Useful fallback while testing deeper search and ordering changes.
- `3600-agents/YolandaApurbo/agent.py`
  - Partner experiment copy of current `Yolanda`.
  - Use this for Apurbo's independent edits without changing the main bot.
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
- Created `YolandaV4` as the depth-4 minimax checkpoint.
- Created `YolandaApurbo` as a partner sandbox copy of the current bot.
- Added rat belief tracking and a conservative search rule to `Yolanda`.
- Added a lightweight opponent-aware movement heuristic.
- Refactored `play()` into smaller helpers for belief update, search choice, and movement choice.
- Prototyped minimax movement search and moved the active branch to depth 5.
- Added alpha-beta pruning and lightweight move ordering to make deeper search more practical.
- Added `depth_sweep.py` for controlled minimax-depth experiments with elapsed-time reporting.

## Recent Test Results

- `Yolanda` strongly beats `RandomAgent` in both seat orders.
- `Yolanda` beats `YolandaV2` on larger batch samples.
- The EV-based search experiment underperformed and was reverted.
- Depth-4 minimax beat `YolandaV3` in a larger confirmation sample and was saved as `YolandaV4`.
- Alpha-beta pruning reduced depth-4 runtime in a timed comparison while keeping performance in the same range.
- Current depth-5 `Yolanda` with alpha-beta and move ordering looks competitive with `YolandaV4` and still clearly beats `YolandaV3` in local batches.
- Recent heuristic tweaks were mixed; small coefficient changes still swing results noticeably.
- Current conclusion:
  - `YolandaV3` remains the strongest simple threshold-search checkpoint.
  - `YolandaV4` is the stable depth-4 minimax checkpoint.
  - Current `Yolanda` is the active depth-5 alpha-beta branch.

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
python run_many.py Yolanda YolandaV4 -n 10
python run_many.py YolandaApurbo Yolanda -n 10
python run_many.py Yolanda YolandaApurbo -n 10
```

## Next Steps

- Test against George when available.
- Compare the depth-5 branch against stronger reference bots.
- Compare `YolandaApurbo` against `Yolanda` before merging any partner changes.
- Add `time_left` safety if deeper search starts getting too expensive.
- Tune leaf heuristics only when batch data shows a clear benefit.
- Keep `YolandaV4` as the minimax fallback checkpoint while experimenting.
