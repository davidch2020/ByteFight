# ByteFight

## Status

Last updated: 2026-04-09 EDT

Current goal: keep `Yolanda` as the active minimax bot, tuned around future carpet potential and local time-budget handling, while preserving the major experiments as snapshots.

## Current Bot Setup

- `3600-agents/Yolanda/agent.py`
  - Main active bot.
  - Uses:
    - rat belief tracking with transition + sensor updates
    - threshold-based root search
    - depth-5 minimax with alpha-beta and move ordering
    - adaptive local time-budget detection for search depth
    - heuristic evaluation built around score margin, carpet quality, future carpet setup, mobility, opponent threat, and a small dynamic position/value-board term
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
- `3600-agents/YolandaV5/agent.py`
  - Frozen checkpoint of the depth-5 minimax branch before the latest heuristic experiments.
  - Still the main local comparison baseline.
- `3600-agents/YolandaV6/agent.py`
  - Frozen expectiminimax / belief-aware experiment snapshot.
  - Kept for reference, not the current main path.
- `3600-agents/YolandaPosOnly/agent.py`
  - Position/value-board experiment branch.
  - Used to isolate how much dynamic board-value helps on its own or in reduced heuristics.
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
- Created `YolandaV5` as the frozen depth-5 minimax checkpoint before the latest heuristic experiments.
- Created `YolandaV6` to preserve the expectiminimax / belief-aware branch.
- Created `YolandaPosOnly` to isolate dynamic position/value-board heuristics.
- Created `YolandaApurbo` as a partner sandbox copy of the current bot.
- Added rat belief tracking and a conservative search rule to `Yolanda`.
- Added a lightweight opponent-aware movement heuristic.
- Refactored `play()` into smaller helpers for belief update, search choice, and movement choice.
- Prototyped minimax movement search and moved the active branch to depth 5.
- Added alpha-beta pruning and lightweight move ordering to make deeper search more practical.
- Added `depth_sweep.py` for controlled minimax-depth experiments with elapsed-time reporting.
- Reworked `time_left` handling so the active bot detects the actual local time budget instead of assuming 240 seconds.
- Updated the carpet heuristic to use the actual `CARPET_POINTS_TABLE` instead of raw `roll_length`.
- Added helper heuristics for:
  - best carpet available now
  - best carpet available after one setup move
  - dynamic board-value / position scoring
- Tested reduced and isolated heuristic variants to measure the value of:
  - future carpet setup
  - opponent future-carpet penalties
  - dynamic board-value / position
- Updated `run_many.py` to decode subprocess output with replacement so batch tests no longer crash on occasional invalid bytes.
- Cleaned local repo noise by ignoring `bot_matches/` and removing `.DS_Store` / `__pycache__` clutter from version control.

## Recent Test Results

- `Yolanda` strongly beats `RandomAgent` in both seat orders.
- `Yolanda` beats `YolandaV2` on larger batch samples.
- The EV-based root-search experiment underperformed and was reverted.
- Depth-4 minimax beat `YolandaV3` in a larger confirmation sample and was saved as `YolandaV4`.
- Alpha-beta pruning reduced depth-4 runtime in a timed comparison while keeping performance in the same range.
- The large expectiminimax branch was educational but not competitive enough to replace the minimax bot, so it was frozen as `YolandaV6`.
- Fixing the local time-budget handling changed the `Yolanda` vs `YolandaV5` comparison substantially; fair local comparisons now use that fix in both bots.
- Future carpet setup is a real signal: increasing the weight on one-turn-ahead carpet potential produced large gains against the unfixed `YolandaV5`, but a fair rematch showed that time-budget handling had been a major confounder.
- Dynamic board-value / position has some initiative value, but by itself it is not strong enough to replace the main carpet heuristic.
- `YolandaPosOnly` showed:
  - strong first-player initiative in some samples
  - weak second-player defense
  - worse performance once opponent future-carpet penalties were added too aggressively
- Against the reference bots as Team A:
  - `George`: `7-2-1`
  - `Albert`: `3-7`
- Match review against `Albert` suggests the biggest gap is future carpet conversion and search quality, not opening play.
- Current conclusion:
  - `Yolanda` is back on the minimax path and is the main bot to keep tuning.
  - `YolandaV5` remains the clean stable baseline for comparison.
  - `YolandaV6` and `YolandaPosOnly` are useful snapshots, not the current primary direction.

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
python run_many.py Yolanda YolandaV5 -n 10
python run_many.py YolandaV5 Yolanda -n 10
python run_many.py YolandaPosOnly YolandaV5 -n 10
python run_many.py YolandaApurbo Yolanda -n 10
python run_many.py Yolanda YolandaApurbo -n 10
```

## Next Steps

- Simplify the active heuristic so it focuses on the signals that have actually helped:
  - score margin
  - immediate carpet strength
  - future carpet setup
  - light mobility / position support
- Keep testing changes against `YolandaV5` in both seat orders before drawing conclusions.
- Compare the best current `Yolanda` build against Albert again after the next heuristic pass.
- Compare `YolandaApurbo` against `Yolanda` before merging any partner changes.
- Keep `YolandaV5` as the stable fallback and `YolandaV6` / `YolandaPosOnly` as archived experiments.
