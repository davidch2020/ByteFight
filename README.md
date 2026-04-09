# ByteFight

## Status

Last updated: 2026-04-08 EDT

Current goal: keep `YolandaV5` as the stable tournament fallback while using `Yolanda` as the active expectiminimax experiment branch.

## Current Bot Setup

- `3600-agents/Yolanda/agent.py`
  - Main experimental bot.
  - Uses:
    - rat belief tracking with transition + sensor updates
    - root search chosen by expected value plus a confidence floor
    - expectiminimax-style movement search with alpha-beta at decision nodes
    - bounded chance branching over likely rat outcomes
    - timing instrumentation for per-turn runtime checks
    - heuristic leaf evaluation over score margin, carpet potential, opponent potential, and mobility
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
  - Frozen checkpoint of the depth-5 minimax branch with alpha-beta, move ordering, `time_left` fallback, and the carpet-points heuristic.
  - Current strongest stable local baseline.
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
- Created `YolandaV5` as the frozen depth-5 minimax checkpoint before expectiminimax experimentation.
- Created `YolandaApurbo` as a partner sandbox copy of the current bot.
- Added rat belief tracking and a conservative search rule to `Yolanda`.
- Added a lightweight opponent-aware movement heuristic.
- Refactored `play()` into smaller helpers for belief update, search choice, and movement choice.
- Prototyped minimax movement search and moved the active branch to depth 5.
- Added alpha-beta pruning and lightweight move ordering to make deeper search more practical.
- Added `depth_sweep.py` for controlled minimax-depth experiments with elapsed-time reporting.
- Added a `time_left` depth selector so the active bot can reduce minimax depth when the remaining time budget gets lower.
- Updated the carpet heuristic to use the actual `CARPET_POINTS_TABLE` instead of raw `roll_length`.
- Tested a generalized opponent-potential leaf heuristic; early results are mixed, so it should be kept under watch before freezing another checkpoint.
- Prototyped an expectiminimax branch in `Yolanda` with chance nodes over likely rat outcomes and alpha-beta pruning at decision nodes.
- Reorganized `Yolanda/agent.py` into grouped sections and added plain-English block comments for partner readability.
- Added per-turn timing output and node-count reporting to help diagnose local search runtime.
- Updated `run_many.py` to decode subprocess output with replacement so batch tests no longer crash on occasional invalid bytes.
- Tweaked the local runner/process setup for more reliable local testing on this machine, including a spawn start method and more permissive initialization timing.

## Recent Test Results

- `Yolanda` strongly beats `RandomAgent` in both seat orders.
- `Yolanda` beats `YolandaV2` on larger batch samples.
- The EV-based search experiment underperformed and was reverted.
- Depth-4 minimax beat `YolandaV3` in a larger confirmation sample and was saved as `YolandaV4`.
- Alpha-beta pruning reduced depth-4 runtime in a timed comparison while keeping performance in the same range.
- Current depth-5 `Yolanda` with alpha-beta and move ordering looks competitive with `YolandaV4` and still clearly beats `YolandaV3` in local batches.
- After adding `time_left` and the carpet-points heuristic, `Yolanda` continued to beat `YolandaV4` in 20-game local samples around the same range as before.
- A generalized opponent-potential heuristic produced mixed 20-game samples against `YolandaV4`, including `11-9` and `13-7`; this is not clearly better yet.
- Recent heuristic tweaks were mixed; small coefficient changes still swing results noticeably.
- Against the reference bots as Team A:
  - `George`: `7-2-1`
  - `Albert`: `3-7`
- Match review against `Albert` suggests the biggest gap is future carpet conversion and search quality, not opening play.
- The current expectiminimax branch underperformed badly at first, but tightening root search and fixing local depth handling made it much more competitive with `YolandaV5` in the first 5-game seat-order rerun (`3-1-1` as Player A).
- Current conclusion:
  - `YolandaV5` is the strongest stable fallback checkpoint.
  - Current `Yolanda` is the active expectiminimax lab branch and still needs more tuning before replacing `YolandaV5`.

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
python run_many.py YolandaApurbo Yolanda -n 10
python run_many.py Yolanda YolandaApurbo -n 10
```

## Next Steps

- Finish reverse-seat validation of the tightened expectiminimax branch against `YolandaV5`.
- Reduce unnecessary interior search branching or make branch-level search more selective.
- Improve future carpet-potential features before pushing expectiminimax depth higher.
- Compare the current branch against Albert again after the next search/heuristic pass.
- Compare `YolandaApurbo` against `Yolanda` before merging any partner changes.
- Keep `YolandaV5` as the stable fallback while `Yolanda` stays experimental.
