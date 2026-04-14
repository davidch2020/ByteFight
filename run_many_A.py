import argparse
import io
import os
import pathlib
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout

# Make sure the engine directory is on the path
ENGINE_DIR = pathlib.Path(__file__).parent / "engine"
sys.path.insert(0, str(ENGINE_DIR))

from board_utils import get_history_json
from gameplay import play_game

TOP_LEVEL = pathlib.Path(__file__).parent.resolve()
PLAY_DIRECTORY = str(TOP_LEVEL / "3600-agents")

WINNER_MAP = {0: "PLAYER_A", 1: "PLAYER_B", 2: "TIE"}
REASON_MAP = {0: "POINTS", 1: "TIMEOUT", 2: "INVALID_TURN", 3: "CODE_CRASH", 4: "MEMORY_ERROR", 5: "FAILED_INIT"}


def run_game(args: tuple) -> dict:
    player_a, player_b, game_num = args

    try:
        t0 = time.perf_counter()

        captured = io.StringIO()
        with redirect_stdout(captured):
            final_board, rat_position_history, spawn_a, spawn_b, message_a, message_b = play_game(
                PLAY_DIRECTORY,
                PLAY_DIRECTORY,
                player_a,
                player_b,
                display_game=False,
                delay=0.0,
                clear_screen=False,
                record=True,
                limit_resources=False,
            )

        elapsed = time.perf_counter() - t0

        # Save JSON with game_num in filename — no race condition
        records_dir = os.path.join(PLAY_DIRECTORY, "matches")
        os.makedirs(records_dir, exist_ok=True)
        out_path = os.path.join(records_dir, f"{player_a}_{player_b}_{game_num}.json")
        with open(out_path, "w") as fp:
            fp.write(get_history_json(
                final_board, rat_position_history,
                spawn_a, spawn_b,
                message_a, message_b,
            ))

        winner = WINNER_MAP.get(final_board.winner, f"UNKNOWN({final_board.winner})")
        reason = REASON_MAP.get(final_board.win_reason, f"UNKNOWN({final_board.win_reason})")

        return {
            "game_num": game_num,
            "ok": True,
            "winner": winner,
            "reason": reason,
            "elapsed": elapsed,
        }

    except Exception as e:
        return {
            "game_num": game_num,
            "ok": False,
            "error": str(e),
            "elapsed": 0.0,
        }


def _process_result(game: dict, winner_counts: Counter, reason_counts: Counter, results: dict) -> None:
    game_num = game["game_num"]
    if not game["ok"]:
        results[game_num] = f"Game {game_num:>3}: FAILED — {game.get('error', 'unknown error')}"
    else:
        winner_counts[game["winner"]] += 1
        reason_counts[game["reason"]] += 1
        results[game_num] = (
            f"Game {game_num:>3}: winner={game['winner']:<10} "
            f"reason={game['reason']:<15} "
            f"({game['elapsed']:.1f}s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple bot matches in parallel.")
    parser.add_argument("player_a", help="Folder name of player A under 3600-agents/")
    parser.add_argument("player_b", help="Folder name of player B under 3600-agents/")
    parser.add_argument("-n", "--games", type=int, default=10, help="Number of games (default: 10)")
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=min(os.cpu_count() or 4, 16),
        help="Parallel workers (default: CPU count)",
    )
    args = parser.parse_args()

    print(f"Running {args.games} games | {args.workers} workers | display=OFF\n")

    winner_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    failed_games = 0
    results: dict[int, str] = {}

    game_args = [(args.player_a, args.player_b, i) for i in range(1, args.games + 1)]

    wall_start = time.perf_counter()

    if args.games <= 2:
        for g in game_args:
            game = run_game(g)
            _process_result(game, winner_counts, reason_counts, results)
            if not game["ok"]:
                failed_games += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_game, g): g[2] for g in game_args}
            for future in as_completed(futures):
                game = future.result()
                _process_result(game, winner_counts, reason_counts, results)
                if not game["ok"]:
                    failed_games += 1

    wall_elapsed = time.perf_counter() - wall_start

    for game_num in sorted(results):
        print(results[game_num])

    played = args.games - failed_games

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"Player A  : {args.player_a}")
    print(f"Player B  : {args.player_b}")
    print(f"Games     : {args.games}  (played: {played}, failed: {failed_games})")
    print(f"Wall time : {wall_elapsed:.1f}s")

    if played:
        print("\nWin counts:")
        for winner, count in sorted(winner_counts.items()):
            pct = count / played * 100
            print(f"  {winner:<12}: {count:>3}  ({pct:.1f}%)")

        print("\nWin reasons:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason:<20}: {count}")


if __name__ == "__main__":
    main()