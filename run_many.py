import argparse
import os
import pathlib
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
WINNER_RE = re.compile(r"(PLAYER_A|PLAYER_B|TIE) wins by (\w+)")
SCORE_RE = re.compile(r"POINTS A:([-\d]+)\s+B:([-\d]+)")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def matches_dir() -> pathlib.Path:
    top_level = pathlib.Path(__file__).resolve().parent
    return top_level / "3600-agents" / "matches"


def report_path(player_a: str, player_b: str, game_num: int) -> pathlib.Path:
    return matches_dir() / f"{player_a}_{player_b}_{game_num:03d}.json"


def run_game(player_a: str, player_b: str, game_num: int) -> dict:
    out_path = report_path(player_a, player_b, game_num)
    env = os.environ.copy()
    env["MATCH_OUTPUT_PATH"] = str(out_path)

    result = subprocess.run(
        [sys.executable, "engine/run_local_agents.py", player_a, player_b],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    output = strip_ansi(result.stdout)

    winner_match = WINNER_RE.search(output)
    score_matches = SCORE_RE.findall(output)

    if winner_match:
        winner = winner_match.group(1)
        reason = winner_match.group(2)
    else:
        winner = "UNKNOWN"
        reason = "UNKNOWN"

    if score_matches:
        score_a, score_b = score_matches[-1]
    else:
        score_a, score_b = "?", "?"

    return {
        "game_num": game_num,
        "returncode": result.returncode,
        "winner": winner,
        "reason": reason,
        "score_a": score_a,
        "score_b": score_b,
        "stdout": output,
        "stderr": result.stderr,
        "report_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple local bot matches.")
    parser.add_argument("player_a", help="Folder name of player A under 3600-agents/")
    parser.add_argument("player_b", help="Folder name of player B under 3600-agents/")
    parser.add_argument(
        "-n",
        "--games",
        type=int,
        default=10,
        help="Number of games to run (default: 10)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel workers to use (default: min(4, cpu_count))",
    )
    args = parser.parse_args()

    out_dir = matches_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    winner_counts = Counter()
    reason_counts = Counter()
    failed_games = 0
    jobs = max(1, min(args.jobs, args.games))

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(run_game, args.player_a, args.player_b, game_num)
            for game_num in range(1, args.games + 1)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    for game in sorted(results, key=lambda item: item["game_num"]):
        if game["returncode"] != 0:
            failed_games += 1
            print(f"Game {game['game_num']}: FAILED (exit code {game['returncode']})")
            if game["stderr"].strip():
                print(game["stderr"].strip())
            continue

        winner_counts[game["winner"]] += 1
        reason_counts[game["reason"]] += 1
        print(
            f"Game {game['game_num']}: winner={game['winner']}, "
            f"reason={game['reason']}, score A={game['score_a']}, B={game['score_b']}, "
            f"report={game['report_path']}"
        )

    print("\nSummary")
    print(f"Player A: {args.player_a}")
    print(f"Player B: {args.player_b}")
    print(f"Games run: {args.games}")
    print(f"Parallel jobs: {jobs}")
    print(f"Failures: {failed_games}")
    print(f"Reports saved in: {out_dir}")

    for winner, count in sorted(winner_counts.items()):
        print(f"{winner}: {count}")

    for reason, count in sorted(reason_counts.items()):
        print(f"{reason}: {count}")


if __name__ == "__main__":
    main()
