import argparse
import re
import subprocess
import sys
from collections import Counter


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
WINNER_RE = re.compile(r"(PLAYER_A|PLAYER_B|TIE) wins by (\w+)")
SCORE_RE = re.compile(r"POINTS A:([-\d]+)\s+B:([-\d]+)")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def run_game(player_a: str, player_b: str) -> dict:
    result = subprocess.run(
        [sys.executable, "engine/run_local_agents.py", player_a, player_b],
        capture_output=True,
        text=True,
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
        "returncode": result.returncode,
        "winner": winner,
        "reason": reason,
        "score_a": score_a,
        "score_b": score_b,
        "stdout": output,
        "stderr": result.stderr,
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
    args = parser.parse_args()

    winner_counts = Counter()
    reason_counts = Counter()
    failed_games = 0

    for game_num in range(1, args.games + 1):
        game = run_game(args.player_a, args.player_b)

        if game["returncode"] != 0:
            failed_games += 1
            print(f"Game {game_num}: FAILED (exit code {game['returncode']})")
            if game["stderr"].strip():
                print(game["stderr"].strip())
            continue

        winner_counts[game["winner"]] += 1
        reason_counts[game["reason"]] += 1
        print(
            f"Game {game_num}: winner={game['winner']}, "
            f"reason={game['reason']}, score A={game['score_a']}, B={game['score_b']}"
        )

    print("\nSummary")
    print(f"Player A: {args.player_a}")
    print(f"Player B: {args.player_b}")
    print(f"Games run: {args.games}")
    print(f"Failures: {failed_games}")

    for winner, count in sorted(winner_counts.items()):
        print(f"{winner}: {count}")

    for reason, count in sorted(reason_counts.items()):
        print(f"{reason}: {count}")


if __name__ == "__main__":
    main()
