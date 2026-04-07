import re
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
AGENT_PATH = ROOT / "3600-agents" / "Yolanda" / "agent.py"
RESULTS_PATH = ROOT / "depth_sweep_results.txt"
PYTHON = ROOT / ".venv" / "bin" / "python"
DEPTHS = [2, 4]
GAMES_PER_SEAT = 10


def parse_summary(output: str, yolanda_is_player_a: bool) -> tuple[int, int, int]:
    player_a_wins = player_b_wins = ties = 0

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("PLAYER_A:"):
            player_a_wins = int(line.split(":", 1)[1].strip())
        elif line.startswith("PLAYER_B:"):
            player_b_wins = int(line.split(":", 1)[1].strip())
        elif line.startswith("TIE:"):
            ties = int(line.split(":", 1)[1].strip())

    if yolanda_is_player_a:
        return player_a_wins, player_b_wins, ties
    return player_b_wins, player_a_wins, ties


def run_matchup(player_a: str, player_b: str) -> str:
    result = subprocess.run(
        [str(PYTHON), "run_many.py", player_a, player_b, "-n", str(GAMES_PER_SEAT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def main() -> None:
    original = AGENT_PATH.read_text()
    pattern = re.compile(r"^MINIMAX_DEPTH = \d+", re.MULTILINE)
    if not pattern.search(original):
        raise RuntimeError("Could not find MINIMAX_DEPTH constant in agent.py")

    lines = [
        f"Depth sweep started: {datetime.now().isoformat()}",
        f"Games per seat: {GAMES_PER_SEAT}",
        "",
    ]

    try:
        for depth in DEPTHS:
            updated = pattern.sub(f"MINIMAX_DEPTH = {depth}", original, count=1)
            AGENT_PATH.write_text(updated)

            start = perf_counter()
            forward = run_matchup("Yolanda", "YolandaV3")
            reverse = run_matchup("YolandaV3", "Yolanda")
            elapsed = perf_counter() - start
            games_run = 2 * GAMES_PER_SEAT

            wins_f, losses_f, ties_f = parse_summary(forward, yolanda_is_player_a=True)
            wins_r, losses_r, ties_r = parse_summary(reverse, yolanda_is_player_a=False)

            total_wins = wins_f + wins_r
            total_losses = losses_f + losses_r
            total_ties = ties_f + ties_r

            lines.append(
                f"Depth {depth}: Yolanda {total_wins} / {2 * GAMES_PER_SEAT}, "
                f"YolandaV3 {total_losses} / {2 * GAMES_PER_SEAT}, ties {total_ties}"
            )
            lines.append(f"Elapsed: {elapsed:.2f}s total, {elapsed / games_run:.2f}s/game")
            lines.append("Forward summary:")
            lines.extend(forward.strip().splitlines()[-7:])
            lines.append("Reverse summary:")
            lines.extend(reverse.strip().splitlines()[-7:])
            lines.append("")

            RESULTS_PATH.write_text("\n".join(lines) + "\n")
    finally:
        AGENT_PATH.write_text(original)

    lines.append(f"Depth sweep finished: {datetime.now().isoformat()}")
    RESULTS_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
