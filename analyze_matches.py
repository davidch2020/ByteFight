#!/usr/bin/env python3
"""Analyze bot match data from player_a and player_b folders."""

import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
PLAYER_A_DIR = os.path.join(BASE, "bot_matches", "player_a")
PLAYER_B_DIR = os.path.join(BASE, "bot_matches", "player_b")


def load_matches(folder):
    matches = []
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            with open(os.path.join(folder, fname)) as f:
                m = json.load(f)
                m["_filename"] = fname
                matches.append(m)
    # Sort by filename for consistent ordering
    matches.sort(key=lambda m: m["_filename"])
    return matches


def analyze_match(m, we_are):
    """Analyze a single match. we_are is 'a' or 'b'."""
    opp = "b" if we_are == "a" else "a"
    steps = len(m["a_points"])

    our_points = m[f"{we_are}_points"]
    opp_points = m[f"{opp}_points"]
    our_final = our_points[-1]
    opp_final = opp_points[-1]
    diff = our_final - opp_final

    # Determine outcome
    if we_are == "a":
        we_won = m["result"] == 0
    else:
        we_won = m["result"] == 1
    tie = our_final == opp_final
    if tie:
        outcome = "tie"
    elif we_won:
        outcome = "win"
    else:
        outcome = "loss"

    # Turn count
    turn_count = m.get("turn_count", steps)

    # Points per turn
    our_ppt = our_final / max(turn_count, 1)
    opp_ppt = opp_final / max(turn_count, 1)

    # Time usage
    our_time = m[f"{we_are}_time_left"]
    opp_time = m[f"{opp}_time_left"]
    our_time_used = our_time[0] - our_time[-1]
    opp_time_used = opp_time[0] - opp_time[-1]

    # Per-turn time usage (only on our turns)
    our_turn_times = []
    opp_turn_times = []
    for i in range(1, len(our_time)):
        dt = our_time[i - 1] - our_time[i]
        if dt > 0.001:  # this was our turn
            our_turn_times.append(dt)
        dt2 = opp_time[i - 1] - opp_time[i]
        if dt2 > 0.001:
            opp_turn_times.append(dt2)

    our_avg_turn_time = (
        sum(our_turn_times) / len(our_turn_times) if our_turn_times else 0
    )
    opp_avg_turn_time = (
        sum(opp_turn_times) / len(opp_turn_times) if opp_turn_times else 0
    )
    our_max_turn_time = max(our_turn_times) if our_turn_times else 0
    opp_max_turn_time = max(opp_turn_times) if opp_turn_times else 0

    # Rat catches per player
    # Rat is caught when rat_caught[i] is True. Need to figure out WHO caught it.
    # The step index alternates: even steps = player A's turn result, odd = player B's
    # Actually the arrays record state after each half-turn. Index 0 is initial,
    # index 1 is after A's first move, index 2 is after B's first move, etc.
    our_rat_catches = 0
    opp_rat_catches = 0
    for i in range(1, len(m["rat_caught"])):
        if m["rat_caught"][i] and not m["rat_caught"][i - 1]:
            # A catch happened at step i
            # Odd indices = player A just moved, even indices = player B just moved
            if we_are == "a":
                if i % 2 == 1:
                    our_rat_catches += 1
                else:
                    opp_rat_catches += 1
            else:
                if i % 2 == 0:
                    our_rat_catches += 1
                else:
                    opp_rat_catches += 1

    # Actually, rat_caught might stay True for a while. Let's also check via points jumps.
    # Let's count rat catches differently: when rat_caught transitions to True
    total_rat_catches = 0
    for i in range(1, len(m["rat_caught"])):
        if m["rat_caught"][i] and (i == 0 or not m["rat_caught"][i - 1]):
            total_rat_catches += 1

    # Cell types left behind
    left = m["left_behind"]
    our_cells = defaultdict(int)
    opp_cells = defaultdict(int)
    # Step 0 is initial state. Step 1 = after A moves, step 2 = after B moves...
    for i in range(1, len(left)):
        cell_type = left[i]
        if we_are == "a":
            if i % 2 == 1:
                our_cells[cell_type] += 1
            else:
                opp_cells[cell_type] += 1
        else:
            if i % 2 == 0:
                our_cells[cell_type] += 1
            else:
                opp_cells[cell_type] += 1

    # Carpet placements
    total_carpets = sum(1 for c in m["new_carpets"] if c)
    our_carpets = 0
    opp_carpets = 0
    carpet_timing = []  # (step_fraction, who)
    for i in range(1, len(m["new_carpets"])):
        if m["new_carpets"][i]:
            frac = i / len(m["new_carpets"])
            if we_are == "a":
                if i % 2 == 1:
                    our_carpets += 1
                    carpet_timing.append((frac, "us"))
                else:
                    opp_carpets += 1
                    carpet_timing.append((frac, "opp"))
            else:
                if i % 2 == 0:
                    our_carpets += 1
                    carpet_timing.append((frac, "us"))
                else:
                    opp_carpets += 1
                    carpet_timing.append((frac, "opp"))

    # Point swings: compute score differential over time, find big changes
    diffs_over_time = [
        our_points[i] - opp_points[i] for i in range(len(our_points))
    ]
    swings = []
    for i in range(1, len(diffs_over_time)):
        change = diffs_over_time[i] - diffs_over_time[i - 1]
        if abs(change) >= 3:
            swings.append(
                {
                    "step": i,
                    "frac": i / len(diffs_over_time),
                    "change": change,
                    "new_diff": diffs_over_time[i],
                }
            )

    # Early vs late game scoring (first half vs second half)
    mid = len(our_points) // 2
    our_early = our_points[mid] - our_points[0]
    our_late = our_points[-1] - our_points[mid]
    opp_early = opp_points[mid] - opp_points[0]
    opp_late = opp_points[-1] - opp_points[mid]

    # Score trajectory at quartiles
    q1 = len(our_points) // 4
    q2 = len(our_points) // 2
    q3 = 3 * len(our_points) // 4
    our_q_scores = [our_points[q1], our_points[q2], our_points[q3], our_points[-1]]
    opp_q_scores = [opp_points[q1], opp_points[q2], opp_points[q3], opp_points[-1]]

    return {
        "filename": m["_filename"],
        "outcome": outcome,
        "our_final": our_final,
        "opp_final": opp_final,
        "diff": diff,
        "turn_count": turn_count,
        "our_ppt": our_ppt,
        "opp_ppt": opp_ppt,
        "our_time_used": our_time_used,
        "opp_time_used": opp_time_used,
        "our_avg_turn_time": our_avg_turn_time,
        "opp_avg_turn_time": opp_avg_turn_time,
        "our_max_turn_time": our_max_turn_time,
        "opp_max_turn_time": opp_max_turn_time,
        "our_time_remaining": our_time[-1],
        "opp_time_remaining": opp_time[-1],
        "our_rat_catches": our_rat_catches,
        "opp_rat_catches": opp_rat_catches,
        "total_rat_catches": total_rat_catches,
        "our_cells": dict(our_cells),
        "opp_cells": dict(opp_cells),
        "our_carpets": our_carpets,
        "opp_carpets": opp_carpets,
        "total_carpets": total_carpets,
        "carpet_timing": carpet_timing,
        "swings": swings,
        "our_early": our_early,
        "our_late": our_late,
        "opp_early": opp_early,
        "opp_late": opp_late,
        "our_q_scores": our_q_scores,
        "opp_q_scores": opp_q_scores,
        "reason": m.get("reason", "?"),
        "we_are": we_are,
        "max_diff": max(diffs_over_time),
        "min_diff": min(diffs_over_time),
        "final_diff_trajectory": diffs_over_time[-1],
    }


def avg(lst):
    return sum(lst) / len(lst) if lst else 0


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_subsection(title):
    print(f"\n--- {title} ---")


def main():
    # Load matches
    a_matches = load_matches(PLAYER_A_DIR)
    b_matches = load_matches(PLAYER_B_DIR)
    print(f"Loaded {len(a_matches)} matches from player_a, {len(b_matches)} from player_b")

    # Analyze all
    results_a = [analyze_match(m, "a") for m in a_matches]
    results_b = [analyze_match(m, "b") for m in b_matches]
    all_results = results_a + results_b

    # ========== SECTION 1: Win/Loss Record ==========
    print_section("WIN/LOSS/TIE RECORD")

    for label, results in [
        ("Player A (16 matches)", results_a),
        ("Player B (10 matches)", results_b),
        ("OVERALL (26 matches)", all_results),
    ]:
        wins = sum(1 for r in results if r["outcome"] == "win")
        losses = sum(1 for r in results if r["outcome"] == "loss")
        ties = sum(1 for r in results if r["outcome"] == "tie")
        print(f"\n  {label}:")
        print(f"    Wins: {wins}  Losses: {losses}  Ties: {ties}  Win%: {wins/len(results)*100:.1f}%")

    # ========== SECTION 2: Per-Match Summary ==========
    print_section("PER-MATCH SUMMARY")

    for label, results in [("AS PLAYER A", results_a), ("AS PLAYER B", results_b)]:
        print_subsection(label)
        print(
            f"  {'File':<20} {'Result':>6} {'Our':>4} {'Opp':>4} {'Diff':>5} "
            f"{'Our PPT':>8} {'Opp PPT':>8} {'Reason':>8} {'Turns':>5}"
        )
        print("  " + "-" * 80)
        for r in results:
            tag = r["outcome"].upper()
            print(
                f"  {r['filename']:<20} {tag:>6} {r['our_final']:>4} {r['opp_final']:>4} "
                f"{r['diff']:>+5} {r['our_ppt']:>8.2f} {r['opp_ppt']:>8.2f} "
                f"{r['reason']:>8} {r['turn_count']:>5}"
            )

    # ========== SECTION 3: Score Differentials ==========
    print_section("SCORE DIFFERENTIALS")
    diffs = [r["diff"] for r in all_results]
    print(f"  Average differential: {avg(diffs):+.1f}")
    print(f"  Median differential:  {sorted(diffs)[len(diffs)//2]:+d}")
    print(f"  Best result:          {max(diffs):+d}")
    print(f"  Worst result:         {min(diffs):+d}")

    win_diffs = [r["diff"] for r in all_results if r["outcome"] == "win"]
    loss_diffs = [r["diff"] for r in all_results if r["outcome"] == "loss"]
    if win_diffs:
        print(f"  Avg win margin:       {avg(win_diffs):+.1f}")
    if loss_diffs:
        print(f"  Avg loss margin:      {avg(loss_diffs):+.1f}")

    # ========== SECTION 4: Time Management ==========
    print_section("TIME MANAGEMENT")

    for label, results in [
        ("Overall", all_results),
        ("Wins", [r for r in all_results if r["outcome"] == "win"]),
        ("Losses", [r for r in all_results if r["outcome"] == "loss"]),
    ]:
        if not results:
            continue
        print_subsection(f"{label} ({len(results)} matches)")
        print(f"  Our avg time used:      {avg([r['our_time_used'] for r in results]):.1f}s")
        print(f"  Opp avg time used:      {avg([r['opp_time_used'] for r in results]):.1f}s")
        print(f"  Our avg turn time:      {avg([r['our_avg_turn_time'] for r in results]):.3f}s")
        print(f"  Opp avg turn time:      {avg([r['opp_avg_turn_time'] for r in results]):.3f}s")
        print(f"  Our avg max turn time:  {avg([r['our_max_turn_time'] for r in results]):.3f}s")
        print(f"  Opp avg max turn time:  {avg([r['opp_max_turn_time'] for r in results]):.3f}s")
        print(f"  Our avg time remaining: {avg([r['our_time_remaining'] for r in results]):.1f}s")
        print(f"  Opp avg time remaining: {avg([r['opp_time_remaining'] for r in results]):.1f}s")

    # ========== SECTION 5: Rat Catching ==========
    print_section("RAT CATCHING")

    for label, results in [
        ("Overall", all_results),
        ("Wins", [r for r in all_results if r["outcome"] == "win"]),
        ("Losses", [r for r in all_results if r["outcome"] == "loss"]),
    ]:
        if not results:
            continue
        print_subsection(f"{label} ({len(results)} matches)")
        print(f"  Our avg rat catches:   {avg([r['our_rat_catches'] for r in results]):.2f}")
        print(f"  Opp avg rat catches:   {avg([r['opp_rat_catches'] for r in results]):.2f}")
        print(f"  Total rat catches/game:{avg([r['total_rat_catches'] for r in results]):.2f}")

    # ========== SECTION 6: Cell Types Left Behind ==========
    print_section("CELL TYPES LEFT BEHIND (average per match)")

    cell_types = ["plain", "prime", "carpet", "search"]
    for label, results in [
        ("Overall", all_results),
        ("Wins", [r for r in all_results if r["outcome"] == "win"]),
        ("Losses", [r for r in all_results if r["outcome"] == "loss"]),
    ]:
        if not results:
            continue
        print_subsection(f"{label} ({len(results)} matches)")
        print(f"  {'Type':<10} {'Our Avg':>8} {'Opp Avg':>8} {'Our Total':>10} {'Opp Total':>10}")
        for ct in cell_types:
            our_vals = [r["our_cells"].get(ct, 0) for r in results]
            opp_vals = [r["opp_cells"].get(ct, 0) for r in results]
            print(
                f"  {ct:<10} {avg(our_vals):>8.1f} {avg(opp_vals):>8.1f} "
                f"{sum(our_vals):>10} {sum(opp_vals):>10}"
            )

    # ========== SECTION 7: Carpet Placements ==========
    print_section("CARPET PLACEMENTS")

    for label, results in [
        ("Overall", all_results),
        ("Wins", [r for r in all_results if r["outcome"] == "win"]),
        ("Losses", [r for r in all_results if r["outcome"] == "loss"]),
    ]:
        if not results:
            continue
        print_subsection(f"{label} ({len(results)} matches)")
        print(f"  Our avg carpets placed:  {avg([r['our_carpets'] for r in results]):.2f}")
        print(f"  Opp avg carpets placed:  {avg([r['opp_carpets'] for r in results]):.2f}")
        print(f"  Total carpets per game:  {avg([r['total_carpets'] for r in results]):.2f}")

    # Carpet timing analysis
    print_subsection("Carpet Placement Timing (game fraction)")
    our_carpet_fracs = []
    opp_carpet_fracs = []
    for r in all_results:
        for frac, who in r["carpet_timing"]:
            if who == "us":
                our_carpet_fracs.append(frac)
            else:
                opp_carpet_fracs.append(frac)
    if our_carpet_fracs:
        print(f"  Our avg carpet timing:   {avg(our_carpet_fracs):.2f} (0=start, 1=end)")
        print(f"  Our earliest carpet:     {min(our_carpet_fracs):.2f}")
        print(f"  Our latest carpet:       {max(our_carpet_fracs):.2f}")
    if opp_carpet_fracs:
        print(f"  Opp avg carpet timing:   {avg(opp_carpet_fracs):.2f}")

    # ========== SECTION 8: Point Swings ==========
    print_section("SIGNIFICANT POINT SWINGS (diff change >= 3)")

    for r in all_results:
        if r["swings"]:
            tag = r["outcome"].upper()
            role = f"as {r['we_are'].upper()}"
            print(f"\n  {r['filename']} ({tag}, {role}):")
            for s in r["swings"]:
                direction = "FOR US" if s["change"] > 0 else "AGAINST US"
                game_phase = (
                    "early" if s["frac"] < 0.33 else "mid" if s["frac"] < 0.66 else "late"
                )
                print(
                    f"    Step {s['step']:>3} ({game_phase:>5}, {s['frac']:.0%}): "
                    f"diff changed by {s['change']:+d} -> now {s['new_diff']:+d} ({direction})"
                )

    # ========== SECTION 9: Early vs Late Game ==========
    print_section("EARLY vs LATE GAME SCORING (first half vs second half)")

    for label, results in [
        ("Overall", all_results),
        ("Wins", [r for r in all_results if r["outcome"] == "win"]),
        ("Losses", [r for r in all_results if r["outcome"] == "loss"]),
    ]:
        if not results:
            continue
        print_subsection(f"{label} ({len(results)} matches)")
        print(f"  Our avg early-game pts: {avg([r['our_early'] for r in results]):.1f}")
        print(f"  Our avg late-game pts:  {avg([r['our_late'] for r in results]):.1f}")
        print(f"  Opp avg early-game pts: {avg([r['opp_early'] for r in results]):.1f}")
        print(f"  Opp avg late-game pts:  {avg([r['opp_late'] for r in results]):.1f}")
        our_early_adv = avg([r["our_early"] - r["opp_early"] for r in results])
        our_late_adv = avg([r["our_late"] - r["opp_late"] for r in results])
        print(f"  Our early-game advantage: {our_early_adv:+.1f}")
        print(f"  Our late-game advantage:  {our_late_adv:+.1f}")

    # Quartile analysis
    print_subsection("Score Trajectory (quartile scores)")
    for label, results in [
        ("Wins", [r for r in all_results if r["outcome"] == "win"]),
        ("Losses", [r for r in all_results if r["outcome"] == "loss"]),
    ]:
        if not results:
            continue
        print(f"\n  {label}:")
        for qi, qlabel in enumerate(["Q1 (25%)", "Q2 (50%)", "Q3 (75%)", "Final"]):
            our_avg = avg([r["our_q_scores"][qi] for r in results])
            opp_avg = avg([r["opp_q_scores"][qi] for r in results])
            print(f"    {qlabel:<12}: Our {our_avg:>5.1f}  Opp {opp_avg:>5.1f}  Diff {our_avg-opp_avg:>+5.1f}")

    # ========== SECTION 10: Wins vs Losses Deep Comparison ==========
    print_section("WINS vs LOSSES: KEY DIFFERENCES")

    wins = [r for r in all_results if r["outcome"] == "win"]
    losses = [r for r in all_results if r["outcome"] == "loss"]

    if wins and losses:
        metrics = [
            ("Points per turn (us)", "our_ppt"),
            ("Points per turn (opp)", "opp_ppt"),
            ("Time used (us)", "our_time_used"),
            ("Time used (opp)", "opp_time_used"),
            ("Avg turn time (us)", "our_avg_turn_time"),
            ("Avg turn time (opp)", "opp_avg_turn_time"),
            ("Max turn time (us)", "our_max_turn_time"),
            ("Max turn time (opp)", "opp_max_turn_time"),
            ("Rat catches (us)", "our_rat_catches"),
            ("Rat catches (opp)", "opp_rat_catches"),
            ("Carpets placed (us)", "our_carpets"),
            ("Carpets placed (opp)", "opp_carpets"),
            ("Early-game pts (us)", "our_early"),
            ("Late-game pts (us)", "our_late"),
            ("Early-game pts (opp)", "opp_early"),
            ("Late-game pts (opp)", "opp_late"),
            ("Max lead during game", "max_diff"),
            ("Max deficit during game", "min_diff"),
        ]

        print(f"\n  {'Metric':<30} {'Wins Avg':>10} {'Losses Avg':>10} {'Delta':>10}")
        print("  " + "-" * 62)
        for name, key in metrics:
            w_avg = avg([r[key] for r in wins])
            l_avg = avg([r[key] for r in losses])
            delta = w_avg - l_avg
            print(f"  {name:<30} {w_avg:>10.2f} {l_avg:>10.2f} {delta:>+10.2f}")

        # Cell type comparison
        print_subsection("Cell Type Usage: Wins vs Losses")
        print(f"  {'Type':<10} {'Win (Us)':>9} {'Loss (Us)':>10} {'Win (Opp)':>10} {'Loss (Opp)':>11}")
        for ct in cell_types:
            w_our = avg([r["our_cells"].get(ct, 0) for r in wins])
            l_our = avg([r["our_cells"].get(ct, 0) for r in losses])
            w_opp = avg([r["opp_cells"].get(ct, 0) for r in wins])
            l_opp = avg([r["opp_cells"].get(ct, 0) for r in losses])
            print(f"  {ct:<10} {w_our:>9.1f} {l_our:>10.1f} {w_opp:>10.1f} {l_opp:>11.1f}")

    # ========== SECTION 11: Game End Reasons ==========
    print_section("GAME END REASONS")
    reason_counts = defaultdict(lambda: {"win": 0, "loss": 0, "tie": 0})
    for r in all_results:
        reason_counts[r["reason"]][r["outcome"]] += 1
    for reason, counts in sorted(reason_counts.items()):
        total = sum(counts.values())
        print(
            f"  {reason}: {total} games "
            f"(W:{counts['win']} L:{counts['loss']} T:{counts['tie']})"
        )

    # ========== SECTION 12: Player A vs Player B Performance ==========
    print_section("PERFORMANCE BY ROLE (Player A vs Player B)")
    for label, results in [("As Player A", results_a), ("As Player B", results_b)]:
        if not results:
            continue
        wins_here = sum(1 for r in results if r["outcome"] == "win")
        print(f"\n  {label}: {wins_here}/{len(results)} wins ({wins_here/len(results)*100:.0f}%)")
        print(f"    Avg score diff:     {avg([r['diff'] for r in results]):+.1f}")
        print(f"    Avg our PPT:        {avg([r['our_ppt'] for r in results]):.2f}")
        print(f"    Avg opp PPT:        {avg([r['opp_ppt'] for r in results]):.2f}")
        print(f"    Avg our time used:  {avg([r['our_time_used'] for r in results]):.1f}s")
        print(f"    Avg rat catches:    {avg([r['our_rat_catches'] for r in results]):.2f}")

    # ========== SECTION 13: Individual Loss Breakdown ==========
    print_section("INDIVIDUAL LOSS BREAKDOWN")
    for r in all_results:
        if r["outcome"] == "loss":
            print(f"\n  {r['filename']} (as {r['we_are'].upper()}):")
            print(f"    Score: {r['our_final']} - {r['opp_final']} (diff {r['diff']:+d})")
            print(f"    Reason: {r['reason']}")
            print(f"    Our PPT: {r['our_ppt']:.2f}  Opp PPT: {r['opp_ppt']:.2f}")
            print(f"    Time used: us {r['our_time_used']:.1f}s  opp {r['opp_time_used']:.1f}s")
            print(f"    Rat catches: us {r['our_rat_catches']}  opp {r['opp_rat_catches']}")
            print(
                f"    Our cells: "
                + ", ".join(f"{k}={v}" for k, v in sorted(r["our_cells"].items()))
            )
            print(
                f"    Opp cells: "
                + ", ".join(f"{k}={v}" for k, v in sorted(r["opp_cells"].items()))
            )
            print(f"    Early: us {r['our_early']} opp {r['opp_early']}  "
                  f"Late: us {r['our_late']} opp {r['opp_late']}")
            if r["swings"]:
                print(f"    Big swings: {len(r['swings'])} "
                      f"(worst: {min(s['change'] for s in r['swings']):+d})")

    print(f"\n{'='*70}")
    print("  END OF ANALYSIS")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
