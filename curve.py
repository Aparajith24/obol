"""
Cost vs quality: is a strategy actually better than mixing the baselines at random?

    uv run curve.py runs/combined-v3.jsonl

Ideas borrowed from RouteLLM (Ong et al. 2024):

  random baseline  Send a fraction p of tasks to strategy B and the rest to A.
                   Expected cost and pass rate both move p of the way from A to B,
                   so every random mix lies on the straight line between them.
                   A strategy only "routes well" if it sits ABOVE the line between
                   the always-cheap and always-strong baselines at its own cost.
  PGR              Performance gap recovered: how much of the cheap->strong pass-rate
                   gap a strategy closes. 0 = as good as cheap, 1 = as good as strong.
  frontier         Strategies on the upper convex hull of (cost, pass rate). Anything
                   below it is beaten by some random mix of the strategies on it.
  oracle           Per task, pick the strategy with the best pass rate (ties: cheaper).
                   It needs the answer key, so it's not achievable; it's the ceiling
                   for a perfect per-task strategy router.
"""

import argparse
import json
import math
import random
from collections import defaultdict


def wilson(k, n, z=1.96):
    """95% confidence interval for a pass rate of k/n (Wilson score)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def upper_hull(points):
    """Upper-left convex hull of (cost, pass) points: the best achievable mixes."""
    pts = sorted(points)
    hull = []
    for p in pts:
        if hull and p[1] <= hull[-1][1]:
            continue  # costs more, passes no more: dominated
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # drop the middle point if it lies on or below the line from x1 to p
            if (y2 - y1) * (p[0] - x1) <= (p[1] - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", default="runs/bench.jsonl")
    p.add_argument("--cheap", default="react-cheap", help="always-cheap baseline label")
    p.add_argument("--strong", default="react-strong", help="always-strong baseline label")
    args = p.parse_args()

    by_label = defaultdict(list)
    for line in open(args.log):
        r = json.loads(line)
        by_label[r.get("label", r["strategy"])].append(r)

    stats = {}
    for label, rs in by_label.items():
        n, k = len(rs), sum(r["passed"] for r in rs)
        stats[label] = {"n": n, "k": k, "pass": k / n,
                        "cost": sum(r["est_cost"] for r in rs) / n,  # mean $ per task attempted
                        "ci": wilson(k, n)}

    cheap, strong = stats[args.cheap], stats[args.strong]
    gap_cost, gap_pass = strong["cost"] - cheap["cost"], strong["pass"] - cheap["pass"]
    hull = upper_hull([(s["cost"], s["pass"]) for s in stats.values()])

    print(f"{len(sum(by_label.values(), []))} runs from {args.log}  (hypothetical prices)")
    print(f"baselines: cheap={args.cheap}  strong={args.strong}\n")
    print(f"{'strategy':14} {'pass':>7} {'95% CI':>13} {'$ / task':>10} {'% strong $':>10}"
          f" {'PGR':>5} {'random@same$':>12} {'lift':>6}  frontier")
    for label, s in sorted(stats.items(), key=lambda x: x[1]["cost"]):
        pct_cost = s["cost"] / strong["cost"]
        pgr = (s["pass"] - cheap["pass"]) / gap_pass if gap_pass else float("nan")
        # Random mix of the two baselines that spends the same per task (capped at the ends).
        mix = min(max((s["cost"] - cheap["cost"]) / gap_cost, 0), 1) if gap_cost else 0
        rand = cheap["pass"] + mix * gap_pass
        lo, hi = s["ci"]
        on_hull = "*" if (s["cost"], s["pass"]) in hull else ""
        print(f"{label:14} {s['k']:>3}/{s['n']:<3} {f'{lo:.0%}-{hi:.0%}':>13} {s['cost']:10.7f}"
              f" {pct_cost:10.1%} {pgr:5.2f} {rand:12.0%} {s['pass'] - rand:+6.0%}  {on_hull}")

    # Oracle: per task, the strategy with the best pass rate there (ties: cheaper).
    per_task = defaultdict(lambda: defaultdict(list))
    for label, rs in by_label.items():
        for r in rs:
            per_task[r["task"]][label].append(r)
    o_pass = o_cost = 0.0
    picks = defaultdict(int)
    for task, labels in per_task.items():
        best = min(labels, key=lambda l: (-sum(r["passed"] for r in labels[l]) / len(labels[l]),
                                          sum(r["est_cost"] for r in labels[l]) / len(labels[l])))
        rs = labels[best]
        o_pass += sum(r["passed"] for r in rs) / len(rs)
        o_cost += sum(r["est_cost"] for r in rs) / len(rs)
        picks[best] += 1
    t = len(per_task)
    print(f"\noracle router (needs the answer key, a ceiling): pass {o_pass / t:.0%}  "
          f"$ / task {o_cost / t:.7f}  ({o_cost / t / strong['cost']:.1%} of strong)")
    print("  picks: " + ", ".join(f"{l}={c}" for l, c in sorted(picks.items(), key=lambda x: -x[1])))

    print("\nfrontier (cheapest -> best), mix neighbours at random to hit any point between:")
    names = {(s["cost"], s["pass"]): l for l, s in stats.items()}
    print("  " + "  ->  ".join(f"{names[pt]} ({pt[1]:.0%}, ${pt[0]:.7f})" for pt in hull))

    # Non-inferiority vs strong, paired by task: the same tasks run under both strategies,
    # so compare per-task pass rates and resample whole tasks (repeats of a task aren't independent).
    print(f"\npass-rate difference vs {args.strong}, paired by task (95% bootstrap CI over tasks):")
    rng = random.Random(0)
    for label in sorted(stats, key=lambda l: stats[l]["cost"]):
        if label == args.strong:
            continue
        diffs = []
        for labels in per_task.values():
            if label in labels and args.strong in labels:
                rate = lambda l: sum(r["passed"] for r in labels[l]) / len(labels[l])
                diffs.append(rate(label) - rate(args.strong))
        means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10_000))
        mean = sum(diffs) / len(diffs)
        print(f"  {label:14} {mean:+6.1%}  [{means[249]:+6.1%}, {means[9749]:+6.1%}]  ({len(diffs)} tasks)")


if __name__ == "__main__":
    main()
