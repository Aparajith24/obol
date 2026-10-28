"""
Summarise a bench log: success, cost per success, calls and tokens per strategy.

    uv run report.py runs/bench.jsonl
"""

import json
import sys
from collections import defaultdict

DIFFICULTIES = ["easy", "medium", "hard", "edge", "complex", "surprise"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/bench.jsonl"
    runs = [json.loads(line) for line in open(path)]
    by_label = defaultdict(list)
    for r in runs:
        by_label[r.get("label", r["strategy"])].append(r)

    print(f"{len(runs)} runs from {path}  (costs: {runs[0]['prices_version']}, hypothetical)\n")
    print(f"{'strategy':14} {'pass':>7}  " + "  ".join(f"{d:>6}" for d in DIFFICULTIES)
          + f"  {'calls':>5} {'in tok':>6} {'out tok':>7} {'secs':>5}  {'total $':>10} {'$/success':>10}"
          + f"  {'escalated':>9}")
    for label, rs in by_label.items():
        n, ok = len(rs), sum(r["passed"] for r in rs)
        spend = sum(r["est_cost"] for r in rs)
        per_diff = []
        for d in DIFFICULTIES:
            ds = [r for r in rs if r["difficulty"] == d]
            per_diff.append(f"{sum(r['passed'] for r in ds)}/{len(ds)}" if ds else "-")
        avg = lambda f: sum(f(r) for r in rs) / n
        esc = sum(r.get("escalated", False) for r in rs)
        print(f"{label:14} {ok:>3}/{n:<3}  " + "  ".join(f"{x:>6}" for x in per_diff)
              + f"  {avg(lambda r: len(r['steps'])):5.1f}"
              + f" {avg(lambda r: sum(s['input_tokens'] for s in r['steps'])):6.0f}"
              + f" {avg(lambda r: sum(s['output_tokens'] for s in r['steps'])):7.0f}"
              + f" {avg(lambda r: r['seconds']):5.1f}"
              + f"  {spend:10.7f} {(f'{spend / ok:10.7f}' if ok else '       inf')}"
              + f"  {esc if esc else '-':>9}")

    print("\nper task (passes / runs):")
    tasks = list(dict.fromkeys(r["task"] for r in runs))
    print(f"{'':14} " + " ".join(f"{t:>13}" for t in tasks))
    for label, rs in by_label.items():
        cells = []
        for t in tasks:
            ts = [r for r in rs if r["task"] == t]
            cells.append(f"{sum(r['passed'] for r in ts)}/{len(ts)}")
        print(f"{label:14} " + " ".join(f"{c:>13}" for c in cells))

    errors = defaultdict(int)
    for r in runs:
        for s in r["steps"]:
            if s.get("error"):
                errors[(s["model"], s["error"].split(":")[0])] += 1
    if errors:
        print("\nplan failures we could detect (model, kind): count")
        for (m, kind), c in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {m:13} {kind:30} {c}")


if __name__ == "__main__":
    main()
