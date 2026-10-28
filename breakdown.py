"""
Where do the tokens (and dollars) go? Reads the run log written by agent.py.

    uv run breakdown.py                          # all runs in runs/bench.jsonl
    uv run breakdown.py --label react-strong

Every input token on step k is one of three things:

  fixed prefix    system prompt + tool definitions + the question. Identical on
                  every call. Measured exactly: it's the input of step 1.
  re-sent history everything the previous call already saw beyond the prefix
                  (earlier replies and tool results), sent AGAIN.
                  = input(k-1) - prefix
  new content     what was added since the previous call.
                  = input(k) - input(k-1)
                  We split it into the model's own last reply echoed back
                  (~ output(k-1) tokens) and new tool results (the rest).
                  That split is approximate: chat-template wrapping lands
                  in "tool results".

Output tokens are what the model generated. "Wasted" is everything spent on
runs that failed the evaluator.
"""

import argparse
import json
from collections import defaultdict

from agent import PRICES

BUCKETS = ["fixed_prefix", "resent_history", "echoed_reply", "new_tool_results", "output"]


def split_run(steps, strategy="react"):
    """Token counts per bucket for each step of one run (a list, one dict per step)."""
    out = []
    for i, s in enumerate(steps):
        b = dict.fromkeys(BUCKETS, 0)
        b["output"] = s["output_tokens"]
        if strategy != "react" or i == 0:
            # Plan strategies: every call is a fresh prompt, no history to re-send.
            b["fixed_prefix"] = s["input_tokens"]
        else:
            prefix, prev = steps[0]["input_tokens"], steps[i - 1]
            b["fixed_prefix"] = prefix
            b["resent_history"] = prev["input_tokens"] - prefix
            new = s["input_tokens"] - prev["input_tokens"]
            b["echoed_reply"] = min(prev["output_tokens"], new)
            b["new_tool_results"] = new - b["echoed_reply"]
        out.append(b)
    return out


def dollars(model, bucket, tokens):
    side = "output" if bucket == "output" else "input"
    return tokens * PRICES[model][side] / 1_000_000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="runs/bench.jsonl")
    p.add_argument("--label", help="only this strategy label, e.g. react-strong")
    args = p.parse_args()

    groups = defaultdict(list)
    for line in open(args.log):
        r = json.loads(line)
        label = r.get("label") or f"{r.get('strategy', 'react')}-{r['model']}"
        if not args.label or label == args.label:
            groups[label].append(r)

    for label, rs in groups.items():
        tok = dict.fromkeys(BUCKETS, 0)
        usd = dict.fromkeys(BUCKETS, 0.0)
        wasted_usd = 0.0
        for r in rs:
            run_usd = 0.0
            for step, b in zip(r["steps"], split_run(r["steps"], r.get("strategy", "react"))):
                model = step.get("model", r["model"])  # cascades mix models within one run
                for k, v in b.items():
                    tok[k] += v
                    usd[k] += dollars(model, k, v)
                    run_usd += dollars(model, k, v)
            if not r["passed"]:
                wasted_usd += run_usd

        n, ok = len(rs), sum(r["passed"] for r in rs)
        total_tok, total_usd = sum(tok.values()), sum(usd.values())
        print(f"\n{label}")
        print(f"runs={n}  passed={ok}  avg model calls={sum(len(r['steps']) for r in rs) / n:.1f}")
        print(f"  {'bucket':18} {'tokens':>9} {'% tok':>6} {'est $':>11} {'% $':>6}")
        for k in BUCKETS:
            print(f"  {k:18} {tok[k]:9d} {100 * tok[k] / total_tok:5.1f}% "
                  f"{usd[k]:11.7f} {100 * usd[k] / total_usd:5.1f}%")
        print(f"  {'TOTAL':18} {total_tok:9d}        {total_usd:11.7f}")
        print(f"  spent on failed runs: ${wasted_usd:.7f} ({100 * wasted_usd / total_usd:.1f}% of spend)")
        print(f"  cost per success:     " + (f"${total_usd / ok:.7f}" if ok else "inf"))


if __name__ == "__main__":
    main()
