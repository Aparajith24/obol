"""
Trace checks: catch plans that ran cleanly but probably computed the wrong thing.

    uv run checks.py runs/traces-v1.jsonl      # score the checks against the answer key

No answer key and no knowledge of the tools' domain: each check looks only at the
plan's code and what its tool calls returned. The answer key is used here only to
GRADE the checks (caught / missed / false alarm), never inside them.

  empty_lookup   A tool returned nothing ([], {}, None, ''). Usually a wrong input:
                 a name that isn't an ID, an order ID passed as a customer.
                 False alarm when "nothing" is the honest answer (a customer with no
                 orders) - escalating then costs money but shouldn't hurt the answer.
  unseen_value   The code compares something to a string literal ('nuts', 'BOLT')
                 that never appears in any tool result. A filter on a value the
                 data doesn't contain can't match; the model guessed the spelling.
"""

import argparse
import ast
import json
from collections import Counter

EMPTY = {"[]", "{}", "None", "''", "()", "set()"}


def empty_lookup(code, tool_trace):
    return [f"{c['tool']}({', '.join(c['args'])}) returned {c['result']}"
            for c in tool_trace if c.get("result") in EMPTY]


def unseen_value(code, tool_trace):
    seen = " ".join(c.get("result", "") for c in tool_trace)
    reasons = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Compare) or not all(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            continue
        for side in [node.left, *node.comparators]:
            if isinstance(side, ast.Constant) and isinstance(side.value, str) and repr(side.value) not in seen:
                reasons.append(f"compares against {side.value!r}, which no tool result contains")
    return reasons


CHECKS = {"empty_lookup": empty_lookup, "unseen_value": unseen_value}


def run_checks(code, tool_trace, enabled=tuple(CHECKS)):
    """All reasons to distrust a clean run, as (check name, reason) pairs."""
    return [(name, reason) for name in enabled for reason in CHECKS[name](code, tool_trace)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", default="runs/traces-v1.jsonl")
    p.add_argument("-v", "--verbose", action="store_true", help="print every flagged or missed run")
    args = p.parse_args()

    # One row per clean plan attempt: the only runs where a check can change anything.
    rows = []
    for line in open(args.log):
        r = json.loads(line)
        for s in r["steps"]:
            if s.get("code") is not None and s.get("error") is None and "tool_trace" in s:
                right = r["passed"] if len(r["steps"]) == 1 else None  # cascades: can't tell which rung was right
                if right is not None:
                    rows.append((r, s, right, run_checks(s["code"], s["tool_trace"])))

    def score(name=None):
        hit = lambda flags: any(n == name for n, _ in flags) if name else bool(flags)
        caught = sum(1 for *_, right, f in rows if not right and hit(f))
        missed = sum(1 for *_, right, f in rows if not right and not hit(f))
        false = sum(1 for *_, right, f in rows if right and hit(f))
        ok = sum(1 for *_, right, f in rows if right and not hit(f))
        return caught, missed, false, ok

    print(f"{len(rows)} clean plan runs from {args.log}\n")
    print(f"{'check':14} {'caught':>7} {'missed':>7} {'recall':>7}  {'false alarm':>11} {'of right':>9}")
    for name in [*CHECKS, None]:
        c, m, f, ok = score(name)
        print(f"{name or 'ANY':14} {c:7} {m:7} {c / max(c + m, 1):7.0%}  {f:11} {f / max(f + ok, 1):9.0%}")

    alarms = Counter(r["task"] for r, s, right, f in rows if right and f)
    if alarms:
        print("\nfalse alarms by task: " + ", ".join(f"{t}={n}" for t, n in alarms.most_common()))

    if args.verbose:
        for r, s, right, flags in rows:
            if flags or not right:
                tag = "CAUGHT" if flags and not right else "FALSE ALARM" if flags else "MISSED"
                print(f"\n[{tag}] {r['label']} {r['task']} got={r['answer']} expected={r['expected']}")
                for name, reason in flags:
                    print(f"   {name}: {reason}")


if __name__ == "__main__":
    main()
