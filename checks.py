"""
Trace checks: catch plans that ran cleanly but probably computed the wrong thing.

    uv run checks.py runs/traces-v1.jsonl      # score the checks against the answer key

No answer key and no knowledge of the tools' domain: each check looks only at the
plan's code and what its tool calls returned. The answer key is used here only to
GRADE the checks (caught / missed / false alarm), never inside them.

  empty_lookup   A tool returned nothing ([], {}, None, '') AND the input looks like a
                 mistake: it nearly matches a value some listing tool knows ('Globex
                 Corp.' vs 'globex'), or it works as input to another lookup tool
                 ('A-1006' passed as a customer, but get_order('A-1006') works).
                 Or the input came from nowhere: not in the question, not in any earlier
                 tool result ('customer123', 'All Customers' - the model invented it).
                 An input from the question that matches nothing anywhere ('wayne') is
                 taken as an honest "doesn't exist", so the empty result is accepted.
                 Needs the tools (read-only, called again by the check); zero-argument
                 tools are treated as listings of known values.
  unseen_value   The code compares something to a string literal ('nuts', 'BOLT')
                 that never appears in any tool result. A filter on a value the
                 data doesn't contain can't match; the model guessed the spelling.
"""

import argparse
import ast
import difflib
import inspect
import json
import re
from collections import Counter

EMPTY = {"[]", "{}", "None", "''", "()", "set()"}


def _is_empty(value):
    return value is None or (hasattr(value, "__len__") and len(value) == 0)


def _near(a: str, b: str) -> bool:
    """Same thing, spelled differently: case, a suffix like ' Inc.', or a small typo."""
    a, b = a.lower().strip(" ."), b.lower().strip(" .")
    return a == b or (min(len(a), len(b)) >= 3 and (a in b or b in a)) \
        or difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def _known_values(tools):
    """Strings returned by zero-argument tools: our best list of valid IDs and names."""
    values = set()
    for name, fn in tools.items():
        if not inspect.signature(fn).parameters:
            result = fn()
            values |= {v for v in (result if isinstance(result, (list, set, tuple)) else []) if isinstance(v, str)}
    return values


def _arg_value(logged: str):
    """Logged args are repr()s, keyword args as 'name=repr'."""
    if re.match(r"^\w+=", logged):
        logged = logged.split("=", 1)[1]
    try:
        return ast.literal_eval(logged)
    except (ValueError, SyntaxError):
        return None


def empty_lookup(code, tool_trace, tools, question=""):
    reasons = []
    known = _known_values(tools)
    for i, c in enumerate(tool_trace):
        if c.get("result") not in EMPTY or len(c["args"]) != 1:
            continue
        arg = _arg_value(c["args"][0])
        if not isinstance(arg, str) or arg in known:
            continue  # a real value with genuinely nothing behind it: honest
        call = f"{c['tool']}({arg!r}) returned {c['result']}"
        earlier = " ".join(e.get("result", "") for e in tool_trace[:i])
        if question and (not arg.strip() or (arg.lower() not in question.lower() and arg not in earlier)):
            reasons.append(f"{call}, and {arg!r} appears neither in the question nor in any earlier tool result")
            continue
        near = sorted(v for v in known if _near(arg, v))
        if near:
            reasons.append(f"{call}, but {', '.join(map(repr, near))} is a known value - did you mean that?")
            continue
        for name, fn in tools.items():  # does it work somewhere else?
            if name == c["tool"] or len(inspect.signature(fn).parameters) != 1:
                continue
            try:
                if not _is_empty(fn(arg)):
                    reasons.append(f"{call}, but {name}({arg!r}) works - wrong kind of input?")
                    break
            except Exception:
                pass
    return reasons


def unseen_value(code, tool_trace, tools=None, question=""):
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


def run_checks(code, tool_trace, tools, question="", enabled=tuple(CHECKS)):
    """All reasons to distrust a clean run, as (check name, reason) pairs."""
    return [(name, reason) for name in enabled for reason in CHECKS[name](code, tool_trace, tools, question)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", default="runs/traces-v1.jsonl")
    p.add_argument("-v", "--verbose", action="store_true", help="print every flagged or missed run")
    p.add_argument("--logs", nargs="*", default=[], help="more logs to pool with the first")
    args = p.parse_args()
    from worlds import world_of
    from tasks import TASKS_BY_ID

    # One row per clean plan attempt: the only runs where a check can change anything.
    # Re-run the CURRENT checks on each logged attempt; grade the attempt's own answer.
    rows = []
    for path in [args.log, *args.logs]:
        for line in open(path):
            r = json.loads(line)
            for s in r["steps"]:
                if s.get("code") is None or "tool_trace" not in s or not s.get("final_vars"):
                    continue
                if s.get("error") and not s["error"].startswith("check failed"):
                    continue  # crashed: already escalated without any check
                try:
                    answer = float(s["final_vars"].get("answer", "None"))
                except ValueError:
                    continue
                right = abs(answer - r["expected"]) < 0.01
                task = TASKS_BY_ID[r["task"]]
                rows.append((r, s, right, run_checks(s["code"], s["tool_trace"], world_of(task)["functions"],
                                                        task["question"])))

    def score(name=None):
        hit = lambda flags: any(n == name for n, _ in flags) if name else bool(flags)
        caught = sum(1 for *_, right, f in rows if not right and hit(f))
        missed = sum(1 for *_, right, f in rows if not right and not hit(f))
        false = sum(1 for *_, right, f in rows if right and hit(f))
        ok = sum(1 for *_, right, f in rows if right and not hit(f))
        return caught, missed, false, ok

    print(f"{len(rows)} clean plan attempts from {', '.join([args.log, *args.logs])}\n")
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
