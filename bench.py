"""
Run strategies on tasks, several times, into one log.

    uv run bench.py                                          # local Ollama pair, default strategies
    uv run bench.py --split library --repeats 3 --log runs/holdout-v3.jsonl
    uv run bench.py --cheap anthropic/claude-haiku-5-5 --strong anthropic/claude-sonnet-5-5 \\
                    --split library --repeats 2 --budget 5 --log runs/claude.jsonl
    uv run curve.py runs/claude.jsonl                        # then compare
"""

import argparse
import time

import llm
from agent import log_record, run_react
from planner import run_hybrid, run_plan
from prices import price_of, set_price
from tasks import SPLITS

# Each strategy gets the task and the two models it may use.
STRATEGIES = {
    "react-cheap":  lambda t, cheap, strong: run_react(t, cheap, verbose=False),
    "react-strong": lambda t, cheap, strong: run_react(t, strong, verbose=False),
    "plan-cheap":   lambda t, cheap, strong: run_plan(t, [cheap], verbose=False),
    "plan-strong":  lambda t, cheap, strong: run_plan(t, [strong], verbose=False),
    "plan-cascade": lambda t, cheap, strong: run_plan(t, [cheap, strong], verbose=False),
    # Fall back to step-by-step when planning fails detectably.
    "plan>react":    lambda t, cheap, strong: run_hybrid(t, [cheap], strong, verbose=False),
    "cascade>react": lambda t, cheap, strong: run_hybrid(t, [cheap, strong], strong, verbose=False),
    # Also reject clean plans that fail a trace check; optionally tell the next rung why.
    "cascade>react+checks":    lambda t, cheap, strong: run_hybrid(t, [cheap, strong], strong, checks=True,
                                                                   verbose=False),
    "cascade>react+checks+fb": lambda t, cheap, strong: run_hybrid(t, [cheap, strong], strong, checks=True,
                                                                   feedback=True, verbose=False),
    # Before climbing a rung, let the same model repair its rejected plan once.
    "cascade>react+checks+repair": lambda t, cheap, strong: run_hybrid(t, [cheap, strong], strong, checks=True,
                                                                       repairs=1, verbose=False),
    # Repair only on the strong rung and only after a failed check: in experiment 9 those repairs
    # fixed 10 of 12, while the cheap model's repairs and repairs after a crash almost never worked.
    # (Called "+repair7b" in experiment 10.)
    "cascade>react+checks+repair-strong": lambda t, cheap, strong: run_hybrid(
        t, [cheap, strong], strong, checks=True, repairs={strong: 1}, repair_crashes=False, verbose=False),
}
DEFAULT = ["plan-cheap", "cascade>react", "cascade>react+checks", "react-strong"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cheap", default="ollama/qwen2.5:1.5b", help="provider/model, see llm.py")
    p.add_argument("--strong", default="ollama/qwen2.5:7b", help="provider/model, see llm.py")
    p.add_argument("--strategies", nargs="*", default=DEFAULT, choices=STRATEGIES)
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--tasks", nargs="*", help="only these task ids from the split")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--budget", type=float, help="stop once the estimated spend passes this many dollars")
    p.add_argument("--price", action="append", default=[], metavar="MODEL=IN,OUT",
                   help="USD per 1M input/output tokens for a model not in prices.py")
    p.add_argument("--max-output-tokens", type=int, default=llm.MAX_TOKENS,
                   help="per-call output cap; raise it for models that think before answering")
    p.add_argument("--log", default="runs/bench.jsonl")
    args = p.parse_args()
    llm.MAX_TOKENS = args.max_output_tokens
    for spec in args.price:
        set_price(spec)
    for model in (args.cheap, args.strong):
        try:
            price_of(model)  # fail now, not halfway through the run
        except KeyError as e:
            p.error(e.args[0])

    tasks = SPLITS[args.split]
    if args.tasks:
        unknown = set(args.tasks) - {t["id"] for t in tasks}
        if unknown:
            p.error(f"not in split {args.split!r}: {', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t["id"] in args.tasks]
    total = args.repeats * len(args.strategies) * len(tasks)
    done, spent, start = 0, 0.0, time.perf_counter()
    width = max(map(len, args.strategies))
    for rep in range(1, args.repeats + 1):  # repeats outermost, so a partial run still covers everything
        for name in args.strategies:
            for task in tasks:
                if args.budget is not None and spent >= args.budget:
                    print(f"budget ${args.budget:.2f} reached after {done}/{total} runs (${spent:.4f}); stopping")
                    return
                record = STRATEGIES[name](task, args.cheap, args.strong)
                record.update(label=name, repeat=rep, split=args.split, cheap=args.cheap, strong=args.strong)
                log_record(record, args.log)
                done += 1
                spent += record["est_cost"]
                print(f"[{done}/{total}] {time.perf_counter() - start:6.0f}s  {name:{width}} {task['id']:13} "
                      f"{'PASS' if record['passed'] else 'fail'}  calls={len(record['steps'])}  "
                      f"${record['est_cost']:.7f}  (total ${spent:.4f})", flush=True)


if __name__ == "__main__":
    main()
