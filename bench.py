"""
Run every strategy on every task, several times, into one log.

    uv run bench.py                    # 3 repeats -> runs/bench.jsonl
    uv run bench.py --repeats 5 --log runs/bench-5x.jsonl
    uv run report.py runs/bench.jsonl  # then summarise
"""

import argparse
import time

from agent import log_record, run_react
from planner import run_plan
from tasks import TASKS

CHEAP, STRONG = "qwen2.5:1.5b", "qwen2.5:7b"

STRATEGIES = {
    "react-cheap":  lambda t: run_react(t, CHEAP, verbose=False),
    "react-strong": lambda t: run_react(t, STRONG, verbose=False),
    "plan-cheap":   lambda t: run_plan(t, [CHEAP], verbose=False),
    "plan-strong":  lambda t: run_plan(t, [STRONG], verbose=False),
    "plan-cascade": lambda t: run_plan(t, [CHEAP, STRONG], verbose=False),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--strategies", nargs="*", default=list(STRATEGIES), choices=STRATEGIES)
    p.add_argument("--log", default="runs/bench.jsonl")
    args = p.parse_args()

    total = args.repeats * len(args.strategies) * len(TASKS)
    done, start = 0, time.perf_counter()
    for rep in range(1, args.repeats + 1):  # repeats outermost, so a partial run still covers everything
        for name in args.strategies:
            for task in TASKS:
                record = STRATEGIES[name](task)
                record["label"] = name
                record["repeat"] = rep
                log_record(record, args.log)
                done += 1
                print(f"[{done}/{total}] {time.perf_counter() - start:6.0f}s  {name:13} {task['id']:13} "
                      f"{'PASS' if record['passed'] else 'fail'}  calls={len(record['steps'])}  "
                      f"${record['est_cost']:.7f}", flush=True)


if __name__ == "__main__":
    main()
