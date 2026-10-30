"""
A bare agent loop, written by hand, that prints what every step costs.

    uv run agent.py                                          # default task, local cheap model
    uv run agent.py --model ollama/qwen2.5:7b --task hard-two
    uv run agent.py --model anthropic/claude-haiku-5-5 --task lib-med-fee

The agent answers a question about a tiny fake database (world.py, or library.py
for tasks set in the library; see worlds.py). It
can't see the data directly; it has to call tools to look things up. Each loop
iteration is one model call, and we print the token usage and estimated cost
of each one. This is the "step-by-step" (ReAct-style) strategy.

Any model llm.py can reach works. Token counts come from the provider; the $
figures are token counts times the table in prices.py (made up for local models).
"""

import argparse
import json
import re
import time
from pathlib import Path

import llm
from prices import PRICES_VERSION, price_of, set_price, step_cost
from tasks import TASKS_BY_ID
from world import CALCULATOR, run_tool
from worlds import world_of

# Experiment 1: tell the model how to sequence its tool calls.
BASE_PROMPT = "You answer questions using the tools provided. End your answer with a line 'ANSWER: <number>'."
SEQUENCING_RULES = (
    " Work step by step: call a tool, read its result, then decide the next call."
    " Only use {ids} that appeared in earlier tool results - never guess them."
)
# Experiment 2: hand the arithmetic to a deterministic tool.
CALCULATOR_RULE = " Use the calculate tool for all arithmetic; do not do math in your head."


def parse_answer(text: str):
    match = re.search(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", text or "")
    return float(match.group(1).replace(",", "")) if match else None


def is_correct(got, expected) -> bool:
    return got is not None and abs(got - expected) < 0.01


def usage_of(reply: llm.Reply, model: str, elapsed: float) -> dict:
    """One step's usage, in the shape every strategy logs."""
    return {"model": model,
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
            "seconds": round(elapsed, 3),
            "done_reason": reply.done_reason}  # "length" = hit the output cap


def log_record(record: dict, path: str):
    """Append-only run log: one JSON object per line, never rewritten."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_react(task: dict, model: str, rules=True, calculator=False,
              max_steps=20, max_output_tokens=None, verbose=True) -> dict:
    say = print if verbose else (lambda *a, **k: None)

    world = world_of(task)
    system = BASE_PROMPT
    tools = list(world["schemas"])
    if rules:
        system += SEQUENCING_RULES.format(ids=world["react_ids"])
    if calculator:
        system += CALCULATOR_RULE
        tools.append(CALCULATOR)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task["question"]},
    ]

    total_cost = 0.0
    run_start = time.perf_counter()
    steps = []  # per-step usage, saved to the log for later analysis
    reply, tool_calls = None, []

    say(f"task={task['id']}  model={model}  rules={rules}  calculator={calculator}  "
        f"prices: {price_of(model)['source']}\n")

    for step in range(1, max_steps + 1):
        # ---- decision point A: call the model (always the same one, for now) ----
        t0 = time.perf_counter()
        reply = llm.chat(model, messages, tools, max_output_tokens)
        u = usage_of(reply, model, time.perf_counter() - t0)
        cost = step_cost(model, u["input_tokens"], u["output_tokens"])
        total_cost += cost

        tool_calls = reply.tool_calls
        u["tool_calls"] = len(tool_calls)
        steps.append(u)

        say(f"── step {step} ── {u['seconds']:.2f}s  done_reason={u['done_reason']}")
        say(f"   tokens: in={u['input_tokens']}  out={u['output_tokens']}")
        say(f"   cost:   ${cost:.8f}  (running ${total_cost:.8f})")
        if reply.text.strip():
            say(f"   [text] {reply.text.strip()}")
        for call in tool_calls:
            say(f"   [tool_call] {call['name']}({call['args']})")

        # The model has no memory: we append its full reply so the next call sees it.
        messages.append(llm.assistant_turn(model, reply))

        # ---- decision point C: the model decides it's done by not asking for tools ----
        if not tool_calls:
            break

        # ---- decision point B: run every tool it asked for ----
        results = []
        for call in tool_calls:
            result = run_tool(call["name"], call["args"], world["react_functions"])
            say(f"   [tool_result] {call['name']} -> {result}")
            results.append((call, result))
        messages += llm.tool_results(model, results)
        say()
    else:
        say(f"\n!! hit max_steps={max_steps} without finishing")

    # ---- evaluation: don't trust the model's word, check the number ----
    got = parse_answer(reply.text if reply else "")
    passed = is_correct(got, task["expected"])
    wall = time.perf_counter() - run_start

    say("\n══ summary ══")
    say(f"model calls:   {len(steps)}")
    say(f"tokens:        in={sum(s['input_tokens'] for s in steps)}  "
        f"out={sum(s['output_tokens'] for s in steps)}")
    say(f"est. cost:     ${total_cost:.8f}  (prices.py {PRICES_VERSION}: {price_of(model)['source']})")
    say(f"wall time:     {wall:.2f}s")
    say(f"answer:        got={got}  expected={task['expected']}  -> {'PASS' if passed else 'FAIL'}")

    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "strategy": "react", "task": task["id"], "difficulty": task["difficulty"],
        "model": model, "rules": rules, "calculator": calculator,
        "max_steps": max_steps, "prices_version": PRICES_VERSION,
        "passed": passed, "answer": got, "expected": task["expected"],
        "hit_max_steps": bool(tool_calls),  # loop only exits with pending tool calls at the step limit
        "est_cost": total_cost, "seconds": round(wall, 3), "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="hard-acme", choices=TASKS_BY_ID)
    parser.add_argument("--model", default="ollama/qwen2.5:1.5b", help="provider/model, see llm.py")
    parser.add_argument("--price", action="append", default=[], metavar="MODEL=IN,OUT",
                        help="USD per 1M input/output tokens for a model not in prices.py")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--no-rules", action="store_true", help="drop the sequencing rules from the prompt")
    parser.add_argument("--calculator", action="store_true", help="experiment 2: add the calculate tool")
    parser.add_argument("--max-output-tokens", type=int, default=llm.MAX_TOKENS,
                        help="per-call output cap; raise it for models that think before answering")
    parser.add_argument("--log", default="runs/runs.jsonl", help="append one JSON record per run here")
    args = parser.parse_args()
    for spec in args.price:
        set_price(spec)

    record = run_react(TASKS_BY_ID[args.task], args.model, rules=not args.no_rules,
                       calculator=args.calculator, max_steps=args.max_steps,
                       max_output_tokens=args.max_output_tokens)
    log_record(record, args.log)


if __name__ == "__main__":
    main()
