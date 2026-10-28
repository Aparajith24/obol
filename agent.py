"""
A bare agent loop, written by hand, that prints what every step costs.

    uv run agent.py                                  # default task, cheap model
    uv run agent.py --model qwen2.5:7b --task hard-two

The agent answers a question about a tiny fake order database (world.py). It
can't see the data directly; it has to call tools to look things up. Each loop
iteration is one model call, and we print the token usage and estimated cost
of each one. This is the "step-by-step" (ReAct-style) strategy.

Models run locally through Ollama (`ollama serve` must be running), so the real
bill is $0. Token counts and timings are real; the $ figures come from a
HYPOTHETICAL price table so we can practise the accounting.
"""

import argparse
import json
import re
import time
from pathlib import Path

import ollama

from tasks import TASKS_BY_ID
from world import CALCULATOR, TOOLS, run_tool

# ---------------------------------------------------------------------------
# HYPOTHETICAL prices, USD per 1M tokens. Local inference is free; these are
# placeholders standing in for "what a hosted provider might charge for this
# model". Change them freely - every cost printed is tagged with this version.
# ---------------------------------------------------------------------------
PRICES_VERSION = "hypothetical-v2"
PRICES = {
    "qwen2.5:1.5b": {"input": 0.05, "output": 0.10},  # "cheap"
    "qwen2.5:7b":   {"input": 0.25, "output": 0.50},  # "strong": 5x, roughly tracking model size
}

# Experiment 1: tell the model how to sequence its tool calls.
BASE_PROMPT = "You answer questions using the tools provided. End your answer with a line 'ANSWER: <number>'."
SEQUENCING_RULES = (
    " Work step by step: call a tool, read its result, then decide the next call."
    " Only use order IDs and skus that appeared in earlier tool results - never guess them."
)
# Experiment 2: hand the arithmetic to a deterministic tool.
CALCULATOR_RULE = " Use the calculate tool for all arithmetic; do not do math in your head."


def step_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def parse_answer(text: str):
    match = re.search(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", text or "")
    return float(match.group(1).replace(",", "")) if match else None


def is_correct(got, expected) -> bool:
    return got is not None and abs(got - expected) < 0.01


def usage_of(response, model: str, elapsed: float) -> dict:
    """One step's usage, in the shape every strategy logs."""
    return {"model": model,
            "input_tokens": response.prompt_eval_count or 0,
            "output_tokens": response.eval_count or 0,
            "seconds": round(elapsed, 3),
            "done_reason": response.done_reason}  # "length" = hit the output cap


def log_record(record: dict, path: str):
    """Append-only run log: one JSON object per line, never rewritten."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_react(task: dict, model: str, rules=True, calculator=False,
              max_steps=20, max_output_tokens=2048, verbose=True) -> dict:
    say = print if verbose else (lambda *a, **k: None)

    system = BASE_PROMPT
    tools = list(TOOLS)
    if rules:
        system += SEQUENCING_RULES
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
    response, tool_calls = None, []

    say(f"task={task['id']}  model={model}  rules={rules}  calculator={calculator}  "
        f"prices={PRICES_VERSION} (hypothetical)\n")

    for step in range(1, max_steps + 1):
        # ---- decision point A: call the model (always the same one, for now) ----
        t0 = time.perf_counter()
        response = ollama.chat(model=model, messages=messages, tools=tools,
                               options={"num_predict": max_output_tokens})
        u = usage_of(response, model, time.perf_counter() - t0)

        # Ollama's usage metadata: token counts plus nanosecond timings.
        in_tps = u["input_tokens"] / (response.prompt_eval_duration / 1e9) if response.prompt_eval_duration else 0
        out_tps = u["output_tokens"] / (response.eval_duration / 1e9) if response.eval_duration else 0
        cost = step_cost(model, u["input_tokens"], u["output_tokens"])
        total_cost += cost

        msg = response.message
        tool_calls = msg.tool_calls or []
        u["tool_calls"] = len(tool_calls)
        steps.append(u)

        say(f"── step {step} ── {u['seconds']:.2f}s  done_reason={u['done_reason']}")
        say(f"   tokens: in={u['input_tokens']} ({in_tps:.0f} tok/s)  out={u['output_tokens']} ({out_tps:.0f} tok/s)")
        say(f"   cost:   ${cost:.8f}  (running ${total_cost:.8f})")
        if msg.content.strip():
            say(f"   [text] {msg.content.strip()}")
        for call in tool_calls:
            say(f"   [tool_call] {call.function.name}({call.function.arguments})")

        # The model has no memory: we append its full reply so the next call sees it.
        messages.append(msg)

        # ---- decision point C: the model decides it's done by not asking for tools ----
        if not tool_calls:
            break

        # ---- decision point B: run every tool it asked for ----
        for call in tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            say(f"   [tool_result] {call.function.name} -> {result}")
            messages.append({"role": "tool", "tool_name": call.function.name, "content": result})
        say()
    else:
        say(f"\n!! hit max_steps={max_steps} without finishing")

    # ---- evaluation: don't trust the model's word, check the number ----
    got = parse_answer(response.message.content if response else "")
    passed = is_correct(got, task["expected"])
    wall = time.perf_counter() - run_start

    say("\n══ summary ══")
    say(f"model calls:   {len(steps)}")
    say(f"tokens:        in={sum(s['input_tokens'] for s in steps)}  "
        f"out={sum(s['output_tokens'] for s in steps)}")
    say(f"est. cost:     ${total_cost:.8f}  ({PRICES_VERSION}, hypothetical)")
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
    parser.add_argument("--model", default="qwen2.5:1.5b", choices=PRICES)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--no-rules", action="store_true", help="drop the sequencing rules from the prompt")
    parser.add_argument("--calculator", action="store_true", help="experiment 2: add the calculate tool")
    parser.add_argument("--max-output-tokens", type=int, default=2048,
                        help="per-call output cap (Ollama's default is unlimited)")
    parser.add_argument("--log", default="runs/runs.jsonl", help="append one JSON record per run here")
    args = parser.parse_args()

    record = run_react(TASKS_BY_ID[args.task], args.model, rules=not args.no_rules,
                       calculator=args.calculator, max_steps=args.max_steps,
                       max_output_tokens=args.max_output_tokens)
    log_record(record, args.log)


if __name__ == "__main__":
    main()
