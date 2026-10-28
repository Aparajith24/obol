"""
A bare agent loop, written by hand, that prints what every step costs.

    uv run agent.py                                        # default model
    uv run agent.py --model qwen2.5:7b                     # the "strong" model

The agent answers a question about a tiny fake order database. It can't see the
data directly; it has to call tools to look things up. Each loop iteration is
one model call, and we print the token usage and estimated cost of each one.

Models run locally through Ollama (`ollama serve` must be running), so the real
bill is $0. Token counts and timings are real; the $ figures come from a
HYPOTHETICAL price table so we can practise the accounting.
"""

import argparse
import ast
import json
import operator
import re
import time
from pathlib import Path

import ollama

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

# ---------------------------------------------------------------------------
# The "world" the agent works in. Plain Python data; the model never sees it
# except through tool results.
# ---------------------------------------------------------------------------
ORDERS = {
    "A-1001": {"customer": "acme",    "items": [{"sku": "bolt", "qty": 120}, {"sku": "nut", "qty": 300}]},
    "A-1002": {"customer": "globex",  "items": [{"sku": "bracket", "qty": 15}]},
    "A-1003": {"customer": "acme",    "items": [{"sku": "bracket", "qty": 8}, {"sku": "washer", "qty": 500}]},
    "A-1004": {"customer": "acme",    "items": [{"sku": "bolt", "qty": 40}]},
    "A-1005": {"customer": "initech", "items": [{"sku": "nut", "qty": 1000}]},
}
UNIT_PRICES = {"bolt": 0.25, "nut": 0.05, "washer": 0.02, "bracket": 3.40}

QUESTION = "What is the total value in dollars of all orders placed by the customer 'acme'?"


def expected_answer() -> float:
    """Ground truth, computed without any model. This is our evaluator."""
    return round(sum(
        item["qty"] * UNIT_PRICES[item["sku"]]
        for order in ORDERS.values() if order["customer"] == "acme"
        for item in order["items"]
    ), 2)


# ---------------------------------------------------------------------------
# Tools: the JSON schema we send to the model, and the Python that runs them.
# Note: these definitions are re-sent (and counted as input) on EVERY call.
# ---------------------------------------------------------------------------
def _tool(name, description, param):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {param: {"type": "string"}},
                "required": [param],
            },
        },
    }


TOOLS = [
    _tool("list_orders", "List the order IDs placed by a customer.", "customer"),
    _tool("get_order", "Get the line items (sku and quantity) for one order.", "order_id"),
    _tool("get_unit_price", "Get the unit price in dollars for a product sku.", "sku"),
]
CALCULATOR = _tool("calculate", "Evaluate an arithmetic expression using + - * / and parentheses, "
                                "e.g. '120*0.25 + 300*0.05'.", "expression")

# Experiment 1: tell the model how to sequence its tool calls.
BASE_PROMPT = "You answer questions using the tools provided. End your answer with a line 'TOTAL: <number>'."
SEQUENCING_RULES = (
    " Work step by step: call a tool, read its result, then decide the next call."
    " Only use order IDs and skus that appeared in earlier tool results - never guess them."
)
# Experiment 2: hand the arithmetic to a deterministic tool.
CALCULATOR_RULE = " Use the calculate tool for all arithmetic; do not do math in your head."

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(expression: str) -> float:
    """Safe arithmetic: walk the syntax tree, allow only numbers and + - * /. Never eval()."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        raise ValueError("only numbers and + - * / are allowed")
    return round(walk(ast.parse(expression, mode="eval").body), 6)


def run_tool(name: str, args: dict) -> str:
    """Model output is untrusted, so check what it asked for."""
    if name == "list_orders":
        return str([oid for oid, o in ORDERS.items() if o["customer"] == args.get("customer")])
    if name == "get_order":
        order = ORDERS.get(args.get("order_id"))
        return str(order["items"]) if order else "error: no such order"
    if name == "get_unit_price":
        price = UNIT_PRICES.get(args.get("sku"))
        return str(price) if price is not None else "error: no such sku"
    if name == "calculate":
        try:
            return str(calculate(str(args.get("expression", ""))))
        except (ValueError, SyntaxError, ZeroDivisionError) as e:
            return f"error: {e}"
    return f"error: unknown tool {name}"


def step_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:1.5b", choices=PRICES)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--rules", action="store_true", help="experiment 1: add sequencing rules to the prompt")
    parser.add_argument("--calculator", action="store_true", help="experiment 2: add the calculate tool")
    parser.add_argument("--max-output-tokens", type=int, default=2048,
                        help="per-call output cap (Ollama's default is unlimited)")
    parser.add_argument("--log", default="runs/runs.jsonl", help="append one JSON record per run here")
    args = parser.parse_args()

    system = BASE_PROMPT
    tools = list(TOOLS)
    if args.rules:
        system += SEQUENCING_RULES
    if args.calculator:
        system += CALCULATOR_RULE
        tools.append(CALCULATOR)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": QUESTION},
    ]

    total_cost = 0.0
    total_in = total_out = 0
    run_start = time.perf_counter()
    response = None
    steps = []  # per-step usage, saved to the log for later analysis

    print(f"model={args.model}  rules={args.rules}  calculator={args.calculator}  "
          f"prices={PRICES_VERSION} (hypothetical)\n")

    for step in range(1, args.max_steps + 1):
        # ---- decision point A: call the model (always the same one, for now) ----
        t0 = time.perf_counter()
        response = ollama.chat(model=args.model, messages=messages, tools=tools,
                               options={"num_predict": args.max_output_tokens})
        elapsed = time.perf_counter() - t0

        # Ollama's usage metadata: token counts plus nanosecond timings.
        in_tok = response.prompt_eval_count or 0
        out_tok = response.eval_count or 0
        in_tps = in_tok / (response.prompt_eval_duration / 1e9) if response.prompt_eval_duration else 0
        out_tps = out_tok / (response.eval_duration / 1e9) if response.eval_duration else 0

        cost = step_cost(args.model, in_tok, out_tok)
        total_cost += cost
        total_in += in_tok
        total_out += out_tok

        msg = response.message
        tool_calls = msg.tool_calls or []
        steps.append({"step": step, "input_tokens": in_tok, "output_tokens": out_tok,
                      "seconds": round(elapsed, 3), "tool_calls": len(tool_calls),
                      "done_reason": response.done_reason})  # "length" = hit the output cap

        print(f"── step {step} ── {elapsed:.2f}s  done_reason={response.done_reason}")
        print(f"   tokens: in={in_tok} ({in_tps:.0f} tok/s)  out={out_tok} ({out_tps:.0f} tok/s)")
        print(f"   cost:   ${cost:.8f}  (running ${total_cost:.8f})")
        if msg.content.strip():
            print(f"   [text] {msg.content.strip()}")
        for call in tool_calls:
            print(f"   [tool_call] {call.function.name}({call.function.arguments})")

        # The model has no memory: we append its full reply so the next call sees it.
        messages.append(msg)

        # ---- decision point C: the model decides it's done by not asking for tools ----
        if not tool_calls:
            break

        # ---- decision point B: run every tool it asked for ----
        for call in tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            print(f"   [tool_result] {call.function.name} -> {result}")
            messages.append({"role": "tool", "tool_name": call.function.name, "content": result})
        print()
    else:
        print(f"\n!! hit max_steps={args.max_steps} without finishing")

    # ---- evaluation: don't trust the model's word, check the number ----
    final_text = response.message.content if response else ""
    match = re.search(r"TOTAL:\s*\$?([\d,]+(?:\.\d+)?)", final_text)
    got = float(match.group(1).replace(",", "")) if match else None
    want = expected_answer()
    passed = got is not None and abs(got - want) < 0.01

    print("\n══ summary ══")
    print(f"model calls:   {step}")
    print(f"tokens:        in={total_in}  out={total_out}")
    print(f"est. cost:     ${total_cost:.8f}  ({PRICES_VERSION}, hypothetical)")
    print(f"wall time:     {time.perf_counter() - run_start:.2f}s")
    print(f"answer:        got={got}  expected={want}  -> {'PASS' if passed else 'FAIL'}")

    # Append-only run log: one JSON object per line, never rewritten.
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model, "rules": args.rules, "calculator": args.calculator,
        "max_steps": args.max_steps, "prices_version": PRICES_VERSION,
        "passed": passed, "answer": got, "expected": want,
        "hit_max_steps": bool(tool_calls),  # loop only exits with pending tool calls at the step limit
        "est_cost": total_cost, "steps": steps,
    }
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
