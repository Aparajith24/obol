"""
"Plan once, then execute" strategy.

    uv run planner.py --task hard-acme --model qwen2.5:7b
    uv run planner.py --task hard-acme --cascade        # 1.5B first, 7B if it visibly fails

Instead of one model call per tool call, the model is called ONCE and writes a
short Python program that uses the tools. Plain Python then runs that program:
tool calls cost no tokens, and nothing is re-sent. (Same idea as ReWOO /
LLMCompiler / programmatic tool calling; a program rather than a JSON graph
because the plan has to loop over results it hasn't seen yet.)

Escalation only uses failures we can DETECT without knowing the right answer:
no code, disallowed code, a crash, or no numeric `answer`. A plan that runs
cleanly but computes the wrong number is not caught - in the real world there
is no answer key to compare against.

SAFETY: the program is model output, i.e. untrusted. Before running it we walk
its syntax tree and reject anything outside a small allowlist (no imports, no
dunder access, no while-loops), run it with a tiny set of builtins, and kill
it after a few seconds. That's fine for a local experiment with fake data; it
is NOT a real sandbox.
"""

import argparse
import ast
import re
import signal
import time

import ollama

from agent import PRICES, PRICES_VERSION, is_correct, log_record, run_react, step_cost, usage_of
from tasks import TASKS_BY_ID
from world import get_order, get_unit_price, list_customers, list_orders

PLAN_PROMPT = """You answer questions about an order database by writing a short Python program.

These functions are available:
    list_customers() -> list[str]                    # IDs of all customers
    list_orders(customer: str) -> list[str]          # order IDs placed by a customer
    get_order(order_id: str) -> list[dict]           # line items, each {"sku": str, "qty": int}
    get_unit_price(sku: str) -> float                # unit price in dollars

Rules:
- Call the functions to look up everything you need; never guess IDs, skus or prices.
- Use plain Python only: variables, for-loops, if, arithmetic, lists, dicts. No imports, no while-loops.
- Store the final numeric result in a variable named `answer`.
- Reply with only the program, inside a ```python code block."""

ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Break, ast.Continue, ast.Pass,
    ast.FunctionDef, ast.Return, ast.arguments, ast.arg, ast.Lambda,
    ast.Name, ast.Constant, ast.Attribute, ast.Subscript, ast.Slice, ast.Call, ast.keyword,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context,
)
# No str.format: "{0.__class__}" would reach attributes the AST check never sees.
SAFE_METHODS = {"append", "extend", "get", "items", "keys", "values", "lower", "upper", "strip",
                "startswith", "endswith", "count", "split", "join"}
SAFE_BUILTINS = {
    "len": len, "sum": sum, "range": range, "round": round, "min": min, "max": max, "abs": abs,
    "any": any, "all": all,
    "float": float, "int": int, "str": str, "list": list, "dict": dict, "set": set,
    "sorted": sorted, "enumerate": enumerate, "zip": zip,
    "print": lambda *a, **k: None,  # models like to print; make it harmless
}
TIMEOUT_SECONDS = 5


class PlanError(Exception):
    """A failure we can see without an answer key - the escalation signal."""


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        raise PlanError("no code block in reply")
    return match.group(1)


def check_code(code: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise PlanError(f"syntax error: {e.msg}")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise PlanError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise PlanError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_METHODS:
            raise PlanError(f"disallowed attribute: .{node.attr}")
    return tree


def execute(code: str) -> tuple[float, int]:
    """Run the checked program. Returns (answer, number of tool calls it made)."""
    tree = check_code(code)
    calls = 0

    def counted(fn):
        def wrapper(*args, **kwargs):
            nonlocal calls
            calls += 1
            return fn(*args, **kwargs)
        return wrapper

    env = {"__builtins__": SAFE_BUILTINS,
           "list_customers": counted(list_customers),
           "list_orders": counted(list_orders),
           "get_order": counted(get_order),
           "get_unit_price": counted(get_unit_price)}

    def timeout(*_):
        raise PlanError(f"program ran longer than {TIMEOUT_SECONDS}s")

    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(TIMEOUT_SECONDS)
    try:
        exec(compile(tree, "<plan>", "exec"), env)
    except PlanError:
        raise
    except Exception as e:  # the plan's own bug: wrong key, bad ID, type error...
        raise PlanError(f"program crashed: {type(e).__name__}: {e}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    answer = env.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, (int, float)):
        raise PlanError(f"`answer` is not a number: {answer!r}")
    return float(answer), calls


def run_plan(task: dict, models: list[str], max_output_tokens=2048, verbose=True) -> dict:
    """Try each model in order; move to the next only on a detectable failure."""
    say = print if verbose else (lambda *a, **k: None)
    say(f"task={task['id']}  models={' -> '.join(models)}  prices={PRICES_VERSION} (hypothetical)\n")

    run_start = time.perf_counter()
    steps, total_cost = [], 0.0
    got, error = None, None

    for attempt, model in enumerate(models, 1):
        t0 = time.perf_counter()
        response = ollama.chat(model=model, options={"num_predict": max_output_tokens}, messages=[
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": task["question"]},
        ])
        u = usage_of(response, model, time.perf_counter() - t0)
        cost = step_cost(model, u["input_tokens"], u["output_tokens"])
        total_cost += cost

        say(f"── attempt {attempt}: {model} ── {u['seconds']:.2f}s  "
            f"in={u['input_tokens']} out={u['output_tokens']}  ${cost:.8f}")
        try:
            code = extract_code(response.message.content)
            say("   " + code.strip().replace("\n", "\n   "))
            got, u["tool_calls"] = execute(code)
            error = None
            say(f"   -> answer={got}  ({u['tool_calls']} tool calls, 0 tokens)")
        except PlanError as e:
            got, error, u["tool_calls"] = None, str(e), 0
            say(f"   !! {error}")
        u["error"] = error
        steps.append(u)
        if error is None:
            break  # ran cleanly: we have no way to know if it's wrong, so stop
        say()

    passed = is_correct(got, task["expected"])
    wall = time.perf_counter() - run_start
    say(f"\nanswer: got={got}  expected={task['expected']}  -> {'PASS' if passed else 'FAIL'}  "
        f"model calls={len(steps)}  est. cost=${total_cost:.8f}  wall={wall:.2f}s")

    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "strategy": "plan-cascade" if len(models) > 1 else "plan",
        "task": task["id"], "difficulty": task["difficulty"],
        "model": "+".join(models), "prices_version": PRICES_VERSION,
        "passed": passed, "answer": got, "expected": task["expected"],
        "error": error, "escalated": len(steps) > 1,
        "est_cost": total_cost, "seconds": round(wall, 3), "steps": steps,
    }


def run_hybrid(task: dict, plan_models: list[str], fallback_model: str, verbose=True) -> dict:
    """Plan first; if every plan fails DETECTABLY, fall back to step-by-step.

    The trigger is the same no-answer-key signal as the cascade. A plan that runs
    cleanly but is wrong still gets accepted.
    """
    plan = run_plan(task, plan_models, verbose=verbose)
    for s in plan["steps"]:
        s["phase"] = "plan"
    record = dict(plan, strategy="hybrid", model="+".join(plan_models) + ">" + fallback_model,
                  fell_back=False)
    if plan["error"] is None:
        return record

    if verbose:
        print(f"\n== plan failed detectably ({plan['error']}); falling back to step-by-step ==\n")
    react = run_react(task, fallback_model, verbose=verbose)
    for s in react["steps"]:
        s["phase"] = "react"
    record.update(passed=react["passed"], answer=react["answer"], fell_back=True,
                  est_cost=plan["est_cost"] + react["est_cost"],
                  seconds=round(plan["seconds"] + react["seconds"], 3),
                  steps=plan["steps"] + react["steps"])
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="hard-acme", choices=TASKS_BY_ID)
    parser.add_argument("--model", default="qwen2.5:7b", choices=PRICES)
    parser.add_argument("--cascade", action="store_true", help="try qwen2.5:1.5b first, then qwen2.5:7b")
    parser.add_argument("--log", default="runs/runs.jsonl")
    args = parser.parse_args()

    models = ["qwen2.5:1.5b", "qwen2.5:7b"] if args.cascade else [args.model]
    log_record(run_plan(TASKS_BY_ID[args.task], models), args.log)


if __name__ == "__main__":
    main()
