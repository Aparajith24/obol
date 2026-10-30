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
no code, disallowed code, a crash, or no numeric `answer`. With checks=True, a
clean run is also rejected when a trace check (checks.py) finds it suspicious.
With feedback=True, the next rung is told why the previous attempt was rejected.
With repairs=N, a rejected plan first goes back to the SAME model up to N times,
with its own program and the reason, before moving to the next rung.

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

from checks import run_checks
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
PLAN_TOOLS = {f.__name__: f for f in (list_customers, list_orders, get_order, get_unit_price)}


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


def _short(value, limit=200):
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def execute(code: str) -> tuple[float, dict]:
    """Run the checked program. Returns (answer, trace).

    The trace records every tool call (args, result or error, duration) and the
    program's variables at the end, so a wrong answer can be diagnosed later.
    On a crash the partial trace rides along on the PlanError.
    """
    tree = check_code(code)
    calls = []

    def traced(fn):
        def wrapper(*args, **kwargs):
            entry = {"tool": fn.__name__, "args": [_short(a) for a in args] + [f"{k}={_short(v)}" for k, v in kwargs.items()]}
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                entry["result"] = _short(result)
                return result
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"
                raise
            finally:
                entry["ms"] = round((time.perf_counter() - t0) * 1000, 3)
                calls.append(entry)
        return wrapper

    tools = {name: traced(f) for name, f in PLAN_TOOLS.items()}
    env = {"__builtins__": SAFE_BUILTINS, **tools}

    def final_vars():
        return {k: _short(v) for k, v in env.items()
                if k != "__builtins__" and k not in tools and not callable(v)}

    def timeout(*_):
        raise PlanError(f"program ran longer than {TIMEOUT_SECONDS}s")

    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(TIMEOUT_SECONDS)
    try:
        exec(compile(tree, "<plan>", "exec"), env)
    except PlanError as e:
        e.trace = {"tool_trace": calls, "final_vars": final_vars()}
        raise
    except Exception as e:  # the plan's own bug: wrong key, bad ID, type error...
        err = PlanError(f"program crashed: {type(e).__name__}: {e}")
        err.trace = {"tool_trace": calls, "final_vars": final_vars()}
        raise err
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    trace = {"tool_trace": calls, "final_vars": final_vars()}
    answer = env.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, (int, float)):
        err = PlanError(f"`answer` is not a number: {answer!r}")
        err.trace = trace
        raise err
    return float(answer), trace


def feedback_note(error: str, tool_trace: list) -> str:
    """What the next rung is told about a rejected attempt: the reason and the tool calls."""
    calls = [f"  {c['tool']}({', '.join(c['args'])}) -> {c.get('result', c.get('error', ''))[:80]}"
             for c in tool_trace[:8]]
    return ("\n\nNote: a previous attempt at this question was rejected.\n"
            f"Reason: {error}\n"
            + ("Tool calls it made:\n" + "\n".join(calls) + "\n" if calls else "")
            + "Don't repeat that mistake; look up valid IDs and values first if needed.")


REPAIR_PROMPT = """Your program was rejected.
Why: {reason}
{calls}
Write a corrected program that avoids this mistake. Same rules as before: reply with only
the program, inside a ```python code block."""


def repair_message(error: str, tool_trace: list) -> str:
    calls = [f"  {c['tool']}({', '.join(c['args'])}) -> {c.get('result', c.get('error', ''))[:80]}"
             for c in tool_trace[:8]]
    return REPAIR_PROMPT.format(reason=error, calls="Tool calls it made:\n" + "\n".join(calls) if calls else "")


def run_plan(task: dict, models: list[str], checks=False, feedback=False, repairs=0,
             max_output_tokens=2048, verbose=True) -> dict:
    """Try each model in order; move to the next only on a detectable failure."""
    say = print if verbose else (lambda *a, **k: None)
    say(f"task={task['id']}  models={' -> '.join(models)}  prices={PRICES_VERSION} (hypothetical)\n")

    run_start = time.perf_counter()
    steps, total_cost = [], 0.0
    got, error, note = None, None, ""
    flagged_answer = None  # a clean but suspicious answer: better than nothing if every rung fails

    schedule = [(model, r) for model in models for r in range(1 + repairs)]
    for attempt, (model, repair) in enumerate(schedule, 1):
        messages = [{"role": "system", "content": PLAN_PROMPT},
                    {"role": "user", "content": task["question"] + note}]
        if repair:  # same model, sees its own rejected reply and why: one cheap call
            messages += [{"role": "assistant", "content": rejected_reply},
                         {"role": "user", "content": repair_message(error, trace["tool_trace"])}]
        t0 = time.perf_counter()
        response = ollama.chat(model=model, options={"num_predict": max_output_tokens}, messages=messages)
        u = usage_of(response, model, time.perf_counter() - t0)
        cost = step_cost(model, u["input_tokens"], u["output_tokens"])
        total_cost += cost

        u["repair"] = bool(repair)
        say(f"── attempt {attempt}: {model}{' (repair)' if repair else ''} ── {u['seconds']:.2f}s  "
            f"in={u['input_tokens']} out={u['output_tokens']}  ${cost:.8f}")
        u["code"] = None
        try:
            code = u["code"] = extract_code(response.message.content)
            say("   " + code.strip().replace("\n", "\n   "))
            got, trace = execute(code)
            error = None
            say(f"   -> answer={got}  ({len(trace['tool_trace'])} tool calls, 0 tokens)")
            if checks:
                u["flags"] = run_checks(code, trace["tool_trace"], PLAN_TOOLS, task["question"])
                if u["flags"]:
                    flagged_answer, got = got, None
                    error = "check failed: " + "; ".join(reason for _, reason in u["flags"])
                    say(f"   ?? {error}")
        except PlanError as e:
            got, error, trace = None, str(e), getattr(e, "trace", {"tool_trace": [], "final_vars": {}})
            if u["code"] is None:
                u["reply"] = response.message.content[:2000]  # no code block: keep what it said instead
            say(f"   !! {error}")
        u.update(trace, tool_calls=len(trace["tool_trace"]), error=error)
        steps.append(u)
        if error is None:
            break  # ran cleanly and (if enabled) passed the checks: accept it
        rejected_reply = response.message.content
        if feedback and repair == repairs:  # leaving this model: tell the next rung
            note = feedback_note(error, trace["tool_trace"])
        say()

    if got is None and flagged_answer is not None:
        got = flagged_answer
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
        "error": error, "escalated": len(steps) > 1, "checks": checks, "feedback": feedback,
        "repairs": repairs,
        "feedback_note": note,
        "est_cost": total_cost, "seconds": round(wall, 3), "steps": steps,
    }


def run_hybrid(task: dict, plan_models: list[str], fallback_model: str,
               checks=False, feedback=False, repairs=0, verbose=True) -> dict:
    """Plan first; if every plan fails DETECTABLY, fall back to step-by-step.

    The trigger is the same no-answer-key signal as the cascade. A plan that runs
    cleanly but is wrong still gets accepted.
    """
    plan = run_plan(task, plan_models, checks=checks, feedback=feedback, repairs=repairs, verbose=verbose)
    for s in plan["steps"]:
        s["phase"] = "plan"
    record = dict(plan, strategy="hybrid", model="+".join(plan_models) + ">" + fallback_model,
                  fell_back=False)
    if plan["error"] is None:
        return record

    if verbose:
        print(f"\n== plan failed detectably ({plan['error']}); falling back to step-by-step ==\n")
    # With feedback the step-by-step agent sees why the plans were rejected; it's graded on the same answer.
    react = run_react(dict(task, question=task["question"] + plan["feedback_note"]), fallback_model, verbose=verbose)
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
