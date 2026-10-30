"""
The "world" the agent works in: a tiny fake order database and the tools that
read it. Plain Python; a model only ever sees it through tool results.
"""

import ast
import inspect
import operator

ORDERS = {
    "A-1001": {"customer": "acme",    "items": [{"sku": "bolt", "qty": 120}, {"sku": "nut", "qty": 300}]},
    "A-1002": {"customer": "globex",  "items": [{"sku": "bracket", "qty": 15}]},
    "A-1003": {"customer": "acme",    "items": [{"sku": "bracket", "qty": 8}, {"sku": "washer", "qty": 500}]},
    "A-1004": {"customer": "acme",    "items": [{"sku": "bolt", "qty": 40}]},
    "A-1005": {"customer": "initech", "items": [{"sku": "nut", "qty": 1000}]},
    "A-1006": {"customer": "hooli",   "items": [{"sku": "gear", "qty": 20}, {"sku": "bolt", "qty": 100}]},
    "A-1007": {"customer": "hooli",   "items": [{"sku": "gear", "qty": 5}]},
}
UNIT_PRICES = {"bolt": 0.25, "nut": 0.05, "washer": 0.02, "bracket": 3.40, "gear-v2": 1.10}
# A mid-task surprise: asking for a discontinued sku's price fails, and the error
# says which sku to use instead. Nothing in the tool descriptions warns about it.
DISCONTINUED = {"gear": "gear-v2"}


# ---------------------------------------------------------------------------
# Raw tools: return Python values, raise LookupError on bad input.
# ---------------------------------------------------------------------------
def list_customers() -> list[str]:
    return sorted({o["customer"] for o in ORDERS.values()})


def list_orders(customer: str) -> list[str]:
    return [oid for oid, o in ORDERS.items() if o["customer"] == customer]


def get_order(order_id: str) -> list[dict]:
    if order_id not in ORDERS:
        raise LookupError(f"no such order {order_id!r}")
    return [dict(item) for item in ORDERS[order_id]["items"]]


def get_unit_price(sku: str) -> float:
    if sku in DISCONTINUED:
        raise LookupError(f"sku {sku!r} is discontinued; use the price of {DISCONTINUED[sku]!r} instead")
    if sku not in UNIT_PRICES:
        raise LookupError(f"no such sku {sku!r}")
    return UNIT_PRICES[sku]


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


TOOL_FUNCTIONS = {
    "list_customers": list_customers,
    "list_orders": list_orders,
    "get_order": get_order,
    "get_unit_price": get_unit_price,
    "calculate": calculate,
}


def run_tool(name: str, args: dict, functions=None) -> str:
    """Tool call from a model: arguments are untrusted, result goes back as text."""
    fn = (functions or TOOL_FUNCTIONS).get(name)
    if fn is None:
        return f"error: unknown tool {name}"
    try:
        values = {k: str(v) for k, v in args.items()}
        if set(values) <= set(inspect.signature(fn).parameters):  # by name when the names fit
            return str(fn(**values))
        return str(fn(*values.values()))
    except (LookupError, ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
        return f"error: {e}"


# ---------------------------------------------------------------------------
# JSON schemas sent to the model in the step-by-step loop.
# Note: these definitions are re-sent (and counted as input) on EVERY call.
# ---------------------------------------------------------------------------
def _tool(name, description, param=None):
    params = [param] if param else []
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string"} for p in params},
                "required": params,
            },
        },
    }


TOOLS = [
    _tool("list_customers", "List the IDs of all customers."),
    _tool("list_orders", "List the order IDs placed by a customer.", "customer"),
    _tool("get_order", "Get the line items (sku and quantity) for one order.", "order_id"),
    _tool("get_unit_price", "Get the unit price in dollars for a product sku.", "sku"),
]
CALCULATOR = _tool("calculate", "Evaluate an arithmetic expression using + - * / and parentheses, "
                                "e.g. '120*0.25 + 300*0.05'.", "expression")

PLAN_DOC = """\
    list_customers() -> list[str]                    # IDs of all customers
    list_orders(customer: str) -> list[str]          # order IDs placed by a customer
    get_order(order_id: str) -> list[dict]           # line items, each {"sku": str, "qty": int}
    get_unit_price(sku: str) -> float                # unit price in dollars"""

WORLD = {
    "name": "orders",
    "about": "an order database",
    "functions": {f.__name__: f for f in (list_customers, list_orders, get_order, get_unit_price)},
    "react_functions": TOOL_FUNCTIONS,  # also has calculate, offered only with --calculator
    "plan_doc": PLAN_DOC,
    "never_guess": "IDs, skus or prices",
    "react_ids": "order IDs and skus",
    "schemas": TOOLS,
}
