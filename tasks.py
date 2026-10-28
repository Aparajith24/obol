"""
The benchmark: questions about the order database, from easy to hard.

Each expected answer is computed from the data in plain Python, never by a
model. That's the evaluator.
"""

from world import ORDERS, UNIT_PRICES


def _order_value(order_id):
    return sum(i["qty"] * UNIT_PRICES[i["sku"]] for i in ORDERS[order_id]["items"])


def _customer_value(customer):
    return sum(_order_value(oid) for oid, o in ORDERS.items() if o["customer"] == customer)


TASKS = [
    {
        "id": "easy-count", "difficulty": "easy",
        "question": "How many orders has the customer 'acme' placed?",
        "expected": sum(o["customer"] == "acme" for o in ORDERS.values()),
    },
    {
        "id": "easy-price", "difficulty": "easy",
        "question": "What is the unit price in dollars of the product 'bracket'?",
        "expected": UNIT_PRICES["bracket"],
    },
    {
        "id": "medium-order", "difficulty": "medium",
        "question": "What is the total value in dollars of order A-1002?",
        "expected": _order_value("A-1002"),
    },
    {
        "id": "medium-qty", "difficulty": "medium",
        "question": "How many bolts in total has the customer 'acme' ordered across all its orders?",
        "expected": sum(i["qty"] for o in ORDERS.values() if o["customer"] == "acme"
                        for i in o["items"] if i["sku"] == "bolt"),
    },
    {
        "id": "hard-acme", "difficulty": "hard",
        "question": "What is the total value in dollars of all orders placed by the customer 'acme'?",
        "expected": _customer_value("acme"),
    },
    {
        "id": "hard-two", "difficulty": "hard",
        "question": "What is the combined total value in dollars of all orders placed by "
                    "the customers 'globex' and 'initech'?",
        "expected": _customer_value("globex") + _customer_value("initech"),
    },
    {
        # Surprise: the customer has no orders. Tests handling an empty result.
        "id": "edge-none", "difficulty": "edge",
        "question": "What is the total value in dollars of all orders placed by the customer 'umbrella'?",
        "expected": _customer_value("umbrella"),
    },
]
for t in TASKS:
    t["expected"] = round(float(t["expected"]), 2)

TASKS_BY_ID = {t["id"]: t for t in TASKS}
