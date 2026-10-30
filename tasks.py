"""
The benchmark: questions about the order database, from easy to hard.

Each expected answer is computed from the data in plain Python, never by a
model. That's the evaluator.
"""

from world import DISCONTINUED, ORDERS, UNIT_PRICES


def _price(sku):
    return UNIT_PRICES[DISCONTINUED.get(sku, sku)]


def _order_value(order_id):
    return sum(i["qty"] * _price(i["sku"]) for i in ORDERS[order_id]["items"])


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
    # ---- complex: more logic over the same lookups -------------------------
    {
        "id": "complex-max", "difficulty": "complex",
        "question": "What is the value in dollars of the single most valuable order placed by 'acme'?",
        "expected": max(_order_value(oid) for oid, o in ORDERS.items() if o["customer"] == "acme"),
    },
    {
        "id": "complex-discount", "difficulty": "complex",
        "question": "Any single order worth more than $40 gets a 10% discount. "
                    "How much does 'acme' pay in total after discounts?",
        "expected": sum(v * 0.9 if v > 40 else v for v in
                        (_order_value(oid) for oid, o in ORDERS.items() if o["customer"] == "acme")),
    },
    {
        "id": "complex-skus", "difficulty": "complex",
        "question": "How many different product skus has 'acme' ordered?",
        "expected": len({i["sku"] for o in ORDERS.values() if o["customer"] == "acme" for i in o["items"]}),
    },
    # ---- surprise: something only visible once you see a tool result -------
    {
        # One sku's price lookup fails with "discontinued, use X instead".
        "id": "surprise-discontinued", "difficulty": "surprise",
        "question": "What is the total value in dollars of all orders placed by the customer 'hooli'?",
        "expected": _customer_value("hooli"),
    },
    {
        # The question's name doesn't match the customer ID; list_orders("Initech Inc.") is just [].
        "id": "surprise-name", "difficulty": "surprise",
        "question": "What is the total value in dollars of all orders placed by Initech Inc.?",
        "expected": _customer_value("initech"),
    },
    {
        # Everything at once: discover all customers, and hit the discontinued sku on the way.
        "id": "surprise-all", "difficulty": "surprise",
        "question": "What is the total value in dollars of every order from every customer?",
        "expected": sum(_order_value(oid) for oid in ORDERS),
    },
]
# ---- holdout: never used to tune prompts; run only to check the training result ----
HOLDOUT_TASKS = [
    {
        "id": "h-easy-washer", "difficulty": "easy",
        "question": "What is the unit price in dollars of the product 'washer'?",
        "expected": UNIT_PRICES["washer"],
    },
    {
        "id": "h-medium-order", "difficulty": "medium",
        "question": "What is the total value in dollars of order A-1003?",
        "expected": _order_value("A-1003"),
    },
    {
        "id": "h-medium-nuts", "difficulty": "medium",
        "question": "How many nuts in total has the customer 'initech' ordered?",
        "expected": sum(i["qty"] for o in ORDERS.values() if o["customer"] == "initech"
                        for i in o["items"] if i["sku"] == "nut"),
    },
    {
        "id": "h-complex-avg", "difficulty": "complex",
        "question": "What is the average value in dollars of an order placed by 'acme'?",
        "expected": _customer_value("acme") / len([o for o in ORDERS.values() if o["customer"] == "acme"]),
    },
    {
        "id": "h-complex-diff", "difficulty": "complex",
        "question": "How many dollars more has 'acme' spent in total than 'globex'?",
        "expected": _customer_value("acme") - _customer_value("globex"),
    },
    {
        "id": "h-complex-min", "difficulty": "complex",
        "question": "What is the value in dollars of the cheapest single order placed by 'acme'?",
        "expected": min(_order_value(oid) for oid, o in ORDERS.items() if o["customer"] == "acme"),
    },
    {
        "id": "h-complex-bolts", "difficulty": "complex",
        "question": "How many orders, across all customers, contain at least one bolt?",
        "expected": sum(any(i["sku"] == "bolt" for i in o["items"]) for o in ORDERS.values()),
    },
    {
        # Hits the discontinued sku via an order lookup rather than a customer total.
        "id": "h-surprise-order", "difficulty": "surprise",
        "question": "What is the total value in dollars of order A-1006?",
        "expected": _order_value("A-1006"),
    },
    {
        "id": "h-surprise-name", "difficulty": "surprise",
        "question": "What is the total value in dollars of all orders placed by Globex Corp.?",
        "expected": _customer_value("globex"),
    },
    {
        "id": "h-surprise-two", "difficulty": "surprise",
        "question": "What is the combined total value in dollars of all orders placed by "
                    "'hooli' and 'acme'?",
        "expected": _customer_value("hooli") + _customer_value("acme"),
    },
]
# ---- holdout v2: written 2026-10-30 BEFORE seeing how the trace checks score ----
# Holdout v1 was used to design the checks, so it is development data now.
_ACME = [oid for oid, o in ORDERS.items() if o["customer"] == "acme"]
_CUSTOMERS = sorted({o["customer"] for o in ORDERS.values()})


def _qty(sku, customer=None):
    return sum(i["qty"] for o in ORDERS.values() if customer in (None, o["customer"])
               for i in o["items"] if i["sku"] == sku)


def _t(id, difficulty, question, expected):
    return {"id": "v2-" + id, "difficulty": difficulty, "question": question, "expected": expected}


HOLDOUT_V2_TASKS = [
    _t("easy-nut-price", "easy", "What is the unit price in dollars of the product 'nut'?", UNIT_PRICES["nut"]),
    _t("easy-customers", "easy", "How many different customers are there?", len(_CUSTOMERS)),
    _t("easy-hooli-count", "easy", "How many orders has the customer 'hooli' placed?",
       sum(o["customer"] == "hooli" for o in ORDERS.values())),
    _t("easy-lines", "easy", "How many line items does order A-1001 have?", len(ORDERS["A-1001"]["items"])),

    _t("med-order1", "medium", "What is the total value in dollars of order A-1001?", _order_value("A-1001")),
    _t("med-order4", "medium", "What is the total value in dollars of order A-1004?", _order_value("A-1004")),
    _t("med-washers", "medium", "How many washers in total has the customer 'acme' ordered?", _qty("washer", "acme")),
    _t("med-brackets", "medium", "How many brackets in total has the customer 'globex' ordered?",
       _qty("bracket", "globex")),
    _t("med-initech", "medium", "What is the total value in dollars of all orders placed by the customer 'initech'?",
       _customer_value("initech")),
    _t("med-units", "medium", "How many units in total (all products) has the customer 'acme' ordered?",
       sum(i["qty"] for oid in _ACME for i in ORDERS[oid]["items"])),

    _t("cx-max-all", "complex", "What is the value in dollars of the single most valuable order, across all customers?",
       max(_order_value(oid) for oid in ORDERS)),
    _t("cx-top-customer", "complex", "How many dollars has the customer who spent the most spent in total?",
       max(_customer_value(c) for c in _CUSTOMERS)),
    _t("cx-small-orders", "complex", "How many orders, across all customers, are worth less than $20?",
       sum(_order_value(oid) < 20 for oid in ORDERS)),
    _t("cx-avg-all", "complex", "What is the average value in dollars of an order, across all orders?",
       sum(_order_value(oid) for oid in ORDERS) / len(ORDERS)),
    _t("cx-big-customers", "complex", "How many customers have placed at least one order worth more than $46?",
       sum(any(_order_value(oid) > 46 for oid, o in ORDERS.items() if o["customer"] == c) for c in _CUSTOMERS)),
    _t("cx-bolts-all", "complex", "How many bolts have been ordered in total, across all customers?", _qty("bolt")),
    _t("cx-acme-bolt-spend", "complex", "How many dollars has 'acme' spent on bolts only?",
       _qty("bolt", "acme") * _price("bolt")),
    _t("cx-price-diff", "complex", "How many dollars more does one bracket cost than one bolt?",
       _price("bracket") - _price("bolt")),
    _t("cx-distinct-skus", "complex", "How many different product skus appear across all orders?",
       len({i["sku"] for o in ORDERS.values() for i in o["items"]})),
    _t("cx-free-washers", "complex", "If washers were free, what would 'acme' have paid in total in dollars?",
       _customer_value("acme") - _qty("washer", "acme") * _price("washer")),

    _t("sp-order7", "surprise", "What is the total value in dollars of order A-1007?", _order_value("A-1007")),
    _t("sp-caps", "surprise", "What is the total value in dollars of all orders placed by ACME?",
       _customer_value("acme")),
    _t("sp-hooli-corp", "surprise", "What is the total value in dollars of all orders placed by Hooli Corporation?",
       _customer_value("hooli")),
    _t("sp-gear-spend", "surprise", "How many dollars has 'hooli' spent on gears?", _qty("gear", "hooli") * _price("gear")),
    _t("sp-no-hyphen", "surprise", "What is the total value in dollars of order A1002?", _order_value("A-1002")),
    _t("sp-two", "surprise", "What is the combined total value in dollars of all orders placed by 'initech' and 'hooli'?",
       _customer_value("initech") + _customer_value("hooli")),
    _t("sp-globex-inc", "surprise", "How many orders has Globex Inc placed?",
       sum(o["customer"] == "globex" for o in ORDERS.values())),
    _t("sp-lowercase", "surprise", "What is the total value in dollars of order a-1005?", _order_value("A-1005")),

    _t("edge-wayne", "edge", "How many orders has the customer 'wayne' placed?",
       sum(o["customer"] == "wayne" for o in ORDERS.values())),
    _t("edge-stark", "edge", "What is the total value in dollars of all orders placed by the customer 'stark'?",
       _customer_value("stark")),
]

for t in TASKS + HOLDOUT_TASKS + HOLDOUT_V2_TASKS:
    t["expected"] = round(float(t["expected"]), 2)

TASKS_BY_ID = {t["id"]: t for t in TASKS + HOLDOUT_TASKS + HOLDOUT_V2_TASKS}
SPLITS = {"train": TASKS, "holdout": HOLDOUT_TASKS, "holdout-v2": HOLDOUT_V2_TASKS}
