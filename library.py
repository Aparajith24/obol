"""
A second world: a small lending library. Written for holdout v3 to test whether the
checks and the ladder carry over to tools they were not designed on.

Different shapes from the order world on purpose, the way real APIs differ:
members are found by a name search, loans come back in pages of 3, the fee policy
is data the program has to apply (grace days, a cap), and one tier has no policy at all.
"""

MEMBERS = {
    "m01": {"name": "Alice Wong",  "tier": "standard"},
    "m02": {"name": "Bilal Khan",  "tier": "student"},
    "m03": {"name": "Chen Li",     "tier": "senior"},
    "m04": {"name": "Dana Ruiz",   "tier": "standard"},
    "m05": {"name": "Erik Olsen",  "tier": "student"},
    "m06": {"name": "Fatima Noor", "tier": "standard"},  # a member with no loans
}
BOOKS = {
    "B-201": {"title": "Dune",                "genre": "scifi",    "loan_days": 21},
    "B-202": {"title": "Emma",                "genre": "classics", "loan_days": 21},
    "B-203": {"title": "Neuromancer",         "genre": "scifi",    "loan_days": 14},
    "B-204": {"title": "Python Crash Course", "genre": "tech",     "loan_days": 7},
    "B-205": {"title": "The Hobbit",          "genre": "fantasy",  "loan_days": 21},
    "B-206": {"title": "Clean Code",          "genre": "tech",     "loan_days": 7},
}
LOANS = {
    "L-01": {"member": "m01", "book": "B-201", "days_out": 30, "returned": True},
    "L-02": {"member": "m01", "book": "B-204", "days_out": 10, "returned": False},
    "L-03": {"member": "m01", "book": "B-206", "days_out": 5,  "returned": False},
    "L-04": {"member": "m01", "book": "B-203", "days_out": 16, "returned": True},
    "L-05": {"member": "m02", "book": "B-204", "days_out": 12, "returned": False},
    "L-06": {"member": "m02", "book": "B-205", "days_out": 40, "returned": False},
    "L-07": {"member": "m03", "book": "B-202", "days_out": 25, "returned": True},
    "L-08": {"member": "m03", "book": "B-203", "days_out": 20, "returned": False},
    "L-09": {"member": "m04", "book": "B-205", "days_out": 3,  "returned": False},
    "L-10": {"member": "m05", "book": "B-206", "days_out": 9,  "returned": True},
    "L-11": {"member": "m05", "book": "B-201", "days_out": 25, "returned": False},
    "L-12": {"member": "m05", "book": "B-203", "days_out": 14, "returned": False},
    "L-13": {"member": "m05", "book": "B-202", "days_out": 50, "returned": False},
}
FEE_POLICIES = {
    "standard": {"per_day": 0.25, "grace_days": 2, "cap": 5.00},
    "student":  {"per_day": 0.10, "grace_days": 3, "cap": 2.00},
    # "senior" has no policy: asking for it fails, and the error says why.
}
PAGE_SIZE = 3


def list_members() -> list[str]:
    return sorted(MEMBERS)


def find_member(name: str) -> list[str]:
    return [mid for mid, m in MEMBERS.items() if name.lower() in m["name"].lower()]


def get_member(member_id: str) -> dict:
    if member_id not in MEMBERS:
        raise LookupError(f"no such member {member_id!r}")
    return dict(MEMBERS[member_id])


def list_loans(member_id: str, page=1) -> dict:
    page = int(page)
    ids = [lid for lid, l in LOANS.items() if l["member"] == member_id]
    chunk = ids[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    return {"loans": chunk, "next_page": page + 1 if page * PAGE_SIZE < len(ids) else None}


def get_loan(loan_id: str) -> dict:
    if loan_id not in LOANS:
        raise LookupError(f"no such loan {loan_id!r}")
    return dict(LOANS[loan_id])


def list_books() -> list[str]:
    return sorted(BOOKS)


def get_book(book_id: str) -> dict:
    if book_id not in BOOKS:
        raise LookupError(f"no such book {book_id!r}")
    return dict(BOOKS[book_id])


def get_fee_policy(tier: str) -> dict:
    if tier == "senior":
        raise LookupError("tier 'senior' has no fee policy: senior members are never charged late fees")
    if tier not in FEE_POLICIES:
        raise LookupError(f"no such tier {tier!r}")
    return dict(FEE_POLICIES[tier])


FUNCTIONS = {f.__name__: f for f in (list_members, find_member, get_member, list_loans, get_loan,
                                     list_books, get_book, get_fee_policy)}

FEE_RULE = ("late fee for one loan = per_day * max(0, days overdue - grace_days), at most cap; "
            "days overdue = days_out - the book's loan_days")

PLAN_DOC = f"""\
    list_members() -> list[str]                        # IDs of all members
    find_member(name: str) -> list[str]                # IDs of members whose name contains `name` (any case)
    get_member(member_id: str) -> dict                 # {{"name": str, "tier": str}}
    list_loans(member_id: str, page: int = 1) -> dict  # {{"loans": [loan IDs, at most 3], "next_page": int or None}}
    get_loan(loan_id: str) -> dict                     # {{"member": str, "book": str, "days_out": int, "returned": bool}}
    list_books() -> list[str]                          # IDs of all books in the catalogue
    get_book(book_id: str) -> dict                     # {{"title": str, "genre": str, "loan_days": int}}
    get_fee_policy(tier: str) -> dict                  # {{"per_day": float, "grace_days": int, "cap": float}}
                                                       # {FEE_RULE}"""


def _tool(name, description, required=(), optional=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object",
                       "properties": {p: {"type": "string"} for p in (*required, *optional)},
                       "required": list(required)}}}


SCHEMAS = [
    _tool("list_members", "List the IDs of all members."),
    _tool("find_member", "Find the IDs of members whose name contains the given text (any case).", ["name"]),
    _tool("get_member", "Get a member's name and tier.", ["member_id"]),
    _tool("list_loans", "List a member's loan IDs, at most 3 per page. Returns the loans and next_page "
                        "(the page number to ask for next, or None when there are no more).", ["member_id"], ["page"]),
    _tool("get_loan", "Get one loan: member ID, book ID, days_out (days the book has been or was out) "
                      "and whether it was returned.", ["loan_id"]),
    _tool("list_books", "List the IDs of all books in the catalogue."),
    _tool("get_book", "Get a book's title, genre and loan period in days (loan_days).", ["book_id"]),
    _tool("get_fee_policy", f"Get the late-fee policy for a member tier: per_day, grace_days, cap. {FEE_RULE}.",
          ["tier"]),
]

WORLD = {
    "name": "library",
    "about": "a lending library's database",
    "functions": FUNCTIONS,          # what plans (and the checks) can call
    "react_functions": FUNCTIONS,    # what the step-by-step agent can call
    "plan_doc": PLAN_DOC,
    "never_guess": "IDs, names or fees",
    "react_ids": "member IDs, loan IDs and book IDs",
    "schemas": SCHEMAS,
}
