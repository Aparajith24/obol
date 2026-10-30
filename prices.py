"""
Prices in USD per 1M tokens, used to turn token counts into dollars.

Every cost Obol prints is an ESTIMATE: tokens counted by the provider times the price
here. Add your own models here, or pass --price MODEL=INPUT,OUTPUT on the command line.
A model with no price stops the run instead of silently costing $0.
"""

PRICES_VERSION = "v3"
PRICES = {
    # Local models are free to run. These made-up prices stand in for "what a hosted
    # provider might charge" so local experiments can practise the accounting.
    "ollama/qwen2.5:1.5b": {"input": 0.05, "output": 0.10, "source": "hypothetical"},
    "ollama/qwen2.5:7b":   {"input": 0.25, "output": 0.50, "source": "hypothetical"},
    # List prices as of 2026-10-30; check the provider's pricing page before relying on them.
    "anthropic/claude-haiku-5-5":  {"input": 0.10, "output": 0.50, "source": "list price, prompts <= 100K tokens"},
    "anthropic/claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "source": "list price"},
    "anthropic/claude-opus-5-5":   {"input": 4.00, "output": 20.00, "source": "list price"},
}


def price_of(model: str) -> dict:
    """Logs from before v3 name Ollama models without the provider: 'qwen2.5:7b'."""
    p = PRICES.get(model) or PRICES.get("ollama/" + model)
    if p is None:
        raise KeyError(f"no price for {model!r}: add it to prices.py or pass --price {model}=INPUT,OUTPUT")
    return p


def set_price(spec: str):
    """'provider/model=0.10,0.50' (USD per 1M input, output tokens)."""
    model, _, nums = spec.rpartition("=")
    try:
        inp, out = (float(x) for x in nums.split(","))
    except ValueError:
        raise ValueError(f"--price wants MODEL=INPUT,OUTPUT, got {spec!r}")
    PRICES[model] = {"input": inp, "output": out, "source": "--price"}


def step_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = price_of(model)
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
