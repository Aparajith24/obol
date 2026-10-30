"""
One way to call any chat model, so the strategies don't care who serves it.

A model is named "provider/model":

    ollama/qwen2.5:7b             local Ollama; OLLAMA_HOST, default http://localhost:11434
    openai/<model>                OPENAI_API_KEY
    openrouter/<vendor>/<model>   OPENROUTER_API_KEY
    anthropic/<model>             ANTHROPIC_API_KEY, e.g. anthropic/claude-haiku-5-5
    compat/<model>                any other OpenAI-compatible server (vLLM, LM Studio, Groq...):
                                  OBOL_BASE_URL and, if it needs one, OBOL_API_KEY

Keys come from the environment, or from a .env file in the working directory (gitignored).

Messages are plain {"role", "content"} dicts for system, user and assistant text. Tool
turns differ by provider, so they are built here: append assistant_turn(model, reply)
after each call and tool_results(model, ...) after running the tools.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

OPENAI_COMPATIBLE = {
    # provider: (base URL, env var holding the key, key to send when none is needed)
    "ollama":     (os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1", None, "ollama"),
    "openai":     ("https://api.openai.com/v1", "OPENAI_API_KEY", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None),
    "compat":     (os.environ.get("OBOL_BASE_URL"), "OBOL_API_KEY", "none"),
}
PROVIDERS = [*OPENAI_COMPATIBLE, "anthropic"]
MAX_TOKENS = 2048  # per-call output cap unless a caller sets one; thinking models may need more


def _load_dotenv(path=".env"):
    """KEY=value lines into os.environ, without overriding what's already set."""
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


@dataclass
class Reply:
    text: str
    tool_calls: list = field(default_factory=list)  # [{"id", "name", "args": dict}]
    input_tokens: int = 0     # everything the provider billed as input, cached or not
    output_tokens: int = 0    # includes any thinking tokens
    done_reason: str = "stop"  # "stop", "tool_calls", "length" (hit the output cap), "refusal"...
    raw: object = None        # the provider's own assistant message, sent back as-is


def split(model: str) -> tuple[str, str]:
    provider, sep, name = model.partition("/")
    if not sep or provider not in PROVIDERS:
        raise ValueError(f"model {model!r} should look like 'provider/model', provider one of {PROVIDERS}")
    return provider, name


_clients = {}


def _client(provider):
    if provider not in _clients:
        if provider == "anthropic":
            import anthropic
            _clients[provider] = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        else:
            from openai import OpenAI
            base_url, key_var, default_key = OPENAI_COMPATIBLE[provider]
            key = os.environ.get(key_var) if key_var else None
            if not base_url:
                raise RuntimeError("compat/ models need OBOL_BASE_URL")
            if not (key or default_key):
                raise RuntimeError(f"{provider}/ models need {key_var} (environment or .env)")
            _clients[provider] = OpenAI(base_url=base_url, api_key=key or default_key)
    return _clients[provider]


def chat(model: str, messages: list[dict], tools: list | None = None, max_tokens: int | None = None) -> Reply:
    """One model call. `tools` are OpenAI-style function schemas."""
    provider, name = split(model)
    max_tokens = max_tokens or MAX_TOKENS
    if provider == "anthropic":
        return _chat_anthropic(name, messages, tools, max_tokens)
    kwargs = {"max_completion_tokens" if provider == "openai" else "max_tokens": max_tokens}
    if tools:
        kwargs["tools"] = tools
    r = _client(provider).chat.completions.create(model=name, messages=messages, **kwargs)
    msg, usage = r.choices[0].message, r.usage
    calls = []
    for c in msg.tool_calls or []:
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed": c.function.arguments}  # the tool runner reports it as an error
        calls.append({"id": c.id, "name": c.function.name, "args": args if isinstance(args, dict) else {}})
    return Reply(text=msg.content or "", tool_calls=calls,
                 input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                 output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                 done_reason=r.choices[0].finish_reason or "stop", raw=msg)


_ANTHROPIC_STOP = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length"}


def _chat_anthropic(name, messages, tools, max_tokens):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    kwargs = {"system": system} if system else {}
    if tools:
        kwargs["tools"] = [{"name": t["function"]["name"], "description": t["function"]["description"],
                            "input_schema": t["function"]["parameters"]} for t in tools]
    r = _client("anthropic").messages.create(
        model=name, max_tokens=max_tokens, messages=[m for m in messages if m["role"] != "system"], **kwargs)
    u = r.usage
    return Reply(
        text="".join(b.text for b in r.content if b.type == "text"),
        tool_calls=[{"id": b.id, "name": b.name, "args": dict(b.input)} for b in r.content if b.type == "tool_use"],
        input_tokens=u.input_tokens + (getattr(u, "cache_creation_input_tokens", 0) or 0)
                     + (getattr(u, "cache_read_input_tokens", 0) or 0),
        output_tokens=u.output_tokens,
        done_reason=_ANTHROPIC_STOP.get(r.stop_reason, r.stop_reason or "stop"),
        raw=r.content)


def assistant_turn(model: str, reply: Reply) -> dict:
    """The model's reply, in the shape its provider wants it back in the history."""
    if split(model)[0] == "anthropic":
        return {"role": "assistant", "content": reply.raw}  # keeps tool_use and thinking blocks intact
    turn = {"role": "assistant", "content": reply.text}
    if reply.tool_calls:
        turn["tool_calls"] = [{"id": c["id"], "type": "function",
                               "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                              for c in reply.tool_calls]
    return turn


def tool_results(model: str, results: list[tuple[dict, str]]) -> list[dict]:
    """Messages carrying each (tool call, result text) back to the model."""
    if split(model)[0] == "anthropic":
        return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": c["id"], "content": text}
                                             for c, text in results]}]
    return [{"role": "tool", "tool_call_id": c["id"], "content": text} for c, text in results]
