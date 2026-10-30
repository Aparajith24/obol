# Obol

**Measure what a tool-using agent costs per *successful* task, and test cheaper ways to run it against a strong baseline, with any model.**

Most agent cost reports give dollars per call or per run. That hides the number that matters: a cheap agent that's wrong half the time isn't cheap. Obol runs the same tasks several ways and reports:
- pass rate and cost per task
- **cost per success**
- a paired comparison against a strong baseline, with confidence intervals

The answers are checked in plain Python, never by a model.

It's a small, readable research harness, not a framework. Each file is a few hundred lines you can read in one sitting.

> An obol was the small coin paid to the ferryman: the cost of getting across.

## What's in the box

- **Strategies.** These are ways to run the same agent:
  - classic step-by-step tool loops (ReAct)
  - plan-once, where the model writes one small program and plain Python runs the tools
  - a cheap-first ladder that climbs to a stronger model only when it can see that the cheaper attempt failed
- **Trace checks.** These flag a plan that ran cleanly but probably computed the wrong thing, without an answer key. Examples: an empty lookup on a suspicious input, or a comparison against a value no tool ever returned.
- **Two small test worlds** with 83 tasks and answers computed from the data:
  - an order database
  - a lending library
- **Honest evaluation:**
  - always-cheap and always-strong baselines
  - a random-mix baseline and the cost-quality frontier
  - an oracle ceiling
  - paired bootstrap confidence intervals over tasks
- **Any model:** local Ollama, OpenAI, Anthropic, OpenRouter, or any OpenAI-compatible server.

## Quickstart (local and free)

You need [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
ollama pull qwen2.5:1.5b && ollama pull qwen2.5:7b
uv sync

uv run agent.py --task hard-acme                      # one step-by-step run; prints every step's tokens and cost
uv run bench.py --split train --repeats 1             # the default strategies on 13 tasks
uv run curve.py runs/bench.jsonl                      # compare them
```

## Use your own models

Models are named `provider/model`:

| Provider | Example | Needs |
|---|---|---|
| `ollama` | `ollama/qwen2.5:7b` | Ollama running (`OLLAMA_HOST` if it isn't on localhost) |
| `anthropic` | `anthropic/claude-haiku-5-5` | `ANTHROPIC_API_KEY` |
| `openai` | `openai/<model>` | `OPENAI_API_KEY` |
| `openrouter` | `openrouter/<vendor>/<model>` | `OPENROUTER_API_KEY` |
| `compat` | `compat/<model>` | any OpenAI-compatible server (vLLM, LM Studio, Groq...): `OBOL_BASE_URL`, optional `OBOL_API_KEY` |

Put keys in your environment or in a `.env` file in the project folder. It's gitignored; never commit keys.

```bash
uv run bench.py --cheap anthropic/claude-haiku-5-5 --strong anthropic/claude-sonnet-5-5 \
    --split library --repeats 2 --budget 5 --log runs/claude.jsonl
uv run curve.py runs/claude.jsonl
```

**Costs are estimates.** Tokens are counted by the provider and multiplied by the table in [`prices.py`](prices.py). The local Qwen prices there are made up so local runs can practise the accounting. A model that isn't in the table stops the run instead of silently costing $0. Add it to `prices.py`, or pass `--price provider/model=INPUT,OUTPUT` (USD per 1M tokens).

**`--budget` is a hard stop.** The benchmark stops once estimated spend passes it. Set one on any paid run: a step-by-step agent on a hard task can make 20 calls.

**Models that think before answering** use output tokens for it. If replies come back cut off (`done_reason=length` in the log), raise the per-call cap with `--max-output-tokens`, which defaults to 2048.

**Tested so far:** only Ollama has been run end to end. The Anthropic backend has been checked with a fake client, but not yet against the live API. The OpenAI-compatible backend is the same one Ollama uses.

## Strategies

| Name | What it does |
|---|---|
| `react-cheap` / `react-strong` | Step-by-step tool loop with one model. These are the always-cheap and always-strong baselines. |
| `plan-cheap` / `plan-strong` | The model writes one Python program that calls the tools, run in a restricted interpreter. |
| `plan-cascade` | Cheap plan; a strong plan if it **visibly** fails (no code, crash, no numeric answer). |
| `cascade>react` | Cheap plan, then strong plan, then strong step-by-step, climbing only on visible failures. |
| `+checks` | Also climbs when a clean plan fails a trace check. |
| `+checks+fb` | Also tells the next rung why the last attempt was rejected. |
| `+checks+repair` | Before climbing, the same model gets one try to fix its rejected plan. |
| `+checks+repair-strong` | Repair only on the strong model, and only after a failed check. |

The default run is `plan-cheap`, `cascade>react`, `cascade>react+checks` and `react-strong`. Use `--strategies` to pick others, and `--tasks` to run a few task IDs.

## Reading the results

`curve.py` prints one row per strategy:

- **pass, 95% CI**: the share of runs with the right answer (Wilson interval).
- **$ / task, % strong $**: average estimated cost, and the same as a share of the strong baseline's.
- **random@same$, lift**: randomly mixing the cheap and strong baselines gives a straight line between them. `random@same$` is that line's pass rate at this strategy's cost, and `lift` is how far above it the strategy sits. A strategy below the line is beaten by a coin flip.
- **frontier**: `*` if the strategy is on the cost-quality frontier (upper convex hull).
- **oracle router**: the best strategy per task, picked with the answer key. It isn't achievable; it's the ceiling for a perfect router.
- **paired difference vs strong**: per-task pass-rate differences with a bootstrap CI over **tasks**. This is the comparison to trust. With 30 tasks the interval is roughly ±15 points, so small differences are noise.

`report.py` breaks pass rates down by task and difficulty. `checks.py` scores the trace checks against logged runs. `breakdown.py` splits a step-by-step agent's tokens into fixed prompt, re-sent history, and new content.

## What we found

The full record, including the results that went against us, is in **[EXPERIMENTS.md](EXPERIMENTS.md)**. In short, with two small local models:

- **Most of a step-by-step agent's bill is re-sending the conversation.** Only ~3% of input tokens were new information.
- **Plan-once is 20-60× cheaper**, but it breaks on surprises it can't see coming. Most of its failures are *silent*: the program runs fine and returns a wrong number.
- **The cheap-first ladder matched the strong agent's quality, within noise, at 42-72% of its cost on two of three unseen holdout sets, in two different worlds.** On the first holdout it lost by 20 points. It never met our own predeclared win bar: one holdout lost on quality, one on cost, one was inconclusive.
- **Hand-written trace checks worked in the world they were designed on and did not transfer to a new one.** In the library world, plans silently ignored pagination, and the checks never looked for that.

## Add your own world

A world is a dict of plain Python functions plus a short description of them. See [`library.py`](library.py) for a complete one:
- `FUNCTIONS`: what plans can call.
- `PLAN_DOC`: one signature line per function, shown to the planner.
- `SCHEMAS`: JSON tool schemas for the step-by-step agent.

Register it in [`worlds.py`](worlds.py). Then add tasks in [`tasks.py`](tasks.py) with `"world": "<name>"` and an `expected` answer **computed from your data in Python**, and add them to `SPLITS`.

Tasks should have numeric answers; that's what makes grading objective.

## Safety

Plan strategies run **model-written Python**. It goes through an AST allowlist with no imports and no dunder access, a handful of safe builtins, and a 5-second timeout. **That is not a sandbox.** Use read-only tools only, and don't point it at anything you can't afford to have called with arbitrary arguments.

## Layout

| File | What it is |
|---|---|
| `agent.py` | step-by-step tool loop |
| `planner.py` | plan-once, the cheap-first ladder, repair |
| `checks.py` | trace checks, and scoring them against logs |
| `llm.py` | one call interface for every provider |
| `prices.py` | price table and cost estimates |
| `world.py`, `library.py`, `worlds.py` | the test worlds |
| `tasks.py` | tasks and answer keys, grouped into splits |
| `bench.py` | runs strategies × tasks × repeats into a JSONL log |
| `curve.py`, `report.py`, `breakdown.py` | analysis |

## Limitations

- Toy worlds: small read-only databases with numeric answers. Real agents have large tool outputs, write actions and fuzzy goals.
- The experiments in EXPERIMENTS.md used two small local models and made-up prices. With frontier models the ratios may be quite different. Run it and find out.
- 30-task holdouts give wide confidence intervals.

## License

MIT, see [LICENSE](LICENSE).
