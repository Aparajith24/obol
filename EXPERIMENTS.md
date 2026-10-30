# Obol experiment log

Obol is a small runtime for tool-using agents. It aims to finish each task **the cheapest way
that still gets the right answer**, and to report **cost per successful task**, not cost per call.

This file is the full record of how we got here, including the results that went against us.
Each experiment lists its setup, its numbers and what we took from it. Commit hashes point at
the code that produced each result.

---

## TL;DR

- **Most of an agent's bill is re-sending the conversation.** Step-by-step agents re-send the
  whole history on every call, so cost grows quadratically with the number of steps. In our
  runs, only ~3% of input tokens were new information.
- **Plan-once is 20-60x cheaper.** The model writes one small program and plain Python runs the
  tools. But it breaks on surprises it can't see in advance.
- **A cheap-first ladder looked great on our own tasks and lost on unseen ones.** The ladder is
  cheap plan -> strong plan -> strong step-by-step. Most of its failures were **silent**: the
  program ran fine and returned a wrong number, so nothing escalated.
- **Generic trace checks catch most silent failures without an answer key.** The checks are
  "an empty lookup on a suspicious input" and "a comparison against a value the data never
  contained". With them, the ladder **matched or beat the strong step-by-step agent on unseen
  tasks at 1.5x lower cost per success**. It still missed our predeclared cost bar.
- **The open problem is escalation, not detection.** Escalating a caught failure to an
  expensive rung that also fails burns money.
- **Letting the strong model repair its own rejected plan helps.** Given the check's reason, it
  fixed 10 of 12. That cut cost per task by 28% with no loss in pass rate (development data).
  Repairs by the small model almost never worked, and a few turned crashes into silent wrong
  answers.

---

## Setup

| | |
|---|---|
| Models | Local, via Ollama: `qwen2.5:1.5b` ("cheap") and `qwen2.5:7b` ("strong") |
| Prices | **Hypothetical** table `hypothetical-v2`: cheap $0.05 / $0.10, strong $0.25 / $0.50 per 1M input / output tokens. Token counts are real (Ollama's `prompt_eval_count` / `eval_count`); the dollar figures are only as meaningful as this table. |
| World | A fake order database (`world.py`): 7 orders, 4 customers, 5 products, plus one discontinued product whose price lookup fails with "use gear-v2 instead". |
| Tools | `list_customers()`, `list_orders(customer)`, `get_order(order_id)`, `get_unit_price(sku)` |
| Evaluator | Every task's correct answer is computed from the data in plain Python (`tasks.py`), never by a model. Pass = the numeric answer is within 0.01. |

### Strategies

| Name | What it does |
|---|---|
| `react-cheap` / `react-strong` | Classic step-by-step tool loop with one model: one tool call, see the result, decide the next. These are the always-cheap and always-strong baselines. |
| `plan-cheap` / `plan-strong` | The model writes one Python program that calls the tools. It runs in a restricted interpreter: AST allowlist, no imports, a few safe builtins, 5s timeout. This is not a real sandbox. |
| `plan-cascade` | Cheap plan; if it **visibly** fails (no code, crash, no numeric answer), a strong plan. |
| `cascade>react` | Cheap plan -> strong plan -> strong step-by-step, climbing only on visible failures. |
| `+checks` | Also climbs when a clean plan fails a trace check (`checks.py`). |
| `+checks+fb` | Also tells the next rung why the previous attempt was rejected. |
| `+checks+repair` | Before climbing, the same model gets its rejected plan and the reason, and one try to fix it. |

### Evaluation method

These were adopted along the way, mostly borrowed from RouteLLM:

- **Baselines:** always-cheap and always-strong.
- **Random baseline:** randomly mixing two strategies gives every point on the straight line
  between them. A strategy has to sit *above* that line to be worth anything.
- **Cost-quality frontier:** the upper convex hull of (cost per task, pass rate). Anything below
  it is beaten by a random mix of strategies on it.
- **Oracle:** pick the best strategy per task using the answer key. This isn't achievable; it's
  the ceiling for a perfect per-task router.
- **Paired comparison:** per-task pass-rate differences against the strong baseline, with a 95%
  bootstrap confidence interval that resamples **tasks**, not runs.
- **Holdout sets with win criteria written before the run.** Prompts and checks are frozen
  before a holdout is run. Once a holdout's failures have been looked at, it becomes
  development data.
- **The answer key only grades detectors.** It is never used inside them.

---

## Experiments

### 1. A hand-written agent loop and where the tokens go
`c2ca76a`, `a9c81af` · one order-total task, 10 runs per setting

- **No instructions:** both models went 0/10. All the spend was wasted, while each individual
  call looked cheap.
- **One line of sequencing rules** took the 7B from 0/5 to 4/5. The harness mattered more than
  model choice.
- **Step cap:** every 7B success landed exactly on the 10-step cap. With the cap raised to 20,
  it went 9/10. A badly set budget creates failures that look like model failures.
- **Adding a calculator tool** made things worse: 0/5 at cap 10 and 2/5 at cap 20. More tools
  mean more steps and more context.
- **Token breakdown** for the 7B with rules: fixed prefix 43%, re-sent history 37%, the model's
  own replies echoed back ~8%, new tool results 3%, output 9.5% of tokens (17% of dollars).
  Failed runs were 7% of spend.
- **Quadratic growth checks out:** total input ≈ nP + d·n(n−1)/2. With P≈297 and d≈78 that
  predicts 6,470 input tokens; we measured 6,243.

### 2. Plan-once on a 7-task benchmark
`87dc434`, `a1721e4`, `d4a8886` · 7 tasks × 3 repeats

| Strategy | Pass | $ per success |
|---|---|---|
| react-strong | 18/21 | $0.000748 |
| plan-cascade | 20/21 | $0.0000375 |

That's ~20x cheaper and ~6x faster at the same or better pass rate. 7 plan runs ran cleanly
but returned a wrong number, and escalation can't see those.

### 3. Harder tasks and the cheap-first ladder
`1b2b848`, `f69337a` · 13 tasks (easy, medium, hard, edge, complex logic, mid-task surprises) × 3 repeats

| Strategy | Pass | $ per success | Avg seconds |
|---|---|---|---|
| react-cheap | 0/39 | ∞ | |
| react-strong | 26/39 | $0.00186 | 26.7 |
| plan-cheap | 11/39 | $0.000059 | |
| plan-strong | 21/39 | $0.00017 | |
| plan-cascade | 21/39 | $0.00011 | |
| plan>react | 27/39 | $0.00109 | |
| **cascade>react** | **29/39** | **$0.000714** | **14.2** |

- Plans scored 0/9 on surprise tasks. Step-by-step scored 4/9.
- Escalating a crashed plan to a stronger planner didn't help on the discontinued-sku task:
  both models wrote the same naive plan.
- 7 of `cascade>react`'s 10 failures were silent.

### 4. Cost-quality curve and random baseline on experiment 3's data
`8c2409d`

- The frontier was plan-cheap -> plan-cascade -> cascade>react. react-strong was below it.
- The oracle reached **90% pass at 23% of react-strong's cost**, which shows how much room
  per-task routing has.
- Paired against react-strong, cascade>react came out **+7.7 points [−17.9, +33.3]**. That's a
  tie: with 13 tasks the interval is about 50 points wide.

### 5. Holdout v1: **lost**
`2ccf306` · 10 new tasks × 5 repeats

Criteria written before the run:
- Win: paired CI lower bound ≥ −10 points, ≤ 50% of react-strong's cost per task, and on the frontier.
- Lose: point estimate < −10 points, or > 70% of the cost.

| Strategy | Pass | Cost per task vs react-strong |
|---|---|---|
| react-strong | 35/50 | 100% |
| cascade>react | 25/50 | 46% |

The paired difference was **−20 points [−46, +8] → LOSE**. 18 of 25 failures were silent, and
16 of those were a clean plan returning `0.0`. The win on the training set did not generalise.

### 6. Why plans fail silently
`6e79534` · full traces (plan code, every tool call, final variables) · plan-cheap and plan-strong on 23 tasks × 3

Of 80 failures, 21 were silent:

| Cause | Count | Example |
|---|---|---|
| Wrong input -> empty lookup | 10 | `list_orders("Globex Corp.")`; an order ID passed as a customer |
| Guessed a value the data never contains | 6 | `sku == "nuts"` when the data says `'nut'`; `"BOLT"` |
| Logic bug | 5 | Summed bolt quantities when asked for the number of orders |
| Skipped a lookup | 0 | (Invented fields crash instead, and crashes are already caught) |

The 7B made the same spelling guess repeatedly, so agreement between samples would not catch it.

### 7. Trace checks, version 1
`bdd4895` · development data only

The checks are generic and need no knowledge of the domain:
- `empty_lookup`: a tool returned nothing.
- `unseen_value`: the code compares against a string literal that no tool result contains.

Offline, on the 21 silent failures: 17 caught (81%). There were 6 false alarms, all on the
honest "customer has no orders" task.

On development data (23 tasks):

| Strategy | Pass | Cost per task vs strong | $ per success |
|---|---|---|---|
| react-strong | 61/89 (69%) | 100% | $0.00196 |
| cascade>react | 54/89 (61%) | 45% | $0.00099 |
| **+checks** | **57/69 (83%)** | 52% | **$0.00085** |
| +checks+fb | 54/69 (78%) | 62% | $0.00106 |

- Silent wrong answers that got accepted fell from **25 to 3**.
- Paired against react-strong: **+14.5 points [+2.0, +28.7]**. This is optimistic, because the
  checks were designed on this data.
- The feedback note did not help.

### 8. Holdout v2: **lost on cost**
`dbaeabd` · 30 new tasks × 3 repeats, written before the checks' development results were
seen · same criteria as v1

| Strategy | Pass | Cost per task vs react-strong | $ per success |
|---|---|---|---|
| react-strong | 47/90 (52%) | 100% | $0.00282 |
| cascade>react | 42/90 (47%) | 51% | $0.00161 |
| **cascade>react+checks** | **50/90 (56%)** | **71.5%** | **$0.00190** |
| plan-cascade | 32/90 (36%) | 4% | $0.00017 |
| plan-cheap | 23/90 (26%) | 1% | $0.00006 |

- The paired difference was **+3.3 points [−11.1, +17.8]**, a tie, and the ladder was on the
  frontier. But it cost **71.5% > 70% → LOSE**.
- The checks generalised: `unseen_value` had 8 catches and 0 false alarms, `empty_lookup` 23
  catches.
- `empty_lookup`'s 10 false alarms (honest "doesn't exist" answers) made edge tasks *worse*:
  4/6 vs 6/6. A false alarm is not free.
- **Where the money went:** 50 of 90 runs fell back to step-by-step. Those passed only 19/50,
  yet were ~98% of the spend. Catching a failure and then escalating to a rung that also fails
  is the cost problem.
- Development data predicted 83% pass at 52% cost. The holdout gave 56% at 72%.

### 9. Trace checks, version 2, and plan repair
`74b5b05`, `98eae95` · all 53 development tasks (train + holdout v1 + holdout v2) × 2 repeats

- **`empty_lookup` now asks "does this look like a mistake?"** It flags an empty result only
  when the input
  - nearly matches a value a listing tool knows ("did you mean 'globex'?"), or
  - works as input to another tool (`get_order('A-1006')` works, so it's the wrong kind of input), or
  - appears neither in the question nor in any earlier tool result (invented, e.g. `'customer123'`).

  An input taken from the question that matches nothing anywhere (`'wayne'`) is accepted as
  honestly not existing. Rescored offline on 459 logged plan attempts:

  | | Wrong answers caught | False alarms |
  |---|---|---|
  | checks v1 | 132 | 42 |
  | checks v2 | 122 | 1 |

- **Plan repair.** A rejected plan goes back to the same model once, with its own program, the
  reason and the tool calls, before the ladder climbs. That costs ~1% (1.5B) or ~6% (7B) of a
  step-by-step run. In a smoke test, a repair turned a crash into a clean wrong answer, so
  repair can hide the signal that would have escalated.
- Results. This is development data, and the react-strong numbers come from the earlier runs
  of experiments 7 and 8 on the same tasks:

  | Strategy | Pass | Cost per task vs react-strong | $ per success | Fell back to step-by-step |
  |---|---|---|---|---|
  | react-strong | 108/179 (60%) | 100% | $0.00234 | |
  | cascade>react | 96/179 (54%) | 48% | $0.00126 | |
  | +checks (v2) | 69/106 (65%) | 60% | $0.00129 | 48 runs |
  | **+checks (v2) +repair** | **68/106 (64%)** | **43%** | **$0.00094** | **29 runs** |

  - **The empty_lookup fix worked.** Edge tasks went 6/6 for both, up from 4/6 in experiment 8.
  - **Repair cut cost per task by 28% at the same pass rate.** Paired against +checks it came
    out −0.9 points [−9.4, +7.5]. Against react-strong it came out +5.0 [−6.2, +16.7], at
    2.5x lower cost per success. Both are ties on pass rate.
  - **Only the 7B's repairs were useful.** What each repair attempt did:

    | Model | Rejected for | Repair outcome |
    |---|---|---|
    | 7B | check failed | **10/10 fixed and correct**, 2 crashed |
    | 7B | crash | 25 crashed again, 2 flagged |
    | 1.5B | check failed | 16 flagged again, 3 crashed: **0 fixed** |
    | 1.5B | crash | 41 crashed again, 2 flagged, 2 correct, **5 clean but wrong** |

  - **The smoke-test risk was real.** In 5 runs a 1.5B repair turned a crash, which would have
    escalated, into a clean wrong answer that was accepted. Accepted wrong answers went from
    12 to 20 overall, though about half of that is sampling noise in first attempts.
  - Step-by-step was still ~90% of the spend.
  - **What we took from it:** a repair helps when the reason is specific ("'nuts' never appears
    in any result") and the model is strong enough to act on it. Repairing a crash from the
    small model mostly wastes a call, and sometimes hides the failure. The next step is to
    repair only on the 7B.

---

## Lessons so far

1. **Measure cost per successful task.** Per-call cost hid an infinite cost per success in our
   very first run.
2. **The number of steps is the main cost driver** for step-by-step agents. Prompt caching
   makes re-sending cheaper but doesn't remove it.
3. **Cheap-first only works if you can tell when cheap failed.** Crash-based escalation misses
   most real failures.
4. **Trace checks beat asking the model.** Execution traces (inputs, outputs, what the code
   compared against) give strong, label-free signals. Prior work agrees that models are poor at
   catching their own errors.
5. **False alarms cost money and can cost quality.**
6. **Detection is half the problem; where you escalate to is the other half.**
7. **Development numbers overstate results.** Both holdouts came in well below development
   data. Predeclared criteria kept us honest twice.
8. **10-30 tasks give wide confidence intervals.** Pass rates differ a lot between tasks, so
   more distinct tasks help more than more repeats.

## Limitations

- **Toy world:** one small fake database, short tool outputs, numeric answers. Real agents have
  large tool outputs, write actions and fuzzy goals.
- **Hypothetical prices** and **small local models**. The ratios may change with frontier models
  and real price sheets, including cache pricing, which we don't model.
- **One author** wrote the prompts, the tasks and the checks. Both holdouts use the same world
  the checks were designed in, so the next fair test should be a different domain.
- **Latency:** with checks, many tasks reach the slow step-by-step rung. The ladder averaged
  ~25s vs react-strong's ~29-35s, so it saves money, not much time.
- **The plan interpreter is not a security sandbox.**

## Prior art we read and borrowed from

| Topic | Work |
|---|---|
| Routing and evaluation | RouteLLM (Ong et al. 2024, arXiv 2406.18665); LiteLLM (budgets, fallbacks, Auto Router) |
| Cascades | FrugalGPT (arXiv 2305.05176); AutoMix (arXiv 2310.12963); Yue et al., ICLR 2024 (arXiv 2310.03094); Gupta et al., ICLR 2024 (arXiv 2404.10136); Agreement-Based Cascading (arXiv 2407.02348) |
| Verification | LEVER (arXiv 2302.08468); CodeT (arXiv 2207.10397); CRITIC (arXiv 2305.11738); semantic entropy (Farquhar et al., Nature 2024) |
| Limits of self-correction | Huang et al., ICLR 2024 (arXiv 2310.01798); Tyen et al. (arXiv 2311.08516) |
| Silent failures in agents | Reason Less, Verify More (arXiv 2607.07405); Outcome Monitors (arXiv 2608.19303); Characterizing False Success (arXiv 2606.09863) |
| Survey | Moslem & Kelleher 2026 (arXiv 2603.04445) |

## Reproducing

Requires Ollama with `qwen2.5:1.5b` and `qwen2.5:7b`, and `uv`.

```sh
uv run bench.py --split train --repeats 3 --log runs/bench.jsonl          # or holdout / holdout-v2
uv run bench.py --split holdout-v2 --strategies react-strong "cascade>react+checks" --log runs/x.jsonl
uv run report.py runs/x.jsonl                                              # pass rates, cost, per task
uv run curve.py runs/x.jsonl --cheap plan-cheap                            # frontier, random baseline, paired CI
uv run checks.py runs/x.jsonl -v                                           # score trace checks offline
uv run breakdown.py --log runs/x.jsonl                                     # where the tokens go
```

Run logs (`runs/`) are not committed. Every number above can be regenerated from the commit listed.
