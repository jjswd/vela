# Architecture

How a request moves through Vela, and why the boundaries sit where they do.

## Layers

```
AgentLoop      orchestration: sessions, persistence, compaction, isolation
AgentRunner    execution: the tool loop and its recovery paths
Provider/Tool  capabilities: wire protocols and tool contracts
```

Dependencies point downward only. `core` never imports from a provider, and
providers never learn about sessions. That constraint is what lets a backend
swap avoid touching routing, and a transport change avoid touching the model
loop.

```
  ┌─────────────────────────────────────────────┐
  │                AgentLoop                    │  session key, workspace,
  │  SessionStore · compaction · per-session     │  compaction, concurrency
  │  worker · emit hooks                         │
  └───────────────────┬─────────────────────────┘
                      │  RunSpec (messages, tools, limits)
  ┌───────────────────▼─────────────────────────┐
  │              AgentRunner                    │  ReAct loop, stream
  │  recovery paths · tool dispatch · budget    │  reassembly, usage
  └───────┬─────────────────────────┬───────────┘
          │                         │
  ┌───────▼────────┐       ┌────────▼─────────┐
  │   Provider     │       │   ToolRegistry   │
  │  wire formats  │       │  schema + exec   │
  └────────────────┘       └──────────────────┘
```

## The split that matters

`AgentLoop` answers "what does it mean to handle a message from a platform?" —
routing, session identity, context assembly, lifecycle.

`AgentRunner` answers "how do I talk to the model?" — streaming, tool
dispatch, iteration limits, recovery.

The failure modes differ, so the debugging entry point differs: routing and
session problems start in `loop.py`, model and tool problems in `runner.py`.

## Control flow of one turn

1. Caller submits text for a session key.
2. The loop resolves the session, appends the user message, and compacts the
   transcript if it has grown past the threshold.
3. A `RunSpec` is built and handed to the runner.
4. The runner loops: request → tool calls → dispatch → append results → repeat.
5. Each request streams text to the emitter as fragments arrive.
6. On completion the transcript is persisted, or discarded when ephemeral.
7. The reply is returned with usage and a stop reason.

## Session isolation

One worker per session key, processing that session's messages in order.

Concurrent turns within a session are not an optimisation to be unlocked, they
are a correctness hazard: tool results enter a transcript whose order must match
the requests that produced them. Different sessions run genuinely in parallel.

## Recovery paths

Model output is untrusted input. Each of these has an explicit bound, because an
unbounded recovery against untrusted input is a way to burn tokens.

| Condition | Response | Bound |
| --- | --- | --- |
| Empty reply | retry, then force a finalisation request | 2 retries |
| Truncated output | collect segments, ask for continuation, join at the end | 3 segments |
| Tool call without a name | dropped during parsing, before it can be persisted | 0 |
| Invalid arguments | all errors reported in one reply so the model can fix them all | — |
| Tool crash or timeout | converted to an error result; the turn continues | — |
| Model error | surfaced to the user; a neutral placeholder enters history | — |
| Iteration budget spent | one tool-free finalisation request, then a local fallback | — |
| Input arriving mid-turn | drained at an iteration boundary | 3 messages / 5 cycles |
| Prompt cache disabled when unreused | decided by cost model, not by hope | — |

Three of these deserve their reasoning spelled out.

**Exhausting the budget is not a hard stop.** The remaining text is whatever
mid-thought fragment happened to exist. One extra request with tool definitions
*removed* converts that into a usable summary. Removing the definitions rather
than asking in the prompt means the model structurally cannot start work it
cannot finish.

**An error is not something the assistant said.** When a request fails, history
receives a neutral placeholder instead of the error text. Otherwise later turns
read the error as a previous answer.

**Missing is not zero.** `cache_read_tokens=None` means the backend did not
report the metric; `0` means it reported zero. Addition preserves the
distinction, because a silently missing metric is worse than a wrong number.

## Provider abstraction

`Provider.complete()` is the whole surface. It takes the internal transcript,
returns a normalised response, and never raises for ordinary model failures —
those come back as `finish_reason="error"` so the loop can react.

The first real test of this boundary was adding a second protocol. Anthropic
differs structurally from the OpenAI shape in three ways, all absorbed inside
the provider:

- the system prompt is a top-level field, not a message;
- tool results are `tool_result` blocks inside a **single** user message that
  must directly follow the assistant turn which requested them, rather than one
  `role="tool"` message each;
- tool schemas use `input_schema`, and stop reasons are `end_turn` /
  `max_tokens` / `tool_use`.

The second is the one that bites: emitting one user message per result produces
consecutive user turns and a 400. The conversion therefore looks ahead and
merges each run of results.

`tests/test_backends.py` asserts that the same transcript survives both
protocols with equivalent tool round-trip information, which is the evidence
that the boundary holds rather than a claim that it does.

## Streaming

`complete()` takes an `on_delta` callback invoked as each fragment is parsed.
Returning the finished response and emitting once would not be streaming in any
sense a user could observe.

Two failure modes are handled explicitly:

- **A frame split across network packets.** The reader must buffer a partial
  line; dropping it truncates the answer mid-word. Tests fragment the body at
  byte boundaries that deliberately do not align with frame boundaries.
- **A backend that ignores the callback.** The runner re-checks the completed
  response and catches up when nothing was forwarded, so an older backend
  degrades to a single delivery instead of silence.

Text is never delivered twice: the emitter tracks whether it already received
fragments for the current request.

## Prompt caching

Caching is not a free win. Writes cost a premium (1.25× for 5 minutes, 2.0× for
1 hour) while reads are discounted to 0.1×, so a prefix must be reused enough
times to repay the write:

| TTL | Write | Break-even requests |
| --- | --- | --- |
| 5 minutes | 1.25× | 2 |
| 1 hour | 2.0× | 3 |

Below that, enabling caching makes every request more expensive. The policy is
therefore off by default and gated on the expected conversation length.

Breakpoints go only on stable prefixes — tool definitions, the system prompt,
and history old enough to have stopped changing. A marker on content that
changes next turn invalidates everything before it, which means writing the cache
again on every request for no benefit. The API also caps breakpoints at four,
so placement is planned centrally.

`CacheStats` records realised hits and writes and derives a cost ratio. A ratio
above 1.0 means the write premium has not been repaid yet, which is the signal
to stop rather than a detail to ignore.

## Testing strategy

The suite concentrates on recovery paths, because those exist to defend against
a model that misbehaves and would otherwise go unexercised.

Doubles are held to the same contract as real backends. A test double that
ignores `on_delta`, or that skips validation, tests nothing — it will happily
pass while the real integration is broken. That is also why assertions check
callback counts, not only final content: a single catch-up emission satisfies a
content assertion while defeating the purpose of streaming.

Where a bug could hide behind convenient mocks, the mock was made harder. An
earlier version returned whole response bodies at once, which meant the
incremental reader's buffering path was never executed.

## The web layer

FastAPI, as an optional dependency. The core runtime does not import it, so
`pip install vela` pulls in one dependency and `pip install vela[server]` adds
the web surface. A test blocks the web packages from importing and checks the
core still works; CI runs the same check.

What the framework replaced:

- Request validation, previously hand-written and returning 400 with a
  hand-shaped error body, is now declared on Pydantic models and rejected with
  422 by the framework.
- SSE framing, previously assembled by hand, is a `StreamingResponse` over an
  async generator.
- The agent used to run on a private event loop spun up inside the request
  thread, polled with `sleep(0)`, because the standard-library server was
  thread-based. Under ASGI the agent, the provider, and the response share one
  loop, so a fragment is forwarded the moment it is produced.

The OpenAI-compatible shape is unchanged, and `/docs` now serves an
interactive API page, which answers "how do I call this?" better than prose.

## Known limits

- Single process; sessions are JSON files guarded by an in-process lock.
- Compaction thresholds use a character-per-token estimate, not an exact
  tokenizer.
- Only the OpenAI-compatible and Anthropic protocols are implemented.
- Caching markers are placed statically, not adapted from observed hit rates.
- No connection limit or backpressure in the HTTP server.
