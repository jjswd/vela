# Interview preparation

Notes on what to say about this project, and what each answer should contain.
Numbers below are measured from this codebase, not estimated.

## One-sentence description

> A small agent runtime in Python that separates orchestration from model
> execution, absorbs two LLM wire protocols behind one provider contract, and
> treats model output as untrusted input — every recovery path is bounded.

## Scale to quote accurately

| Metric | Value |
| --- | --- |
| Source | 3,573 lines across 16 modules |
| Tests | 179 across 8 files (2,890 lines) |
| Offline checks | 46, no API key required |
| Providers | 2 protocols, 6 registered backends |
| Python | 3.11 – 3.13 (CI matrix) |

Quote these exactly: 3,573 lines of source, 2,890 of tests, 179 test cases, 46
offline checks. If asked about the recovery paths, there are eight and each has an
explicit bound.

## Questions to expect, with the shape of a good answer

### Why split the loop from the runner?

The two answer different questions: "what does handling a platform message
mean?" versus "how do I talk to the model?" Routing, session identity and
compaction live in the loop; streaming, tool dispatch, iteration limits and
recovery live in the runner.

The payoff is orthogonal extension. Adding a protocol required zero changes to
`core/`, and `tests/test_backends.py` asserts that the same transcript survives
both protocols with equivalent tool round-trip information.

The cost: context has to be handed across the boundary in `RunSpec`, and a bug
can sit on either side of it — which is why the docs say routing problems start
in `loop.py` and model or tool problems in `runner.py`.

### Why is every recovery path bounded?

Because model output is untrusted input, and an unbounded recovery against
untrusted input is just a way to burn tokens. A model that always returns
`finish_reason="length"` would loop forever without the cap on recovery
segments.

The bounds are stated as constants next to the code that enforces them:
2 empty-reply retries, 3 continuation segments, 3 injected messages, 5
injection cycles.

### Why does exhausting the budget not just fail?

The remaining output is whatever mid-thought fragment existed, which is not a
useful answer. One extra request with the tool definitions removed lets the
model summarise what it already gathered.

Removing the definitions rather than instructing the model in the prompt is
deliberate: with no tools available it structurally cannot start work it cannot
finish. `test_iteration_limit_triggers_tool_free_finalization` asserts the
final request carries no tools.

### What is the most interesting bug you fixed?

Two are worth telling; pick based on the role.

**Streaming usage accounting (Anthropic).** The API reports input tokens in
`message_start` and output tokens in `message_delta`. The code assigned rather
than merged, so `input_tokens` was always reported as 0 and cost tracking was
silently wrong. The subtle part: the fix cannot compare values, because the
usage parser normalises a missing field to `0`, which is indistinguishable from
a reported zero. The merge has to test for key presence.

**Optional-parameter schema inference.** Tool schemas are inferred from type
annotations. A parameter written `count: int | None` inferred as a string,
because the code did not look inside unions. Since optional parameters are the
common way tools are written, most real tool schemas were misdescribed — the
model would have been told to send numbers as text.

### How do you know the tests are actually testing anything?

Mutation checks on the two fixes above: reintroduce the bug, watch the specific
tests fail, restore, watch them pass.

One structural rule earned its place: doubles implement the same contract as
real backends. An earlier test provider ignored the `on_delta` callback and
skipped validation, which meant a broken streaming path would still have passed.
That is also why assertions check callback counts rather than only final
content — a single catch-up emission satisfies a content assertion while
defeating the purpose of streaming.

### How do you know streaming works?

Two layers of test. Protocol level drives realistic SSE byte streams through a
mock transport that splits the body at byte boundaries deliberately misaligned
with frame boundaries, because a mock returning the whole body at once never
exercises the reader's buffering path. Runtime level asserts the emitter
receives fragments in order, that the stream is not closed before tool work
finishes, and that text is never delivered twice.

The mutation test for this: break the argument passing from the provider to the
stream callback, and the fragment-count assertions fail.

### How is the prompt caching decision made?

Caching is not free. Writes cost a premium (1.25× for 5 minutes, 2.0× for 1
hour) and reads are discounted to 0.1×, so a prefix must be reused enough times
to repay the write. Break-even is 2 requests for the 5-minute TTL and 3 for the
1-hour one, computed from the price table rather than hard-coded.

So the policy is off by default and gated on the expected conversation length.
Below the threshold, enabling it would make every request more expensive.

Breakpoints go only on stable prefixes — tools, system prompt, and history old
enough to have stopped changing. A marker on content that changes next turn
invalidates everything before it, which means rewriting the cache on every
request for no benefit. The API also caps breakpoints at four.

Measured on a 10-turn session with a 9,000-token stable prefix and a 1,000-token
changing tail: 81% hit rate and input cost at 0.29× the uncached baseline.

### How do you know caching is worth it in practice?

`CacheStats` records realised hits and writes and derives a cost ratio against
the same usage without caching. A ratio above 1.0 means the write premium has
not been repaid yet. The policy is a prediction; the statistics are the
measurement.

Getting that accounting right mattered more than getting it fast. Two bugs here
would each have made the savings look better than the invoice: excluding cached
writes from the baseline, and subtracting cache hits from the uncached input
count when the backend already reports them separately.

### What would you do differently?

- Replace the character-per-token estimate used for compaction thresholds with
  the backend's own tokenizer. It is approximate now, which means compaction
  fires at a token count that does not match what the model will actually
  count.
- Add cross-process locking for sessions. Two processes sharing a workspace can
  currently overwrite each other's transcripts.
- The cache breakpoints are placed statically. Reading the observed hit rate
  back into the placement logic would place them where they actually help.
- The offline checks verify behaviour, not the claims in the README. A test
  that asserts the documented cost figures still match the model would keep
  the two from drifting.

## Honest limits to state up front

Saying these before being asked reads as judgement rather than defensiveness.

- No scheduler, no worker pool, no rate limiting across sessions.
- Sessions are JSON files; no cross-process safety.
- Only two wire protocols.
- No retry policy on the HTTP layer beyond transient errors at the runner.
- Compaction summarisation uses the same model call, which is a cost line item
  that is easy to overlook.

## What to show, in order

1. `python examples/verify.py` — 46 checks passing without any credentials.
2. `ARCHITECTURE.md` recovery table, and the reasoning behind one bound.
3. The streaming test with the misaligned byte-boundary stream.
4. `tests/test_backends.py` — the same transcript through two protocols.
5. The caching cost model and its measured result.

Open the code when asked; do not narrate it unprompted.

## Questions to ask them

- How do you decide what gets built into a service versus kept out of the
  runtime? The loop/runner split only pays off because both stayed small.
- What is your position on a feature that is correct but rarely exercised?
- When a system has to degrade rather than fail, where do you draw the line?
