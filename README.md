# free-model-aggregator-01

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier model providers, in JSON and in
server-sent events, routing each request to the best free model that can
serve it.

## Run

```sh
node main.ts            # trial mode: reads $MAKEFUN_PROMPT_FILE, registers
                        # with $MAKEFUN_ENV_URL, serves, exits 0 when done
node main.ts --list     # print the catalog document and exit 0
```

Node 22, standard library only. No keys, no network beyond the trial env.

## How it routes

- Candidates are the free models whose quality-table row meets the
  request's requirements (tools, vision, JSON mode, min context). A paid
  model is never picked. An OpenAI JSON-mode request never goes to an
  Anthropic-wire provider, whose wire cannot carry `response_format`.
- Ranked by the sealed quality table's `rating_milli`, with a per-provider
  soft cap taken from the generator's published `calls_min` (the budget
  every trial guarantees). Providers at their cap, or that failed twice in
  a row, are tried only after healthier ones.
- On a refusal (402/429/5xx) the same provider is retried once — a refusal
  on its failure schedule succeeds on the next call number — then the
  request fails over to the next-best provider, and finally retries each
  refused provider once more. Every provider call is sequential, so the
  one-call-at-a-time providers never see a concurrent call.
- Every choice is a pure function of the brief, the manifest and the
  providers' answers so far: no clock, no unseeded randomness. Two runs
  of the same trial leave byte-identical state exports.

## Wires

All six combinations are translated (client OpenAI/Anthropic × provider
OpenAI/Anthropic/Gemini). Requests pass through untouched — system prompt,
images, tools, stop sequences, `max_tokens`, JSON mode — so the provider's
echo matches what the client sent; answers come back with the provider's
text whole, its usage counts and its stop reason mapped
(`stop`/`end_turn`/`STOP` → the client's family).

## Files

- `main.ts` — server, routing, failover, register/poll loop
- `lib/providers.ts` — brief parsing, catalog probe, capabilities
- `lib/wire.ts` — client↔provider translations, SSE writers
- `LICENSE` — MIT
- `BUILD_LOG.md` — build history and local validation numbers
