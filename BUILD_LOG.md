# BUILD_LOG — free-model-aggregator-01

Entry: one Node 22 server (`main.ts`, stdlib only) serving OpenAI Chat
Completions + Anthropic Messages over the ten simulated free providers,
JSON and SSE. Started from the kit starter's `lib/providers.ts` (brief
parsing, catalog probe) and `lib/wire.ts` (OpenAI↔Anthropic translation);
added the Gemini wire both ways plus OpenAI→Anthropic request and
Anthropic→OpenAI answer/stream translation, and replaced the starter's
routing with rating-ordered routing under per-provider soft caps.

## v1 — first working version

- Routing: free models meeting the request's requirements, ranked by
  `rating_milli` desc; soft cap per provider = generator's published
  `calls_min` (groq 9, cerebras 5, gemini 6, zai 4, openrouter 10,
  sambanova 4, pollinations 10, cloudflare 7, ovh 5, kilol 12).
  Providers at cap or with 2 consecutive failures are deprioritised.
- Failover: on 402/429/5xx try the next provider; after all fail, retry
  each refused provider once.
- All 6 wire combinations translated; requests passed through untouched
  (system, images, tools, stops, max_tokens, JSON mode); JSON-mode
  OpenAI requests never routed to Anthropic-wire providers.
- Local validation (env server + entry + sealed scorer, no docker):
  - trial-0: score 99 (quality 99, served 100, wire 100, failover 100,
    catalog 100), 20/20 valid, 25 upstream calls
  - trial-1: score 96 (quality 95, served 95, wire 99, failover 100,
    catalog 100), 19/20 valid, 28 upstream calls
  - trial-2: score 99 (quality 99, served 100, wire 100, failover 100,
    catalog 100), 20/20 valid, 30 upstream calls
- Trial-1 failure found by the scorer: r20 invalid, `echo_user` mismatch
  (provider echoed empty user text). Root cause: `anthropicToGemini`
  dropped user/assistant messages whose `content` is a plain string
  instead of a block array.

## v2 — genuine revision: string-content bug in Anthropic→Gemini

- `lib/wire.ts` `anthropicToGemini`: handle `content` as a plain string
  for both user and assistant messages (was: only block arrays).
- Re-ran all three practice trials:
  - trial-0: score 99 (99/100/100/100/100), 20/20 valid, 25 calls
  - trial-1: score 99 (99/100/100/100/100), 20/20 valid, 28 calls
  - trial-2: score 99 (99/100/100/100/100), 20/20 valid, 30 calls

## v3 — genuine revision: immediate same-provider retry

- `main.ts` `serve()`: on a refusal, retry the same provider once
  immediately before failing over. A refusal on a provider's failure
  schedule succeeds on the next call number, so the request keeps its
  best-rated model; a provider that is truly out (budget spent, outage)
  refuses again and the request moves on.
- Re-ran all three practice trials:
  - trial-0: score 99 (99/100/100/100/100), 20/20 valid, 25 calls
  - trial-1: score 99 (99/100/100/100/100), 20/20 valid, 28 calls
  - trial-2: score 99 (99/100/100/100/100), 20/20 valid, 30 calls
- Trial-0 detail after v3: r04 and r12 (openrouter scheduled 2nd/10th
  call failures) recovered to the 1475 model instead of falling back to
  1470. Remaining quality gap is structural, not a bug: only one model
  rates 1475 and its budget tops out at 12 calls while 17 requests could
  use it, so quality 100 is unreachable on this workload shape; 99 is
  the practical ceiling (score 99 = max achievable here).

## Determinism proof

- Trial-0 run twice: `state.json` sha256 identical both runs
  (`e96c088395b7e1e65e75c5b619e05f9291eee91d99249077bcea134b2d4dddff`).
- `node main.ts --list` outside a trial prints `{"providers":[]}`, exit 0,
  byte-identical across runs.
- `node main.ts --list` inside trial-0 prints a catalog that equals the
  registered catalog in the state export exactly (10 providers, 27 models:
  26 free + 1 paid).

## Notes / limits

- No docker on this machine, so validation ran the sealed env server,
  entry and scorer as local processes (`/tmp/run-trial.sh`), which is
  exactly what `grade-local --driver process` does minus the containers.
- `grade-local --driver docker` was not run (no docker binary).
- The entry makes no model calls of its own; `model_calls: optional`
  waives the LLM floor.
- No GitHub repo created, no bundle/check/submit run — parent's step.
