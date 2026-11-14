# Free model aggregator (free-model-aggregator-01 entry)

One Python 3.13 server (`main.py`, standard library only) that serves the OpenAI Chat Completions
API and the Anthropic Messages API, JSON and server-sent events, over the trial's ten free-tier
providers. Every request goes to the highest-rated **free** model that meets its requirements, in
whichever wire that provider speaks.

```
python3 main.py --list     # print the catalog document and exit 0 ({"providers":[]} outside a trial)
python3 main.py            # probe, listen, register, serve, exit 0 when GET /work says done
```

## How it works

- **Brief**: the three JSON documents of `$MAKEFUN_PROMPT_FILE` (providers, quality table, workload
  manifest) are found by their `schema` member, whatever prose surrounds them.
- **Catalog**: each provider's listing is probed in its own wire (`GET /v1/models` or
  `GET /v1beta/models`, with the provider's own auth scheme, which is not always its wire's usual one).
  Models keep the provider's listing order and its `free` flag, and `base_url` is the provider's
  path (`/p/<id>`). The catalog is serialized once, and `--list` prints the same bytes the register body carries.
- **Routing**: candidates are free models meeting the request's requirements (the manifest row for
  this arrival, tightened by what the body itself carries: tools, images, JSON mode, size). They are
  ranked by rating; on rating ties the openai wire wins (bigger budgets, so fewer refusals; see below). A JSON-mode request never
  goes to the Anthropic wire, which cannot carry it, unless nothing else is left.
- **Failover**: a 402/429/5xx (or an unparseable answer) moves the request to the next provider.
  The refusal itself is read: "budget is spent" or "is out" retires the provider for the rest of the
  trial, a scheduled refusal only skips it for this request, a `concurrent_requests` refusal costs no
  budget, and a 400 unknown-model mark retires only that model. One call per provider runs at a time.
- **Wires**: client bodies are translated to the upstream wire field by field: system prompt, user
  text, images (data URLs ↔ base64 blocks ↔ `inlineData`), tools and tool choice, tool calls and
  results, stop sequences, `max_tokens`, temperature, JSON mode (`response_format` ↔
  `responseMimeType`). Same-wire requests pass through untouched except `model` (and
  `stream_options.include_usage`, so streamed usage is always known). Answers (JSON or SSE) are
  parsed into one normalized result and rendered in the client's wire with the provider's usage
  counts and the mapped stop reason. OpenAI streams end with a usage chunk and `[DONE]`. Anthropic
  streams run `message_start → content blocks → message_delta → message_stop`.
- **Determinism**: no wall clock and no randomness reach a response or a provider call. Ids are
  counters, `created` is `MAKEFUN_TRIAL_NOW`, and every routing choice is a function of the brief and
  the answers seen so far. A monotonic guard exits before the 180 s wall.

## Local practice runs

`tools/practice.py <kit>` runs the kit's own env (`program/env/server.mjs`, process-driver layout)
and scorer (`program/scorer/score.mjs`) with node against `practice/trial-N`, and prints every
column and the state export's sha256. `tools/routesim.py <dir>` replays the router alone over
generator-drawn instances, with the env's refusal rules, to tune routing offline.

Practice results (two runs, identical state exports):

| trial | score | quality | served | wire | failover | catalog | calls |
|---|---|---|---|---|---|---|---|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |

Quality cannot reach 100: its bound ignores budgets, and the single 1475-rated model runs out.

## Tests

```
python3 -m unittest discover -s tests
```

The tests cover brief parsing, `--list` outside a trial, catalog order, router ranking and
refusal handling, and an in-process fake host that checks every client wire against every
upstream wire (JSON and SSE) the way the env's echo checks do.

## License

MIT, see [LICENSE](LICENSE).
