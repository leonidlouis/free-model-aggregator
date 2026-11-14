#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""free-model-aggregator-01 entry: one server, two client wires, ten free providers.

Standard library only (Python 3.13). The trial runs it as `python3 -P -s main.py`, so everything
lives in this one file: no sibling module could be imported under -P.

Layout of this file:
  1. the brief: its three JSON documents, found by `schema`
  2. HTTP helpers and provider auth
  3. the catalog probe (each provider in its own wire) and the one catalog document
  4. the model pool and the router (hazard model, quality guard, free only, requirements, failover)
  5. request translation: OpenAI / Anthropic client bodies -> openai / anthropic / gemini upstream
  6. answer parsing: upstream JSON or SSE -> one normalized result
  7. answer rendering: normalized result -> the client's wire, JSON or SSE
  8. the HTTP server, register, the /work poll and the entrypoint

Determinism: nothing reads the wall clock or unseeded randomness into anything sent; ids are
counters, `created` comes from MAKEFUN_TRIAL_NOW (or a fixed epoch), routing depends only on the
brief, the requests and the providers' answers so far. The only clock use is the monotonic safety
guard that exits before the trial's wall, which never reaches a request.
"""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"
WIRES = ("openai", "anthropic", "gemini")
AUTHS = ("bearer", "x-api-key", "x-goog-api-key", "none")
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
FIXED_CREATED = 1767225600
POLL_SECONDS = 0.1
UPSTREAM_TIMEOUT = 25.0
PROBE_TIMEOUT = 5.0
WALL_GUARD_SECONDS = 170.0
MAX_BODY_BYTES = 32 * 1024 * 1024


def log(message: str) -> None:
    try:
        sys.stderr.write(f"aggregator: {message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def is_record(v) -> bool:
    return isinstance(v, dict)


def num_or_none(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    return None


def dumps(value) -> str:
    """Compact JSON, as JavaScript's JSON.stringify writes it."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------------------------
# 1. The brief
# ---------------------------------------------------------------------------------------------


def documents_in(text: str) -> list[dict]:
    """Every JSON object in `text` that carries a string `schema`, in order of appearance."""
    decoder = json.JSONDecoder()
    out: list[dict] = []
    at = text.find("{")
    while at != -1:
        try:
            value, end = decoder.raw_decode(text, at)
        except ValueError:
            at = text.find("{", at + 1)
            continue
        if is_record(value) and isinstance(value.get("schema"), str):
            out.append(value)
            at = text.find("{", end)
        else:
            at = text.find("{", at + 1)
    return out


class Provider:
    __slots__ = ("id", "name", "wire", "auth", "path", "key_env", "key", "order")

    def __init__(self, raw: dict, order: int):
        self.id = str(raw.get("id") or "")
        self.name = str(raw.get("name") or self.id)
        self.wire = raw.get("wire")
        auth = raw.get("auth")
        self.auth = auth if auth in AUTHS else "bearer"
        base = ""
        for k in ("path", "base_path", "base_url", "base"):
            if isinstance(raw.get(k), str) and raw[k] != "":
                base = raw[k]
                break
        self.path = base
        self.key_env = raw.get("key_env") if isinstance(raw.get("key_env"), str) else ""
        self.key = raw.get("key") if isinstance(raw.get("key"), str) else ""
        self.order = order


class Brief:
    def __init__(self):
        self.providers: list[Provider] = []
        self.ratings: dict[tuple[str, str], dict] = {}
        self.best: dict[str, int] = {}
        self.manifest: list[dict] = []
        self.error: str | None = None


def parse_brief_text(text: str) -> Brief:
    brief = Brief()
    docs = documents_in(text)
    for d in docs:
        if d["schema"] == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list) and not brief.providers:
            for raw in d["providers"]:
                if is_record(raw):
                    p = Provider(raw, len(brief.providers))
                    if p.id and p.wire in WIRES:
                        brief.providers.append(p)
        elif d["schema"] == RATINGS_SCHEMA and isinstance(d.get("ratings"), list) and not brief.ratings:
            for r in d["ratings"]:
                if is_record(r) and isinstance(r.get("model"), str) and isinstance(r.get("provider"), str):
                    brief.ratings[(r["provider"], r["model"])] = r
            for b in d.get("best") or []:
                if is_record(b) and isinstance(b.get("id"), str):
                    brief.best[b["id"]] = num_or_none(b.get("rating_milli")) or 0
        elif d["schema"] == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list) and not brief.manifest:
            for m in d["manifest"]:
                if is_record(m) and isinstance(m.get("id"), str):
                    brief.manifest.append(m)
    if not brief.providers:
        brief.error = "the brief holds no provider directory"
    return brief


def read_brief(path: str) -> Brief:
    if not path:
        b = Brief()
        b.error = "MAKEFUN_PROMPT_FILE is not set"
        return b
    try:
        with open(path, encoding="utf-8") as f:
            return parse_brief_text(f.read())
    except OSError as err:
        b = Brief()
        b.error = f"cannot read {path}: {err}"
        return b


def requirements_of(raw) -> dict:
    out = {"tools": False, "vision": False, "json": False, "min_context": None}
    if is_record(raw):
        out["tools"] = raw.get("tools") is True
        out["vision"] = raw.get("vision") is True
        out["json"] = raw.get("json") is True
        mc = num_or_none(raw.get("min_context"))
        out["min_context"] = mc
    return out


# ---------------------------------------------------------------------------------------------
# 2. HTTP helpers
# ---------------------------------------------------------------------------------------------


def join_base(env_url: str, base: str) -> str:
    b = base.strip()
    if re.match(r"^https?://", b, re.I):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth: str, key: str) -> dict:
    if auth == "none" or key == "":
        return {}
    if auth == "bearer":
        return {"authorization": f"Bearer {key}"}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


class HttpAnswer:
    __slots__ = ("status", "headers", "body", "error")

    def __init__(self, status: int, headers: dict, body: bytes, error: str | None = None):
        self.status = status
        self.headers = headers
        self.body = body
        self.error = error

    def json(self):
        try:
            return json.loads(self.body.decode("utf-8"))
        except Exception:
            return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # never follow a redirect off the env
        return None


_OPENER = urllib.request.build_opener(_NoRedirect, urllib.request.ProxyHandler({}))


def http(method: str, url: str, headers: dict | None = None, body: bytes | None = None, timeout: float = UPSTREAM_TIMEOUT) -> HttpAnswer:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            return HttpAnswer(res.status, {k.lower(): v for k, v in res.headers.items()}, res.read())
    except urllib.error.HTTPError as err:
        try:
            data = err.read()
        except Exception:
            data = b""
        return HttpAnswer(err.code, {k.lower(): v for k, v in (err.headers or {}).items()}, data)
    except Exception as err:  # refused, reset, timeout
        return HttpAnswer(0, {}, b"", f"{type(err).__name__}: {err}")


# ---------------------------------------------------------------------------------------------
# 3. The catalog
# ---------------------------------------------------------------------------------------------


def bool_or_none(v):
    return v if isinstance(v, bool) else None


def zero_priced(v: dict):
    pricing = v.get("pricing")
    if not is_record(pricing):
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(float(x))
        elif isinstance(x, str) and x.strip():
            try:
                values.append(float(x))
            except ValueError:
                return False
    if not values:
        return None
    return all(n == 0 for n in values)


def model_in(wire: str, v) -> dict | None:
    if not is_record(v):
        return None
    if wire == "gemini":
        mid = v.get("name") if isinstance(v.get("name"), str) else ""
        mid = mid[len("models/"):] if mid.startswith("models/") else mid
    else:
        mid = v.get("id") if isinstance(v.get("id"), str) else ""
    if not mid:
        return None
    ctx = None
    for k in ("context_length", "context_window", "inputTokenLimit", "max_input_tokens"):
        n = num_or_none(v.get(k))
        if n is not None:
            ctx = int(n)
            break
    free = bool_or_none(v.get("free"))
    if free is None:
        free = zero_priced(v)
    if free is None and isinstance(mid, str) and mid.endswith(":free"):
        free = True
    name = v.get("display_name") or v.get("displayName") or mid
    return {
        "id": mid,
        "display_name": name if isinstance(name, str) else mid,
        "free": free,
        "context_length": ctx,
        "tools": bool_or_none(v.get("tools")),
        "vision": bool_or_none(v.get("vision")),
        "json": bool_or_none(v.get("json")),
    }


def models_in(wire: str, body) -> list[dict] | None:
    if not is_record(body):
        return None
    listing = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return None
    out = []
    for v in listing:
        m = model_in(wire, v)
        if m is not None:
            out.append(m)
    return out


def catalog_url(env_url: str, p: Provider) -> str:
    base = join_base(env_url, p.path)
    return f"{base}/v1beta/models" if p.wire == "gemini" else f"{base}/v1/models"


def probe(env_url: str, p: Provider) -> list[dict] | None:
    """One provider's listing in its own wire, following Anthropic and Gemini pagination."""
    headers = {"accept": "application/json", **auth_headers(p.auth, p.key)}
    if p.wire == "anthropic":
        headers["anthropic-version"] = DEFAULT_ANTHROPIC_VERSION
    url = catalog_url(env_url, p)
    models: list[dict] = []
    for attempt in range(2):
        models = []
        next_url = url
        ok = True
        for _page in range(20):
            ans = http("GET", next_url, headers, timeout=PROBE_TIMEOUT)
            if ans.status != 200:
                ok = False
                break
            body = ans.json()
            page = models_in(p.wire, body)
            if page is None:
                ok = False
                break
            models.extend(page)
            token = None
            if p.wire == "gemini" and isinstance(body.get("nextPageToken"), str) and body["nextPageToken"]:
                token = ("pageToken", body["nextPageToken"])
            elif p.wire == "anthropic" and body.get("has_more") is True and isinstance(body.get("last_id"), str):
                token = ("after_id", body["last_id"])
            if token is None:
                break
            next_url = f"{url}?{urllib.parse.urlencode([token])}"
        if ok:
            return models
        log(f"probe of {p.id} failed (attempt {attempt + 1})")
    return None


class ModelEntry:
    """One servable (provider, model) pair with what the router needs to know."""

    __slots__ = ("provider", "id", "display_name", "free", "rating", "context_length", "tools", "vision", "json", "listed_at")

    def __init__(self, provider: Provider, listed: dict, row: dict | None, listed_at: int):
        self.provider = provider
        self.id = listed["id"]
        self.display_name = listed.get("display_name") or self.id
        self.listed_at = listed_at
        # Free: the provider's own listing is the truth the catalog is scored on; where it says
        # nothing, the brief's quality table (which only lists free models) decides.
        listed_free = listed.get("free")
        if listed_free is None:
            listed_free = bool(row is not None and row.get("free") is True)
        self.free = bool(listed_free)
        self.rating = int(num_or_none(row.get("rating_milli")) or 0) if row is not None else 0
        ctx = num_or_none(row.get("context_length")) if row is not None else None
        self.context_length = int(ctx) if ctx is not None else listed.get("context_length")

        def cap(name):
            if row is not None and isinstance(row.get(name), bool):
                return row[name]
            return listed.get(name) is True

        self.tools = cap("tools")
        self.vision = cap("vision")
        self.json = cap("json")

    def routable(self) -> bool:
        return self.free

    def meets(self, req: dict) -> bool:
        if req["tools"] and not self.tools:
            return False
        if req["vision"] and not self.vision:
            return False
        if req["json"] and not self.json:
            return False
        mc = req.get("min_context")
        if mc is not None and (self.context_length is None or self.context_length < mc):
            return False
        return True


class World:
    """The brief, the probed listings, the catalog document and the routable pool."""

    def __init__(self, brief: Brief, env_url: str):
        self.brief = brief
        self.env_url = env_url
        self.listings: dict[str, list[dict]] = {}
        self.catalog: dict = {"providers": []}
        self.pool: list[ModelEntry] = []

    def load(self, do_probe: bool) -> None:
        for p in self.brief.providers:
            listed = probe(self.env_url, p) if do_probe else None
            if listed is None:
                # No answer (or no env): fall back to the brief's table rows for this provider.
                listed = [
                    {"id": m, "display_name": m, "free": r.get("free") is True,
                     "context_length": num_or_none(r.get("context_length")), "tools": None, "vision": None, "json": None}
                    for (pid, m), r in self.brief.ratings.items() if pid == p.id
                ]
            self.listings[p.id] = listed
        self.catalog = self.build_catalog()
        self.pool = []
        for p in self.brief.providers:
            for i, m in enumerate(self.unique(self.listings.get(p.id, []))):
                self.pool.append(ModelEntry(p, m, self.brief.ratings.get((p.id, m["id"])), i))

    @staticmethod
    def unique(models: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for m in models:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            out.append(m)
        return out

    def build_catalog(self) -> dict:
        providers = []
        for p in self.brief.providers:
            models = []
            for m in self.unique(self.listings.get(p.id, [])):
                row = self.brief.ratings.get((p.id, m["id"]))
                free = m.get("free")
                if free is None:
                    free = bool(row is not None and row.get("free") is True)
                ctx = m.get("context_length")
                if ctx is None and row is not None:
                    ctx = num_or_none(row.get("context_length"))
                models.append({"id": m["id"], "free": bool(free), "context_length": ctx})
            # The provider's own path names it; the env's origin is joined only to dial.
            providers.append({"id": p.id, "base_url": p.path, "key_env": p.key_env, "wire": p.wire, "models": models})
        return {"providers": providers}

    def catalog_text(self) -> str:
        return dumps(self.catalog)

    def served_models(self) -> list[ModelEntry]:
        """The free models GET /v1/models lists: one row per id, first provider first."""
        seen = set()
        out = []
        for e in self.pool:
            if e.free and e.id not in seen:
                seen.add(e.id)
                out.append(e)
        return out


# ---------------------------------------------------------------------------------------------
# 4. The router
# ---------------------------------------------------------------------------------------------

PERMANENT_HINTS = ("budget is spent", "budget", " is out", "exhausted", "outage", "quota exceeded", "credits")
TRANSIENT_HINTS = ("refused this call", "try again", "temporarily", "concurrent")


def error_message_of(body) -> tuple[str, str]:
    """(message, code) out of any of the three wires' error envelopes."""
    if not is_record(body):
        return "", ""
    err = body.get("error")
    if is_record(err):
        msg = err.get("message") if isinstance(err.get("message"), str) else ""
        code = err.get("code")
        if not isinstance(code, str):
            code = err.get("status") if isinstance(err.get("status"), str) else err.get("type") if isinstance(err.get("type"), str) else ""
        return msg, code
    if isinstance(err, str):
        return str(body.get("message") or ""), err
    return "", ""


WIRE_PREF = {"openai": 0, "gemini": 1, "anthropic": 2}

# Free-tier schedules as the kit's generator draws them, per provider id: the trial budget B is a
# whole number in [calls_min, calls_max]; 1 or 2 scheduled refusals fall on distinct call numbers
# in [2, B-1]; one provider of the ten (the designated one) always refuses its call 2; one provider
# of the ten has an outage from a call number in [2, B-1] on. A call refuses as "budget is spent"
# past B, else as "is out" once the outage has begun, else as "refused this call" when scheduled.
SCHEDULES = {
    "groq": (9, 12), "cerebras": (5, 6), "gemini": (6, 8), "zai": (4, 5), "openrouter": (10, 12),
    "sambanova": (4, 5), "pollinations": (10, 14), "cloudflare": (7, 8), "ovh": (5, 6), "kilol": (12, 16),
}
OK, SCHED, OUT, BUDGET, OTHER = "ok", "sched", "out", "budget", "other"


class HazardModel:
    """P(a provider's next call is refused | the outcomes of its calls so far in this trial).

    Every scenario the generator can draw for a provider (budget, refusal set, designated or not,
    outage start or none) is enumerated once with its probability and folded into a distribution
    over outcome sequences for calls 1..B+1 (the last is always the budget refusal). The hazard of
    a history is then the posterior mass of the sequences extending it whose next outcome is a
    refusal. An unclassified outcome (a 5xx, an unparseable answer) matches any scheduled one at
    its position. Unknown providers and histories no scenario explains get a conservative fallback."""

    FALLBACK_FIRST = 0.0
    FALLBACK = 0.3

    def __init__(self, schedules: dict = SCHEDULES, providers: int | None = None):
        n = providers or len(schedules) or 1
        self.p_designated = 1.0 / n
        self.p_outage = 1.0 / n
        self.paths = {pid: self._paths(lo, hi) for pid, (lo, hi) in schedules.items()}
        self.cache: dict[tuple[str, tuple], float] = {}

    def _paths(self, lo: int, hi: int) -> list[tuple[tuple, float]]:
        acc: dict[tuple, float] = {}
        for budget in range(lo, hi + 1):
            p_budget = 1.0 / (hi - lo + 1)
            span = list(range(2, budget))
            if not span:
                continue
            sets = [(a,) for a in span] + [(a, b) for i, a in enumerate(span) for b in span[i + 1:]]
            per_count = {1: sum(1 for s in sets if len(s) == 1), 2: sum(1 for s in sets if len(s) == 2)}
            for s in sets:
                if per_count[len(s)] == 0:
                    continue
                p_set = p_budget * 0.5 / per_count[len(s)]
                for designated in (False, True):
                    fails = s if (not designated or 2 in s) else (2,) + s[1:]
                    p_des = self.p_designated if designated else 1.0 - self.p_designated
                    for outage in (None, *span):
                        p_out = (1.0 - self.p_outage) if outage is None else self.p_outage / len(span)
                        seq = []
                        for c in range(1, budget + 2):
                            if c > budget:
                                seq.append(BUDGET)
                            elif outage is not None and c >= outage:
                                seq.append(OUT)
                            elif c in fails:
                                seq.append(SCHED)
                            else:
                                seq.append(OK)
                        key = tuple(seq)
                        acc[key] = acc.get(key, 0.0) + p_set * p_des * p_out
        return sorted(acc.items())

    def hazard(self, pid: str, history: tuple) -> float:
        key = (pid, history)
        h = self.cache.get(key)
        if h is None:
            h = self._hazard(pid, history)
            self.cache[key] = h
        return h

    def _hazard(self, pid: str, history: tuple) -> float:
        paths = self.paths.get(pid)
        if paths is None:
            return self.FALLBACK_FIRST if not history else self.FALLBACK
        n = len(history)
        if OTHER in history:  # an unclassified refusal spent a call: any scheduled outcome fits it
            def fits(seq):
                return all(h == OTHER or h == s for h, s in zip(history, seq))
        else:
            def fits(seq):
                return seq[:n] == history
        total = refused = 0.0
        for seq, p in paths:
            if len(seq) > n and fits(seq):
                total += p
                if seq[n] != OK:
                    refused += p
        if total <= 0.0:
            return self.FALLBACK
        return refused / total


_HAZARD_MODEL: HazardModel | None = None


def hazard_model() -> HazardModel:
    """The shared table: built once per process (it depends on nothing but SCHEDULES)."""
    global _HAZARD_MODEL
    if _HAZARD_MODEL is None:
        _HAZARD_MODEL = HazardModel()
    return _HAZARD_MODEL


class ProviderState:
    __slots__ = ("dead", "calls", "failures", "consecutive_failures", "lock", "bad_models", "last_failed", "history")

    def __init__(self):
        self.dead = False
        self.calls = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.last_failed = False
        self.lock = threading.Lock()  # one inference call at a time per provider
        self.bad_models: set[str] = set()
        self.history: tuple = ()  # outcome of each budget-spending call, in order


class Slot:
    """Where one arrival sits in the manifest: its weight, the best free rating that meets it, and
    how many requests are still to come after it."""

    __slots__ = ("weight", "best", "remaining")

    def __init__(self, weight: int, best: int | None, remaining: int):
        self.weight = weight
        self.best = best
        self.remaining = remaining


class Router:
    """Picks, for each request, the free model that meets it with the fewest expected wasted calls,
    without giving up the quality column.

    Candidates fall in tiers: a wire that carries the request faithfully first (no JSON mode or
    missing max_tokens on the Anthropic wire), a model the client named, and a provider that has not
    refused this request yet. Within the best tier, a model is admissible if serving it keeps the
    trial's rating loss (sum of weight * (best - rating)) within CAP - RESERVE * (requests left
    after this one), where CAP is the largest loss that keeps floor(100 * earned / bound) at
    QUALITY_TARGET. Among admissible models the one minimizing
        hazard(provider's history) + LAMBDA * weight * (best - rating)
    wins (ties: higher rating, then openai wire, provider order, listing order). If none is
    admissible, the best-rated one is taken. Without a manifest row (or its best rating) only the
    top-rated models are admissible."""

    LAMBDA = 0.0005
    RESERVE = 5
    QUALITY_TARGET = 98
    MAX_CALLS_PER_REQUEST = 40

    def __init__(self, world: World, hazards: HazardModel | None = None):
        self.world = world
        self.state = {p.id: ProviderState() for p in world.brief.providers}
        self.hazards = hazards or hazard_model()
        self.loss = 0
        self.cap = self.loss_cap(world.brief)

    def loss_cap(self, brief: Brief) -> int | None:
        """The largest rating loss keeping floor(100 * earned / bound) >= QUALITY_TARGET, or None
        when the brief does not give every manifest row's best rating."""
        if not brief.manifest:
            return None
        bound = 0
        for m in brief.manifest:
            best = brief.best.get(m["id"])
            if not best:
                return None
            bound += self.weight_of(m) * best
        need = -(-self.QUALITY_TARGET * bound // 100)  # ceil
        return bound - need

    @staticmethod
    def weight_of(row: dict) -> int:
        w = num_or_none(row.get("weight"))
        return int(w) if w is not None and w > 0 else 1

    def slot_for(self, index: int) -> Slot | None:
        manifest = self.world.brief.manifest
        if index >= len(manifest):
            return None
        row = manifest[index]
        return Slot(self.weight_of(row), self.world.brief.best.get(row["id"]), len(manifest) - index - 1)

    def hazard(self, pid: str) -> float:
        return self.hazards.hazard(pid, self.state[pid].history)

    def candidates(self, req: dict, client_wire: str, features: dict, requested_model: str | None) -> list[ModelEntry]:
        """Free models meeting `req`, best rated first. A wire that cannot carry the request faithfully
        (JSON mode on the Anthropic wire, a missing max_tokens there) only comes after every faithful one."""
        pool = [e for e in self.world.pool if e.routable() and not self.state[e.provider.id].dead
                and e.id not in self.state[e.provider.id].bad_models and e.meets(req)]
        pool.sort(key=lambda e: (self.faithful(e, features), self.named(e, requested_model)) + self.order_key(e))
        return pool

    @staticmethod
    def faithful(e: ModelEntry, features: dict) -> int:
        return 1 if e.provider.wire == "anthropic" and (features["json_mode"] or features["max_tokens"] is None) else 0

    @staticmethod
    def named(e: ModelEntry, requested_model: str | None) -> int:
        if not requested_model:
            return 1
        return 0 if requested_model in (e.id, f"{e.provider.id}/{e.id}") else 1

    @staticmethod
    def order_key(e: ModelEntry) -> tuple:
        return (-e.rating, WIRE_PREF.get(e.provider.wire, 3), e.provider.order, e.listed_at)

    def choose(self, tier: list[ModelEntry], slot: Slot | None) -> ModelEntry:
        """The pick within one tier (see the class docstring)."""
        top = max(e.rating for e in tier)
        if slot is None or slot.best is None or self.cap is None:
            fit = [e for e in tier if e.rating == top]
            best, weight = top, 1
        else:
            best, weight = slot.best, slot.weight
            limit = self.cap - self.RESERVE * slot.remaining
            fit = [e for e in tier if self.loss + weight * max(0, best - e.rating) <= limit]
            if not fit:
                fit = [e for e in tier if e.rating == top]

        def cost(e: ModelEntry):
            return (self.hazard(e.provider.id) + self.LAMBDA * weight * max(0, best - e.rating),) + self.order_key(e)

        return min(fit, key=cost)

    def note_success(self, pid: str, entry: ModelEntry | None = None, slot: Slot | None = None) -> None:
        st = self.state[pid]
        st.calls += 1
        st.consecutive_failures = 0
        st.last_failed = False
        st.history += (OK,)
        if entry is not None and slot is not None and slot.best is not None:
            self.loss += slot.weight * max(0, slot.best - entry.rating)

    def note_failure(self, pid: str, model: str, status: int, body) -> str:
        """Classify one refused call; returns 'dead', 'transient', 'busy' or 'model'."""
        st = self.state[pid]
        msg, code = error_message_of(body)
        low = f"{msg} {code}".lower()
        if status in (400, 404, 422) and ("model" in low or status == 404):
            st.bad_models.add(model)
            return "model"
        if status in (401, 403):
            st.dead = True
            return "dead"
        if "concurrent" in low:
            return "busy"  # a second concurrent call spends none of the budget
        st.calls += 1
        st.failures += 1
        st.consecutive_failures += 1
        st.last_failed = True
        if "refused this call" in low:
            verdict, outcome = "transient", SCHED
        elif "budget" in low:
            verdict, outcome = "dead", BUDGET
        elif " is out" in low or "outage" in low:
            verdict, outcome = "dead", OUT
        elif any(h in low for h in PERMANENT_HINTS):
            verdict, outcome = "dead", OTHER
        else:
            verdict, outcome = "transient", OTHER
        st.history += (outcome,)
        if st.consecutive_failures >= 3:
            verdict = "dead"
        if verdict == "dead":
            st.dead = True
        return verdict

    def route(self, req: dict, client_wire: str, feats: dict, requested: str | None, call, slot: Slot | None = None):
        """Try candidates until one answers. `call(entry)` makes one provider call and returns
        (status, error body or None, result or None). Returns (entry, result, last status)."""
        refused: set[str] = set()
        busy_retries = 0
        last_status = 0
        for _ in range(self.MAX_CALLS_PER_REQUEST):
            cands = self.candidates(req, client_wire, feats, requested)
            if not cands:
                break
            tier_of = {id(e): (self.faithful(e, feats), self.named(e, requested), e.provider.id in refused) for e in cands}
            first = min(tier_of.values())
            entry = self.choose([e for e in cands if tier_of[id(e)] == first], slot)
            pid = entry.provider.id
            status, err_body, result = call(entry)
            last_status = status
            if status == 200 and result is not None and result["ok"]:
                self.note_success(pid, entry, slot)
                return entry, result, status
            if status == 200:
                verdict = self.note_failure(pid, entry.id, 502, None)
            else:
                verdict = self.note_failure(pid, entry.id, status, err_body)
            log(f"{pid}/{entry.id} answered {status}: {verdict}")
            if verdict == "busy" and busy_retries < 2:
                busy_retries += 1
                continue
            if verdict != "model":
                refused.add(pid)
        return None, None, last_status


# ---------------------------------------------------------------------------------------------
# 5. Request translation
# ---------------------------------------------------------------------------------------------


def text_of_blocks(content, sep: str = "\n") -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return sep.join(b["text"] for b in content if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))


def parse_data_url(url: str):
    m = re.match(r"^data:([^;,]*)(;base64)?,(.*)$", url, re.S)
    if m is None:
        return None
    media = m.group(1) or "application/octet-stream"
    if m.group(2):
        return media, m.group(3)
    return media, base64.b64encode(urllib.parse.unquote_to_bytes(m.group(3))).decode("ascii")


def guess_media(url: str) -> str:
    low = url.lower().split("?")[0]
    for ext, media in ((".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg"), (".gif", "image/gif"), (".webp", "image/webp")):
        if low.endswith(ext):
            return media
    return "image/png"


def parse_json_args(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except ValueError:
            pass
    return {}


# The normalized request (IR):
#   system: list[str]
#   messages: [{role: user|assistant|tool, parts: [part]}]
#     part: {type: text, text} | {type: image, media, data} | {type: image_url, url}
#           | {type: tool_call, id, name, arguments(str)} | {type: tool_result, id, name, content(str), is_error}
#   tools: [{name, description, parameters}]
#   tool_choice: None | auto | required | none | {name}
#   max_tokens, stops(list), temperature, top_p, json_mode(None|'object'|{'schema':...}), stream


def ir_from_openai(body: dict) -> dict:
    ir = {"system": [], "messages": [], "tools": [], "tool_choice": None, "parallel": None}
    names_by_id: dict[str, str] = {}
    for msg in body.get("messages") or []:
        if not is_record(msg):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in ("system", "developer"):
            t = text_of_blocks(content)
            if t != "":
                ir["system"].append(t)
            continue
        if role == "tool":
            tid = msg.get("tool_call_id") if isinstance(msg.get("tool_call_id"), str) else ""
            text = content if isinstance(content, str) else text_of_blocks(content)
            if not isinstance(content, (str, list)) and content is not None:
                text = dumps(content)
            ir["messages"].append({"role": "tool", "parts": [{"type": "tool_result", "id": tid, "name": names_by_id.get(tid, msg.get("name") or ""), "content": text, "is_error": False}]})
            continue
        parts = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                if not is_record(b):
                    continue
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    parts.append({"type": "text", "text": b["text"]})
                elif b.get("type") == "image_url":
                    iu = b.get("image_url")
                    url = iu.get("url") if is_record(iu) else iu if isinstance(iu, str) else None
                    if isinstance(url, str):
                        d = parse_data_url(url)
                        if d is not None:
                            parts.append({"type": "image", "media": d[0], "data": d[1]})
                        else:
                            parts.append({"type": "image_url", "url": url})
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                cid = call.get("id") if isinstance(call.get("id"), str) else f"call_{len(names_by_id)}"
                name = fn.get("name") if isinstance(fn.get("name"), str) else ""
                names_by_id[cid] = name
                args = fn.get("arguments")
                parts.append({"type": "tool_call", "id": cid, "name": name, "arguments": args if isinstance(args, str) else dumps(args or {})})
            ir["messages"].append({"role": "assistant", "parts": parts})
        else:
            ir["messages"].append({"role": "user", "parts": parts})
    for t in body.get("tools") or []:
        if is_record(t) and is_record(t.get("function")) and isinstance(t["function"].get("name"), str):
            fn = t["function"]
            ir["tools"].append({"name": fn["name"], "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                                "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}}})
    tc = body.get("tool_choice")
    if tc in ("auto", "required", "none"):
        ir["tool_choice"] = tc
    elif is_record(tc) and is_record(tc.get("function")) and isinstance(tc["function"].get("name"), str):
        ir["tool_choice"] = {"name": tc["function"]["name"]}
    if isinstance(body.get("parallel_tool_calls"), bool):
        ir["parallel"] = body["parallel_tool_calls"]
    mt = num_or_none(body.get("max_tokens"))
    if mt is None:
        mt = num_or_none(body.get("max_completion_tokens"))
    ir["max_tokens"] = mt
    stop = body.get("stop")
    ir["stops"] = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else []
    ir["temperature"] = num_or_none(body.get("temperature"))
    ir["top_p"] = num_or_none(body.get("top_p"))
    rf = body.get("response_format")
    ir["json_mode"] = None
    if is_record(rf):
        if rf.get("type") == "json_object":
            ir["json_mode"] = "object"
        elif rf.get("type") == "json_schema":
            js = rf.get("json_schema")
            ir["json_mode"] = {"schema": js.get("schema") if is_record(js) and is_record(js.get("schema")) else None}
    ir["stream"] = body.get("stream") is True
    return ir


def ir_from_anthropic(body: dict) -> dict:
    ir = {"system": [], "messages": [], "tools": [], "tool_choice": None, "parallel": None}
    sysv = body.get("system")
    if isinstance(sysv, str):
        if sysv != "":
            ir["system"].append(sysv)
    elif isinstance(sysv, list):
        t = text_of_blocks(sysv)
        if t != "":
            ir["system"].append(t)
    names_by_id: dict[str, str] = {}
    for msg in body.get("messages") or []:
        if not is_record(msg):
            continue
        role = "assistant" if msg.get("role") == "assistant" else "user"
        content = msg.get("content")
        parts = []
        results = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                if not is_record(b):
                    continue
                kind = b.get("type")
                if kind == "text" and isinstance(b.get("text"), str):
                    parts.append({"type": "text", "text": b["text"]})
                elif kind == "image" and is_record(b.get("source")):
                    src = b["source"]
                    if src.get("type") == "base64" and isinstance(src.get("data"), str):
                        parts.append({"type": "image", "media": src.get("media_type") or "image/png", "data": src["data"]})
                    elif src.get("type") == "url" and isinstance(src.get("url"), str):
                        parts.append({"type": "image_url", "url": src["url"]})
                elif kind == "tool_use":
                    cid = b.get("id") if isinstance(b.get("id"), str) else f"toolu_{len(names_by_id)}"
                    name = b.get("name") if isinstance(b.get("name"), str) else ""
                    names_by_id[cid] = name
                    parts.append({"type": "tool_call", "id": cid, "name": name, "arguments": dumps(b.get("input") if is_record(b.get("input")) else {})})
                elif kind == "tool_result":
                    tid = b.get("tool_use_id") if isinstance(b.get("tool_use_id"), str) else ""
                    c = b.get("content")
                    if isinstance(c, str):
                        text = c
                    elif isinstance(c, list):
                        text = text_of_blocks(c)
                        if text == "" and c:
                            text = dumps(c)
                    else:
                        text = "" if c is None else dumps(c)
                    results.append({"type": "tool_result", "id": tid, "name": names_by_id.get(tid, ""), "content": text, "is_error": b.get("is_error") is True})
        if results:
            ir["messages"].append({"role": "tool", "parts": results})
        if parts or not results:
            ir["messages"].append({"role": role, "parts": parts})
    for t in body.get("tools") or []:
        if is_record(t) and isinstance(t.get("name"), str):
            ir["tools"].append({"name": t["name"], "description": t.get("description") if isinstance(t.get("description"), str) else "",
                                "parameters": t.get("input_schema") if is_record(t.get("input_schema")) else {"type": "object", "properties": {}}})
    tc = body.get("tool_choice")
    if is_record(tc):
        if tc.get("type") == "auto":
            ir["tool_choice"] = "auto"
        elif tc.get("type") == "any":
            ir["tool_choice"] = "required"
        elif tc.get("type") == "none":
            ir["tool_choice"] = "none"
        elif tc.get("type") == "tool" and isinstance(tc.get("name"), str):
            ir["tool_choice"] = {"name": tc["name"]}
        if tc.get("disable_parallel_tool_use") is True:
            ir["parallel"] = False
    ir["max_tokens"] = num_or_none(body.get("max_tokens"))
    ss = body.get("stop_sequences")
    ir["stops"] = [s for s in ss if isinstance(s, str)] if isinstance(ss, list) else []
    ir["temperature"] = num_or_none(body.get("temperature"))
    ir["top_p"] = num_or_none(body.get("top_p"))
    ir["json_mode"] = None
    ir["stream"] = body.get("stream") is True
    return ir


def features_of(api: str, body: dict) -> dict:
    """What a request body asks of a model and of a wire, read off the body itself."""
    ir = ir_from_openai(body) if api == "openai" else ir_from_anthropic(body)
    vision = any(p["type"] in ("image", "image_url") for m in ir["messages"] for p in m["parts"])
    chars = len(dumps(body))
    return {
        "tools": len(ir["tools"]) > 0,
        "vision": vision,
        "json_mode": ir["json_mode"] is not None,
        "max_tokens": ir["max_tokens"],
        "approx_tokens": chars // 4,
    }


def openai_from_ir(ir: dict, model: str) -> dict:
    out: dict = {"model": model, "messages": []}
    for s in ir["system"]:
        out["messages"].append({"role": "system", "content": s})
    for m in ir["messages"]:
        if m["role"] == "tool":
            for p in m["parts"]:
                out["messages"].append({"role": "tool", "tool_call_id": p["id"], "content": p["content"]})
            continue
        if m["role"] == "assistant":
            text = "".join(p["text"] for p in m["parts"] if p["type"] == "text")
            calls = [{"id": p["id"], "type": "function", "function": {"name": p["name"], "arguments": p["arguments"]}} for p in m["parts"] if p["type"] == "tool_call"]
            msg = {"role": "assistant", "content": text if text != "" or not calls else None}
            if calls:
                msg["tool_calls"] = calls
            out["messages"].append(msg)
            continue
        parts = []
        for p in m["parts"]:
            if p["type"] == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                parts.append({"type": "image_url", "image_url": {"url": f"data:{p['media']};base64,{p['data']}"}})
            elif p["type"] == "image_url":
                parts.append({"type": "image_url", "image_url": {"url": p["url"]}})
        if len(parts) == 1 and parts[0]["type"] == "text":
            out["messages"].append({"role": "user", "content": parts[0]["text"]})
        else:
            out["messages"].append({"role": "user", "content": parts})
    if ir["max_tokens"] is not None:
        out["max_tokens"] = ir["max_tokens"]
    if ir["temperature"] is not None:
        out["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        out["top_p"] = ir["top_p"]
    if ir["stops"]:
        out["stop"] = list(ir["stops"])
    if ir["tools"]:
        out["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in ir["tools"]]
        tc = ir["tool_choice"]
        if isinstance(tc, str):
            out["tool_choice"] = tc
        elif is_record(tc):
            out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
        if ir["parallel"] is not None:
            out["parallel_tool_calls"] = ir["parallel"]
    jm = ir["json_mode"]
    if jm == "object":
        out["response_format"] = {"type": "json_object"}
    elif is_record(jm):
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": jm.get("schema") or {"type": "object"}}}
    if ir["stream"]:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def anthropic_from_ir(ir: dict, model: str) -> dict:
    out: dict = {"model": model}
    # Real Anthropic requires max_tokens; the router keeps such requests off this wire when it can.
    out["max_tokens"] = ir["max_tokens"] if ir["max_tokens"] is not None else 4096
    if ir["system"]:
        out["system"] = "\n".join(ir["system"])
    messages: list[dict] = []

    def push(role: str, blocks: list[dict]) -> None:
        if messages and messages[-1]["role"] == role and role == "user" and any(b.get("type") == "tool_result" for b in messages[-1]["content"]):
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for m in ir["messages"]:
        blocks = []
        for p in m["parts"]:
            if p["type"] == "text":
                blocks.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": p["media"], "data": p["data"]}})
            elif p["type"] == "image_url":
                blocks.append({"type": "image", "source": {"type": "url", "url": p["url"]}})
            elif p["type"] == "tool_call":
                blocks.append({"type": "tool_use", "id": p["id"], "name": p["name"], "input": parse_json_args(p["arguments"])})
            elif p["type"] == "tool_result":
                r = {"type": "tool_result", "tool_use_id": p["id"], "content": p["content"]}
                if p.get("is_error"):
                    r["is_error"] = True
                blocks.append(r)
        if m["role"] == "user" and len(blocks) == 1 and blocks[0]["type"] == "text":
            messages.append({"role": "user", "content": blocks[0]["text"]})
            continue
        push("assistant" if m["role"] == "assistant" else "user", blocks)
    out["messages"] = messages
    if ir["temperature"] is not None:
        out["temperature"] = max(0, min(1, ir["temperature"]))
    if ir["top_p"] is not None:
        out["top_p"] = ir["top_p"]
    if ir["stops"]:
        out["stop_sequences"] = list(ir["stops"])
    if ir["tools"]:
        out["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in ir["tools"]]
        tc = ir["tool_choice"]
        choice = None
        if tc == "auto":
            choice = {"type": "auto"}
        elif tc == "required":
            choice = {"type": "any"}
        elif tc == "none":
            choice = {"type": "none"}
        elif is_record(tc):
            choice = {"type": "tool", "name": tc["name"]}
        if ir["parallel"] is False:
            choice = choice or {"type": "auto"}
            choice["disable_parallel_tool_use"] = True
        if choice is not None:
            out["tool_choice"] = choice
    if ir["stream"]:
        out["stream"] = True
    return out


GEMINI_SCHEMA_DROP = {"additionalProperties", "$schema", "$id", "$ref", "$defs", "definitions", "examples", "strict"}


def gemini_schema(schema):
    """A JSON schema with the keys Gemini's function declarations refuse taken out."""
    if isinstance(schema, list):
        return [gemini_schema(x) for x in schema]
    if not is_record(schema):
        return schema
    out = {}
    for k, v in schema.items():
        if k in GEMINI_SCHEMA_DROP:
            continue
        if k == "properties" and is_record(v):
            out[k] = {pk: gemini_schema(pv) for pk, pv in v.items()}
        else:
            out[k] = gemini_schema(v)
    return out


def gemini_from_ir(ir: dict) -> dict:
    out: dict = {"contents": []}
    if ir["system"]:
        out["systemInstruction"] = {"parts": [{"text": s} for s in ir["system"]]}
    for m in ir["messages"]:
        parts = []
        for p in m["parts"]:
            if p["type"] == "text":
                parts.append({"text": p["text"]})
            elif p["type"] == "image":
                parts.append({"inlineData": {"mimeType": p["media"], "data": p["data"]}})
            elif p["type"] == "image_url":
                parts.append({"fileData": {"mimeType": guess_media(p["url"]), "fileUri": p["url"]}})
            elif p["type"] == "tool_call":
                parts.append({"functionCall": {"name": p["name"], "args": parse_json_args(p["arguments"])}})
            elif p["type"] == "tool_result":
                parts.append({"functionResponse": {"name": p["name"] or p["id"], "response": {"content": p["content"]}}})
        role = "model" if m["role"] == "assistant" else "user"
        if out["contents"] and m["role"] == "tool" and out["contents"][-1]["role"] == "user" and all("functionResponse" in x for x in out["contents"][-1]["parts"]):
            out["contents"][-1]["parts"].extend(parts)
        else:
            out["contents"].append({"role": role, "parts": parts})
    if ir["tools"]:
        out["tools"] = [{"functionDeclarations": [{"name": t["name"], "description": t["description"], "parameters": gemini_schema(t["parameters"])} for t in ir["tools"]]}]
        tc = ir["tool_choice"]
        mode = None
        allowed = None
        if tc == "auto":
            mode = "AUTO"
        elif tc == "required":
            mode = "ANY"
        elif tc == "none":
            mode = "NONE"
        elif is_record(tc):
            mode, allowed = "ANY", [tc["name"]]
        if mode is not None:
            fc = {"mode": mode}
            if allowed:
                fc["allowedFunctionNames"] = allowed
            out["toolConfig"] = {"functionCallingConfig": fc}
    gc: dict = {}
    if ir["max_tokens"] is not None:
        gc["maxOutputTokens"] = ir["max_tokens"]
    if ir["stops"]:
        gc["stopSequences"] = list(ir["stops"])
    if ir["temperature"] is not None:
        gc["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        gc["topP"] = ir["top_p"]
    jm = ir["json_mode"]
    if jm is not None:
        gc["responseMimeType"] = "application/json"
        if is_record(jm) and is_record(jm.get("schema")):
            gc["responseSchema"] = gemini_schema(jm["schema"])
    if gc:
        out["generationConfig"] = gc
    return out


def upstream_request(entry: ModelEntry, api: str, body: dict, client_headers: dict, env_url: str):
    """(url, headers, body bytes) for one call of `entry` serving a client request."""
    p = entry.provider
    base = join_base(env_url, p.path)
    headers = {"content-type": "application/json", **auth_headers(p.auth, p.key)}
    stream = body.get("stream") is True
    if p.wire == "openai":
        if api == "openai":
            out = dict(body)  # the same wire: pass it through untouched but for the model
            out["model"] = entry.id
            if stream:
                so = dict(out["stream_options"]) if is_record(out.get("stream_options")) else {}
                so["include_usage"] = True
                out["stream_options"] = so
        else:
            out = openai_from_ir(ir_from_anthropic(body), entry.id)
        if stream:
            headers["accept"] = "text/event-stream"
        return f"{base}/v1/chat/completions", headers, dumps(out).encode("utf-8")
    if p.wire == "anthropic":
        if api == "anthropic":
            out = dict(body)
            out["model"] = entry.id
            headers["anthropic-version"] = client_headers.get("anthropic-version") or DEFAULT_ANTHROPIC_VERSION
            if client_headers.get("anthropic-beta"):
                headers["anthropic-beta"] = client_headers["anthropic-beta"]
        else:
            out = anthropic_from_ir(ir_from_openai(body), entry.id)
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_VERSION
        if stream:
            headers["accept"] = "text/event-stream"
        return f"{base}/v1/messages", headers, dumps(out).encode("utf-8")
    ir = ir_from_openai(body) if api == "openai" else ir_from_anthropic(body)
    out = gemini_from_ir(ir)
    model = urllib.parse.quote(entry.id, safe="")
    if stream:
        headers["accept"] = "text/event-stream"
        url = f"{base}/v1beta/models/{model}:streamGenerateContent?alt=sse"
    else:
        url = f"{base}/v1beta/models/{model}:generateContent"
    return url, headers, dumps(out).encode("utf-8")


# ---------------------------------------------------------------------------------------------
# 6. Answer parsing (upstream -> normalized result)
# ---------------------------------------------------------------------------------------------
#   result: {text, tool_calls: [{id, name, arguments}], stop: end|max|tool|filter|stop_sequence,
#            usage: {input, output} | None, model, ok}

OPENAI_STOP = {"stop": "end", "length": "max", "tool_calls": "tool", "function_call": "tool", "content_filter": "filter"}
ANTHROPIC_STOP = {"end_turn": "end", "stop_sequence": "stop_sequence", "max_tokens": "max", "tool_use": "tool", "refusal": "filter", "pause_turn": "end"}
GEMINI_STOP = {"STOP": "end", "MAX_TOKENS": "max", "SAFETY": "filter", "RECITATION": "filter", "BLOCKLIST": "filter",
               "PROHIBITED_CONTENT": "filter", "SPII": "filter", "IMAGE_SAFETY": "filter", "MALFORMED_FUNCTION_CALL": "end", "OTHER": "end"}


def sse_frames(text: str) -> list[tuple[str | None, str]]:
    out = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:][1:] if line[5:].startswith(" ") else line[5:])
        if data:
            out.append((event, "\n".join(data)))
    return out


def is_sse(ans: HttpAnswer) -> bool:
    ct = ans.headers.get("content-type", "")
    if "text/event-stream" in ct:
        return True
    return ans.body[:6] in (b"data: ", b"event:", b"data:{")


def new_result(model: str) -> dict:
    return {"text": "", "tool_calls": [], "stop": None, "usage": None, "model": model, "ok": True}


def parse_openai_answer(ans: HttpAnswer, model: str) -> dict:
    r = new_result(model)
    if is_sse(ans):
        calls: dict[int, dict] = {}
        usage = None
        for _ev, data in sse_frames(ans.body.decode("utf-8", "replace")):
            if data.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get("usage")):
                usage = chunk["usage"]
            if isinstance(chunk.get("model"), str):
                r["model"] = chunk["model"]
            for choice in chunk.get("choices") or []:
                if not is_record(choice) or choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") if is_record(choice.get("delta")) else {}
                if isinstance(delta.get("content"), str):
                    r["text"] += delta["content"]
                for call in delta.get("tool_calls") or []:
                    if not is_record(call):
                        continue
                    idx = call.get("index") if isinstance(call.get("index"), int) else len(calls)
                    slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if isinstance(call.get("id"), str) and call["id"]:
                        slot["id"] = call["id"]
                    fn = call.get("function") if is_record(call.get("function")) else {}
                    if isinstance(fn.get("name"), str) and fn["name"]:
                        slot["name"] = fn["name"]
                    if isinstance(fn.get("arguments"), str):
                        slot["arguments"] += fn["arguments"]
                if isinstance(choice.get("finish_reason"), str):
                    r["stop"] = OPENAI_STOP.get(choice["finish_reason"], "end")
        r["tool_calls"] = [calls[k] for k in sorted(calls)]
    else:
        body = ans.json()
        if not is_record(body) or not isinstance(body.get("choices"), list) or not body["choices"]:
            r["ok"] = False
            return r
        if isinstance(body.get("model"), str):
            r["model"] = body["model"]
        choice = body["choices"][0] if is_record(body["choices"][0]) else {}
        msg = choice.get("message") if is_record(choice.get("message")) else {}
        c = msg.get("content")
        r["text"] = c if isinstance(c, str) else text_of_blocks(c, "")
        for call in msg.get("tool_calls") or []:
            if is_record(call) and is_record(call.get("function")):
                fn = call["function"]
                args = fn.get("arguments")
                r["tool_calls"].append({"id": call.get("id") or "", "name": fn.get("name") or "", "arguments": args if isinstance(args, str) else dumps(args or {})})
        if isinstance(choice.get("finish_reason"), str):
            r["stop"] = OPENAI_STOP.get(choice["finish_reason"], "end")
        usage = body.get("usage")
    if is_record(usage):
        r["usage"] = {"input": num_or_none(usage.get("prompt_tokens")) or 0, "output": num_or_none(usage.get("completion_tokens")) or 0}
    if r["stop"] is None:
        r["stop"] = "tool" if r["tool_calls"] else "end"
    return r


def parse_anthropic_answer(ans: HttpAnswer, model: str) -> dict:
    r = new_result(model)
    if is_sse(ans):
        blocks: dict[int, dict] = {}
        inp = None
        outp = None
        for ev, data in sse_frames(ans.body.decode("utf-8", "replace")):
            try:
                d = json.loads(data)
            except ValueError:
                continue
            if not is_record(d):
                continue
            kind = d.get("type") or ev
            if kind == "message_start" and is_record(d.get("message")):
                m = d["message"]
                if isinstance(m.get("model"), str):
                    r["model"] = m["model"]
                if is_record(m.get("usage")):
                    inp = num_or_none(m["usage"].get("input_tokens"))
                    outp = num_or_none(m["usage"].get("output_tokens"))
            elif kind == "content_block_start" and isinstance(d.get("index"), int):
                cb = d.get("content_block") if is_record(d.get("content_block")) else {}
                blocks[d["index"]] = {"type": cb.get("type"), "text": cb.get("text") or "", "id": cb.get("id") or "", "name": cb.get("name") or "", "json": ""}
            elif kind == "content_block_delta" and isinstance(d.get("index"), int):
                b = blocks.setdefault(d["index"], {"type": "text", "text": "", "id": "", "name": "", "json": ""})
                delta = d.get("delta") if is_record(d.get("delta")) else {}
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    b["text"] += delta["text"]
                elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                    b["json"] += delta["partial_json"]
            elif kind == "message_delta":
                delta = d.get("delta") if is_record(d.get("delta")) else {}
                if isinstance(delta.get("stop_reason"), str):
                    r["stop"] = ANTHROPIC_STOP.get(delta["stop_reason"], "end")
                if is_record(d.get("usage")):
                    o = num_or_none(d["usage"].get("output_tokens"))
                    if o is not None:
                        outp = o
                    i = num_or_none(d["usage"].get("input_tokens"))
                    if i is not None and inp is None:
                        inp = i
            elif kind == "error":
                r["ok"] = False
        for k in sorted(blocks):
            b = blocks[k]
            if b["type"] == "text":
                r["text"] += b["text"]
            elif b["type"] == "tool_use":
                r["tool_calls"].append({"id": b["id"], "name": b["name"], "arguments": b["json"] or "{}"})
        if inp is not None or outp is not None:
            r["usage"] = {"input": inp or 0, "output": outp or 0}
    else:
        body = ans.json()
        if not is_record(body) or body.get("type") != "message":
            r["ok"] = False
            return r
        if isinstance(body.get("model"), str):
            r["model"] = body["model"]
        for b in body.get("content") or []:
            if not is_record(b):
                continue
            if b.get("type") == "text" and isinstance(b.get("text"), str):
                r["text"] += b["text"]
            elif b.get("type") == "tool_use":
                r["tool_calls"].append({"id": b.get("id") or "", "name": b.get("name") or "", "arguments": dumps(b.get("input") if is_record(b.get("input")) else {})})
        if isinstance(body.get("stop_reason"), str):
            r["stop"] = ANTHROPIC_STOP.get(body["stop_reason"], "end")
        u = body.get("usage")
        if is_record(u):
            r["usage"] = {"input": num_or_none(u.get("input_tokens")) or 0, "output": num_or_none(u.get("output_tokens")) or 0}
    if r["stop"] is None:
        r["stop"] = "tool" if r["tool_calls"] else "end"
    return r


def parse_gemini_answer(ans: HttpAnswer, model: str) -> dict:
    r = new_result(model)
    if is_sse(ans):
        chunks = []
        for _ev, data in sse_frames(ans.body.decode("utf-8", "replace")):
            try:
                chunks.append(json.loads(data))
            except ValueError:
                continue
    else:
        body = ans.json()
        chunks = body if isinstance(body, list) else [body]  # a non-SSE stream answers a JSON array
    finish = None
    usage = None
    seen_any = False
    for c in chunks:
        if not is_record(c):
            continue
        if is_record(c.get("usageMetadata")):
            usage = c["usageMetadata"]
        if isinstance(c.get("modelVersion"), str):
            r["model"] = c["modelVersion"]
        cands = c.get("candidates")
        if not isinstance(cands, list) or not cands:
            continue
        seen_any = True
        cand = cands[0] if is_record(cands[0]) else {}
        content = cand.get("content") if is_record(cand.get("content")) else {}
        for part in content.get("parts") or []:
            if not is_record(part) or part.get("thought") is True:
                continue
            if isinstance(part.get("text"), str):
                r["text"] += part["text"]
            if is_record(part.get("functionCall")):
                fc = part["functionCall"]
                cid = fc.get("id") if isinstance(fc.get("id"), str) and fc.get("id") else f"call_{len(r['tool_calls'])}"
                r["tool_calls"].append({"id": cid, "name": fc.get("name") or "", "arguments": dumps(fc.get("args") if is_record(fc.get("args")) else {})})
        if isinstance(cand.get("finishReason"), str):
            finish = cand["finishReason"]
    if not seen_any and usage is None:
        r["ok"] = False
        return r
    stop = GEMINI_STOP.get(finish, "end") if finish is not None else "end"
    if r["tool_calls"] and stop == "end":
        stop = "tool"
    r["stop"] = stop
    if is_record(usage):
        r["usage"] = {"input": num_or_none(usage.get("promptTokenCount")) or 0,
                      "output": (num_or_none(usage.get("candidatesTokenCount")) or 0) + (num_or_none(usage.get("thoughtsTokenCount")) or 0)}
    return r


def parse_answer(wire: str, ans: HttpAnswer, model: str) -> dict:
    try:
        if wire == "openai":
            return parse_openai_answer(ans, model)
        if wire == "anthropic":
            return parse_anthropic_answer(ans, model)
        return parse_gemini_answer(ans, model)
    except Exception as err:  # a malformed answer is a failed call, never a crash
        log(f"cannot parse a {wire} answer: {err}")
        r = new_result(model)
        r["ok"] = False
        return r


# ---------------------------------------------------------------------------------------------
# 7. Answer rendering (normalized result -> client wire)
# ---------------------------------------------------------------------------------------------

TO_OPENAI_FINISH = {"end": "stop", "stop_sequence": "stop", "max": "length", "tool": "tool_calls", "filter": "content_filter"}
TO_ANTHROPIC_STOP = {"end": "end_turn", "stop_sequence": "stop_sequence", "max": "max_tokens", "tool": "tool_use", "filter": "refusal"}
SSE_PIECE = 64


def sse(event: str | None, data) -> str:
    return (f"event: {event}\n" if event is not None else "") + f"data: {dumps(data)}\n\n"


def pieces(text: str) -> list[str]:
    return [text[i:i + SSE_PIECE] for i in range(0, len(text), SSE_PIECE)]


def usage_pair(r: dict) -> tuple[int, int]:
    u = r["usage"] or {"input": 0, "output": 0}
    return int(u["input"]), int(u["output"])


def openai_tool_calls(r: dict, seq: int) -> list[dict]:
    return [{"id": c["id"] or f"call_{seq}_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
            for i, c in enumerate(r["tool_calls"])]


def render_openai_json(r: dict, seq: int, created: int) -> bytes:
    inp, out = usage_pair(r)
    msg: dict = {"role": "assistant", "content": r["text"] if (r["text"] != "" or not r["tool_calls"]) else None}
    if r["tool_calls"]:
        msg["tool_calls"] = openai_tool_calls(r, seq)
    body = {
        "id": f"chatcmpl-agg{seq:04d}",
        "object": "chat.completion",
        "created": created,
        "model": r["model"],
        "choices": [{"index": 0, "message": msg, "finish_reason": TO_OPENAI_FINISH.get(r["stop"], "stop"), "logprobs": None}],
        "usage": {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out},
    }
    return dumps(body).encode("utf-8")


def render_openai_sse(r: dict, seq: int, created: int) -> bytes:
    cid = f"chatcmpl-agg{seq:04d}"

    def chunk(delta, finish):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": r["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]}

    frames = [sse(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in pieces(r["text"]):
        frames.append(sse(None, chunk({"content": piece}, None)))
    for i, call in enumerate(openai_tool_calls(r, seq)):
        frames.append(sse(None, chunk({"tool_calls": [{"index": i, "id": call["id"], "type": "function", "function": {"name": call["function"]["name"], "arguments": ""}}]}, None)))
        for piece in pieces(call["function"]["arguments"]):
            frames.append(sse(None, chunk({"tool_calls": [{"index": i, "function": {"arguments": piece}}]}, None)))
    frames.append(sse(None, chunk({}, TO_OPENAI_FINISH.get(r["stop"], "stop"))))
    inp, out = usage_pair(r)
    frames.append(sse(None, {"id": cid, "object": "chat.completion.chunk", "created": created, "model": r["model"], "choices": [],
                             "usage": {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode("utf-8")


def anthropic_blocks(r: dict, seq: int) -> list[dict]:
    blocks: list[dict] = []
    if r["text"] != "" or not r["tool_calls"]:
        blocks.append({"type": "text", "text": r["text"]})
    for i, c in enumerate(r["tool_calls"]):
        blocks.append({"type": "tool_use", "id": c["id"] or f"toolu_agg{seq:04d}_{i}", "name": c["name"], "input": parse_json_args(c["arguments"])})
    return blocks


def render_anthropic_json(r: dict, seq: int) -> bytes:
    inp, out = usage_pair(r)
    body = {
        "id": f"msg_agg{seq:04d}",
        "type": "message",
        "role": "assistant",
        "model": r["model"],
        "content": anthropic_blocks(r, seq),
        "stop_reason": TO_ANTHROPIC_STOP.get(r["stop"], "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": inp, "output_tokens": out},
    }
    return dumps(body).encode("utf-8")


def render_anthropic_sse(r: dict, seq: int) -> bytes:
    inp, out = usage_pair(r)
    frames = [sse("message_start", {"type": "message_start", "message": {
        "id": f"msg_agg{seq:04d}", "type": "message", "role": "assistant", "model": r["model"], "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": inp, "output_tokens": 0}}})]
    frames.append(sse("ping", {"type": "ping"}))
    for index, block in enumerate(anthropic_blocks(r, seq)):
        if block["type"] == "text":
            frames.append(sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}))
            for piece in pieces(block["text"]):
                frames.append(sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": piece}}))
        else:
            frames.append(sse("content_block_start", {"type": "content_block_start", "index": index,
                                                      "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}))
            for piece in pieces(dumps(block["input"])):
                frames.append(sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": piece}}))
        frames.append(sse("content_block_stop", {"type": "content_block_stop", "index": index}))
    frames.append(sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": TO_ANTHROPIC_STOP.get(r["stop"], "end_turn"), "stop_sequence": None},
                                        "usage": {"output_tokens": out}}))
    frames.append(sse("message_stop", {"type": "message_stop"}))
    return "".join(frames).encode("utf-8")


def openai_error(message: str, etype: str, code) -> dict:
    return {"error": {"message": message, "type": etype, "param": None, "code": code}}


def anthropic_error(etype: str, message: str) -> dict:
    return {"type": "error", "error": {"type": etype, "message": message}}


def openai_model_list(models: list[ModelEntry], created: int) -> dict:
    return {"object": "list", "data": [{"id": m.id, "object": "model", "created": created, "owned_by": m.provider.id} for m in models]}


def anthropic_model_list(models: list[ModelEntry], created_at: str) -> dict:
    data = [{"type": "model", "id": m.id, "display_name": m.display_name, "created_at": created_at} for m in models]
    return {"data": data, "has_more": False, "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None}


# ---------------------------------------------------------------------------------------------
# 8. The server
# ---------------------------------------------------------------------------------------------


def trial_now() -> tuple[int, str]:
    """(epoch seconds, ISO string) of MAKEFUN_TRIAL_NOW, else a fixed instant: never the clock."""
    raw = os.environ.get("MAKEFUN_TRIAL_NOW", "")
    if raw:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return int(dt.timestamp()), dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            if raw.isdigit():
                n = int(raw)
                n = n // 1000 if n > 10**11 else n
                return n, datetime.fromtimestamp(n, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return FIXED_CREATED, "2026-01-01T00:00:00Z"


class Aggregator:
    def __init__(self, world: World):
        self.world = world
        self.router = Router(world)
        self.lock = threading.Lock()  # requests are routed one at a time, in arrival order
        self.arrivals = 0
        self.created, self.created_at = trial_now()

    def requirements_for(self, api: str, body: dict) -> tuple[dict, dict]:
        """The manifest's row for this arrival, tightened by what the body itself asks for."""
        manifest = self.world.brief.manifest
        entry = None
        if self.arrivals < len(manifest):
            entry = manifest[self.arrivals]
        self.arrivals += 1
        req = requirements_of(entry.get("requires") if entry else None)
        feats = features_of(api, body)
        req["tools"] = req["tools"] or feats["tools"]
        req["vision"] = req["vision"] or feats["vision"]
        req["json"] = req["json"] or feats["json_mode"]
        need = feats["approx_tokens"]
        if req["min_context"] is None or need > req["min_context"]:
            req["min_context"] = need if need > 0 else req["min_context"]
        return req, feats

    def serve(self, api: str, body: dict, client_headers: dict) -> tuple[int, str, bytes]:
        """(status, content type, body) for one client inference request."""
        with self.lock:
            seq = self.arrivals + 1
            slot = self.router.slot_for(self.arrivals)
            req, feats = self.requirements_for(api, body)
            requested = body.get("model") if isinstance(body.get("model"), str) and body.get("model") not in ("", "auto") else None
            stream = body.get("stream") is True

            def call(entry: ModelEntry):
                url, headers, data = upstream_request(entry, api, body, client_headers, self.world.env_url)
                with self.router.state[entry.provider.id].lock:
                    ans = http("POST", url, headers, data)
                if ans.status == 200:
                    return 200, None, parse_answer(entry.provider.wire, ans, entry.id)
                return ans.status, ans.json(), None

            entry, result, last_status = self.router.route(req, api, feats, requested, call, slot)
            if entry is not None:
                log(f"#{seq} {api}{' stream' if stream else ''} -> {entry.provider.id}/{entry.id} ({entry.rating})")
                return self.render(api, stream, result, seq)
            msg = "no free model that meets this request could answer it"
            status = 429 if last_status in (402, 429) else 503
            if api == "openai":
                return status, "application/json", dumps(openai_error(msg, "rate_limit_error" if status == 429 else "api_error", "no_free_model")).encode()
            return status, "application/json", dumps(anthropic_error("rate_limit_error" if status == 429 else "overloaded_error", msg)).encode()

    def render(self, api: str, stream: bool, result: dict, seq: int) -> tuple[int, str, bytes]:
        if api == "openai":
            if stream:
                return 200, "text/event-stream; charset=utf-8", render_openai_sse(result, seq, self.created)
            return 200, "application/json", render_openai_json(result, seq, self.created)
        if stream:
            return 200, "text/event-stream; charset=utf-8", render_anthropic_sse(result, seq)
        return 200, "application/json", render_anthropic_json(result, seq)


def make_handler(agg: Aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "aggregator/1"
        sys_version = ""

        def log_message(self, fmt, *args):  # quiet: the trial's log is stderr
            pass

        def send(self, status: int, ctype: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            if ctype.startswith("text/event-stream"):
                self.send_header("cache-control", "no-cache")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
            self.wfile.flush()

        def client_wire(self, path: str) -> str:
            if path.startswith("/v1/messages"):
                return "anthropic"
            if self.headers.get("x-api-key") is not None and self.headers.get("authorization") is None:
                return "anthropic"
            return "openai"

        def error(self, wire: str, status: int, etype: str, message: str) -> None:
            body = anthropic_error(etype, message) if wire == "anthropic" else openai_error(message, etype, None)
            self.send(status, "application/json", dumps(body).encode())

        def read_body(self):
            try:
                length = int(self.headers.get("content-length") or "0")
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY_BYTES:
                return None
            raw = self.rfile.read(length)
            try:
                value = json.loads(raw.decode("utf-8"))
            except ValueError:
                return None
            return value if is_record(value) else None

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path.rstrip("/")
            wire = self.client_wire(path)
            models = agg.world.served_models()
            if path == "/v1/models":
                body = anthropic_model_list(models, agg.created_at) if wire == "anthropic" else openai_model_list(models, agg.created)
                return self.send(200, "application/json", dumps(body).encode())
            if path.startswith("/v1/models/"):
                mid = urllib.parse.unquote(path[len("/v1/models/"):])
                for m in models:
                    if m.id == mid:
                        if wire == "anthropic":
                            body = {"type": "model", "id": m.id, "display_name": m.display_name, "created_at": agg.created_at}
                        else:
                            body = {"id": m.id, "object": "model", "created": agg.created, "owned_by": m.provider.id}
                        return self.send(200, "application/json", dumps(body).encode())
                return self.error(wire, 404, "not_found_error" if wire == "anthropic" else "invalid_request_error", f"no model {mid}")
            if path in ("", "/health", "/healthz"):
                return self.send(200, "application/json", b'{"ok":true}')
            self.error(wire, 404, "not_found_error" if wire == "anthropic" else "invalid_request_error", f"no route GET {path}")

        do_HEAD = do_GET

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path.rstrip("/")
            if path not in ("/v1/chat/completions", "/v1/messages"):
                self.read_body()
                return self.error(self.client_wire(path), 404, "invalid_request_error", f"no route POST {path}")
            api = "openai" if path == "/v1/chat/completions" else "anthropic"
            body = self.read_body()
            if body is None:
                return self.error(api, 400, "invalid_request_error", "the request body is not one JSON object")
            headers = {k.lower(): v for k, v in self.headers.items()}
            try:
                status, ctype, data = agg.serve(api, body, headers)
            except Exception as err:
                log(f"serve failed: {type(err).__name__}: {err}")
                return self.error(api, 500, "api_error", "the aggregator failed to answer")
            self.send(status, ctype, data)

    return Handler


def register(env_url: str, port: int, catalog_text: str) -> bool:
    body = ('{"protocol_version":%d,"port":%d,"catalog":%s}' % (PROTOCOL_VERSION, port, catalog_text)).encode("utf-8")
    for attempt in range(20):
        ans = http("POST", f"{env_url}/register", {"content-type": "application/json"}, body, timeout=5.0)
        if ans.status == 200:
            reply = ans.json()
            log(f"registered on port {port}: {reply}")
            return True
        log(f"register answered {ans.status or ans.error} (attempt {attempt + 1})")
        if 400 <= ans.status < 500:
            return False
        time.sleep(0.25)
    return False


def poll_until_done(env_url: str, started: float) -> None:
    while True:
        if time.monotonic() - started > WALL_GUARD_SECONDS:
            log("the wall is near: exiting")
            return
        ans = http("GET", f"{env_url}/work", timeout=2.0)
        if ans.status == 200:
            work = ans.json()
            if is_record(work) and work.get("done") is True:
                log("the env says done")
                return
        time.sleep(POLL_SECONDS)


def build_world(do_probe: bool) -> World:
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    prompt = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if not prompt and not env_url:
        return World(Brief(), "")
    brief = read_brief(prompt)
    if brief.error:
        log(brief.error)
    world = World(brief, env_url)
    world.load(do_probe and env_url != "")
    return world


def finish(code: int) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


def main(argv: list[str]) -> int:
    started = time.monotonic()
    if "--list" in argv:
        world = build_world(True)
        sys.stdout.write(world.catalog_text() + "\n")
        sys.stdout.flush()
        return 0
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    world = build_world(True)
    agg = Aggregator(world)
    server = None
    for _ in range(5):
        server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(agg))
        if server.server_address[1] != 8080:
            break
        server.server_close()
    server.daemon_threads = True
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    log(f"listening on 0.0.0.0:{port}: {len(world.catalog['providers'])} providers, {len(world.served_models())} free models")
    signal.signal(signal.SIGTERM, lambda *_: finish(0))
    if not register(env_url, port, world.catalog_text()):
        log("register failed; polling anyway")
    poll_until_done(env_url, started)
    return 0


if __name__ == "__main__":
    finish(main(sys.argv[1:]))
