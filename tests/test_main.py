# SPDX-License-Identifier: MIT
"""Unit and in-process integration tests for main.py (python3 -m unittest discover -s tests)."""

import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("agg_main", os.path.join(REPO, "main.py"))
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)
agg.log = lambda message: None

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

DIRECTORY = {"schema": agg.PROVIDERS_SCHEMA, "providers": [
    {"id": "oa", "name": "OA", "wire": "openai", "auth": "bearer", "key_env": "OA_KEY", "key": "k-oa", "path": "/p/oa"},
    {"id": "an", "name": "AN", "wire": "anthropic", "auth": "bearer", "key_env": "AN_KEY", "key": "k-an", "path": "/p/an"},
    {"id": "ge", "name": "GE", "wire": "gemini", "auth": "x-goog-api-key", "key_env": "GE_KEY", "key": "k-ge", "path": "/p/ge"},
]}
RATINGS = {"schema": agg.RATINGS_SCHEMA, "ratings": [
    {"model": "oa-top", "provider": "oa", "rating_milli": 1475, "context_length": 204800, "tools": True, "vision": False, "json": True, "free": True},
    {"model": "oa-low", "provider": "oa", "rating_milli": 1200, "context_length": 32768, "tools": True, "vision": False, "json": False, "free": True},
    {"model": "an-mid", "provider": "an", "rating_milli": 1470, "context_length": 204800, "tools": True, "vision": True, "json": True, "free": True},
    {"model": "ge-vis", "provider": "ge", "rating_milli": 1458, "context_length": 1048576, "tools": True, "vision": True, "json": True, "free": True},
], "best": []}
WORKLOAD = {"schema": agg.WORKLOAD_SCHEMA, "manifest": [
    {"id": "r01", "api": "openai", "stream": False, "weight": 2, "requires": {"tools": True, "vision": True}},
], "bodies": {}}
LISTINGS = {
    "oa": [("oa-top", True, 204800), ("oa-paid", False, 200000), ("oa-low", True, 32768)],
    "an": [("an-mid", True, 204800)],
    "ge": [("ge-vis", True, 1048576)],
}


def brief_text() -> str:
    return "\n".join(["Prose before.", json.dumps(DIRECTORY), "more prose {not json}", json.dumps(RATINGS), json.dumps(WORKLOAD)])


# --- a fake provider host that echoes what it parsed, as the kit's env does ----------------------

def sha8(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]


def blocks_text(c):
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return ""
    return "\n".join(b["text"] for b in c if isinstance(b, dict) and b.get("type") == "text")


def echo(system, user, tools, max_tokens, stops, image, json_mode):
    return {"system_present": system != "", "system_sha8": sha8(system.encode()) if system else None, "user_sha8": sha8(user.encode()),
            "tools": tools, "max_tokens": max_tokens, "stops": stops, "image_sha8": image, "json_mode": json_mode}


def image_of_url(url):
    if url.startswith("data:"):
        return sha8(base64.b64decode(url.split(",", 1)[1]))
    return sha8(url.encode())


def openai_echo(body):
    sysparts, user, image = [], "", None
    for m in body.get("messages", []):
        if m.get("role") == "system":
            t = blocks_text(m.get("content"))
            if t:
                sysparts.append(t)
        elif m.get("role") == "user":
            user = blocks_text(m.get("content"))
            if isinstance(m.get("content"), list):
                for b in m["content"]:
                    if b.get("type") == "image_url":
                        image = image_of_url(b["image_url"]["url"])
    rf = body.get("response_format") or {}
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else (stop or [])
    return echo("\n".join(sysparts), user, [t["function"]["name"] for t in body.get("tools", [])],
                body.get("max_tokens", body.get("max_completion_tokens")), stops, image, rf.get("type") in ("json_object", "json_schema"))


def anthropic_echo(body):
    user, image = "", None
    for m in body.get("messages", []):
        if m.get("role") != "user":
            continue
        user = blocks_text(m.get("content"))
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "image":
                    s = b["source"]
                    image = sha8(base64.b64decode(s["data"])) if s["type"] == "base64" else image_of_url(s["url"])
    return echo(blocks_text(body.get("system")), user, [t["name"] for t in body.get("tools", [])], body.get("max_tokens"),
                body.get("stop_sequences", []), image, False)


def gemini_echo(body):
    def ptext(parts):
        return "\n".join(p["text"] for p in parts or [] if "text" in p)
    user, image = "", None
    for c in body.get("contents", []):
        if c.get("role") == "model":
            continue
        user = ptext(c.get("parts"))
        for p in c.get("parts", []):
            if "inlineData" in p:
                image = sha8(base64.b64decode(p["inlineData"]["data"]))
    gc = body.get("generationConfig", {})
    tools = [f["name"] for t in body.get("tools", []) for f in t.get("functionDeclarations", [])]
    return echo(ptext((body.get("systemInstruction") or {}).get("parts")), user, tools, gc.get("maxOutputTokens"),
                gc.get("stopSequences", []), image, gc.get("responseMimeType") == "application/json")


class FakeHost:
    """Serves the three providers; `refuse` maps provider id -> list of (status, message) to answer first."""

    def __init__(self):
        self.refuse = {"oa": [], "an": [], "ge": []}
        self.calls = []
        host = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def reply(self, status, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                pid = self.path.split("/")[2]
                models = LISTINGS[pid]
                if pid == "ge":
                    return self.reply(200, {"models": [{"name": f"models/{m}", "free": f, "inputTokenLimit": c} for m, f, c in models]})
                return self.reply(200, {"data": [{"id": m, "free": f, "context_length": c} for m, f, c in models]})

            def do_POST(self):
                pid = self.path.split("/")[2]
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                host.calls.append((pid, self.path, body, dict(self.headers)))
                if host.refuse[pid]:
                    status, msg = host.refuse[pid].pop(0)
                    return self.reply(status, {"error": {"message": msg, "code": "rate_limit"}})
                if pid == "oa":
                    e, model, stream = openai_echo(body), body["model"], body.get("stream") is True
                elif pid == "an":
                    e, model, stream = anthropic_echo(body), body["model"], body.get("stream") is True
                else:
                    e, model, stream = gemini_echo(body), self.path.split("/models/")[1].split(":")[0], ":streamGenerateContent" in self.path
                text = "ANSWER ECHO" + json.dumps(e, separators=(",", ":"))
                u_in, u_out = 7, 1 + len(text) // 4
                if pid == "oa":
                    if not stream:
                        return self.reply(200, {"id": "x", "object": "chat.completion", "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                                                "usage": {"prompt_tokens": u_in, "completion_tokens": u_out, "total_tokens": u_in + u_out}})
                    fr = [{"choices": [{"index": 0, "delta": {"content": text[i:i + 13]}, "finish_reason": None}]} for i in range(0, len(text), 13)]
                    fr.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                    if (body.get("stream_options") or {}).get("include_usage"):
                        fr.append({"choices": [], "usage": {"prompt_tokens": u_in, "completion_tokens": u_out}})
                    return self.reply(200, ("".join(f"data: {json.dumps(f)}\n\n" for f in fr) + "data: [DONE]\n\n").encode(), "text/event-stream")
                if pid == "an":
                    if not stream:
                        return self.reply(200, {"id": "m", "type": "message", "role": "assistant", "model": model, "content": [{"type": "text", "text": text}],
                                                "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": u_in, "output_tokens": u_out}})
                    ev = [("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": u_in, "output_tokens": 0}}}),
                          ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                          ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
                          ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                          ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": u_out}}),
                          ("message_stop", {"type": "message_stop"})]
                    return self.reply(200, "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in ev).encode(), "text/event-stream")
                meta = {"promptTokenCount": u_in, "candidatesTokenCount": u_out}
                if not stream:
                    return self.reply(200, {"candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": "STOP"}], "usageMetadata": meta})
                fr = [{"candidates": [{"content": {"parts": [{"text": text[i:i + 13]}], "role": "model"}}]} for i in range(0, len(text), 13)]
                fr.append({"candidates": [{"content": {"parts": [{"text": ""}]}, "finishReason": "STOP"}], "usageMetadata": meta})
                return self.reply(200, "".join(f"data: {json.dumps(f)}\n\n" for f in fr).encode(), "text/event-stream")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def make_world(host):
    world = agg.World(agg.parse_brief_text(brief_text()), host.url)
    world.load(True)
    return world


def openai_body(**extra):
    body = {"model": "auto", "max_tokens": 100, "temperature": 0.3,
            "messages": [{"role": "system", "content": "be careful"},
                         {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}]}],
            "tools": [{"type": "function", "function": {"name": "lookup", "description": "d", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}],
            "tool_choice": "auto", "stop": ["\n\n###"]}
    body.update(extra)
    return body


def anthropic_body(**extra):
    body = {"model": "auto", "max_tokens": 80, "system": "sys text",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}]}],
            "tools": [{"name": "lookup", "description": "d", "input_schema": {"type": "object", "properties": {}}}], "stop_sequences": ["\n\nHuman:"]}
    body.update(extra)
    return body


def answer_text(api, stream, data: bytes):
    if not stream:
        b = json.loads(data)
        return b["choices"][0]["message"]["content"] if api == "openai" else "".join(c["text"] for c in b["content"] if c["type"] == "text")
    text = ""
    for ev, d in agg.sse_frames(data.decode()):
        if d == "[DONE]":
            continue
        j = json.loads(d)
        if api == "openai":
            for c in j.get("choices", []):
                text += c.get("delta", {}).get("content") or ""
        elif j.get("type") == "content_block_delta" and j["delta"].get("type") == "text_delta":
            text += j["delta"]["text"]
    return text


class BriefTests(unittest.TestCase):
    def test_documents_found_by_schema(self):
        b = agg.parse_brief_text(brief_text())
        self.assertEqual([p.id for p in b.providers], ["oa", "an", "ge"])
        self.assertEqual(b.providers[1].auth, "bearer")
        self.assertEqual(len(b.ratings), 4)
        self.assertEqual(b.manifest[0]["id"], "r01")

    def test_kit_brief_parses_if_present(self):
        path = os.path.join(REPO, "..", "kit", "free-model-aggregator-01-kit", "practice", "trial-0", "input", "prompt.txt")
        if not os.path.exists(path):
            self.skipTest("kit not beside the repository")
        b = agg.read_brief(path)
        self.assertEqual(len(b.providers), 10)
        self.assertEqual(len(b.manifest), 20)

    def test_list_outside_a_trial(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("MAKEFUN_")}
        out = subprocess.run([sys.executable, "-P", "-s", os.path.join(REPO, "main.py"), "--list"], env=env, capture_output=True, timeout=30)
        self.assertEqual(out.returncode, 0)
        self.assertEqual(out.stdout, b'{"providers":[]}\n')


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = FakeHost()
        cls.world = make_world(cls.host)

    @classmethod
    def tearDownClass(cls):
        cls.host.close()

    def test_catalog_keeps_listing_order_and_paths(self):
        oa = self.world.catalog["providers"][0]
        self.assertEqual(oa["base_url"], "/p/oa")
        self.assertEqual([m["id"] for m in oa["models"]], ["oa-top", "oa-paid", "oa-low"])
        self.assertEqual([m["free"] for m in oa["models"]], [True, False, True])
        self.assertEqual(self.world.catalog["providers"][2]["models"][0], {"id": "ge-vis", "free": True, "context_length": 1048576})

    def test_catalog_text_is_stable(self):
        self.assertEqual(self.world.catalog_text(), make_world(self.host).catalog_text())

    def test_served_models_are_free_only(self):
        self.assertEqual([m.id for m in self.world.served_models()], ["oa-top", "oa-low", "an-mid", "ge-vis"])



def brute_hazard(lo, hi, history, n_providers=10):
    """P(next call refused | history) by direct enumeration of the generator's draw (independent of main.py)."""
    import itertools
    total = refused = 0.0
    for budget in range(lo, hi + 1):
        span = range(2, budget)
        for count in (1, 2):
            sets = list(itertools.combinations(span, count))
            for s in sets:
                for designated in (False, True):
                    fails = set(s) if (not designated or 2 in s) else {2, *s[1:]}
                    for outage in (None, *span):
                        p = (1 / (hi - lo + 1)) * 0.5 / len(sets)
                        p *= (1 / n_providers) if designated else (1 - 1 / n_providers)
                        p *= (1 - 1 / n_providers) if outage is None else (1 / n_providers) / len(span)

                        def outcome(c):
                            if c > budget:
                                return "budget"
                            if outage is not None and c >= outage:
                                return "out"
                            return "sched" if c in fails else "ok"
                        if all(outcome(i + 1) == h for i, h in enumerate(history)):
                            total += p
                            refused += p if outcome(len(history) + 1) != "ok" else 0.0
    return refused / total


class HazardAndGuardTests(unittest.TestCase):
    def setUp(self):
        self.brief = agg.parse_brief_text(brief_text())
        self.brief.manifest = [
            {"id": "q1", "api": "openai", "weight": 2, "requires": {}},
            {"id": "q2", "api": "openai", "weight": 1, "requires": {}},
        ]
        self.brief.best = {"q1": 1475, "q2": 1475}
        self.world = agg.World(self.brief, "")
        self.world.load(False)
        self.router = agg.Router(self.world)
        self.feats = {"tools": False, "vision": False, "json_mode": False, "max_tokens": 100, "approx_tokens": 0}

    def test_hazard_matches_brute_force(self):
        h = agg.hazard_model()
        for pid in ("zai", "cerebras"):
            lo, hi = agg.SCHEDULES[pid]
            for hist in [(), ("ok",), ("ok", "ok"), ("ok", "sched"), ("ok", "sched", "ok"), ("ok", "ok", "sched", "ok")]:
                self.assertAlmostEqual(h.hazard(pid, hist), brute_hazard(lo, hi, hist), places=12, msg=(pid, hist))

    def test_first_call_is_safe_and_fallbacks(self):
        h = agg.hazard_model()
        for pid in agg.SCHEDULES:
            self.assertEqual(h.hazard(pid, ()), 0.0)
        self.assertEqual(h.hazard("unknown", ()), h.FALLBACK_FIRST)
        self.assertEqual(h.hazard("unknown", ("ok",)), h.FALLBACK)
        self.assertEqual(h.hazard("zai", ("ok", "sched", "sched", "sched")), h.FALLBACK)  # no scenario explains it
        wild = h.hazard("kilol", ("ok", "other"))
        self.assertTrue(0.0 < wild < 1.0)
        self.assertNotEqual(wild, h.FALLBACK)

    def test_loss_cap(self):
        bound = 2 * 1475 + 1475
        self.assertEqual(self.router.cap, bound - -(-98 * bound // 100))
        self.brief.best = {"q1": 1475}
        self.assertIsNone(self.router.loss_cap(self.brief))

    def test_history_is_recorded(self):
        self.router.note_success("oa")
        self.router.note_failure("oa", "oa-top", 429, {"error": {"message": "OA refused this call"}})
        self.router.note_failure("oa", "oa-top", 500, {"error": {"message": "upstream hiccup"}})
        self.router.note_failure("oa", "oa-top", 429, {"error": {"message": "x", "code": "concurrent_requests"}})
        self.assertEqual(self.router.state["oa"].history, ("ok", "sched", "other"))
        self.router.note_failure("an", "an-mid", 429, {"error": {"message": "AN is out"}})
        self.router.note_failure("ge", "ge-vis", 429, {"error": {"message": "GE's free-tier budget is spent"}})
        self.assertEqual((self.router.state["an"].history, self.router.state["ge"].history), (("out",), ("budget",)))

    def test_guard_trades_rating_for_a_fresh_provider_only_within_the_cap(self):
        self.router.note_success("oa")  # oa now carries the fallback hazard, an/ge are fresh
        tier = self.router.candidates(agg.requirements_of({}), "openai", self.feats, None)
        slot = self.router.slot_for(0)
        self.assertEqual((slot.weight, slot.best, slot.remaining), (2, 1475, 1))
        self.assertEqual(self.router.choose(tier, slot).id, "an-mid")  # 2 * 5 loss fits the cap
        self.router.loss = self.router.cap - self.router.RESERVE * slot.remaining - 9
        self.assertEqual(self.router.choose(tier, slot).id, "oa-top")  # 10 more would not
        self.assertEqual(self.router.choose(tier, None).id, "oa-top")  # no manifest row: top rating only

    def test_hazard_and_lambda_tie_break(self):
        tier = self.router.candidates(agg.requirements_of({}), "openai", self.feats, None)
        self.assertEqual(self.router.choose(tier, self.router.slot_for(0)).id, "oa-top")  # all fresh: no gap wins

    def test_scheduled_refusal_skips_provider_unless_alone(self):
        calls = []
        script = {"oa": [429], "an": [], "ge": [429]}

        def call(entry):
            pid = entry.provider.id
            calls.append(pid)
            if script[pid]:
                return script[pid].pop(0), {"error": {"message": f"{pid} refused this call"}}, None
            return 200, None, {"ok": True}

        entry, _, _ = self.router.route(agg.requirements_of({}), "openai", self.feats, None, call, self.router.slot_for(0))
        self.assertEqual(calls, ["oa", "an"])  # not straight back to oa
        self.assertEqual(self.router.loss, 2 * 5)
        calls.clear()
        entry, _, _ = self.router.route(agg.requirements_of({"min_context": 300000}), "openai", self.feats, None, call, self.router.slot_for(1))
        self.assertEqual((calls, entry.id), (["ge", "ge"], "ge-vis"))  # the only candidate is retried

    def test_permanent_refusal_retires(self):
        calls = []

        def call(entry):
            calls.append(entry.provider.id)
            if entry.provider.id == "oa":
                return 429, {"error": {"message": "OA is out"}}, None
            return 200, None, {"ok": True}

        req = agg.requirements_of({"json": True})
        self.router.route(req, "openai", self.feats, None, call, self.router.slot_for(0))
        self.router.route(req, "openai", self.feats, None, call, self.router.slot_for(1))
        self.assertEqual(calls[:2], ["oa", "an"])
        self.assertEqual(calls[2], "ge")  # the dead provider is never called again; ge is fresh and fits the cap
        self.assertTrue(self.router.state["oa"].dead)


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.host = FakeHost()
        self.world = make_world(self.host)
        self.router = agg.Router(self.world)

    def tearDown(self):
        self.host.close()

    def feats(self, **kw):
        f = {"tools": False, "vision": False, "json_mode": False, "max_tokens": 100, "approx_tokens": 0}
        f.update(kw)
        return f

    def test_best_free_model_first_never_paid(self):
        req = agg.requirements_of({})
        ids = [e.id for e in self.router.candidates(req, "openai", self.feats(), None)]
        self.assertEqual(ids[0], "oa-top")
        self.assertNotIn("oa-paid", ids)

    def test_requirements_filter(self):
        ids = [e.id for e in self.router.candidates(agg.requirements_of({"vision": True, "tools": True}), "openai", self.feats(), None)]
        self.assertEqual(ids, ["an-mid", "ge-vis"])
        ids = [e.id for e in self.router.candidates(agg.requirements_of({"min_context": 300000}), "openai", self.feats(), None)]
        self.assertEqual(ids, ["ge-vis"])

    def test_json_mode_avoids_anthropic_wire(self):
        req = agg.requirements_of({"json": True, "vision": True})
        ids = [e.id for e in self.router.candidates(req, "openai", self.feats(json_mode=True), None)]
        self.assertEqual(ids, ["ge-vis", "an-mid"])

    def test_refusal_classification(self):
        self.assertEqual(self.router.note_failure("oa", "oa-top", 429, {"error": {"message": "OA refused this call"}}), "transient")
        self.assertFalse(self.router.state["oa"].dead)
        self.assertEqual(self.router.note_failure("oa", "oa-top", 429, {"error": {"message": "OA's free-tier budget is spent"}}), "dead")
        self.assertTrue(self.router.state["oa"].dead)
        self.assertEqual(self.router.note_failure("an", "an-mid", 429, {"type": "error", "error": {"type": "rate_limit_error", "message": "AN is out"}}), "dead")
        calls = self.router.state["ge"].calls
        self.assertEqual(self.router.note_failure("ge", "ge-vis", 429, {"error": {"message": "x", "code": "concurrent_requests"}}), "busy")
        self.assertEqual(self.router.state["ge"].calls, calls)

    def test_failover_serves_on_next_model(self):
        self.world.brief.manifest = []  # requirements from the bodies alone
        agg_ = agg.Aggregator(self.world)
        self.router = agg_.router
        self.host.refuse["oa"] = [(402, "OA's free-tier budget is spent")]
        status, _, data = agg_.serve("openai", {"model": "auto", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}, {})
        self.assertEqual(status, 200)
        self.assertEqual([c[0] for c in self.host.calls], ["oa", "an"])
        self.assertTrue(self.router.state["oa"].dead)
        status, _, _ = agg_.serve("openai", {"model": "auto", "max_tokens": 10, "messages": [{"role": "user", "content": "again"}]}, {})
        self.assertEqual([c[0] for c in self.host.calls], ["oa", "an", "an"])  # the dead provider is never called again


class WireTests(unittest.TestCase):
    """Every client wire through every upstream wire, JSON and SSE: the echo must match the client."""

    @classmethod
    def setUpClass(cls):
        cls.host = FakeHost()
        cls.world = make_world(cls.host)

    @classmethod
    def tearDownClass(cls):
        cls.host.close()

    def run_case(self, api, stream, provider):
        entry = next(e for e in self.world.pool if e.provider.id == provider and e.free)
        body = (openai_body if api == "openai" else anthropic_body)(**({"stream": True} if stream else {}))
        url, headers, data = agg.upstream_request(entry, api, body, {"anthropic-version": "2023-06-01"}, self.host.url)
        ans = agg.http("POST", url, headers, data)
        self.assertEqual(ans.status, 200)
        result = agg.parse_answer(entry.provider.wire, ans, entry.id)
        self.assertTrue(result["ok"])
        a = agg.Aggregator(self.world)
        status, ctype, out = a.render(api, stream, result, 1)
        text = answer_text(api, stream, out)
        self.assertTrue(text.startswith("ANSWER ECHO"), text[:40])
        got = json.loads(text[len("ANSWER ECHO"):])
        want = openai_echo(body) if api == "openai" else anthropic_echo(body)
        self.assertEqual(got, want, f"{api} stream={stream} via {provider}")
        self.assertIsNotNone(result["usage"])
        return out

    def test_all_combinations(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                for provider in ("oa", "an", "ge"):
                    with self.subTest(api=api, stream=stream, provider=provider):
                        self.run_case(api, stream, provider)

    def test_json_mode_reaches_gemini(self):
        entry = next(e for e in self.world.pool if e.provider.id == "ge")
        body = openai_body(response_format={"type": "json_object"})
        _, _, data = agg.upstream_request(entry, "openai", body, {}, self.host.url)
        self.assertEqual(json.loads(data)["generationConfig"]["responseMimeType"], "application/json")
        self.assertNotIn("additionalProperties", json.dumps(json.loads(data)["tools"]))

    def test_anthropic_stream_event_order(self):
        out = self.run_case("anthropic", True, "oa")
        names = [ev for ev, _ in agg.sse_frames(out.decode())]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        self.assertLess(names.index("content_block_stop"), names.index("message_delta"))

    def test_openai_stream_ends_with_usage_and_done(self):
        out = self.run_case("openai", True, "ge")
        frames = [d for _, d in agg.sse_frames(out.decode())]
        self.assertEqual(frames[-1], "[DONE]")
        self.assertIn("usage", json.loads(frames[-2]))
        self.assertEqual(json.loads(frames[-3])["choices"][0]["finish_reason"], "stop")


class RenderTests(unittest.TestCase):
    def test_tool_calls_and_stop_mapping(self):
        r = {"text": "", "tool_calls": [{"id": "c1", "name": "lookup", "arguments": '{"a":1}'}], "stop": "tool", "usage": {"input": 3, "output": 4}, "model": "m", "ok": True}
        o = json.loads(agg.render_openai_json(r, 1, 0))
        self.assertEqual(o["choices"][0]["finish_reason"], "tool_calls")
        self.assertIsNone(o["choices"][0]["message"]["content"])
        a = json.loads(agg.render_anthropic_json(r, 1))
        self.assertEqual(a["stop_reason"], "tool_use")
        self.assertEqual(a["content"][0], {"type": "tool_use", "id": "c1", "name": "lookup", "input": {"a": 1}})
        events = agg.sse_frames(agg.render_anthropic_sse(r, 1).decode())
        partial = "".join(json.loads(d)["delta"].get("partial_json", "") for e, d in events if e == "content_block_delta")
        self.assertEqual(json.loads(partial), {"a": 1})

    def test_stop_families(self):
        for upstream, oa, an in (("max", "length", "max_tokens"), ("filter", "content_filter", "refusal"), ("end", "stop", "end_turn")):
            self.assertEqual(agg.TO_OPENAI_FINISH[upstream], oa)
            self.assertEqual(agg.TO_ANTHROPIC_STOP[upstream], an)
        self.assertEqual(agg.GEMINI_STOP["MAX_TOKENS"], "max")


if __name__ == "__main__":
    unittest.main()
