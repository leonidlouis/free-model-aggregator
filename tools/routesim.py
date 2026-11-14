#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay main.py's Router against drawn instances, offline, with the env's refusal rules.

    python3 tools/routesim.py <dir of drawn instances> [RISK_MILLI HAZARD_BASE HAZARD_AFTER_FAIL]

Each instance dir holds input/prompt.txt, input/env/providers.json and reference/ratings.json (as
the kit's generator writes them). Only routing is simulated: every successful call is assumed to
pass the wire checks (tools/practice.py checks those against the real env). Prints the mean
quality column, the mean calls and the share of trials whose quality column falls below 99.
"""

import glob
import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("agg_main", os.path.join(REPO, "main.py"))
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)
agg.log = lambda message: None


def simulate(inst_dir: str, params: dict) -> tuple[int, int, int]:
    with open(os.path.join(inst_dir, "input", "prompt.txt"), encoding="utf-8") as f:
        brief = agg.parse_brief_text(f.read())
    with open(os.path.join(inst_dir, "input", "env", "providers.json")) as f:
        env = {p["id"]: p for p in json.load(f)["providers"]}
    with open(os.path.join(inst_dir, "reference", "ratings.json")) as f:
        best = {b["id"]: b["rating_milli"] for b in json.load(f)["best"]}
    world = agg.World(brief, "")
    world.listings = {pid: [{"id": m["id"], "display_name": m["id"], "free": m["free"], "context_length": m["context_length"],
                             "tools": m["tools"], "vision": m["vision"], "json": m["json"]} for m in p["models"]] for pid, p in env.items()}
    for p in brief.providers:
        for i, m in enumerate(world.listings[p.id]):
            world.pool.append(agg.ModelEntry(p, m, brief.ratings.get((p.id, m["id"])), i))
    router = agg.Router(world)
    for k, v in params.items():
        setattr(router, k, v)
    counts = {pid: 0 for pid in env}
    calls = 0
    earned = bound = 0
    for m in brief.manifest:
        req = agg.requirements_of(m["requires"])
        feats = {"tools": req["tools"], "vision": req["vision"], "json_mode": req["json"] and m["api"] == "openai", "max_tokens": 100, "approx_tokens": 0}
        weight = m["weight"]

        def call(entry):
            nonlocal calls
            calls += 1
            p = env[entry.provider.id]
            counts[p["id"]] += 1
            n = counts[p["id"]]
            if n > p["budget"]:
                msg = f"{p['name']}'s free-tier budget is spent"
            elif p["outage_from"] is not None and n >= p["outage_from"]:
                msg = f"{p['name']} is out"
            elif n in p["fail_calls"]:
                msg = f"{p['name']} refused this call"
            else:
                return 200, None, {"ok": True}
            return 429, {"error": {"message": msg, "code": "rate_limit"}}, None

        entry, _result, _ = router.route(req, m["api"], feats, None, call)
        bound += weight * best[m["id"]]
        if entry is not None:
            earned += weight * min(entry.rating, best[m["id"]])
    return (100 * earned) // bound, calls, earned


def main(argv):
    dirs = sorted(glob.glob(os.path.join(argv[0], "out*")))
    names = ["RISK_MILLI", "HAZARD_BASE", "HAZARD_AFTER_FAIL"]
    params = {n: float(v) for n, v in zip(names, argv[1:])}
    qs, cs = [], []
    for d in dirs:
        q, c, _ = simulate(d, params)
        qs.append(q)
        cs.append(c)
    n = len(dirs)
    print(f"{params} n={n} quality={sum(qs)/n:.3f} calls={sum(cs)/n:.2f} below99={sum(q < 99 for q in qs)/n:.3f} max_calls={max(cs)}")


if __name__ == "__main__":
    main(sys.argv[1:])
