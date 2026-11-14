#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay main.py's Router against drawn instances, offline, with the env's refusal rules.

    python3 tools/draw.py <kit> /tmp/inst 1000            # draw instances with the kit's generator
    python3 tools/routesim.py /tmp/inst [NAME=VALUE ...] [--main path/to/main.py]

Each instance dir holds input/prompt.txt, input/env/providers.json, input/env/workload.json and
reference/ratings.json (as the kit's generator writes them). Requests go through the same
Aggregator.requirements_for and Router.route as a live trial, with the workload's real bodies;
only the upstream call is simulated (every unrefused call is assumed to pass the wire checks:
tools/practice.py checks those against the real env). NAME=VALUE overrides a Router class
attribute (LAMBDA, RESERVE, ...). --main replays another main.py (e.g. `git show HEAD~1:main.py`).
Prints the mean calls, their spread, and the share of trials whose quality column is below 98.
"""

import glob
import importlib.util
import inspect
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path: str):
    spec = importlib.util.spec_from_file_location("agg_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.log = lambda message: None
    return mod


def simulate(agg, inst_dir: str, params: dict) -> tuple[int, int]:
    """(quality column, upstream calls) for one instance."""
    with open(os.path.join(inst_dir, "input", "prompt.txt"), encoding="utf-8") as f:
        brief = agg.parse_brief_text(f.read())
    with open(os.path.join(inst_dir, "input", "env", "providers.json")) as f:
        env = {p["id"]: p for p in json.load(f)["providers"]}
    with open(os.path.join(inst_dir, "input", "env", "workload.json")) as f:
        bodies = json.load(f)["bodies"]
    with open(os.path.join(inst_dir, "reference", "ratings.json")) as f:
        best = {b["id"]: b["rating_milli"] for b in json.load(f)["best"]}
    world = agg.World(brief, "")
    world.listings = {pid: [{"id": m["id"], "display_name": m["id"], "free": m["free"], "context_length": m["context_length"],
                             "tools": m["tools"], "vision": m["vision"], "json": m["json"]} for m in p["models"]] for pid, p in env.items()}
    for p in brief.providers:
        for i, m in enumerate(world.listings[p.id]):
            world.pool.append(agg.ModelEntry(p, m, brief.ratings.get((p.id, m["id"])), i))
    aggr = agg.Aggregator(world)
    router = aggr.router
    for k, v in params.items():
        setattr(router, k, v)
    if hasattr(router, "loss_cap"):
        router.cap = router.loss_cap(brief)
    takes_slot = "slot" in inspect.signature(router.route).parameters
    counts = {pid: 0 for pid in env}
    calls = 0
    earned = bound = 0
    for m in brief.manifest:
        body = bodies[m["id"]]
        slot = router.slot_for(aggr.arrivals) if takes_slot else None
        req, feats = aggr.requirements_for(m["api"], body)
        requested = body.get("model") if isinstance(body.get("model"), str) and body.get("model") not in ("", "auto") else None

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

        if takes_slot:
            entry, _result, _ = router.route(req, m["api"], feats, requested, call, slot)
        else:
            entry, _result, _ = router.route(req, m["api"], feats, requested, call)
        bound += m["weight"] * best[m["id"]]
        if entry is not None:
            earned += m["weight"] * min(entry.rating, best[m["id"]])
    return (100 * earned) // bound, calls


def main(argv):
    main_py = os.path.join(REPO, "main.py")
    if "--main" in argv:
        i = argv.index("--main")
        main_py = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        print(__doc__)
        return 2
    agg = load(main_py)
    dirs = sorted(glob.glob(os.path.join(argv[0], "out*")))
    params = {}
    for kv in argv[1:]:
        k, v = kv.split("=", 1)
        params[k] = float(v) if "." in v else int(v)
    qs, cs = [], []
    for d in dirs:
        q, c = simulate(agg, d, params)
        qs.append(q)
        cs.append(c)
    n = len(dirs)
    spread = {c: cs.count(c) for c in sorted(set(cs))}
    print(f"{os.path.relpath(main_py)} {params} n={n} mean_calls={statistics.mean(cs):.3f} min={min(cs)} max={max(cs)} "
          f"quality_mean={statistics.mean(qs):.3f} quality_min={min(qs)} share_below98={sum(q < 98 for q in qs) / n:.4f}")
    print(f"calls histogram {spread}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
