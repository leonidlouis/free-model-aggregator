#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run main.py against a kit's practice instances with the kit's own env and scorer (node).

    python3 tools/practice.py <kit dir> [trial ...] [--out out/]

For each trial it starts program/env/server.mjs on a free loopback port, seeded from
practice/trial-N/input/env, runs `python3 -P -s main.py` against it, sends the env SIGTERM so it
writes state.json, then scores it with program/scorer/score.mjs against practice/trial-N/reference.
It prints every score column and the sha256 of the state export (compare two runs for replay).
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def seeds(kit: str) -> list[str]:
    out = []
    with open(os.path.join(kit, "rules", "instance.yaml"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("seed:"):
                out.append(line.split(":", 1)[1].strip().strip('"'))
    return out


def run_trial(kit: str, t: int, out_root: str) -> dict:
    trial = os.path.join(kit, "practice", f"trial-{t}")
    state = os.path.abspath(os.path.join(out_root, f"trial-{t}", "state"))
    scores = os.path.abspath(os.path.join(out_root, f"trial-{t}", "scores"))
    os.makedirs(state, exist_ok=True)
    for name in ("ready",):
        try:
            os.remove(os.path.join(state, name))
        except FileNotFoundError:
            pass
    port = free_port()
    env = dict(os.environ, MAKEFUN_STATE_DIR=state, MAKEFUN_ENV_DATA_DIR=os.path.join(trial, "input", "env"), MAKEFUN_ENV_PORT=str(port))
    srv = subprocess.Popen(["node", os.path.join(kit, "program", "env", "server.mjs")], env=env, stdout=subprocess.DEVNULL, stderr=open(os.path.join(state, "env.log"), "w"))
    for _ in range(100):
        if os.path.exists(os.path.join(state, "ready")):
            break
        time.sleep(0.05)
    seed_list = seeds(kit)
    bot_env = {k: v for k, v in os.environ.items() if not k.startswith("MAKEFUN_")}
    bot_env.update(MAKEFUN_ENV_URL=f"http://127.0.0.1:{port}", MAKEFUN_PROMPT_FILE=os.path.join(trial, "input", "prompt.txt"),
                   MAKEFUN_TRIAL_SEED=seed_list[t] if t < len(seed_list) else "0" * 64, MAKEFUN_TRIAL_NOW="2026-11-14T00:00:00Z")
    started = time.monotonic()
    with open(os.path.join(state, "bot.log"), "w") as bot_log:
        list_out = subprocess.run([sys.executable, "-P", "-s", os.path.join(REPO, "main.py"), "--list"], env=bot_env, capture_output=True, timeout=60)
        bot = subprocess.run([sys.executable, "-P", "-s", os.path.join(REPO, "main.py")], env=bot_env, stdout=bot_log, stderr=bot_log, timeout=200)
    elapsed = time.monotonic() - started
    srv.send_signal(15)
    srv.wait(timeout=10)
    export = os.path.join(state, "export")
    res = subprocess.run(["node", os.path.join(kit, "program", "scorer", "score.mjs")], capture_output=True, text=True,
                         env=dict(os.environ, MAKEFUN_OUTPUT_DIR=export, MAKEFUN_REFERENCE_DIR=os.path.join(trial, "reference"), MAKEFUN_SCORES_DIR=scores))
    with open(os.path.join(export, "state.json"), "rb") as f:
        state_bytes = f.read()
    st = json.loads(state_bytes)
    with open(os.path.join(scores, "scores.json")) as f:
        sc = json.load(f)
    registered = json.dumps(st["catalog"], separators=(",", ":"))
    return {
        "trial": t, "exit": bot.returncode, "seconds": round(elapsed, 1), "scores": sc, "scorer": res.stdout.strip() + res.stderr.strip(),
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "list_matches_register": list_out.stdout.decode().strip() == registered,
        "invalid": [(r["id"], r["error"]) for r in st["requests"] if not r["valid"]],
        "served_by": [(r["id"], r["model"], r["upstream_calls"]) for r in st["requests"]],
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    kit = os.path.abspath(argv[0])
    out_root = os.path.join(REPO, "out")
    rest = argv[1:]
    if "--out" in rest:
        i = rest.index("--out")
        out_root = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    trials = [int(x) for x in rest] or [0, 1, 2]
    for t in trials:
        r = run_trial(kit, t, out_root)
        print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
