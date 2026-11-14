#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Draw instances with a kit's own generator (node), for tools/routesim.py.

    python3 tools/draw.py <kit dir> <out dir> <count> [first]

Instance i gets the seed sha256("routesim/<i>") (64 hex chars, agent-fun/instance-seed/v1) and is
written to <out dir>/out<i:05d>/ exactly as program/generator/draw.mjs writes /output. Existing
instance dirs are kept, so the draw can be extended.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    kit, out, count = os.path.abspath(argv[0]), os.path.abspath(argv[1]), int(argv[2])
    first = int(argv[3]) if len(argv) > 3 else 0
    draw = os.path.join(kit, "program", "generator", "draw.mjs")
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as seed_dir:
        for i in range(first, first + count):
            dest = os.path.join(out, f"out{i:05d}")
            if os.path.exists(os.path.join(dest, "input", "env", "providers.json")):
                continue
            seed = hashlib.sha256(f"routesim/{i}".encode()).hexdigest()
            with open(os.path.join(seed_dir, "seed.json"), "w") as f:
                json.dump({"schema": "agent-fun/instance-seed/v1", "seed": seed, "purpose": "routesim", "k": i}, f)
            env = dict(os.environ, MAKEFUN_INPUT_DIR=seed_dir, MAKEFUN_OUTPUT_DIR=dest)
            subprocess.run(["node", draw], env=env, check=True, stdout=subprocess.DEVNULL)
    print(f"{count} instances in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
