"""Time each jitted function's first compile, one at a time, in a cold process.

Loads copies of bitgen and the agent with their warm() calls stripped, so nothing is
compiled until this script asks for it and every piece can be timed on its own. The
stripped copies go in a scratch directory that is placed FIRST on sys.path, otherwise
the real warmed modules win and every measurement lands in the wrong bucket.
"""

import os
import re
import shutil
import sys
import time

ROOT = "/home/harry/aichessathon-starter"
SCRATCH = "/tmp/compile_profile"

shutil.rmtree(SCRATCH, ignore_errors=True)
os.makedirs(SCRATCH)
for name, path in (
    ("bitgen", f"{ROOT}/bitgen.py"),
    ("agentmod", f"{ROOT}/baselines/invictus/invictus_moveGen/agent.py"),
):
    with open(path) as handle:
        src = re.sub(r"^_?warm\(\)$", "", handle.read(), flags=re.M)
    with open(f"{SCRATCH}/{name}.py", "w") as handle:
        handle.write(src)

sys.path.insert(0, ROOT)
sys.path.insert(0, SCRATCH)  # must come first, or the warmed originals shadow these

t0 = time.monotonic()
import numba  # noqa: E402,F401
import numpy as np  # noqa: E402,F401

t_numba = time.monotonic() - t0

t0 = time.monotonic()
import bitgen as bg  # noqa: E402

t_tables = time.monotonic() - t0
assert bg.__file__.startswith(SCRATCH), f"imported the wrong bitgen: {bg.__file__}"

t0 = time.monotonic()
import agentmod as ag  # noqa: E402

t_agent_body = time.monotonic() - t0

bb, sq, st = bg.from_fen(bg.STARTING_FEN)
buf, hist = bg.new_buffers()
steps = []


def step(label, fn):
    t = time.monotonic()
    fn()
    steps.append((label, time.monotonic() - t))


step("gen_moves_ex  (_generate,_gen_pawns,_danger,_pinned,attackers_to)",
     lambda: bg.gen_moves_ex(bb, st, buf[0], 0))
step("make_move", lambda: bg.make_move(bb, sq, st, int(buf[0, 0]), hist, 0))
step("unmake_move", lambda: bg.unmake_move(bb, sq, st, int(buf[0, 0]), hist, 0))
step("gen_moves", lambda: bg.gen_moves(bb, st, buf[0], 0))
step("gen_captures", lambda: bg.gen_captures(bb, st, buf[0], 0))
step("in_check", lambda: bg.in_check(bb, st))
step("queen_attacks", lambda: bg.queen_attacks(0, bb[14]))
step("perft", lambda: bg.perft(bb, sq, st, 2, buf, hist, 0))
step("perft_nodes", lambda: bg.perft_nodes(bb, sq, st, 2, buf, hist, 0))

ag.CTRL[0] = time.monotonic() + 100.0
ag.CTRL[1] = 0.0
ag.CTRL[2] = 0.0
ag.CTRL[3] = ag.CHECK_INTERVAL
step("quiescence  (evaluate,_tick,_now,_select,_keep_captures)",
     lambda: ag.quiescence(bb, sq, st, -ag.INF, ag.INF, 1, 0,
                           ag.BUF, ag.SCORES, ag.HIST, ag.CTRL))
step("negamax",
     lambda: ag.negamax(bb, sq, st, -ag.INF, ag.INF, 1, 0,
                        ag.BUF, ag.SCORES, ag.HIST, ag.CTRL))
step("think", lambda: ag.think(bb, sq, st, ag.BUF, ag.SCORES, ag.HIST, ag.CTRL, 1))

width = 64
print(f"{'import numpy + numba':{width}s} {t_numba:6.2f}s")
print(f"{'bitgen module body (tables + magic search)':{width}s} {t_tables:6.2f}s")
print(f"{'agent module body':{width}s} {t_agent_body:6.2f}s")
for label, dt in steps:
    print(f"{label:{width}s} {dt:6.2f}s")
total = t_numba + t_tables + t_agent_body + sum(d for _, d in steps)
print(f"{'TOTAL':{width}s} {total:6.2f}s")
