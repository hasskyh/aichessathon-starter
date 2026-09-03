"""Search throughput: the jitted bitboard search against the python-chess original.

Fixed depth, not fixed time. A time-limited search does however many nodes it can, and
on a noisy laptop that swings 25% run to run; a fixed depth does exactly the same work
every time, so the only thing left varying is the clock. Times are best-of-N for the
same reason.

Both engines run the same algorithm, so the root score must agree. Node counts will
not: the two generators emit moves in different orders, so alpha-beta cuts in different
places. Nodes per second is the honest comparison.
"""

import math
import os
import sys
import time

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import chess  # noqa: E402

import bitgen  # noqa: E402

sys.path.insert(0, os.path.join(root, "baselines/invictus/invictus_moveGen"))
import agent as fast  # noqa: E402

sys.path.insert(0, os.path.join(root, "baselines/invictus/invictus_quiesce"))
del sys.modules["agent"]
import agent as slow  # noqa: E402

POSITIONS = [
    ("startpos", bitgen.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("midgame", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/2NP1N2/PPPQ1PPP/R4RK1 w - - 0 10"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]

DEPTH = 3
REPEATS = 3


def run_fast(fen, depth):
    bb, sq, st = bitgen.from_fen(fen)
    fast.CTRL[0] = time.monotonic() + 1e6  # effectively no deadline
    fast.CTRL[1] = 0.0
    fast.CTRL[2] = 0.0
    fast.CTRL[3] = fast.CHECK_INTERVAL
    start = time.perf_counter()
    score = fast.negamax(
        bb, sq, st, -fast.INF, fast.INF, depth, 0,
        fast.BUF, fast.SCORES, fast.HIST, fast.CTRL,
    )
    return time.perf_counter() - start, int(fast.CTRL[1]), int(score)


def run_slow(fen, depth):
    board = chess.Board(fen)
    slow._deadline = time.monotonic() + 1e6
    slow._nodes = 0
    start = time.perf_counter()
    score = slow.negamax(-math.inf, math.inf, board, depth)
    return time.perf_counter() - start, slow._nodes, int(score)


print(f"--- fixed depth {DEPTH}, best of {REPEATS} ---\n")
print(f"{'position':10s} {'engine':14s} {'nodes':>12s} {'time':>9s} {'nps':>12s} {'score':>9s}")
ratios = []
for name, fen in POSITIONS:
    row = {}
    for label, runner in (("bitgen+numba", run_fast), ("python-chess", run_slow)):
        best = None
        for _ in range(REPEATS):
            elapsed, nodes, score = runner(fen, DEPTH)
            if best is None or elapsed < best[0]:
                best = (elapsed, nodes, score)
        elapsed, nodes, score = best
        row[label] = (nodes / elapsed, score)
        print(
            f"{name:10s} {label:14s} {nodes:>12,} {elapsed:>8.3f}s "
            f"{nodes / elapsed / 1000:>10.1f}k {score:>9,}"
        )
    ratio = row["bitgen+numba"][0] / row["python-chess"][0]
    ratios.append(ratio)
    agree = "same" if row["bitgen+numba"][1] == row["python-chess"][1] else "DIFFERENT"
    print(f"{'':10s} {'-> speedup':14s} {ratio:>12.1f}x   root score {agree}\n")

print(f"mean search speedup {sum(ratios) / len(ratios):.1f}x")
