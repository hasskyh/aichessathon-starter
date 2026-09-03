"""Time move generation over a fixed corpus of positions.

Two things this fixes over timing perft. First, work is fixed: the same N positions
are generated from every time, so an ablated build that produces wrong moves still does
comparable work and the times stay meaningful. Second, generating from a *changing*
position each iteration stops LLVM hoisting the call out of the loop, which is what made
an earlier single-position micro-benchmark report an impossible 235M generations/sec.

Usage:  python perf/bench_gen.py [label]
The bitgen it measures is whichever one is first on sys.path, so an ablated copy can be
dropped in a scratch directory and measured with the same script.
"""

import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bitgen  # noqa: E402
from bitgen import gen_moves, make_move  # noqa: E402
from numba import njit  # noqa: E402

CORPUS = "/tmp/bitgen_corpus.npz"
POSITIONS = 4000
REPEATS = 7


def build_corpus():
    """Random games, sampling every position, so the mix is realistic rather than
    a handful of hand-picked openings."""
    rng = random.Random(4242)
    buf, hist = bitgen.new_buffers()
    bbs = np.zeros((POSITIONS, 15), dtype=np.uint64)
    sqs = np.zeros((POSITIONS, 64), dtype=np.int8)
    sts = np.zeros((POSITIONS, 5), dtype=np.int64)
    n = 0
    while n < POSITIONS:
        bb, sq, st = bitgen.from_fen(bitgen.STARTING_FEN)
        for _ in range(160):
            count = gen_moves(bb, st, buf[0], 0)
            if count == 0:
                break
            bbs[n] = bb
            sqs[n] = sq
            sts[n] = st
            n += 1
            if n == POSITIONS:
                break
            move = buf[0, rng.randrange(count)]
            make_move(bb, sq, st, move, hist, 0)
    np.savez(CORPUS, bbs=bbs, sqs=sqs, sts=sts)
    return bbs, sqs, sts


@njit(cache=False)
def _sweep(bbs, sts, out, reps):
    total = 0
    for _ in range(reps):
        for i in range(bbs.shape[0]):
            total += gen_moves(bbs[i], sts[i], out, 0)
    return total


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "bitgen"
    if os.path.exists(CORPUS):
        data = np.load(CORPUS)
        bbs, sqs, sts = data["bbs"], data["sqs"], data["sts"]
    else:
        bbs, sqs, sts = build_corpus()

    buf, _hist = bitgen.new_buffers()
    out = buf[0]
    _sweep(bbs[:8], sts[:8], out, 1)  # compile

    best = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        moves = _sweep(bbs, sts, out, 400)
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
            best_moves = moves
    calls = bbs.shape[0] * 400
    print(
        f"{label:22s} {calls:>8,} generations  {best:7.4f}s  "
        f"{calls / best / 1e6:6.2f}M gen/s  {best / calls * 1e9:6.0f} ns/gen  "
        f"({best_moves / calls:.1f} moves/pos)"
    )


if __name__ == "__main__":
    main()
