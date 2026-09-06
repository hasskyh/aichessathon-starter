"""Search depth and throughput: fixed time, head-to-head between any two agent
baselines via their own think() entry point.

Usage: perf/bench_search.py <baseline-dir-A> <baseline-dir-B> [seconds]

Fixed time, not fixed depth -- a full-depth benchmark on positions like kiwipete can
take many minutes per engine once pruning (or the lack of it) swings node counts by
orders of magnitude, which isn't a fair per-position time budget either. Fixed time
matches how the engine is actually used (a real clock budget per move) and reports
back whatever depth and node count that time bought -- reaching a deeper depth in the
same time, or the same depth on fewer nodes, is exactly what the pruning additions
(LMR/LMP/RFP/killers) are supposed to deliver.

Each run resets the engine's own transposition table, history, and killer-move globals
to their fresh-process values first -- otherwise a later position would hit cached TT
entries left over from an earlier one and look artificially fast. A fresh, empty
game-history array is passed to think() every call for the same reason (no
repetition-draw state leaking across positions).
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POSITIONS = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("midgame", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/2NP1N2/PPPQ1PPP/R4RK1 w - - 0 10"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]


def load(path: str):
    sys.path.insert(0, os.path.join(ROOT, path))
    sys.modules.pop("agent", None)
    import agent
    return agent


def reset_state(mod) -> None:
    mod.TT_KEY.fill(0)
    mod.TT_MOVE.fill(-1)
    mod.TT_SCORE.fill(0)
    mod.TT_DEPTH.fill(0)
    mod.TT_TYPE.fill(0)
    mod.HISTORY.fill(0)
    if hasattr(mod, "KILLERS"):  # not every baseline has killer moves
        mod.KILLERS.fill(-1)


def run(mod, fen: str, seconds: float) -> tuple[float, int, int, int]:
    bb, sq, st = mod.bitgen.from_fen(fen)
    reset_state(mod)
    game_history = np.zeros(101, dtype=np.uint64)
    mod.CTRL[0] = time.monotonic() + seconds
    mod.CTRL[1] = 0.0
    mod.CTRL[2] = 0.0
    mod.CTRL[3] = mod.CHECK_INTERVAL
    start = time.perf_counter()
    _move, reached_depth, score = mod.think(
        bb, sq, st, mod.BUF, mod.SCORES, mod.HIST, mod.CTRL, mod.STACK, mod.KING_SQ,
        mod.MAX_DEPTH, mod.HASH_STACK, mod.ASPIRATION_MARGIN, game_history, 0,
    )
    elapsed = time.perf_counter() - start
    return elapsed, int(mod.CTRL[1]), int(score), reached_depth


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <baseline-dir-A> <baseline-dir-B> [seconds]", file=sys.stderr)
        sys.exit(2)

    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

    label_a = os.path.basename(sys.argv[1].rstrip("/"))
    label_b = os.path.basename(sys.argv[2].rstrip("/"))
    mod_a = load(sys.argv[1])
    mod_b = load(sys.argv[2])

    print(f"--- fixed time {seconds:.0f}s per position ---\n")
    print(f"{'position':10s} {'engine':34s} {'depth':>6s} {'nodes':>12s} {'time':>9s} {'nps':>12s} {'score':>9s}")
    depth_deltas = []
    for name, fen in POSITIONS:
        row = {}
        for label, mod in ((label_a, mod_a), (label_b, mod_b)):
            elapsed, nodes, score, depth = run(mod, fen, seconds)
            row[label] = (depth, nodes, nodes / elapsed)
            print(
                f"{name:10s} {label:34s} {depth:>6d} {nodes:>12,} {elapsed:>8.3f}s "
                f"{nodes / elapsed / 1000:>10.1f}k {score:>9,}"
            )
        depth_delta = row[label_b][0] - row[label_a][0]
        depth_deltas.append(depth_delta)
        print(
            f"{'':10s} {'-> ' + label_b + ' vs ' + label_a:34s} "
            f"depth {depth_delta:+d}   nps {row[label_b][2] / row[label_a][2]:.2f}x\n"
        )

    print(f"mean depth gained ({label_b} vs {label_a}): {sum(depth_deltas) / len(depth_deltas):+.1f} ply")


if __name__ == "__main__":
    main()
