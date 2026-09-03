"""bitgen against python-chess, on the same tree walk.

Both sides do the same work per node: generate, make, recurse, unmake. `perft_nodes`
deliberately skips the usual depth-1 bulk-count shortcut so the node counts are the
same ones a search would visit, and nodes/sec compares honestly.
"""

import time

import chess

import bitgen

POSITIONS = [
    ("startpos", bitgen.STARTING_FEN, 4),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5),
    ("midgame", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/2NP1N2/PPPQ1PPP/R4RK1 w - - 0 10", 4),
]

# Measured once by perf/bench_baseline.py so a run here stays quick: python-chess
# nodes/sec for the identical walk, using board.legal_moves with push and pop.
BASELINE_NPS = {
    ("startpos", 4): 90_600.0,
    ("kiwipete", 4): 106_700.0,
    ("endgame", 5): 107_900.0,
    ("midgame", 4): 129_300.0,
}


def ref_perft(board, depth):
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += ref_perft(board, depth - 1)
        board.pop()
    return total


buf, hist = bitgen.new_buffers()

print("--- bitgen ---")
results = []
for name, fen, depth in POSITIONS:
    bb, sq, st = bitgen.from_fen(fen)
    best = 0.0
    for _ in range(3):
        bb, sq, st = bitgen.from_fen(fen)
        start = time.perf_counter()
        nodes = bitgen.perft_nodes(bb, sq, st, depth, buf, hist, 0)
        elapsed = time.perf_counter() - start
        best = max(best, nodes / elapsed)
    results.append((name, depth, nodes, best))
    print(f"{name:10s} depth {depth}  {nodes:>10,} nodes  {best / 1e6:7.2f}M nps")

print("\n--- speedup over python-chess on the same walk ---")
total_ratio = 0.0
for name, depth, nodes, nps in results:
    base = BASELINE_NPS.get((name, depth))
    if base is None:
        board = chess.Board(dict(((n, f) for n, f, _ in POSITIONS))[name])
        start = time.perf_counter()
        ref_perft(board, depth)
        base = nodes / (time.perf_counter() - start)
    ratio = nps / base
    total_ratio += ratio
    print(
        f"{name:10s} {base / 1000:7.1f}k -> {nps / 1e6:6.2f}M nps    {ratio:6.1f}x"
    )
print(f"\nmean speedup {total_ratio / len(results):.1f}x")

