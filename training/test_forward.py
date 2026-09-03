"""Is forward correct, and what does a whole leaf evaluation cost?"""

import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path.home() / "aichessathon-starter"))
import features
import nnue

rng = np.random.default_rng(5)
W1 = rng.integers(-64, 64, size=(nnue.FEATURES, nnue.HIDDEN), dtype=np.int16)
B1 = rng.integers(-16, 16, size=nnue.HIDDEN, dtype=np.int16)
W2 = rng.integers(-32, 32, size=(2 * nnue.HIDDEN, nnue.OUTPUTS), dtype=np.int8)
B2 = rng.integers(-500, 500, size=nnue.OUTPUTS, dtype=np.int32)
W3 = rng.integers(-32, 32, size=nnue.OUTPUTS, dtype=np.int8)
B3 = np.int32(37)

ok = True


def reference(acc: np.ndarray, stm: int) -> int:
    """Vectorised numpy, written deliberately differently from the jitted loops."""
    joined = np.concatenate([acc[stm], acc[1 - stm]]).astype(np.int64)
    activations = np.clip(joined, 0, nnue.ACT_MAX)
    hidden = B2.astype(np.int64) + activations @ W2.astype(np.int64)
    hidden = np.clip(hidden >> nnue.HIDDEN_SHIFT, 0, nnue.ACT_MAX)
    return int(np.int64(B3) + hidden @ W3.astype(np.int64))


def accumulator(board: chess.Board) -> np.ndarray:
    acc = np.zeros((nnue.PERSPECTIVES, nnue.HIDDEN), dtype=np.int32)
    nnue.refresh(acc, W1, B1, features.active(board))
    return acc


POSITIONS = [
    ("start", chess.STARTING_FEN),
    ("midgame white", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 5 4"),
    ("midgame black", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1"),
    ("endgame", "8/5k2/8/3P4/8/2K5/8/8 w - - 0 1"),
    ("bare kings", "8/8/8/4k3/8/8/8/K7 w - - 0 1"),
]

print("=== correctness against a numpy reference ===")
for label, fen in POSITIONS:
    board = chess.Board(fen)
    acc = accumulator(board)
    stm = 0 if board.turn == chess.WHITE else 1
    got = int(nnue.forward(acc, stm, W2, B2, W3, B3))
    want = reference(acc, stm)
    same = got == want
    ok = ok and same
    print(f"  {'ok  ' if same else 'FAIL'} {label:15s} stm={stm}  "
          f"forward={got:>9}  reference={want:>9}")

print("\n=== the mover's half comes first, so stm changes the score ===")
board = chess.Board(POSITIONS[1][1])
acc = accumulator(board)
as_white = int(nnue.forward(acc, 0, W2, B2, W3, B3))
as_black = int(nnue.forward(acc, 1, W2, B2, W3, B3))
differs = as_white != as_black
ok = ok and differs
print(f"  {'ok  ' if differs else 'FAIL'} stm=0 -> {as_white}, stm=1 -> {as_black}")

print("\n=== no int32 overflow: worst case is far inside the range ===")
saturated = np.full((2, nnue.HIDDEN), 10_000, dtype=np.int32)
got = int(nnue.forward(saturated, 0, W2, B2, W3, B3))
want = reference(saturated, 0)
same = got == want
ok = ok and same
print(f"  {'ok  ' if same else 'FAIL'} all activations clamped high: {got} vs {want}")
print(f"  theoretical max before the shift: 512 * {nnue.ACT_MAX} * 127 = "
      f"{512 * nnue.ACT_MAX * 127:,} (int32 holds 2,147,483,647)")

print("\n=== cost of a full leaf evaluation ===")
board = chess.Board(POSITIONS[1][1])
acc = accumulator(board)
move = chess.Move.from_uci("f3e5")
assert move in board.legal_moves
off, on = features.deltas(board, move)
idx = features.active(board)

runs = 30_000


def timed(fn) -> float:
    started = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - started) / runs * 1e6


t_deltas = timed(lambda: features.deltas(board, move))
t_update = timed(lambda: nnue.update(acc, W1, off, on))
t_forward = timed(lambda: nnue.forward(acc, 0, W2, B2, W3, B3))
t_refresh = timed(lambda: nnue.refresh(acc, W1, B1, idx))

print(f"  features.deltas  {t_deltas:7.2f} us   pure Python")
print(f"  nnue.update      {t_update:7.2f} us   jitted")
print(f"  nnue.forward     {t_forward:7.2f} us   jitted")
print(f"  nnue.refresh     {t_refresh:7.2f} us   jitted, root only")
interior = t_deltas + t_update
leaf = interior + t_forward
print(f"\n  interior node (deltas + update)      {interior:7.2f} us")
print(f"  leaf node    (+ forward)             {leaf:7.2f} us")
print(f"  a node costs 57.4 us today, so a leaf becomes {57.4 + leaf:.1f} us")
print(f"  implied node rate: {1e6 / (57.4 + leaf):,.0f}/sec  (from 17,400)")
print("  gate for step 4 was 8,000/sec")

print("\nALL CHECKS PASSED" if ok else "\nSOMETHING FAILED")
sys.exit(0 if ok else 1)
