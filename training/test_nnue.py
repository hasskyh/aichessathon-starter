"""Does update agree with refresh, move after move, and what does it cost?"""

import random
import sys
import time
from collections import Counter
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path.home() / "aichessathon-starter"))
import features
import nnue

rng = np.random.default_rng(23)
W1 = rng.integers(-64, 64, size=(nnue.FEATURES, nnue.HIDDEN), dtype=np.int16)
B1 = rng.integers(-16, 16, size=nnue.HIDDEN, dtype=np.int16)

ok = True


def fresh(board: chess.Board) -> np.ndarray:
    acc = np.zeros((nnue.PERSPECTIVES, nnue.HIDDEN), dtype=np.int32)
    nnue.refresh(acc, W1, B1, features.active(board))
    return acc


FIXTURES = [
    ("quiet move", chess.STARTING_FEN, "g1f3"),
    ("capture", "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "e4d5"),
    ("promotion", "8/P7/8/8/8/8/8/K6k w - - 0 1", "a7a8q"),
    ("promotion + capture", "1n6/P7/8/8/8/8/8/K6k w - - 0 1", "a7b8q"),
    ("en passant", "8/8/8/3pP3/8/8/8/K6k w - d6 0 1", "e5d6"),
    ("castling O-O", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("castling O-O-O", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
    ("black O-O", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
    ("black en passant", "K6k/8/8/8/4pP2/8/8/8 b - f3 0 1", "e4f3"),
]

print("=== every move shape: update must match a from-scratch refresh ===")
for label, fen, uci in FIXTURES:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    acc = fresh(board)
    off, on = features.deltas(board, move)
    board.push(move)
    nnue.update(acc, W1, off, on)
    want = fresh(board)
    same = np.array_equal(acc, want)
    ok = ok and same
    drift = int(np.abs(acc - want).max()) if not same else 0
    print(f"  {'ok  ' if same else 'FAIL'} {label:22s}"
          f"{'' if same else f'  max drift {drift}'}")

print("\n=== drift over long random games ===")
random.seed(101)
shapes: Counter[str] = Counter()
moves_checked = 0
failures = 0
for game in range(150):
    board = chess.Board()
    acc = fresh(board)
    while not board.is_game_over(claim_draw=False) and board.ply() < 140:
        move = random.choice(list(board.legal_moves))
        if board.is_castling(move):
            shapes["castling"] += 1
        elif board.is_en_passant(move):
            shapes["en passant"] += 1
        elif move.promotion:
            shapes["promotion"] += 1
        elif board.piece_at(move.to_square):
            shapes["capture"] += 1
        else:
            shapes["quiet"] += 1
        off, on = features.deltas(board, move)
        board.push(move)
        nnue.update(acc, W1, off, on)
        moves_checked += 1
        if not np.array_equal(acc, fresh(board)):
            failures += 1
            if failures <= 3:
                print(f"  FAIL game {game} ply {board.ply()} {move.uci()}  {board.fen()}")
            acc = fresh(board)
print(f"  {moves_checked} moves, {failures} drifted")
for name, count in shapes.most_common():
    print(f"    {name:12s} {count}")
ok = ok and failures == 0

print("\n=== cost: update against refresh ===")
board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 5 4")
idx = features.active(board)
move = chess.Move.from_uci("f3e5")
assert move in board.legal_moves
off, on = features.deltas(board, move)
acc = np.zeros((nnue.PERSPECTIVES, nnue.HIDDEN), dtype=np.int32)

runs = 50_000
started = time.perf_counter()
for _ in range(runs):
    nnue.refresh(acc, W1, B1, idx)
refresh_us = (time.perf_counter() - started) / runs * 1e6

started = time.perf_counter()
for _ in range(runs):
    nnue.update(acc, W1, off, on)
update_us = (time.perf_counter() - started) / runs * 1e6

print(f"  refresh  {refresh_us:6.2f} us   (32 features x 2 perspectives)")
print(f"  update   {update_us:6.2f} us   ({off.shape[1]} off, {on.shape[1]} on)")
print(f"  update is {refresh_us / update_us:.1f}x cheaper")
print(f"  a node costs 57.4 us today, so update is {update_us / 57.4 * 100:.1f}% of one node")

print("\n=== and what deltas() itself costs, the part that cannot be jitted ===")
started = time.perf_counter()
for _ in range(runs):
    features.deltas(board, move)
deltas_us = (time.perf_counter() - started) / runs * 1e6
print(f"  features.deltas  {deltas_us:6.2f} us  (pure Python)")
print(f"  deltas + update  {deltas_us + update_us:6.2f} us  = "
      f"{(deltas_us + update_us) / 57.4 * 100:.0f}% of a 57.4 us node")

print("\nALL CHECKS PASSED" if ok else "\nSOMETHING FAILED")
sys.exit(0 if ok else 1)
