"""Structural (non-judgmental) check: what fraction of the independent binpack
data is "tactically loud" -- side to move is in check, or has at least one legal
capture available -- versus fully quiet? This is move-generation bookkeeping via
python-chess, not a chess-quality judgment call, so it doesn't fall under "always
ask Stockfish" (that rule is about evaluating which side is better / what's best,
not about counting legal move types).

If the dataset is almost entirely quiet positions, the net would have had very
little training signal on positions where captures are actually on the table --
exactly the class of position (queen sac, hanging piece) where it's failing.
"""
import json
import sys

import chess

path = sys.argv[1] if len(sys.argv) > 1 else "data/binpack_labeled.jsonl"
sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else 50000

with open(path, encoding="utf-8") as f:
    lines = f.readlines()

import random
random.seed(0)
random.shuffle(lines)
lines = lines[:sample_n]

in_check = 0
has_capture = 0
loud = 0
quiet = 0
total = 0

for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
        board = chess.Board(data["fen"])
    except (json.JSONDecodeError, KeyError, ValueError):
        continue
    total += 1
    check = board.is_check()
    capture = any(board.is_capture(m) for m in board.legal_moves)
    if check:
        in_check += 1
    if capture:
        has_capture += 1
    if check or capture:
        loud += 1
    else:
        quiet += 1

print(f"total checked: {total:,}")
print(f"in check: {in_check:,} ({100*in_check/total:.1f}%)")
print(f"has a legal capture available: {has_capture:,} ({100*has_capture/total:.1f}%)")
print(f"tactically loud (check or capture available): {loud:,} ({100*loud/total:.1f}%)")
print(f"fully quiet (no check, no capture available): {quiet:,} ({100*quiet/total:.1f}%)")
