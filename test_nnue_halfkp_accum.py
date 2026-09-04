import random
import sys

import numpy as np

sys.path.insert(0, ".")
import bitgen
import features_halfkp as fk
import nnue_halfkp as nn

random.seed(2)

MAX_PLY = 200
HIDDEN = nn.HIDDEN
BUF = np.zeros((MAX_PLY, 256), dtype=np.int32)
HIST = np.zeros((MAX_PLY, 4), dtype=np.int64)

W1 = np.random.randint(-50, 50, size=(nn.FEATURES, HIDDEN)).astype(np.int16)
B1 = np.zeros(HIDDEN, dtype=np.int16)

n_games = 30
n_moves_checked = 0
n_mismatches = 0

for game_num in range(n_games):
    bb, sq, st = bitgen.from_fen(bitgen.STARTING_FEN)
    wk, bk = fk.find_king_squares(sq)

    acc = np.zeros((2, HIDDEN), dtype=np.int32)
    nn.refresh(acc, W1, B1, fk.halfkp_active_bb(sq, wk, bk))

    for ply in range(120):
        count = bitgen.gen_moves(bb, st, BUF[ply], 0)
        if count == 0:
            break
        move = np.int32(BUF[ply, random.randrange(count)])

        is_king_move = sq[move & 63] % 6 == bitgen.KING
        moving_side = int(st[0])  # 0=white, 1=black; captured before make_move flips it

        # halfkp_deltas_bb is safe to call for ANY move, king or not -- it never
        # emits a feature entry for a king itself, only for captures/castling-rook
        # moves, which is exactly what the *other* perspective still needs applied
        # even when this move's mover is a king.
        off, on = fk.halfkp_deltas_bb(sq, st, move, wk, bk)

        bitgen.make_move(bb, sq, st, move, HIST, ply)

        if is_king_move:
            wk, bk = fk.find_king_squares(sq)
            other_side = 1 - moving_side
            # the moved king's own perspective: nothing short of a full recompute
            # means anything, since every one of its features is keyed on its
            # king square.
            fresh_row = fk.halfkp_active_bb(sq, wk, bk)
            nn.refresh(
                acc[moving_side : moving_side + 1], W1, B1,
                fresh_row[moving_side : moving_side + 1],
            )
            # the other perspective's own king did not move, so it is still a
            # normal incremental delta -- typically empty (a quiet king move),
            # non-empty exactly when the king move was a capture.
            nn.update(
                acc[other_side : other_side + 1], W1,
                off[other_side : other_side + 1], on[other_side : other_side + 1],
            )
        else:
            nn.update(acc, W1, off, on)

        fresh = np.zeros((2, HIDDEN), dtype=np.int32)
        nn.refresh(fresh, W1, B1, fk.halfkp_active_bb(sq, wk, bk))

        n_moves_checked += 1
        if not np.array_equal(fresh, acc):
            diff = np.abs(fresh.astype(np.int64) - acc.astype(np.int64))
            print(f"MISMATCH game {game_num} ply {ply} (king_move={is_king_move}): "
                  f"max|diff|={diff.max()} mean|diff|={diff.mean():.3f}")
            n_mismatches += 1
            if n_mismatches > 5:
                break

print(f"\n{n_moves_checked} moves checked across {n_games} games, {n_mismatches} mismatches")
