import random
import sys

import chess
import numpy as np

sys.path.insert(0, ".")
import bitgen
import features_halfkp as fk

random.seed(1)

MAX_PLY = 200
BUF = np.zeros((MAX_PLY, 256), dtype=np.int32)
HIST = np.zeros((MAX_PLY, 4), dtype=np.int64)


def rows_match(chess_row, bb_row):
    return sorted(chess_row.tolist()) == sorted(int(x) for x in bb_row if x >= 0)


def check_active(board, sq, wk, bk, label):
    white, black = fk.halfkp_active(board)
    active_bb = fk.halfkp_active_bb(sq, wk, bk)
    ok_w = rows_match(white, active_bb[0])
    ok_b = rows_match(black, active_bb[1])
    if not (ok_w and ok_b):
        print(f"MISMATCH (active) at {label}: fen={board.fen()}")
        print("  chess white:", sorted(white.tolist()))
        print("  bb    white:", sorted(int(x) for x in active_bb[0] if x >= 0))
        print("  chess black:", sorted(black.tolist()))
        print("  bb    black:", sorted(int(x) for x in active_bb[1] if x >= 0))
        return False
    return True


n_games = 200
n_mismatches = 0
n_positions = 0
n_delta_checks = 0

for game_num in range(n_games):
    board = chess.Board()
    bb, sq, st = bitgen.from_fen(bitgen.STARTING_FEN)
    wk, bk = fk.find_king_squares(sq)

    if not check_active(board, sq, wk, bk, f"game {game_num} start"):
        n_mismatches += 1
    n_positions += 1

    for ply in range(80):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = random.choice(legal)

        count = bitgen.gen_moves(bb, st, BUF[0], 0)
        bb_move = None
        for i in range(count):
            if bitgen.to_uci(int(BUF[0, i])) == move.uci():
                bb_move = np.int32(BUF[0, i])
                break
        if bb_move is None:
            print(
                f"MISMATCH (move not found): game {game_num} ply {ply} "
                f"move {move.uci()} fen={board.fen()}"
            )
            n_mismatches += 1
            break

        off, on = fk.halfkp_deltas_bb(sq, st, bb_move, wk, bk)
        off_c, on_c, wr, br = fk.halfkp_deltas(board, move)
        n_delta_checks += 1

        bb_is_king_move = sq[bb_move & 63] % 6 == bitgen.KING
        if bb_is_king_move != (wr or br):
            print(
                f"MISMATCH (refresh flag disagreement): game {game_num} ply {ply} "
                f"move {move.uci()} bb_king_move={bb_is_king_move} chess wr={wr} br={br}"
            )
            n_mismatches += 1

        ok_off_w = rows_match(off_c[0], off[0])
        ok_off_b = rows_match(off_c[1], off[1])
        ok_on_w = rows_match(on_c[0], on[0])
        ok_on_b = rows_match(on_c[1], on[1])
        if not (ok_off_w and ok_off_b and ok_on_w and ok_on_b):
            print(
                f"MISMATCH (deltas) game {game_num} ply {ply} "
                f"move {move.uci()} fen={board.fen()} king_move={bb_is_king_move}"
            )
            print("  off chess:", off_c.tolist(), "off bb:", off.tolist())
            print("  on  chess:", on_c.tolist(), "on  bb:", on.tolist())
            n_mismatches += 1

        board.push(move)
        bitgen.make_move(bb, sq, st, bb_move, HIST, 0)

        if bb_is_king_move:
            wk, bk = fk.find_king_squares(sq)

        if not check_active(board, sq, wk, bk, f"game {game_num} ply {ply}"):
            n_mismatches += 1
        n_positions += 1

print(
    f"\n{n_positions} positions checked, {n_delta_checks} delta checks, "
    f"{n_mismatches} mismatches"
)
