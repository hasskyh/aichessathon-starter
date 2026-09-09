"""Actually check features_768.py's own claim: active()/deltas() (training, via
pack.py) and active_bb()/deltas_bb() (runtime, via agent.py's search) must agree
bit for bit. The module docstring says test_features_bb.py checks this "against
thousands of real positions, not by assumption" -- but that file doesn't exist
anywhere in the repo, so the claim has never actually been verified for this
feature set. Two independently-trained nets (Lichess data, and now a genuinely
independent binpack-derived dataset) both reproduce the exact same knight-hanging
blunder, which rules data out -- if training and runtime don't actually agree on
what a feature index MEANS, that would explain identical behaviour regardless of
what data trained the weights.

Checks, per random real game:
  1. active(board) sorted == active_bb(sq) sorted (with -1 padding stripped), for
     every position reached by playing real moves from the start position.
  2. deltas(board, move) == deltas_bb(sq, st, move_int) for every move played,
     called BEFORE the move is applied on both sides (matching both functions'
     documented calling convention).
"""
import random
import sys

sys.path.insert(0, "/home/harry/aichessathon-starter")

import chess
import numpy as np

import bitgen
import features_768 as features

N_GAMES = 200
MAX_PLY = 60


def sq_from_board(board: chess.Board) -> np.ndarray:
    """bitgen-style sq[64] array: -1 empty, else colour*6+piece_type (0-11),
    piece_type 0=pawn..5=king, matching bitgen's own documented code order."""
    sq = np.full(64, -1, dtype=np.int64)
    for square, piece in board.piece_map().items():
        colour = 0 if piece.color == chess.WHITE else 1
        sq[square] = colour * 6 + (piece.piece_type - 1)
    return sq


def move_to_int(board: chess.Board, move: chess.Move) -> int:
    frm, to = move.from_square, move.to_square
    promo = 0
    if move.promotion:
        promo = move.promotion - 1  # bitgen: knight=1..queen=4? check PIECE order below
    flag = 0
    if board.is_en_passant(move):
        flag = bitgen.FLAG_EP
    elif board.is_castling(move):
        flag = bitgen.FLAG_CASTLE
    return frm | (to << 6) | (promo << 12) | (flag << 15)


def sorted_pairs(idx_2xn: np.ndarray) -> list[tuple[int, int]]:
    white = [v for v in idx_2xn[0] if v >= 0]
    black = [v for v in idx_2xn[1] if v >= 0]
    return sorted(zip(white, black))


def sorted_pairs_padded(idx_2xk) -> list[tuple[int, int]]:
    white = [int(v) for v in idx_2xk[0] if v >= 0]
    black = [int(v) for v in idx_2xk[1] if v >= 0]
    return sorted(zip(white, black))


def main() -> None:
    random.seed(0)
    active_mismatches = 0
    active_checks = 0
    delta_mismatches = 0
    delta_checks = 0
    promo_map = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

    for game_i in range(N_GAMES):
        board = chess.Board()
        for ply in range(MAX_PLY):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = random.choice(legal)

            sq = sq_from_board(board)
            st = np.array([0 if board.turn == chess.WHITE else 1], dtype=np.int64)

            # --- check 1: active() vs active_bb() on the pre-move position ---
            a_train = sorted_pairs(features.active(board))
            a_bb = sorted_pairs_padded(features.active_bb(sq))
            active_checks += 1
            if a_train != a_bb:
                active_mismatches += 1
                print(f"ACTIVE MISMATCH game {game_i} ply {ply} fen={board.fen()}")
                print(f"  train: {a_train}")
                print(f"  bb:    {a_bb}")

            # --- check 2: deltas() vs deltas_bb(), called before the move ---
            promo_code = promo_map.get(move.promotion, 0) if move.promotion else 0
            frm, to = move.from_square, move.to_square
            flag = 0
            if board.is_en_passant(move):
                flag = bitgen.FLAG_EP
            elif board.is_castling(move):
                flag = bitgen.FLAG_CASTLE
            move_int = frm | (to << 6) | (promo_code << 12) | (flag << 15)

            off_train, on_train = features.deltas(board, move)
            off_bb, on_bb = features.deltas_bb(sq, st, move_int)
            delta_checks += 1
            if sorted_pairs(off_train) != sorted_pairs_padded(off_bb) or \
               sorted_pairs(on_train) != sorted_pairs_padded(on_bb):
                delta_mismatches += 1
                print(f"DELTA MISMATCH game {game_i} ply {ply} fen={board.fen()} move={move.uci()}")
                print(f"  off train: {sorted_pairs(off_train)}  off bb: {sorted_pairs_padded(off_bb)}")
                print(f"  on  train: {sorted_pairs(on_train)}  on  bb: {sorted_pairs_padded(on_bb)}")

            board.push(move)

    print(f"\nactive(): {active_checks} checked, {active_mismatches} mismatches")
    print(f"deltas(): {delta_checks} checked, {delta_mismatches} mismatches")


if __name__ == "__main__":
    main()
