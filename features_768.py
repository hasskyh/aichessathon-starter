"""The 768-feature encoding, shared by the engine and the trainer.

Every function here produces both perspectives. A single side-to-move-relative
encoding cannot be updated incrementally, because the meaning of every index
changes with the turn; two fixed-perspective sets each stay stable across a move,
and the search reads whichever row belongs to the side to move.

A feature is one yes/no question: "is there a friendly knight on c3?". There are
768 of them, 2 owners * 6 piece types * 64 squares, and at most 32 are on at once.

Two encoders live here. active()/deltas() take a chess.Board, for pack.py's
offline data preparation. active_bb()/deltas_bb() do the identical encoding
against bitgen's bb/sq/st arrays and a packed move int, jitted so they can be
called from inside agent.py's fully compiled search. They must agree bit for bit
-- the trained weights' meaning depends on it -- and test_features_bb.py checks
that directly against thousands of real positions, not by assumption.
"""

import chess
import numpy as np
from numba import njit

import bitgen

SQUARES = 64
PIECE_TYPES = 6
PERSPECTIVE = PIECE_TYPES * SQUARES  # 384, the size of one owner's half
MIRROR = 56  # square ^ MIRROR flips the rank and leaves the file alone
MAX_CHANGES = 2  # castling is the largest case: two pieces off, two on

Change = tuple[chess.Piece, chess.Square]


def index(piece: chess.Piece, square: chess.Square, perspective: chess.Color) -> int:
    """The feature number for one piece on one square, seen from one side."""
    owner = 0 if piece.color == perspective else 1
    seen = square if perspective == chess.WHITE else square ^ MIRROR
    return owner * PERSPECTIVE + (piece.piece_type - 1) * SQUARES + seen


def active(board: chess.Board) -> np.ndarray:
    """Every feature that is on: int16[2, n], row 0 white's view, row 1 black's."""
    white: list[int] = []
    black: list[int] = []
    for square, piece in board.piece_map().items():
        white.append(index(piece, square, chess.WHITE))
        black.append(index(piece, square, chess.BLACK))
    return np.array([white, black], dtype=np.int16)


def deltas(board: chess.Board, move: chess.Move) -> tuple[np.ndarray, np.ndarray]:
    """The features switching off and on, each int16[2, k].

    Call this before board.push(move). It reads the piece being moved and the piece
    being captured, and neither is still there afterwards.
    """
    off, on = changes(board, move)
    return _encode(off), _encode(on)


def changes(board: chess.Board, move: chess.Move) -> tuple[list[Change], list[Change]]:
    """The physical (piece, square) pairs leaving and arriving, before any encoding."""
    mover = board.piece_at(move.from_square)
    if mover is None:
        raise ValueError(f"{move.uci()} has no piece on its from-square")

    off: list[Change] = [(mover, move.from_square)]
    on: list[Change] = [(chess.Piece(move.promotion or mover.piece_type, mover.color),
                         move.to_square)]

    taken = _captured_square(board, move)
    if taken is not None:
        captured = board.piece_at(taken)
        if captured is not None:
            off.append((captured, taken))

    if board.is_castling(move):
        rook = chess.Piece(chess.ROOK, mover.color)
        rook_from, rook_to = _rook_squares(move)
        off.append((rook, rook_from))
        on.append((rook, rook_to))

    return off, on


def _captured_square(board: chess.Board, move: chess.Move) -> chess.Square | None:
    """Where the captured piece stands, which is not always the move's destination."""
    if board.is_castling(move):
        return None
    if board.is_en_passant(move):
        # the taken pawn is beside the mover, not on the square the mover lands on
        return chess.square(chess.square_file(move.to_square),
                            chess.square_rank(move.from_square))
    return move.to_square if board.piece_at(move.to_square) is not None else None


def _rook_squares(move: chess.Move) -> tuple[chess.Square, chess.Square]:
    """The rook's journey during a castle, derived from the king's destination file."""
    rank = chess.square_rank(move.from_square)
    if chess.square_file(move.to_square) > chess.square_file(move.from_square):
        return chess.square(7, rank), chess.square(5, rank)  # h-file rook to the f-file
    return chess.square(0, rank), chess.square(3, rank)  # a-file rook to the d-file


def _encode(items: list[Change]) -> np.ndarray:
    white = [index(piece, square, chess.WHITE) for piece, square in items]
    black = [index(piece, square, chess.BLACK) for piece, square in items]
    return np.array([white, black], dtype=np.int16)


# --- bitgen-native encoding, jitted for use inside the search --------------------


@njit(inline="always", cache=False)
def feature_index_bb(code: int, square: int, perspective: int) -> int:
    """Same formula as index(), taking a bitgen piece code (0-11) instead of a
    chess.Piece. code // 6 is the colour, code % 6 the piece type -- and bitgen's
    own type order (PAWN..KING = 0..5) already matches chess.PAWN..KING minus 1."""
    colour = code // 6
    piece_type = code % 6
    owner = 0 if colour == perspective else 1
    seen = square if perspective == bitgen.WHITE else square ^ MIRROR
    return owner * PERSPECTIVE + piece_type * SQUARES + seen


@njit(cache=False)
def active_bb(sq: np.ndarray) -> np.ndarray:
    """Every feature that is on, fixed-width and -1 padded: int16[2, 32].

    Unlike active(), this returns ready-to-use rows -- the same shape nnue.refresh
    already expects -- since a jitted caller has no convenient way to trim a
    variable-length result the way pack.py does for the training data.
    """
    out = np.full((2, 32), -1, dtype=np.int16)
    n = 0
    for square in range(64):
        code = sq[square]
        if code < 0:
            continue
        out[0, n] = feature_index_bb(code, square, bitgen.WHITE)
        out[1, n] = feature_index_bb(code, square, bitgen.BLACK)
        n += 1
    return out


@njit(cache=False)
def deltas_bb(sq: np.ndarray, st: np.ndarray, move: int) -> tuple[np.ndarray, np.ndarray]:
    """The features switching off and on, each int16[2, MAX_CHANGES], -1 padded.

    Call this before bitgen.make_move(move). It reads the piece being moved and the
    piece being captured off of sq, and neither is still there afterwards.
    """
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 3
    us = st[0]

    off_code = np.full(MAX_CHANGES, -1, dtype=np.int64)
    off_sq = np.full(MAX_CHANGES, -1, dtype=np.int64)
    on_code = np.full(MAX_CHANGES, -1, dtype=np.int64)
    on_sq = np.full(MAX_CHANGES, -1, dtype=np.int64)

    mover_code = sq[frm]
    off_code[0] = mover_code
    off_sq[0] = frm
    n_off = 1

    arriving_code = us * 6 + promo if promo > 0 else mover_code
    on_code[0] = arriving_code
    on_sq[0] = to
    n_on = 1

    if flag == bitgen.FLAG_EP:
        # the taken pawn is beside the mover, not on the square the mover lands on
        captured_sq = to - bitgen.PUSH_DIR[us]
        off_code[1] = sq[captured_sq]
        off_sq[1] = captured_sq
        n_off = 2
    elif flag != bitgen.FLAG_CASTLE:
        captured_code = sq[to]
        if captured_code >= 0:
            off_code[1] = captured_code
            off_sq[1] = to
            n_off = 2

    if flag == bitgen.FLAG_CASTLE:
        rook_code = us * 6 + bitgen.ROOK
        if to > frm:  # kingside: h-file rook to the f-file
            rook_from, rook_to = frm + 3, frm + 1
        else:  # queenside: a-file rook to the d-file
            rook_from, rook_to = frm - 4, frm - 1
        off_code[1] = rook_code
        off_sq[1] = rook_from
        n_off = 2
        on_code[1] = rook_code
        on_sq[1] = rook_to
        n_on = 2

    off_idx = np.full((2, MAX_CHANGES), -1, dtype=np.int16)
    on_idx = np.full((2, MAX_CHANGES), -1, dtype=np.int16)
    for k in range(n_off):
        off_idx[0, k] = feature_index_bb(off_code[k], off_sq[k], bitgen.WHITE)
        off_idx[1, k] = feature_index_bb(off_code[k], off_sq[k], bitgen.BLACK)
    for k in range(n_on):
        on_idx[0, k] = feature_index_bb(on_code[k], on_sq[k], bitgen.WHITE)
        on_idx[1, k] = feature_index_bb(on_code[k], on_sq[k], bitgen.BLACK)

    return off_idx, on_idx
