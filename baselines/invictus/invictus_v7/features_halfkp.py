"""HalfKP: king-relative features, a separate track from features.py's flat 768.

Every feature here is a (my king's square, piece owner, piece type, piece square)
tuple -- excluding kings as pieces, since a king's own position is what indexes the
feature, not something it can also be one of. 64 king squares * 64 piece squares *
5 piece types * 2 owners = 40,960 features per perspective, matching the standard
HalfKP size used by real NNUE nets.

The consequence that makes this a genuinely different architecture, not just a
bigger flat net: every one of a perspective's 40,960 features changes meaning the
moment that perspective's own king moves, because they are all relative to where
it stands. A flat, color-relative feature (features.py's) never needs this -- a
piece's feature index only depends on the piece and the perspective's colour, so
incremental updates always work. Here, a king move invalidates an entire
perspective's accumulator at once; the caller must call refresh() for that side
instead of update(), never both at once for the same move.

halfkp_deltas()/halfkp_deltas_bb() are safe to call for ANY move, including a king
move -- they never produce a feature entry for the king's own square, but DO
produce one for a captured piece or a castling rook, because the perspective whose
OWN king did not move still needs that applied as an ordinary incremental update.
The caller's protocol for a move whose mover is a king:
  - the mover's own colour's perspective: ignore these off/on rows entirely, and
    call halfkp_active() (or the bb equivalent) fresh for that side instead.
  - the other perspective: apply off/on via update() exactly as normal. It is
    usually empty (a quiet king move touches nothing else), non-empty exactly
    when the king move was a capture or a castle.
For any non-king move, apply off/on to both perspectives via update(), same as
features.py's flat set.

40,960 exceeds int16's positive range (32,767), unlike the flat set's 768 -- every
feature *index* here is int32, even though the *weight values* stored at those
indices are still the same small int16 numbers nnue.py already uses.

Two encoders, matching features.py's split: halfkp_active()/halfkp_deltas() take a
chess.Board, for pack.py's offline data preparation. halfkp_active_bb()/
halfkp_deltas_bb() do the identical encoding against bitgen's bb/sq/st arrays,
jitted so they can be called from inside agent.py's compiled search. They must
agree bit for bit, the same way features.py's two encoders do, and are checked
against each other the same way -- see test_features_halfkp.py, including the
king-move cases above, which is exactly where the first version of this file had
two real bugs: halfkp_deltas() returned nothing at all for a king move, discarding
a castling rook's or a captured piece's change; halfkp_deltas_bb() went the other
way and encoded the king's own move as if it were a real feature, silently
aliasing a completely different, valid feature (piece_type=5 for a king wraps
around to owner=1, piece_type=0's slot, since PIECE_TYPES is only 5), and never
handled castling's rook move at all. Caught by test_nnue_halfkp_accum2.py-style
incremental-vs-fresh-refresh testing before either reached a trained model.
"""

import chess
import numpy as np
from numba import njit

import bitgen

SQUARES = 64
PIECE_TYPES = 5  # pawn, knight, bishop, rook, queen -- kings are never a feature
OWNERS = 2
PER_KING_SQUARE = SQUARES * PIECE_TYPES * OWNERS  # 640
FEATURES = SQUARES * PER_KING_SQUARE  # 40,960
MIRROR = 56  # square ^ MIRROR flips the rank and leaves the file alone
MAX_PIECES = 30  # 32 squares minus the two kings, which are never features
MAX_CHANGES = 2  # a capture, or castling's rook: one square off, one square on


def halfkp_index(
    king_square: chess.Square, piece_type_idx: int, owner: int, piece_square: chess.Square,
    perspective: chess.Color,
) -> int:
    """The feature number for one piece on one square, seen by one king's owner.

    king_square is that perspective's OWN king -- always chess.WHITE's king for the
    white row, chess.BLACK's king for the black row -- never the opponent's.
    """
    ks = king_square if perspective == chess.WHITE else king_square ^ MIRROR
    ps = piece_square if perspective == chess.WHITE else piece_square ^ MIRROR
    return ks * PER_KING_SQUARE + (owner * PIECE_TYPES + piece_type_idx) * SQUARES + ps


def halfkp_active(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    """Every non-king feature that is on: int32[2, n], row 0 white's view, row 1 black's."""
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    white: list[int] = []
    black: list[int] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        pt = piece.piece_type - 1  # chess.PAWN=1 .. chess.QUEEN=5 -> 0..4
        owner_w = 0 if piece.color == chess.WHITE else 1
        owner_b = 0 if piece.color == chess.BLACK else 1
        white.append(halfkp_index(white_king, pt, owner_w, square, chess.WHITE))
        black.append(halfkp_index(black_king, pt, owner_b, square, chess.BLACK))
    return np.array(white, dtype=np.int32), np.array(black, dtype=np.int32)


def halfkp_deltas(
    board: chess.Board, move: chess.Move
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """The features switching off and on, each int32[2, k], plus a full-refresh flag
    per perspective (white_refresh, black_refresh) for when that side's own king moved.

    See the module docstring for the full king-move protocol. Call this before
    board.push(move), exactly like features.deltas().
    """
    mover = board.piece_at(move.from_square)
    if mover is None:
        raise ValueError(f"{move.uci()} has no piece on its from-square")

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    white_refresh = mover.piece_type == chess.KING and mover.color == chess.WHITE
    black_refresh = mover.piece_type == chess.KING and mover.color == chess.BLACK

    off: list[tuple[chess.Piece, chess.Square]] = []
    on: list[tuple[chess.Piece, chess.Square]] = []

    if mover.piece_type != chess.KING:
        off.append((mover, move.from_square))
        on.append((chess.Piece(move.promotion or mover.piece_type, mover.color), move.to_square))

    if board.is_castling(move):
        rook = chess.Piece(chess.ROOK, mover.color)
        rook_from, rook_to = _rook_squares(move)
        off.append((rook, rook_from))
        on.append((rook, rook_to))
    else:
        taken = _captured_square(board, move)
        if taken is not None:
            captured = board.piece_at(taken)
            if captured is not None:
                off.append((captured, taken))

    def encode(items: list[tuple[chess.Piece, chess.Square]]) -> np.ndarray:
        if not items:
            return np.zeros((2, 0), dtype=np.int32)
        white = []
        black = []
        for piece, square in items:
            pt = piece.piece_type - 1
            owner_w = 0 if piece.color == chess.WHITE else 1
            owner_b = 0 if piece.color == chess.BLACK else 1
            white.append(halfkp_index(white_king, pt, owner_w, square, chess.WHITE))
            black.append(halfkp_index(black_king, pt, owner_b, square, chess.BLACK))
        return np.array([white, black], dtype=np.int32)

    return encode(off), encode(on), white_refresh, black_refresh


def _captured_square(board: chess.Board, move: chess.Move) -> chess.Square | None:
    if board.is_castling(move):
        return None
    if board.is_en_passant(move):
        return chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
    return move.to_square if board.piece_at(move.to_square) is not None else None


def _rook_squares(move: chess.Move) -> tuple[chess.Square, chess.Square]:
    """The rook's journey during a castle, derived from the king's destination file."""
    rank = chess.square_rank(move.from_square)
    if chess.square_file(move.to_square) > chess.square_file(move.from_square):
        return chess.square(7, rank), chess.square(5, rank)  # h-file rook to the f-file
    return chess.square(0, rank), chess.square(3, rank)  # a-file rook to the d-file


# --- bitgen-native encoding, jitted for use inside the search --------------------


@njit(inline="always", cache=False)
def halfkp_feature_index_bb(king_square: int, code: int, square: int, perspective: int) -> int:
    """Same formula as halfkp_index(), taking a bitgen piece code (0-11) instead of
    a chess.Piece. code // 6 is the colour, code % 6 the piece type. Never call this
    with a king code (code % 6 == bitgen.KING) -- piece_type=5 silently aliases into
    a real, valid feature slot rather than erroring (see the module docstring)."""
    colour = code // 6
    piece_type = code % 6
    owner = 0 if colour == perspective else 1
    ks = king_square if perspective == bitgen.WHITE else king_square ^ MIRROR
    ps = square if perspective == bitgen.WHITE else square ^ MIRROR
    return ks * PER_KING_SQUARE + (owner * PIECE_TYPES + piece_type) * SQUARES + ps


@njit(cache=False)
def find_king_squares(sq: np.ndarray) -> tuple[int, int]:
    """Scan for both kings. Only ever called at a refresh, never in the hot path --
    the search itself tracks king squares incrementally (see agent.py) rather than
    re-scanning here on every move."""
    white_king = -1
    black_king = -1
    for square in range(64):
        code = sq[square]
        if code == bitgen.KING:
            white_king = square
        elif code == bitgen.KING + 6:
            black_king = square
    return white_king, black_king


@njit(cache=False)
def halfkp_active_bb(sq: np.ndarray, white_king: int, black_king: int) -> np.ndarray:
    """Every non-king feature that is on, fixed-width and -1 padded: int32[2, 30]."""
    out = np.full((2, MAX_PIECES), -1, dtype=np.int32)
    n = 0
    for square in range(64):
        code = sq[square]
        if code < 0 or code % 6 == bitgen.KING:
            continue
        out[0, n] = halfkp_feature_index_bb(white_king, code, square, bitgen.WHITE)
        out[1, n] = halfkp_feature_index_bb(black_king, code, square, bitgen.BLACK)
        n += 1
    return out


@njit(cache=False)
def halfkp_deltas_bb(
    sq: np.ndarray, st: np.ndarray, move: int, white_king: int, black_king: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The features switching off and on, each int32[2, MAX_CHANGES], -1 padded.

    Safe to call for ANY move, including one whose mover is a king -- see the
    module docstring for the full protocol. Call this before bitgen.make_move(move).
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
    n_off = 0
    n_on = 0

    mover_code = sq[frm]
    if mover_code % 6 != bitgen.KING:
        off_code[0] = mover_code
        off_sq[0] = frm
        n_off = 1
        arriving_code = us * 6 + promo if promo > 0 else mover_code
        on_code[0] = arriving_code
        on_sq[0] = to
        n_on = 1

    if flag == bitgen.FLAG_EP:
        captured_sq = to - bitgen.PUSH_DIR[us]
        off_code[n_off] = sq[captured_sq]
        off_sq[n_off] = captured_sq
        n_off += 1
    elif flag == bitgen.FLAG_CASTLE:
        rook_code = us * 6 + bitgen.ROOK
        if to > frm:  # kingside: h-file rook to the f-file
            rook_from, rook_to = frm + 3, frm + 1
        else:  # queenside: a-file rook to the d-file
            rook_from, rook_to = frm - 4, frm - 1
        off_code[n_off] = rook_code
        off_sq[n_off] = rook_from
        n_off += 1
        on_code[n_on] = rook_code
        on_sq[n_on] = rook_to
        n_on += 1
    else:
        captured_code = sq[to]
        if captured_code >= 0:
            off_code[n_off] = captured_code
            off_sq[n_off] = to
            n_off += 1

    off_idx = np.full((2, MAX_CHANGES), -1, dtype=np.int32)
    on_idx = np.full((2, MAX_CHANGES), -1, dtype=np.int32)
    for k in range(n_off):
        off_idx[0, k] = halfkp_feature_index_bb(white_king, off_code[k], off_sq[k], bitgen.WHITE)
        off_idx[1, k] = halfkp_feature_index_bb(black_king, off_code[k], off_sq[k], bitgen.BLACK)
    for k in range(n_on):
        on_idx[0, k] = halfkp_feature_index_bb(white_king, on_code[k], on_sq[k], bitgen.WHITE)
        on_idx[1, k] = halfkp_feature_index_bb(black_king, on_code[k], on_sq[k], bitgen.BLACK)

    return off_idx, on_idx
