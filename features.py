"""The 768-feature encoding, shared by the engine and the trainer.

Every function here produces both perspectives. A single side-to-move-relative
encoding cannot be updated incrementally, because the meaning of every index
changes with the turn; two fixed-perspective sets each stay stable across a move,
and the search reads whichever row belongs to the side to move.

A feature is one yes/no question: "is there a friendly knight on c3?". There are
768 of them, 2 owners * 6 piece types * 64 squares, and at most 32 are on at once.
"""

import chess
import numpy as np

SQUARES = 64
PIECE_TYPES = 6
PERSPECTIVE = PIECE_TYPES * SQUARES  # 384, the size of one owner's half
MIRROR = 56  # square ^ MIRROR flips the rank and leaves the file alone

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
