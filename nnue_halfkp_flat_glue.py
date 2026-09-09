"""Bridges the search's stack/board representation to the FLAT HalfKP NNUE
architecture (nnue_halfkp.forward_flat) and its feature encoding (features_halfkp.py).

Structurally identical to nnue_halfkp_glue.py -- same apply_move_nnue, root_refresh,
move_deltas, since those only touch the accumulator stage (w1/b1), which is
unchanged by whether the output stage has a hidden layer or not. Only nnue_evaluate
differs: it calls nnue.forward_flat (accumulator -> single linear layer -> score)
instead of nnue.forward (accumulator -> hidden layer -> score).

nnue_evaluate still accepts w2/b2 for call-site compatibility with agent.py (see
nnue_halfkp_glue.py's own docstring on this pattern, already used there for
king_sq) -- they are simply unused here.
"""

import numpy as np
from numba import njit

import bitgen
import features_halfkp
import nnue_halfkp as nnue

HIDDEN = nnue.HIDDEN


@njit(inline="always", cache=False)
def nnue_evaluate(
    stack: np.ndarray,
    ply: int,
    st: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: int,
) -> int:
    """NNUE score for the position at this ply, from the side to move's perspective.

    w2/b2 are accepted but unused -- the flat architecture has no hidden layer, so
    w3 is the full 2*HIDDEN-wide output layer applied directly to the clamped
    accumulator (see nnue_halfkp.forward_flat).
    """
    return nnue.forward_flat(stack[ply], st[0], w3, b3)


@njit(inline="always", cache=False)
def apply_move_nnue(
    stack: np.ndarray,
    king_sq: np.ndarray,
    ply: int,
    sq: np.ndarray,
    st_us: int,
    move: int,
    w1: np.ndarray,
    b1: np.ndarray,
    off: np.ndarray,
    on: np.ndarray,
) -> None:
    """Maintain the accumulator and king-square trackers across one make_move,
    already applied to bb/sq/st by the caller -- st_us is the mover's colour from
    BEFORE that flip, sq is the board AFTER it (so bitgen.KING lookups here would
    see the moved piece, which is why the caller passes off/on and the pre-move
    king squares rather than this function re-deriving anything from sq itself).

    off/on come from move_deltas, called by the caller BEFORE make_move (its own
    documented protocol) -- safe to pass here for ANY move, king or not, since a
    king move's own off/on rows are simply unused on the mover's side below.
    """
    stack[ply + 1] = stack[ply]
    king_sq[ply + 1] = king_sq[ply]
    moved_piece_is_king = sq[(move >> 6) & 63] % 6 == bitgen.KING
    if moved_piece_is_king:
        to_square = (move >> 6) & 63
        king_sq[ply + 1, st_us] = to_square
        other = 1 - st_us
        active = features_halfkp.halfkp_active_bb(sq, king_sq[ply + 1, 0], king_sq[ply + 1, 1])
        nnue.refresh(stack[ply + 1, st_us:st_us + 1], w1, b1, active[st_us:st_us + 1])
        nnue.update(stack[ply + 1, other:other + 1], w1, off[other:other + 1], on[other:other + 1])
    else:
        nnue.update(stack[ply + 1], w1, off, on)


@njit(inline="always", cache=False)
def root_refresh(
    stack: np.ndarray,
    king_sq: np.ndarray,
    sq: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
) -> None:
    """Set up stack[0]/king_sq[0] from scratch for a brand-new root position."""
    white_king, black_king = features_halfkp.find_king_squares(sq)
    king_sq[0, 0] = white_king
    king_sq[0, 1] = black_king
    nnue.refresh(stack[0], w1, b1, features_halfkp.halfkp_active_bb(sq, white_king, black_king))


@njit(inline="always", cache=False)
def move_deltas(
    sq: np.ndarray, st: np.ndarray, move: int, king_sq_0: int, king_sq_1: int
):
    """Feature deltas for one move, computed BEFORE make_move (see apply_move_nnue)."""
    return features_halfkp.halfkp_deltas_bb(sq, st, move, king_sq_0, king_sq_1)
