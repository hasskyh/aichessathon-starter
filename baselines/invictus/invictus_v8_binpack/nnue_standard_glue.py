"""Bridges the search's stack/board representation to the standard NNUE
architecture (nnue_768.py) and its feature encoding (features_768.py): 768
features, piece-on-square only, no king-relative structure, two-layer
768-512-32-1 shape (accumulator -> hidden -> output, via nnue.forward). The
checkpoint this loads must have w2/b2 -- nnue_768.py's OTHER forward variant,
forward_flat, is the single-layer 768-512-1 architecture (no hidden layer at
all) and belongs to a separate nnue_flat_glue.py, not this file.

Same shape as nnue_halfkp_glue.py (HIDDEN, nnue_evaluate, apply_move_nnue,
root_refresh, move_deltas) so agent.py can import either module under the same
names and never touch negamax/quiescence/think. king_sq is accepted everywhere
nnue_halfkp_glue.py's versions take it, purely so every call site is identical
either way -- this architecture never reads or writes it, since there's no king
move special case without king-relative features.
"""

import numpy as np
from numba import njit

import features_768 as features
import nnue_768 as nnue

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
    return nnue.forward(stack[ply], st[0], w2, b2, w3, b3)


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
    stack[ply + 1] = stack[ply]
    king_sq[ply + 1] = king_sq[ply]  # never read by this architecture, kept for parity
    nnue.update(stack[ply + 1], w1, off, on)


@njit(inline="always", cache=False)
def root_refresh(
    stack: np.ndarray,
    king_sq: np.ndarray,
    sq: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
) -> None:
    nnue.refresh(stack[0], w1, b1, features.active_bb(sq))


@njit(inline="always", cache=False)
def move_deltas(
    sq: np.ndarray, st: np.ndarray, move: int, king_sq_0: int, king_sq_1: int
):
    return features.deltas_bb(sq, st, move)
