# bitgen speed, the original invictus_quiesce evaluation, both known bugs fixed.
#
# This is what the bitgen port looked like before nnue_evaluate replaced evaluate()
# throughout -- same alpha-beta, same quiescence depth, same move ordering, so the
# strength difference against invictus_quiesce is speed and nothing else. It also
# carried invictus_quiesce's root-window bug (see get_move there): think()'s root
# loop called negamax with beta=-best_score instead of a full window, so a move
# searched after best_score had already improved could return an early, unsound
# fail-high bound that was then compared directly against other moves' exact
# scores as if it were one too. Fixed here the same way: a full window for every
# root move.
#
# No features.py or nnue.py needed at all -- this eval only looks at bb and st.

import time

import numpy as np
from numba import njit, objmode

import bitgen
from bitgen import (
    FLAG_EP,
    MAX_MOVES,
    MAX_PLY,
    gen_moves,
    gen_moves_ex,
    make_move,
    popcount,
    unmake_move,
)

PIECE_VALUE = np.array([100, 320, 330, 500, 900, 0], dtype=np.int64)
# The original scores a quiet move as if it captured a pawn, because piece_at returns
# None and its fallback is a pawn. Kept, so ordering matches move for move.
QUIET_SCORE = 100
MOBILITY_WEIGHT = 4
MATE = 1_000_000
INF = 1 << 30
QUIESCE_DEPTH = 6
MAX_DEPTH = 64

# ctrl[0] deadline, ctrl[1] nodes, ctrl[2] abort flag, ctrl[3] next node to check the
# clock at. Counting down to a checkpoint beats a modulo on every node.
CHECK_INTERVAL = 2048.0


@njit(cache=False)
def _now() -> float:
    """Wall clock inside nopython mode. objmode costs ~700ns, so call it rarely."""
    with objmode(t="f8"):
        t = time.monotonic()
    return t


@njit(inline="always", cache=False)
def _tick(ctrl: np.ndarray) -> bool:
    ctrl[1] += 1.0
    if ctrl[1] >= ctrl[3]:
        ctrl[3] = ctrl[1] + CHECK_INTERVAL
        if _now() > ctrl[0]:
            ctrl[2] = 1.0
    return ctrl[2] != 0.0


@njit(inline="always", cache=False)
def evaluate(bb: np.ndarray, st: np.ndarray, mobility: int) -> int:
    """Material plus mobility, from the side to move's point of view."""
    us = st[0]
    them = 1 - us
    material = 0
    for kind in range(5):  # pawn through queen; the king is never counted
        material += PIECE_VALUE[kind] * (
            popcount(bb[us * 6 + kind]) - popcount(bb[them * 6 + kind])
        )
    return material + MOBILITY_WEIGHT * mobility


@njit(inline="always", cache=False)
def _order_score(sq: np.ndarray, move: int) -> int:
    victim = sq[(move >> 6) & 63]
    if victim < 0:
        return QUIET_SCORE
    return PIECE_VALUE[victim % 6]


@njit(inline="always", cache=False)
def _select(moves: np.ndarray, scores: np.ndarray, start: int, count: int) -> None:
    """Swap the best remaining move into position `start`.

    Picking the earliest maximum each time reproduces a stable descending sort exactly,
    but stops as soon as a beta cutoff does, so most nodes never order their whole list.
    """
    best = start
    for i in range(start + 1, count):
        if scores[i] > scores[best]:
            best = i
    if best != start:
        moves[start], moves[best] = moves[best], moves[start]
        scores[start], scores[best] = scores[best], scores[start]


@njit(inline="always", cache=False)
def _keep_captures(sq: np.ndarray, moves: np.ndarray, count: int) -> int:
    """Compact the capture moves to the front, preserving their order. Returns how many.

    Generation emits moves in a fixed order, so the survivors here are exactly the list
    gen_captures would build, and a linear scan beats a second full generation.
    """
    kept = 0
    for i in range(count):
        move = moves[i]
        if sq[(move >> 6) & 63] >= 0 or (move >> 15) & 3 == FLAG_EP:
            moves[kept] = move
            kept += 1
    return kept


@njit(cache=False)
def quiescence(
    bb: np.ndarray,
    sq: np.ndarray,
    st: np.ndarray,
    alpha: int,
    beta: int,
    qdepth: int,
    ply: int,
    buf: np.ndarray,
    scores: np.ndarray,
    hist: np.ndarray,
    ctrl: np.ndarray,
) -> int:
    if _tick(ctrl):
        return 0

    count, checkers = gen_moves_ex(bb, st, buf[ply], 0)
    if ply >= MAX_PLY - 2:
        return evaluate(bb, st, count)
    if count == 0:
        return -MATE if checkers else 0

    if checkers:
        # In check the original searches every legal move, unordered. Left alone.
        best = -INF
        ordered = False
    else:
        best = evaluate(bb, st, count)
        if best >= beta or qdepth == 0:
            return best
        if best > alpha:
            alpha = best
        # The captures are already in this list, in the order gen_captures would have
        # produced them, so compact them in place rather than generating a second time.
        count = _keep_captures(sq, buf[ply], count)
        for i in range(count):
            scores[ply, i] = _order_score(sq, buf[ply, i])
        ordered = True

    for i in range(count):
        if ordered:
            _select(buf[ply], scores[ply], i, count)
        move = buf[ply, i]
        make_move(bb, sq, st, move, hist, ply)
        score = -quiescence(
            bb, sq, st, -beta, -alpha, qdepth - 1, ply + 1, buf, scores, hist, ctrl,
        )
        unmake_move(bb, sq, st, move, hist, ply)
        if ctrl[2] != 0.0:
            return 0
        if score >= beta:
            return score
        if score > best:
            best = score
        if score > alpha:
            alpha = score
    return best


@njit(cache=False)
def negamax(
    bb: np.ndarray,
    sq: np.ndarray,
    st: np.ndarray,
    alpha: int,
    beta: int,
    depth: int,
    ply: int,
    buf: np.ndarray,
    scores: np.ndarray,
    hist: np.ndarray,
    ctrl: np.ndarray,
) -> int:
    if _tick(ctrl):
        return 0
    if depth == 0:
        # Quiescence returns the same mate and stalemate scores, so handing off before
        # generating avoids generating this node's moves twice.
        return quiescence(bb, sq, st, alpha, beta, QUIESCE_DEPTH, ply, buf, scores, hist, ctrl)

    count, checkers = gen_moves_ex(bb, st, buf[ply], 0)
    if count == 0:
        return -MATE if checkers else 0
    for i in range(count):
        scores[ply, i] = _order_score(sq, buf[ply, i])

    best = -MATE
    for i in range(count):
        _select(buf[ply], scores[ply], i, count)
        move = buf[ply, i]
        make_move(bb, sq, st, move, hist, ply)
        score = -negamax(
            bb, sq, st, -beta, -alpha, depth - 1, ply + 1, buf, scores, hist, ctrl,
        )
        unmake_move(bb, sq, st, move, hist, ply)
        if ctrl[2] != 0.0:
            return 0
        if score > best:
            best = score
            if score > alpha:
                alpha = score
        if score >= beta:
            return score
    return best


def think(
    bb: np.ndarray,
    sq: np.ndarray,
    st: np.ndarray,
    buf: np.ndarray,
    scores: np.ndarray,
    hist: np.ndarray,
    ctrl: np.ndarray,
    max_depth: int,
) -> tuple[int, int, int]:
    """Iterative deepening. Returns the chosen move, the depth it survived, its score.

    Deliberately plain Python. The root runs a few hundred iterations per move against
    a tree of millions of nodes, so jitting it saved no measurable time while costing
    seconds of the import budget to compile. Everything below the root is jitted.
    """
    count = gen_moves(bb, st, buf[0], 0)
    if count == 0:
        return -1, 0, 0

    moves = [np.int32(buf[0, i]) for i in range(count)]
    pv = moves[0]
    reached = 0
    value = 0
    for depth in range(1, max_depth + 1):
        best_move = -1
        best_score = -INF
        for move in [pv] + [m for m in moves if m != pv]:
            make_move(bb, sq, st, move, hist, 0)
            # Full window for every move (see module docstring): a narrower one here
            # would make later moves' returned scores unsound fail-high bounds rather
            # than exact values, no longer safely comparable to best_score.
            score = -negamax(bb, sq, st, -INF, INF, depth - 1, 1, buf, scores, hist, ctrl)
            unmake_move(bb, sq, st, move, hist, 0)
            if ctrl[2] != 0.0:
                break
            if score > best_score:
                best_score = score
                best_move = move
        if ctrl[2] != 0.0:
            break  # an unfinished depth is discarded, as in the original
        if best_move != -1:
            pv = best_move
            reached = depth
            value = best_score
        if time.monotonic() > ctrl[0]:
            break
    return pv, reached, value


# Module state survives between moves in one game, so allocate the scratch once.
BUF = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
SCORES = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int64)
HIST = np.zeros((MAX_PLY, 4), dtype=np.int64)
CTRL = np.zeros(4, dtype=np.float64)

last_depth = 0
last_nodes = 0.0


def get_move(fen: str, time_left_ms: int) -> str:
    global last_depth, last_nodes
    bb, sq, st = bitgen.from_fen(fen)
    CTRL[0] = time.monotonic() + max(time_left_ms / 25_000.0, 0.01)
    CTRL[1] = 0.0
    CTRL[2] = 0.0
    CTRL[3] = CHECK_INTERVAL
    move, depth, _score = think(bb, sq, st, BUF, SCORES, HIST, CTRL, MAX_DEPTH)
    last_depth = depth
    last_nodes = CTRL[1]
    if move < 0:
        return "0000"  # unreachable: the platform never asks for a move once mated
    return bitgen.to_uci(move)


def _warm() -> None:
    """Compile the search at import, where the 60 second init budget pays for it."""
    bb, sq, st = bitgen.from_fen(bitgen.STARTING_FEN)
    CTRL[0] = time.monotonic() + 60.0
    CTRL[1] = 0.0
    CTRL[2] = 0.0
    CTRL[3] = CHECK_INTERVAL
    negamax(bb, sq, st, -INF, INF, 2, 0, BUF, SCORES, HIST, CTRL)


_warm()
