# The invictus_quiesce search, unchanged in behaviour, moved onto bitgen's bitboards.
#
# The search itself has to be jitted, not just move generation. A jitted generator
# called from a Python search still pays interpreter overhead at every node, which is
# most of the cost, so the whole tree walk lives in nopython mode here and Python is
# entered exactly twice per move: once to parse the fen, once to format the reply.
#
# The algorithm is a faithful port, deliberately: same evaluation, same alpha-beta,
# same quiescence depth, same move ordering including its quirks, so the strength
# difference against invictus_quiesce is speed and nothing else.
#
# v6 adds a second, phase-specific NNUE: once a line's own piece count drops below
# ENDGAME_PIECE_LIMIT, the search switches from the main net's weights to a net
# trained only on such positions (see training/train.py --outputs, ckpt-endgame-*.pt).
# Piece count is monotonic non-increasing down any single line (captures and
# promotions never add a piece), so once a node crosses the threshold every
# descendant of it has too -- the regime never has to switch back, only forward.
#
# An earlier pass measured this design's import time at 79s on one core -- over the
# platform's 60s budget -- and briefly replaced it with a once-per-move version that
# picks a net before entering the tree at all, at the cost of never switching mid-search.
# That 79s measurement turned out to be contaminated: it was taken while an unrelated
# HalfKP training run hammered the machine's other cores, and shared cache/memory
# bandwidth contention slows a single pinned core down too, independent of CPU
# affinity. Measured clean (nothing else running), this design compiles in ~23s --
# comfortably inside budget -- so it was restored in favour of the once-per-move
# fallback, which is a strictly weaker approximation of the same idea.

import time
from pathlib import Path

import numpy as np
from numba import njit, objmode

import bitgen
import features
import nnue
from bitgen import (
    FLAG_EP,
    MAX_MOVES,
    MAX_PLY,
    gen_moves,
    gen_moves_ex,
    make_move,
    popcount,
    unmake_move,
    has_non_pawn_material,
)
from zobrist import (
    ZOBRIST_EP,
    ZOBRIST_SIDE,
    zobrist_delta_bb,
    zobrist_hash_bb,
)

PIECE_VALUE = np.array([100, 320, 330, 500, 900, 0], dtype=np.int64)
MOBILITY_WEIGHT = 4
MATE = 1_000_000
INF = 1 << 30
QUIESCE_DEPTH = 6
MAX_DEPTH = 64
R = 2 # How deep to null move prune
ASPIRATION_MARGIN = 50
ENDGAME_PIECE_LIMIT = 8  # total pieces (both sides, kings included) below which the endgame net takes over

# ctrl[0] deadline, ctrl[1] nodes, ctrl[2] abort flag, ctrl[3] next node to check the
# clock at. Counting down to a checkpoint beats a modulo on every node.
CHECK_INTERVAL = 2048.0

# Transposition table
TT_SIZE = 1 << 22
TT_MASK = TT_SIZE - 1

TT_KEY   = np.zeros(TT_SIZE, dtype=np.uint64)
TT_MOVE  = np.full(TT_SIZE, -1, dtype=np.int32)
TT_SCORE = np.zeros(TT_SIZE, dtype=np.int32)
TT_DEPTH = np.zeros(TT_SIZE, dtype=np.int8)
TT_TYPE  = np.zeros(TT_SIZE, dtype=np.int8)
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
HASH_MOVE_SCORE = 10_000

HASH_STACK = np.zeros(MAX_PLY + 1, dtype=np.uint64)

HISTORY = np.zeros((2, 64, 64), dtype=np.int32)
HISTORY_CAP = 80

GAME_HISTORY = np.zeros(101, dtype=np.uint64)
GAME_HISTORY_LEN = 0

STACK = np.zeros((MAX_PLY + 1, 2, nnue.HIDDEN), dtype=np.int32)

REGIME = np.zeros(MAX_PLY + 1, dtype=np.int8)

_WEIGHTS = np.load(Path(__file__).resolve().parent / "weights" / "nnue.npz")
W1 = np.ascontiguousarray(_WEIGHTS["w1"])
B1 = np.ascontiguousarray(_WEIGHTS["b1"])
W2 = np.ascontiguousarray(_WEIGHTS["w2"])
B2 = np.ascontiguousarray(_WEIGHTS["b2"])
W3 = np.ascontiguousarray(_WEIGHTS["w3"])
B3 = int(_WEIGHTS["b3"])

_WEIGHTS_ENDGAME = np.load(Path(__file__).resolve().parent / "weights" / "endgame" / "nnue.npz")
W1E = np.ascontiguousarray(_WEIGHTS_ENDGAME["w1"])
B1E = np.ascontiguousarray(_WEIGHTS_ENDGAME["b1"])
W2E = np.ascontiguousarray(_WEIGHTS_ENDGAME["w2"])
B2E = np.ascontiguousarray(_WEIGHTS_ENDGAME["b2"])
W3E = np.ascontiguousarray(_WEIGHTS_ENDGAME["w3"])
B3E = int(_WEIGHTS_ENDGAME["b3"])

@njit(cache=False)
def _now() -> float:
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
def total_pieces(bb: np.ndarray) -> int:
    count = 0
    for i in range(bb.shape[0]):
        count += popcount(bb[i])
    return count


@njit(inline="always", cache=False)
def evaluate(bb: np.ndarray, st: np.ndarray, mobility: int) -> int:
    us = st[0]
    them = 1 - us
    material = 0
    for kind in range(5):
        material += PIECE_VALUE[kind] * (
            popcount(bb[us * 6 + kind]) - popcount(bb[them * 6 + kind])
        )
    return material + MOBILITY_WEIGHT * mobility


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
def _order_score(sq: np.ndarray, move: int, butterfly: np.ndarray) -> int:
    victim = sq[(move >> 6) & 63]
    if victim < 0:
        us = sq[move & 63] // 6
        return min(butterfly[us, move & 63, (move >> 6) & 63], HISTORY_CAP)
    attacker = sq[move & 63] % 6
    return PIECE_VALUE[victim % 6] - attacker


@njit(inline="always", cache=False)
def _select(moves: np.ndarray, scores: np.ndarray, start: int, count: int) -> None:
    best = start
    for i in range(start + 1, count):
        if scores[i] > scores[best]:
            best = i
    if best != start:
        moves[start], moves[best] = moves[best], moves[start]
        scores[start], scores[best] = scores[best], scores[start]


@njit(inline="always", cache=False)
def _keep_captures(sq: np.ndarray, moves: np.ndarray, count: int) -> int:
    kept = 0
    for i in range(count):
        move = moves[i]
        if sq[(move >> 6) & 63] >= 0 or (move >> 15) & 3 == FLAG_EP:
            moves[kept] = move
            kept += 1
    return kept

@njit(inline="always", cache=False)
def tt_probe(
    tt_key: np.ndarray, tt_move: np.ndarray, tt_score: np.ndarray, tt_depth: np.ndarray,
    tt_type: np.ndarray, key: int, depth: int, alpha: int, beta: int,
) -> tuple[bool, int, int]:
    idx = key & TT_MASK
    if tt_key[idx] != key:
        return False, 0, -1
    hash_move = tt_move[idx]
    if tt_depth[idx] < depth:
        return False, 0, hash_move
    score, node_type = tt_score[idx], tt_type[idx]
    if node_type == TT_EXACT:
        return True, score, hash_move
    if node_type == TT_LOWER and score >= beta:
        return True, score, hash_move
    if node_type == TT_UPPER and score <= alpha:
        return True, score, hash_move
    return False, 0, hash_move

@njit(inline="always", cache=False)
def tt_store(
    tt_key: np.ndarray, tt_move: np.ndarray, tt_score: np.ndarray, tt_depth: np.ndarray,
    tt_type: np.ndarray, key: int, depth: int, score: int, best_move: int, alpha_orig: int,
    beta: int,
) -> None:
    idx = key & TT_MASK
    if tt_key[idx] == key and tt_depth[idx] > depth:
        return
    node_type = TT_UPPER if score <= alpha_orig else TT_LOWER if score >= beta else TT_EXACT
    tt_key[idx] = key
    tt_move[idx] = best_move
    tt_score[idx] = score
    tt_depth[idx] = depth
    tt_type[idx] = node_type

@njit(inline="always", cache=False)
def make_null_move(st: np.ndarray, hist: np.ndarray, ply: int) -> None:
    hist[ply, 2] = st[2]
    hist[ply, 3] = st[3]
    us = st[0]
    st[2] = -1
    st[3] += 1
    st[0] = 1 - us
    st[4] += us

@njit(inline="always", cache=False)
def unmake_null_move(st: np.ndarray, hist: np.ndarray, ply: int) -> None:
    us = 1 - st[0]
    st[0] = us
    st[2] = hist[ply, 2]
    st[3] = hist[ply, 3]
    st[4] -= us

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
    stack: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: int,
    regime: np.ndarray,
    w1e: np.ndarray,
    b1e: np.ndarray,
    w2e: np.ndarray,
    b2e: np.ndarray,
    w3e: np.ndarray,
    b3e: int,
    butterfly: np.ndarray
) -> int:
    if _tick(ctrl):
        return 0
    if ply >= MAX_PLY - 2:
        if regime[ply] == 1:
            return nnue_evaluate(stack, ply, st, w2e, b2e, w3e, b3e)
        return nnue_evaluate(stack, ply, st, w2, b2, w3, b3)

    count, checkers = gen_moves_ex(bb, st, buf[ply], 0)
    if count == 0:
        return -MATE if checkers else 0

    if checkers:
        best = -INF
        ordered = False
    else:
        if regime[ply] == 1:
            best = nnue_evaluate(stack, ply, st, w2e, b2e, w3e, b3e)
        else:
            best = nnue_evaluate(stack, ply, st, w2, b2, w3, b3)
        if best >= beta or qdepth == 0:
            return best
        if best > alpha:
            alpha = best
        count = _keep_captures(sq, buf[ply], count)
        for i in range(count):
            scores[ply, i] = _order_score(sq, buf[ply, i], butterfly)
        ordered = True

    for i in range(count):
        if ordered:
            _select(buf[ply], scores[ply], i, count)
        move = buf[ply, i]
        off, on = features.deltas_bb(sq, st, move)
        make_move(bb, sq, st, move, hist, ply)
        stack[ply + 1] = stack[ply]
        if regime[ply] == 1:
            nnue.update(stack[ply + 1], w1e, off, on)
            regime[ply + 1] = 1
        elif total_pieces(bb) < ENDGAME_PIECE_LIMIT:
            nnue.refresh(stack[ply + 1], w1e, b1e, features.active_bb(sq))
            regime[ply + 1] = 1
        else:
            nnue.update(stack[ply + 1], w1, off, on)
            regime[ply + 1] = 0
        score = -quiescence(
            bb, sq, st, -beta, -alpha, qdepth - 1, ply + 1, buf, scores, hist, ctrl,
            stack, w1, w2, b2, w3, b3, regime, w1e, b1e, w2e, b2e, w3e, b3e, butterfly
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
    stack: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: int,
    regime: np.ndarray,
    w1e: np.ndarray,
    b1e: np.ndarray,
    w2e: np.ndarray,
    b2e: np.ndarray,
    w3e: np.ndarray,
    b3e: int,
    hash_stack: np.ndarray,
    tt_key: np.ndarray,
    tt_move: np.ndarray,
    tt_score: np.ndarray,
    tt_depth: np.ndarray,
    tt_type: np.ndarray,
    butterfly: np.ndarray,
    game_history: np.ndarray,
    game_history_len: int,
) -> tuple[int, int]:
    if _tick(ctrl):
        return 0, -1
    if depth == 0:
        return quiescence(
            bb, sq, st, alpha, beta, QUIESCE_DEPTH, ply, buf, scores, hist, ctrl,
            stack, w1, w2, b2, w3, b3, regime, w1e, b1e, w2e, b2e, w3e, b3e, butterfly
        ), -1

    key = hash_stack[ply]
    for i in range(game_history_len):
        if game_history[i] == key:
            return 0, -1
    for p in range(ply):
        if hash_stack[p] == key:
            return 0, -1

    alpha_orig = alpha
    found, found_score, hash_move = tt_probe(
        tt_key, tt_move, tt_score, tt_depth, tt_type, key, depth, alpha, beta
    )
    if found:
        return found_score, hash_move

    count, checkers = gen_moves_ex(bb, st, buf[ply], 0)
    if count == 0:
        return (-MATE if checkers else 0), -1
    if not checkers and depth >= 3 and has_non_pawn_material(bb, st[0]):
        hash_delta = ZOBRIST_SIDE
        if st[2] >= 0:
            hash_delta ^= ZOBRIST_EP[st[2] & 7]
        make_null_move(st, hist, ply)
        stack[ply + 1] = stack[ply]
        regime[ply + 1] = regime[ply]
        hash_stack[ply + 1] = hash_stack[ply] ^ hash_delta
        child_score, _ = negamax(bb, sq, st, -beta, -beta + 1, depth - 1 - R, ply + 1,
                                 buf, scores, hist, ctrl, stack, w1, w2, b2, w3, b3,
                                 regime, w1e, b1e, w2e, b2e, w3e, b3e,
                                 hash_stack, tt_key, tt_move, tt_score, tt_depth, tt_type,
                                 butterfly, game_history, game_history_len
                            )
        score = - child_score
        unmake_null_move(st, hist, ply)
        if ctrl[2] != 0.0:
            return 0, -1
        if score >= beta:
            return score, -1
    for i in range(count):
        scores[ply, i] = _order_score(sq, buf[ply, i], butterfly)
    if hash_move != -1:
        for i in range(count):
            if buf[ply, i] == hash_move:
                scores[ply, i] = HASH_MOVE_SCORE
                break

    best = -MATE
    best_move = -1
    for i in range(count):
        _select(buf[ply], scores[ply], i, count)
        move = buf[ply, i]
        off, on = features.deltas_bb(sq, st, move)
        hash_delta = zobrist_delta_bb(sq, st, move)
        make_move(bb, sq, st, move, hist, ply)
        stack[ply + 1] = stack[ply]
        if regime[ply] == 1:
            nnue.update(stack[ply + 1], w1e, off, on)
            regime[ply + 1] = 1
        elif total_pieces(bb) < ENDGAME_PIECE_LIMIT:
            nnue.refresh(stack[ply + 1], w1e, b1e, features.active_bb(sq))
            regime[ply + 1] = 1
        else:
            nnue.update(stack[ply + 1], w1, off, on)
            regime[ply + 1] = 0
        hash_stack[ply + 1] = hash_stack[ply] ^ hash_delta
        child_score, _ = negamax(
            bb, sq, st, -beta, -alpha, depth - 1, ply + 1, buf, scores, hist, ctrl,
            stack, w1, w2, b2, w3, b3, regime, w1e, b1e, w2e, b2e, w3e, b3e, hash_stack,
            tt_key, tt_move, tt_score, tt_depth, tt_type, butterfly, game_history, game_history_len
        )
        score = -child_score
        unmake_move(bb, sq, st, move, hist, ply)
        if ctrl[2] != 0.0:
            return 0, -1
        if score > best:
            best = score
            best_move = move
            if score > alpha:
                alpha = score
        if score >= beta:
            if sq[(move >> 6) & 63] < 0:
                us = sq[move & 63] // 6
                butterfly[us, move & 63, (move >> 6) & 63] += depth * depth
            tt_store(
                tt_key, tt_move, tt_score, tt_depth, tt_type,
                key, depth, score, move, alpha_orig, beta,
            )
            return score, move

    tt_store(
        tt_key, tt_move, tt_score, tt_depth, tt_type,
        key, depth, best, best_move, alpha_orig, beta,
    )
    return best, best_move


def think(
    bb: np.ndarray,
    sq: np.ndarray,
    st: np.ndarray,
    buf: np.ndarray,
    scores: np.ndarray,
    hist: np.ndarray,
    ctrl: np.ndarray,
    stack: np.ndarray,
    max_depth: int,
    hash_stack: np.ndarray,
    margin: int,
    game_history: np.ndarray,
    game_history_len: int,
    regime: np.ndarray,
) -> tuple[int, int, int]:
    count = gen_moves(bb, st, buf[0], 0)
    if count == 0:
        return -1, 0, 0

    moves = [np.int32(buf[0, i]) for i in range(count)]
    pv = moves[0]
    reached = 0
    value = 0

    if total_pieces(bb) < ENDGAME_PIECE_LIMIT:
        nnue.refresh(stack[0], W1E, B1E, features.active_bb(sq))
        regime[0] = 1
    else:
        nnue.refresh(stack[0], W1, B1, features.active_bb(sq))
        regime[0] = 0
    hash_stack[0] = zobrist_hash_bb(sq, st)
    for depth in range(1, max_depth + 1):
        alpha, beta = (-INF, INF) if depth == 1 else (value - margin, value + margin)
        aborted = False
        while True:
            best_move = -1
            best_score = -INF
            for move in [pv] + [m for m in moves if m != pv]:
                off, on = features.deltas_bb(sq, st, move)
                hash_delta = zobrist_delta_bb(sq, st, move)
                make_move(bb, sq, st, move, hist, 0)
                stack[1] = stack[0]
                if regime[0] == 1:
                    nnue.update(stack[1], W1E, off, on)
                    regime[1] = 1
                elif total_pieces(bb) < ENDGAME_PIECE_LIMIT:
                    nnue.refresh(stack[1], W1E, B1E, features.active_bb(sq))
                    regime[1] = 1
                else:
                    nnue.update(stack[1], W1, off, on)
                    regime[1] = 0
                hash_stack[1] = hash_stack[0] ^ hash_delta
                child_score, _ = negamax(
                    bb, sq, st, -beta, -alpha, depth - 1, 1, buf, scores, hist, ctrl,
                    stack, W1, W2, B2, W3, B3, regime, W1E, B1E, W2E, B2E, W3E, B3E, hash_stack,
                    TT_KEY, TT_MOVE, TT_SCORE, TT_DEPTH, TT_TYPE,
                    HISTORY, game_history, game_history_len,
                )
                score = -child_score
                unmake_move(bb, sq, st, move, hist, 0)
                if ctrl[2] != 0.0:
                    aborted = True
                    break
                if score > best_score:
                    best_score = score
                    best_move = move
            if aborted:
                break
            if best_score <= alpha and alpha > -INF:
                alpha = -INF
                continue
            if best_score >= beta and beta < INF:
                beta = INF
                continue
            break
        if aborted and best_score <= alpha:
            break
        if best_move != -1:
            pv = best_move
            reached = depth
            value = best_score
        if aborted or time.monotonic() > ctrl[0]:
            break
    return pv, reached, value


BUF = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
SCORES = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int64)
HIST = np.zeros((MAX_PLY, 4), dtype=np.int64)
CTRL = np.zeros(4, dtype=np.float64)

last_depth = 0
last_nodes = 0.0


def get_move(fen: str, time_left_ms: int) -> str:
    global last_depth, last_nodes, GAME_HISTORY_LEN
    bb, sq, st = bitgen.from_fen(fen)
    current_hash = zobrist_hash_bb(sq, st)

    if st[3] == 0:
        GAME_HISTORY[:] = 0
        GAME_HISTORY_LEN = 0
    GAME_HISTORY[GAME_HISTORY_LEN] = current_hash
    GAME_HISTORY_LEN += 1

    CTRL[0] = time.monotonic() + max(time_left_ms / 25_000.0, 0.01)
    CTRL[1] = 0.0
    CTRL[2] = 0.0
    CTRL[3] = CHECK_INTERVAL
    move, depth, _score = think(bb, sq, st, BUF, SCORES, HIST, CTRL, STACK, MAX_DEPTH, HASH_STACK,
                                ASPIRATION_MARGIN, GAME_HISTORY, GAME_HISTORY_LEN, REGIME
                            )
    last_depth = depth
    last_nodes = CTRL[1]
    if move < 0:
        return "0000"

    hash_delta = zobrist_delta_bb(sq, st, move)
    GAME_HISTORY[GAME_HISTORY_LEN] = current_hash ^ hash_delta
    GAME_HISTORY_LEN += 1

    return bitgen.to_uci(move)


def _warm() -> None:
    bb, sq, st = bitgen.from_fen(bitgen.STARTING_FEN)
    CTRL[0] = time.monotonic() + 60.0
    CTRL[1] = 0.0
    CTRL[2] = 0.0
    CTRL[3] = CHECK_INTERVAL
    gen_moves(bb, st, BUF[0], 0)
    move = np.int32(BUF[0, 0])

    nnue.refresh(STACK[0], W1, B1, features.active_bb(sq))
    HASH_STACK[0] = zobrist_hash_bb(sq, st)
    off, on = features.deltas_bb(sq, st, move)
    hash_delta = zobrist_delta_bb(sq, st, move)
    make_move(bb, sq, st, move, HIST, 0)
    STACK[1] = STACK[0]
    nnue.update(STACK[1], W1, off, on)
    HASH_STACK[1] = HASH_STACK[0] ^ hash_delta
    negamax(
        bb, sq, st, -INF, INF, 1, 1, BUF, SCORES, HIST, CTRL, STACK, W1, W2, B2, W3, B3,
        REGIME, W1E, B1E, W2E, B2E, W3E, B3E, HASH_STACK,
        TT_KEY, TT_MOVE, TT_SCORE, TT_DEPTH, TT_TYPE, HISTORY, GAME_HISTORY, GAME_HISTORY_LEN
    )
    unmake_move(bb, sq, st, move, HIST, 0)


_warm()
