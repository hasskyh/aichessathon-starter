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
)
from zobrist import (
    zobrist_delta_bb,
    zobrist_hash_bb,
)

PIECE_VALUE = np.array([100, 320, 330, 500, 900, 0], dtype=np.int64)
MOBILITY_WEIGHT = 4
MATE = 1_000_000
INF = 1 << 30
QUIESCE_DEPTH = 6
MAX_DEPTH = 64

# ctrl[0] deadline, ctrl[1] nodes, ctrl[2] abort flag, ctrl[3] next node to check the
# clock at. Counting down to a checkpoint beats a modulo on every node.
CHECK_INTERVAL = 2048.0

# Transposition table
TT_SIZE = 1 << 22
TT_MASK = TT_SIZE - 1

TT_KEY   = np.zeros(TT_SIZE, dtype=np.uint64)   # full hash, for collision detection -- must
                                                 # be uint64 like zobrist_hash_bb's return value;
                                                 # int64 would make a hash with its top bit set
                                                 # compare as negative and corrupt TT_MASK indexing
TT_MOVE  = np.full(TT_SIZE, -1, dtype=np.int32)  # best move found, else -1
TT_SCORE = np.zeros(TT_SIZE, dtype=np.int32)
TT_DEPTH = np.zeros(TT_SIZE, dtype=np.int8)      # depth this score was searched to
TT_TYPE  = np.zeros(TT_SIZE, dtype=np.int8)      # 0=EXACT, 1=LOWER_BOUND, 2=UPPER_BOUND
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
HASH_MOVE_SCORE = 10_000  # comfortably above PIECE_VALUE's max (900, a queen): always sorts first

# The Zobrist hash of the position at each ply, maintained exactly like STACK is for
# the NNUE accumulator: refreshed once at the root (in think()), then updated at every
# make_move via zobrist_delta_bb -- never recomputed from scratch mid-search.
HASH_STACK = np.zeros(MAX_PLY + 1, dtype=np.uint64)

STACK = np.zeros((MAX_PLY + 1, 2, nnue.HIDDEN), dtype=np.int32)  # the NNUE accumulator stack

_WEIGHTS = np.load(Path(__file__).resolve().parent / "weights" / "nnue.npz")
# np.load on an .npz can hand back a non-writable, non-C-contiguous view (W2 is
# Fortran-ordered because export.py's transpose preserved that layout through the
# save/load round trip); np.array() forces an independent, C-contiguous copy.
# Separately, and not fixed by any of that: numba types a module-level global array
# as read-only the moment it is touched inside a jitted function, regardless of the
# array's own .flags.writeable. That is why W1..B3 are threaded through negamax,
# quiescence and nnue_evaluate as ordinary parameters below, exactly like stack and
# ctrl already are, rather than read directly as globals from inside jitted code.
# think() and _warm() are plain Python, so they read the globals directly and pass
# them onward -- the restriction only bites once you are inside nopython mode.
# ascontiguousarray, not array(): np.array()'s default order='K' preserves
# whatever memory layout the .npz happened to store. export.py's w2 is built via
# a transpose, which produces Fortran-ordered data even after np.array() copies
# it -- silently correct but with every row access striding across the whole
# array instead of walking contiguous memory, and unable to auto-vectorize under
# nnue.py's now-declared-contiguous signatures. ascontiguousarray forces genuine
# C order regardless of what was on disk.
W1 = np.ascontiguousarray(_WEIGHTS["w1"])
B1 = np.ascontiguousarray(_WEIGHTS["b1"])
W2 = np.ascontiguousarray(_WEIGHTS["w2"])
B2 = np.ascontiguousarray(_WEIGHTS["b2"])
W3 = np.ascontiguousarray(_WEIGHTS["w3"])
B3 = int(_WEIGHTS["b3"])

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
    """Material plus mobility, from the side to move's point of view.

    Superseded by nnue_evaluate below; kept as a reference and for A/B testing.
    """
    us = st[0]
    them = 1 - us
    material = 0
    for kind in range(5):  # pawn through queen; the king is never counted
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
    """NNUE score for the position at this ply, from the side to move's perspective.

    stack[ply] must already reflect this exact position: refreshed once at the
    root, then kept current by nnue.update at every make_move since.
    """
    return nnue.forward(stack[ply], st[0], w2, b2, w3, b3)


@njit(inline="always", cache=False)
def _order_score(sq: np.ndarray, move: int) -> int:
    victim = sq[(move >> 6) & 63]
    if victim < 0:
        return 0
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
        return  # no point overwriting a deeper, more valuable search
    node_type = TT_UPPER if score <= alpha_orig else TT_LOWER if score >= beta else TT_EXACT
    tt_key[idx] = key
    tt_move[idx] = best_move
    tt_score[idx] = score
    tt_depth[idx] = depth
    tt_type[idx] = node_type

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
) -> int:
    if _tick(ctrl):
        return 0
    if ply >= MAX_PLY - 2:
        return nnue_evaluate(stack, ply, st, w2, b2, w3, b3)

    count, checkers = gen_moves_ex(bb, st, buf[ply], 0)
    if count == 0:
        return -MATE if checkers else 0

    if checkers:
        # In check the original searches every legal move, unordered. Left alone.
        best = -INF
        ordered = False
    else:
        best = nnue_evaluate(stack, ply, st, w2, b2, w3, b3)
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
        off, on = features.deltas_bb(sq, st, move)
        make_move(bb, sq, st, move, hist, ply)
        stack[ply + 1] = stack[ply]
        nnue.update(stack[ply + 1], w1, off, on)
        score = -quiescence(
            bb, sq, st, -beta, -alpha, qdepth - 1, ply + 1, buf, scores, hist, ctrl,
            stack, w1, w2, b2, w3, b3,
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
    hash_stack: np.ndarray,
    tt_key: np.ndarray,
    tt_move: np.ndarray,
    tt_score: np.ndarray,
    tt_depth: np.ndarray,
    tt_type: np.ndarray,
) -> tuple[int, int]:
    # Every return path below returns (score, best_move) -- the caller negates just
    # the score (score = -child_score), never the pair, since a move one ply down
    # is that node's own candidate, not something meaningful to this one.
    if _tick(ctrl):
        return 0, -1
    if depth == 0:
        # Quiescence returns a plain int score, not a (score, move) pair -- the TT
        # only covers the main search, not quiescence. Quiescence nodes are far more
        # numerous and shallow, so the standard, simpler choice (what most engines
        # do) is to skip TT bookkeeping there; -1 stands in for "no move of my own",
        # since this ply never generated or looped over any candidates itself.
        return quiescence(
            bb, sq, st, alpha, beta, QUIESCE_DEPTH, ply, buf, scores, hist, ctrl,
            stack, w1, w2, b2, w3, b3,
        ), -1

    key = hash_stack[ply]
    alpha_orig = alpha
    found, found_score, hash_move = tt_probe(
        tt_key, tt_move, tt_score, tt_depth, tt_type, key, depth, alpha, beta
    )
    if found:
        return found_score, hash_move

    count, checkers = gen_moves_ex(bb, st, buf[ply], 0)
    if count == 0:
        return (-MATE if checkers else 0), -1
    for i in range(count):
        scores[ply, i] = _order_score(sq, buf[ply, i])
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
        nnue.update(stack[ply + 1], w1, off, on)
        hash_stack[ply + 1] = hash_stack[ply] ^ hash_delta
        child_score, _ = negamax(
            bb, sq, st, -beta, -alpha, depth - 1, ply + 1, buf, scores, hist, ctrl,
            stack, w1, w2, b2, w3, b3, hash_stack,
            tt_key, tt_move, tt_score, tt_depth, tt_type,
        )
        score = -child_score
        unmake_move(bb, sq, st, move, hist, ply)
        if ctrl[2] != 0.0:
            # The search was cut short by the clock -- score is whatever a partial,
            # possibly-aborted child call happened to return, not a real result.
            # Storing it would poison the table with a fake "fully searched" entry.
            return 0, -1
        if score > best:
            best = score
            best_move = move
            if score > alpha:
                alpha = score
        if score >= beta:
            # This move caused the cutoff, so it -- not whatever the loop was on
            # before -- is the refutation worth remembering here.
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
) -> tuple[int, int, int]:
    """Iterative deepening. Returns the chosen move, the depth it survived, its score.

    Deliberately plain Python. The root runs a few hundred iterations per move against
    a tree of millions of nodes, so jitting it saved no measurable time while costing
    7.5 seconds of the import budget to compile. Everything below the root is jitted.

    Moves stay as numpy int32 rather than Python ints because that is the type the
    jitted search passes to make_move; handing it a Python int would make numba compile
    a second int64 specialisation, on the clock.
    """
    count = gen_moves(bb, st, buf[0], 0)
    if count == 0:
        return -1, 0, 0

    moves = [np.int32(buf[0, i]) for i in range(count)]
    pv = moves[0]
    reached = 0
    value = 0
    nnue.refresh(stack[0], W1, B1, features.active_bb(sq))
    hash_stack[0] = zobrist_hash_bb(sq, st)
    for depth in range(1, max_depth + 1):
        best_move = -1
        best_score = -INF
        for move in [pv] + [m for m in moves if m != pv]:
            off, on = features.deltas_bb(sq, st, move)
            hash_delta = zobrist_delta_bb(sq, st, move)
            make_move(bb, sq, st, move, hist, 0)
            stack[1] = stack[0]
            nnue.update(stack[1], W1, off, on)
            hash_stack[1] = hash_stack[0] ^ hash_delta
            child_score, _ = negamax(
                bb, sq, st, -INF, INF, depth - 1, 1, buf, scores, hist, ctrl,
                stack, W1, W2, B2, W3, B3, hash_stack,
                TT_KEY, TT_MOVE, TT_SCORE, TT_DEPTH, TT_TYPE,
            )
            score = -child_score
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
    move, depth, _score = think(bb, sq, st, BUF, SCORES, HIST, CTRL, STACK, MAX_DEPTH, HASH_STACK)
    last_depth = depth
    last_nodes = CTRL[1]
    if move < 0:
        return "0000"  # unreachable: the platform never asks for a move once mated
    return bitgen.to_uci(move)


def _warm() -> None:
    """Compile the search at import, where the 60 second init budget pays for it.

    One negamax call at depth 2 pulls in the whole jitted tree: quiescence, evaluate,
    the ordering helpers, bitgen's generation and make/unmake at the int32 move type
    the search really uses, and the NNUE path (refresh, update, forward) at the real
    weight shapes. Argument types must match the real call exactly, or numba compiles
    a second specialisation later, on the clock.
    """
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
        bb, sq, st, -INF, INF, 1, 1, BUF, SCORES, HIST, CTRL, STACK, W1, W2, B2, W3, B3, HASH_STACK,
        TT_KEY, TT_MOVE, TT_SCORE, TT_DEPTH, TT_TYPE,
    )
    unmake_move(bb, sq, st, move, HIST, 0)


_warm()
