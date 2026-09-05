# PROFILING COPY of baselines/invictus/invictus_v5/agent.py -- not used by compare.sh
# or the competition harness. Adds cheap per-node counters (plain integer increments,
# the same cost class as the existing node counter in _tick) and real wall-clock
# per-depth timing in think() (plain Python already, no objmode needed there).
#
# Deliberately NOT wall-clock timing inside jitted per-node sections like tt_probe or
# the NMP block: objmode costs ~700ns per call (see _now()'s own docstring), and those
# sections fire on a large fraction of the millions of nodes a real search visits --
# bracketing them with _now() would make the timer's own overhead dwarf whatever it's
# measuring. For the actual cost of a specific hot section, the stub-and-measure
# approach (disable it, remeasure nodes/sec) that found the game_history bug is the
# right tool; this file only tells you how often each branch fires and how time
# divides across iterative-deepening depths.

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

# ctrl[0] deadline, ctrl[1] nodes, ctrl[2] abort flag, ctrl[3] next node to check the
# clock at. Counting down to a checkpoint beats a modulo on every node.
CHECK_INTERVAL = 2048.0

# Transposition table
TT_SIZE = 1 << 22
TT_MASK = TT_SIZE - 1

TT_KEY   = np.zeros(TT_SIZE, dtype=np.uint64)    # full hash, for collision detection -- must
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

HISTORY = np.zeros((2, 64, 64), dtype=np.int32)   # Butterfly history board [color][from][to]
HISTORY_CAP = 80 # Capped score so that it's never more than a capture, no matter how bad a capture

# Repetition detection. Bounded at 101: the halfmove clock (st[3]) resets to 0 on every
# irreversible move (pawn push or capture), and a repetition can never reach back across
# one of those, so that's the hard ceiling on how far back this ever needs to look. Two
# entries get appended per get_move() call (the position we were handed, and the position
# our own chosen move creates -- get_move() only ever observes every OTHER real ply, since
# it's called once per our own turn), so 101 covers the worst case of 50 calls between one
# clock reset and the next (halfmove clock 0 -> 100).
GAME_HISTORY = np.zeros(101, dtype=np.uint64)
GAME_HISTORY_LEN = 0

STACK = np.zeros((MAX_PLY + 1, 2, nnue.HIDDEN), dtype=np.int32)  # the NNUE accumulator stack

# --- Profiling counters -----------------------------------------------------------
# Plain int64 increments only (no timing) -- cheap enough to leave on every node,
# the same cost class as ctrl[1] += 1.0 in _tick.
N_MAIN_NODES = 0        # every call to negamax (before the quiescence handoff)
N_QUIESCE_NODES = 1      # every call to quiescence
N_TT_PROBES = 2          # every tt_probe call
N_TT_HITS = 3            # tt_probe returned found=True
N_NMP_ATTEMPTS = 4       # NMP's guard conditions passed, null-move search attempted
N_NMP_CUTOFFS = 5        # that null-move search caused a beta cutoff
N_REPETITION_HITS = 6    # a node was resolved as a repetition draw
N_CUTOFF_MOVE_0 = 7      # a beta cutoff caused by the first move tried (best case ordering)
N_CUTOFF_MOVE_LATER = 8  # a beta cutoff caused by any later move (ordering missed it)
N_COUNTERS = 9
COUNTERS = np.zeros(N_COUNTERS, dtype=np.int64)

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
def _order_score(sq: np.ndarray, move: int, butterfly: np.ndarray) -> int:
    victim = sq[(move >> 6) & 63]
    if victim < 0:
        us = sq[move & 63] // 6
        return min(butterfly[us, move & 63, (move >> 6) & 63], HISTORY_CAP)
    attacker = sq[move & 63] % 6
    return PIECE_VALUE[victim % 6] - attacker


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
    tt_type: np.ndarray, key: int, depth: int, alpha: int, beta: int, counters: np.ndarray,
) -> tuple[bool, int, int]:
    counters[N_TT_PROBES] += 1
    idx = key & TT_MASK
    if tt_key[idx] != key:
        return False, 0, -1
    hash_move = tt_move[idx]
    if tt_depth[idx] < depth:
        return False, 0, hash_move
    score, node_type = tt_score[idx], tt_type[idx]
    if node_type == TT_EXACT:
        counters[N_TT_HITS] += 1
        return True, score, hash_move
    if node_type == TT_LOWER and score >= beta:
        counters[N_TT_HITS] += 1
        return True, score, hash_move
    if node_type == TT_UPPER and score <= alpha:
        counters[N_TT_HITS] += 1
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

@njit(inline="always", cache=False)
def make_null_move(st: np.ndarray, hist: np.ndarray, ply: int) -> None:
    hist[ply, 2] = st[2]    # Saves en passant rights
    hist[ply, 3] = st[3]    # Saves halfmove clock
    us = st[0]
    st[2] = -1              # No en passant after being passed a turn
    st[3] += 1              # Halfmove count obviously goes up
    st[0] = 1 - us          # Flip side to move
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
    butterfly: np.ndarray,
    counters: np.ndarray,
) -> int:
    counters[N_QUIESCE_NODES] += 1
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
            scores[ply, i] = _order_score(sq, buf[ply, i], butterfly)
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
            stack, w1, w2, b2, w3, b3, butterfly, counters,
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
    butterfly: np.ndarray,
    game_history: np.ndarray,
    game_history_len: int,
    counters: np.ndarray,
) -> tuple[int, int]:
    counters[N_MAIN_NODES] += 1
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
            stack, w1, w2, b2, w3, b3, butterfly, counters,
        ), -1

    key = hash_stack[ply]

    # A repeat of an earlier real-game position, or of a position already visited
    # earlier in THIS search path, is a draw -- checked before the TT, since it's an
    # unconditional fact about the position, not a heuristic bound. Flagged the first
    # time it's seen (not only on a strict third occurrence): if a repeat is reachable
    # at all, the side that wants it can force it, so treating it as available the
    # moment the search notices it is the standard, conservative choice.
    for i in range(game_history_len):
        if game_history[i] == key:
            counters[N_REPETITION_HITS] += 1
            return 0, -1
    for p in range(ply):
        if hash_stack[p] == key:
            counters[N_REPETITION_HITS] += 1
            return 0, -1

    alpha_orig = alpha
    found, found_score, hash_move = tt_probe(
        tt_key, tt_move, tt_score, tt_depth, tt_type, key, depth, alpha, beta, counters
    )
    if found:
        return found_score, hash_move

    count, checkers = gen_moves_ex(bb, st, buf[ply], 0)
    if count == 0:
        return (-MATE if checkers else 0), -1
    if not checkers and depth >= 3 and has_non_pawn_material(bb, st[0]):
        counters[N_NMP_ATTEMPTS] += 1
        hash_delta = ZOBRIST_SIDE
        if st[2] >= 0:
            hash_delta ^= ZOBRIST_EP[st[2] & 7]
        make_null_move(st, hist, ply)
        stack[ply + 1] = stack[ply]
        hash_stack[ply + 1] = hash_stack[ply] ^ hash_delta
        child_score, _ = negamax(bb, sq, st, -beta, -beta + 1, depth - 1 - R, ply + 1,
                                 buf, scores, hist, ctrl, stack, w1, w2, b2, w3, b3,
                                 hash_stack, tt_key, tt_move, tt_score, tt_depth, tt_type,
                                 butterfly, game_history, game_history_len, counters,
                            )
        score = - child_score
        unmake_null_move(st, hist, ply)
        if ctrl[2] != 0.0:
            return 0, -1
        if score >= beta:
            counters[N_NMP_CUTOFFS] += 1
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
        nnue.update(stack[ply + 1], w1, off, on)
        hash_stack[ply + 1] = hash_stack[ply] ^ hash_delta
        child_score, _ = negamax(
            bb, sq, st, -beta, -alpha, depth - 1, ply + 1, buf, scores, hist, ctrl,
            stack, w1, w2, b2, w3, b3, hash_stack,
            tt_key, tt_move, tt_score, tt_depth, tt_type, butterfly, game_history,
            game_history_len, counters,
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
            if i == 0:
                counters[N_CUTOFF_MOVE_0] += 1
            else:
                counters[N_CUTOFF_MOVE_LATER] += 1
            if sq[(move >> 6) & 63] < 0: # Meaning this is not a capture
                us = sq[move & 63] // 6  # Mover's colour
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
    counters: np.ndarray,
    depth_log: list,
) -> tuple[int, int, int]:
    """Iterative deepening. Returns the chosen move, the depth it survived, its score.

    depth_log gets one (depth, elapsed_seconds, nodes_at_depth_end) tuple appended per
    completed depth -- real wall-clock timing, no objmode needed since this function
    is already plain Python.
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
        depth_start = time.monotonic()
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
                nnue.update(stack[1], W1, off, on)
                hash_stack[1] = hash_stack[0] ^ hash_delta
                child_score, _ = negamax(
                    bb, sq, st, -beta, -alpha, depth - 1, 1, buf, scores, hist, ctrl,
                    stack, W1, W2, B2, W3, B3, hash_stack,
                    TT_KEY, TT_MOVE, TT_SCORE, TT_DEPTH, TT_TYPE,
                    HISTORY, game_history, game_history_len, counters,
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
        # An aborted scan is still salvageable if what it found so far already cleared
        # the window -- that's a genuine validated score, not a fail-low bound, so a
        # well-ordered partial result really can beat the previous depth here. But if
        # best_score never cleared alpha, every score examined is an unresolved bound
        # (the same condition that would have triggered a widen-and-retry had there
        # been time), and comparing bounds against each other is meaningless -- discard
        # the whole depth in that case, same as the fully-aborted case always did.
        if aborted and best_score <= alpha:
            break
        if best_move != -1:
            pv = best_move
            reached = depth
            value = best_score
        depth_log.append((depth, time.monotonic() - depth_start, ctrl[1]))
        if aborted or time.monotonic() > ctrl[0]:
            break
    return pv, reached, value


# Module state survives between moves in one game, so allocate the scratch once.
BUF = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
SCORES = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int64)
HIST = np.zeros((MAX_PLY, 4), dtype=np.int64)
CTRL = np.zeros(4, dtype=np.float64)

last_depth = 0
last_nodes = 0.0


def get_move(fen: str, time_left_ms: int, verbose: bool = True) -> str:
    """Same interface as the real agent's get_move, plus a printed profiling summary.

    verbose=True (the default here, unlike the production file) prints the counter
    breakdown and per-depth timing after every call -- fine for a profiling script
    calling this a handful of times, not something the real agent would ever do.
    """
    global last_depth, last_nodes, GAME_HISTORY_LEN
    COUNTERS[:] = 0
    depth_log: list = []

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
    t0 = time.monotonic()
    move, depth, _score = think(bb, sq, st, BUF, SCORES, HIST, CTRL, STACK, MAX_DEPTH, HASH_STACK,
                                ASPIRATION_MARGIN, GAME_HISTORY, GAME_HISTORY_LEN, COUNTERS,
                                depth_log,
                            )
    wall = time.monotonic() - t0
    last_depth = depth
    last_nodes = CTRL[1]

    if verbose:
        _print_profile(wall, depth_log)

    if move < 0:
        return "0000"  # unreachable: the platform never asks for a move once mated

    hash_delta = zobrist_delta_bb(sq, st, move)
    GAME_HISTORY[GAME_HISTORY_LEN] = current_hash ^ hash_delta
    GAME_HISTORY_LEN += 1

    return bitgen.to_uci(move)


def _print_profile(wall: float, depth_log: list) -> None:
    c = COUNTERS
    print(f"--- profile: wall={wall:.3f}s  final depth={last_depth}  nodes={last_nodes:.0f}  "
          f"nodes/sec={last_nodes / wall if wall > 0 else 0:.0f} ---")
    print(f"  main nodes:       {c[N_MAIN_NODES]:>12,}")
    print(f"  quiescence nodes: {c[N_QUIESCE_NODES]:>12,}")
    print(f"  TT probes:        {c[N_TT_PROBES]:>12,}   hits: {c[N_TT_HITS]:>12,}"
          f"  ({100 * c[N_TT_HITS] / c[N_TT_PROBES] if c[N_TT_PROBES] else 0:.1f}%)")
    print(f"  NMP attempts:     {c[N_NMP_ATTEMPTS]:>12,}   cutoffs: {c[N_NMP_CUTOFFS]:>12,}"
          f"  ({100 * c[N_NMP_CUTOFFS] / c[N_NMP_ATTEMPTS] if c[N_NMP_ATTEMPTS] else 0:.1f}%)")
    print(f"  repetition hits:  {c[N_REPETITION_HITS]:>12,}")
    total_cutoffs = c[N_CUTOFF_MOVE_0] + c[N_CUTOFF_MOVE_LATER]
    print(f"  beta cutoffs on move 0 (best-case ordering): {c[N_CUTOFF_MOVE_0]:>12,}"
          f"  ({100 * c[N_CUTOFF_MOVE_0] / total_cutoffs if total_cutoffs else 0:.1f}%)")
    print(f"  beta cutoffs on a later move (ordering missed it): {c[N_CUTOFF_MOVE_LATER]:>12,}"
          f"  ({100 * c[N_CUTOFF_MOVE_LATER] / total_cutoffs if total_cutoffs else 0:.1f}%)")
    print(f"  {'depth':>5} {'time(s)':>9} {'cum_nodes':>12} {'nodes_this_depth':>16} {'nodes/sec_this_depth':>20}")
    prev_nodes = 0.0
    for d, elapsed, cum_nodes in depth_log:
        this_depth_nodes = cum_nodes - prev_nodes
        prev_nodes = cum_nodes
        nps = this_depth_nodes / elapsed if elapsed > 0 else 0
        print(f"  {d:>5} {elapsed:>9.4f} {cum_nodes:>12,.0f} {this_depth_nodes:>16,.0f} {nps:>20,.0f}")


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
        TT_KEY, TT_MOVE, TT_SCORE, TT_DEPTH, TT_TYPE, HISTORY, GAME_HISTORY, GAME_HISTORY_LEN,
        COUNTERS,
    )
    unmake_move(bb, sq, st, move, HIST, 0)


_warm()
