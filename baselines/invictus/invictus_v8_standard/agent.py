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
# v7 is invictus_v5 with a HalfKP net (features_halfkp.py, nnue_halfkp.py) instead of
# the flat 768-feature one -- same TT/null-move/aspiration-window search, same
# abort-handling fix, only the evaluation changes. HalfKP's features are king-relative
# (see features_halfkp.py's module docstring), so every one of a perspective's
# features changes meaning the moment that perspective's own king moves. The search
# has to know this: KING_SQ tracks both kings' current squares per ply (mirroring
# STACK's own per-ply convention), and every make-move site below checks whether the
# mover was a king. A non-king move updates both perspectives incrementally exactly
# like the flat net does. A king move takes the documented HalfKP protocol instead:
# the mover's own perspective gets a full refresh (its features are relative to a
# king square that just changed, so no incremental delta from the old position means
# anything), while the OTHER perspective -- whose king did not move -- still applies
# the ordinary incremental delta, which is non-empty exactly when the king move was
# also a capture or a castle (the rook's own relocation still needs applying there).

import time
import math
from pathlib import Path

import numpy as np
from numba import njit, objmode

import bitgen
from nnue_standard_glue import apply_move_nnue, move_deltas, nnue_evaluate, root_refresh
from nnue_standard_glue import HIDDEN as NNUE_HIDDEN
from bitgen import (
    FLAG_EP,
    MAX_MOVES,
    MAX_PLY,
    gen_moves,
    gen_moves_ex,
    has_non_pawn_material,
    make_move,
    popcount,
    unmake_move,
    attackers_to,
    bit,
    lsb,
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
MATE_THRESHOLD = MATE - MAX_PLY # Anything larger than this is mate, but closer mates are better
FUTILITY_DEPTH_LIMIT = 6
FUTILITY_MARGIN = 150

INF = 1 << 30
QUIESCE_DEPTH = 6
MAX_DEPTH = 64
R = 2 # How deep to null move prune
ASPIRATION_MARGIN = 50
LMP_DEPTH_LIMIT = 8
RFP_DEPTH_LIMIT = 6
RFP_MARGIN = 120 # Slightly more than a pawn per remaining ply

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

# Killer moves are moves that have caused a beta cutoff for several moves, so they should be tried
# early in similar positions
KILLERS = np.full((MAX_PLY + 1, 2), -1, dtype=np.int32)
KILLER_SCORE = 90 

# Repetition detection. Bounded at 101: the halfmove clock (st[3]) resets to 0 on every
# irreversible move (pawn push or capture), and a repetition can never reach back across
# one of those, so that's the hard ceiling on how far back this ever needs to look. Two
# entries get appended per get_move() call (the position we were handed, and the position
# our own chosen move creates -- get_move() only ever observes every OTHER real ply, since
# it's called once per our own turn), so 101 covers the worst case of 50 calls between one
# clock reset and the next (halfmove clock 0 -> 100).
GAME_HISTORY = np.zeros(101, dtype=np.uint64)
GAME_HISTORY_LEN = 0

STACK = np.zeros((MAX_PLY + 1, 2, NNUE_HIDDEN), dtype=np.int32)  # the NNUE accumulator stack

# Both kings' current squares at each ply, maintained the same way STACK is: copied
# down from the parent ply at every make_move, then the mover's own entry corrected
# in place if (and only if) the move just made was that king's own move. Needed
# because halfkp_deltas_bb/halfkp_active_bb take king squares as explicit arguments
# rather than re-deriving them from sq -- find_king_squares exists for exactly the
# refresh case (root, or a king move) and is deliberately never called elsewhere.
KING_SQ = np.zeros((MAX_PLY + 1, 2), dtype=np.int32)

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
# Some architectures (e.g. the flat/general net's forward_flat) have no hidden
# layer at all, so their checkpoints simply have no w2/b2 -- fall back to an
# empty array of the right dtype/ndim so agent.py loads either checkpoint shape
# without numba ever seeing a different type signature for W2/B2.
W2 = (
    np.ascontiguousarray(_WEIGHTS["w2"]) if "w2" in _WEIGHTS
    else np.zeros((0, 0), dtype=np.int8)
)
B2 = (
    np.ascontiguousarray(_WEIGHTS["b2"]) if "b2" in _WEIGHTS
    else np.zeros(0, dtype=np.int32)
)
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
def _order_score(sq: np.ndarray, move: int, butterfly: np.ndarray, killers: np.ndarray, ply: int) -> int:
    victim = sq[(move >> 6) & 63]
    if victim < 0:
        if move == killers[ply, 0] or move == killers[ply, 1]:
            return KILLER_SCORE
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
def _lmr_reduction(depth: int, i: int) -> int:
    # This performs log-logarithmic reduction on depth for the ith move
    if depth < 3 or i < 3:
        return 0
    r = int(0.75 + math.log(depth) * math.log(i) / 2.25)
    if r > depth - 1: # This ensures it is never negative
        return depth - 1
    return r

@njit(inline="always", cache=False)
def see(bb: np.ndarray, sq: np.ndarray, move: int) -> int:
    # This is a static exchange evaluation, basically guessing at what captures will occur
    # and is used to see if a node has any hope of being better than alpha
    from_sq = move & 63
    to_sq = (move >> 6) & 63
    us = sq[from_sq] // 6

    gain = np.empty(32, dtype=np.int64)
    if (move >> 15) & 3 == FLAG_EP:
        gain[0] = PIECE_VALUE[0]
    else:
        gain[0] = PIECE_VALUE[sq[to_sq] % 6]
    last_value = PIECE_VALUE[sq[from_sq] % 6]

    occupied = bb[14] ^ bit(from_sq)
    side = 1 - us
    depth = 0
    while True:
        attackers = attackers_to(bb, to_sq, occupied, side) & occupied
        if not attackers:
            break
        least_kind = -1
        least_sq = -1
        for kind in range(6):
            candidates = attackers & bb[side * 6 + kind]
            if candidates:
                least_kind = kind
                least_sq = lsb(candidates)
                break
        depth += 1
        gain[depth] = last_value - gain[depth - 1]
        occupied ^= bit(least_sq)
        last_value = PIECE_VALUE[least_kind]
        side = 1 - side

    while depth > 0:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1
    return gain[0]


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
    tt_type: np.ndarray, key: int, depth: int, alpha: int, beta: int, ply: int
) -> tuple[bool, int, int]:
    idx = key & TT_MASK
    if tt_key[idx] != key:
        return False, 0, -1
    hash_move = tt_move[idx]
    if tt_depth[idx] < depth:
        return False, 0, hash_move
    score, node_type = tt_score[idx], tt_type[idx]
    if score > MATE_THRESHOLD:
        score -= ply
    elif score < -MATE_THRESHOLD:
        score += ply
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
    beta: int, ply: int
) -> None:
    idx = key & TT_MASK
    if tt_key[idx] == key and tt_depth[idx] > depth:
        return  # no point overwriting a deeper, more valuable search
    node_type = TT_UPPER if score <= alpha_orig else TT_LOWER if score >= beta else TT_EXACT
    store_score = score
    # Have to undo adjustments
    if score > MATE_THRESHOLD:
        store_score = score + ply
    elif score < -MATE_THRESHOLD:
        store_score = score - ply
    tt_key[idx] = key
    tt_move[idx] = best_move
    tt_score[idx] = store_score
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
    king_sq: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: int,
    butterfly: np.ndarray,
    killers: np.ndarray
) -> int:
    if _tick(ctrl):
        return 0
    if ply >= MAX_PLY - 2:
        return nnue_evaluate(stack, ply, st, w2, b2, w3, b3)

    count, checkers = gen_moves_ex(bb, st, buf[ply], 0)
    if count == 0:
        return -MATE + ply if checkers else 0

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
        kept = 0
        for i in range(count):
            if see(bb, sq, buf[ply, i]) >= 0:
                buf[ply, kept] = buf[ply, i]
                kept += 1
        count = kept
        for i in range(count):
            scores[ply, i] = _order_score(sq, buf[ply, i], butterfly, killers, ply)
        ordered = True

    for i in range(count):
        if ordered:
            _select(buf[ply], scores[ply], i, count)
        move = buf[ply, i]
        us = st[0]
        off, on = move_deltas(sq, st, move, king_sq[ply, 0], king_sq[ply, 1])
        make_move(bb, sq, st, move, hist, ply)
        apply_move_nnue(stack, king_sq, ply, sq, us, move, w1, b1, off, on)
        score = -quiescence(
            bb, sq, st, -beta, -alpha, qdepth - 1, ply + 1, buf, scores, hist, ctrl,
            stack, king_sq, w1, b1, w2, b2, w3, b3, butterfly, killers
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
    king_sq: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
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
    killers: np.ndarray,
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
            stack, king_sq, w1, b1, w2, b2, w3, b3, butterfly, killers,
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
            return 0, -1
    for p in range(ply):
        if hash_stack[p] == key:
            return 0, -1

    alpha_orig = alpha
    found, found_score, hash_move = tt_probe(
        tt_key, tt_move, tt_score, tt_depth, tt_type, key, depth, alpha, beta, ply
    )
    if found:
        return found_score, hash_move

    count, checkers = gen_moves_ex(bb, st, buf[ply], 0)
    if count == 0:
        return (-MATE + ply if checkers else 0), -1

    static_eval = 0
    not_too_deep = depth <= max(RFP_DEPTH_LIMIT, FUTILITY_DEPTH_LIMIT)
    within_bounds = abs(alpha) < MATE_THRESHOLD and abs(beta) < MATE_THRESHOLD
    if not checkers and not_too_deep and within_bounds:
        static_eval = nnue_evaluate(stack, ply, st, w2, b2, w3, b3)
        if depth <= RFP_DEPTH_LIMIT:
            margin = RFP_MARGIN * depth
            if static_eval - margin >= beta:
                return static_eval - margin, -1
    
    if not checkers and depth >= 3 and has_non_pawn_material(bb, st[0]):
        hash_delta = ZOBRIST_SIDE
        if st[2] >= 0:
            hash_delta ^= ZOBRIST_EP[st[2] & 7]
        make_null_move(st, hist, ply)
        stack[ply + 1] = stack[ply]
        king_sq[ply + 1] = king_sq[ply]  # a null move never moves anyone's king
        hash_stack[ply + 1] = hash_stack[ply] ^ hash_delta
        child_score, _ = negamax(bb, sq, st, -beta, -beta + 1, depth - 1 - R, ply + 1,
                                 buf, scores, hist, ctrl, stack, king_sq, w1, b1, w2, b2, w3, b3,
                                 hash_stack, tt_key, tt_move, tt_score, tt_depth, tt_type,
                                 butterfly, game_history, game_history_len, killers
                            )
        score = - child_score
        unmake_null_move(st, hist, ply)
        if ctrl[2] != 0.0:
            return 0, -1
        if score >= beta:
            return score, -1
    for i in range(count):
        scores[ply, i] = _order_score(sq, buf[ply, i], butterfly, killers, ply)
    if hash_move != -1:
        for i in range(count):
            if buf[ply, i] == hash_move:
                scores[ply, i] = HASH_MOVE_SCORE
                break

    quiets_tried = 0
    best = -MATE
    best_move = -1
    for i in range(count):
        _select(buf[ply], scores[ply], i, count)
        move = buf[ply, i]
        us = st[0]

        is_quiet = sq[(move >> 6) & 63] < 0
        if (is_quiet and not checkers and depth <= LMP_DEPTH_LIMIT 
                and quiets_tried >= (3 + depth * depth)):
            continue
        if (is_quiet and not checkers and depth <= FUTILITY_DEPTH_LIMIT
                and best_move != -1 and abs(alpha) < MATE_THRESHOLD
                and static_eval + FUTILITY_MARGIN * depth <= alpha):
            continue
        if is_quiet:
            quiets_tried += 1

        off, on = move_deltas(sq, st, move, king_sq[ply, 0], king_sq[ply, 1])
        hash_delta = zobrist_delta_bb(sq, st, move)
        make_move(bb, sq, st, move, hist, ply)
        apply_move_nnue(stack, king_sq, ply, sq, us, move, w1, b1, off, on)
        hash_stack[ply + 1] = hash_stack[ply] ^ hash_delta
        reduction = _lmr_reduction(depth, i) if (is_quiet and not checkers) else 0
        child_score, _ = negamax(
            bb, sq, st, -beta, -alpha, depth - 1 - reduction, ply + 1, buf, scores, hist, ctrl,
            stack, king_sq, w1, b1, w2, b2, w3, b3, hash_stack,
            tt_key, tt_move, tt_score, tt_depth, tt_type, butterfly,
            game_history, game_history_len, killers,
        )
        score = -child_score
        if reduction > 0 and score > alpha:
            child_score, _ = negamax(
                bb, sq, st, -beta, -alpha, depth - 1, ply + 1, buf, scores, hist, ctrl,
                stack, king_sq, w1, b1, w2, b2, w3, b3, hash_stack,
                tt_key, tt_move, tt_score, tt_depth, tt_type, butterfly,
                game_history, game_history_len, killers,
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
            if sq[(move >> 6) & 63] < 0: # Meaning this is not a capture
                us = sq[move & 63] // 6  # Mover's colour
                butterfly[us, move & 63, (move >> 6) & 63] += depth * depth
                if move != killers[ply, 0]:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = move
            tt_store(
                tt_key, tt_move, tt_score, tt_depth, tt_type,
                key, depth, score, move, alpha_orig, beta, ply 
            )
            return score, move

    tt_store(
        tt_key, tt_move, tt_score, tt_depth, tt_type,
        key, depth, best, best_move, alpha_orig, beta, ply
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
    king_sq: np.ndarray,
    max_depth: int,
    hash_stack: np.ndarray,
    margin: int,
    game_history: np.ndarray,
    game_history_len: int,
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

    for i in range(count):
        scores[0, i] = _order_score(sq, buf[0, i], HISTORY, KILLERS, 0,)
    for i in range(count):
        _select(buf[0], scores[0], i, count)

    moves = [np.int32(buf[0, i]) for i in range(count)]
    pv = moves[0]
    reached = 0
    value = 0

    root_refresh(stack, king_sq, sq, W1, B1)
    hash_stack[0] = zobrist_hash_bb(sq, st)
    for depth in range(1, max_depth + 1):
        for i in range(count):
            scores[0, i] = _order_score(sq, buf[0, i], HISTORY, KILLERS, 0)
        for i in range(count):
            _select(buf[0], scores[0], i, count)
        alpha, beta = (-INF, INF) if depth == 1 else (value - margin, value + margin)
        aborted = False
        while True:
            best_move = -1
            best_score = -INF
            for move in [pv] + [m for m in moves if m != pv]:
                us = st[0]
                off, on = move_deltas(sq, st, move, king_sq[0, 0], king_sq[0, 1])
                hash_delta = zobrist_delta_bb(sq, st, move)
                make_move(bb, sq, st, move, hist, 0)
                apply_move_nnue(stack, king_sq, 0, sq, us, move, W1, B1, off, on)
                hash_stack[1] = hash_stack[0] ^ hash_delta
                child_score, _ = negamax(
                    bb, sq, st, -beta, -alpha, depth - 1, 1, buf, scores, hist, ctrl,
                    stack, king_sq, W1, B1, W2, B2, W3, B3, hash_stack,
                    TT_KEY, TT_MOVE, TT_SCORE, TT_DEPTH, TT_TYPE,
                    HISTORY, game_history, game_history_len, KILLERS
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
        # A scan that is aborted mid iteration is still better than the previous
        # iteration. It will have searched the move we were going to recommend 
        # anyway. If it found that it's still the best, then there's no issue,
        # and if it turns out it's not that good at high depth then we wouldn't
        # want to play it anyway. However, if it failed outside of the bounds, the
        # score it gives is not actually accurate, and we can't order things based
        # on that, so we have to discard it.
        if aborted and not (alpha < best_score < beta):
            break
        if best_move != -1:
            pv = best_move
            reached = depth
            value = best_score
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


def get_move(fen: str, time_left_ms: int) -> str:
    global last_depth, last_nodes, GAME_HISTORY_LEN
    bb, sq, st = bitgen.from_fen(fen)
    current_hash = zobrist_hash_bb(sq, st)

    # st[3] == 0 means the last move played (ours or theirs) was irreversible, so no
    # earlier position can possibly recur -- everything before this point is moot.
    if st[3] == 0:
        GAME_HISTORY[:] = 0
        GAME_HISTORY_LEN = 0
    GAME_HISTORY[GAME_HISTORY_LEN] = current_hash
    GAME_HISTORY_LEN += 1

    CTRL[0] = time.monotonic() + max(time_left_ms / 25_000.0, 0.01)
    CTRL[1] = 0.0
    CTRL[2] = 0.0
    CTRL[3] = CHECK_INTERVAL
    move, depth, _score = think(bb, sq, st, BUF, SCORES, HIST, CTRL, STACK, KING_SQ, MAX_DEPTH,
                                HASH_STACK, ASPIRATION_MARGIN, GAME_HISTORY, GAME_HISTORY_LEN
                            )
    last_depth = depth
    last_nodes = CTRL[1]
    if move < 0:
        return "0000"  # unreachable: the platform never asks for a move once mated

    # Also record the position OUR move creates -- get_move() is only ever called at
    # our own turns, so this is the one real ply the search itself never hands back.
    hash_delta = zobrist_delta_bb(sq, st, move)
    GAME_HISTORY[GAME_HISTORY_LEN] = current_hash ^ hash_delta
    GAME_HISTORY_LEN += 1

    return bitgen.to_uci(move)


def _warm() -> None:
    """Compile the search at import, where the 60 second init budget pays for it.

    One negamax call at depth 2 pulls in the whole jitted tree: quiescence, evaluate,
    the ordering helpers, bitgen's generation and make/unmake at the int32 move type
    the search really uses, and the NNUE path (refresh, update, forward) at the real
    weight shapes. Argument types must match the real call exactly, or numba compiles
    a second specialisation later, on the clock.

    This one non-king first move never exercises the king-move-refresh branch inside
    quiescence/negamax's make-move sites, but numba compiles a jitted function's whole
    body -- every branch of every `if` -- the first time it is called with a given set
    of argument types, not lazily by which branch actually runs a given call. Passing
    b1/king_sq here (matching their real types) is what makes that branch compile too.
    """
    bb, sq, st = bitgen.from_fen(bitgen.STARTING_FEN)
    CTRL[0] = time.monotonic() + 60.0
    CTRL[1] = 0.0
    CTRL[2] = 0.0
    CTRL[3] = CHECK_INTERVAL
    gen_moves(bb, st, BUF[0], 0)
    move = np.int32(BUF[0, 0])

    root_refresh(STACK, KING_SQ, sq, W1, B1)
    HASH_STACK[0] = zobrist_hash_bb(sq, st)
    us = st[0]
    off, on = move_deltas(sq, st, move, KING_SQ[0, 0], KING_SQ[0, 1])
    hash_delta = zobrist_delta_bb(sq, st, move)
    make_move(bb, sq, st, move, HIST, 0)
    apply_move_nnue(STACK, KING_SQ, 0, sq, us, move, W1, B1, off, on)
    HASH_STACK[1] = HASH_STACK[0] ^ hash_delta
    negamax(
        bb, sq, st, -INF, INF, 1, 1, BUF, SCORES, HIST, CTRL, STACK, KING_SQ,
        W1, B1, W2, B2, W3, B3, HASH_STACK,
        TT_KEY, TT_MOVE, TT_SCORE, TT_DEPTH, TT_TYPE, HISTORY, GAME_HISTORY, GAME_HISTORY_LEN, KILLERS
    )
    unmake_move(bb, sq, st, move, HIST, 0)


_warm()
