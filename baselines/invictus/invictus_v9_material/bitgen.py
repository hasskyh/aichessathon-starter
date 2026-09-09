"""Bitboard move generation for chess, compiled with numba.

The board is three flat numpy arrays, so every function here is nopython-clean and a
search can call them in a tight loop without ever re-entering the interpreter:

    bb  uint64[15]  0..5 white P N B R Q K, 6..11 black, 12 white occ, 13 black occ, 14 all
    sq  int8[64]    piece code 0..11 standing on that square, -1 for empty
    st  int64[5]    side to move, castling mask, ep square (-1 for none), halfmove, fullmove

Squares use python-chess's numbering: a1 = 0, h1 = 7, a8 = 56, h8 = 63.

Moves pack into one int32:

    bits 0-5    from square
    bits 6-11   to square
    bits 12-14  promotion piece kind, 0 when there is none
    bits 15-16  flag: 0 normal, 1 double pawn push, 2 castle, 3 en passant

Sliding attacks use magic bitboards, with the magics searched for at import so the
tables are paid for inside the 60 second init budget rather than on the clock.
"""

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic

# --- hardware bit primitives -------------------------------------------------------
# numba has no popcount or count-trailing-zeros builtin, so reach for the LLVM
# intrinsics. Both become a single instruction on any x86-64 worth playing chess on.

_I64 = ir.IntType(64)


@intrinsic
def popcount(typingctx, x):  # type: ignore[no-untyped-def]
    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        fnty = ir.FunctionType(_I64, [_I64])
        fn = builder.module.declare_intrinsic("llvm.ctpop.i64", [], fnty)
        return builder.call(fn, [args[0]])

    return types.int64(types.uint64), codegen


@intrinsic
def lsb(typingctx, x):  # type: ignore[no-untyped-def]
    """Index of the lowest set bit. Undefined at zero, so never call it on zero."""

    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        fnty = ir.FunctionType(_I64, [_I64, ir.IntType(1)])
        fn = builder.module.declare_intrinsic("llvm.cttz.i64", [], fnty)
        return builder.call(fn, [args[0], ir.Constant(ir.IntType(1), 1)])

    return types.int64(types.uint64), codegen


U64 = np.uint64
ONE = U64(1)
ZERO = U64(0)
FULL = U64(0xFFFFFFFFFFFFFFFF)


@njit(inline="always", cache=False)
def bit(square):
    return ONE << U64(square)


# --- geometry tables ---------------------------------------------------------------

FILE_A = U64(0x0101010101010101)
FILE_H = U64(0x8080808080808080)
NOT_FILE_A = ~FILE_A
NOT_FILE_H = ~FILE_H

_KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING_DELTAS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
_ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _step(square, dr, df):
    rank, file = divmod(square, 8)
    rank += dr
    file += df
    if 0 <= rank < 8 and 0 <= file < 8:
        return rank * 8 + file
    return -1


def _hop_table(deltas):
    table = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        acc = 0
        for dr, df in deltas:
            target = _step(square, dr, df)
            if target >= 0:
                acc |= 1 << target
        table[square] = U64(acc)
    return table


KNIGHT_ATT = _hop_table(_KNIGHT_DELTAS)
KING_ATT = _hop_table(_KING_DELTAS)

# PAWN_ATT[colour * 64 + square]: squares a pawn of that colour on that square attacks.
PAWN_ATT = np.zeros(128, dtype=np.uint64)
for _sq in range(64):
    _w = 0
    _b = 0
    for _df in (-1, 1):
        _t = _step(_sq, 1, _df)
        if _t >= 0:
            _w |= 1 << _t
        _t = _step(_sq, -1, _df)
        if _t >= 0:
            _b |= 1 << _t
    PAWN_ATT[_sq] = U64(_w)
    PAWN_ATT[64 + _sq] = U64(_b)


def _ray(square, dirs, blockers=0):
    """Squares reachable along dirs, stopping on and including the first blocker."""
    attacks = 0
    for dr, df in dirs:
        current = square
        while True:
            current = _step(current, dr, df)
            if current < 0:
                break
            attacks |= 1 << current
            if blockers >> current & 1:
                break
    return attacks


ROOK_RAYS = np.array([U64(_ray(s, _ROOK_DIRS)) for s in range(64)], dtype=np.uint64)
BISHOP_RAYS = np.array([U64(_ray(s, _BISHOP_DIRS)) for s in range(64)], dtype=np.uint64)

# BETWEEN[a * 64 + b]: squares strictly between two aligned squares, else empty.
# LINE[a * 64 + b]: the whole line through both squares, else empty. A pinned piece
# may only ever move inside LINE[king, itself].
BETWEEN = np.zeros(4096, dtype=np.uint64)
LINE = np.zeros(4096, dtype=np.uint64)
for _a in range(64):
    for _dirs in (_ROOK_DIRS, _BISHOP_DIRS):
        for _dr, _df in _dirs:
            _path = 0
            _cur = _a
            while True:
                _cur = _step(_cur, _dr, _df)
                if _cur < 0:
                    break
                BETWEEN[_a * 64 + _cur] = U64(_path)
                _path |= 1 << _cur
        _full = _ray(_a, _dirs)
        for _t in range(64):
            if _full >> _t & 1:
                LINE[_a * 64 + _t] = U64((_full & _ray(_t, _dirs)) | (1 << _a) | (1 << _t))

# Trimming the edges off a slider's relevant-occupancy mask is what keeps the magic
# tables small: a blocker on the last square of a ray cannot change what is attacked.
_EDGES = np.zeros(64, dtype=np.uint64)
for _sq in range(64):
    _r, _f = divmod(_sq, 8)
    _mask = 0
    if _r > 0:
        _mask |= 0x00000000000000FF
    if _r < 7:
        _mask |= 0xFF00000000000000
    if _f > 0:
        _mask |= 0x0101010101010101
    if _f < 7:
        _mask |= 0x8080808080808080
    _EDGES[_sq] = U64(_mask)

ROOK_MASK = np.array(
    [U64(int(ROOK_RAYS[s]) & ~int(_EDGES[s]) & 0xFFFFFFFFFFFFFFFF) for s in range(64)],
    dtype=np.uint64,
)
BISHOP_MASK = np.array(
    [U64(int(BISHOP_RAYS[s]) & ~int(_EDGES[s]) & 0xFFFFFFFFFFFFFFFF) for s in range(64)],
    dtype=np.uint64,
)


# --- magic bitboards --------------------------------------------------------------


@njit(cache=False)
def _xorshift(state):
    state ^= state >> U64(12)
    state ^= state << U64(25)
    state ^= state >> U64(27)
    return state * U64(0x2545F4914F6CDD1D)


_ROOK_DIR_ARRAY = np.array(_ROOK_DIRS, dtype=np.int64)
_BISHOP_DIR_ARRAY = np.array(_BISHOP_DIRS, dtype=np.int64)


@njit(cache=False)
def _ray_jit(square, dirs, blockers):
    """The jitted twin of _ray. Building the magic tables calls this ~300k times, and
    in pure Python that alone cost several seconds of the import budget."""
    attacks = ZERO
    for d in range(dirs.shape[0]):
        rank = square // 8
        file = square % 8
        while True:
            rank += dirs[d, 0]
            file += dirs[d, 1]
            if rank < 0 or rank > 7 or file < 0 or file > 7:
                break
            attacks |= bit(rank * 8 + file)
            if blockers & bit(rank * 8 + file):
                break
    return attacks


@njit(cache=False)
def _search_magic(occupancies, attacks, count, bits, seed, table):
    """Find a multiplier mapping every occupancy of one square to a distinct index.

    Collisions are allowed when two occupancies attack the same squares, which is what
    makes a fixed-shift magic findable in a few thousand tries.
    """
    shift = U64(64 - bits)
    state = seed
    while True:
        # A sparse multiplier scatters the relevant bits into the high word far better
        # than a dense one, so AND three draws together.
        state = _xorshift(state)
        magic = state
        state = _xorshift(state)
        magic &= state
        state = _xorshift(state)
        magic &= state
        table[:] = ZERO
        ok = True
        for i in range(count):
            index = (occupancies[i] * magic) >> shift
            if table[index] == ZERO:
                table[index] = attacks[i]
            elif table[index] != attacks[i]:
                ok = False
                break
        if ok:
            return magic, state


@njit(cache=False)
def _build_magics(masks, dirs, max_bits, seed):
    """One packed attack table for all 64 squares, indexed by per-square offset.

    Giving every square its own width instead of a fixed 2^max_bits shrinks the rook
    table from 2 MB to 800 KB. Sliding lookups are the hottest memory access in the
    generator, so keeping the table inside cache is worth the extra offset add.
    """
    widths = np.zeros(64, dtype=np.int64)
    offsets = np.zeros(64, dtype=np.uint64)
    total = 0
    for square in range(64):
        widths[square] = popcount(masks[square])
        offsets[square] = U64(total)
        total += 1 << widths[square]

    magics = np.zeros(64, dtype=np.uint64)
    shifts = np.zeros(64, dtype=np.uint64)
    table = np.zeros(total, dtype=np.uint64)
    scratch = np.zeros(1 << max_bits, dtype=np.uint64)
    occupancies = np.zeros(1 << max_bits, dtype=np.uint64)
    attacks = np.zeros(1 << max_bits, dtype=np.uint64)
    state = seed
    for square in range(64):
        mask = masks[square]
        bits = widths[square]
        count = 1 << bits
        subset = ZERO
        for i in range(count):
            occupancies[i] = subset
            attacks[i] = _ray_jit(square, dirs, subset)
            subset = (subset - mask) & mask  # carry-rippler walk over every subset
        magic, state = _search_magic(occupancies, attacks, count, bits, state, scratch)
        magics[square] = magic
        shifts[square] = U64(64 - bits)
        for i in range(count):
            table[offsets[square] + ((occupancies[i] * magic) >> shifts[square])] = attacks[i]
    return magics, shifts, offsets, table


# Magics found once by _search_magic above (seed 0x9E3779B97F4A7C15) and pasted in.
# They are deterministic - same seed, same masks, same search order always finds the
# same numbers - so re-searching for them on every import bought nothing but compile
# time: _build_magics, _search_magic, _xorshift and _ray_jit are ~2.5s of numba
# compilation that a search-based build pays and this path skips.
EMBEDDED_ROOK_MAGIC = np.array([
    0x1180002240028050, 0x80400020001000C0, 0x8100200100409029, 0x0080080050008004,
    0x8A00116002000408, 0x4300040013002886, 0x090001002C008200, 0x0100024180650002,
    0x0000800081400122, 0x0C04400420005000, 0x400300200010410A, 0x2000801001080080,
    0x0044800400080080, 0x4188801401800200, 0xA011004200940100, 0x0000803444800100,
    0x8005208004400081, 0x1030104002402000, 0x0810008020008050, 0x0200090030010021,
    0x0084808004001800, 0x220080801A000400, 0x2B0054000510080A, 0x00000600088404C1,
    0x2C00400080008020, 0x00300040400C2000, 0x0400410100200610, 0x0008008080500008,
    0x00020032000420C8, 0x2011000900140006, 0x000200C200040128, 0x6408128200204104,
    0x024020C004800080, 0x0000C01081802000, 0x8600403303002000, 0x0200100180800800,
    0x2000800800802C02, 0x1000800400800200, 0x0018180194004210, 0x0000448402000451,
    0x0400400081208000, 0x8420003000484000, 0x088440A009010010, 0x2110000891010021,
    0x100A00C420120008, 0x4001001400090002, 0x8002002304020008, 0x0140068400C60001,
    0x9008402280090100, 0x0020043000400240, 0x4010080020040020, 0x2023033000201900,
    0xA002800800840080, 0x0008804400020080, 0x010A000881140200, 0x4000040043088200,
    0x0004228001423901, 0x8000400024130081, 0x0540130020002841, 0x0080100004082101,
    0x801200A00C504882, 0x0221000204000881, 0x2901004402002581, 0x61200C0082C02112,
], dtype=np.uint64)
EMBEDDED_BISHOP_MAGIC = np.array([
    0x0608208480820080, 0x0020082901052600, 0xF130390210A90044, 0x20C8218028800700,
    0x8801114000001240, 0x8442021004041000, 0x0211009010080001, 0x2002002208048400,
    0x0254A12017010110, 0x2200501011012060, 0x0824080801023052, 0x2280211141000120,
    0x0008440420000C20, 0x280B02080A481024, 0x24800208044C0C04, 0x8082048418821008,
    0x1040002808010C20, 0x0029081042008400, 0x0830000A09820008, 0x00020004012200A0,
    0x8404042882A0200C, 0x8002000900A20900, 0x080140110108A001, 0x001080152094100A,
    0x0260044810850812, 0x0808040A82040810, 0xE80090000A040010, 0xA101080004006020,
    0x0801840008822008, 0x4025020020480400, 0x0818208040C44400, 0x0000821001010090,
    0x0010042108100200, 0x40084404C0100110, 0x80004248021000A0, 0x1402004240840100,
    0x0084040C00003100, 0x6C10010110320040, 0x000A0A44001A4404, 0x1164040420008084,
    0x1342300424002080, 0x0404018211080800, 0x00800C0048000400, 0x2004002218004400,
    0x1000082008200100, 0x002089020A000020, 0x000A0C0800840204, 0x2309022409402108,
    0x4004040304500004, 0x2050808847102018, 0x0102010043100060, 0x0901000460880000,
    0x805A681650440000, 0x0010200404082200, 0x0004180204040000, 0x0008100C14A02010,
    0x5010120230020800, 0x0048020082413000, 0x4800022600420884, 0x0100015040840400,
    0x0004012030060220, 0x80000041A86D0308, 0x0290041044210400, 0x8111140808802200,
], dtype=np.uint64)


def _fill_tables_from_magics(masks, dirs, magics):
    """Fill packed attack tables from known-good magics, in plain Python.

    No numba involved, so this costs interpreter time, never compile time. Every slot
    is checked against the true attack set (via the pure-Python `_ray`, not `_ray_jit`)
    as it's written: if a magic no longer covers its square collision-free, the masks
    it was found for have changed since these constants were captured, and `ok` comes
    back False so the caller can fall back to a real search instead of trusting a table
    that might be wrong.
    """
    widths = [bin(int(masks[s])).count("1") for s in range(64)]
    offsets = [0] * 64
    total = 0
    for s in range(64):
        offsets[s] = total
        total += 1 << widths[s]
    shifts = np.array([64 - w for w in widths], dtype=np.uint64)
    offset_arr = np.array(offsets, dtype=np.uint64)
    table = np.zeros(total, dtype=np.uint64)

    for square in range(64):
        mask = int(masks[square])
        magic = int(magics[square])
        shift = widths[square] and 64 - widths[square]
        base = offsets[square]
        filled = {}
        subset = 0
        for _ in range(1 << widths[square]):
            attacks = _ray(square, dirs, subset)
            index = ((subset * magic) & 0xFFFFFFFFFFFFFFFF) >> shift
            slot = base + index
            if slot in filled and filled[slot] != attacks:
                return shifts, offset_arr, table, False
            filled[slot] = attacks
            table[slot] = U64(attacks)
            subset = (subset - mask) & mask  # carry-rippler walk over every subset
    return shifts, offset_arr, table, True


def _load_magics(masks, dirs, dir_array, embedded_magics, max_bits, seed):
    shifts, offsets, table, ok = _fill_tables_from_magics(masks, dirs, embedded_magics)
    if ok:
        return embedded_magics, shifts, offsets, table
    return _build_magics(masks, dir_array, max_bits, seed)  # constants stale: re-search


_SEED = U64(0x9E3779B97F4A7C15)
ROOK_MAGIC, ROOK_SHIFT, ROOK_OFFSET, ROOK_TABLE = _load_magics(
    ROOK_MASK, _ROOK_DIRS, _ROOK_DIR_ARRAY, EMBEDDED_ROOK_MAGIC, 12, _SEED
)
BISHOP_MAGIC, BISHOP_SHIFT, BISHOP_OFFSET, BISHOP_TABLE = _load_magics(
    BISHOP_MASK, _BISHOP_DIRS, _BISHOP_DIR_ARRAY, EMBEDDED_BISHOP_MAGIC, 9, _SEED
)


@njit(inline="always", cache=False)
def rook_attacks(square, occupied):
    index = ((occupied & ROOK_MASK[square]) * ROOK_MAGIC[square]) >> ROOK_SHIFT[square]
    return ROOK_TABLE[ROOK_OFFSET[square] + index]


@njit(inline="always", cache=False)
def bishop_attacks(square, occupied):
    index = ((occupied & BISHOP_MASK[square]) * BISHOP_MAGIC[square]) >> BISHOP_SHIFT[square]
    return BISHOP_TABLE[BISHOP_OFFSET[square] + index]


@njit(inline="always", cache=False)
def queen_attacks(square, occupied):
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied)


# --- position plumbing -------------------------------------------------------------

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
WHITE, BLACK = 0, 1

FLAG_NORMAL, FLAG_DPUSH, FLAG_CASTLE, FLAG_EP = 0, 1, 2, 3

# Castling rights: bit 0 white kingside, 1 white queenside, 2 black kingside, 3 black
# queenside. A move from or to one of these squares clears the rights it touches.
CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[0] = 15 & ~2
CASTLE_MASK[4] = 15 & ~3
CASTLE_MASK[7] = 15 & ~1
CASTLE_MASK[56] = 15 & ~8
CASTLE_MASK[60] = 15 & ~12
CASTLE_MASK[63] = 15 & ~4

# Per-colour pawn geometry, indexed by side to move.
PUSH_DIR = np.array([8, -8], dtype=np.int64)
RANK_DOUBLE = np.array([U64(0x0000000000FF0000), U64(0x0000FF0000000000)], dtype=np.uint64)
RANK_PROMO = np.array([U64(0xFF00000000000000), U64(0x00000000000000FF)], dtype=np.uint64)
RANK_HOME = np.array([U64(0x000000000000FF00), U64(0x00FF000000000000)], dtype=np.uint64)

MAX_MOVES = 256
MAX_PLY = 128


@njit(inline="always", cache=False)
def _shift_up(bits, side):
    if side == WHITE:
        return bits << U64(8)
    return bits >> U64(8)


@njit(inline="always", cache=False)
def _place(bb, sq, code, square):
    mask = bit(square)
    bb[code] |= mask
    bb[12 + code // 6] |= mask
    bb[14] |= mask
    sq[square] = code


@njit(inline="always", cache=False)
def _lift(bb, sq, code, square):
    mask = ~bit(square)
    bb[code] &= mask
    bb[12 + code // 6] &= mask
    bb[14] &= mask
    sq[square] = -1


@njit(cache=False)
def attackers_to(bb, square, occupied, side):
    """Every piece of `side` attacking `square` under `occupied`."""
    base = side * 6
    return (
        (PAWN_ATT[(1 - side) * 64 + square] & bb[base + PAWN])
        | (KNIGHT_ATT[square] & bb[base + KNIGHT])
        | (KING_ATT[square] & bb[base + KING])
        | (bishop_attacks(square, occupied) & (bb[base + BISHOP] | bb[base + QUEEN]))
        | (rook_attacks(square, occupied) & (bb[base + ROOK] | bb[base + QUEEN]))
    )


@njit(cache=False)
def _danger(bb, side, occupied):
    """Every square `side` attacks, with our king already lifted off the board.

    Lifting the king is what stops it stepping backwards along a checking ray.
    """
    base = side * 6
    pawns = bb[base + PAWN]
    if side == WHITE:
        danger = ((pawns & NOT_FILE_A) << U64(7)) | ((pawns & NOT_FILE_H) << U64(9))
    else:
        danger = ((pawns & NOT_FILE_A) >> U64(9)) | ((pawns & NOT_FILE_H) >> U64(7))
    pieces = bb[base + KNIGHT]
    while pieces:
        danger |= KNIGHT_ATT[lsb(pieces)]
        pieces &= pieces - ONE
    pieces = bb[base + BISHOP] | bb[base + QUEEN]
    while pieces:
        danger |= bishop_attacks(lsb(pieces), occupied)
        pieces &= pieces - ONE
    pieces = bb[base + ROOK] | bb[base + QUEEN]
    while pieces:
        danger |= rook_attacks(lsb(pieces), occupied)
        pieces &= pieces - ONE
    return danger | KING_ATT[lsb(bb[base + KING])]


@njit(cache=False)
def _pinned(bb, side, king, occupied):
    """Our pieces standing alone between our own king and an enemy slider."""
    them = (1 - side) * 6
    snipers = (ROOK_RAYS[king] & (bb[them + ROOK] | bb[them + QUEEN])) | (
        BISHOP_RAYS[king] & (bb[them + BISHOP] | bb[them + QUEEN])
    )
    pins = ZERO
    while snipers:
        sniper = lsb(snipers)
        snipers &= snipers - ONE
        blockers = BETWEEN[king * 64 + sniper] & occupied
        if blockers and (blockers & (blockers - ONE)) == ZERO:
            pins |= blockers & bb[12 + side]
    return pins


@njit(cache=False)
def in_check(bb, st):
    side = st[0]
    return attackers_to(bb, lsb(bb[side * 6 + KING]), bb[14], 1 - side) != ZERO


@njit(cache=False)
def gen_moves(bb, st, out, base):
    """Every legal move, written into out[base:]. Returns the new end index."""
    count, _checkers = _generate(bb, st, out, base, 0)
    return count


@njit(cache=False)
def gen_moves_ex(bb, st, out, base):
    """gen_moves, but also handing back the checkers bitboard it computed anyway.

    A search needs to know whether it is in check at nearly every node. Taking that
    from generation costs nothing and saves a whole attackers_to sweep per node.
    """
    return _generate(bb, st, out, base, 0)


@njit(cache=False)
def gen_captures(bb, st, out, base):
    """Only legal moves that land on an enemy piece, plus en passant."""
    count, _checkers = _generate(bb, st, out, base, 1)
    return count


@njit(cache=False)
def _generate(bb, st, out, base, only_caps):
    """Fully legal generation. Check evasions and pins are resolved with masks here,
    rather than by making each move and testing the king afterwards.

    `only_caps` narrows every target mask to the enemy's pieces. It is a plain runtime
    branch outside the bit loops, so the full-generation path pays nothing for it.
    """
    us = st[0]
    them = 1 - us
    mine = us * 6
    occupied = bb[14]
    own = bb[12 + us]
    king = lsb(bb[mine + KING])
    n = base
    wanted = bb[12 + them] if only_caps else FULL

    danger = _danger(bb, them, occupied & ~bit(king))
    targets = KING_ATT[king] & ~own & ~danger & wanted
    while targets:
        to = lsb(targets)
        targets &= targets - ONE
        out[n] = king | (to << 6)
        n += 1

    checkers = attackers_to(bb, king, occupied, them)
    if checkers:
        if checkers & (checkers - ONE):
            return n, checkers  # double check: nothing but the king can help
        checker = lsb(checkers)
        allowed = BETWEEN[king * 64 + checker] | bit(checker)
    elif only_caps:
        allowed = FULL  # castling is never a capture, so there is nothing else to add
    else:
        allowed = FULL
        rights = st[1]
        home = us * 56
        if us == WHITE:
            king_free = U64(0x60)
            king_safe = U64(0x70)
            queen_free = U64(0x0E)
            queen_safe = U64(0x1C)
        else:
            king_free = U64(0x6000000000000000)
            king_safe = U64(0x7000000000000000)
            queen_free = U64(0x0E00000000000000)
            queen_safe = U64(0x1C00000000000000)
        if (rights >> (us * 2)) & 1 and not (occupied & king_free) and not (danger & king_safe):
            out[n] = (home + 4) | ((home + 6) << 6) | (FLAG_CASTLE << 15)
            n += 1
        if (
            (rights >> (us * 2 + 1)) & 1
            and not (occupied & queen_free)
            and not (danger & queen_safe)
        ):
            out[n] = (home + 4) | ((home + 2) << 6) | (FLAG_CASTLE << 15)
            n += 1

    pins = _pinned(bb, us, king, occupied)
    quiet = ~own & allowed & wanted

    pieces = bb[mine + KNIGHT] & ~pins  # a pinned knight can never move at all
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - ONE
        targets = KNIGHT_ATT[frm] & quiet
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            out[n] = frm | (to << 6)
            n += 1

    pieces = bb[mine + BISHOP] | bb[mine + QUEEN]
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - ONE
        targets = bishop_attacks(frm, occupied) & quiet
        if bit(frm) & pins:
            targets &= LINE[king * 64 + frm]
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            out[n] = frm | (to << 6)
            n += 1

    pieces = bb[mine + ROOK] | bb[mine + QUEEN]
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - ONE
        targets = rook_attacks(frm, occupied) & quiet
        if bit(frm) & pins:
            targets &= LINE[king * 64 + frm]
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            out[n] = frm | (to << 6)
            n += 1

    return _gen_pawns(bb, st, out, n, us, king, pins, allowed, only_caps), checkers


@njit(cache=False)
def _gen_pawns(bb, st, out, n, us, king, pins, allowed, only_caps):
    them = 1 - us
    occupied = bb[14]
    empty = ~occupied
    enemy = bb[12 + them]
    up = PUSH_DIR[us]
    promo_rank = RANK_PROMO[us]
    pawns = bb[us * 6 + PAWN]

    # Unpinned pawns move set-wise, four shifts for the whole army. Pinned pawns are
    # rare enough to walk one at a time.
    free = pawns & ~pins
    single = _shift_up(free, us) & empty
    double = _shift_up(single & RANK_DOUBLE[us], us) & empty & allowed

    if only_caps:
        single = ZERO
        double = ZERO

    targets = single & allowed & ~promo_rank
    while targets:
        to = lsb(targets)
        targets &= targets - ONE
        out[n] = (to - up) | (to << 6)
        n += 1
    while double:
        to = lsb(double)
        double &= double - ONE
        out[n] = (to - up - up) | (to << 6) | (FLAG_DPUSH << 15)
        n += 1
    targets = single & allowed & promo_rank
    while targets:
        to = lsb(targets)
        targets &= targets - ONE
        frm = to - up
        for kind in range(QUEEN, PAWN, -1):
            out[n] = frm | (to << 6) | (kind << 12)
            n += 1

    if us == WHITE:
        west = ((free & NOT_FILE_A) << U64(7)) & enemy & allowed
        east = ((free & NOT_FILE_H) << U64(9)) & enemy & allowed
        west_delta = 7
        east_delta = 9
    else:
        west = ((free & NOT_FILE_A) >> U64(9)) & enemy & allowed
        east = ((free & NOT_FILE_H) >> U64(7)) & enemy & allowed
        west_delta = -9
        east_delta = -7
    for side in range(2):
        captures = west if side == 0 else east
        delta = west_delta if side == 0 else east_delta
        while captures:
            to = lsb(captures)
            captures &= captures - ONE
            frm = to - delta
            if bit(to) & promo_rank:
                for kind in range(QUEEN, PAWN, -1):
                    out[n] = frm | (to << 6) | (kind << 12)
                    n += 1
            else:
                out[n] = frm | (to << 6)
                n += 1

    stuck = pawns & pins
    while stuck:
        frm = lsb(stuck)
        stuck &= stuck - ONE
        ray = LINE[king * 64 + frm]
        to = frm + up
        if not only_caps and bit(to) & empty & ray:
            if bit(to) & allowed:
                if bit(to) & promo_rank:
                    for kind in range(QUEEN, PAWN, -1):
                        out[n] = frm | (to << 6) | (kind << 12)
                        n += 1
                else:
                    out[n] = frm | (to << 6)
                    n += 1
            if bit(frm) & RANK_HOME[us] and bit(to + up) & empty & ray & allowed:
                out[n] = frm | ((to + up) << 6) | (FLAG_DPUSH << 15)
                n += 1
        captures = PAWN_ATT[us * 64 + frm] & enemy & ray & allowed
        while captures:
            to = lsb(captures)
            captures &= captures - ONE
            if bit(to) & promo_rank:
                for kind in range(QUEEN, PAWN, -1):
                    out[n] = frm | (to << 6) | (kind << 12)
                    n += 1
            else:
                out[n] = frm | (to << 6)
                n += 1

    ep = st[2]
    if ep >= 0:
        captured = ep - up
        base = them * 6
        candidates = PAWN_ATT[them * 64 + ep] & pawns
        while candidates:
            frm = lsb(candidates)
            candidates &= candidates - ONE
            # En passant is the one move that empties two squares at once, so no mask
            # catches every illegal case. Test this one against the board directly.
            after = (occupied ^ bit(frm) ^ bit(captured)) | bit(ep)
            if rook_attacks(king, after) & (bb[base + ROOK] | bb[base + QUEEN]):
                continue
            if bishop_attacks(king, after) & (bb[base + BISHOP] | bb[base + QUEEN]):
                continue
            if KNIGHT_ATT[king] & bb[base + KNIGHT]:
                continue
            if PAWN_ATT[us * 64 + king] & bb[base + PAWN] & ~bit(captured):
                continue
            out[n] = frm | (ep << 6) | (FLAG_EP << 15)
            n += 1
    return n


@njit(cache=False)
def make_move(bb, sq, st, move, hist, ply):
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 3
    us = st[0]
    mine = us * 6
    piece = sq[frm]
    kind = piece - mine
    captured = sq[to]

    hist[ply, 0] = captured
    hist[ply, 1] = st[1]
    hist[ply, 2] = st[2]
    hist[ply, 3] = st[3]

    if captured >= 0:
        _lift(bb, sq, captured, to)
    _lift(bb, sq, piece, frm)
    if promo:
        _place(bb, sq, mine + promo, to)
    else:
        _place(bb, sq, piece, to)

    if flag == FLAG_EP:
        _lift(bb, sq, them_pawn(us), to - PUSH_DIR[us])
    elif flag == FLAG_CASTLE:
        if to > frm:
            _lift(bb, sq, mine + ROOK, frm + 3)
            _place(bb, sq, mine + ROOK, frm + 1)
        else:
            _lift(bb, sq, mine + ROOK, frm - 4)
            _place(bb, sq, mine + ROOK, frm - 1)

    st[1] &= CASTLE_MASK[frm] & CASTLE_MASK[to]
    st[2] = frm + PUSH_DIR[us] if flag == FLAG_DPUSH else -1
    st[3] = 0 if (kind == PAWN or captured >= 0) else st[3] + 1
    st[0] = 1 - us
    st[4] += us


@njit(inline="always", cache=False)
def them_pawn(us):
    return (1 - us) * 6 + PAWN


@njit(cache=False)
def unmake_move(bb, sq, st, move, hist, ply):
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 3
    us = 1 - st[0]
    mine = us * 6

    st[0] = us
    st[1] = hist[ply, 1]
    st[2] = hist[ply, 2]
    st[3] = hist[ply, 3]
    st[4] -= us

    if promo:
        _lift(bb, sq, mine + promo, to)
        _place(bb, sq, mine + PAWN, frm)
    else:
        piece = sq[to]
        _lift(bb, sq, piece, to)
        _place(bb, sq, piece, frm)

    captured = hist[ply, 0]
    if captured >= 0:
        _place(bb, sq, captured, to)

    if flag == FLAG_EP:
        _place(bb, sq, them_pawn(us), to - PUSH_DIR[us])
    elif flag == FLAG_CASTLE:
        if to > frm:
            _lift(bb, sq, mine + ROOK, frm + 1)
            _place(bb, sq, mine + ROOK, frm + 3)
        else:
            _lift(bb, sq, mine + ROOK, frm - 1)
            _place(bb, sq, mine + ROOK, frm - 4)


@njit(cache=False)
def perft(bb, sq, st, depth, buf, hist, ply):
    if depth == 1:
        return gen_moves(bb, st, buf[ply], 0)
    if depth == 0:
        return 1
    n = gen_moves(bb, st, buf[ply], 0)
    total = 0
    for i in range(n):
        move = buf[ply, i]
        make_move(bb, sq, st, move, hist, ply)
        total += perft(bb, sq, st, depth - 1, buf, hist, ply + 1)
        unmake_move(bb, sq, st, move, hist, ply)
    return total


@njit(cache=False)
def perft_nodes(bb, sq, st, depth, buf, hist, ply):
    """perft without the depth-1 shortcut, so nodes/sec is comparable to a search."""
    if depth == 0:
        return 1
    n = gen_moves(bb, st, buf[ply], 0)
    total = 0
    for i in range(n):
        move = buf[ply, i]
        make_move(bb, sq, st, move, hist, ply)
        total += perft_nodes(bb, sq, st, depth - 1, buf, hist, ply + 1)
        unmake_move(bb, sq, st, move, hist, ply)
    return total


# --- python-side helpers ------------------------------------------------------------

_KIND_OF = {"p": PAWN, "n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN, "k": KING}
_PROMO_CHAR = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

@njit(inline="always", cache=False)
def has_non_pawn_material(bb: np.ndarray, us: np.int32) -> bool:
    return (bb[us*6+KNIGHT] or bb[us*6+BISHOP] or bb[us*6+ROOK] or bb[us*6+QUEEN]) != 0


def new_buffers():
    """The scratch a search needs: one move list and one undo record per ply."""
    return (
        np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32),
        np.zeros((MAX_PLY, 4), dtype=np.int64),
    )


def from_fen(fen):
    bb = np.zeros(15, dtype=np.uint64) # The bit boards for the pieces, 0-5 white, 6-11 black, 12-13 white and black, 14 all pieces
    sq = np.full(64, -1, dtype=np.int8) # What piece is on this square?
    st = np.zeros(5, dtype=np.int64) # Other info about the board state like side to move, castling, en passant square, halfmove count and fullmove count

    parts = fen.split()
    board = parts[0]
    turn = parts[1] if len(parts) > 1 else "w"
    castling = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"

    square = 56
    for char in board:
        if char == "/":
            square -= 16
        elif char.isdigit():
            square += int(char)
        else:
            code = _KIND_OF[char.lower()] + (0 if char.isupper() else 6)
            mask = U64(1) << U64(square)
            bb[code] |= mask
            bb[12 + code // 6] |= mask
            bb[14] |= mask
            sq[square] = code
            square += 1

    st[0] = WHITE if turn == "w" else BLACK
    st[1] = (
        (1 if "K" in castling else 0)
        | (2 if "Q" in castling else 0)
        | (4 if "k" in castling else 0)
        | (8 if "q" in castling else 0)
    )
    st[2] = -1 if ep == "-" else (ord(ep[0]) - 97) + 8 * (int(ep[1]) - 1)
    st[3] = int(parts[4]) if len(parts) > 4 else 0
    st[4] = int(parts[5]) if len(parts) > 5 else 1
    return bb, sq, st


def to_uci(move):
    move = int(move)
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    text = f"{chr(97 + frm % 8)}{frm // 8 + 1}{chr(97 + to % 8)}{to // 8 + 1}"
    return text + _PROMO_CHAR[promo] if promo else text


def legal_ucis(bb, st):
    buf, _ = new_buffers()
    return [to_uci(buf[0, i]) for i in range(gen_moves(bb, st, buf[0], 0))]


def capture_ucis(bb, st):
    buf, _ = new_buffers()
    return [to_uci(buf[0, i]) for i in range(gen_captures(bb, st, buf[0], 0))]


def warm():
    """Compile what a search calls, with the argument types it will really see.

    Deliberately minimal. numba compiles lazily, so anything left out of here costs
    nothing until something calls it, and every function warmed here is spent from the
    60 second init budget an agent shares with loading its weights. Warming perft,
    gen_captures, in_check and queen_attacks as well cost 8 seconds for functions a
    search never touches.

    Types matter as much as coverage: a move read out of the buffer is an int32, so
    warming make_move with a Python int would compile an int64 version and leave the
    real one to compile on the clock.
    """
    bb, sq, st = from_fen(STARTING_FEN)
    buf, hist = new_buffers()
    gen_moves(bb, st, buf[0], 0)
    gen_moves_ex(bb, st, buf[0], 0)
    move = np.int32(buf[0, 0])
    make_move(bb, sq, st, move, hist, 0)
    unmake_move(bb, sq, st, move, hist, 0)


def warm_extras():
    """Compile the rest: perft, capture generation, the standalone attack queries.

    Call this from tests, or from an agent that actually uses gen_captures. It is not
    part of import because most agents never need any of it.
    """
    bb, sq, st = from_fen(STARTING_FEN)
    buf, hist = new_buffers()
    gen_captures(bb, st, buf[0], 0)
    in_check(bb, st)
    attackers_to(bb, 0, bb[14], BLACK)
    queen_attacks(0, bb[14])
    perft(bb, sq, st, 2, buf, hist, 0)
    perft_nodes(bb, sq, st, 2, buf, hist, 0)


warm()
