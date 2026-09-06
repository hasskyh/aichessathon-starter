"""Zobrist hashing for bitgen's board representation, for the transposition table.

The idea: assign one random 64-bit number to every independent fact a position can
have ("a white knight sits on c3", "it's Black's turn", "White can still castle
kingside", "there's an en-passant capture on the e-file"), and XOR together the
numbers for every fact that's currently true. XOR is its own inverse, so toggling
one fact on or off is one XOR, in either direction -- which is exactly what makes
this maintainable incrementally at every make/unmake instead of recomputed from
scratch at every node. Two different positions collide only by chance (astronomically
unlikely with 64-bit numbers), and two occurrences of the *same* position, reached by
different move orders, always hash identically -- that second property is the entire
reason a transposition table works.

Four independent fact families, each with its own random table:
  ZOBRIST_PIECE[code, square]  -- one of the 12 piece codes sits on one of 64 squares
  ZOBRIST_SIDE                 -- a single number, XORed in whenever it's Black to move
  ZOBRIST_CASTLE[right]        -- one of the 4 castling rights (see bitgen.CASTLE_MASK's
                                   bit order: 0=white kingside, 1=white queenside,
                                   2=black kingside, 3=black queenside) is still held
  ZOBRIST_EP[file]             -- an en-passant capture is possible on this file

Seeded fixed, not random-per-process: two different processes (or two runs of a
test) must agree on what a given position hashes to, or nothing that persists or
gets compared across them (including this module's own tests) would be meaningful.

zobrist_hash_bb() builds a hash from scratch -- call it once, at the root of a
search, exactly like nnue.refresh(). zobrist_delta_bb() returns the XOR delta for
one move -- call it before bitgen.make_move(), exactly like features.deltas_bb(),
and XOR it into the running hash; XOR the same delta in again to undo it after
bitgen.unmake_move(), since XOR is its own inverse.
"""

import numpy as np
from numba import njit

import bitgen

_RNG = np.random.default_rng(0xC0FFEE)
ZOBRIST_PIECE = _RNG.integers(0, 2**64, size=(12, 64), dtype=np.uint64)
ZOBRIST_SIDE = np.uint64(_RNG.integers(0, 2**64, dtype=np.uint64))
ZOBRIST_CASTLE = _RNG.integers(0, 2**64, size=4, dtype=np.uint64)
ZOBRIST_EP = _RNG.integers(0, 2**64, size=8, dtype=np.uint64)


@njit(cache=False)
def zobrist_hash_bb(sq: np.ndarray, st: np.ndarray) -> np.uint64:
    """Hash the position from scratch. Call once, at the root of a search."""
    h = np.uint64(0)
    for square in range(64):
        code = sq[square]
        if code >= 0:
            h ^= ZOBRIST_PIECE[code, square]
    if st[0] == bitgen.BLACK:
        h ^= ZOBRIST_SIDE
    rights = st[1]
    for right in range(4):
        if rights & (1 << right):
            h ^= ZOBRIST_CASTLE[right]
    if st[2] >= 0:
        h ^= ZOBRIST_EP[st[2] & 7]
    return h


@njit(cache=False)
def zobrist_delta_bb(sq: np.ndarray, st: np.ndarray, move: np.int32) -> np.uint64:
    """The XOR delta for one move, computed from the PRE-move sq/st.

    Call this before bitgen.make_move(move) -- like features.deltas_bb(), it reads
    the piece being moved and the piece being captured off of sq, and both are
    stale the instant make_move actually runs. XOR the result into the running
    hash after make_move; XOR it in again after unmake_move to restore the parent's
    hash exactly (XOR is its own inverse, so no separate "undo" formula is needed).
    """
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 3
    us = st[0]
    mine = us * 6

    h = ZOBRIST_SIDE  # side to move always flips, every move, unconditionally

    mover_code = sq[frm]
    h ^= ZOBRIST_PIECE[mover_code, frm]  # mover leaves frm
    arriving_code = mine + promo if promo > 0 else mover_code
    h ^= ZOBRIST_PIECE[arriving_code, to]  # mover (or its promotion) arrives at to

    if flag == bitgen.FLAG_EP:
        captured_sq = to - bitgen.PUSH_DIR[us]
        h ^= ZOBRIST_PIECE[sq[captured_sq], captured_sq]
    else:
        captured_code = sq[to]
        if captured_code >= 0:
            h ^= ZOBRIST_PIECE[captured_code, to]  # ordinary capture, removed from to

    if flag == bitgen.FLAG_CASTLE:
        rook_code = mine + bitgen.ROOK
        if to > frm:  # kingside: h-file rook to the f-file
            rook_from, rook_to = frm + 3, frm + 1
        else:  # queenside: a-file rook to the d-file
            rook_from, rook_to = frm - 4, frm - 1
        h ^= ZOBRIST_PIECE[rook_code, rook_from]
        h ^= ZOBRIST_PIECE[rook_code, rook_to]

    # Castling rights only ever get taken away (bitgen.CASTLE_MASK clears bits via
    # AND, never sets one), so old_rights & ~new_rights is exactly the bits this
    # move clears -- toggle off just those, not the ones already gone before it.
    old_rights = st[1]
    new_rights = old_rights & bitgen.CASTLE_MASK[frm] & bitgen.CASTLE_MASK[to]
    cleared = old_rights & ~new_rights
    for right in range(4):
        if cleared & (1 << right):
            h ^= ZOBRIST_CASTLE[right]

    # En passant is set-or-cleared outright each move (never partially), so this is
    # a plain "toggle off the old one if any, toggle on the new one if any" rather
    # than needing a diff the way castling rights do.
    if st[2] >= 0:
        h ^= ZOBRIST_EP[st[2] & 7]
    if flag == bitgen.FLAG_DPUSH:
        new_ep = frm + bitgen.PUSH_DIR[us]
        h ^= ZOBRIST_EP[new_ep & 7]

    return h
