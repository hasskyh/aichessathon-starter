"""Integer inference for the HalfKP NNUE evaluation, jitted with numba.

Structurally identical to nnue.py -- same accumulator-then-clipped-ReLU-then-two-
layer-MLP shape, same quantisation scheme -- with two differences forced by
features_halfkp.py's feature set:

  1. FEATURES is 40,960, not 768, so w1 is a much bigger table (40,960 * HIDDEN),
     but each row is still the same small int16 weight vector nnue.py already uses.
  2. Feature *indices* are int32 here, not int16: 40,960 exceeds int16's positive
     range (32,767), even though 768 fits easily. refresh()/update() take int32
     idx/off/on arrays for exactly this reason -- everything else about them,
     including the -1 padding convention, is unchanged from nnue.py.

There is no separate "king moved" case inside refresh() or update() themselves --
that decision belongs to the caller. A king move invalidates an entire
perspective's features (see features_halfkp.py's module docstring), so the caller
must call refresh() for that side instead of update(), and never call update() with
deltas computed across a king move. refresh() and update() here have no way to
detect that misuse; they just do what they are told for whichever side they are
given, exactly like nnue.py's do.
"""

import numpy as np
from numba import njit

FEATURES = 40_960
HIDDEN = 256
PERSPECTIVES = 2
OUTPUTS = 32

# acc: int32[2, HIDDEN]   w1: int16[FEATURES, HIDDEN]   b1: int16[HIDDEN]
# idx: int32[2, n], row 0 white's view and row 1 black's, negatives ignored as padding
_REFRESH_SIG = "void(int32[:, ::1], int16[:, ::1], int16[::1], int32[:, ::1])"


@njit(_REFRESH_SIG, cache=False, fastmath=False)
def refresh(acc, w1, b1, idx):  # type: ignore[no-untyped-def]
    """Rebuild both accumulators from scratch: acc[side] = b1 + sum of w1 rows.

    Call this once at the root of a search, and again for whichever single
    perspective's own king just moved -- never for both sides on a non-king move,
    and never skip it for the side whose king did move.
    """
    for side in range(acc.shape[0]):
        for h in range(acc.shape[1]):
            acc[side, h] = b1[h]
        for k in range(idx.shape[1]):
            feature = idx[side, k]
            if feature < 0:
                continue
            for h in range(acc.shape[1]):
                acc[side, h] += w1[feature, h]


_UPDATE_SIG = "void(int32[:, ::1], int16[:, ::1], int32[:, ::1], int32[:, ::1])"


@njit(_UPDATE_SIG, cache=False, fastmath=False)
def update(acc, w1, off, on):  # type: ignore[no-untyped-def]
    """Fold one non-king move into an existing accumulator: subtract off, add on.

    off and on are what features_halfkp.halfkp_deltas_bb returns for a move whose
    mover is not a king. The two are a plain sum so the order does not matter here,
    though adding first keeps the intermediate values larger, which matters once
    acc is int16 rather than int32.
    """
    for side in range(acc.shape[0]):
        for k in range(off.shape[1]):
            feature = off[side, k]
            if feature < 0:
                continue
            for h in range(acc.shape[1]):
                acc[side, h] -= w1[feature, h]
        for k in range(on.shape[1]):
            feature = on[side, k]
            if feature < 0:
                continue
            for h in range(acc.shape[1]):
                acc[side, h] += w1[feature, h]


# Quantisation constants, identical in meaning to nnue.py's -- export.py must scale
# the trained weights to match, and if the two disagree the net evaluates nonsense
# without ever erroring.
ACT_MAX = 127  # clipped ReLU ceiling, so activations stay inside int8 range
HIDDEN_SHIFT = 6  # >> after the hidden layer, to bring products back near ACT_MAX

# acc: int32[2, HIDDEN]   stm: 0 white to move, 1 black
# w2: int8[2 * HIDDEN, n]   b2: int32[n]   w3: int8[n]   b3: int32
_FORWARD_SIG = "int32(int32[:, ::1], int64, int8[:, ::1], int32[::1], int8[::1], int32)"


@njit(_FORWARD_SIG, cache=False, fastmath=False)
def forward(acc, stm, w2, b2, w3, b3):  # type: ignore[no-untyped-def]
    """Score the position in the accumulator, from the mover's point of view.

    Byte-for-byte the same computation as nnue.forward -- the feature set changes
    what is summed into the accumulator, never how the accumulator is turned into
    a score.
    """
    half = acc.shape[1]
    other = 1 - stm

    hidden = np.empty(w3.shape[0], dtype=np.int32)
    for j in range(w3.shape[0]):
        hidden[j] = b2[j]

    for i in range(2 * half):
        raw = acc[stm, i] if i < half else acc[other, i - half]
        if raw <= 0:
            continue  # clipped ReLU zeroes these, so skip the whole row
        activation = ACT_MAX if raw > ACT_MAX else raw
        for j in range(w3.shape[0]):
            hidden[j] += activation * w2[i, j]

    total = b3
    for j in range(w3.shape[0]):
        value = hidden[j] >> HIDDEN_SHIFT
        if value < 0:
            value = 0
        elif value > ACT_MAX:
            value = ACT_MAX
        total += value * w3[j]
    return total


def _warm() -> None:
    """Pay numba's compilation cost at import, inside the 60 second init budget."""
    acc = np.zeros((PERSPECTIVES, HIDDEN), dtype=np.int32)
    w1 = np.zeros((FEATURES, HIDDEN), dtype=np.int16)
    b1 = np.zeros(HIDDEN, dtype=np.int16)
    idx = np.zeros((PERSPECTIVES, 1), dtype=np.int32)
    refresh(acc, w1, b1, idx)
    off = np.zeros((PERSPECTIVES, 1), dtype=np.int32)
    on = np.zeros((PERSPECTIVES, 1), dtype=np.int32)
    update(acc, w1, off, on)
    w2 = np.zeros((PERSPECTIVES * HIDDEN, OUTPUTS), dtype=np.int8)
    b2 = np.zeros(OUTPUTS, dtype=np.int32)
    w3 = np.zeros(OUTPUTS, dtype=np.int8)
    forward(acc, 0, w2, b2, w3, 0)


_warm()
