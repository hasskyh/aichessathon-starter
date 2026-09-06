"""Integer inference for the NNUE evaluation, jitted with numba.

The accumulator is the first layer's output: for each perspective, the bias plus
one row of w1 for every feature that is on. Because it is a plain sum, a move that
changes three features costs three row-adds instead of 768 dot products. That is
the entire reason NNUE is affordable at every node.

Weights are passed in as arguments, never read from module globals. numba freezes
globals at compile time, so a function warmed before the weights load would bake in
whatever array happened to exist at that moment.
"""

import numpy as np
from numba import njit

FEATURES = 768
HIDDEN = 256
PERSPECTIVES = 2
OUTPUTS = 32

# acc: int32[2, HIDDEN]   w1: int16[FEATURES, HIDDEN]   b1: int16[HIDDEN]
# idx: int16[2, n], row 0 white's view and row 1 black's, negatives ignored as padding
#
# Every 2D array here is declared C-contiguous (::1 on the last axis), not just
# "some strides" (: on both axes). agent.py already forces every one of these into
# a contiguous layout before it is ever passed in (np.array() copies on load, and
# slicing STACK's leading axis preserves contiguity of what is left) -- declaring
# that here, rather than leaving numba to assume nothing about it, is what lets
# LLVM auto-vectorize the row-wise add below into SIMD instead of walking it with
# runtime-computed strides. A caller that ever passes a non-contiguous view (e.g.
# a transpose) now fails loudly with a signature-mismatch TypeError instead of
# silently compiling a slower specialisation.
_REFRESH_SIG = "void(int32[:, ::1], int16[:, ::1], int16[::1], int16[:, ::1])"


@njit(_REFRESH_SIG, cache=False, fastmath=False)
def refresh(acc, w1, b1, idx):  # type: ignore[no-untyped-def]
    """Rebuild both accumulators from scratch: acc[side] = b1 + sum of w1 rows.

    Call this once at the root of a search, and never inside it. idx is what
    features.active() returns; any negative entry is padding and is skipped, so the
    same function works on the fixed-width rows that pack.py writes.
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

_UPDATE_SIG = "void(int32[:, ::1], int16[:, ::1], int16[:, ::1], int16[:, ::1])"

@njit(_UPDATE_SIG, cache=False, fastmath=False)
def update(acc, w1, off, on):  # type: ignore[no-untyped-def]
    """Fold one move into an existing accumulator: subtract off, add on.

    off and on are what features.deltas() returns. The two are a plain sum so the
    order does not matter here, though adding first keeps the intermediate values
    larger, which matters once acc is int16 rather than int32.
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

# Quantisation constants. These are provisional: export.py must scale the trained
# weights to match them, and if the two disagree the net evaluates nonsense without
# ever erroring.
ACT_MAX = 127  # clipped ReLU ceiling, so activations stay inside int8 range
HIDDEN_SHIFT = 6  # >> after the hidden layer, to bring products back near ACT_MAX

# acc: int32[2, HIDDEN]   stm: 0 white to move, 1 black
# w2: int8[2 * HIDDEN, n]   b2: int32[n]   w3: int8[n]   b3: int32
_FORWARD_SIG = "int32(int32[:, ::1], int64, int8[:, ::1], int32[::1], int8[::1], int32)"


@njit(_FORWARD_SIG, cache=False, fastmath=False)
def forward(acc, stm, w2, b2, w3, b3):  # type: ignore[no-untyped-def]
    """Score the position in the accumulator, from the mover's point of view.

    The two perspectives are concatenated with the mover's half first, which is how
    the net knows whose turn it is. Everything stays in integers: activations are
    clamped into int8 range so the int8 weights multiply into an int32 without
    overflowing.
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
    idx = np.zeros((PERSPECTIVES, 1), dtype=np.int16)
    refresh(acc, w1, b1, idx)
    off = np.zeros((PERSPECTIVES, 1), dtype=np.int16)
    on = np.zeros((PERSPECTIVES, 1), dtype=np.int16)
    update(acc, w1, off, on)
    w2 = np.zeros((PERSPECTIVES * HIDDEN, OUTPUTS), dtype=np.int8)
    b2 = np.zeros(OUTPUTS, dtype=np.int32)
    w3 = np.zeros(OUTPUTS, dtype=np.int8)
    forward(acc, 0, w2, b2, w3, 0)

_warm()
