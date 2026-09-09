"""Position-intrinsic (net-independent) tactical-loudness check: does either side
have a genuinely profitable capture on the board, using agent.py's own SEE
(static exchange evaluation), not any trained net's opinion?

Checks two things per position:
  1. The mover's own best available capture SEE value (a capture I can make now).
  2. The best SEE value available to the OPPONENT after a hypothetical null move
     (a capture they could make if I just passed) -- this is exactly how a
     "hanging piece" shows up: the mover has no good capture themselves, but one
     of their own pieces is a profitable target for the opponent.

A position is "loud" if either of those is a real material gain (> threshold, in
pawns-equivalent centipawns via agent.py's own PIECE_VALUE scale). This is fully
deterministic and independent of any trained NNUE -- it only uses move generation
and SEE, both pure board-mechanics -- so it isn't coupled to our current net's own
(possibly buggy) judgment the way comparing static-eval-vs-quiescence-eval was.

Must run from the repo root. Pays agent.py's ~60s import-time JIT compile once.
"""
import json
import random
import sys

import numpy as np

sys.path.insert(0, ".")
import bitgen
import agent


def loudness(fen: str) -> int | None:
    """Returns the largest profitable-capture SEE value available to either side
    (mover now, or opponent after a hypothetical pass), or None if the mover is
    already in check (forced to respond, not a "hanging piece" situation)."""
    bb, sq, st = bitgen.from_fen(fen)
    count, checkers = agent.gen_moves_ex(bb, st, agent.BUF[0], 0)
    if checkers or count == 0:
        return None

    own_captures = agent._keep_captures(sq, agent.BUF[0], count)
    own_best = max(
        (agent.see(bb, sq, agent.BUF[0, i]) for i in range(own_captures)), default=-10**9
    )

    agent.make_null_move(st, agent.HIST, 0)
    count2, checkers2 = agent.gen_moves_ex(bb, st, agent.BUF[1], 0)
    opp_best = -10**9
    if not checkers2 and count2 > 0:
        opp_captures = agent._keep_captures(sq, agent.BUF[1], count2)
        opp_best = max(
            (agent.see(bb, sq, agent.BUF[1, i]) for i in range(opp_captures)), default=-10**9
        )
    agent.unmake_null_move(st, agent.HIST, 0)

    return max(own_best, opp_best)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/binpack_labeled.jsonl"
    sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    random.seed(0)
    random.shuffle(lines)

    values = []
    checked = 0
    skipped_in_check = 0
    for line in lines:
        if checked >= sample_n:
            break
        line = line.strip()
        if not line:
            continue
        try:
            fen = json.loads(line)["fen"]
            result = loudness(fen)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        if result is None:
            skipped_in_check += 1
            continue
        values.append(result)
        checked += 1

    values = np.array(values)
    print(f"{path}: {checked:,} positions checked ({skipped_in_check:,} in-check skipped)")
    for threshold in (0, 50, 100, 200, 300):
        frac = (values > threshold).mean() * 100
        print(f"  a genuinely profitable capture (SEE > {threshold}) exists for someone: "
              f"{frac:.1f}% of positions")


if __name__ == "__main__":
    main()
