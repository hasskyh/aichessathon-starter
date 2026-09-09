"""Run OUR OWN quiescence search (agent.py's real, jitted quiescence()) on sample
positions from a training data file, and compare its result to the RAW static
NNUE eval at the same position (what quiescence starts from before exploring any
captures). A big gap between the two means the position was NOT quiescence-quiet
to begin with -- there were real tactics on the board that only resolve once you
search captures out, exactly the situation real NNUE training pipelines are
supposed to avoid by only labelling quiescence-settled leaf positions.

QUIESCE_DEPTH is 6 plies (agent.py:66) -- generous enough to resolve a simple
hanging piece (1 ply), so a large swing here reflects real tactical noise in the
position, not an artificially shallow search cutting the comparison short.

Must run from the repo root (agent.py's own weights/module lookups are relative).
Pays agent.py's ~60s import-time JIT compile once.
"""
import json
import random
import sys

import numpy as np

sys.path.insert(0, ".")
import bitgen
import agent

QUIESCE_DEPTH = agent.QUIESCE_DEPTH


def eval_position(fen: str) -> tuple[int, int] | None:
    """Returns (raw_static_eval, post_quiescence_eval), both side-to-move relative,
    or None if the position is in check (quiescence skips the stand-pat static eval
    entirely when in check, so there's no single static baseline to compare against)."""
    bb, sq, st = bitgen.from_fen(fen)
    agent.root_refresh(agent.STACK, agent.KING_SQ, sq, agent.W1, agent.B1)
    count, checkers = agent.gen_moves_ex(bb, st, agent.BUF[0], 0)
    if checkers or count == 0:
        return None
    static = agent.nnue_evaluate(agent.STACK, 0, st, agent.W2, agent.B2, agent.W3, agent.B3)
    agent.CTRL[2] = 0.0
    post_q = agent.quiescence(
        bb, sq, st, -agent.INF, agent.INF, QUIESCE_DEPTH, 0,
        agent.BUF, agent.SCORES, agent.HIST, agent.CTRL, agent.STACK, agent.KING_SQ,
        agent.W1, agent.B1, agent.W2, agent.B2, agent.W3, agent.B3,
        agent.HISTORY, agent.KILLERS,
    )
    return int(static), int(post_q)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/binpack_labeled.jsonl"
    sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    random.seed(0)
    random.shuffle(lines)

    swings = []
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
            result = eval_position(fen)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        if result is None:
            skipped_in_check += 1
            continue
        static, post_q = result
        swings.append(abs(post_q - static))
        checked += 1

    swings = np.array(swings)
    print(f"{path}: {checked:,} positions checked ({skipped_in_check:,} in-check skipped)")
    print(f"  mean |quiescence swing|: {swings.mean():.1f} (internal eval units)")
    print(f"  median: {np.median(swings):.1f}")
    for threshold in (0, 50, 150, 300, 600):
        frac = (swings > threshold).mean() * 100
        print(f"  swing > {threshold}: {frac:.1f}% of positions")


if __name__ == "__main__":
    main()
