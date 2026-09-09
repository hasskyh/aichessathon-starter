"""Filter a raw (fen, cp) jsonl file down to genuinely quiet positions only, using
agent.py's own SEE (static exchange evaluation) -- a position-intrinsic, net-
independent measure of whether either side has a real profitable capture on the
board (checked directly, and for the opponent after a hypothetical null move, to
catch "my own piece is hanging" as well as "I have a good capture available").

"Filter harshly" per instruction: keeps only positions where NEITHER side has any
profitable capture at all (SEE > 0 for anyone -> excluded), not just a lenient
material-swing threshold. In-check positions are also excluded (the mover is
forced to respond, not a free choice, so it isn't the kind of "quiet" position a
static evaluator should be trained to judge on its own).

Must run from the repo root. Pays agent.py's ~60s import-time JIT compile once.

    uv run python filter_quiet_positions.py data/raw.jsonl data/raw_quiet.jsonl
"""
import json
import sys

sys.path.insert(0, ".")
import bitgen
import agent


def is_quiet(fen: str) -> bool:
    bb, sq, st = bitgen.from_fen(fen)
    count, checkers = agent.gen_moves_ex(bb, st, agent.BUF[0], 0)
    if checkers or count == 0:
        return False

    own_captures = agent._keep_captures(sq, agent.BUF[0], count)
    for i in range(own_captures):
        if agent.see(bb, sq, agent.BUF[0, i]) > 0:
            return False

    agent.make_null_move(st, agent.HIST, 0)
    count2, checkers2 = agent.gen_moves_ex(bb, st, agent.BUF[1], 0)
    quiet = True
    if not checkers2 and count2 > 0:
        opp_captures = agent._keep_captures(sq, agent.BUF[1], count2)
        for i in range(opp_captures):
            if agent.see(bb, sq, agent.BUF[1, i]) > 0:
                quiet = False
                break
    agent.unmake_null_move(st, agent.HIST, 0)
    return quiet


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    progress_every = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000

    kept = 0
    total = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                data = json.loads(line)
                if is_quiet(data["fen"]):
                    fout.write(line + "\n")
                    kept += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if total % progress_every == 0:
                print(f"  {total:,} scanned, {kept:,} kept ({100*kept/total:.1f}%)",
                      file=sys.stderr)

    print(f"done: {kept:,} of {total:,} kept ({100*kept/total:.1f}%) -> {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
