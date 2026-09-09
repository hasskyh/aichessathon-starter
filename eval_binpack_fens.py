"""Label a list of FENs (one per line, extracted from a third-party .binpack via
extract_binpack_fens.py) with our own local Stockfish, producing (fen, cp) rows in
the exact same format/convention as data/raw.jsonl. Keeping our own Stockfish +
sign convention here (not the binpack's own embedded score) means the only thing
that differs from raw.jsonl is the SOURCE of positions -- Leela/SF self-play
training games instead of Lichess's played-game evals -- so a difference in
trained behaviour on this set can't be explained by a labelling mismatch.

    uv run python eval_binpack_fens.py binpack_fens.txt out.jsonl /path/to/stockfish [n]
"""
import json
import subprocess
import sys


def sf_eval(proc: subprocess.Popen, fen: str, movetime_ms: int = 100) -> int | None:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(f"position fen {fen}\n")
    proc.stdin.write(f"go movetime {movetime_ms}\n")
    proc.stdin.flush()
    last_score_line = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        if line.startswith("info") and "score" in line:
            last_score_line = line
        if line.startswith("bestmove"):
            break
    if "score mate" in last_score_line:
        mate_in = int(last_score_line.split("score mate")[1].split()[0])
        return 3000 if mate_in > 0 else -3000
    if "score cp" in last_score_line:
        return int(last_score_line.split("score cp")[1].split()[0])
    return None


def main() -> None:
    in_path, out_path, sf_path = sys.argv[1], sys.argv[2], sys.argv[3]
    n_wanted = int(sys.argv[4]) if len(sys.argv) > 4 else None

    proc = subprocess.Popen(
        [sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if "uciok" in line:
            break

    written = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for fen in fin:
            fen = fen.strip()
            if not fen:
                continue
            if n_wanted is not None and written >= n_wanted:
                break
            # sf_eval is side-to-move-relative; flip to White-relative to match
            # raw.jsonl's own convention (same fix already applied in
            # gen_synthetic_sacrifice.py / gen_queen_confound.py).
            side_to_move_cp = sf_eval(proc, fen)
            if side_to_move_cp is None:
                continue
            is_white_to_move = fen.split(" ")[1] == "w"
            cp = side_to_move_cp if is_white_to_move else -side_to_move_cp
            fout.write(json.dumps({"fen": fen, "cp": cp}) + "\n")
            written += 1
            if written % 5000 == 0:
                print(f"  {written:,} written", file=sys.stderr)

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=5)
    print(f"done: wrote {written:,} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
