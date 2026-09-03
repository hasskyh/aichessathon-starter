"""Filter Lichess's Stockfish-annotated positions into a plain (fen, cp) stream.

    curl -r 0-10000000000 https://database.lichess.org/lichess_db_eval.jsonl.zst \
      | zstd -dc \
      | uv run python training/fetch_data.py --out data/raw.jsonl --limit 20000000

Reads newline-delimited JSON from stdin, one line at a time, so the multi-gigabyte
source never sits in memory. Each input line looks like:

    {"fen": "...", "evals": [{"pvs": [{"cp": 30, "line": "e2e4 e7e5"}],
                               "knodes": 2206942, "depth": 38}, ...]}

or with {"mate": N} in place of "cp" for a forced mate.

Positions in check, or whose best move is a capture, are dropped: those are exactly
what quiescence search is for, and training on them teaches the net to duplicate
work the search already does. Positions are deduplicated on the FEN's first four
fields (board, side to move, castling, en passant), ignoring the halfmove and
fullmove counters, which do not change what a position means. A truncated final
line at the byte-range cutoff is expected and skipped, not an error.

zstd exits non-zero when curl's byte range cuts it off mid-stream. That is
expected -- the range is what bounds the download -- and does not affect the
lines already written to --out.
"""

import argparse
import json
import sys
from pathlib import Path

import chess

CP_CLAMP = 2000


def best_move_uci(position: dict) -> str | None:
    evals = position.get("evals")
    if not evals:
        return None
    deepest = max(evals, key=lambda e: (e.get("depth", 0), e.get("knodes", 0)))
    pvs = deepest.get("pvs")
    if not pvs:
        return None
    line = pvs[0].get("line")
    if not line:
        return None
    return line.split()[0]


def label_cp(position: dict) -> int | None:
    evals = position.get("evals")
    if not evals:
        return None
    deepest = max(evals, key=lambda e: (e.get("depth", 0), e.get("knodes", 0)))
    pvs = deepest.get("pvs")
    if not pvs:
        return None
    pv = pvs[0]
    if "mate" in pv:
        return CP_CLAMP if pv["mate"] > 0 else -CP_CLAMP
    if "cp" in pv:
        return max(-CP_CLAMP, min(CP_CLAMP, pv["cp"]))
    return None


def dedup_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Lichess eval JSONL into (fen, cp) lines.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5_000_000, help="positions to keep")
    parser.add_argument("--progress-every", type=int, default=500_000)
    arguments = parser.parse_args()

    seen: set[str] = set()
    kept = read = malformed = duplicate = in_check = capture = 0

    with arguments.out.open("w", encoding="utf-8") as out:
        for raw_line in sys.stdin:
            read += 1
            if read % arguments.progress_every == 0:
                print(f"  read {read:,}  kept {kept:,}", file=sys.stderr)

            try:
                position = json.loads(raw_line)
                fen = position["fen"]
                uci = best_move_uci(position)
                cp = label_cp(position)
            except (json.JSONDecodeError, KeyError, ValueError, IndexError):
                malformed += 1
                continue
            if uci is None or cp is None:
                malformed += 1
                continue

            key = dedup_key(fen)
            if key in seen:
                duplicate += 1
                continue

            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(uci)
            except (ValueError, IndexError):
                malformed += 1
                continue

            if board.is_check():
                in_check += 1
                continue
            if board.is_capture(move):
                capture += 1
                continue

            seen.add(key)
            out.write(json.dumps({"fen": fen, "cp": cp}) + "\n")
            kept += 1
            if kept >= arguments.limit:
                break

    print(
        f"read {read:,}  kept {kept:,}  "
        f"dropped: malformed {malformed:,}, duplicate {duplicate:,}, "
        f"in_check {in_check:,}, capture {capture:,}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
