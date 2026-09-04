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
import hashlib
import json
import math
import sys
from pathlib import Path

import chess

CP_CLAMP = 2000


class BloomFilter:
    """Fixed-memory approximate membership test, in place of a plain set[str].

    A set[str] grows without bound across a run this large: at the 250M-position
    scale this project fetches at, that is hundreds of millions of live Python
    string objects, and it exhausted this machine's memory (3.7 GB RAM, 1 GB swap)
    well before the fetch could finish -- observed directly, swap usage climbed to
    75% within the first ~15-20M kept positions alone, and was still climbing. A
    Bloom filter's bit array is a fixed size chosen up front from the expected item
    count, so memory never grows no matter how long the run goes.

    The cost is a small, tunable false-positive rate: a brand-new position
    occasionally, wrongly, treated as already seen and skipped. That is free here --
    one lost row out of an eventual quarter-billion is not a correctness problem the
    way silently exhausting memory and crashing the whole fetch is.

    might_contain() and add() are separate, matching the set[str] they replace:
    the caller checks early and cheaply (a duplicate of an already-kept position
    should never pay for a board reconstruction it's just going to throw away), and
    only commits the add once a position survives every other filter -- exactly the
    dedup_key()-then-later-seen.add() split main() already had.
    """

    def __init__(self, expected_items: int, false_positive_rate: float = 0.01) -> None:
        self.size = max(
            8, int(-expected_items * math.log(false_positive_rate) / (math.log(2) ** 2))
        )
        self.hash_count = max(1, round((self.size / expected_items) * math.log(2)))
        self.bits = bytearray(self.size // 8 + 1)

    def _indexes(self, key: str) -> list[int]:
        # Double hashing: two independent-enough hashes from one digest simulate
        # hash_count hash functions without actually computing that many.
        digest = hashlib.blake2b(key.encode(), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little")
        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]

    def might_contain(self, key: str) -> bool:
        return all(self.bits[i // 8] & (1 << (i % 8)) for i in self._indexes(key))

    def add(self, key: str) -> None:
        for i in self._indexes(key):
            self.bits[i // 8] |= 1 << (i % 8)


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

    seen = BloomFilter(expected_items=arguments.limit)
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
            if seen.might_contain(key):
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
