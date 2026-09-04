"""Filter Lichess's Stockfish-annotated positions into a plain (fen, cp) stream.

    curl -sS https://database.lichess.org/lichess_db_eval.jsonl.zst \
      | zstd -dc \
      | uv run python -m training.fetch_data --out data/raw.jsonl --limit 2000000000 \
            --state data/fetch_state.json --bloom-state data/fetch_bloom.bin

Reads newline-delimited JSON from stdin, one line at a time, so the multi-gigabyte
source never sits in memory. Each input line looks like:

    {"fen": "...", "evals": [{"pvs": [{"cp": 30, "line": "e2e4 e7e5"}],
                               "knodes": 2206942, "depth": 38}, ...]}

or with {"mate": N} in place of "cp" for a forced mate.

Positions in check, or whose best move is a capture, are dropped: those are exactly
what quiescence search is for, and training on them teaches the net to duplicate
work the search already does. Positions are deduplicated on the FEN's first four
fields (board, side to move, castling, en passant), ignoring the halfmove and
fullmove counters, which do not change what a position means.

Resumable, for a fetch large enough to span an unattended run that outlives one
sitting: --state and --bloom-state persist enough to pick back up cleanly after
this process is killed for any reason (a machine restart chief among them, on this
project's hardware). The design this settled on, and why:

  - zstd's compressed format cannot be resumed from an arbitrary mid-file byte
    offset the way the plain-text output can -- its internal block/frame structure
    doesn't generally align with an arbitrary cut point, so re-issuing curl with a
    byte range starting partway into the *compressed* file does not reliably
    decompress on its own. Resuming therefore always re-downloads and
    re-decompresses from the start of the source file, not from a saved byte
    offset into it.
  - What IS saved is how many *lines* of decompressed input have already been
    read (state["lines_read"]), so a resumed run skips that many lines cheaply
    (no json.loads, no board reconstruction) before processing anything new. This
    project's fetches are CPU-bound on the per-line filtering, not on download
    bandwidth (confirmed directly: curl and zstd sit near-idle while this script's
    own process saturates a core), so re-transferring and skipping already-seen
    lines costs a few minutes, not the hours a full redo from position zero would.
  - --append opens --out in "a" mode so previously kept rows are never discarded.
  - --bloom-state persists the dedup Bloom filter's bit array itself (not just a
    count), so a position seen in an earlier segment of the run is still correctly
    recognised as a duplicate after a restart -- without this, resuming would
    silently lose all deduplication memory and let cross-restart duplicates
    through.
  - state["kept_total"] make --limit mean the cumulative total across every
    segment of the run, not just the current process's own count, so it stops in
    the right place regardless of how many times it has been restarted.

Both state files are updated every --progress-every rows, not only at a clean
exit, since the whole point is surviving an unclean one.
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

    A set[str] grows without bound across a run this large: at the scale this
    project fetches at, that is hundreds of millions of live Python string
    objects, and it exhausted this machine's memory (3.7 GB RAM, 1 GB swap) well
    before a 250M-position fetch could finish -- observed directly, swap usage
    climbed to 75% within the first ~15-20M kept positions alone, and was still
    climbing. A Bloom filter's bit array is a fixed size chosen up front from the
    expected item count, so memory never grows no matter how long the run goes.

    The cost is a small, tunable false-positive rate: a brand-new position
    occasionally, wrongly, treated as already seen and skipped. That is free here --
    a handful of lost rows out of a dataset this size is not a correctness problem
    the way silently exhausting memory and crashing the whole fetch is.

    might_contain() and add() are separate, matching the set[str] they replace:
    the caller checks early and cheaply (a duplicate of an already-kept position
    should never pay for a board reconstruction it's just going to throw away), and
    only commits the add once a position survives every other filter -- exactly the
    dedup_key()-then-later-seen.add() split main() already had.

    save()/load() persist and restore the bit array verbatim, so resuming a fetch
    does not lose deduplication memory from segments already processed.
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

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(bytes(self.bits))
        tmp.replace(path)  # atomic on the same filesystem: never a half-written state file

    @classmethod
    def load(
        cls, path: Path, expected_items: int, false_positive_rate: float = 0.01
    ) -> "BloomFilter":
        bf = cls(expected_items, false_positive_rate)
        data = path.read_bytes()
        if len(data) != len(bf.bits):
            raise ValueError(
                f"bloom state at {path} has {len(data)} bytes, expected {len(bf.bits)} -- "
                "was it created with a different --limit?"
            )
        bf.bits = bytearray(data)
        return bf


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


def load_state(path: Path | None) -> dict:
    if path is not None and path.exists():
        return json.loads(path.read_text())
    return {"lines_read": 0, "kept_total": 0}


def save_state(path: Path | None, state: dict) -> None:
    if path is None:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Lichess eval JSONL into (fen, cp) lines.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, default=5_000_000, help="cumulative positions to keep"
    )
    parser.add_argument(
        "--append", action="store_true", help="resume: append instead of overwrite"
    )
    parser.add_argument(
        "--state", type=Path, default=None, help="JSON file: lines_read, kept_total"
    )
    parser.add_argument(
        "--bloom-state", type=Path, default=None, help="persisted dedup filter bits"
    )
    parser.add_argument("--progress-every", type=int, default=500_000)
    arguments = parser.parse_args()

    state = load_state(arguments.state)
    lines_to_skip = state["lines_read"]
    kept_total = state["kept_total"]

    if arguments.bloom_state is not None and arguments.bloom_state.exists():
        seen = BloomFilter.load(arguments.bloom_state, expected_items=arguments.limit)
    else:
        seen = BloomFilter(expected_items=arguments.limit)

    kept = read = malformed = duplicate = in_check = capture = 0
    mode = "a" if arguments.append else "w"

    with arguments.out.open(mode, encoding="utf-8") as out:
        for raw_line in sys.stdin:
            read += 1
            if read <= lines_to_skip:
                continue

            if read % arguments.progress_every == 0:
                print(
                    f"  read {read:,}  kept {kept:,} this run "
                    f"(total {kept_total + kept:,})",
                    file=sys.stderr,
                )
                state["lines_read"] = read
                state["kept_total"] = kept_total + kept
                save_state(arguments.state, state)
                seen.save(arguments.bloom_state) if arguments.bloom_state else None

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
            if kept_total + kept >= arguments.limit:
                break

    state["lines_read"] = read
    state["kept_total"] = kept_total + kept
    save_state(arguments.state, state)
    if arguments.bloom_state is not None:
        seen.save(arguments.bloom_state)

    print(
        f"read {read:,} ({lines_to_skip:,} skipped as already processed)  "
        f"kept {kept:,} this run (total {kept_total + kept:,})  "
        f"dropped: malformed {malformed:,}, duplicate {duplicate:,}, "
        f"in_check {in_check:,}, capture {capture:,}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
