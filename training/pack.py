"""Pack labelled positions from fetch_data.py into one training array.

    uv run python -m training.pack --in data/raw.jsonl --out data/train.npz

Parsing a FEN and walking its piece map is far slower than anything train.py does
with the result, so that work happens once here rather than once per epoch.

idx[i, 0] is the mover's own active features in that position, idx[i, 1] the
opponent's, both -1 padded to width 32 (the most pieces a board can hold). Lichew's
cp is already relative to the side to move, and nnue.forward concatenates the
mover's accumulator row first, so storing rows in mover-then-opponent order here
means train.py never has to know whose turn a position was.
"""

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

import features

SIGMOID_SCALE = 400.0
MAX_FEATURES = 32  # one board can hold at most 32 pieces


def process_position(fen: str, cp: float, idx: np.ndarray, target: np.ndarray, i: int) -> None:
    """Write one position into row i of idx and target.

    idx is pre-filled with -1, so writing only the first len(row) columns of each
    perspective leaves the rest correctly padded.
    """
    board = chess.Board(fen)
    white_row, black_row = features.active(board)
    mover, opponent = (
        (white_row, black_row) if board.turn == chess.WHITE else (black_row, white_row)
    )
    idx[i, 0, : len(mover)] = mover
    idx[i, 1, : len(opponent)] = opponent
    target[i] = 1.0 / (1.0 + np.exp(-cp / SIGMOID_SCALE))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack labelled FENs into training arrays.")
    parser.add_argument("--in", dest="input", type=Path, default=Path("data/raw.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/train.npz"))
    parser.add_argument("--progress-every", type=int, default=100_000)
    arguments = parser.parse_args()

    with arguments.input.open(encoding="utf-8") as file:
        n = sum(1 for _ in file)
    print(f"{n:,} lines in {arguments.input}", file=sys.stderr)

    idx = np.full((n, 2, MAX_FEATURES), -1, dtype=np.int16)
    target = np.zeros(n, dtype=np.float32)

    kept = 0
    skipped = 0
    with arguments.input.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                process_position(data["fen"], data["cp"], idx, target, kept)
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                print(f"skipping line {line_number}: {error}", file=sys.stderr)
                skipped += 1
                continue
            kept += 1
            if kept % arguments.progress_every == 0:
                print(f"  packed {kept:,}", file=sys.stderr)

    idx = idx[:kept]
    target = target[:kept]
    np.savez(arguments.out, idx=idx, target=target)
    print(f"packed {kept:,} positions ({skipped:,} skipped) -> {arguments.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
