"""Pack labelled positions from fetch_data.py into training arrays on disk.

    uv run python -m training.pack --in data/raw.jsonl

Writes data/train_idx.npy and data/train_target.npy directly through numpy memmaps,
so resident memory stays small and bounded regardless of dataset size -- this box
has under 2 GB genuinely free once VS Code's own WSL server is running, and an
11M-row dataset as one in-memory array (~1.4 GB) was enough on its own to crash the
WSL VM. Nothing here ever holds more than one position's data in ordinary memory;
the OS pages the memmapped arrays to disk as needed.

idx[i, 0] is the mover's own active features in that position, idx[i, 1] the
opponent's, both -1 padded to width 32 (the most pieces a board can hold). Lichess's
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


def process_position(fen: str, cp: float, idx: np.memmap, target: np.memmap, i: int) -> None:
    """Write one position into row i of idx and target.

    A fresh memmap is zero-filled, and 0 is a real feature (a friendly pawn on a1),
    not an empty slot -- so every row's padding must be set to -1 explicitly here,
    rather than relying on however the array happened to start out.
    """
    board = chess.Board(fen)
    white_row, black_row = features.active(board)
    mover, opponent = (
        (white_row, black_row) if board.turn == chess.WHITE else (black_row, white_row)
    )
    idx[i, 0, :] = -1
    idx[i, 1, :] = -1
    idx[i, 0, : len(mover)] = mover
    idx[i, 1, : len(opponent)] = opponent
    target[i] = 1.0 / (1.0 + np.exp(-cp / SIGMOID_SCALE))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack labelled FENs into training arrays.")
    parser.add_argument("--in", dest="input", type=Path, default=Path("data/raw.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    arguments = parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    with arguments.input.open(encoding="utf-8") as file:
        n = sum(1 for _ in file)
    print(f"{n:,} lines in {arguments.input}", file=sys.stderr)

    idx_path = arguments.out_dir / "train_idx.npy"
    target_path = arguments.out_dir / "train_target.npy"
    idx = np.lib.format.open_memmap(idx_path, mode="w+", dtype=np.int16, shape=(n, 2, MAX_FEATURES))
    target = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float32, shape=(n,))

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
                idx.flush()
                target.flush()
                print(f"  packed {kept:,}", file=sys.stderr)

    idx.flush()
    target.flush()
    print(
        f"packed {kept:,} positions ({skipped:,} skipped) of {n:,} preallocated rows "
        f"-> {idx_path}, {target_path}",
        file=sys.stderr,
    )
    if kept != n:
        print(
            f"warning: {n - kept:,} trailing rows are unused (all -1 / 0.0); "
            f"slice [:{kept}] when loading, or investigate why raw.jsonl had bad lines",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
