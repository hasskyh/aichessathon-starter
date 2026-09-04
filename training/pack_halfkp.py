"""Pack labelled positions from fetch_data.py into HalfKP training arrays on disk.

    uv run python -m training.pack_halfkp --in data/raw_halfkp.jsonl

Structurally identical to pack.py -- same memmap-through-disk approach so resident
memory stays small regardless of dataset size, same mover-then-opponent row order,
same White-relative-to-mover-relative cp fix (see pack.py's module docstring for
how that was found and verified) -- with two differences forced by
features_halfkp.py's feature set:

  1. features_halfkp.halfkp_active() replaces features.active(): every row is now
     relative to that perspective's own king, not just its colour.
  2. idx is int32, not int16: HalfKP feature indices go up to 40,959, past int16's
     positive range (32,767). MAX_FEATURES is 30, not 32 -- kings are excluded from
     HalfKP entirely, so the most a row can hold is 32 squares minus the two kings.
"""

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

import features_halfkp as fk

SIGMOID_SCALE = 400.0
MAX_FEATURES = fk.MAX_PIECES  # 30: 32 squares minus the two kings, never features here


def process_position(fen: str, cp: float, idx: np.memmap, target: np.memmap, i: int) -> None:
    """Write one position into row i of idx and target.

    A fresh memmap is zero-filled, and 0 is a real feature (a friendly pawn's own
    king on a1, say), not an empty slot -- so every row's padding must be set to -1
    explicitly here, rather than relying on however the array happened to start out.
    """
    board = chess.Board(fen)
    white_row, black_row = fk.halfkp_active(board)
    mover, opponent = (
        (white_row, black_row) if board.turn == chess.WHITE else (black_row, white_row)
    )
    idx[i, 0, :] = -1
    idx[i, 1, :] = -1
    idx[i, 0, : len(mover)] = mover
    idx[i, 1, : len(opponent)] = opponent
    # cp is White-relative regardless of whose move it is; flip it to mover-relative
    # here so it lines up with idx's mover-then-opponent row order (see pack.py).
    mover_cp = cp if board.turn == chess.WHITE else -cp
    target[i] = 1.0 / (1.0 + np.exp(-mover_cp / SIGMOID_SCALE))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack labelled FENs into HalfKP training arrays.")
    parser.add_argument("--in", dest="input", type=Path, default=Path("data/raw_halfkp.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    arguments = parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    with arguments.input.open(encoding="utf-8") as file:
        n = sum(1 for _ in file)
    print(f"{n:,} lines in {arguments.input}", file=sys.stderr)

    idx_path = arguments.out_dir / "train_idx_halfkp.npy"
    target_path = arguments.out_dir / "train_target_halfkp.npy"
    idx = np.lib.format.open_memmap(idx_path, mode="w+", dtype=np.int32, shape=(n, 2, MAX_FEATURES))
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
            f"slice [:{kept}] when loading, or investigate why raw_halfkp.jsonl had bad lines",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
