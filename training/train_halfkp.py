"""Train the HalfKP NNUE on data/train_idx_halfkp.npy and train_target_halfkp.npy.

    uv run python -m training.train_halfkp --epochs 10

Sanity-check on a small slice before trusting a full run:
    uv run python -m training.train_halfkp --rows 50000 --epochs 1

Structurally identical to train.py -- same accumulator-then-clipped-ReLU-then-MLP
architecture, same ACCUMULATOR_NORM fix, same real bias1 term, same WDL sigmoid
target and MSE loss -- with FEATURES/HIDDEN taken from nnue_halfkp instead of
nnue, and padding_idx (and therefore remap()'s sentinel) at nnue_halfkp.FEATURES
=40,960 instead of 768. See train.py's module docstring for why ACCUMULATOR_NORM
and bias1 exist at all; nothing about either changes with the feature set.

Data loading reads in large sequential BLOCKS, not train.py's global random
permutation -- this dataset is 75M rows (train_idx_halfkp.npy alone is 18 GB),
far past what fits in this machine's 3.7 GB of RAM, unlike the flat net's 11M/
1.4 GB set which sat entirely in page cache and made a global shuffle cheap.
idx[np.random.permutation(n)] fancy-indexes the memmap at row positions
scattered across the whole 18 GB file with zero locality -- confirmed directly
to drive this machine into a severe, near-unresponsive swap crisis within a
couple of minutes on the very first attempt (free -h and even bare `kill -9`
took 20-30 seconds to return). Reading one large contiguous block at a time
(np.array(idx[start:stop]), a plain slice, not fancy indexing) is effectively
one sequential disk read per block instead of thousands of scattered ones;
shuffling happens in memory, within each loaded block and across the order
blocks are visited in, which is real per-epoch randomization without ever
touching the whole file's span at once.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from nnue_halfkp import FEATURES, HIDDEN

OUTPUTS = 32
ACCUMULATOR_NORM = 16.0
BLOCK_SIZE = 1_000_000  # rows per sequential read; ~120 MB of idx data at a time


class NNUE(nn.Module):
    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.transformer = nn.EmbeddingBag(FEATURES + 1, hidden, mode="sum", padding_idx=FEATURES)
        self.bias1 = nn.Parameter(torch.zeros(hidden))
        self.hidden = nn.Linear(2 * hidden, OUTPUTS)
        self.output = nn.Linear(OUTPUTS, 1)

    def forward(self, mover_idx: Tensor, opponent_idx: Tensor) -> Tensor:
        mover_vec = self.transformer(mover_idx) + self.bias1
        opponent_vec = self.transformer(opponent_idx) + self.bias1

        x = torch.cat([mover_vec, opponent_vec], dim=1) / ACCUMULATOR_NORM
        x = torch.clamp(x, 0.0, 1.0)
        x = torch.clamp(self.hidden(x), 0.0, 1.0)

        return torch.sigmoid(self.output(x)).squeeze(1)


def load_arrays(data_dir: Path, n: int | None) -> tuple[np.memmap, np.memmap]:
    idx = np.load(data_dir / "train_idx_halfkp.npy", mmap_mode="r")
    target = np.load(data_dir / "train_target_halfkp.npy", mmap_mode="r")
    if n is not None:
        idx = idx[:n]
        target = target[:n]
    return idx, target


def remap(batch_idx: np.ndarray) -> torch.Tensor:
    batch_idx = batch_idx.copy()
    batch_idx[batch_idx < 0] = FEATURES
    return torch.from_numpy(batch_idx.astype(np.int64))


def prepare_batch(
    block_idx: np.ndarray, block_target: np.ndarray, local_ids: np.ndarray
) -> tuple[Tensor, Tensor, Tensor]:
    """local_ids index into an ALREADY-LOADED in-memory block, never the memmap
    directly -- this is what keeps per-batch cost to plain in-memory fancy
    indexing (cheap) instead of disk-backed fancy indexing (the actual problem)."""
    rows = block_idx[local_ids]
    mover = remap(rows[:, 0, :])
    opponent = remap(rows[:, 1, :])
    y = torch.from_numpy(np.array(block_target[local_ids], dtype=np.float32))
    return mover, opponent, y


def saturation(model: NNUE, block_idx: np.ndarray, block_target: np.ndarray) -> float:
    """Fraction of pre-clamp accumulator values landing OUTSIDE [0, 1] -- the
    fraction of information the clipped-ReLU is throwing away."""
    n = min(4000, len(block_idx))
    mover, opponent, _ = prepare_batch(block_idx, block_target, np.arange(n))
    with torch.no_grad():
        mover_vec = model.transformer(mover) + model.bias1
        opponent_vec = model.transformer(opponent) + model.bias1
        pre = torch.cat([mover_vec, opponent_vec], dim=1) / ACCUMULATOR_NORM
    return float(((pre < 0.0) | (pre > 1.0)).float().mean())


def run_epoch(
    model: NNUE,
    optimizer: torch.optim.Optimizer,
    idx: np.memmap,
    target: np.memmap,
    row_range: tuple[int, int],
    batch_size: int,
    train: bool,
) -> float:
    start, stop = row_range
    block_starts = list(range(start, stop, BLOCK_SIZE))
    if train:
        random.shuffle(block_starts)  # a different block VISIT ORDER each epoch

    result = 0.0
    n_total = 0
    with torch.set_grad_enabled(train):
        for block_start in block_starts:
            block_stop = min(block_start + BLOCK_SIZE, stop)
            # One sequential read per block -- a plain slice, not fancy indexing.
            block_idx = np.array(idx[block_start:block_stop])
            block_target = np.array(target[block_start:block_stop])

            local_order = np.arange(block_stop - block_start)
            if train:
                np.random.shuffle(local_order)  # real randomization, but only within this block

            for b_start in range(0, len(local_order), batch_size):
                local_ids = local_order[b_start : b_start + batch_size]
                mover, opponent, y = prepare_batch(block_idx, block_target, local_ids)
                prediction = model(mover, opponent)
                loss = nn.functional.mse_loss(prediction, y)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                result += loss.item() * len(local_ids)
                n_total += len(local_ids)
    return result / n_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HalfKP NNUE.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--rows", type=int, default=None, help="default: every packed row")
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--tag", type=str, default="", help="checkpoint filename suffix")
    parser.add_argument(
        "--resume", type=Path, default=None, help="checkpoint to continue training from"
    )
    parser.add_argument(
        "--start-epoch", type=int, default=1, help="epoch number to label the first new checkpoint"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    arguments = parser.parse_args()

    torch.set_num_threads(8)
    idx, target = load_arrays(arguments.data_dir, arguments.rows)

    n = len(idx)
    n_val = int(n * arguments.val_fraction)
    # Validation is a fixed slice at the end of the file: contiguous (one cheap
    # sequential read, done once, not every epoch) and never touched by training,
    # unlike train.py's approach of carving both sets out of one global permutation.
    val_range = (n - n_val, n)
    train_range = (0, n - n_val)
    print(f"{train_range[1] - train_range[0]:,} train rows, {n_val:,} validation rows")

    val_idx_block = np.array(idx[val_range[0] : val_range[1]])
    val_target_block = np.array(target[val_range[0] : val_range[1]])

    model = NNUE(hidden=arguments.hidden)
    if arguments.resume is not None:
        model.load_state_dict(torch.load(arguments.resume, map_location="cpu"))
        print(f"resumed from {arguments.resume}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.lr, weight_decay=arguments.weight_decay
    )

    checkpoint_dir = Path(__file__).parent
    last_epoch = arguments.start_epoch + arguments.epochs - 1
    for epoch in range(arguments.start_epoch, last_epoch + 1):
        train_loss = run_epoch(
            model, optimizer, idx, target, train_range, arguments.batch_size, train=True
        )
        val_loss = run_epoch(
            model, optimizer, idx, target, val_range, arguments.batch_size, train=False
        )
        sat = saturation(model, val_idx_block, val_target_block)
        print(f"epoch {epoch}: train {train_loss:.6f}  val {val_loss:.6f}  saturated {sat:.1%}")
        suffix = f"-{arguments.tag}" if arguments.tag else ""
        torch.save(model.state_dict(), checkpoint_dir / f"ckpt-halfkp{suffix}-{epoch}.pt")


if __name__ == "__main__":
    main()
