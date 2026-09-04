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
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from nnue_halfkp import FEATURES, HIDDEN

OUTPUTS = 32
ACCUMULATOR_NORM = 16.0


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
    idx: np.memmap, target: np.memmap, ids: np.ndarray
) -> tuple[Tensor, Tensor, Tensor]:
    rows = np.array(idx[ids])  # one real copy out of the memmap, per batch
    mover = remap(rows[:, 0, :])
    opponent = remap(rows[:, 1, :])
    y = torch.from_numpy(np.array(target[ids], dtype=np.float32))
    return mover, opponent, y


def saturation(model: NNUE, idx: np.memmap, target: np.memmap, ids: np.ndarray) -> float:
    """Fraction of pre-clamp accumulator values landing OUTSIDE [0, 1] -- the
    fraction of information the clipped-ReLU is throwing away."""
    mover, opponent, _ = prepare_batch(idx, target, ids[:4000])
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
    ids: np.ndarray,
    batch_size: int,
    train: bool,
) -> float:
    if train:
        np.random.shuffle(ids)

    result = 0.0
    with torch.set_grad_enabled(train):
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            mover, opponent, y = prepare_batch(idx, target, batch_ids)
            prediction = model(mover, opponent)
            loss = nn.functional.mse_loss(prediction, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            result += loss.item() * len(batch_ids)
    return result / len(ids)


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

    ids = np.random.permutation(len(idx))
    n_val = int(len(idx) * arguments.val_fraction)
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    print(f"{len(train_ids):,} train rows, {len(val_ids):,} validation rows")

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
            model, optimizer, idx, target, train_ids, arguments.batch_size, train=True
        )
        val_loss = run_epoch(
            model, optimizer, idx, target, val_ids, arguments.batch_size, train=False
        )
        sat = saturation(model, idx, target, val_ids)
        print(f"epoch {epoch}: train {train_loss:.6f}  val {val_loss:.6f}  saturated {sat:.1%}")
        suffix = f"-{arguments.tag}" if arguments.tag else ""
        torch.save(model.state_dict(), checkpoint_dir / f"ckpt-halfkp{suffix}-{epoch}.pt")


if __name__ == "__main__":
    main()
