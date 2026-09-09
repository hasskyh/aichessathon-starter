"""Train our own 768-feature NNUE architecture (imported unchanged from
training.train) using PyTorch Lightning's generic, battle-tested Trainer instead
of train.py's own hand-written epoch/optimizer loop. Same model class, same data
format, same loss -- the only thing that changes is which code runs the training
loop. If this also reproduces the material-blindness pattern on the independent
binpack data (data_binpack/), that further implicates something more fundamental
(architecture, feature encoding, or the pattern being real and hard) rather than
a bug specific to train.py's own loop; if it doesn't, that isolates the bug to
train.py specifically.

Sweeps a generous epoch count with per-epoch checkpoints and val-loss logging so
the stabilization point can be read off afterwards, rather than guessing a count
up front (train.py's own earlier binpack run used an arbitrary 25).

    uv run python train_lightning.py --data-dir data_binpack --rows 434942 --epochs 40
"""
import argparse
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

import sys
sys.path.insert(0, "/home/harry/aichessathon-starter")
from training.train import NNUE, ACCUMULATOR_NORM, load_arrays, prepare_batch  # noqa: E402


class BatchIterableDataset(IterableDataset):
    """Yields whole pre-batched tensors via ONE bulk numpy fancy-index call per
    batch (idx[batch_ids], matching train.py's own prepare_batch exactly) instead
    of a map-style Dataset's one-row-per-__getitem__ pattern. The former fixed
    the Python-interpreter-overhead cost of one __getitem__ call per row, but at
    42.9M rows (5.5GB) a full-dataset shuffle still gathers each batch from ids
    scattered across the whole file -- on a 5.8GB box that doesn't comfortably
    cache 5.5GB alongside everything else, this thrashes just as badly as the
    original per-row access did (confirmed: swap climbed steadily, throughput
    projected to ~39h for 8 epochs).

    Fix: bucket ids into fixed-size contiguous row-index blocks, shuffle block
    VISIT ORDER each epoch, and shuffle rows only WITHIN a block (already in
    RAM once read). Each block is one bounded, near-sequential read instead of
    a scatter across the full file, so total I/O per epoch is ~one sequential
    pass rather than repeated eviction/refetch thrashing -- while still
    reshuffling every epoch at both the block and row level."""

    def __init__(self, idx: np.memmap, target: np.memmap, ids: np.ndarray,
                 batch_size: int, shuffle: bool, block_size: int = 1_000_000) -> None:
        self.idx = idx
        self.target = target
        self.ids = np.sort(ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.block_size = block_size

    def __iter__(self):
        n_total = self.idx.shape[0]
        block_starts = list(range(0, n_total, self.block_size))
        if self.shuffle:
            np.random.shuffle(block_starts)
        for block_start in block_starts:
            block_end = block_start + self.block_size
            lo = np.searchsorted(self.ids, block_start, side="left")
            hi = np.searchsorted(self.ids, block_end, side="left")
            block_ids = self.ids[lo:hi]
            if len(block_ids) == 0:
                continue
            if self.shuffle:
                block_ids = block_ids.copy()
                np.random.shuffle(block_ids)
            for start in range(0, len(block_ids), self.batch_size):
                batch_ids = block_ids[start:start + self.batch_size]
                yield prepare_batch(self.idx, self.target, batch_ids)


class LitNNUE(L.LightningModule):
    def __init__(self, hidden: int, outputs: int, lr: float, weight_decay: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = NNUE(hidden=hidden, outputs=outputs)

    def forward(self, mover, opponent):
        return self.model(mover, opponent)

    def _step(self, batch, stage: str):
        mover, opponent, y = batch
        prediction = self.model(mover, opponent)
        loss = nn.functional.mse_loss(prediction, y)
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--outputs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--default-root-dir", type=Path, default=Path("lightning_run"))
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    idx, target = load_arrays(args.data_dir, args.rows)
    ids = np.random.default_rng(0).permutation(len(idx))
    n_val = int(len(idx) * args.val_fraction)
    val_ids, train_ids = ids[:n_val], ids[n_val:]
    print(f"{len(train_ids):,} train rows, {len(val_ids):,} validation rows")

    train_loader = DataLoader(
        BatchIterableDataset(idx, target, train_ids, args.batch_size, shuffle=True),
        batch_size=None, num_workers=0,
    )
    val_loader = DataLoader(
        BatchIterableDataset(idx, target, val_ids, args.batch_size, shuffle=False),
        batch_size=None, num_workers=0,
    )

    model = LitNNUE(args.hidden, args.outputs, args.lr, args.weight_decay)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        devices=1,
        default_root_dir=str(args.default_root_dir),
        enable_progress_bar=True,
        log_every_n_steps=10,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_epochs=1),
        ],
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)


if __name__ == "__main__":
    main()
