"""Fine-tune the standard-net checkpoint on a mix of the original data (a sample,
not all 11M rows) and the new synthetic "uncompensated material loss" positions,
at a low learning rate for a few epochs. Mirrors training/finetune_halfkp.py.

    uv run python finetune_standard.py \
        --resume training/ckpt-fixed-continued-34.pt \
        --synthetic data/synthetic_sacrifice_standard.jsonl \
        --real-sample 200000 --epochs 3 --lr 1e-5 --tag finetune-sac
"""
import argparse
import json
from pathlib import Path

import chess
import numpy as np
import torch

import features
from training.train import NNUE, remap

SIGMOID_SCALE = 400.0
MAX_FEATURES = 32


def pack_jsonl(path: Path, max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows_idx = []
    rows_target = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if max_rows is not None and len(rows_idx) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            board = chess.Board(data["fen"])
            white_row, black_row = features.active(board)
            mover, opponent = (
                (white_row, black_row) if board.turn == chess.WHITE else (black_row, white_row)
            )
            padded = np.full((2, MAX_FEATURES), -1, dtype=np.int16)
            padded[0, : len(mover)] = mover
            padded[1, : len(opponent)] = opponent
            mover_cp = data["cp"] if board.turn == chess.WHITE else -data["cp"]
            target = 1.0 / (1.0 + np.exp(-mover_cp / SIGMOID_SCALE))
            rows_idx.append(padded)
            rows_target.append(target)
    return np.array(rows_idx, dtype=np.int16), np.array(rows_target, dtype=np.float32)


def sample_real_data(data_dir: Path, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.load(data_dir / "train_idx.npy", mmap_mode="r")
    target = np.load(data_dir / "train_target.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(idx), n, replace=False)
    ids.sort()
    return np.array(idx[ids]), np.array(target[ids])


def run_epoch(
    model: NNUE, optimizer: torch.optim.Optimizer | None,
    idx: np.ndarray, target: np.ndarray, batch_size: int, train: bool,
) -> float:
    order = np.arange(len(idx))
    if train:
        np.random.shuffle(order)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            rows = idx[batch_ids]
            mover = remap(rows[:, 0, :])
            opponent = remap(rows[:, 1, :])
            y = torch.from_numpy(target[batch_ids].astype(np.float32))
            prediction = model(mover, opponent)
            loss = torch.nn.functional.mse_loss(prediction, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(batch_ids)
    return total_loss / len(order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--real-sample", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--tag", type=str, default="finetune")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("packing synthetic data...")
    synth_idx, synth_target = pack_jsonl(args.synthetic)
    print(f"  {len(synth_idx):,} synthetic rows")

    print("sampling real data...")
    real_idx, real_target = sample_real_data(args.data_dir, args.real_sample, args.seed)
    print(f"  {len(real_idx):,} real rows")

    all_idx = np.concatenate([synth_idx, real_idx])
    all_target = np.concatenate([synth_target, real_target])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_idx))
    all_idx, all_target = all_idx[perm], all_target[perm]

    n_val = int(len(all_idx) * args.val_fraction)
    val_idx, val_target = all_idx[:n_val], all_target[:n_val]
    train_idx, train_target = all_idx[n_val:], all_target[n_val:]
    print(f"{len(train_idx):,} train rows, {len(val_idx):,} val rows (mixed synthetic+real)")

    model = NNUE(hidden=256, outputs=32)
    model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    print(f"resumed from {args.resume}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    checkpoint_dir = Path("training")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, optimizer, train_idx, train_target, args.batch_size, True)
        val_loss = run_epoch(model, None, val_idx, val_target, args.batch_size, False)
        print(f"epoch {epoch}: train {train_loss:.6f}  val {val_loss:.6f}")
        torch.save(model.state_dict(), checkpoint_dir / f"ckpt-{args.tag}-{epoch}.pt")


if __name__ == "__main__":
    main()
