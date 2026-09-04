"""Quantise a trained HalfKP checkpoint into the int arrays nnue_halfkp.py runs on.

    uv run python -m training.export_halfkp --checkpoint training/ckpt-halfkp-10.pt

Byte-for-byte the same quantisation scheme as export.py -- same ACT_MAX/HIDDEN_SHIFT
scale chain, same ACCUMULATOR_NORM folding, same real bias1 -- pointed at
nnue_halfkp/train_halfkp instead of nnue/train. See export.py's module docstring
and quantize()'s comments for why each scale factor is what it is; none of that
changes with the feature set, only which module supplies FEATURES and the model.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import nnue_halfkp
from nnue_halfkp import ACT_MAX, FEATURES, HIDDEN_SHIFT
from training.train_halfkp import ACCUMULATOR_NORM, NNUE, load_arrays, prepare_batch

WEIGHT_SCALE = 2 ** HIDDEN_SHIFT        # 64
OUTPUT_SCALE = ACT_MAX * WEIGHT_SCALE   # 8128


def load_checkpoint(path: Path, hidden: int) -> NNUE:
    model = NNUE(hidden=hidden)
    result = model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"  load_state_dict: missing={result.missing_keys} "
              f"unexpected={result.unexpected_keys}")
    return model.eval()


def quantize(model: NNUE) -> dict[str, np.ndarray]:
    embedding = model.transformer.weight.detach().numpy()
    w1 = np.clip(
        np.round(embedding[:FEATURES] * ACT_MAX / ACCUMULATOR_NORM), -32768, 32767
    ).astype(np.int16)
    b1 = np.clip(
        np.round(model.bias1.detach().numpy() * ACT_MAX / ACCUMULATOR_NORM), -32768, 32767
    ).astype(np.int16)

    w2 = np.ascontiguousarray(np.clip(
        np.round(model.hidden.weight.detach().numpy().T * WEIGHT_SCALE), -127, 127
    ).astype(np.int8))
    b2 = np.round(model.hidden.bias.detach().numpy() * OUTPUT_SCALE).astype(np.int32)

    w3 = np.clip(
        np.round(model.output.weight.detach().numpy().squeeze(0) * WEIGHT_SCALE), -127, 127
    ).astype(np.int8)
    b3 = np.int32(round(model.output.bias.item() * OUTPUT_SCALE))

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def save_weights(weights: dict[str, np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, array in weights.items():
        print(f"  {name:4s} shape={getattr(array, 'shape', ())} dtype={array.dtype} "
              f"min={array.min()} max={array.max()}")
    np.savez(out_dir / "nnue_halfkp.npz", **weights)


def verify(
    model: NNUE, weights: dict[str, np.ndarray], idx: np.memmap, target: np.memmap, n_samples: int
) -> None:
    sample_ids = np.random.choice(len(idx), n_samples, replace=False)
    mover, opponent, _ = prepare_batch(idx, target, sample_ids)
    with torch.no_grad():
        float_pred = model(mover, opponent).numpy()

    quant_pred = np.zeros(n_samples)
    hidden = weights["w1"].shape[1]
    for k, row_id in enumerate(sample_ids):
        acc = np.zeros((nnue_halfkp.PERSPECTIVES, hidden), dtype=np.int32)
        nnue_halfkp.refresh(
            acc, weights["w1"], weights["b1"], np.array(idx[row_id], dtype=np.int32, order="C")
        )
        total = nnue_halfkp.forward(
            acc, 0, weights["w2"], weights["b2"], weights["w3"], weights["b3"]
        )
        quant_pred[k] = 1.0 / (1.0 + np.exp(-total / OUTPUT_SCALE))

    diff = np.abs(float_pred - quant_pred)
    print(f"  mean |float - quantized| = {diff.mean():.5f}")
    print(f"  max  |float - quantized| = {diff.max():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantise and export the HalfKP NNUE.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=256, help="must match the checkpoint")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("weights_halfkp"))
    parser.add_argument("--verify-samples", type=int, default=500)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, args.hidden)
    weights = quantize(model)
    save_weights(weights, args.out_dir)

    idx, target = load_arrays(args.data_dir, None)
    verify(model, weights, idx, target, args.verify_samples)


if __name__ == "__main__":
    main()
