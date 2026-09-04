"""Quantise a trained checkpoint into the int arrays nnue.py runs on.

    uv run python -m training.export --checkpoint training/ckpt-10.pt

Every scale factor here is forced by two constants nnue.py already fixes: ACT_MAX
(the runtime's clipped-ReLU ceiling) and HIDDEN_SHIFT (the bit-shift applied after
the hidden layer). They are imported, never retyped, so the two files cannot drift
apart the way features.py and pack.py briefly did.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import nnue
from nnue import ACT_MAX, FEATURES, HIDDEN_SHIFT
from training.train import ACCUMULATOR_NORM, KEPT_ROWS, NNUE, load_arrays, prepare_batch

WEIGHT_SCALE = 2 ** HIDDEN_SHIFT        # 64
OUTPUT_SCALE = ACT_MAX * WEIGHT_SCALE   # 8128


def load_checkpoint(path: Path, hidden: int) -> NNUE:
    model = NNUE(hidden=hidden)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def quantize(model: NNUE) -> dict[str, np.ndarray]:
    embedding = model.transformer.weight.detach().numpy()
    # training divides the raw sum by ACCUMULATOR_NORM before its clamp (see train.py);
    # the runtime clamp has no such division, so that factor must be folded into the
    # scale here or the exported weights are ACCUMULATOR_NORM times too large.
    w1 = np.clip(
        np.round(embedding[:FEATURES] * ACT_MAX / ACCUMULATOR_NORM), -32768, 32767
    ).astype(np.int16)
    # derived from the model, not nnue.py's HIDDEN constant: this file must export
    # correctly whatever size was actually trained, matching hidden.
    b1 = np.zeros(model.transformer.embedding_dim, dtype=np.int16)

    w2 = np.clip(
        np.round(model.hidden.weight.detach().numpy().T * WEIGHT_SCALE), -127, 127
    ).astype(np.int8)
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
    np.savez(out_dir / "nnue.npz", **weights)


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
        acc = np.zeros((nnue.PERSPECTIVES, hidden), dtype=np.int32)
        nnue.refresh(acc, weights["w1"], weights["b1"], np.array(idx[row_id]))
        total = nnue.forward(acc, 0, weights["w2"], weights["b2"], weights["w3"], weights["b3"])
        quant_pred[k] = 1.0 / (1.0 + np.exp(-total / OUTPUT_SCALE))

    diff = np.abs(float_pred - quant_pred)
    print(f"  mean |float - quantized| = {diff.mean():.5f}")
    print(f"  max  |float - quantized| = {diff.max():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantise and export the NNUE.")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/ckpt-10.pt"))
    parser.add_argument("--hidden", type=int, default=256, help="must match the checkpoint")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("weights"))
    parser.add_argument("--verify-samples", type=int, default=500)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, args.hidden)
    weights = quantize(model)
    save_weights(weights, args.out_dir)

    idx, target = load_arrays(args.data_dir, KEPT_ROWS)
    verify(model, weights, idx, target, args.verify_samples)


if __name__ == "__main__":
    main()
