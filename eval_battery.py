"""Query the trained standard net directly (no search) on a battery of hand-picked
positions spanning clear material/positional cases, to check whether it now
"understands" basic chess signals qualitatively -- does white-up-a-queen score
much higher than white-down-a-queen, is the start position roughly neutral, etc.
This is a static-eval sanity check, independent of search entirely.
"""
import sys

import chess
import torch

sys.path.insert(0, "/home/harry/aichessathon-starter")
import features_768 as features
from training.train import NNUE

CASES = [
    ("start position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("white up a queen (K+Q vs K)", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"),
    ("black up a queen (K vs K+Q)", "3qk3/8/8/8/8/8/8/4K3 w - - 0 1"),
    ("white up a rook (K+R vs K)", "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
    ("black up a rook (K vs K+R)", "r3k3/8/8/8/8/8/8/4K3 w - - 0 1"),
    ("white up a knight (K+N vs K)", "4k3/8/8/8/8/8/8/N3K3 w - - 0 1"),
    ("dead equal endgame (K vs K)", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
    ("white up a pawn (K+P vs K)", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"),
    ("black up a pawn (K vs K+P)", "4k3/4p3/8/8/8/8/8/4K3 w - - 0 1"),
    ("blindness position, before Rxe5 (roughly balanced per Stockfish)",
     "5b1r/1QR2pp1/p3rqkp/B3p3/4R3/7P/5PP1/6K1 w - - 10 36"),
]

# Derived programmatically (via python-chess push, not hand-edited) to avoid any
# manual FEN-transcription mistake: the position right after the real game's
# Rxe5 Qxe5 exchange, where White is down a rook for a pawn.
_after_board = chess.Board("5b1r/1QR2pp1/p3rqkp/B3p3/4R3/7P/5PP1/6K1 w - - 10 36")
_after_board.push_uci("e4e5")
_after_board.push_uci("f6e5")
CASES.append(("blindness position, after Rxe5 Qxe5 (White down a rook for a pawn)",
              _after_board.fen()))


def query(model, fen: str) -> float:
    board = chess.Board(fen)
    active = features.active(board)
    mover_idx = active[0] if board.turn == chess.WHITE else active[1]
    opp_idx = active[1] if board.turn == chess.WHITE else active[0]
    mover = torch.tensor(mover_idx, dtype=torch.int64).unsqueeze(0)
    opp = torch.tensor(opp_idx, dtype=torch.int64).unsqueeze(0)
    with torch.no_grad():
        pred = model(mover, opp).item()
    return pred if board.turn == chess.WHITE else 1 - pred


def main() -> None:
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "training/ckpt-lightning-quiet-final-stripped.pt"
    model = NNUE(hidden=256, outputs=32)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    print(f"checkpoint: {ckpt_path}")
    print(f"{'case':50s} white_win_prob")
    for label, fen in CASES:
        prob = query(model, fen)
        print(f"{label:50s} {prob:.4f}")


if __name__ == "__main__":
    main()
