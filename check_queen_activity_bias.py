"""Direct test of the sub-population hypothesis: within "down material" bins,
does the mover having an ADVANCED (active-looking) own queen correlate with a
more optimistic (less negative) mean label than a queen still at home, or no
queen at all? Tests on data/raw.jsonl directly, no training/Stockfish needed.
"""
import json
import random
import sys

PIECE_VALUE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}

BINS = [
    ("down a minor (-4..-2)", lambda d: -4 <= d <= -2),
    ("down a pawn (-1)", lambda d: d == -1),
]


def parse_board(board_field: str):
    """Yields (rank, file, char) for every piece, rank 1-8 (1=White's back rank)."""
    for rank_idx, rank_str in enumerate(board_field.split("/")):
        rank = 8 - rank_idx  # FEN ranks go 8 down to 1
        file = 0
        for ch in rank_str:
            if ch.isdigit():
                file += int(ch)
            else:
                yield rank, file, ch
                file += 1


def material_diff_and_queen(board_field: str, white_to_move: bool):
    white = 0
    black = 0
    mover_queen_rank = None
    for rank, _file, ch in parse_board(board_field):
        lower = ch.lower()
        if lower in PIECE_VALUE:
            if ch.isupper():
                white += PIECE_VALUE[lower]
            else:
                black += PIECE_VALUE[lower]
        if lower == "q":
            is_white_piece = ch.isupper()
            if is_white_piece == white_to_move:  # this queen belongs to the mover
                mover_queen_rank = rank
    diff = white - black
    diff = diff if white_to_move else -diff
    return diff, mover_queen_rank


def classify_queen(mover_queen_rank, white_to_move: bool) -> str:
    if mover_queen_rank is None:
        return "no_queen"
    # "advanced" = on the opponent's half of the board, from the mover's own view
    if white_to_move:
        return "advanced" if mover_queen_rank >= 5 else "home"
    else:
        return "advanced" if mover_queen_rank <= 4 else "home"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw.jsonl"
    sample_target = 3_000_000
    with open(path, encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    keep_prob = min(1.0, 2 * sample_target / n_lines)
    print(f"{n_lines:,} lines, keep_prob={keep_prob:.5f}", file=sys.stderr)

    random.seed(0)
    counts = {}
    sums_cp = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if random.random() > keep_prob:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                fen = row["fen"]
                cp = float(row["cp"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            fields = fen.split(" ")
            board_field, turn_field = fields[0], fields[1]
            white_to_move = turn_field == "w"
            diff, mover_queen_rank = material_diff_and_queen(board_field, white_to_move)
            mover_cp = cp if white_to_move else -cp
            queen_class = classify_queen(mover_queen_rank, white_to_move)
            for name, pred in BINS:
                if pred(diff):
                    key = (name, queen_class)
                    counts[key] = counts.get(key, 0) + 1
                    sums_cp[key] = sums_cp.get(key, 0.0) + mover_cp
                    break

    print(f"{'bin':22s} {'queen':>10s} {'count':>10s} {'mean mover_cp':>15s}")
    for name, _ in BINS:
        for qc in ("advanced", "home", "no_queen"):
            key = (name, qc)
            c = counts.get(key, 0)
            if c == 0:
                print(f"{name:22s} {qc:>10s} {0:>10d}")
                continue
            print(f"{name:22s} {qc:>10s} {c:>10,} {sums_cp[key] / c:>15.1f}")


if __name__ == "__main__":
    main()
