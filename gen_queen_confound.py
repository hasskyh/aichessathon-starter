"""Generate synthetic positions that specifically target the queen-activity
confound the autopsy found: within "down material" positions, the mover having
an advanced (opponent's-half) queen correlates with a much more optimistic mean
label in real data (+302cp for down-a-minor+advanced-queen vs -56cp for
down-a-minor+home-queen), because in real games an advanced queen usually DOES
come with real tactical compensation. The net over-generalized this into "active
queen = compensation" without checking whether compensation is actually there.

This strips a random non-king, non-queen piece from real positions (same method
as gen_synthetic_sacrifice.py) but KEEPS ONLY the results where the mover ends up
down material AND with an advanced queen -- i.e. positions that look exactly like
the deceptive pattern but have no real compensation (a piece removed at random
can't create real tactical comp), re-labelled by Stockfish. This directly
oversamples the exact sub-population the net is miscalibrated on, rather than
relying on it turning up by chance in a uniformly random sample.

The reject filter runs on the raw FEN board-field string first (same approach as
check_queen_activity_bias.py) so the ~98% of candidates that don't match never
pay for a chess.Board() construction or legal_moves generation -- an earlier
version built the Board and checked legal_moves/is_valid FIRST, which made an
11M-line pass too slow to finish in any reasonable time.

    uv run python gen_queen_confound.py data/raw.jsonl out.jsonl 40000 /path/to/stockfish
"""
import json
import random
import subprocess
import sys

import chess

PIECE_VALUE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
NON_KING_NON_QUEEN = set("pnbrPNBR")


def parse_board(board_field: str):
    """Yields (rank, file, char) for every piece, rank 1-8 (1=White's back rank)."""
    for rank_idx, rank_str in enumerate(board_field.split("/")):
        rank = 8 - rank_idx
        file = 0
        for ch in rank_str:
            if ch.isdigit():
                file += int(ch)
            else:
                yield rank, file, ch
                file += 1


def find_confound_candidate(fen: str) -> tuple[int, int] | None:
    """Cheap, string-only pass: does removing SOME non-king/queen piece leave the
    mover down material with their own queen advanced? Returns (rank, file) of a
    piece to remove if so, else None. No chess.Board() involved."""
    fields = fen.split(" ")
    board_field, turn_field = fields[0], fields[1]
    mover_is_white = turn_field == "w"

    white = 0
    black = 0
    removable = []  # (rank, file) of non-king/queen pieces
    mover_queen_rank = None
    for rank, file, ch in parse_board(board_field):
        lower = ch.lower()
        if lower in PIECE_VALUE:
            if ch.isupper():
                white += PIECE_VALUE[lower]
            else:
                black += PIECE_VALUE[lower]
        if ch in NON_KING_NON_QUEEN:
            removable.append((rank, file))
        if lower == "q" and ch.isupper() == mover_is_white:
            mover_queen_rank = rank

    if mover_queen_rank is None:
        return None
    queen_advanced = mover_queen_rank >= 5 if mover_is_white else mover_queen_rank <= 4
    if not queen_advanced:
        return None
    if not removable:
        return None

    random.shuffle(removable)
    for rank, file in removable:
        is_white_piece = None
        for r, f, ch in parse_board(board_field):
            if (r, f) == (rank, file):
                is_white_piece = ch.isupper()
                value = PIECE_VALUE[ch.lower()]
                break
        new_white = white - value if is_white_piece else white
        new_black = black - value if not is_white_piece else black
        diff = new_white - new_black
        diff = diff if mover_is_white else -diff
        if -6 <= diff <= -1:  # down a pawn to down a rook -- matches the confound bins
            return rank, file
    return None


def square_from_rank_file(rank: int, file: int) -> int:
    return chess.square(file, rank - 1)  # chess.square wants 0-indexed file/rank


def strip_piece_at(fen: str, rank: int, file: int) -> str | None:
    board = chess.Board(fen)
    board.remove_piece_at(square_from_rank_file(rank, file))
    if not any(board.legal_moves):
        return None
    if board.is_valid() is False:
        return None
    return board.fen()


def sf_eval(proc: subprocess.Popen, fen: str, movetime_ms: int = 100) -> int | None:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(f"position fen {fen}\n")
    proc.stdin.write(f"go movetime {movetime_ms}\n")
    proc.stdin.flush()
    last_score_line = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        if line.startswith("info") and "score" in line:
            last_score_line = line
        if line.startswith("bestmove"):
            break
    if "score mate" in last_score_line:
        mate_in = int(last_score_line.split("score mate")[1].split()[0])
        return 3000 if mate_in > 0 else -3000
    if "score cp" in last_score_line:
        return int(last_score_line.split("score cp")[1].split()[0])
    return None


def main() -> None:
    in_path, out_path, n_wanted, sf_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    random.seed(2)

    with open(in_path, encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    print(f"{n_lines:,} lines available", file=sys.stderr)

    proc = subprocess.Popen(
        [sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if "uciok" in line:
            break

    written = 0
    tried = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_number, line in enumerate(fin, 1):
            if written >= n_wanted:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                fen = data["fen"]
                candidate = find_confound_candidate(fen)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if candidate is None:
                continue
            tried += 1
            new_fen = strip_piece_at(fen, candidate[0], candidate[1])
            if new_fen is None:
                continue
            # sf_eval is side-to-move-relative; flip to White-relative to match
            # raw.jsonl's own convention (see gen_synthetic_sacrifice.py's note --
            # same sign-bug class this project already fixed once, verified again here).
            side_to_move_cp = sf_eval(proc, new_fen)
            if side_to_move_cp is None:
                continue
            is_white_to_move = new_fen.split(" ")[1] == "w"
            cp = side_to_move_cp if is_white_to_move else -side_to_move_cp
            fout.write(json.dumps({"fen": new_fen, "cp": cp}) + "\n")
            written += 1
            if written % 2000 == 0:
                print(f"  {written:,}/{n_wanted:,} written ({tried:,} candidates tried, "
                      f"{line_number:,}/{n_lines:,} source lines scanned)", file=sys.stderr)

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=5)
    print(f"done: wrote {written:,} synthetic positions to {out_path} "
          f"(tried {tried:,} candidates)", file=sys.stderr)


if __name__ == "__main__":
    main()
