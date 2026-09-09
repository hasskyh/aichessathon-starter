"""For a given FEN, play each candidate move and report Stockfish's evaluation of
the resulting position from the ORIGINAL mover's perspective, so different search
configurations' chosen moves can be ranked against each other objectively.
"""
import subprocess
import sys

import chess


def sf_eval(proc, fen, movetime_ms=3000):
    proc.stdin.write(f"position fen {fen}\n")
    proc.stdin.write(f"go movetime {movetime_ms}\n")
    proc.stdin.flush()
    last = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        if line.startswith("info") and "score" in line:
            last = line
        if line.startswith("bestmove"):
            break
    if "score mate" in last:
        mate_in = int(last.split("score mate")[1].split()[0])
        return 30000 if mate_in > 0 else -30000
    if "score cp" in last:
        return int(last.split("score cp")[1].split()[0])
    return None


def main():
    sf_path = sys.argv[1]
    fen = sys.argv[2]
    labeled_moves = sys.argv[3:]  # "label:uci" pairs

    proc = subprocess.Popen([sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, bufsize=1)
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if "uciok" in line:
            break

    board = chess.Board(fen)
    mover_is_white = board.turn == chess.WHITE
    print(f"position: {fen}")
    print(f"mover: {'White' if mover_is_white else 'Black'}")

    for entry in labeled_moves:
        label, uci = entry.split(":", 1)
        b = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in b.legal_moves:
            print(f"  {label:20s} {uci:6s} ILLEGAL MOVE")
            continue
        b.push(move)
        # after any single move it's always the opponent's turn now, so Stockfish's
        # side-to-move-relative eval just needs negating to become mover-relative
        side_to_move_cp = sf_eval(proc, b.fen())
        mover_relative_cp = -side_to_move_cp
        print(f"  {label:20s} {uci:6s} mover-relative eval: {mover_relative_cp:+d}")

    proc.stdin.write("quit\n")
    proc.stdin.flush()


if __name__ == "__main__":
    main()
