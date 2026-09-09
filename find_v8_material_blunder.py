"""Scan a PGN for moves played by invictus_v8_material that Stockfish flags as a
big eval swing against the mover -- a real, observed blunder to use as the "we
know this doesn't work" test case for node-based search tracing, rather than
reusing the NNUE-specific positions (which don't apply to a material-only
evaluator by construction).
"""
import subprocess
import sys

import chess
import chess.pgn


def sf_eval(proc, board, movetime_ms=200):
    proc.stdin.write(f"position fen {board.fen()}\n")
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
        return 3000 if mate_in > 0 else -3000
    if "score cp" in last:
        return int(last.split("score cp")[1].split()[0])
    return None


def main():
    pgn_path, sf_path = sys.argv[1], sys.argv[2]
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 250

    proc = subprocess.Popen([sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, bufsize=1)
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if "uciok" in line:
            break

    with open(pgn_path, encoding="utf-8") as f:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_idx += 1
            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            if "v8_material" not in white and "v8_material" not in black:
                continue
            material_is_white = "v8_material" in white

            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                movers_turn_is_white = board.turn == chess.WHITE
                is_material_move = (movers_turn_is_white == material_is_white)
                if is_material_move:
                    before = sf_eval(proc, board)
                    before_mover = before if movers_turn_is_white else -before
                fen_before = board.fen()
                board.push(move)
                if is_material_move and before is not None:
                    after = sf_eval(proc, board)
                    # after the move, turn has flipped, so "after" is from the
                    # OPPONENT's perspective -- flip back to the mover's perspective
                    after_mover = -after if movers_turn_is_white else after
                    swing = after_mover - before_mover
                    if swing < -threshold:
                        print(f"game {game_idx} ply {ply}: {white} vs {black}, "
                              f"move={move.uci()} swing={swing} "
                              f"(before={before_mover} after={after_mover})")
                        print(f"  fen before move: {fen_before}")
                        print(f"  fen after move:  {board.fen()}")

    proc.stdin.write("quit\n")
    proc.stdin.flush()


if __name__ == "__main__":
    main()
