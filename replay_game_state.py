"""Replay a real game through the ACTUAL engine process, one real get_move() call
per own-side move, so TT/history/killers/GAME_HISTORY accumulate exactly as they
did during the real match -- then stop at a chosen move and report what the
engine does next with genuine accumulated state, instead of the cold-TT/cold-
history state every earlier fresh test in this investigation necessarily started
from.

Time budget per move is reconstructed from the PGN's own {Xs} elapsed-time
annotations, tracking OUR side's clock only (base + increment per move, minus
each move's real elapsed time) -- the engine gets the same shrinking clock it
really had, not a generous flat budget, since get_move()'s own think-time
allocation (time_left_ms / 25_000) depends on it.

    uv run python replay_game_state.py <baseline-dir> <pgn-file> <our-color: white|black> \\
        <stop-at-our-move-number> [base-ms] [increment-ms]
"""
import re
import sys

import chess
import chess.pgn


def main() -> None:
    baseline_dir = sys.argv[1]
    pgn_path = sys.argv[2]
    our_color_str = sys.argv[3]
    stop_at_move = int(sys.argv[4])
    base_ms = int(sys.argv[5]) if len(sys.argv) > 5 else 120_000
    increment_ms = int(sys.argv[6]) if len(sys.argv) > 6 else 500

    sys.path.insert(0, baseline_dir)
    import agent  # noqa: E402

    our_color = chess.WHITE if our_color_str == "white" else chess.BLACK

    with open(pgn_path, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)

    board = game.board()
    our_time_left_ms = base_ms
    move_number = 1

    node = game
    while node.variations:
        node = node.variations[0]
        move = node.move
        movers_turn_is_white = board.turn == chess.WHITE
        is_our_move = movers_turn_is_white == our_color

        if is_our_move:
            fen_before = board.fen()
            elapsed_match = re.search(r"([\d.]+)s", node.comment or "")
            real_elapsed_s = float(elapsed_match.group(1)) if elapsed_match else 0.0

            if move_number == stop_at_move:
                print(f"STOPPING at our move {move_number}, fen: {fen_before}")
                print(f"our_time_left_ms going in: {our_time_left_ms}")
                chosen = agent.get_move(fen_before, our_time_left_ms)
                print(f"ENGINE WITH REAL ACCUMULATED STATE PLAYS: {chosen}")
                print(f"ACTUAL GAME MOVE WAS: {move.uci()} "
                      f"(really took {real_elapsed_s}s)")
                return

            chosen = agent.get_move(fen_before, our_time_left_ms)
            actual_uci = move.uci()
            match = "OK" if chosen == actual_uci else "DIFFERS"
            print(f"replay move {move_number}: engine={chosen} real={actual_uci} "
                  f"[{match}] time_left={our_time_left_ms}ms real_spent={real_elapsed_s}s")
            our_time_left_ms = our_time_left_ms - int(real_elapsed_s * 1000) + increment_ms
            our_time_left_ms = max(our_time_left_ms, 1)

        board.push(move)
        if not movers_turn_is_white:
            move_number += 1

    print("reached end of game without hitting the stop point")


if __name__ == "__main__":
    main()
