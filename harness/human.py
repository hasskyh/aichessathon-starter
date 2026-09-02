"""Play a game yourself against an agent directory, from the terminal."""

import argparse
import contextlib
import time
from pathlib import Path

import chess
import chess.pgn

from harness.rules import BASE_MS, INCREMENT_MS, INIT_BUDGET_S
from harness.sandbox import Agent, AgentFailure, local

SAN_ERRORS = (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError)
QUIT = frozenset({"q", "quit", "resign"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a game against an agent directory.")
    parser.add_argument("--bot", type=Path, default=Path("baselines/greedy"))
    parser.add_argument("--colour", choices=("white", "black"), default="white")
    parser.add_argument("--base-ms", type=int, default=BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=INCREMENT_MS)
    parser.add_argument("--pgn", type=Path)
    arguments = parser.parse_args()

    human = chess.WHITE if arguments.colour == "white" else chess.BLACK
    bot = local(arguments.bot)
    board = chess.Board()

    print(f"you are {arguments.colour}; the bot is {arguments.bot}")
    print("moves as UCI (e2e4) or SAN (Nf3). commands: moves, fen, quit\n")
    try:
        bot.start(INIT_BUDGET_S)
        _show(board, human)
        termination = _play(board, bot, human, float(arguments.base_ms), arguments.increment_ms)
    except AgentFailure as failure:
        termination = f"the bot failed to {failure.reason}"
    finally:
        bot.stop()

    print(f"\n{termination}")
    if bot.stderr_tail:
        print(f"\nthe bot wrote to stderr:\n{bot.stderr_tail.rstrip()}")
    if arguments.pgn:
        game = chess.pgn.Game.from_board(board)
        game.headers["Termination"] = termination
        arguments.pgn.write_text(str(game) + "\n")
        print(f"pgn written to {arguments.pgn}")


def _play(
    board: chess.Board, bot: Agent, human: chess.Color, clock_ms: float, increment: int
) -> str:
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return _describe(outcome, human)

        if board.turn == human:
            move = _ask(board)
            if move is None:
                return "you resigned"
        else:
            started_at = time.monotonic()
            uci = bot.move(board.fen(), int(clock_ms))
            clock_ms -= (time.monotonic() - started_at) * 1000.0
            if clock_ms < 0:
                return "the bot flagged; you win"
            parsed = _legal(board, uci)
            if parsed is None:
                return f"the bot returned an illegal move ({uci}); you win"
            clock_ms += increment
            move = parsed
            print(f"the bot played {board.san(move)} ({uci}), {clock_ms / 1000:.1f}s left")

        board.push(move)
        _show(board, human)


def _ask(board: chess.Board) -> chess.Move | None:
    while True:
        try:
            entry = input("your move: ").strip()
        except EOFError:
            return None
        if entry in QUIT:
            return None
        if entry == "fen":
            print(board.fen())
        elif entry == "moves":
            print(" ".join(sorted(board.san(move) for move in board.legal_moves)))
        else:
            move = _parse(board, entry)
            if move is not None:
                return move
            print("not a legal move here; 'moves' lists what is")


def _parse(board: chess.Board, entry: str) -> chess.Move | None:
    with contextlib.suppress(*SAN_ERRORS):
        return board.parse_san(entry)
    return _legal(board, entry)


def _legal(board: chess.Board, uci: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except chess.InvalidMoveError:
        return None
    return move if move in board.legal_moves else None


def _describe(outcome: chess.Outcome, human: chess.Color) -> str:
    reason = outcome.termination.name.lower().replace("_", " ")
    if outcome.winner is None:
        return f"draw by {reason}"
    winner = "you win" if outcome.winner == human else "the bot wins"
    return f"{winner} by {reason}"


def _show(board: chess.Board, human: chess.Color) -> None:
    print()
    print(board.unicode(invert_color=True, borders=False, orientation=human))
    print()


if __name__ == "__main__":
    main()
