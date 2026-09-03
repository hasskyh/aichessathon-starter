"""Serve an agent directory as a UCI engine, so any chess GUI can play it.

    uv run python -m harness.uci --agent .

The agent runs inside the platform's runner, one process per game, so its
print() output cannot reach the UCI stream and its state is discarded on
ucinewgame exactly as it is between rated games.

Set AICHESS_UCI_LOG to a file path to record every protocol line in both
directions. That is the only reliable way to see what a GUI actually sent when
it misbehaves.

This is not the referee. A crash or an illegal move here does not lose the
game: when the position has a legal move the engine plays one and says so with
an info string, because a GUI given no move just hangs. harness/referee.py, not
this file, is how the platform would rule your agent.
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import chess

from harness.rules import INIT_BUDGET_S
from harness.sandbox import Agent, AgentFailure, local

DEFAULT_BUDGET_MS = 60_000
NO_MOVE = "0000"
# a search started with either of these must not answer until stop or ponderhit
HELD = frozenset({"infinite", "ponder"})
LOG_PATH = os.environ.get("AICHESS_UCI_LOG", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an agent directory over UCI.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--name", default="")
    arguments = parser.parse_args()
    directory = arguments.agent.resolve()
    Engine(directory, arguments.name or f"aichessathon {directory.name}").run()


class Engine:
    """A UCI front end for one agent directory."""

    def __init__(self, directory: Path, name: str) -> None:
        self.directory = directory
        self.name = name
        self.board = chess.Board()
        self.agent: Agent | None = None
        self.pending: str | None = None

    def run(self) -> None:
        _log("--", f"engine up, serving {self.directory}")
        for line in sys.stdin:
            _log(">>", line.rstrip())
            command, _, rest = line.strip().partition(" ")
            if command == "uci":
                self._identify()
            elif command == "isready":
                _send("readyok")
            elif command == "ucinewgame":
                self._shutdown()
                self.board = chess.Board()
                self.pending = None
            elif command == "position":
                self._position(rest)
            elif command == "go":
                self._go(rest)
            elif command in {"stop", "ponderhit"}:
                self._flush()
            elif command == "quit":
                break
        self._shutdown()
        _log("--", "engine down")

    def _identify(self) -> None:
        _send(f"id name {self.name}")
        _send("id author aichessathon-starter")
        _send("uciok")

    def _position(self, rest: str) -> None:
        tokens = rest.split()
        if not tokens:
            return
        if "moves" in tokens:
            split = tokens.index("moves")
            head, moves = tokens[:split], tokens[split + 1 :]
        else:
            head, moves = tokens, []
        if head[0] == "startpos":
            self.board = chess.Board()
        elif head[0] == "fen":
            self.board = chess.Board(" ".join(head[1:]))
        else:
            return
        for uci in moves:
            self.board.push(chess.Move.from_uci(uci))

    def _go(self, rest: str) -> None:
        best = self._best(_budget_ms(rest, self.board.turn))
        if HELD.isdisjoint(rest.split()):
            _send(f"bestmove {best}")
        else:
            self.pending = best

    def _flush(self) -> None:
        if self.pending is None:
            return
        pending, self.pending = self.pending, None
        _send(f"bestmove {pending}")

    def _best(self, budget_ms: int) -> str:
        try:
            uci = self._running().move(self.board.fen(), budget_ms)
        except AgentFailure as failure:
            self._shutdown()
            return self._fallback(f"the agent failed to {failure.reason}")
        move = _legal(self.board, uci)
        if move is None:
            return self._fallback(f"the agent returned an illegal move: {uci!r}")
        return move.uci()

    def _fallback(self, reason: str) -> str:
        _send(f"info string {reason}")
        legal = next(iter(self.board.legal_moves), None)
        if legal is None:
            _send("info string the position is already over, so there is no move to make")
            return NO_MOVE
        _send(f"info string playing {legal.uci()} so the game can continue")
        return legal.uci()

    def _running(self) -> Agent:
        if self.agent is None:
            agent = local(self.directory)
            agent.start(INIT_BUDGET_S)
            self.agent = agent
        return self.agent

    def _shutdown(self) -> None:
        if self.agent is None:
            return
        agent, self.agent = self.agent, None
        agent.stop()
        for entry in agent.stderr_tail.rstrip().splitlines():
            _send(f"info string {entry}")


def _budget_ms(rest: str, turn: chess.Color) -> int:
    tokens = rest.split()
    values = {
        name: int(value)
        for name, value in itertools.pairwise(tokens)
        if value.lstrip("-").isdigit()
    }
    if "movetime" in values:
        return max(values["movetime"], 1)
    clock = values.get("wtime" if turn == chess.WHITE else "btime")
    return DEFAULT_BUDGET_MS if clock is None else max(clock, 1)


def _legal(board: chess.Board, uci: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except chess.InvalidMoveError:
        return None
    return move if move in board.legal_moves else None


def _send(message: str) -> None:
    _log("<<", message)
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _log(direction: str, message: str) -> None:
    if not LOG_PATH:
        return
    with open(LOG_PATH, "a", encoding="utf-8") as stream:
        stream.write(f"{time.time():.3f} {direction} {message}\n")


if __name__ == "__main__":
    main()
