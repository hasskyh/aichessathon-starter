"""Serve an agent directory as a UCI engine, so any chess GUI can play it.

    uv run python -m harness.uci --agent .

The agent still runs inside the platform's runner, one process per game, so its
print() output cannot reach the UCI stream and its state is discarded on
ucinewgame exactly as it is between rated games.
"""

import argparse
import itertools
import sys
from pathlib import Path

import chess

from harness.rules import INIT_BUDGET_S
from harness.sandbox import Agent, AgentFailure, local

DEFAULT_BUDGET_MS = 60_000
NO_MOVE = "0000"
# a search started with either of these must not answer until stop or ponderhit
HELD = frozenset({"infinite", "ponder"})


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
        for line in sys.stdin:
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

    def _best(self, budget_ms: int) -> str:
        try:
            uci = self._running().move(self.board.fen(), budget_ms)
        except AgentFailure as failure:
            _send(f"info string the agent failed to {failure.reason}")
            self._shutdown()
            return NO_MOVE
        move = _legal(self.board, uci)
        if move is None:
            _send(f"info string the agent returned an illegal move: {uci!r}")
            return NO_MOVE
        return move.uci()

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
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
