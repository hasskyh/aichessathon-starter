import argparse
import contextlib
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, Outcome, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100

Played = tuple[int, bool, Outcome]


@dataclass(frozen=True)
class Game:
    """One scheduled game. Must stay picklable to cross into a worker process."""

    index: int
    agent: Path
    opponent: Path
    plays_white: bool
    base_ms: int
    increment_ms: int
    ply_cap: int


def play(game: Game) -> Played:
    white, black = (
        (game.agent, game.opponent) if game.plays_white else (game.opponent, game.agent)
    )
    outcome = play_match(
        local(white), local(black), game.base_ms, game.increment_ms, ply_cap=game.ply_cap
    )
    return game.index, game.plays_white, outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an agent over several games.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/greedy"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="games to run at once; the clock is wall clock, so never exceed your core count",
    )
    arguments = parser.parse_args()

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    games = [
        Game(
            index=index,
            agent=agent,
            opponent=opponent,
            plays_white=index % 2 == 0,
            base_ms=arguments.base_ms,
            increment_ms=arguments.increment_ms,
            ply_cap=arguments.ply_cap,
        )
        for index in range(arguments.games)
    ]

    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    with contextlib.ExitStack() as stack:
        results = _schedule(stack, games, _capped(arguments.concurrency))
        for index, plays_white, outcome in results:
            terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
            if outcome.result in {"draw", "void"}:
                draws += 1
            elif (outcome.result == "white") == plays_white:
                wins += 1
            else:
                losses += 1
            print(f"game {index + 1}/{arguments.games}: {outcome.result} by {outcome.termination}")

    score = (wins + draws / 2) / arguments.games
    print(f"\n{arguments.agent} vs {arguments.opponent} over {arguments.games} games")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print("terminations: " + ", ".join(f"{name} {count}" for name, count in terminations.items()))
    broken = {name: count for name, count in terminations.items() if name in FAILED_TERMINATIONS}
    if broken:
        raise SystemExit(
            "your agent failed to finish a game: "
            + ", ".join(f"{name} {count}" for name, count in broken.items())
        )


def _schedule(
    stack: contextlib.ExitStack, games: list[Game], concurrency: int
) -> Iterator[Played]:
    """Results in scheduling order either way, so the printed log stays readable."""
    if concurrency == 1:
        return map(play, games)
    pool = stack.enter_context(ProcessPoolExecutor(max_workers=concurrency))
    return pool.map(play, games)


def _capped(requested: int) -> int:
    """The referee times agents by wall clock, so oversubscribing makes them flag."""
    cores = os.cpu_count() or 1
    if requested <= 1:
        return 1
    if requested > cores:
        print(f"concurrency {requested} exceeds {cores} cores; capping to avoid false flag losses")
        return cores
    return requested


if __name__ == "__main__":
    main()
