"""Baseline movegen throughput: what python-chess costs the search today."""

import time

import chess

POSITIONS = [
    ("startpos", chess.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("midgame", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/2NP1N2/PPPQ1PPP/R4RK1 w - - 0 10"),
]


def perft_legal(board: chess.Board, depth: int) -> int:
    """Exactly the shape of the agent's search: legal_moves then push/pop."""
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += perft_legal(board, depth - 1)
        board.pop()
    return total


def perft_pseudo(board: chess.Board, depth: int) -> int:
    """pseudo_legal_moves + is_legal filter."""
    if depth == 0:
        return 1
    total = 0
    for move in board.pseudo_legal_moves:
        if not board.is_legal(move):
            continue
        board.push(move)
        total += perft_pseudo(board, depth - 1)
        board.pop()
    return total


def perft_pseudo_nofilter(board: chess.Board, depth: int) -> int:
    """pseudo_legal_moves, illegality detected only by the child capturing a king.

    Node counts differ from perft; this is the cheapest possible python-chess loop.
    """
    if depth == 0:
        return 1
    total = 0
    for move in board.pseudo_legal_moves:
        board.push(move)
        total += perft_pseudo_nofilter(board, depth - 1)
        board.pop()
    return total


def gen_only_legal(board: chess.Board, reps: int) -> int:
    n = 0
    for _ in range(reps):
        n += len(list(board.legal_moves))
    return n


def gen_only_pseudo(board: chess.Board, reps: int) -> int:
    n = 0
    for _ in range(reps):
        n += len(list(board.pseudo_legal_moves))
    return n


def timed(fn, *args):
    t = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t


DEPTH = 4

print(f"--- perft(depth={DEPTH}): full tree walk, nodes/sec is what matters ---")
for name, fen in POSITIONS:
    board = chess.Board(fen)
    nodes, dt = timed(perft_legal, board, DEPTH)
    print(f"{name:10s} legal        {nodes:>10,} nodes {dt:7.3f}s {nodes / dt / 1000:9.1f}k nps")
    board = chess.Board(fen)
    nodes_p, dt_p = timed(perft_pseudo, board, DEPTH)
    rate = nodes_p / dt_p / 1000
    print(f"{name:10s} pseudo+legal {nodes_p:>10,} nodes {dt_p:7.3f}s {rate:9.1f}k nps")
    board = chess.Board(fen)
    nodes_n, dt_n = timed(perft_pseudo_nofilter, board, DEPTH)
    rate = nodes_n / dt_n / 1000
    print(f"{name:10s} pseudo raw   {nodes_n:>10,} nodes {dt_n:7.3f}s {rate:9.1f}k nps")
    print()

print("--- generation only, no push/pop (200k reps) ---")
for name, fen in POSITIONS[:2]:
    board = chess.Board(fen)
    _, dt = timed(gen_only_legal, board, 200_000)
    print(f"{name:10s} legal_moves         {200_000 / dt / 1000:9.1f}k gens/sec")
    _, dt = timed(gen_only_pseudo, board, 200_000)
    print(f"{name:10s} pseudo_legal_moves  {200_000 / dt / 1000:9.1f}k gens/sec")
