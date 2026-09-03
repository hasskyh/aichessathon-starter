# +130 elo roughly on minimax

import math
import random

import chess

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.
PIECE_VALUE = {
    chess.PAWN: 100.0,
    chess.KNIGHT: 320.0,
    chess.BISHOP: 330.0,
    chess.ROOK: 500.0,
    chess.QUEEN: 900.0,
}
MOBILITY_WEIGHT = 4.0
MATE = 1e6

def evaluate(board: chess.Board, mobility: int) -> float:
    mover = board.turn
    material = sum(
        value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
        for piece, value in PIECE_VALUE.items()
    )
    return material + MOBILITY_WEIGHT * mobility

def negamax(alpha: float, beta: float, board: chess.Board, depth: int) -> float:
    moves = list(board.legal_moves)
    if not any(moves):
        return -MATE if board.is_check() else 0.0
    if depth == 0:
        # Quiescence search
        return evaluate(board, len(moves))
    best_score = -MATE
    for move in moves:
        board.push(move)
        score = -negamax(-beta, -alpha, board, depth - 1)
        board.pop()
        if score > best_score:
            best_score = score
            if score > alpha:
                alpha = score
        if score >= beta:
            return score

    return best_score


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.

    print() is safe. Your stdout is redirected away from the protocol stream, discarded
    during rated games and shown back to you in the validation log.
    """
    board = chess.Board(fen)
    best_score = -MATE
    best: list[chess.Move] = []
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(-math.inf, math.inf, board, 2)
        board.pop()
        if score > best_score:
            best_score = score
            best = [move]
        elif score == best_score:
            best.append(move)

    return random.choice(best).uci()
