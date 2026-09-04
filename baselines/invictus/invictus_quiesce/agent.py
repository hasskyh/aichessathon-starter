# invictus_quiesce, with the deadline fix (see git history): the original assigns a local
# __deadline and leaves the module-level _deadline at 0.0, so _tick raises Timeout on
# node 2048 of every move. This copy exists so a speed comparison has an honest target.

import math
import time

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
MOBILITY_WEIGHT = 4
MATE = 1e6
CAPTURE_VALUE = dict(PIECE_VALUE)
CAPTURE_VALUE[chess.KING] = 0.0
_deadline = 0.0
_nodes = 0

class Timeout(Exception):
    pass

def _tick() -> None:
    global _nodes
    _nodes += 1
    if _nodes % 2048 == 0 and time.monotonic() > _deadline:
        raise Timeout

def evaluate(board: chess.Board, mobility: int) -> float:
    mover = board.turn
    material = sum(
        value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
        for piece, value in PIECE_VALUE.items()
    )
    return material + MOBILITY_WEIGHT * mobility

def quiescence(alpha: float, beta: float, board: chess.Board, qdepth: int) -> float:
    _tick()

    if not any(board.legal_moves):
        return -MATE if board.is_check() else 0.0

    moves = list(board.legal_moves)
    if board.is_check():
        best_val = -math.inf
    else:
        best_val = evaluate(board, len(moves))
        if best_val >= beta or qdepth == 0:
            return best_val
        if best_val > alpha:
            alpha = best_val
        moves = sorted(
            board.generate_legal_captures(),
            key = lambda m: -CAPTURE_VALUE[
                (board.piece_at(m.to_square) or chess.Piece(chess.PAWN, True)).piece_type
            ],
        )

    for move in moves:
        board.push(move)
        score = -quiescence(-beta, -alpha, board, qdepth - 1)
        board.pop()

        if score >= beta:
            return score
        if score > best_val:
            best_val = score
        if score > alpha:
            alpha = score
    return best_val

def negamax(alpha: float, beta: float, board: chess.Board, depth: int) -> float:
    _tick()
    if not any(board.legal_moves):
        return -MATE if board.is_check() else 0.0
    if depth == 0:
        # Quiescence search
        return quiescence(alpha, beta, board, 6)
                
    best_score = -MATE
    for move in sorted(board.legal_moves, key=lambda m: -CAPTURE_VALUE[
        (board.piece_at(m.to_square) or chess.Piece(chess.PAWN, True)).piece_type]):
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

def _pv_first(moves: list[chess.Move], pv: chess.Move | None) -> list[chess.Move]:
    if pv is None or pv not in moves:
        return moves
    return [pv] + [m for m in moves if m != pv]


def get_move(fen: str, time_left_ms: int) -> str:
    global _deadline, _nodes
    board = chess.Board(fen)
    _deadline = time.monotonic() + max(time_left_ms / 25_000.0, 0.01)
    _nodes = 0

    moves = list(board.legal_moves)
    pv_move = moves[0]
    depth = 1
    while True:
        best_move: chess.Move | None = None
        best_score = -math.inf
        try:
            for move in _pv_first(moves, pv_move):
                board.push(move)
                # Every root move must be searched with a full (-inf, +inf) window.
                # The previous version passed (-inf, -best_score), which narrows beta
                # as best_score improves through the move loop. A move that then hits
                # that narrowed beta returns early with a fail-high bound -- "at least
                # this good", not its true score -- and that bound was being compared
                # directly against other moves' *exact* scores as if it were one too.
                # A late move could look worse than it truly is (or a bad move look
                # better) purely because of when it happened to be searched, not its
                # actual merit -- observed directly as the same position scoring
                # +100 / -144 / +104 at consecutive depths instead of converging.
                score = -negamax(-math.inf, math.inf, board, depth - 1)
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
        except Timeout:
            break
        if best_move is not None:
            pv_move = best_move
        depth += 1
        if time.monotonic() > _deadline:
            break
    return pv_move.uci()
