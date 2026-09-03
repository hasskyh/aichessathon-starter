"""Correctness gate for bitgen.

Three layers, weakest to strongest:

1. Published perft counts. Cheap, and a wrong total is unmissable.
2. perft divide against python-chess. Same coverage, but it names the move that is
   wrong instead of just the total, which is the difference between a five minute fix
   and an afternoon.
3. Random walks from every suite position, comparing move lists, check status,
   castling rights and the ep square at every node. This is the layer that catches
   what nobody wrote a perft number for.

Run it after any change to generation or make/unmake. It is the only thing standing
between a fast generator and a fast generator that loses on an illegal move.
"""

import random
import sys

import chess

import bitgen

# name, fen, node counts by depth. Every literal here was cross-checked against
# python-chess, because a wrong expectation costs more time than a wrong generator.
SUITE = [
    ("startpos", bitgen.STARTING_FEN, [20, 400, 8902, 197281, 4865609]),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        [48, 2039, 97862, 4085603],
    ),
    ("pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238, 674624]),
    (
        "pos4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        [6, 264, 9467, 422333],
    ),
    (
        "pos4-mirror",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        [6, 264, 9467, 422333],
    ),
    (
        "pos5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        [44, 1486, 62379, 2103487],
    ),
    (
        "midgame",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/2NP1N2/PPPQ1PPP/R4RK1 w - - 0 10",
        [45, 1764, 76031, 2854377],
    ),
    # En passant that is illegal because the capturing pawn is pinned along the rank.
    ("ep-pin", "8/8/8/8/k1p4R/8/3P4/3K4 b - - 0 1", [5, 89, 555, 10094]),
    # Promotion with the promoted piece immediately giving or blocking check.
    ("promo-check", "8/1P6/8/8/8/8/k6K/8 w - - 0 1", [9, 39, 428, 2062]),
    # Castling rights present but both sides of the king can be attacked through.
    ("castle-thru", "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", [26, 112, 3189, 17945]),
]

buf, hist = bitgen.new_buffers()
failures = 0


def ref_perft(board, depth):
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += ref_perft(board, depth - 1)
        board.pop()
    return total


print("--- perft totals ---")
for name, fen, expected in SUITE:
    for depth, want in enumerate(expected, start=1):
        bb, sq, st = bitgen.from_fen(fen)
        got = bitgen.perft(bb, sq, st, depth, buf, hist, 0)
        if got == want:
            print(f"ok   {name:12s} depth {depth}  {got:>10,}")
        else:
            failures += 1
            print(f"FAIL {name:12s} depth {depth}  got {got:>10,}  want {want:>10,}")
            break

print("\n--- perft divide vs python-chess, depth 3 ---")
for name, fen, _ in SUITE:
    bb, sq, st = bitgen.from_fen(fen)
    n = bitgen.gen_moves(bb, st, buf[0], 0)
    mine = {}
    for i in range(n):
        move = int(buf[0, i])
        bitgen.make_move(bb, sq, st, move, hist, 0)
        mine[bitgen.to_uci(move)] = bitgen.perft(bb, sq, st, 2, buf, hist, 1)
        bitgen.unmake_move(bb, sq, st, move, hist, 0)

    board = chess.Board(fen)
    theirs = {}
    for move in board.legal_moves:
        board.push(move)
        theirs[move.uci()] = ref_perft(board, 2)
        board.pop()

    if mine == theirs:
        print(f"ok   {name:12s} {len(mine):>3} moves, {sum(mine.values()):>8,} nodes")
    else:
        failures += 1
        print(f"FAIL {name:12s}")
        for key in sorted(set(mine) | set(theirs)):
            got, want = mine.get(key), theirs.get(key)
            if got != want:
                print(f"       {key}: bitgen {got}, python-chess {want}")

print("\n--- random walks vs python-chess ---")
rng = random.Random(20260902)
mismatch = None
positions = 0
for _name, fen, _ in SUITE:
    for _ in range(60):
        board = chess.Board(fen)
        bb, sq, st = bitgen.from_fen(fen)
        for _ in range(120):
            mine = sorted(bitgen.legal_ucis(bb, st))
            theirs = sorted(m.uci() for m in board.legal_moves)
            positions += 1
            if mine != theirs:
                mismatch = (
                    board.fen(),
                    f"missing {sorted(set(theirs) - set(mine))}"
                    f" extra {sorted(set(mine) - set(theirs))}",
                )
                break
            if bool(bitgen.in_check(bb, st)) != board.is_check():
                mismatch = (board.fen(), "in_check disagrees with python-chess")
                break
            want_rights = (
                (1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
                | (2 if board.has_queenside_castling_rights(chess.WHITE) else 0)
                | (4 if board.has_kingside_castling_rights(chess.BLACK) else 0)
                | (8 if board.has_queenside_castling_rights(chess.BLACK) else 0)
            )
            if st[1] != want_rights:
                mismatch = (board.fen(), f"rights {st[1]} want {want_rights}")
                break
            if not mine or board.is_insufficient_material():
                break
            pick = rng.choice(mine)
            n = bitgen.gen_moves(bb, st, buf[0], 0)
            packed = next(
                int(buf[0, i]) for i in range(n) if bitgen.to_uci(buf[0, i]) == pick
            )
            bitgen.make_move(bb, sq, st, packed, hist, 0)
            board.push(chess.Move.from_uci(pick))
        if mismatch:
            break
    if mismatch:
        break

if mismatch:
    failures += 1
    print(f"FAIL at {mismatch[0]}\n     {mismatch[1]}")
else:
    print(f"ok   {positions:,} positions matched python-chess exactly")

print("\n--- capture generation vs python-chess ---")
# Quiescence only ever sees this list, so a bug here is a bug the perft suite above
# cannot see: perft never calls gen_captures.
cap_bad = 0
cap_rng = random.Random(7)
cap_seen = 0
for _name, fen, _ in SUITE:
    for _ in range(40):
        board = chess.Board(fen)
        bb, sq, st = bitgen.from_fen(fen)
        for _ in range(60):
            mine = sorted(bitgen.capture_ucis(bb, st))
            theirs = sorted(m.uci() for m in board.generate_legal_captures())
            cap_seen += 1
            if mine != theirs:
                print(f"FAIL {board.fen()}")
                print(f"       bitgen       {mine}")
                print(f"       python-chess {theirs}")
                cap_bad += 1
                break
            moves = sorted(bitgen.legal_ucis(bb, st))
            if not moves:
                break
            pick = cap_rng.choice(moves)
            n = bitgen.gen_moves(bb, st, buf[0], 0)
            packed = next(
                int(buf[0, i]) for i in range(n) if bitgen.to_uci(buf[0, i]) == pick
            )
            bitgen.make_move(bb, sq, st, packed, hist, 0)
            board.push(chess.Move.from_uci(pick))
        if cap_bad:
            break
    if cap_bad:
        break
failures += cap_bad
if not cap_bad:
    print(f"ok   {cap_seen:,} positions agree on legal captures")

print("\n--- unmake restores the position bit for bit ---")
bad = 0
for _, fen, _ in SUITE:
    bb, sq, st = bitgen.from_fen(fen)
    before = (bb.copy(), sq.copy(), st.copy())
    n = bitgen.gen_moves(bb, st, buf[0], 0)
    for i in range(n):
        move = int(buf[0, i])
        bitgen.make_move(bb, sq, st, move, hist, 0)
        bitgen.unmake_move(bb, sq, st, move, hist, 0)
        same = (
            (bb == before[0]).all() and (sq == before[1]).all() and (st == before[2]).all()
        )
        if not same:
            print(f"FAIL {fen}  {bitgen.to_uci(move)} did not restore")
            bad += 1
failures += bad
if not bad:
    print("ok   every move round-trips")

print(f"\n{'FAILURES: ' + str(failures) if failures else 'all green'}")
sys.exit(1 if failures else 0)
