import random
import sys

import numpy as np

sys.path.insert(0, ".")
import bitgen
import zobrist as zb

random.seed(3)

MAX_PLY = 200
BUF = np.zeros((MAX_PLY, 256), dtype=np.int32)
HIST = np.zeros((MAX_PLY, 4), dtype=np.int64)

n_games = 200
n_moves = 0
n_mismatches = 0

for game_num in range(n_games):
    bb, sq, st = bitgen.from_fen(bitgen.STARTING_FEN)
    h = zb.zobrist_hash_bb(sq, st)

    fresh0 = zb.zobrist_hash_bb(sq, st)
    if h != fresh0:
        print(f"MISMATCH game {game_num} start")
        n_mismatches += 1

    for ply in range(150):
        count = bitgen.gen_moves(bb, st, BUF[ply], 0)
        if count == 0:
            break
        move = np.int32(BUF[ply, random.randrange(count)])

        delta = zb.zobrist_delta_bb(sq, st, move)
        bitgen.make_move(bb, sq, st, move, HIST, ply)
        h ^= delta

        fresh = zb.zobrist_hash_bb(sq, st)
        n_moves += 1
        if h != fresh:
            print(f"MISMATCH game {game_num} ply {ply} move {bitgen.to_uci(int(move))}: "
                  f"incremental={h} fresh={fresh}")
            n_mismatches += 1
            if n_mismatches > 5:
                break

        # also verify undo: XOR the same delta back in, unmake, and check the
        # PARENT's hash is restored exactly.
        bitgen.unmake_move(bb, sq, st, move, HIST, ply)
        h ^= delta
        fresh_parent = zb.zobrist_hash_bb(sq, st)
        if h != fresh_parent:
            print(f"MISMATCH (undo) game {game_num} ply {ply} move {bitgen.to_uci(int(move))}")
            n_mismatches += 1
            if n_mismatches > 5:
                break
        # redo the move for real to keep the game progressing
        bitgen.make_move(bb, sq, st, move, HIST, ply)
        h ^= delta
    if n_mismatches > 5:
        break

print(f"\n{n_moves} moves checked across {n_games} games, {n_mismatches} mismatches")

# collision sanity check: hash a large batch of distinct positions and confirm no
# accidental collisions in this sample (not a proof, but a basic sanity check)
seen = {}
collisions = 0
bb, sq, st = bitgen.from_fen(bitgen.STARTING_FEN)
h = zb.zobrist_hash_bb(sq, st)
seen[h] = "start"
for _ply in range(400):
    count = bitgen.gen_moves(bb, st, BUF[0], 0)
    if count == 0:
        bb, sq, st = bitgen.from_fen(bitgen.STARTING_FEN)
        h = zb.zobrist_hash_bb(sq, st)
        continue
    move = np.int32(BUF[0, random.randrange(count)])
    delta = zb.zobrist_delta_bb(sq, st, move)
    bitgen.make_move(bb, sq, st, move, HIST, 0)
    h ^= delta
    fen_key = f"{sq.tobytes()}{st[0]}{st[1]}{st[2]}"
    if h in seen and seen[h] != fen_key:
        collisions += 1
    seen[h] = fen_key
print(f"collision check: {len(seen)} distinct hashes stored, {collisions} suspected collisions")
