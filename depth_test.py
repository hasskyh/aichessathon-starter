"""Run a fixed set of positions through one baseline's agent.get_move() at a fixed
time budget each, reporting depth reached / nodes searched / nps / chosen move.

Run once per baseline (as a separate process each time) rather than importing both
in one process -- every baseline's module is literally named agent.py, so only one
can live in sys.modules at a time; subprocess isolation sidesteps that entirely
and also matches how this file's own module-level state (TT, history tables,
GAME_HISTORY) should never leak between the two runs being compared anyway.

    uv run python depth_test.py baselines/invictus/invictus_v9_material
"""
import sys
import time
from pathlib import Path

baseline_dir = sys.argv[1]
sys.path.insert(0, str(Path(baseline_dir).resolve()))
import agent  # noqa: E402

TIME_MS = 5000 * 25  # get_move divides time_left_ms by ~25 to budget this move

POSITIONS = [
    ("start position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("italian-ish middlegame", "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6"),
    ("open tactical middlegame", "r2q1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/1PN1PN2/PB3PPP/R2Q1RK1 w - - 0 10"),
    ("rook endgame", "8/5pk1/6p1/7p/7P/6P1/5PK1/3r4 w - - 0 1"),
    ("queen endgame", "6k1/5ppp/8/8/3Q4/8/5PPP/6K1 w - - 0 1"),
]

print(f"baseline: {baseline_dir}")
print(f"{'position':30s} {'move':6s} {'depth':>6s} {'nodes':>10s} {'nps':>10s} {'time_s':>8s}")
for label, fen in POSITIONS:
    t0 = time.monotonic()
    move = agent.get_move(fen, TIME_MS)
    elapsed = time.monotonic() - t0
    depth = agent.last_depth
    nodes = agent.last_nodes
    nps = nodes / elapsed if elapsed > 0 else 0
    print(f"{label:30s} {move:6s} {depth:6d} {nodes:10.0f} {nps:10.0f} {elapsed:8.2f}")
