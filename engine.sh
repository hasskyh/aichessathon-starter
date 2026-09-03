#!/usr/bin/env sh
# Launch an agent directory as a UCI engine, for a chess GUI to load.
#
#   ./engine.sh                     your agent.py
#   ./engine.sh baselines/minimax   a baseline
#
# This execs the project venv directly instead of going through `uv run`, so a
# GUI never blocks on dependency resolution or a download while it waits for the
# engine to answer. Run `uv sync` yourself after changing pyproject.toml.
set -e
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root"
PY=$root/.venv/bin/python
if [ ! -x "$PY" ]; then
    echo "no interpreter at $PY -- run 'uv sync' first" >&2
    exit 1
fi
exec "$PY" -m harness.uci --agent "${1:-.}"
