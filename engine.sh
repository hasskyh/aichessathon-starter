#!/usr/bin/env sh
# Launch an agent directory as a UCI engine, for a chess GUI to load.
#
# Point your GUI's engine command at this file. The optional first argument is
# the agent directory to serve, relative to the project root; it defaults to
# your own agent.
#
#   ./engine.sh                     your agent.py
#   ./engine.sh baselines/minimax   a baseline
set -e
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root"
# a GUI launched from the desktop may not have ~/.local/bin on its PATH
UV=${UV:-$(command -v uv || printf '%s' "$HOME/.local/bin/uv")}
exec "$UV" run python -m harness.uci --agent "${1:-.}"
