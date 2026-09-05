#!/usr/bin/env bash
# Play two agent directories against each other and print an Elo estimate.
#
#   ./compare.sh . baselines/invictus/invictus_ab
#   ./compare.sh . baselines/minimax 100 120+0.5
#   CONCURRENCY=4 ./compare.sh baselines/greedy baselines/random 200
#
# Defaults: 40 games at 120s+0.1s, one game per core. Both agent.py files are
# syntax checked first, because a typo costs a second here and a whole match later.
#
# Openings are drawn from openings/suite.pgn (override with OPENINGS=path), 16 plies
# deep, in random order; -repeat plays each chosen opening twice with colors swapped,
# so a result reflects engine strength rather than which side got the better of one
# fixed, deterministically-replayed line.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"

if [ $# -lt 2 ]; then
    echo "usage: $(basename "$0") <agent-dir-A> <agent-dir-B> [games] [tc]" >&2
    exit 2
fi

a=${1%/}
b=${2%/}
games=${3:-40}
tc=${4:-120+0.1}
concurrency=${CONCURRENCY:-$(nproc)}
openings=${OPENINGS:-$root/openings/suite.pgn}

cli=$(command -v cutechess-cli 2>/dev/null || true)
if [ -z "$cli" ]; then
    cli=$HOME/cutechess/build/cutechess-cli
fi
if [ ! -x "$cli" ]; then
    echo "cutechess-cli not found (tried PATH and $HOME/cutechess/build)" >&2
    exit 1
fi

py=$root/.venv/bin/python
for dir in "$a" "$b"; do
    if [ ! -f "$dir/agent.py" ]; then
        echo "no agent.py in $dir" >&2
        exit 1
    fi
    if ! err=$("$py" -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' "$dir/agent.py" 2>&1); then
        echo "$dir/agent.py does not parse:" >&2
        printf '%s\n' "$err" | sed 's/^/  /' >&2
        exit 1
    fi
done

label() { basename "$(cd "$1" && pwd)"; }
na=$(label "$a")
nb=$(label "$b")
if [ "$na" = "$nb" ]; then
    nb="$nb-2"
fi

mkdir -p "$root/logs"
pgn=$root/logs/compare-$na-vs-$nb.pgn
echo "$na vs $nb   |   $games games   |   tc=$tc   |   concurrency=$concurrency"
echo
"$cli" \
    -engine name="$na" cmd=./engine.sh arg="$a" proto=uci \
    -engine name="$nb" cmd=./engine.sh arg="$b" proto=uci \
    -each dir="$root" tc="$tc" \
    -openings file="$openings" format=pgn order=random plies=16 \
    -repeat \
    -games "$games" -concurrency "$concurrency" -recover \
    -pgnout "$pgn"
echo
echo "pgn saved to $pgn"
