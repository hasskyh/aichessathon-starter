#!/usr/bin/env bash
# Fetch the entire Lichess eval database (21.68 GB compressed, a hard ceiling --
# there is no more where that came from), resuming automatically across any
# interruption within this session: a transient network failure, fetch_data.py
# being killed, anything short of the whole machine going down. A full shutdown
# kills this loop along with everything else in it -- relaunch this same script
# by hand when the machine is back, and it picks up exactly where it left off via
# --state/--bloom-state, the same as an in-session restart.
#
#   nohup bash fetch_overnight.sh > /tmp/fetch_overnight.log 2>&1 &
#
set -uo pipefail
cd ~/aichessathon-starter
export TERM=dumb

URL="https://database.lichess.org/lichess_db_eval.jsonl.zst"
OUT=data/raw_halfkp.jsonl
STATE=data/fetch_state.json
BLOOM=data/fetch_bloom.bin
LIMIT=2000000000  # effectively unlimited: the real stop is the source running out

while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S'): launching fetch segment ==="
    curl -sS "$URL" | zstd -dc | .venv/bin/python -m training.fetch_data \
        --out "$OUT" --append --limit "$LIMIT" \
        --state "$STATE" --bloom-state "$BLOOM" \
        --progress-every 5000000
    curl_status=${PIPESTATUS[0]}
    py_status=${PIPESTATUS[2]}
    kept_after=$(python3 -c "import json; print(json.load(open('$STATE'))['kept_total'])" 2>/dev/null || echo 0)
    echo "=== $(date '+%Y-%m-%d %H:%M:%S'): curl_status=$curl_status py_status=$py_status kept_total=$kept_after ==="

    if [ "$kept_after" -ge "$LIMIT" ]; then
        echo "=== reached the (effectively unlimited) --limit, done ==="
        break
    fi
    if [ "$curl_status" -eq 0 ] && [ "$py_status" -eq 0 ]; then
        # Both the download and the filter completed cleanly on their own --
        # every byte of the 21.68 GB source has been read, not just this segment
        # of it. There is nothing left to fetch.
        echo "=== source file fully consumed, nothing more to fetch, done ==="
        break
    fi
    echo "=== segment ended early (curl=$curl_status python=$py_status), retrying in 15s ==="
    sleep 15
done
echo "=== fetch_overnight.sh finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
