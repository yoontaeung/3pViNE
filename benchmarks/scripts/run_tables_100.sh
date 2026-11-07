#!/usr/bin/env bash
# Run sequentially to avoid cross-experiment CPU contention.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_DIR="${1:?Usage: bash benchmarks/scripts/run_tables_100.sh NEW_RUN_DIRECTORY}"
mkdir -p "$RUN_DIR"
exec 9>"$ROOT/artifacts/benchmarks/tables_100.lock"
flock -n 9 || { echo "Another table repetition batch is running"; exit 1; }
date -u > "$RUN_DIR/started.txt"
git diff --binary HEAD > "$RUN_DIR/source_changes.patch"
docker images --no-trunc > "$RUN_DIR/docker_images.txt"
lscpu > "$RUN_DIR/cpu.txt"

set +e
timeout --signal=TERM --kill-after=120s 24h \
  .venv/bin/python -u -m benchmarks.compose_protocol_cost.run_end_to_end \
  --run-dir "$RUN_DIR/table2" --repetitions 100 --warmups 1 \
  > "$RUN_DIR/table2.log" 2>&1
TABLE2_STATUS=$?
echo "$TABLE2_STATUS" > "$RUN_DIR/table2.exit"

docker run --rm --name "zkvne-table3-100-$$" --network none \
  --user "$(id -u):$(id -g)" \
  --env NODE_OPTIONS=--max-old-space-size=16384 \
  --env PYTHONUNBUFFERED=1 --env MPLCONFIGDIR=/tmp/matplotlib \
  -v "$ROOT:/workspace" -w /workspace zk-vne-zk:local \
  timeout --signal=TERM --kill-after=60s 12h \
  /opt/zk-vne-venv/bin/python benchmarks/zk_overhead/topology_run.py \
  --run-dir "$RUN_DIR/table3" --profile vn3_sn32_h3 \
  --repetitions 100 --warmups 1 > "$RUN_DIR/table3.log" 2>&1
TABLE3_STATUS=$?
echo "$TABLE3_STATUS" > "$RUN_DIR/table3.exit"
date -u > "$RUN_DIR/finished.txt"
echo "Table 2 exit=$TABLE2_STATUS; Table 3 exit=$TABLE3_STATUS"
test "$TABLE2_STATUS" -eq 0 && test "$TABLE3_STATUS" -eq 0
