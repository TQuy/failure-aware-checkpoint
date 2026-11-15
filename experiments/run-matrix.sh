#!/usr/bin/env bash
# Run a set of cells on one GPU machine, in one boot, then power off.
#
#   SEED=1 MTBF=3600 DURATION=7200 CELLS="ref checkfreq:0.01 dcp:37" \
#     nohup bash ~/facp/experiments/run-matrix.sh > ~/matrix.log 2>&1 &
#
# Every cell with kills uses the same SEED and MTBF, so all cells in one
# matrix are killed at the same moments. A thrown-away warm-up runs first,
# because the first steps of a boot run faster than the rest.
set -uo pipefail

SEED=${SEED:-1}
MTBF=${MTBF:-3600}
DURATION=${DURATION:-7200}
REF_DURATION=${REF_DURATION:-900}
WARMUP_DURATION=${WARMUP_DURATION:-300}
CELLS=${CELLS:?cells}
MODEL=${MODEL:-gpt2-large}
BATCH=${BATCH:-4}
SEQ=${SEQ:-512}
SHUTDOWN=${SHUTDOWN:-1}
OUT=${OUT:-$HOME/results/seed$SEED}
STORE=${STORE:-/mnt/store}

PY=${PY:-python3}
cd "$HOME/facp" || exit 1
mkdir -p "$OUT"
echo "matrix seed=$SEED mtbf=$MTBF duration=$DURATION cells=$CELLS model=$MODEL batch=$BATCH seq=$SEQ" > "$OUT/matrix.txt"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv >> "$OUT/matrix.txt"

# host and GPU memory every 10 s, for the memory check
( echo "time,host_used_mb,host_available_mb,gpu_used_mb"
  while sleep 10; do
    read -r used avail < <(free -m | awk '/^Mem:/ {print $3, $7}')
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "$(date +%s),$used,$avail,$gpu"
  done ) > "$OUT/memory.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

run_cell() {  # name policy duration mtbf
  local name=$1 policy=$2 duration=$3 mtbf=$4 dir="$OUT/$1"
  mkdir -p "$dir"
  rm -rf "${STORE:?}/ckpt-$name"
  echo "started: $(date -Is)" > "$dir/cell.txt"
  $PY -m facp.supervisor --duration "$duration" --mtbf "$mtbf" --seed "$SEED" \
    --record "$dir/supervisor.jsonl" -- \
    $PY examples/train_gpt2.py --model "$MODEL" --policy "$policy" --grad-ckpt \
      --batch "$BATCH" --seq "$SEQ" --seed 0 --data-dir "$HOME/data" \
      --ckpt-dir "$STORE/ckpt-$name" --record "$dir/train.jsonl" > "$dir/train.log" 2>&1
  $PY -m facp.accounting "$dir" > "$dir/accounting.json" 2> "$dir/accounting.err"
  rm -rf "${STORE:?}/ckpt-$name"
  echo "finished: $(date -Is)" >> "$dir/cell.txt"
}

run_cell warmup none "$WARMUP_DURATION" 0
for cell in $CELLS; do
  if [[ $cell == ref ]]; then
    run_cell ref none "$REF_DURATION" 0
  else
    run_cell "${cell//[:.]/-}" "$cell" "$DURATION" "$MTBF"
  fi
done
echo "done: $(date -Is)" >> "$OUT/matrix.txt"
[[ $SHUTDOWN == 1 ]] && sudo shutdown -h +2
exit 0
