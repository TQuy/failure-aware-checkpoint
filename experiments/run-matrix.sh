#!/usr/bin/env bash
# Run a set of cells on one GPU machine, in one boot, then power off.
#
#   SEED=1 CELLS="ref checkfreq:0.01@900 dcp:64@900 checkfreq:0.01@3600" \
#     nohup bash ~/facp/experiments/run-matrix.sh > ~/matrix.log 2>&1 &
#
# A cell is POLICY or POLICY@MTBF (seconds; MTBF falls back to $MTBF). Cells
# with the same SEED and MTBF are killed at the same moments, so they form a
# paired group. A thrown-away warm-up runs first, because the first steps of
# a boot run faster than the rest.
#
# RESUME=1 (after a spot stop) keeps every failure-rate group whose cells all
# finished in one earlier boot, and reruns every other group in full on this
# boot, so the cells compared with each other always share a boot. Results go
# to $OUT and, after each cell, to $STORE/results-seed$SEED on the storage
# machine (results-<name of OUT>), which is not a spot machine and can be read while this one is off.
set -uo pipefail

# a resume reads back the settings of the first launch, so it runs the same matrix
OUT=${OUT:-$HOME/results/seed${SEED:-1}}
if [[ ${RESUME:-0} == 1 && -f $OUT/settings.env ]]; then
  # shellcheck disable=SC1091
  source "$OUT/settings.env"
fi
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
RESUME=${RESUME:-0}
STORE=${STORE:-/mnt/store}
MIRROR=${MIRROR:-$STORE/results-$(basename "$OUT")}

PY=${PY:-python3}
BOOT=$(cat /proc/sys/kernel/random/boot_id)
cd "$HOME/facp" || exit 1
mkdir -p "$OUT"

mirror() {  # copy the results to the storage machine (no rsync on the image)
  mkdir -p "$MIRROR" && cp -a "$OUT/." "$MIRROR/" 2>/dev/null || true
}

if [[ $RESUME == 1 ]]; then
  echo "resume boot=$BOOT at $(date -Is)" >> "$OUT/matrix.txt"
else
  echo "matrix seed=$SEED duration=$DURATION cells=$CELLS model=$MODEL batch=$BATCH seq=$SEQ" > "$OUT/matrix.txt"
  echo "boot=$BOOT at $(date -Is)" >> "$OUT/matrix.txt"
  echo "$CELLS" > "$OUT/cells.txt"
  for v in SEED MTBF DURATION REF_DURATION WARMUP_DURATION CELLS MODEL BATCH SEQ; do
    printf '%s=%q\n' "$v" "${!v}"
  done > "$OUT/settings.env"         # read back by a resume
fi
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv >> "$OUT/matrix.txt"

# host and GPU memory every 10 s, for the memory check
( echo "time,host_used_mb,host_available_mb,gpu_used_mb"
  while sleep 10; do
    read -r used avail < <(free -m | awk '/^Mem:/ {print $3, $7}')
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "$(date +%s),$used,$avail,$gpu"
  done ) >> "$OUT/memory-$BOOT.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

run_cell() {  # dir policy duration mtbf
  local dir=$1 policy=$2 duration=$3 mtbf=$4 ck
  ck="$STORE/ckpt-${dir//\//-}"
  mkdir -p "$dir"
  rm -rf "${ck:?}"
  { echo "boot: $BOOT"; echo "started: $(date -Is)"; } > "$dir/cell.txt"
  $PY -m facp.supervisor --duration "$duration" --mtbf "$mtbf" --seed "$SEED" \
    --record "$dir/supervisor.jsonl" -- \
    $PY examples/train_gpt2.py --model "$MODEL" --policy "$policy" --grad-ckpt \
      --batch "$BATCH" --seq "$SEQ" --seed 0 --data-dir "$HOME/data" \
      --ckpt-dir "$ck" --record "$dir/train.jsonl" > "$dir/train.log" 2>&1
  $PY -m facp.accounting "$dir" > "$dir/accounting.json" 2> "$dir/accounting.err"
  rm -rf "${ck:?}"
  echo "finished: $(date -Is)" >> "$dir/cell.txt"
  mirror
}

cell_mtbf() { [[ $1 == *@* ]] && echo "${1##*@}" || echo "$MTBF"; }
cell_dir() { local p=${1%@*}; echo "$OUT/m$(cell_mtbf "$1")/${p//[:.]/-}"; }

# a group is kept on resume only if all its cells finished, all in one boot
group_done() {  # mtbf
  local cell boots=""
  for cell in $CELLS; do
    [[ $cell == ref ]] && continue
    [[ $(cell_mtbf "$cell") == "$1" ]] || continue
    grep -q '^finished:' "$(cell_dir "$cell")/cell.txt" 2>/dev/null || return 1
    boots+="$(sed -n 's/^boot: //p' "$(cell_dir "$cell")/cell.txt")"$'\n'
  done
  [[ $(sort -u <<< "$boots" | grep -c .) == 1 ]]
}

run_cell "$OUT/warmup-$BOOT" none "$WARMUP_DURATION" 0
declare -A redo=()
for cell in $CELLS; do
  [[ $cell == ref ]] && { run_cell "$OUT/ref-$BOOT" none "$REF_DURATION" 0; continue; }
  m=$(cell_mtbf "$cell")
  if [[ -z ${redo[$m]+x} ]]; then
    if [[ $RESUME == 1 ]] && group_done "$m"; then redo[$m]=0; else redo[$m]=1; fi
    # an unfinished group from an earlier boot is set aside, not mixed in
    if [[ ${redo[$m]} == 1 ]]; then
      stamp=$(date +%s)
      [[ -d $OUT/m$m ]] && mv "$OUT/m$m" "$OUT/m$m-aborted-$stamp"
      [[ -d $MIRROR/m$m ]] && mv "$MIRROR/m$m" "$MIRROR/m$m-aborted-$stamp"
    fi
    echo "group mtbf=$m: $([[ ${redo[$m]} == 1 ]] && echo run || echo kept)" >> "$OUT/matrix.txt"
  fi
  [[ ${redo[$m]} == 1 ]] && run_cell "$(cell_dir "$cell")" "${cell%@*}" "$DURATION" "$m"
done
echo "done: $(date -Is)" >> "$OUT/matrix.txt"
mirror
[[ $SHUTDOWN == 1 ]] && sudo shutdown -h +2
exit 0
