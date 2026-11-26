#!/usr/bin/env bash
# Watch the Week 13 matrix on pairs 1-5 from a helper machine, every 10 min.
#
#   PROJECT=... GCLOUD=.../gcloud SSH_USER=<launching user> WORK=<folder> \
#     bash experiments/watch-w13.sh          # inside tmux
#
# For each pair it:
# - copies the results from the storage machine (not a spot machine) every
#   30 minutes, and once more when the matrix is done;
# - when the matrix is done: copies the results, stops the storage machine;
# - when the spot GPU machine was stopped early: restarts it and resumes the
#   matrix (unfinished failure-rate groups rerun in full on the new boot),
#   but only if the rerun ends before RESTART_BY, the pair has had fewer than
#   MAX_RESTARTS restarts and the cost estimate is under COST_CAP; otherwise
#   the seed is dropped and its storage machine stopped.
# At HARD_STOP every machine of every pair is stopped.
set -uo pipefail

: "${PROJECT:?}" "${SSH_USER:?}" "${WORK:?}"
export PROJECT SSH_USER
PAIRS=${PAIRS:-1 2 3 4 5}
POLL=${POLL:-600}
RESTART_BY=${RESTART_BY:-$(TZ=Asia/Seoul date -d '2026-11-27 21:00' +%s)}
HARD_STOP=${HARD_STOP:-$(TZ=Asia/Seoul date -d '2026-11-27 22:00' +%s)}
MAX_RESTARTS=${MAX_RESTARTS:-2}
COST_CAP=${COST_CAP:-32}          # USD, estimate for this matrix
GPU_RATE=${GPU_RATE:-0.25}        # USD per hour, spot n1-standard-8 + T4 (estimate)
STORE_RATE=${STORE_RATE:-0.07}    # USD per hour, e2-standard-2 on demand
here=$(cd "$(dirname "$0")" && pwd)
cloud() { timeout 600 bash "$here/cloud.sh" "$@"; }
S=$WORK/watch; R=$WORK/results
mkdir -p "$S" "$R"
log() { echo "$(TZ=Asia/Seoul date '+%m-%d %H:%M') $*" | tee -a "$S/watch.log"; }
get() { cat "$S/$1" 2>/dev/null || echo "${2:-}"; }
put() { echo "$2" > "$S/$1"; }

cost() {  # estimate from the minutes each machine was seen running
  local g=0 s=0 n
  for n in $PAIRS; do g=$((g + $(get gpu-min-$n 0))); s=$((s + $(get store-min-$n 0))); done
  awk -v g="$g" -v s="$s" -v a="$GPU_RATE" -v b="$STORE_RATE" 'BEGIN {printf "%.2f", g/60*a + s/60*b}'
}

pull() {  # into a fresh folder, then swap, so files moved aside on the storage machine do not linger
  local tmp; tmp=$(mktemp -d "$R/.pull-$1-XXXX")
  if cloud pull-store "$1" "results-w13-seed$1" "$tmp" >/dev/null 2>&1 && [[ -d $tmp/results-w13-seed$1 ]]; then
    rm -rf "$R/results-w13-seed$1" && mv "$tmp/results-w13-seed$1" "$R/" && log "pair $1: results copied"
  else
    log "pair $1: copy failed"
  fi
  rm -rf "$tmp"
}

# minutes still needed: both groups, or only the second if the first is complete
remaining_min() {
  local done900
  done900=$(cloud ssh "$1" store "grep -l '^finished:' /srv/facp/results-w13-seed$1/m900/*/cell.txt 2>/dev/null | wc -l" 2>/dev/null | tail -1)
  [[ ${done900:-0} -ge 4 ]] && echo 500 || echo 980
}

matrix_done() {
  cloud ssh "$1" store "grep -q '^done:' /srv/facp/results-w13-seed$1/matrix.txt && echo yes" 2>/dev/null | grep -q yes
}

restart() {  # pair minutes
  local n=$1 m=$2 ip
  cloud start "$n" || { log "pair $n: start failed (no spot capacity?), retry next poll"; return 1; }
  sleep 60
  cloud ssh "$n" gpu "sudo shutdown -h +$((m + 45))"
  cloud ssh "$n" store "sudo shutdown -h +$((m + 75))"
  ip=$(cloud store-ip "$n")
  cloud ssh "$n" gpu "mountpoint -q /mnt/store || sudo mount -t nfs -o hard,vers=4.2 $ip:/srv/facp /mnt/store; \
    sudo rm -rf /mnt/store/ckpt-*; cd ~ && (SEED=$n OUT=\$HOME/results/w13-seed$n RESUME=1 SHUTDOWN=1 \
    setsid nohup bash ~/facp/experiments/run-matrix.sh >> ~/w13.log 2>&1 < /dev/null &); sleep 20; \
    tail -3 ~/results/w13-seed$n/matrix.txt"
  put restarts-$n $(( $(get restarts-$n 0) + 1 ))
  log "pair $n: restarted and resumed, $m min left, restart $(get restarts-$n)"
}

log "watcher started: pairs $PAIRS, restart by $(TZ=Asia/Seoul date -d @"$RESTART_BY" '+%m-%d %H:%M'), hard stop $(TZ=Asia/Seoul date -d @"$HARD_STOP" '+%m-%d %H:%M')"
tick=0
while :; do
  now=$(date +%s)
  if (( now >= HARD_STOP )); then
    for n in $PAIRS; do
      [[ $(get end-$n) ]] && continue
      pull "$n"; cloud stop "$n"; put end-$n "stopped at hard stop"; log "pair $n: stopped at the hard stop"
    done
    log "hard stop reached; estimated cost $(cost) USD"; break
  fi
  active=0
  for n in $PAIRS; do
    [[ $(get end-$n) ]] && continue
    active=1
    gs=$(cloud state "$n" gpu 2>/dev/null); ss=$(cloud state "$n" store 2>/dev/null)
    [[ $gs == RUNNING ]] && put gpu-min-$n $(( $(get gpu-min-$n 0) + POLL / 60 ))
    [[ $ss == RUNNING ]] && put store-min-$n $(( $(get store-min-$n 0) + POLL / 60 ))
    if [[ $ss != RUNNING ]]; then
      log "pair $n: storage machine is $ss (backstop?); starting it"; cloud start "$n" >/dev/null 2>&1; continue
    fi
    if matrix_done "$n"; then
      pull "$n"; [[ $gs == RUNNING ]] && sleep 180
      cloud stop "$n" >/dev/null 2>&1; put end-$n "finished"; log "pair $n: matrix done, machines stopped"; continue
    fi
    case $gs in
      RUNNING|PROVISIONING|STAGING|STOPPING) (( tick % 3 == 0 )) && pull "$n" ;;
      TERMINATED|STOPPED|SUSPENDED)
        m=$(remaining_min "$n")
        if (( now + (m + 30) * 60 > RESTART_BY )); then
          pull "$n"; cloud stop "$n"; put end-$n "dropped: no time"; log "pair $n: GPU stopped early, no time to rerun $m min; seed dropped"
        elif (( $(get restarts-$n 0) >= MAX_RESTARTS )); then
          pull "$n"; cloud stop "$n"; put end-$n "dropped: restarts"; log "pair $n: GPU stopped early again; seed dropped"
        elif awk -v c="$(cost)" -v k="$COST_CAP" 'BEGIN {exit !(c > k)}'; then
          pull "$n"; cloud stop "$n"; put end-$n "dropped: cost"; log "pair $n: cost estimate over $COST_CAP USD; seed dropped"
        else
          log "pair $n: GPU is $gs before the matrix ended (spot stop)"; pull "$n"; restart "$n" "$m"
        fi ;;
      *) log "pair $n: GPU state '$gs' (gcloud error?)" ;;
    esac
  done
  { TZ=Asia/Seoul date '+updated %m-%d %H:%M (GMT+09)'
    for n in $PAIRS; do echo "pair $n: $(get end-$n running) restarts=$(get restarts-$n 0) gpu_min=$(get gpu-min-$n 0)"; done
    echo "estimated cost: $(cost) USD"; } > "$S/status.txt"
  (( active )) || { log "all pairs ended; estimated cost $(cost) USD"; break; }
  tick=$((tick + 1)); sleep "$POLL"
done
