#!/usr/bin/env bash
# Launch the 27 November matrix on pair N (Week 13, Plan B validation).
#
#   PROJECT=... bash experiments/launch-w13.sh N
#
# Pair 1 exists and is started; pairs 2-5 are created as spot GPU machines.
# One boot runs: warm-up 5 min, reference 15 min, then the four policies at
# one failure per 15 minutes, then the four at one per 60 minutes, 2 hours
# each (about 16 h 20 min). Seed N sets the kill times, so seeds 1-3 are
# killed at the same moments as in the Week 12 baseline.
set -euo pipefail
n=${1:?pair number}
here=$(dirname "$0")
cloud() { bash "$here/cloud.sh" "$@"; }

# a smoke test overrides CELLS, the durations and BACKSTOP_MIN
CELLS=${CELLS:-"ref checkfreq:0.01@900 checkfreq:0.05@900 dcp:64@900 failure-aware:900@900 \
checkfreq:0.01@3600 checkfreq:0.05@3600 dcp:64@3600 failure-aware:3600@3600"}
DURATION=${DURATION:-7200}; WARMUP_DURATION=${WARMUP_DURATION:-300}; REF_DURATION=${REF_DURATION:-900}
MATRIX_MIN=980            # 16 h 20 min
BACKSTOP_MIN=${BACKSTOP_MIN:-$((MATRIX_MIN + 50))}

if [[ $n == 1 ]]; then
  cloud start 1
else
  SPOT=1 cloud create "$n"
fi
sleep 60                  # let SSH come up
# the watcher's key (a line user:type key comment), so it can reach this pair
[[ -n ${WATCHER_KEY:-} ]] && cloud add-key "$n" "$WATCHER_KEY"
cloud backstop "$n" "$BACKSTOP_MIN"
[[ $n == 1 ]] || cloud ssh "$n" store "bash -s" < "$here/setup-store.sh"
cloud push "$n"
ip=$(cloud store-ip "$n")
if [[ $n == 1 ]]; then
  cloud ssh "$n" gpu "mountpoint -q /mnt/store || sudo mount -t nfs -o hard,vers=4.2 $ip:/srv/facp /mnt/store; \
    python3 -m pip install -q --user -e ~/facp"
else
  cloud ssh "$n" gpu "bash ~/facp/experiments/setup-gpu.sh $ip"
fi
# a fresh matrix: no results left from a smoke test or an earlier launch
# no training left over from an earlier launch may hold the GPU (bracket: no self-match)
cloud ssh "$n" gpu "pkill -9 -f '[r]un-matrix.sh|[f]acp.supervisor|[t]rain_gpt2' || true; sleep 2; \
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo GPU busy; exit 1; } || true"
cloud ssh "$n" gpu "sudo rm -rf /mnt/store/ckpt-* /mnt/store/results-w13-seed$n ~/results/w13-seed$n; df -h /mnt/store | tail -1; free -g | sed -n 2p"
# own subshell with all output redirected, or SSH stays open until the matrix ends
cloud ssh "$n" gpu "cd ~ && (SEED=$n OUT=\$HOME/results/w13-seed$n CELLS='$CELLS' SHUTDOWN=1 \
  DURATION=$DURATION WARMUP_DURATION=$WARMUP_DURATION REF_DURATION=$REF_DURATION \
  setsid nohup bash ~/facp/experiments/run-matrix.sh > ~/w13.log 2>&1 < /dev/null &); \
  sleep 20; cat ~/results/w13-seed$n/matrix.txt"
echo "pair $n launched at $(date -Is); backstop ${BACKSTOP_MIN} min"
