#!/usr/bin/env bash
# Rerun the DCP baseline cell on pair N (16 Nov 2026). The first matrix's
# DCP cells filled the storage disk, because old DCP folders were never
# deleted. The GPU machines were switched to spot pricing for this rerun.
#
#   PROJECT=... bash experiments/rerun-dcp.sh N
#
# Cells: warm-up 5 min, reference 15 min, DCP every 46 steps for 2 hours,
# with the same seed (so the same kill times) as the first matrix.
set -euo pipefail
n=${1:?pair number}
here=$(dirname "$0")
cloud() { bash "$here/cloud.sh" "$@"; }

cloud start "$n"
sleep 30
cloud backstop "$n" 175          # matrix about 145 min plus margin
cloud push "$n"
ip=$(cloud store-ip "$n")
cloud ssh "$n" gpu "mountpoint -q /mnt/store || sudo mount -t nfs -o hard,vers=4.2 $ip:/srv/facp /mnt/store; \
  sudo rm -rf /mnt/store/ckpt-*; df -h /mnt/store | tail -1"
# the launch sits in its own subshell with all output redirected; otherwise
# the background job holds the SSH session open until the matrix ends
cloud ssh "$n" gpu "cd ~ && (SEED=$n MTBF=3600 DURATION=7200 CELLS='ref dcp:46' \
  OUT=\$HOME/results/rerun-seed$n SHUTDOWN=1 setsid nohup bash ~/facp/experiments/run-matrix.sh \
  > ~/rerun.log 2>&1 < /dev/null &); sleep 5; cat ~/results/rerun-seed$n/matrix.txt"
