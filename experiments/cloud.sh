#!/usr/bin/env bash
# Create, start, stop and reach the experiment machines on Google Cloud.
#
# Each pair N is a GPU machine ${PREFIX}-gpu-N (n1-standard-8, one T4, 30 GB
# memory) and a storage machine ${PREFIX}-store-N (e2-standard-2, NFS server).
# Every name this script touches starts with ${PREFIX}-, so machines outside
# the experiment are never affected.
#
#   PROJECT=my-project experiments/cloud.sh create 1
#   PROJECT=my-project experiments/cloud.sh backstop 1 420   # power off in 420 min
#   PROJECT=my-project experiments/cloud.sh stop 1
#
# SPOT=1 makes `create` a spot GPU machine that Google may stop at any time
# (stopped, not deleted, so its disk and results survive). The storage
# machine is always on demand. GCLOUD names the gcloud binary to use.
set -euo pipefail

: "${PROJECT:?set PROJECT to the Google Cloud project}"
ZONE=${ZONE:-us-central1-f}
PREFIX=${PREFIX:-facp}
[[ $PREFIX == facp* ]] || { echo "PREFIX must start with facp" >&2; exit 1; }
GPU_IMAGE_FAMILY=${GPU_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}
G=("${GCLOUD:-gcloud}" --project "$PROJECT")
SPOT=${SPOT:-0}
spot_flags=()
[[ $SPOT == 1 ]] && spot_flags=(--provisioning-model=SPOT --instance-termination-action=STOP)

gpu() { echo "${PREFIX}-gpu-$1"; }
store() { echo "${PREFIX}-store-$1"; }
# SSH_USER logs in as that user, so a second controller (the watcher) reaches
# the same home folder as the one that launched the runs
# SSH_DIRECT=1 uses plain ssh to the external address with SSH_KEY, for a
# controller whose account may only touch these machines (no project read)
remote() {
  if [[ ${SSH_DIRECT:-0} == 1 ]]; then
    local ip
    ip=$("${G[@]}" compute instances describe "$1" --zone "$ZONE" \
      --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
      -o ConnectTimeout=20 -o BatchMode=yes -i "${SSH_KEY:-$HOME/.ssh/google_compute_engine}" \
      "${SSH_USER:?}@$ip" "$2"
  else
    "${G[@]}" compute ssh "${SSH_USER:+$SSH_USER@}$1" --zone "$ZONE" --quiet --command "$2"
  fi
}

require_running() {
  local state
  state=$("${G[@]}" compute instances describe "$1" --zone "$ZONE" --format='value(status)')
  # a start with an expired login can print an error and still exit 0
  [[ $state == RUNNING ]] || { echo "$1 is $state, not RUNNING (run: gcloud auth login)" >&2; exit 1; }
}

cmd=${1:?command}; n=${2:-1}
case $cmd in
  create)
    "${G[@]}" compute instances create "$(store "$n")" --zone "$ZONE" \
      --machine-type e2-standard-2 --image-family debian-12 --image-project debian-cloud \
      --boot-disk-size 100GB --boot-disk-type pd-balanced
    "${G[@]}" compute instances create "$(gpu "$n")" --zone "$ZONE" \
      --machine-type n1-standard-8 --accelerator type=nvidia-tesla-t4,count=1 \
      --maintenance-policy TERMINATE "${spot_flags[@]}" --image-family "$GPU_IMAGE_FAMILY" \
      --image-project deeplearning-platform-release \
      --boot-disk-size 150GB --boot-disk-type pd-balanced \
      --metadata install-nvidia-driver=True
    require_running "$(store "$n")"; require_running "$(gpu "$n")" ;;
  start)
    # storage first: a training job hangs if its storage is missing
    "${G[@]}" compute instances start "$(store "$n")" --zone "$ZONE"
    "${G[@]}" compute instances start "$(gpu "$n")" --zone "$ZONE"
    require_running "$(store "$n")"; require_running "$(gpu "$n")" ;;
  stop)
    # GPU first, storage second, for the same reason
    "${G[@]}" compute instances stop "$(gpu "$n")" --zone "$ZONE"
    "${G[@]}" compute instances stop "$(store "$n")" --zone "$ZONE" ;;
  delete)
    "${G[@]}" compute instances delete "$(gpu "$n")" "$(store "$n")" --zone "$ZONE" --quiet ;;
  backstop)
    minutes=${3:?minutes}
    remote "$(gpu "$n")" "sudo shutdown -h +$minutes"
    remote "$(store "$n")" "sudo shutdown -h +$((minutes + 30))" ;;
  store-ip)
    "${G[@]}" compute instances describe "$(store "$n")" --zone "$ZONE" \
      --format='value(networkInterfaces[0].networkIP)' ;;
  ssh)
    target=${3:-gpu}
    [[ $target == gpu ]] && name=$(gpu "$n") || name=$(store "$n")
    shift 3 || true
    remote "$name" "$*" ;;
  push)
    # copy the working tree to the GPU machine (no rsync on the image)
    tar czf - --exclude .git --exclude '*.egg-info' --exclude results -C "$(dirname "$0")/.." . |
      "${G[@]}" compute ssh "$(gpu "$n")" --zone "$ZONE" --quiet \
        --command "rm -rf ~/facp && mkdir -p ~/facp && tar xzf - -C ~/facp" ;;
  pull)
    out=${3:-results}
    mkdir -p "$out"
    "${G[@]}" compute ssh "$(gpu "$n")" --zone "$ZONE" --quiet --command "tar czf - -C ~ results" |
      tar xzf - -C "$out" ;;
  add-key)
    # append a key line (user:type key comment) to both machines' own ssh-keys,
    # keeping the keys already there
    line=$(cat "${3:?key line file}")
    for name in "$(gpu "$n")" "$(store "$n")"; do
      tmp=$(mktemp)
      "${G[@]}" compute instances describe "$name" --zone "$ZONE" --format=json |
        python3 -c 'import json,sys
items=json.load(sys.stdin).get("metadata",{}).get("items",[])
print(next((i["value"] for i in items if i["key"]=="ssh-keys"),""))' | grep -v '^$' > "$tmp" || true
      grep -qF "$line" "$tmp" || echo "$line" >> "$tmp"
      "${G[@]}" compute instances add-metadata "$name" --zone "$ZONE" --metadata-from-file ssh-keys="$tmp"
      rm -f "$tmp"
    done ;;
  state)
    # one machine's state; needs no project-wide list permission
    [[ ${3:-gpu} == gpu ]] && name=$(gpu "$n") || name=$(store "$n")
    "${G[@]}" compute instances describe "$name" --zone "$ZONE" --format='value(status)' ;;
  pull-store)
    # copy one results folder from the storage machine (readable while the GPU is off)
    dir=${3:?results folder name}; out=${4:-results}
    mkdir -p "$out"
    remote "$(store "$n")" "tar czf - -C /srv/facp $dir" | tar xzf - -C "$out" ;;
  status)
    "${G[@]}" compute instances list --filter="name~^${PREFIX}-" \
      --format='table(name,zone.basename(),machineType.basename(),status)' ;;
  *) echo "unknown command $cmd" >&2; exit 1 ;;
esac
