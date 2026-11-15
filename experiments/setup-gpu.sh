#!/usr/bin/env bash
# Run on the GPU machine after `cloud.sh push`: mount the storage, install
# the library, and cache the tokenised WikiText-2.
#   bash ~/facp/experiments/setup-gpu.sh STORE_IP
set -euo pipefail
store_ip=${1:?storage machine internal IP}

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nfs-common
sudo mkdir -p /mnt/store
mountpoint -q /mnt/store || sudo mount -t nfs -o hard,vers=4.2 "$store_ip:/srv/facp" /mnt/store
df -h /mnt/store

# the image's own PyTorch is kept; only the extras are installed for this user
# (the image has no venv module). The image's torchaudio 2.11 does not match
# its torch 2.9.1 and breaks `import transformers`; audio is not needed.
sudo python3 -m pip uninstall -y -q torchaudio || true
python3 -m pip install -q --user -e "$HOME/facp" "transformers>=4.44" "datasets>=2.20"
python3 - <<'EOF'
import torch
from pathlib import Path
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import sys; sys.path.insert(0, str(Path.home() / "facp" / "examples"))
from train_gpt2 import load_tokens
print("tokens", len(load_tokens("gpt2-large", Path.home() / "data", 0)))
EOF
free -g
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
