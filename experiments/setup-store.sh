#!/usr/bin/env bash
# Run on the storage machine: serve /srv/facp over NFS to the internal network.
set -euo pipefail
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nfs-kernel-server
sudo mkdir -p /srv/facp
sudo chmod 777 /srv/facp
# sync: the server commits each write before it replies, so a finished
# checkpoint write really is on the server's disk
echo "/srv/facp 10.128.0.0/9(rw,sync,no_subtree_check,no_root_squash)" | sudo tee /etc/exports
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
df -h /srv/facp
