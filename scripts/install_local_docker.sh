#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "请用 root 运行: sudo bash scripts/install_local_docker.sh" >&2
  exit 1
fi

install -m 0755 -d /etc/apt/keyrings
rm -f /etc/apt/sources.list.d/docker.sources
cat >/etc/apt/sources.list.d/docker.sources <<'EOF'
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: noble
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt update
apt install -y ca-certificates curl
if [[ ! -s /etc/apt/keyrings/docker.asc ]]; then
  curl --retry 5 --retry-delay 3 --retry-all-errors \
    -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
fi
chmod a+r /etc/apt/keyrings/docker.asc
apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

if id yangyuxin >/dev/null 2>&1; then
  usermod -aG docker yangyuxin
fi

docker run --rm hello-world
