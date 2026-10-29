#!/usr/bin/env bash
# 安装（或覆盖）systemd 单元 kvmem-vllm，把仓库路径/用户写进模板。
# 用法： sudo ./deploy/install-service.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_USER=${RUN_USER:-${SUDO_USER:-$(id -un)}}
RUN_HOME=$(getent passwd "$RUN_USER" | cut -d: -f6)
TEMPLATE="$REPO/deploy/kvmem-vllm.service.in"
UNIT=/etc/systemd/system/kvmem-vllm.service

[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE" >&2; exit 1; }
sed -e "s#@REPO@#$REPO#g" -e "s#@USER@#$RUN_USER#g" -e "s#@HOME@#$RUN_HOME#g" \
    "$TEMPLATE" > /tmp/kvmem-vllm.service.$$
install -m 0644 /tmp/kvmem-vllm.service.$$ "$UNIT"
rm -f /tmp/kvmem-vllm.service.$$
systemctl daemon-reload
echo "installed $UNIT (user=$RUN_USER repo=$REPO)"
echo "next: systemctl enable --now kvmem-vllm"
