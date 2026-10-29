#!/usr/bin/env bash
# 安装（或覆盖）systemd 单元 kvmem-vllm，把仓库路径/用户写进模板。
# 用法： sudo ./deploy/install-service.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_USER=${RUN_USER:-${SUDO_USER:-$(id -un)}}
RUN_HOME=$(getent passwd "$RUN_USER" | cut -d: -f6)
# 站点启动脚本：默认用仓库里的，也可指到自己的配置（例如 ~/start-kvmem-vllm-<site>.sh）
START_SCRIPT=${START_SCRIPT:-$REPO/start-kvmem-vllm.sh}
TEMPLATE="$REPO/deploy/kvmem-vllm.service.in"
UNIT=/etc/systemd/system/kvmem-vllm.service

[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE" >&2; exit 1; }
sed -e "s#@REPO@#$REPO#g" -e "s#@USER@#$RUN_USER#g" -e "s#@HOME@#$RUN_HOME#g" \
    -e "s#@START_SCRIPT@#$START_SCRIPT#g" \
    "$TEMPLATE" > /tmp/kvmem-vllm.service.$$
install -m 0644 /tmp/kvmem-vllm.service.$$ "$UNIT"
rm -f /tmp/kvmem-vllm.service.$$
systemctl daemon-reload
echo "installed $UNIT (user=$RUN_USER repo=$REPO exec=$START_SCRIPT)"

# 启动脚本里 ${VAR:-默认} 遇到 EnvironmentFile 已有的同名变量会保留文件值，
# 这类"改了脚本不生效"最常见。装完顺手提醒一次。
for f in /etc/default/kvmem-vllm "$RUN_HOME/.config/kvmem-vllm.env"; do
  [ -f "$f" ] || continue
  grep -nE '^(MODEL|SERVED_NAME|B|R|SINK|RECENT|CTX|GPU_UTIL|MAX_MODEL_LEN)=' "$f" 2>/dev/null |     sed "s#^#  warn: $f:#" || true
done
echo "next: systemctl enable --now kvmem-vllm"
