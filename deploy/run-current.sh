#!/usr/bin/env bash
# ============================================================================
#  通用启动入口 —— 这里不放任何站点参数
#
#  约定：
#    start-*.sh          每个文件一套配置（B/R/N/GPU_UTIL/MODEL/... ），最后 exec 本文件
#    run-current.sh      只做通用启动，参数全部来自调用者的环境变量
#    systemd ExecStart   指向哪个 start-*.sh，就启动哪套配置
#
#  覆盖方式见 docs/PARAMETERS.md；默认值见 run-kvmem-vllm.sh 里的 ${VAR:-default}。
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_IMPL=${RUN_IMPL:-$HERE/run-kvmem-vllm.sh}
exec "$RUN_IMPL" "$@"
