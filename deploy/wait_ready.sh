#!/usr/bin/env bash
# Thin wrapper: implementation lives in <repo>/scripts/wait_ready.sh.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$HERE/../scripts/wait_ready.sh" "$@"
