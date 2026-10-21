#!/usr/bin/env bash
# Thin wrapper: the maintained implementation lives in the adapter repo.
exec /home/srj/vllm-kvmem-adapter/scripts/wait_ready.sh "$@"
