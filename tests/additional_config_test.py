# KVMem's configuration must travel through vLLM's own channel (VllmConfig.additional_config),
# not through a side file: that is what reaches every TP worker and what makes the startup form
# match kvmem-llama / kvmem-ninfer.
import importlib.util
import json
import os
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def main():
    # 1) the worker emits the payload instead of writing a file
    out = subprocess.run([sys.executable, os.path.join(ROOT, "backends", "vllm", "worker.py"),
                          "--dry-run", "--kvmem-budget", "32768", "--kvmem-gen-reserve", "16384",
                          "--kvmem-sink-tokens", "8192", "--kvmem-recent-tokens", "1024",
                          "--kvmem-sessions", "4", "--kvmem-host-mib", "16384",
                          "--kvmem-block-tokens", "128", "--no-kvmem-lrfu"],
                         capture_output=True, text=True)
    check(out.returncode == 0, f"the backend worker runs ({out.stderr.strip()[:80]})")
    plan = json.loads(out.stdout)
    payload = json.loads(plan["additional_config"])
    check("kvmem" in payload, "the payload is wrapped in additional_config['kvmem']")
    kv = payload["kvmem"]
    check(kv["budget_tokens"] == 32768 and kv["gen_reserve_tokens"] == 16384,
          "B and R ride the vLLM config")
    check(kv["host_mib"] == 16384 and kv["sessions"] == 4, "H and sessions ride it too")
    check(kv["select_block_tokens"] == 128 and kv["lrfu"] is False, "block size and LRFU ride it")

    # 2) the plugin prefers additional_config over any file
    spec = importlib.util.spec_from_file_location(
        "kvmem_plugin", os.path.join(ROOT, "plugin", "vllm_kvmem_plugin.py"))
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    runner = types.SimpleNamespace(vllm_config=types.SimpleNamespace(
        additional_config={"kvmem": {"sparse": True, "budget_tokens": 4096}}))
    got = plugin._config_from_runner(runner)
    check(got == {"sparse": True, "budget_tokens": 4096}, "the runner's additional_config wins")
    check(plugin._config()["budget_tokens"] == 4096, "_config() returns it afterwards")

    # A runner without kvmem settings must not clobber what we already have.
    bare = types.SimpleNamespace(vllm_config=types.SimpleNamespace(additional_config={}))
    plugin._config_from_runner(bare)
    check(plugin._config()["budget_tokens"] == 4096, "an unrelated runner does not reset the config")

    print("PASS additional_config_test" if not failed else "FAILED additional_config_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
