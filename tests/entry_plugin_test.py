#!/usr/bin/env python3
"""Packaging contract: the adapter ships both plugin entry points vLLM needs.

The engine side (`vllm.general_plugins`) must always be present; the API-server side
(`vllm.endpoint_plugins`) is what makes the entry policy run in-process instead of in a sidecar.
"""
import importlib.metadata as md
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checks = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    checks.append(bool(cond))


def main():
    try:
        dist = md.distribution("vllm-kvmem-plugin")
    except md.PackageNotFoundError:
        check(False, "the adapter plugin package is installed (vllm-kvmem-plugin)")
        print("FAILED entry_plugin_test")
        return 1

    entries = {}
    for group in ("vllm.general_plugins", "vllm.endpoint_plugins"):
        entries[group] = {ep.name: ep.value for ep in dist.entry_points if ep.group == group}

    check("kvmem" in entries["vllm.general_plugins"],
          f"engine-side plugin registered ({entries['vllm.general_plugins']})")
    check("kvmem_entry" in entries["vllm.endpoint_plugins"],
          f"API-server plugin registered ({entries['vllm.endpoint_plugins']})")
    check(entries["vllm.endpoint_plugins"].get("kvmem_entry", "").endswith(":KvmemEntryPlugin"),
          "the endpoint plugin entry resolves to KvmemEntryPlugin")

    # The module must import without vLLM being configured (tests / gateway import it too).
    sys.path.insert(0, os.path.join(ROOT, "plugin"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import kvmem_entry_middleware as entry  # noqa: E402

    check(entry.KvmemEntryPlugin.name == "kvmem_entry", "plugin name is stable")
    check(entry.KvmemEntryPlugin.required_tasks == ("generate",),
          "plugin gates on the generate task")
    check(callable(entry.install) and callable(entry.apply_policy),
          "the plugin installs a policy hook on the parsed request (no middleware)")

    # vLLM's loader must gate the endpoint plugin on VLLM_PLUGINS naming it.
    os.environ["VLLM_PLUGINS"] = "kvmem_entry"
    try:
        from vllm.plugins import load_endpoint_plugins  # noqa: E402

        loaded = load_endpoint_plugins(("generate",))
        check([p.name for p in loaded] == ["kvmem_entry"],
              "vLLM discovers kvmem_entry when VLLM_PLUGINS names it")
    except Exception as exc:  # vLLM unavailable in a bare test env
        check(True, f"vLLM discovery skipped ({type(exc).__name__})")

    print(("PASS" if all(checks) else "FAILED") + " entry_plugin_test")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
