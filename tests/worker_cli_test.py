# KVMem owns the CLI: --kvmem-* must be the authority and translate exactly into the backend's
# plugin configuration (same names/meanings as the llama.cpp and NInfer workers).
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PY = sys.executable
WORKER = "scripts/kvmem-vllm-worker.py"
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def run(*args):
    out = subprocess.run([PY, WORKER, "--dry-run", *args], capture_output=True, text=True)
    if out.returncode != 0:
        return None, out.stderr.strip()
    return json.loads(out.stdout), ""


def main():
    plan, err = run("--kvmem-budget", "32768", "--kvmem-gen-reserve", "16384",
                    "--kvmem-sink-tokens", "8192", "--kvmem-recent-tokens", "1024",
                    "--kvmem-host-mib", "16384", "--kvmem-sessions", "4",
                    "--kvmem-block-tokens", "128", "--no-kvmem-lrfu")
    check(plan is not None, f"the worker accepts the contract parameters ({err})")
    cfg = plan["plugin_config"]
    check(cfg["budget_tokens"] == 32768, "B is passed through")
    check(cfg["gen_reserve_tokens"] == 16384, "R is passed through")
    check(cfg["sink_tokens"] == 8192 and cfg["recent_tokens"] == 1024,
          "sink/recent are passed through")
    check(cfg["host_mib"] == 16384 and cfg["sessions"] == 4, "H and sessions are passed through")
    check(cfg["select_block_tokens"] == 128, "the selection block is passed through")
    check(cfg["lrfu"] is False, "--no-kvmem-lrfu disables LRFU")
    check(cfg["reselect_token_budget"] == 0,
          "the cadence is left to KVMem's reserve (no self-built timer)")
    check(plan["backend_args"]["tensor_parallel_size"] == 2, "TP is a backend argument")

    # A selection block must be a whole number of 16-token pages.
    _, err = run("--kvmem-block-tokens", "130")
    check("multiple of the native page" in err, "a misaligned block size is rejected")

    # Sink/recent must be page aligned too.
    _, err = run("--kvmem-sink-tokens", "1000")
    check("page aligned" in err, "a misaligned sink window is rejected")

    # Disk tier needs a path.
    _, err = run("--kvmem-disk-mib", "1024")
    check("requires --kvmem-disk-path" in err, "a disk tier without a path is rejected")

    # --no-kvmem turns the whole mechanism off but still yields a valid plan.
    plan, err = run("--no-kvmem")
    check(plan is not None and plan["plugin_config"]["sparse"] is False,
          "--no-kvmem disables sparsity")

    # Multi-sequence is a backend choice; the worker maps it to the 4-way launcher.
    plan, _ = run("--max-num-seqs", "4", "--max-model-len", "32768")
    check(plan["backend_args"]["max_num_seqs"] == 4, "concurrency is a backend argument")
    _, err = run("--max-num-seqs", "2", "--kvmem-sessions", "1")
    check("must be >= --max-num-seqs" in err,
          "H-tier shares must cover simultaneous active requests")

    # In-request compaction rewrites the block table mid-request, so it must outweigh an explicit
    # async request; with compaction off the caller's choice stands and async stays the default.
    sys.path.insert(0, "backends/vllm")
    import worker as backend_worker  # noqa: E402
    check(backend_worker.async_scheduling_mode({"inrequest_compaction": True}, "0") == "1",
          "compaction forces sync scheduling even against an explicit 0")
    check(backend_worker.async_scheduling_mode({"inrequest_compaction": False}, "") == "0",
          "without compaction the default is async")
    check(backend_worker.async_scheduling_mode({"inrequest_compaction": False}, "1") == "1",
          "an explicit sync choice stands without compaction")

    # In-request compaction is ONE feature: the CLI switch also drives the entry-side ceiling.
    check(backend_worker.entry_compaction_env({"inrequest_compaction": True}, {}) == ("1", None),
          "compaction on exports the entry flag by itself")
    check(backend_worker.entry_compaction_env({"inrequest_compaction": True},
                                              {"KVMEM_ENTRY_INREQUEST_COMPACTION": "0"})
          == ("1", "entry was off"),
          "compaction on overrides a stale entry flag of 0")
    check(backend_worker.entry_compaction_env({"inrequest_compaction": False},
                                              {"KVMEM_ENTRY_INREQUEST_COMPACTION": "1"})
          == ("0", "entry was on"),
          "engine compaction off forces the entry ceiling back to R")
    check(backend_worker.entry_compaction_env({"inrequest_compaction": False}, {}) == ("0", None),
          "without compaction the entry flag is off and silent")

    # Session reuse is opt-in and MTP is refused until draft StateImage is checkpointed.
    for value, expected in (("0", False), ("1", True)):
        out = io.StringIO()
        with patch.dict(os.environ, {"KVMEM_SESSION_REUSE": value}), \
                contextlib.redirect_stdout(out):
            rc = backend_worker.main(["--model", "/tmp/unused-model",
                                     "--kvmem-inrequest-compaction",
                                     "--speculative-tokens", "0", "--dry-run"])
        dry = json.loads(out.getvalue())
        check(rc == 0 and bool(dry["plugin_config"].get("session_reuse", False)) is expected,
              f"KVMEM_SESSION_REUSE={value} maps to session_reuse={expected}")
    with patch.dict(os.environ, {"KVMEM_SESSION_REUSE": "1"}):
        try:
            backend_worker.main(["--model", "/tmp/unused-model",
                                 "--kvmem-inrequest-compaction", "--dry-run"])
        except SystemExit as exc:
            check("requires explicit --speculative-tokens 0" in str(exc),
                  "MTP and session checkpointing fail closed at startup")
        else:
            check(False, "MTP and session checkpointing must not start together")

    # The site script passes 0 explicitly: never let Python's `0 or -1` fall back to the
    # MTP-enabled launcher profile. Mock only the final subprocess, not argument parsing.
    with patch.object(backend_worker.subprocess, "call", return_value=0) as launch:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = backend_worker.main(["--model", "/tmp/unused-model",
                                      "--speculative-tokens", "0"])
    args = launch.call_args.args[0]
    check(rc == 0 and "SPECULATIVE_TOKENS=0" in args and "SPECULATIVE_METHOD=none" in args,
          "zero speculative tokens disables the launcher profile's MTP method")
    check(launch.call_args.kwargs["env"]["SPECULATIVE_TOKENS"] == "0",
          "zero speculative tokens survives the worker environment")

    # Exercise the actual shell -> worker boundary without starting an engine. FOREGROUND avoids
    # the legacy launcher cleanup path; a stub worker only writes its received CLI arguments.
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        backend = root / "backends" / "vllm"
        backend.mkdir(parents=True)
        (backend / "worker.py").write_text(
            "import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        env = {**os.environ, "ADAPTER": temp, "MODEL": temp,
               "VLLM_PY": sys.executable, "FOREGROUND": "1", "SPECULATIVE_TOKENS": "0",
               "LOG": str(root / "launch.log"), "PARAMS_FILE": str(root / "params.env")}
        out = subprocess.run(["bash", "deploy/run-kvmem-vllm.sh"], env=env,
                             text=True, capture_output=True, timeout=10)
        args = json.loads((root / "launch.log").read_text().splitlines()[-1])
        check(out.returncode == 0 and args[-2:] == ["--speculative-tokens", "0"],
              "shell entrypoint forwards explicit zero to the backend worker")

    print("PASS worker_cli_test" if not failed else "FAILED worker_cli_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
