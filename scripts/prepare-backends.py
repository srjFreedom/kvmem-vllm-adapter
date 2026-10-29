#!/usr/bin/env python3
"""Initialize the pinned engine sources and apply the KVMem integration patch.

Same layout kvmem itself uses for its llama.cpp / ninfer backends: a shallow submodule pinned to
`base_commit`, plus one versioned patch per engine whose sha256 and resulting tree hash are recorded
in `backends/versions.json`. Running this twice is safe: an already-applied patch is a no-op.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = json.loads((ROOT / "backends" / "versions.json").read_text())


def run(*args, cwd=None, check=True):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"failed: {' '.join(args)}")
    return result


def ensure_checkout(path: Path, url: str, commit: str) -> None:
    """Fetch the pinned engine into `path`, never reset a checkout we did not create.

    The release flow replaces the engine through the versioned patch only. A directory that
    already holds a different commit is refused instead of being checked out: on a workstation
    that path may be a user's own fork with uncommitted work.
    """
    if (path / ".git").exists():
        head = run("git", "-C", str(path), "rev-parse", "HEAD", check=False).stdout.strip()
        dirt = run("git", "-C", str(path), "status", "--porcelain", check=False).stdout.strip()
        if head != commit:
            raise SystemExit(
                f"{path} is at {head[:12]}, not the pinned {commit[:12]}; refusing to check it out. "
                "Remove the directory (or point --engine at a fresh one) and re-run.")
        if dirt:
            # Tolerated only when the diff IS this released patch (a re-run of this script);
            # main() still refuses anything else before touching the tree.
            print(f"  note: {path} already carries local changes")
        return
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{path} exists but is not a git checkout")
    print(f"  cloning {url} -> {path} (depth 1, ref {commit[:12]})")
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
        # A commit sha: depth-1 clones do not contain it, so fetch it explicitly.
        run("git", "clone", "--depth", "1", "--no-checkout", url, str(path))
        run("git", "-C", str(path), "fetch", "--depth", "1", "origin", commit)
        run("git", "-C", str(path), "checkout", "-q", commit)
    else:
        run("git", "clone", "--depth", "1", "--branch", commit, url, str(path))


def patch_applied(path: Path, patch: Path) -> bool:
    """True only when the patch is already applied verbatim (reverse-apply would succeed)."""
    reverse = run("git", "-C", str(path), "apply", "--reverse", "--check", str(patch), check=False)
    return reverse.returncode == 0


def verify_tree(path: Path, expected: str) -> None:
    """Verify the patched tree hash, using a throwaway index so the user's index is untouched."""
    import os
    import tempfile

    handle, index = tempfile.mkstemp(prefix="kvmem-tree-")
    os.close(handle)
    os.unlink(index)
    env = dict(os.environ, GIT_INDEX_FILE=index)
    try:
        subprocess.run(["git", "-C", str(path), "read-tree", "HEAD"], check=True, env=env)
        subprocess.run(["git", "-C", str(path), "add", "-A"], check=True, env=env)
        actual = subprocess.run(["git", "-C", str(path), "write-tree"], check=True,
                                capture_output=True, text=True, env=env).stdout.strip()
    finally:
        if Path(index).exists():
            os.unlink(index)
    if actual != expected:
        raise SystemExit(f"{path}: tree {actual} != pinned {expected}; "
                         "record engine edits in the patch before releasing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="vllm", choices=("vllm", "kvmem"))
    parser.add_argument("--check-tree", action="store_true")
    args = parser.parse_args()
    entry = VERSIONS["engines"] if args.engine == "vllm" else VERSIONS["engines_kvmem"]
    if args.engine == "kvmem":
        print(f"[prepare] kvmem: {entry['upstream']} @ {entry['base_commit']}")
        ensure_checkout(ROOT / entry["path"], entry["upstream"], entry["base_commit"])
        print(f"  ready: {entry['path']}")
        return 0
    path = ROOT / entry["path"]
    patch = ROOT / entry["patch"]

    actual_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    if actual_sha != entry["patch_sha256"]:
        raise SystemExit("patch changed: update backends/versions.json (patch_sha256, patched_tree)")

    print(f"[prepare] {args.engine}: {entry['upstream']} @ {entry['base_commit'][:12]}")
    ensure_checkout(path, entry["upstream"], entry["base_commit"])

    dirty = run("git", "-C", str(path), "status", "--porcelain", check=False).stdout.strip()
    if patch_applied(path, patch):
        print("  patch already applied")
    else:
        if dirty:
            raise SystemExit(
                f"{path} has local changes that are not this patch; refusing to patch over them: "
                f"{dirty.splitlines()[:3]}")
        run("git", "-C", str(path), "apply", str(patch))
        print(f"  applied {entry['patch']}")

    verify_tree(path, entry["patched_tree"])
    print("  patched tree matches versions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
