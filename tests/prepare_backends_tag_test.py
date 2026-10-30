"""A pinned tag resolves to its commit SHA on repeat backend preparation."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare-backends.py"
spec = importlib.util.spec_from_file_location("prepare_backends", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def git(path, *args):
    subprocess.run(["git", "-C", str(path), *args], check=True,
                   capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        git(path, "init", "-q")
        git(path, "-c", "user.name=test", "-c", "user.email=test@example.invalid",
            "commit", "-q", "--allow-empty", "-m", "base")
        git(path, "tag", "v0.19.0-rc1")
        module.ensure_checkout(path, "unused", "v0.19.0-rc1")
        git(path, "-c", "user.name=test", "-c", "user.email=test@example.invalid",
            "commit", "-q", "--allow-empty", "-m", "different")
        try:
            module.ensure_checkout(path, "unused", "v0.19.0-rc1")
        except SystemExit as exc:
            assert "refusing to check it out" in str(exc)
        else:
            raise AssertionError("non-pinned native checkout was accepted")
    print("prepare_backends_tag_test: OK")


if __name__ == "__main__":
    main()
