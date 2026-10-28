"""In-worker breakpoints for the KVMem engine: stop anywhere, read every local, no print statements.

Why this exists: the engine takes ~80 s to restart and the interesting state lives inside
`execute_model` / `harvest_key_statistics` / `KvmemRecall.scores`. Poking one guessed variable per
run is slow and biased. A tracer installed into the running worker turns any source line into a
breakpoint and dumps the WHOLE frame (locals + requested expressions) the moment it is hit.

Usage (from the plugin console, no restart, no code edits):

    import kvmem_dbg; kvmem_dbg.install()
    # breakpoints live in /tmp/kvmem_dbg.bp, one JSON object per line:
    #   {"file": "kvmem_recall.py", "line": 210, "hits": 2,
    #    "expr": ["scores[:4]", "stats.frontier", "len(layer_rows)"]}
    #   {"func": "compare_scores", "hits": 1}

Everything is written to /tmp/kvmem_dbg.out. When every breakpoint has spent its `hits` the tracer
uninstalls itself, so a forgotten breakpoint cannot tax the engine forever.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback

BP_PATH = "/tmp/kvmem_dbg.bp"
OUT_PATH = "/tmp/kvmem_dbg.out"
MAX_REPR = 300


def _load():
    points = []
    try:
        with open(BP_PATH) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    point = json.loads(line)
                except Exception:
                    continue
                point.setdefault("hits", 1)
                point.setdefault("expr", [])
                point["_left"] = int(point["hits"])
                points.append(point)
    except Exception:
        pass
    return points


_STATE = {"points": [], "installed": False, "dumps": 0}


def _write(text):
    try:
        with open(OUT_PATH, "a") as handle:
            handle.write(text)
    except Exception:
        pass


def _describe(frame, point):
    """One breakpoint hit: the frame's locals plus any expressions the breakpoint asked for."""
    lines = [f"=== BP {point.get('file') or point.get('func')}:{point.get('line') or ''} "
             f"pid={os.getpid()}"]
    code = frame.f_code
    lines.append(f"    at {code.co_filename.split('/')[-1]}:{frame.f_lineno} "
                 f"func={code.co_name}")
    args = code.co_varnames[:code.co_argcount]
    for name, value in list(frame.f_locals.items()):
        try:
            text = repr(value)
        except Exception:
            text = "<unrepresentable>"
        if len(text) > MAX_REPR:
            text = text[:MAX_REPR] + f"...<{len(text)} chars>"
        marker = "arg" if name in args else "   "
        lines.append(f"    {marker} {name} = {text}")
    for expression in point.get("expr", []):
        try:
            value = eval(expression, frame.f_globals, frame.f_locals)     # noqa: S307
            lines.append(f"    EXPR {expression} = {value!r}"[:1200])
        except Exception as exc:
            lines.append(f"    EXPR {expression} !! {type(exc).__name__}: {exc}")
    return "\n".join(lines) + "\n"


def _tracer(frame, event, arg):
    if event != "line":
        return _tracer
    points = _STATE["points"]
    if not points:
        return None
    filename = frame.f_code.co_filename.rsplit("/", 1)[-1]
    for point in points:
        if point["_left"] <= 0:
            continue
        want_file = point.get("file")
        want_func = point.get("func")
        if want_file and want_file.rsplit("/", 1)[-1] != filename:
            continue
        if want_func and want_func != frame.f_code.co_name:
            continue
        if point.get("line") and int(point["line"]) != frame.f_lineno:
            continue
        point["_left"] -= 1
        _STATE["dumps"] += 1
        try:
            _write(_describe(frame, point))
        except Exception:
            _write("=== BP dump failed\n" + traceback.format_exc())
        break
    if all(p["_left"] <= 0 for p in points):
        remove()
    return _tracer


def install():
    """Install the tracer on this thread and (via threading.settrace) on future ones."""
    if _STATE["installed"]:
        return "already installed"
    _STATE["points"] = _load()
    if not _STATE["points"]:
        return "no breakpoints"
    threading.settrace(_tracer)
    sys.settrace(_tracer)
    _STATE["installed"] = True
    _write(f"=== installed {len(_STATE['points'])} breakpoint(s) pid={os.getpid()}\n")
    return f"installed {len(_STATE['points'])}"


def remove():
    try:
        sys.settrace(None)
        threading.settrace(None)
    except Exception:
        pass
    _STATE["installed"] = False
    return "removed"


def status():
    return {"installed": _STATE["installed"], "dumps": _STATE["dumps"],
            "left": [p["_left"] for p in _STATE["points"]]}
