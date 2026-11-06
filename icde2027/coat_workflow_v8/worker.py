"""Persistent, bounded native H/O producer. No dataset or old result reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def checked_entry(entry):
    path = Path(entry["path"]).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("runtime SHA mismatch: " + str(path))


def send(value, limit):
    raw = encode(value)
    if len(raw) > limit:
        raise ValueError("worker response exceeds byte limit")
    sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()


def main():
    started = time.perf_counter_ns()
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-json", required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--max-line-bytes", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.memory_limit_bytes < 1024 ** 2 or args.max_line_bytes <= 0:
            raise ValueError("invalid resource limits")
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_limit_bytes, args.memory_limit_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.sched_setaffinity(0, {args.cpu})
        sys.set_int_max_str_digits(100000)
        config = json.loads(args.runtime_json)
        for entry in config["sources"].values():
            checked_entry(entry)
        root = str(Path(config["project_root"]).resolve(strict=True))
        # -B prevents writes; this fresh nonexistent prefix also avoids loading
        # stale bytecode in place of the source bytes checked above.
        sys.pycache_prefix = str(Path(root) / ".nonexistent-workflow-bytecode" / str(os.getpid()))
        sys.path.insert(0, root)
        from icde2027.coat_workflow_v8.adapter import NativeProducer, strict_loads
        raw = sys.stdin.buffer.readline(args.max_line_bytes + 2)
        if not raw.endswith(b"\n") or len(raw) - 1 > args.max_line_bytes:
            raise ValueError("missing or oversized initialization request")
        initial = strict_loads(raw)
        if set(initial) != {"operation", "arm", "legacy_view"} or initial["operation"] != "initialize":
            raise ValueError("initialization fields differ")
        producer = NativeProducer(initial["arm"], initial["legacy_view"])
        import highspy
        import numpy
        import scipy
        send({"kind": "ready", "schema_version": "coat-workflow-native-ready/1", "pid": os.getpid(),
              "arm": initial["arm"], "setup_ns": time.perf_counter_ns() - started,
              "affinity": sorted(os.sched_getaffinity(0)), "memory_limit_bytes": args.memory_limit_bytes,
              "integer_string_digits": sys.get_int_max_str_digits(),
              "sources": {key: entry["sha256"] for key, entry in config["sources"].items()},
              "runtime": {"python": sys.version, "executable": sys.executable,
                          "highs": producer.engine.native_version,
                          "highspy_file": highspy.__file__, "numpy": numpy.__version__,
                          "scipy": scipy.__version__}}, args.max_line_bytes)
    except Exception as error:
        send({"kind": "initialization_refused", "error": type(error).__name__ + ": " + str(error)},
             args.max_line_bytes)
        return 2
    expected_sequence = 0
    for raw in iter(lambda: sys.stdin.buffer.readline(args.max_line_bytes + 2), b""):
        if not raw.endswith(b"\n") or len(raw) - 1 > args.max_line_bytes:
            raise ValueError("missing or oversized command line")
        request = strict_loads(raw)
        if request == {"operation": "stop"}:
            from icde2027.family_lp_v7.service_worker import runtime_inventory
            inventory = runtime_inventory()
            for entry in config["sources"].values():
                checked_entry(entry)
            send({"kind": "diagnostics", "queries": expected_sequence,
                  "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "runtime_inventory": inventory}, args.max_line_bytes)
            return 0
        if (set(request) != {"operation", "sequence", "objective", "threshold"}
                or request["operation"] != "query" or type(request["sequence"]) is not int
                or request["sequence"] != expected_sequence):
            raise ValueError("query sequence or fields differ")
        sequence = expected_sequence
        expected_sequence += 1
        try:
            result = producer.query(request["objective"], request["threshold"])
            send({"kind": "response", "sequence": sequence, **result}, args.max_line_bytes)
        except (ValueError, ArithmeticError, RuntimeError) as error:
            send({"kind": "producer_refused", "sequence": sequence,
                  "error": type(error).__name__ + ": " + str(error),
                  "partial_diagnostics": producer.last_diagnostics}, args.max_line_bytes)
    raise ValueError("unexpected EOF before stop")


if __name__ == "__main__":
    raise SystemExit(main())
