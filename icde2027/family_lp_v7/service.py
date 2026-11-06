"""Fresh-process, bounded-I/O supervisor for a single exact-family request.

This module does not solve or mathematically verify requests. The pinned worker
performs one independent receiver invocation before its public response. Runtime
files and their ancestors must remain trusted and immutable while a call runs;
this is resource isolation, not a hostile-user security sandbox.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


METHOD_SOURCES = {
    "S": ("core", "old_receiver"),
    "G_partition": ("adapter", "library", "old_receiver"),
    "G_general": ("adapter", "library", "general_receiver"),
    "G_fixed": ("adapter", "library", "general_receiver"),
}
MAX_LINE_BYTES = 32 * 1024 ** 2
INTEGER_STRING_DIGITS = 100000


class ResourceLimitError(ValueError):
    pass


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json_loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def forbidden(_):
        raise ValueError("floating/nonfinite JSON number is forbidden")
    return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
                      parse_float=forbidden, parse_constant=forbidden)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _configuration(config, method):
    _need(type(config) is dict, "runtime_config must be a dictionary")
    allowed = {"worker", "core", "old_receiver", "adapter", "general_receiver", "library"}
    _need(set(config).issubset(allowed), "unknown runtime_config field")
    required = {"worker", *METHOD_SOURCES[method]}
    _need(required.issubset(config), "missing method runtime source")
    result = {}
    for key, entry in config.items():
        _need(type(entry) is dict and set(entry) == {"path", "sha256"}, "invalid source identity")
        path, digest = entry["path"], entry["sha256"]
        _need(type(path) is str and Path(path).is_absolute(), "runtime path must be absolute")
        _need(type(digest) is str and len(digest) == 64 and
              all(c in "0123456789abcdef" for c in digest), "invalid SHA-256")
        result[key] = {"path": path, "sha256": digest}
    return result


def _duration(value, name):
    _need(type(value) in (int, float) and math.isfinite(value) and value > 0,
          name + " must be a positive finite duration")
    result = int(value * 1_000_000_000)
    _need(result > 0, name + " is below the timer resolution")
    return result


def run_call(request, *, method, runtime_config, cpu, startup_limit_s=30,
             service_limit_s=10, diagnostic_limit_s=30,
             memory_limit_bytes=8 * 1024 ** 3, max_line_bytes=MAX_LINE_BYTES,
             max_stderr_bytes=1024 ** 2):
    """Return a complete record, including refusals, timeouts and partial runs.

    Primary service time begins immediately before enabling request writes and
    ends on receiving the complete public response line. Request encoding and
    worker startup are separate costs, and all costs enter total_wall_ns. A
    public response without complete diagnostics and clean exit is never a
    complete/verified-success record. No mathematical receiver runs here.
    """
    call_start = time.perf_counter_ns()
    record = {"schema_version": "family-lp-service-record/1", "method": method if type(method) is str else None,
              "status": "input_refusal", "record_complete": False,
              "verified_success": False, "response": None, "diagnostics": None,
              "ready": None, "exit_code": None, "reaped": False,
              "request_sha256": None, "request_bytes": None,
              "response_bytes": None, "response_sha256": None,
              "diagnostic_bytes": None, "diagnostic_sha256": None,
              "time_ns": {}, "error": None, "peak_memory": None}
    try:
        _need(sys.platform == "linux", "Linux process-group/wait4 implementation required")
        _need(method in METHOD_SOURCES, "unknown service method")
        _need(type(cpu) is int and cpu >= 0 and cpu in os.sched_getaffinity(0),
              "CPU must belong to the supervisor's allowed affinity")
        _need(type(memory_limit_bytes) is int and memory_limit_bytes >= 1024 ** 2,
              "invalid address-space limit")
        _need(type(max_line_bytes) is int and max_line_bytes > 0, "invalid line limit")
        _need(type(max_stderr_bytes) is int and max_stderr_bytes > 0, "invalid stderr limit")
        limits_ns = {"startup": _duration(startup_limit_s, "startup_limit_s"),
                     "service": _duration(service_limit_s, "service_limit_s"),
                     "diagnostics": _duration(diagnostic_limit_s, "diagnostic_limit_s")}
        config = _configuration(runtime_config, method)
        _need(type(request) is dict, "request must be a dictionary")
        encoded = canonical_bytes(request)
        # Check the same strict JSON contract before creating a subprocess. No
        # mathematical/schema acceptance is performed by this transport check.
        strict_json_loads(encoded)
        if len(encoded) > max_line_bytes:
            raise ResourceLimitError("request exceeds line byte limit")
        request_wire = encoded + b"\n"
        record.update(request_sha256=hashlib.sha256(encoded).hexdigest(),
                      request_bytes=len(encoded), runtime_config_sha256=hashlib.sha256(canonical_bytes(config)).hexdigest(),
                      python_executable=str(Path(sys.executable).absolute()), cpu=cpu,
                      memory_limit_bytes=memory_limit_bytes, max_line_bytes=max_line_bytes,
                      integer_string_digits=INTEGER_STRING_DIGITS,
                      max_stderr_bytes=max_stderr_bytes, limits_ns=limits_ns)
        worker_bytes = Path(config["worker"]["path"]).read_bytes()
        _need(hashlib.sha256(worker_bytes).hexdigest() == config["worker"]["sha256"],
              "worker source SHA mismatch")
    except Exception as error:
        if isinstance(error, ResourceLimitError):
            record["status"] = "resource_refusal"
        record["error"] = type(error).__name__ + ": " + str(error)
        record["time_ns"]["total_wall_ns"] = time.perf_counter_ns() - call_start
        return record

    # A small explicit environment excludes unrelated account credentials. The
    # native library's recorded RPATH supplies its dependency resolution.
    environment = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        environment[name] = "1"
    argv = [sys.executable, "-B", "-I", "-S", config["worker"]["path"],
            "--runtime-json", canonical_bytes(config).decode("ascii"),
            "--method", method, "--cpu", str(cpu),
            "--memory-limit-bytes", str(memory_limit_bytes),
            "--max-line-bytes", str(max_line_bytes)]
    launch_start = time.perf_counter_ns()
    record["time_ns"]["preflight_ns"] = launch_start - call_start
    process = None
    selector = selectors.DefaultSelector()
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    stage = "startup"
    stage_started = launch_start
    deadline = stage_started + limits_ns[stage]
    write_offset = 0
    service_started = service_ended = None
    exit_usage = None
    failure = None
    stdout_eof = stderr_eof = False

    def fail(status, message):
        nonlocal failure
        if failure is None:
            failure = (status, message)

    def close_pipe(pipe):
        if pipe is None or pipe.closed:
            return
        try:
            selector.unregister(pipe)
        except (KeyError, ValueError):
            pass
        pipe.close()

    def reap():
        nonlocal exit_usage
        if process is None or record["reaped"]:
            return
        try:
            pid, wait_status, usage = os.wait4(process.pid, os.WNOHANG)
        except ChildProcessError:
            fail("supervisor_error", "worker unexpectedly reaped outside supervisor")
            return
        if pid:
            process.returncode = os.waitstatus_to_exitcode(wait_status)
            record["exit_code"] = process.returncode
            record["reaped"] = True
            exit_usage = usage

    def consume_line(line, received_ns):
        nonlocal stage, stage_started, deadline, service_started, service_ended
        try:
            value = strict_json_loads(line)
            _need(type(value) is dict, "worker line is not an object")
            if stage == "startup":
                if value.get("schema_version") == "family-lp-worker-init-refusal/1":
                    fail("initialization_refused", str(value.get("error", "worker initialization refused")))
                    record["initialization_refusal"] = value
                    return
                _need(set(value) == {"schema_version", "ready", "method", "pid", "setup_ns",
                                    "affinity", "memory_limit_bytes", "integer_string_digits", "loaded_sources"}, "ready fields differ")
                _need(value["schema_version"] == "family-lp-worker-ready/1" and value["ready"] is True,
                      "invalid ready schema/status")
                _need(value["method"] == method and value["pid"] == process.pid, "ready worker identity differs")
                _need(type(value["setup_ns"]) is int and value["setup_ns"] >= 0, "invalid setup timing")
                _need(value["affinity"] == [cpu] and value["memory_limit_bytes"] == memory_limit_bytes,
                      "worker resource declaration differs")
                _need(type(value["integer_string_digits"]) is int and
                      value["integer_string_digits"] == INTEGER_STRING_DIGITS,
                      "worker integer-string representation limit differs")
                expected = {name: config[name]["sha256"] for name in METHOD_SOURCES[method]}
                _need(value["loaded_sources"] == expected, "worker loaded-source declaration differs")
                record["ready"] = value
                record["time_ns"]["startup_ns"] = received_ns - launch_start
                service_started = time.perf_counter_ns()
                stage = "service"
                stage_started = service_started
                deadline = service_started + limits_ns[stage]
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            elif stage == "service":
                _need(write_offset == len(request_wire), "response preceded complete request transmission")
                _need(set(value) == {"status", "certificate", "receiver_receipt"}, "public response fields differ")
                _need(value["status"] in ("verified_optimal", "refused"), "invalid public status")
                if value["status"] == "verified_optimal":
                    cert, receipt = value["certificate"], value["receiver_receipt"]
                    _need(type(cert) is dict and type(receipt) is dict and receipt.get("verified") is True,
                          "successful response lacks a positive receiver receipt")
                    _need(type(cert.get("input_sha256")) is str and
                          cert["input_sha256"] == receipt.get("input_sha256"), "certificate/receipt digest differs")
                else:
                    _need(value["certificate"] is None and value["receiver_receipt"] is None,
                          "refusal cannot carry an accepted certificate")
                record["response"] = value
                record["response_bytes"] = len(line)
                record["response_sha256"] = hashlib.sha256(line).hexdigest()
                service_ended = received_ns
                record["time_ns"]["service_ns"] = service_ended - service_started
                stage = "diagnostics"
                stage_started = service_ended
                deadline = stage_started + limits_ns[stage]
            elif stage == "diagnostics":
                _need(value.get("schema_version") == "family-lp-worker-diagnostics/1", "invalid diagnostics schema")
                _need(value.get("method") == method and value.get("request_sha256") == record["request_sha256"],
                      "diagnostics request/method binding differs")
                record["diagnostics"] = value
                record["diagnostic_bytes"] = len(line)
                record["diagnostic_sha256"] = hashlib.sha256(line).hexdigest()
                if value.get("complete") is False and value.get("resource_refusal") is True:
                    fail("resource_refusal", str(value.get("error", "worker diagnostic resource refusal")))
                    return
                _need(value.get("complete") is True, "worker diagnostics incomplete")
                # This deadline continues through clean process exit, not just
                # through receipt of the diagnostic payload.
                stage = "exit"
            else:
                raise ValueError("unexpected extra worker stdout line")
        except Exception as error:
            record["rejected_stdout_line_utf8"] = line.decode("utf-8", "replace")
            fail("protocol_error", type(error).__name__ + ": " + str(error))

    try:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=environment, start_new_session=True,
                                   bufsize=0, close_fds=True)
        record["worker_pid"] = process.pid
        for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        os.set_blocking(process.stdin.fileno(), False)
        while failure is None:
            now = time.perf_counter_ns()
            if now >= deadline:
                fail(("diagnostic" if stage in ("diagnostics", "exit") else stage) + "_timeout", "outer stage deadline exceeded")
                break
            events = selector.select(min((deadline - now) / 1e9, 0.05))
            for key, mask in events:
                pipe, name = key.fileobj, key.data
                if name == "stdin":
                    try:
                        count = os.write(pipe.fileno(), memoryview(request_wire)[write_offset:write_offset + 65536])
                        write_offset += count
                        if write_offset == len(request_wire):
                            close_pipe(pipe)
                    except BlockingIOError:
                        pass
                    except OSError as error:
                        fail("io_error", "request write failed: " + str(error))
                    continue
                try:
                    data = os.read(pipe.fileno(), 65536)
                except BlockingIOError:
                    continue
                received_ns = time.perf_counter_ns()
                if received_ns >= deadline:
                    fail(("diagnostic" if stage in ("diagnostics", "exit") else stage) + "_timeout", "outer stage deadline exceeded")
                    break
                if not data:
                    close_pipe(pipe)
                    if name == "stdout":
                        stdout_eof = True
                        if stdout_buffer or stage != "exit":
                            fail("premature_eof", "stdout closed before complete expected records")
                    else:
                        stderr_eof = True
                    continue
                if name == "stderr":
                    if len(stderr_buffer) + len(data) > max_stderr_bytes:
                        stderr_buffer.extend(data[:max_stderr_bytes - len(stderr_buffer)])
                        fail("resource_refusal", "stderr byte limit exceeded")
                    else:
                        stderr_buffer.extend(data)
                    continue
                stdout_buffer.extend(data)
                while b"\n" in stdout_buffer and failure is None:
                    end = stdout_buffer.index(10)
                    if end > max_line_bytes:
                        fail("resource_refusal", "stdout line byte limit exceeded")
                        break
                    line = bytes(stdout_buffer[:end])
                    del stdout_buffer[:end + 1]
                    consume_line(line, received_ns)
                if len(stdout_buffer) > max_line_bytes:
                    fail("resource_refusal", "unterminated stdout line byte limit exceeded")
            reap()
            if record["reaped"] and stdout_eof and stderr_eof:
                if record["exit_code"] != 0:
                    fail("process_exit", "worker exited nonzero")
                elif stage != "exit":
                    fail("premature_eof", "worker exited before complete records")
                break
    except Exception as error:
        fail("supervisor_error", type(error).__name__ + ": " + str(error))
    finally:
        if process is not None and (failure is not None or not record["reaped"]):
            # No worker subprocesses are used by the actual implementation. A
            # new process group also lets timeout cleanup terminate accidental
            # descendants; it does not make RLIMIT_AS a tree-wide memory cap.
            try:
                os.killpg(process.pid, signal.SIGKILL)
                record["kill_sent"] = True
            except ProcessLookupError:
                record["kill_sent"] = False
            cleanup_deadline = time.monotonic() + 2
            while not record["reaped"] and time.monotonic() < cleanup_deadline:
                reap()
                if not record["reaped"]:
                    time.sleep(0.005)
        if process is not None:
            for pipe in (process.stdin, process.stdout, process.stderr):
                close_pipe(pipe)
        selector.close()

    ended = time.perf_counter_ns()
    times = record["time_ns"]
    if "startup_ns" not in times:
        times["startup_ns"] = ended - launch_start
    if service_started is not None and "service_ns" not in times:
        times["service_ns"] = ended - service_started
    if service_ended is not None:
        times["diagnostics_ns"] = ended - service_ended
    times["total_wall_ns"] = ended - call_start
    times["supervision_unassigned_ns"] = times["total_wall_ns"] - sum(
        times.get(k, 0) for k in ("preflight_ns", "startup_ns", "service_ns", "diagnostics_ns"))
    record["stderr_bytes"] = len(stderr_buffer)
    record["stderr_capture_complete"] = stderr_eof and (failure is None or failure[1] != "stderr byte limit exceeded")
    record["stderr_sha256"] = hashlib.sha256(stderr_buffer).hexdigest()
    record["stderr_utf8"] = stderr_buffer.decode("utf-8", "replace")
    if stdout_buffer:
        record["partial_stdout_utf8"] = stdout_buffer.decode("utf-8", "replace")
        record["partial_stdout_bytes"] = len(stdout_buffer)
        record["partial_stdout_sha256"] = hashlib.sha256(stdout_buffer).hexdigest()
    if exit_usage is not None:
        record["peak_memory"] = {"peak_rss_kib": int(exit_usage.ru_maxrss), "unit": "KiB",
                                 "source": "Linux wait4 rusage for this worker through exit",
                                 "scope": "one worker; actual worker creates no subprocesses",
                                 "user_cpu_seconds": repr(exit_usage.ru_utime),
                                 "system_cpu_seconds": repr(exit_usage.ru_stime)}
    if failure is not None:
        record["status"], record["error"] = failure
        if record["status"] in ("premature_eof", "io_error") and record["reaped"] and record["exit_code"] != 0:
            record["status"] = "process_exit"
    elif record["reaped"] and record["exit_code"] == 0 and record["diagnostics"] is not None:
        record["record_complete"] = True
        record["status"] = record["response"]["status"]
        if record["status"] == "refused" and record["diagnostics"].get("outcome_kind") == "resource_refusal":
            record["status"] = "resource_refusal"
        record["verified_success"] = record["status"] == "verified_optimal"
    else:
        record["status"] = "incomplete"
        record["error"] = "worker did not supply diagnostics and a clean reaped exit"
    return record


__all__ = ["run_call", "canonical_bytes", "strict_json_loads", "METHOD_SOURCES"]
