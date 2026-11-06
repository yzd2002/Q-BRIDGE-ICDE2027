"""Bounded persistent native workflow with an external rational receiver.

No real workload is selected or launched here.  The caller supplies a separately
admitted stream and pins, owns the full-run schedule and the two-hour deadline.
Full wall time includes cleanup.  Nested producer/receiver clocks are labelled
and never added to the disjoint supervisor phase decomposition.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import ctypes
import os
from pathlib import Path
import resource
import selectors
import signal
import subprocess
import sys
import time
import types

from .adapter import (ARMS, LegacyEvidenceAdapter, binary64_exact, constant_intersection,
                      encode, ideal_exact_vector, partial_wire, point_wire, sha256, strict_loads)

MEMORY_LIMIT_BYTES = 8 * 1024 ** 3
MAX_LINE_BYTES = 32 * 1024 ** 2
PREFIXES = (1, 10, 50, 80, 160, 240)


class StreamTimeout(TimeoutError):
    pass


class CorrectnessMismatch(ValueError):
    pass


def _parent_death_setup(parent_pid):
    """Direct child dies with this supervisor, even in its separate session."""
    def setup():
        library = ctypes.CDLL(None, use_errno=True)
        if library.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            os._exit(126)
        if os.getppid() != parent_pid:
            os._exit(125)
    return setup


class ChildRegistry:
    """Own every native/exact child process group; no unrelated PID inspection."""
    def __init__(self):
        self.children = []

    def launch(self, *args, **kwargs):
        if kwargs.get("preexec_fn") is not None or kwargs.get("start_new_session") is not True:
            raise ValueError("workflow children require a fresh session and owned death signal")
        kwargs["preexec_fn"] = _parent_death_setup(os.getpid())
        # The unchanged exact-service wrapper uses a minimal environment. Make
        # the workflow's CPU-only declaration explicit there as well as native.
        environment = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
        environment["CUDA_VISIBLE_DEVICES"] = ""
        kwargs["env"] = environment
        process = subprocess.Popen(*args, **kwargs)
        self.children.append(process)
        command = args[0] if args else kwargs.get("args", [])
        process._workflow_method = (command[command.index("--method") + 1]
                                    if "--method" in command else "native_producer")
        return process

    def cleanup(self, deadline_ns):
        records = []
        for process in self.children:
            killed = False
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    killed = True
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=max(.001, min(2, (deadline_ns - time.perf_counter_ns()) / 1e9)))
                except subprocess.TimeoutExpired:
                    pass
            records.append({"pid": process.pid, "process_group": process.pid,
                            "method": process._workflow_method,
                            "reaped": process.poll() is not None, "exit_code": process.poll(),
                            "cleanup_kill_sent": killed, "parent_death_signal": "SIGKILL"})
        return records


class _ObservedSubprocess:
    def __init__(self, registry):
        self.registry = registry

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, *args, **kwargs):
        return self.registry.launch(*args, **kwargs)


def exact_service_module(config, registry):
    """Load the unchanged pinned supervisor; observe its actual Popen only.

    This module-local proxy never mutates the global subprocess module or the
    old source, and preserves the service's existing timers/checks/kill logic.
    """
    path = Path(config["project_root"]) / "icde2027/family_lp_v7/service.py"
    entry = config["sources"]["icde2027/family_lp_v7/service.py"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("exact service source pin changed")
    module = types.ModuleType("_coat_workflow_owned_exact_service")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    module.subprocess = _ObservedSubprocess(registry)
    return module


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def runtime_configuration(project_root, library_path):
    """Candidate pins only; independent admission belongs to the outer driver."""
    root = Path(project_root).resolve(strict=True)
    sources = {str(path.relative_to(root)): identity(path)
               for path in sorted((root / "icde2027").rglob("*.py"))}
    exact_paths = {"worker": "icde2027/family_lp_v7/service_worker.py",
                   "core": "icde2027/partition_capacity_v6/core.py",
                   "old_receiver": "icde2027/partition_capacity_v6/verify.py",
                   "adapter": "icde2027/family_lp_v7/adapter.py",
                   "general_receiver": "icde2027/family_lp_v7/verify_general.py"}
    exact = {key: identity(root / path) for key, path in exact_paths.items()}
    exact["library"] = identity(library_path)
    return {"schema_version": "coat-workflow-runtime/1", "project_root": str(root),
            "sources": sources, "worker": identity(root / "icde2027/coat_workflow_v8/worker.py"),
            "exact_runtime": exact}


def check_pins(config):
    if config.get("schema_version") != "coat-workflow-runtime/1":
        raise ValueError("runtime schema differs")
    for entry in [config["worker"], *config["sources"].values(), *config["exact_runtime"].values()]:
        if identity(entry["path"]) != entry:
            raise ValueError("runtime pin changed: " + entry["path"])


class NativeWorker:
    """Nonblocking bounded JSON-lines transport, one process group per stream."""
    def __init__(self, config, arm, view, *, cpu, deadline_ns,
                 memory_limit_bytes=MEMORY_LIMIT_BYTES, max_line_bytes=MAX_LINE_BYTES, registry=None):
        self.process = None
        self.selector = selectors.DefaultSelector()
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.max_line_bytes = max_line_bytes
        self.sequence = 0
        self.request_bytes = self.response_bytes = 0
        self.ready = self.diagnostics = None
        self.last_message = None
        self.registry = ChildRegistry() if registry is None else registry
        environment = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC", "CUDA_VISIBLE_DEVICES": ""}
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            environment[name] = "1"
        argv = [sys.executable, "-B", "-I", "-u", config["worker"]["path"],
                "--runtime-json", encode(config).decode("ascii"), "--cpu", str(cpu),
                "--memory-limit-bytes", str(memory_limit_bytes), "--max-line-bytes", str(max_line_bytes)]
        try:
            self.process = self.registry.launch(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, env=environment, start_new_session=True)
            for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                os.set_blocking(pipe.fileno(), False)
            self.selector.register(self.process.stdout, selectors.EVENT_READ, "stdout")
            self.selector.register(self.process.stderr, selectors.EVENT_READ, "stderr")
            self._write({"operation": "initialize", "arm": arm, "legacy_view": view}, deadline_ns)
            ready = self._read(deadline_ns)
            if (ready.get("kind") != "ready" or ready.get("schema_version") != "coat-workflow-native-ready/1"
                    or ready.get("pid") != self.process.pid
                    or ready.get("arm") != arm or ready.get("affinity") != [cpu]
                    or ready.get("memory_limit_bytes") != memory_limit_bytes
                    or ready.get("integer_string_digits") != 100000
                    or ready.get("sources") != {key: entry["sha256"] for key, entry in config["sources"].items()}):
                raise ValueError("native worker readiness refusal or binding mismatch: " + str(ready))
            self.ready = ready
        except BaseException as error:
            error.native_startup_failure = self.close(force=True, deadline_ns=deadline_ns)
            raise

    @staticmethod
    def _remaining(deadline_ns):
        remaining = (deadline_ns - time.perf_counter_ns()) / 1e9
        if remaining <= 0:
            raise StreamTimeout("stream deadline reached")
        return remaining

    def _write(self, value, deadline_ns):
        raw = encode(value)
        if len(raw) > self.max_line_bytes:
            raise ValueError("native command exceeds byte limit")
        wire, offset = raw + b"\n", 0
        self.selector.register(self.process.stdin, selectors.EVENT_WRITE, "stdin")
        try:
            while offset < len(wire):
                for key, _ in self.selector.select(min(.1, self._remaining(deadline_ns))):
                    if key.data == "stdin":
                        try:
                            offset += os.write(key.fd, wire[offset:])
                        except BlockingIOError:
                            pass
                    else:
                        self._drain(key)
            self.request_bytes += len(raw)
        finally:
            try:
                self.selector.unregister(self.process.stdin)
            except (KeyError, ValueError):
                pass

    def _drain(self, key):
        try:
            raw = os.read(key.fd, 65536)
        except BlockingIOError:
            return
        if not raw:
            self.selector.unregister(key.fileobj)
            return
        buffer = self.stdout if key.data == "stdout" else self.stderr
        buffer.extend(raw)
        limit = self.max_line_bytes + 1 if key.data == "stdout" else 1024 ** 2
        if len(buffer) > limit:
            raise ValueError(key.data + " exceeded registered byte limit")

    def _read(self, deadline_ns):
        while True:
            if b"\n" in self.stdout:
                raw, _, trailing = self.stdout.partition(b"\n")
                self.stdout[:] = trailing
                self.response_bytes += len(raw)
                result = strict_loads(raw)
                if type(result) is not dict:
                    raise ValueError("native response must be an object")
                self.last_message = result
                return result
            if self.process.poll() is not None and not self.selector.get_map():
                raise RuntimeError("native process exited without complete response")
            for key, _ in self.selector.select(min(.1, self._remaining(deadline_ns))):
                self._drain(key)

    def query(self, objective, threshold, deadline_ns):
        sequence = self.sequence
        self.sequence += 1
        self._write({"operation": "query", "sequence": sequence,
                     "objective": objective, "threshold": threshold}, deadline_ns)
        result = self._read(deadline_ns)
        if result.get("sequence") != sequence or result.get("kind") not in ("response", "producer_refused"):
            raise CorrectnessMismatch("native response sequence/schema differs")
        if result["kind"] == "response":
            if (type(result.get("actual_native_calls")) is not int or result["actual_native_calls"] < 0
                    or result["actual_native_calls"] != len(result["answer"].get("native_calls", []))):
                raise CorrectnessMismatch("actual native call accounting differs")
        return result

    def close(self, force=False, deadline_ns=None):
        started = time.perf_counter_ns()
        if deadline_ns is None:
            deadline_ns = started + 4_000_000_000
        error = None
        if self.process is not None:
            try:
                if not force and self.process.poll() is None:
                    deadline = min(deadline_ns, time.perf_counter_ns() + 2_000_000_000)
                    self._write({"operation": "stop"}, deadline)
                    self.diagnostics = self._read(deadline)
                    if self.diagnostics.get("kind") != "diagnostics" or self.diagnostics.get("queries") != self.sequence:
                        raise ValueError("native final diagnostics differ")
                    self.process.wait(timeout=max(.001, (deadline - time.perf_counter_ns()) / 1e9))
                    if self.process.returncode != 0:
                        raise RuntimeError("native worker exited uncleanly")
            except Exception as caught:
                error = type(caught).__name__ + ": " + str(caught)
            finally:
                if self.process.poll() is None:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        self.process.wait(timeout=max(.001, min(2, (deadline_ns - time.perf_counter_ns()) / 1e9)))
                    except subprocess.TimeoutExpired:
                        error = "native process group cleanup timed out"
                for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                    if pipe is not None:
                        pipe.close()
        self.selector.close()
        return {"cleanup_ns": time.perf_counter_ns() - started,
                "exit_code": None if self.process is None else self.process.poll(),
                "reaped": self.process is None or self.process.poll() is not None,
                "error": error, "diagnostics": self.diagnostics,
                "request_bytes": self.request_bytes, "response_bytes": self.response_bytes,
                "stderr_bytes": len(self.stderr), "stderr_utf8": self.stderr.decode("utf-8", "replace"),
                "partial_stdout_bytes": len(self.stdout),
                "partial_stdout_utf8": self.stdout.decode("utf-8", "replace"),
                "last_received_message": self.last_message}


class Workflow:
    """Common exact cache/point/failure rules for all four arms."""
    def __init__(self, stream_input, arm, *, native, exact_call, clock_start_ns=None):
        from .receiver import RationalReceiver, program_input_digest
        from icde2027.family_lp_v7.verify_general import family_input_digest
        self.input, self.arm, self.native, self.exact_call = stream_input, arm, native, exact_call
        # The saved task envelope has an identity distinct from the four-field
        # mathematical request accepted by the unchanged exact service. Keep
        # the raw input intact and reject unknown metadata rather than silently
        # dropping arbitrary fields. Four-field inputs support saved toy cases.
        family = stream_input["family_request"]
        fields = ("program", "context_sizes", "exact_weights", "exact_reference")
        if type(family) is not dict or set(family) not in (set(fields), set(fields) | {"schema_version"}):
            raise ValueError("family task envelope fields differ")
        if "schema_version" in family and family["schema_version"] != "verified-coat-family-task/1":
            raise ValueError("unsupported family task envelope schema")
        self.family = {name: family[name] for name in fields}
        family_input_digest(**self.family)
        self.receiver = RationalReceiver(self.family["program"],
                                        expected_program_sha256=program_input_digest(self.family["program"]),
                                        view_sha256=stream_input["view_id"])
        self.legacy = LegacyEvidenceAdapter(stream_input["legacy_view"], self.family["program"])
        bindings = stream_input.get("input_bindings", {})
        for name, actual in (("family_task_id", stream_input["view_id"]),
                             ("program_sha256", self.receiver.program_sha256),
                             ("legacy_view_sha256", self.legacy.checker.digest)):
            if name in bindings and bindings[name] != actual:
                raise ValueError("prepared input binding differs: " + name)
        self.bindings = {**bindings, "family_task_id": stream_input["view_id"],
                         "program_sha256": self.receiver.program_sha256,
                         "legacy_view_sha256": self.legacy.checker.digest}
        seed = stream_input["seed_point_exact"]
        if list(map(binary64_exact, stream_input["legacy_view"]["feasible_point"])) != seed:
            raise ValueError("seed differs between rational receiver and native producer")
        self.receiver.accept_point(point_wire(self.receiver, seed))
        self.constant_point = constant_intersection(self.family)
        if self.constant_point is not None:
            self.receiver.accept_point(point_wire(self.receiver, self.constant_point))
        self.producer_attempts, self.exact_attempts = set(), set()
        self.stats = Counter()
        self.stats["constant_shortcuts"] = int(self.constant_point is not None)
        self.family_record = None
        self.pending = None
        self.seen_objectives = set()
        self.clock_start_ns = time.perf_counter_ns() if clock_start_ns is None else clock_start_ns
        self.timing = Counter()

    @contextmanager
    def timed(self, phase):
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.timing[phase] += time.perf_counter_ns() - started

    def preprocess(self):
        if self.arm not in ("O+S", "O+G"):
            self.family_record = {"status": "not_requested"}
            return self.family_record
        if self.constant_point is not None:
            self.family_record = {"status": "constant_shortcut", "point": self.constant_point,
                                  "family_solve_performed": False}
            return self.family_record
        method = "S" if self.arm == "O+S" else "G_partition"
        self.stats["family_calls"] += 1
        with self.timed("family_service_ns"):
            result = self.exact_call(self.family, method)
        self.family_record = result
        if result.get("verified_success"):
            from icde2027.partition_capacity_v6.verify import input_digest, verify_certificate
            certificate = result["response"]["certificate"]
            with self.timed("receiver_ns"):
                try:
                    verified = verify_certificate(**self.family, certificate=certificate,
                                                  expected_input_sha256=input_digest(**self.family))
                    accepted = self.receiver.accept_point(point_wire(self.receiver, certificate["point"]))
                except ValueError as error:
                    raise CorrectnessMismatch("invalid family certificate: " + str(error)) from error
            result["external_family_receiver"] = verified
            result["accepted_point"] = accepted
        else:
            self.stats["family_failures"] += 1
        return result

    def evaluate(self, bound, wire=None):
        with self.timed("receiver_ns"):
            return self.receiver.evaluate(bound, wire)

    def query(self, query):
        started = time.perf_counter_ns()
        record = {"query": query, "request_sha256": None,
                  "native": None, "legacy_adaptation": None, "exact_fixed": None,
                  "native_point_checks": []}
        self.pending = record
        with self.timed("receiver_ns"):
            canonical_ideal = ideal_exact_vector(query["ideal_objective_exact"])
            bound = self.receiver.bind_request(list(map(binary64_exact, query["objective"])),
                    canonical_ideal, binary64_exact(query["threshold"]),
                    request_index=query["request_index"], policy_index=query["policy_index"])
        objective_key = tuple(map(binary64_exact, query["objective"]))
        pair = (objective_key, binary64_exact(query["threshold"]))
        record["request_sha256"] = bound.request_sha256
        record["ideal_source_mapping"] = {"source": "query.ideal_objective_exact",
                 "canonical": "receipt.ideal.objective", "order": "same coordinate indices; no reordering",
                 "operation": "Fraction(integer numerator, positive integer denominator); no binary64 conversion",
                 "source_sha256": sha256(query["ideal_objective_exact"]),
                 "canonical_sha256": sha256(canonical_ideal)}
        self.stats["requests"] += 1
        cached_before = objective_key in self.seen_objectives
        self.seen_objectives.add(objective_key)
        result = self.evaluate(bound)
        record["before_native_receipt"] = result
        initial_decision = result["represented"]["decision"]
        if initial_decision != "unknown":
            self.stats["common_evidence_hits"] += 1
            self.stats["objective_cache_hits"] += int(cached_before)
            self.stats["point_pool_decisive_hits"] += int(initial_decision == "false")
            self.stats["lower_evidence_decisive_hits"] += int(initial_decision == "true")
            record["route"] = "common_cache_or_point"
        elif pair not in self.producer_attempts:
            self.producer_attempts.add(pair)
            self.stats["producer_attempts"] += 1
            with self.timed("native_service_ns"):
                native = self.native(query["objective"], query["threshold"])
            record["native"] = native
            native_diagnostics = native if native.get("kind") == "response" else native.get("partial_diagnostics") or {}
            self.stats["actual_native_calls"] += native_diagnostics.get("actual_native_calls", 0)
            # Every returned native point is merely a proposal. The shared
            # receiver checks it independently; invalid proposals are retained
            # as explicit rejections, not confused with invalid accepted proof.
            for proposed in native_diagnostics.get("proposed_points", []):
                proposal_record = {"binary64_point": proposed}
                with self.timed("receiver_ns"):
                    try:
                        proposal_record["receipt"] = self.receiver.accept_point(
                            point_wire(self.receiver, list(map(binary64_exact, proposed))))
                        proposal_record["accepted"] = True
                        self.stats["accepted_native_points"] += 1
                    except ValueError as error:
                        proposal_record.update(accepted=False, error=str(error))
                        self.stats["rejected_native_points"] += 1
                record["native_point_checks"].append(proposal_record)
            if native.get("kind") == "response":
                with self.timed("legacy_receiver_ns"):
                    try:
                        wire, adaptation = self.legacy.verify_and_adapt(native["answer"], query["objective"],
                                        query["threshold"], self.receiver, bound)
                    except ValueError as error:
                        raise CorrectnessMismatch("invalid native packet: " + str(error)) from error
                record["legacy_adaptation"] = adaptation
                try:
                    result = self.evaluate(bound, wire)
                except ValueError as error:
                    raise CorrectnessMismatch("adapted native proof rejected: " + str(error)) from error
                adaptation["new_represented_decision"] = result["represented"]["decision"]
                adaptation["new_represented_lower_bound"] = result["represented"]["lower_bound"]
                adaptation["new_represented_upper_bound"] = result["represented"]["upper_bound"]
                old_exact_verdict = "unknown" if adaptation["old_decision"] is None else "true" if adaptation["old_decision"] else "false"
                adaptation["verdict_changed"] = old_exact_verdict != result["represented"]["decision"]
                adaptation["reason"] = ("independent exact weak dual and shared checked points; original packet already validated"
                                        if adaptation["verdict_changed"] else "same decision after exact receiver")
                if old_exact_verdict != "unknown" and result["represented"]["decision"] != old_exact_verdict:
                    raise CorrectnessMismatch("new exact receiver contradicts verified old decision")
            else:
                self.stats["producer_refusals"] += 1
                result = self.evaluate(bound)
            record["route"] = "native_producer"
        else:
            self.stats["failed_pair_not_retried"] += 1
            record["route"] = "previous_pair_unresolved"
        if result["represented"]["decision"] == "unknown" and objective_key not in self.exact_attempts:
            self.exact_attempts.add(objective_key)
            self.stats["exact_fixed_calls"] += 1
            with self.timed("exact_fixed_service_ns"):
                exact = self.exact_call({"program": self.family["program"],
                                         "exact_objective": list(objective_key)}, "G_fixed")
            record["exact_fixed"] = exact
            if exact.get("verified_success"):
                from icde2027.family_lp_v7.verify_general import fixed_input_digest, verify_fixed_certificate
                certificate = exact["response"]["certificate"]
                with self.timed("receiver_ns"):
                    try:
                        exact["external_fixed_receiver"] = verify_fixed_certificate(self.family["program"], list(objective_key),
                             certificate, expected_input_sha256=fixed_input_digest(self.family["program"], list(objective_key)))
                    except ValueError as error:
                        raise CorrectnessMismatch("invalid fixed certificate: " + str(error)) from error
                try:
                    result = self.evaluate(bound, partial_wire(self.receiver, bound, point=certificate["point"],
                             lower=certificate["original_row_lower_dual"], upper=certificate["original_row_upper_dual"]))
                except ValueError as error:
                    raise CorrectnessMismatch("adapted exact certificate rejected: " + str(error)) from error
                if result["represented"]["optimum"] != certificate["optimum"]:
                    raise CorrectnessMismatch("exact fixed optimum differs from external rational receipt")
                record["route"] = "exact_fixed_fallback"
            else:
                self.stats["exact_fixed_failures"] += 1
        elif result["represented"]["decision"] == "unknown" and objective_key in self.exact_attempts:
            self.stats["failed_exact_objective_not_retried"] += 1
        record["receipt"] = result
        record["status"] = "verified" if result["represented"]["decision"] != "unknown" else "unknown"
        record["elapsed_query_ns"] = time.perf_counter_ns() - started
        record["cumulative_ready_ns"] = time.perf_counter_ns() - self.clock_start_ns
        record["point_count"] = self.receiver.point_count
        record["statistics"] = dict(self.stats)
        self.pending = None
        return record


def run_stream(stream_input, *, arm, runtime_config, expected_input_sha256, cpu,
               stream_limit_s=180, memory_limit_bytes=MEMORY_LIMIT_BYTES,
               max_line_bytes=MAX_LINE_BYTES, exact_service_limit_s=10):
    """Run one admitted stream once; preserve partial output on every failure."""
    started = time.perf_counter_ns()
    deadline = started + int(stream_limit_s * 1e9)
    service_deadline = deadline - int(min(5, max(.01, stream_limit_s / 3)) * 1e9)
    record = {"schema_version": "coat-workflow-stream/1", "arm": arm, "status": "input_refusal",
              "complete": False, "input_sha256": expected_input_sha256, "responses": [],
              "family": None, "error": None, "time_ns": {}, "prefixes": {}}
    phases = Counter()
    worker = workflow = None
    active_response = None
    registry = ChildRegistry()
    ready_at = final_response_at = None
    old_affinity, old_guard = None, None
    old_memory = old_alarm_handler = None
    timer_enabled = False
    try:
        phase = time.perf_counter_ns()
        if arm not in ARMS or sha256(stream_input) != expected_input_sha256:
            raise ValueError("unadmitted arm or stream input digest mismatch")
        if (type(cpu) is not int or cpu not in os.sched_getaffinity(0)
                or stream_limit_s <= 0 or stream_limit_s > 180 or exact_service_limit_s <= 0 or exact_service_limit_s > 10
                or memory_limit_bytes != MEMORY_LIMIT_BYTES):
            raise ValueError("registered resource limits differ")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise ValueError("workflow supervisor requires no preexisting real-time alarm")
        old_alarm_handler = signal.getsignal(signal.SIGALRM)
        def deadline_alarm(_signal, _frame):
            raise StreamTimeout("absolute stream service deadline; cleanup budget reserved")
        signal.signal(signal.SIGALRM, deadline_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(.001, (service_deadline - time.perf_counter_ns()) / 1e9))
        timer_enabled = True
        check_pins(runtime_config)
        queries = stream_input["queries"]
        if not isinstance(queries, list) or any(q.get("request_index") != i for i, q in enumerate(queries)):
            raise ValueError("query sequence must retain contiguous original request indices")
        old_affinity, old_guard = os.sched_getaffinity(0), sys.get_int_max_str_digits()
        old_memory = resource.getrlimit(resource.RLIMIT_AS)
        if old_memory[1] != resource.RLIM_INFINITY and old_memory[1] < memory_limit_bytes:
            raise ValueError("inherited hard address-space limit below registered limit")
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, old_memory[1]))
        os.sched_setaffinity(0, {cpu})
        sys.set_int_max_str_digits(100000)
        phases["preflight_ns"] += time.perf_counter_ns() - phase
        exact_module = exact_service_module(runtime_config, registry)

        def exact_call(request, method):
            # Existing service is bounded per phase; reserve its two-second
            # process-group cleanup before dividing the remaining stream time.
            remaining = (service_deadline - time.perf_counter_ns()) / 1e9 - 2.1
            if remaining <= .003:
                raise StreamTimeout("insufficient remaining stream budget for bounded exact service")
            allocation = remaining / 3
            return exact_module.run_call(request, method=method, runtime_config=runtime_config["exact_runtime"], cpu=cpu,
                            startup_limit_s=min(30, allocation), service_limit_s=min(exact_service_limit_s, allocation),
                            diagnostic_limit_s=min(30, allocation), memory_limit_bytes=memory_limit_bytes,
                            max_line_bytes=max_line_bytes)

        phase = time.perf_counter_ns()
        workflow = Workflow(stream_input, arm, native=lambda c, d: worker.query(c, d, service_deadline), exact_call=exact_call)
        phases["common_preparation_ns"] += time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        worker = NativeWorker(runtime_config, arm, stream_input["legacy_view"], cpu=cpu,
                              deadline_ns=min(service_deadline, time.perf_counter_ns() + 30_000_000_000),
                              memory_limit_bytes=memory_limit_bytes, max_line_bytes=max_line_bytes, registry=registry)
        ready_at = time.perf_counter_ns()
        phases["native_startup_ns"] += ready_at - phase
        workflow.clock_start_ns = ready_at
        record["ready"] = worker.ready
        record["family"] = workflow.preprocess()
        record["status"] = "running"
        for query in queries:
            NativeWorker._remaining(service_deadline)
            response = workflow.query(query)
            active_response = response
            phase = time.perf_counter_ns()
            public = {"schema_version": "coat-workflow-public-response/1", "status": response["status"],
                      "request_sha256": response["request_sha256"], "receipt": response["receipt"]}
            raw = encode(public)
            if len(raw) > max_line_bytes:
                raise ValueError("complete workflow response exceeds registered byte limit")
            response["public_response_wire_utf8"] = raw.decode("ascii")
            response["public_response_bytes"] = len(raw)
            response["public_response_sha256"] = hashlib.sha256(raw).hexdigest()
            response["public_certificate_bytes"] = len(encode(response["receipt"]))
            phases["response_encoding_ns"] += time.perf_counter_ns() - phase
            final_response_at = time.perf_counter_ns()
            response["cumulative_full_ns"] = final_response_at - started
            response["cumulative_ready_ns"] = final_response_at - ready_at
            record["responses"].append(response)
            active_response = None
            position = len(record["responses"])
            if position in PREFIXES:
                record["prefixes"][str(position)] = {"full_cumulative_ns": final_response_at - started,
                                                     "ready_cumulative_ns": final_response_at - ready_at}
        record["status"], record["complete"] = "complete", True
    except StreamTimeout as error:
        record["status"], record["error"] = "timeout", str(error)
        if hasattr(error, "native_startup_failure"):
            record["native_startup_failure"] = error.native_startup_failure
    except (CorrectnessMismatch, AssertionError) as error:
        record["status"], record["error"] = "correctness_mismatch", type(error).__name__ + ": " + str(error)
    except Exception as error:
        record["status"] = "input_refusal" if workflow is None else "refused"
        record["error"] = type(error).__name__ + ": " + str(error)
        if hasattr(error, "native_startup_failure"):
            record["native_startup_failure"] = error.native_startup_failure
    finally:
        phase = time.perf_counter_ns()
        if timer_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm_handler)
        if worker is not None:
            record["native_cleanup"] = worker.close(force=not record["complete"], deadline_ns=deadline)
            if record["complete"] and (record["native_cleanup"]["error"] or not record["native_cleanup"]["reaped"]):
                record["complete"], record["status"] = False, "cleanup_incomplete"
        record["children"] = registry.cleanup(deadline)
        if any(not item["reaped"] for item in record["children"]):
            record["complete"], record["status"] = False, "cleanup_incomplete"
        if old_affinity is not None:
            os.sched_setaffinity(0, old_affinity)
            sys.set_int_max_str_digits(old_guard)
        if old_memory is not None:
            resource.setrlimit(resource.RLIMIT_AS, old_memory)
        from icde2027.family_lp_v7.service_worker import runtime_inventory
        record["supervisor_runtime_inventory"] = runtime_inventory()
        phases["cleanup_ns"] += time.perf_counter_ns() - phase
    ended = time.perf_counter_ns()
    # The service ends after bounded cleanup. The following assembly only
    # copies/aggregates already emitted evidence for the caller's run report;
    # its distinct wall clock is excluded from service and included in the
    # outer two-hour execution/logging budget.
    if workflow is not None:
        phases.update(workflow.timing)
        record["statistics"] = dict(workflow.stats)
        record["statistics"]["actual_exact_worker_launches"] = sum(
            item["method"] != "native_producer" for item in record["children"])
        record["retained_points"] = workflow.receiver.export_points()
        record["input_bindings"] = workflow.bindings
        record["incomplete_request"] = workflow.pending if workflow.pending is not None else active_response
        if record["family"] is None:
            record["family"] = workflow.family_record
    answered = len(record["responses"])
    incomplete_count = int(record.get("incomplete_request") is not None)
    native_inflight_unobserved = bool(record.get("incomplete_request")
        and record["incomplete_request"].get("native") is None
        and record.get("statistics", {}).get("producer_attempts", 0) > sum(r.get("native") is not None for r in record["responses"]))
    known_calls = record.get("statistics", {}).get("actual_native_calls", 0)
    record["native_call_accounting"] = {"complete": not native_inflight_unobserved,
        "actual_native_calls": None if native_inflight_unobserved else known_calls,
        "completed_response_call_count": known_calls,
        "inflight_call_count_unknown": native_inflight_unobserved,
        "note": "an interrupted native invocation cannot report its completed diagnostics; never impute zero calls"}
    if "statistics" in record:
        record["statistics"]["completed_response_native_calls"] = known_calls
        if native_inflight_unobserved:
            record["statistics"]["actual_native_calls"] = None
    record["unstarted_request_indices"] = [q["request_index"] for q in stream_input.get("queries", [])[answered + incomplete_count:]]
    record["counts"] = {domain: dict(Counter(r["receipt"][domain]["decision"]
                                  for r in record["responses"])) for domain in ("represented", "ideal")}
    record["bytes"] = {"public_responses": sum(r["public_response_bytes"] for r in record["responses"]),
                       "public_certificates_nested_in_responses": sum(r["public_certificate_bytes"] for r in record["responses"]),
                       "native_ipc_request": record.get("native_cleanup", {}).get("request_bytes", 0),
                       "native_ipc_response": record.get("native_cleanup", {}).get("response_bytes", 0)}
    record["workload_census"] = {"requested_positions": len(stream_input.get("queries", [])),
             "completed_positions": answered, "incomplete_positions": incomplete_count,
             "unstarted_positions": len(record["unstarted_request_indices"])}
    try:
        record["workload_census"].update(
             distinct_represented_objectives=len({tuple(map(binary64_exact, q["objective"])) for q in stream_input.get("queries", [])}),
             distinct_represented_pairs=len({(tuple(map(binary64_exact, q["objective"])), binary64_exact(q["threshold"])) for q in stream_input.get("queries", [])}))
    except (KeyError, TypeError, ValueError) as error:
        record["workload_census"]["census_refusal"] = str(error)
    call_records = ([record["family"]] if isinstance(record["family"], dict) and "verified_success" in record["family"] else [])
    call_records += [r["exact_fixed"] for r in record["responses"] if r["exact_fixed"] is not None]
    if record.get("incomplete_request") and record["incomplete_request"].get("exact_fixed") is not None:
        call_records.append(record["incomplete_request"]["exact_fixed"])
    record["exact_call_outcomes"] = dict(Counter(item["status"] for item in call_records))
    record["refusal_timeout_counts"] = {"unknown_responses": record["counts"]["represented"].get("unknown", 0),
               "native_refusals": record.get("statistics", {}).get("producer_refusals", 0),
               "exact_non_successful_calls": sum(not item.get("verified_success") for item in call_records),
               "stream_timeout": int(record["status"] == "timeout"),
               "incomplete_positions": incomplete_count, "unstarted_positions": len(record["unstarted_request_indices"])}
    record["time_ns"] = {**dict(phases), "full_wall_ns": ended - started,
                         "ready_to_final_response_ns": None if ready_at is None or final_response_at is None else final_response_at - ready_at,
                         "supervisor_residual_ns": ended - started - sum(phases.values())}
    record["timing_scope"] = {"disjoint_phase_keys": list(phases),
               "nested_not_additive": ["ready_to_final_response_ns", "native.producer_ns", "exact service time_ns", "receiver receipts"],
               "full_wall_includes": "preflight, receiver/common initialization, native startup, family/fixed service, IPC, receiver, response encoding, cleanup",
               "excluded": "post-service report assembly, outer disk logging, and post-run mathematical audit"}
    record["resources"] = {"stream_limit_s": stream_limit_s, "absolute_deadline_ns": deadline,
                           "service_deadline_ns": service_deadline, "cleanup_reserved_ns": deadline - service_deadline,
                           "full_wall_exceeded_registered_limit": ended > deadline,
                           "memory_limit_bytes": memory_limit_bytes,
                           "memory_scope": "8 GiB address space per supervisor/native/exact worker process; not tree sum",
                           "exact_service_limit_s": exact_service_limit_s,
                           "exact_limit_scope": "ready to full response; startup/diagnostics/cleanup included in full stream wall"}
    assembled = time.perf_counter_ns()
    record["post_service_report_assembly_ns"] = assembled - ended
    record["total_run_stream_call_wall_ns"] = assembled - started
    return record
