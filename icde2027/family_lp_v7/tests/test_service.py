"""Synthetic protocol/resource probes and real tiny S/G integration only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "icde2027/family_lp_v7/service.py"
s = types.ModuleType("family_service_synthetic_tests")
s.__file__ = str(SOURCE)
exec(compile(SOURCE.read_bytes(), str(SOURCE), "exec"), s.__dict__)
CPU = 12


def identity(path):
    path = path.absolute()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def configuration():
    return {key: identity(ROOT / path) for key, path in {
        "worker": "icde2027/family_lp_v7/service_worker.py",
        "core": "icde2027/partition_capacity_v6/core.py",
        "old_receiver": "icde2027/partition_capacity_v6/verify.py",
        "adapter": "icde2027/family_lp_v7/adapter.py",
        "general_receiver": "icde2027/family_lp_v7/verify_general.py",
        "library": "runs/icde2027/analysis/family_lp_v7_implementation/build_attempt_04/libfamily_lp_v7.so",
    }.items()}


def family():
    return {"program": {"lower": ["1/3", "0/1"], "upper": ["1/1", "1/4"], "rows": []},
            "context_sizes": [2], "exact_weights": ["1/1"], "exact_reference": [["1/2", "1/2"]]}


PRELUDE = '''import argparse,hashlib,json,os,resource,signal,sys,time
p=argparse.ArgumentParser()
for name in ('runtime-json','method','cpu','memory-limit-bytes','max-line-bytes'):p.add_argument('--'+name)
a=p.parse_args();c=json.loads(a.runtime_json);cpu=int(a.cpu);memory=int(a.memory_limit_bytes)
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.sched_setaffinity(0,{cpu})
sys.set_int_max_str_digits(100000)
def dump(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def send(v):sys.stdout.buffer.write(dump(v)+b'\\n');sys.stdout.buffer.flush()
ready={'schema_version':'family-lp-worker-ready/1','ready':True,'method':a.method,'pid':os.getpid(),
       'setup_ns':0,'affinity':[cpu],'memory_limit_bytes':memory,'integer_string_digits':sys.get_int_max_str_digits(),
       'loaded_sources':{k:c[k]['sha256'] for k in ('core','old_receiver')}}
response={'status':'refused','certificate':None,'receiver_receipt':None}
def diagnostics(raw):return {'schema_version':'family-lp-worker-diagnostics/1','method':a.method,
  'request_sha256':hashlib.sha256(raw.rstrip(b'\\n')).hexdigest(),'complete':True,'operation':None}
'''


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CPU not in os.sched_getaffinity(0):
            raise RuntimeError("synthetic tests are assigned CPU12")
        cls.config = configuration()

    def call(self, request=None, method="S", config=None, **kwargs):
        return s.run_call(family() if request is None else request, method=method,
                          runtime_config=self.config if config is None else config, cpu=CPU, **kwargs)

    def fixture(self, body, **kwargs):
        with tempfile.TemporaryDirectory(prefix="family-service-synthetic-") as directory:
            path = Path(directory) / "fixture.py"
            path.write_text(PRELUDE + body)
            config = copy.deepcopy(self.config)
            config["worker"] = identity(path)
            return self.call(config=config, **kwargs)

    def successful(self, record):
        self.assertTrue(record["record_complete"], record)
        self.assertTrue(record["verified_success"], record)
        self.assertEqual(record["exit_code"], 0)
        self.assertTrue(record["reaped"])
        self.assertGreater(record["peak_memory"]["peak_rss_kib"], 0)
        self.assertGreater(record["time_ns"]["service_ns"], 0)
        self.assertGreaterEqual(record["time_ns"]["supervision_unassigned_ns"], 0)
        self.assertEqual(record["diagnostics"]["operation"]["mathematical_receiver_invocations"], 1)
        self.assertEqual(record["diagnostics"]["request_sha256"], record["request_sha256"])
        self.assertEqual(record["integer_string_digits"], 100000)
        self.assertEqual(record["ready"]["integer_string_digits"], 100000)

    def long_fraction_call(self, method):
        caller_guard = sys.get_int_max_str_digits()
        denominator = '1' + '0'*4999 + '1'
        self.assertEqual(len(denominator), 5001)
        exact = '1/' + denominator
        program = {'lower': [exact], 'upper': [exact], 'rows': []}
        if method == 'G_fixed':
            request = {'program': program, 'exact_objective': ['1/1']}
        else:
            request = {'program': program, 'context_sizes': [1], 'exact_weights': ['1/1'], 'exact_reference': [['1/1']]}
        r = self.call(request, method=method)
        self.successful(r)
        self.assertEqual(r['response']['certificate']['point'], [exact])
        self.assertEqual(r['response']['certificate']['optimum' if method == 'G_fixed' else 'capacity'], exact if method == 'G_fixed' else '0/1')
        self.assertEqual(sys.get_int_max_str_digits(), caller_guard)
        return r

    def test_long_fraction_S_preserves_registered_domain(self):
        self.long_fraction_call('S')

    def test_long_fraction_G_partition_preserves_registered_domain(self):
        self.long_fraction_call('G_partition')

    def test_long_fraction_G_general_preserves_registered_domain(self):
        self.long_fraction_call('G_general')

    def test_long_fraction_G_fixed_preserves_registered_domain(self):
        self.long_fraction_call('G_fixed')

    def test_wrong_ready_integer_limit_rejected(self):
        r = self.fixture("ready['integer_string_digits']=4300\nsend(ready)\ntime.sleep(5)\n")
        self.killed(r, 'protocol_error')
        self.assertIn('integer-string', r['error'])

    def killed(self, record, status):
        self.assertEqual(record["status"], status, record)
        self.assertFalse(record["record_complete"])
        self.assertFalse(record["verified_success"])
        self.assertTrue(record["reaped"])
        self.assertTrue(record["kill_sent"])
        with self.assertRaises(ProcessLookupError):
            os.kill(record["worker_pid"], 0)

    def test_real_S(self):
        r = self.call(); self.successful(r)
        self.assertEqual(r["response"]["certificate"]["capacity"], "1/24")
        self.assertEqual(r["response"]["certificate"]["contexts"], [{}])
        self.assertEqual(r["response"]["certificate"]["internal_check"], {})
        self.assertTrue(r["diagnostics"]["operation"]["construction_metadata"]["contexts"])
        inventory = r["diagnostics"]["runtime_inventory"]
        self.assertIn(str((ROOT/'icde2027/partition_capacity_v6/core.py').resolve()), inventory["module_files"])
        self.assertFalse(inventory["missing_module_files"])
        self.assertFalse(inventory["missing_mapped_files"])
        self.assertFalse(inventory["deleted_mappings"])

    def test_real_G_partition_matches_S(self):
        g = self.call(method="G_partition"); self.successful(g)
        self.assertEqual(g["response"]["certificate"]["capacity"], "1/24")
        self.assertNotIn("core", g["ready"]["loaded_sources"])
        self.assertIn("native_solution", g["diagnostics"]["operation"])
        self.assertNotIn("native_solution", g["response"])
        inventory = g["diagnostics"]["runtime_inventory"]
        self.assertIn(str(Path(self.config['library']['path']).resolve()), inventory["mapped_files"])
        self.assertTrue(any('libgmp' in p for p in inventory['mapped_files']))
        self.assertNotIn(str((ROOT/'icde2027/partition_capacity_v6/core.py').resolve()), inventory["module_files"])
        for key in ('module_files','mapped_files','module_cache_files'):
            self.assertEqual(inventory[key], sorted(set(inventory[key])))
            self.assertTrue(all(Path(p).is_absolute() and Path(p).is_file() for p in inventory[key]))

    def test_real_G_general_overlap(self):
        request = family()
        request["program"]["rows"] = [
            {"coefficients": ["1/1", "1/1"], "lower": "1/2", "upper": "1/2"},
            {"coefficients": ["1/1", "-1/1"], "lower": "1/6", "upper": "1/6"}]
        r = self.call(request, method="G_general"); self.successful(r)
        self.assertEqual(r["response"]["certificate"]["capacity"], "1/12")

    def test_real_G_fixed(self):
        request = {"program": family()["program"], "exact_objective": ["3/1", "-2/1"]}
        r = self.call(request, method="G_fixed"); self.successful(r)
        self.assertEqual(r["response"]["certificate"]["optimum"], "1/2")

    def test_real_invalid_request_is_complete_refusal(self):
        request = family(); request["exact_weights"] = ["2/1"]
        r = self.call(request)
        self.assertTrue(r["record_complete"], r)
        self.assertEqual(r["status"], "refused")
        self.assertFalse(r["verified_success"])
        self.assertIsNone(r["response"]["certificate"])
        self.assertTrue(r["diagnostics"]["error"])
        self.assertEqual(r["diagnostics"]["failure_stage"], "request_binding")
        self.assertGreater(r["diagnostics"]["worker_timing_ns"]["request_binding_ns"], 0)

    def test_real_generic_refusal_keeps_adapter_diagnostics(self):
        request = family(); request["exact_weights"] = ["2/1"]
        r = self.call(request, method="G_partition")
        self.assertTrue(r["record_complete"], r)
        self.assertEqual(r["status"], "refused")
        self.assertEqual(r["diagnostics"]["failure_stage"], "generic_call")
        self.assertIn("timing_ns", r["diagnostics"]["operation"])

    def test_real_wrong_source_hash_refuses_initialization(self):
        config = copy.deepcopy(self.config); config["core"]["sha256"] = "0" * 64
        r = self.call(config=config)
        self.assertEqual(r["status"], "initialization_refused", r)
        self.assertFalse(r["record_complete"])
        self.assertIsNone(r["response"])

    def test_wrong_worker_hash_no_process(self):
        config = copy.deepcopy(self.config); config["worker"]["sha256"] = "0" * 64
        r = self.call(config=config)
        self.assertEqual(r["status"], "input_refusal")
        self.assertNotIn("worker_pid", r)

    def test_runtime_other_method_files_not_read(self):
        config = copy.deepcopy(self.config)
        for key in ("adapter", "library", "general_receiver"):
            config[key]["path"] = "/does/not/exist"
        self.successful(self.call(config=config))
        config = copy.deepcopy(self.config)
        config["core"]["path"] = "/does/not/exist"
        self.successful(self.call(method="G_partition", config=config))

    def test_startup_timeout(self):
        r = self.fixture("time.sleep(5)\n", startup_limit_s=0.15)
        self.killed(r, "startup_timeout")

    def test_service_timeout(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\ntime.sleep(5)\n", service_limit_s=0.1)
        self.killed(r, "service_timeout")

    def test_diagnostics_timeout_preserves_public_response(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nsend(response)\ntime.sleep(5)\n", diagnostic_limit_s=0.1)
        self.killed(r, "diagnostic_timeout")
        self.assertIsNotNone(r["response"])

    def test_post_diagnostics_exit_timeout(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nsend(response)\nsend(diagnostics(raw))\ntime.sleep(5)\n", diagnostic_limit_s=0.1)
        self.killed(r, "diagnostic_timeout")
        self.assertIsNotNone(r["diagnostics"])

    def test_complete_response_missing_diagnostics(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nsend(response)\n")
        self.assertEqual(r["status"], "premature_eof", r)
        self.assertFalse(r["record_complete"])
        self.assertIsNotNone(r["response"])

    def test_crash_is_explicit(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nos.kill(os.getpid(),signal.SIGSEGV)\n")
        self.assertEqual(r["status"], "process_exit", r)
        self.assertEqual(r["exit_code"], -signal.SIGSEGV)
        self.assertFalse(r["record_complete"])

    def test_stderr_pipe_is_drained(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nos.write(2,b'x'*200000)\nsend(response)\nsend(diagnostics(raw))\n")
        self.assertTrue(r["record_complete"], r)
        self.assertEqual(r["stderr_bytes"], 200000)

    def test_stderr_limit(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nos.write(2,b'x'*200000)\ntime.sleep(5)\n", max_stderr_bytes=1024)
        self.killed(r, "resource_refusal")
        self.assertFalse(r["stderr_capture_complete"])

    def test_partial_stdout_lines(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nwire=dump(response)+b'\\n'\nfor i in range(0,len(wire),3):\n os.write(1,wire[i:i+3]);time.sleep(.001)\nsend(diagnostics(raw))\n")
        self.assertTrue(r["record_complete"], r)

    def test_partial_request_writes(self):
        request = family(); request["padding"] = "x" * 200000
        r = self.fixture("send(ready)\nraw=b''\nwhile not raw.endswith(b'\\n'):\n raw+=os.read(0,1000);time.sleep(.001)\nsend(response)\nsend(diagnostics(raw))\n", request=request)
        self.assertTrue(r["record_complete"], r)
        self.assertGreater(r["request_bytes"], 200000)

    def test_blocked_request_write_times_out(self):
        request = family(); request["padding"] = "x" * 200000
        r = self.fixture("send(ready)\ntime.sleep(5)\n", request=request, service_limit_s=0.1)
        self.killed(r, "service_timeout")

    def test_duplicate_response_key_rejected(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nos.write(1,b'{\"status\":\"refused\",\"status\":\"refused\",\"certificate\":null,\"receiver_receipt\":null}\\n')\n")
        self.assertEqual(r["status"], "protocol_error", r)

    def test_nonfinite_response_rejected(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nos.write(1,b'{\"x\":NaN}\\n')\n")
        self.assertEqual(r["status"], "protocol_error", r)

    def test_wrong_diagnostic_binding(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nsend(response)\nd=diagnostics(raw);d['request_sha256']='0'*64;send(d)\n")
        self.assertEqual(r["status"], "protocol_error", r)
        self.assertFalse(r["record_complete"])

    def test_extra_stdout_rejected(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nsend(response)\nsend(diagnostics(raw))\nprint('extra',flush=True)\n")
        self.assertEqual(r["status"], "protocol_error", r)

    def test_oversized_unterminated_stdout(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nos.write(1,b'x'*20000)\ntime.sleep(5)\n", max_line_bytes=4096)
        self.killed(r, "resource_refusal")

    def test_diagnostic_resource_refusal(self):
        r = self.fixture("send(ready)\nraw=sys.stdin.buffer.readline()\nsend(response)\nd=diagnostics(raw);d.update(complete=False,resource_refusal=True,error='limit');send(d)\n")
        self.assertEqual(r["status"], "resource_refusal", r)
        self.assertFalse(r["record_complete"])

    def test_input_size_refusal_no_process(self):
        r = self.call(max_line_bytes=10)
        self.assertEqual(r["status"], "resource_refusal")
        self.assertNotIn("worker_pid", r)

    def test_invalid_cpu_no_process(self):
        r = s.run_call(family(), method="S", runtime_config=self.config, cpu=-1)
        self.assertEqual(r["status"], "input_refusal")
        self.assertNotIn("worker_pid", r)

    def test_nonfinite_timeout_no_process(self):
        r = self.call(service_limit_s=float("nan"))
        self.assertEqual(r["status"], "input_refusal")
        self.assertNotIn("worker_pid", r)

    def test_float_request_no_process(self):
        request = family(); request["exact_weights"] = [1.0]
        r = self.call(request)
        self.assertEqual(r["status"], "input_refusal")
        self.assertNotIn("worker_pid", r)

    def test_invalid_method_still_json_safe(self):
        r = self.call(method=object())
        self.assertEqual(r["status"], "input_refusal")
        json.dumps(r, allow_nan=False)

    def test_subnanosecond_limit_no_process(self):
        r = self.call(service_limit_s=1e-20)
        self.assertEqual(r["status"], "input_refusal")
        self.assertNotIn("worker_pid", r)

    def test_actual_worker_enforces_virtual_memory_limit(self):
        with tempfile.TemporaryDirectory(prefix="family-service-memory-") as directory:
            source = Path(directory) / "allocation_core.py"
            source.write_text("def solve(*args):\n    return bytearray(256*1024**2)\n")
            config = copy.deepcopy(self.config); config["core"] = identity(source)
            r = self.call(config=config, memory_limit_bytes=64*1024**2)
        self.assertTrue(r["record_complete"], r)
        self.assertEqual(r["status"], "resource_refusal", r)
        self.assertIn("MemoryError", r["diagnostics"]["error"])

    def test_actual_public_size_limit_is_not_truncated_success(self):
        r = self.call(max_line_bytes=1600)
        self.assertEqual(r["status"], "resource_refusal", r)
        self.assertFalse(r["verified_success"])
        self.assertEqual(r["response"]["status"], "refused")
        self.assertIsNone(r["response"]["certificate"])

    def test_timeout_kills_accidental_process_group_child(self):
        with tempfile.TemporaryDirectory(prefix="family-service-child-") as directory:
            pidfile = Path(directory)/"child.pid"
            body = ("send(ready)\nraw=sys.stdin.buffer.readline()\nchild=os.fork()\n"
                    "if child==0:\n time.sleep(5);os._exit(0)\n"
                    "open(" + repr(str(pidfile)) + ",'w').write(str(child))\ntime.sleep(5)\n")
            r = self.fixture(body, service_limit_s=0.2)
            self.killed(r, "service_timeout")
            child_pid = int(pidfile.read_text())
            # A reparented zombie may briefly remain until the host's reaper
            # handles it. It must not still be executing after group SIGKILL.
            stat = Path('/proc')/str(child_pid)/'stat'
            if stat.exists():
                self.assertEqual(stat.read_text().split()[2], 'Z')


if __name__ == "__main__":
    unittest.main(verbosity=2)
