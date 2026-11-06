"""Registered single-process CPU execution benchmarks; no simulated timings."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd
import scipy

from .aggregate import count_events,make_group_moments,solve_bound
from .experiment import json_write


def benchmark(out):
    if out.exists():
        raise SystemExit("Refusing existing output")
    out.mkdir(parents=True)
    manifest={"host":platform.node(),"platform":platform.platform(),"python":platform.python_version(),
              "numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"highs_threads":1,
              "thread_environment":{k:os.getenv(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},
              "aggregate_sha256":hashlib.sha256(Path(__file__).with_name("aggregate.py").read_bytes()).hexdigest(),
              "protocol":"runs/icde2027/systems_protocol.md","source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "repeats":5,"status":"running"}
    json_write(out/"manifest.json",manifest)
    aggregation=[]
    for n in [10000,100000,1000000,10000000]:
        rng=np.random.default_rng(602700+n)
        start=time.perf_counter()
        x=rng.integers(0,8,size=n,dtype=np.int32)
        a=rng.integers(0,8,size=n,dtype=np.int32)
        y=rng.integers(0,2,size=n,dtype=np.int8)
        generation=time.perf_counter()-start
        start=time.perf_counter()
        frame=pd.DataFrame({"x":x,"a":a,"y":y},copy=False)
        frame_seconds=time.perf_counter()-start
        for kind in ["bincount_stream","pandas_groupby"]:
            times=[]
            for repeat in range(6):
                start=time.perf_counter()
                if kind=="bincount_stream":
                    counts=np.zeros((8,8,2),dtype=np.int64)
                    for offset in range(0,n,100000):
                        counts += count_events(x[offset:offset+100000],a[offset:offset+100000],y[offset:offset+100000],8,8)
                else:
                    grouped=frame.groupby(["x","a","y"],sort=False,observed=True).size()
                    counts=np.zeros((8,8,2),dtype=np.int64)
                    for key,value in grouped.items():
                        counts[key]=value
                times.append(time.perf_counter()-start)
                if repeat==0 and kind=="bincount_stream":
                    reference=counts.copy()
                if not np.array_equal(reference,counts):
                    raise AssertionError("row aggregation mismatch")
            aggregation.append({"n":n,"method":kind,"cold_seconds":times[0],"warm_seconds":times[1:],
                                "generation_seconds_excluded":generation,"dataframe_seconds_separate":frame_seconds,
                                "input_array_bytes":x.nbytes+a.nbytes+y.nbytes,"aggregate_state_bytes":counts.nbytes,
                                "count_total":int(counts.sum()),"count_sha256":hashlib.sha256(counts.tobytes()).hexdigest()})
        json_write(out/"aggregation.json",aggregation)
        print(f"aggregated actual rows n={n}",flush=True)
        del frame,x,a,y
    solvers=[]
    for d in [64,256,1024,4096,16384]:
        rng=np.random.default_rng(702700+d)
        nx,na=8,d//8
        start=time.perf_counter()
        rows=make_group_moments(np.ones((nx,na)),np.arange(na)//4)
        compilation=time.perf_counter()-start
        lower=rng.uniform(.05,.25,d)
        upper=rng.uniform(.75,.95,d)
        truth=rng.uniform(.3,.7,d)
        center=rows@truth
        epsilon=np.full(len(center),.01)
        c=rng.normal(size=(nx,na))
        c-=c.mean(1,keepdims=True)
        c=(c/np.sum(abs(c))).ravel()
        for backend in ["knapsack","highs"]:
            results=[]
            for _ in range(6):
                result=solve_bound(c,lower,upper,rows,center-epsilon,center+epsilon,backend=backend,highs_threads=1)
                if result.status!="ok":
                    raise RuntimeError(result)
                results.append(result.to_dict())
            if backend=="knapsack":
                objective=results[0]["lower_bound"]
            if abs(objective-results[0]["lower_bound"])>1e-7:
                raise AssertionError("backend numerical disagreement")
            solvers.append({"cells":d,"moments":rows.shape[0],"nonzeros":rows.nnz,"backend":backend,
                            "compilation_seconds":compilation,"sparse_plan_bytes":rows.data.nbytes+rows.indices.nbytes+rows.indptr.nbytes,
                            "cold":results[0],"warm":results[1:]})
        json_write(out/"solvers.json",solvers)
        print(f"solved cells={d}",flush=True)
    manifest["status"]="complete"
    manifest["process_thread_count_at_end"] = next(line.split(":")[1].strip() for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Threads:"))
    manifest["peak_process_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest["rss_scope"]="overall benchmark process including generated input arrays and both methods; not a per-method comparison"
    json_write(out/"manifest.json",manifest)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    benchmark(parser.parse_args().out)
