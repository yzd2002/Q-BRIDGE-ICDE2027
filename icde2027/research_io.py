"""Small provenance helpers for the registered CPU studies."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import platform
import shutil
import time
from importlib.metadata import version,PackageNotFoundError

import numpy as np
import scipy

from .experiment import json_write

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start_run(out, protocol):
    out=Path(out);protocol=Path(protocol)
    if out.exists():
        raise ValueError(f"Refusing to overwrite existing run: {out}")
    out.mkdir(parents=True)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    hashes={}
    for path in sorted((ROOT/'icde2027').glob('*.py')):
        shutil.copy2(path,snapshot/path.name);hashes[path.name]=sha256(path)
    shutil.copy2(protocol,out/'protocol.json')
    cpu=next((line.partition(':')[2].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
              if line.startswith('model name')),'unknown')
    manifest={'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'protocol_sha256':sha256(protocol),'source_sha256':hashes,
              'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'host':platform.node(),'platform':platform.platform(),'cpu':cpu,
              'cpu_affinity':sorted(os.sched_getaffinity(0)), 'load_at_start':list(os.getloadavg()),
              'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
              'native_highs_threads':1,'gpu_used':False,'git':'workspace is not a git repository; immutable source snapshot and hashes recorded'}
    try:
        manifest['highspy']=version('highspy')
    except PackageNotFoundError:
        manifest['highspy']=None
    from scipy.optimize._highspy import _core
    manifest['scipy_bundled_highs']='.'.join(str(getattr(_core,k)) for k in ['HIGHS_VERSION_MAJOR','HIGHS_VERSION_MINOR','HIGHS_VERSION_PATCH'])
    json_write(out/'manifest.json',manifest)
    return manifest


def finish_run(out,manifest,**details):
    manifest.update(details,status='complete',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    json_write(Path(out)/'manifest.json',manifest)
