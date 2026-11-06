"""Compile only this bridge against the already-built parent-owned SoPlex archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def identity(path):
    raw=path.read_bytes();return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soplex-source',type=Path,required=True)
    parser.add_argument('--soplex-library',type=Path,help='Explicit PIC archive; default is this source tree library')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if sorted(os.sched_getaffinity(0))!=[10]:raise ValueError('Compile this wrapper on allocated CPU10')
    source=args.soplex_source.absolute();out=args.output.absolute();out.mkdir(parents=True,exist_ok=False)
    own=Path(__file__).absolute().parent
    archive=args.soplex_library.absolute() if args.soplex_library else source/'lib/libsoplex-8.1.0.linux.x86_64.gnu.opt.a'
    config=source/'src/soplex/config.h'
    if not all(flag in config.read_text() for flag in ('#define SOPLEX_WITH_GMP','#define SOPLEX_WITH_BOOST','#define SOPLEX_WITH_MPFR')):
        raise ValueError('Parent exact dependency build is not ready')
    binary=out/'libfamily_lp_v7.so'
    command=['g++','-std=c++17','-O3','-DNDEBUG','-fPIC','-shared','-I'+str(source/'src'),
        '-I/home/chaojin/newconda/include',str(own/'bridge.cpp'),str(archive),
        '-L/home/chaojin/newconda/lib','-Wl,-rpath,/home/chaojin/newconda/lib','-lgmp','-lmpfr','-o',str(binary)]
    inputs={str(p):identity(p) for p in (own/'bridge.cpp',own/'bridge.h',archive,config,source/'src/soplex.h')}
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ('bridge.cpp','bridge.h','build_wrapper.py'):(snapshot/name).write_bytes((own/name).read_bytes())
    start=time.monotonic()
    with (out/'build.log').open('wb') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    record={'scope':'wrapper build only; no dependency rebuild or scientific input','command':command,
        'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start,
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'cpu_affinity':[10],
        'compiler':subprocess.run(['g++','--version'],capture_output=True,text=True).stdout,
        'inputs':inputs,'inputs_unchanged':all(identity(Path(p))==v for p,v in inputs.items()),
        'dependency_commit':'a00610534e044f35aa298059eb963312336c4c9c',
        'dependency_versions':{'SoPlex':'8.1.0','Boost':'1.73.0','GMP':'6.2.1','MPFR':'4.0.2'}}
    if result.returncode==0:
        record['library']=identity(binary)
        links=subprocess.run(['ldd',str(binary)],capture_output=True,text=True)
        (out/'dynamic_links.txt').write_text(links.stdout+links.stderr)
        record['dynamic_links_exit_code']=links.returncode
    (out/'BUILD_RESULT.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    print(json.dumps(record,sort_keys=True))
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
