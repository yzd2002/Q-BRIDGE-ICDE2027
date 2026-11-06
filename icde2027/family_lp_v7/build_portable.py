"""Build the existing bridge from a supplied PIC SoPlex archive.

This helper installs nothing and does not run a scientific experiment. Supply
the compiler, headers and libraries available in the reproduction environment.
The historical host-specific build remains a separate immutable record.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soplex-source", type=Path, required=True)
    parser.add_argument("--soplex-library", type=Path, required=True)
    parser.add_argument("--include-dir", type=Path, action="append", default=[])
    parser.add_argument("--library-dir", type=Path, action="append", default=[])
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError("Requested CPU is unavailable")
    os.sched_setaffinity(0, {args.cpu})
    source = args.soplex_source.resolve(strict=True)
    archive = args.soplex_library.resolve(strict=True)
    own = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(own):
        raise ValueError("Use a new output directory outside the source trees")
    includes = [p.resolve(strict=True) for p in args.include_dir]
    libraries = [p.resolve(strict=True) for p in args.library_dir]
    if any(not p.is_dir() for p in includes + libraries):
        raise ValueError("Include/library paths must be directories")
    if any("," in str(p) or ":" in str(p) for p in libraries):
        raise ValueError("Library paths cannot contain linker-list separators")
    compiler = shutil.which(args.cxx)
    if compiler is None:
        raise ValueError("C++ compiler was not found")
    compiler_path = Path(compiler).resolve(strict=True)
    config = source / "src/soplex/config.h"
    config_text = config.read_text()
    for flag in ("SOPLEX_WITH_GMP", "SOPLEX_WITH_BOOST", "SOPLEX_WITH_MPFR"):
        if "#define " + flag not in config_text:
            raise ValueError("SoPlex must be built with " + flag)
    input_paths = [own / "bridge.cpp", own / "bridge.h", Path(__file__).resolve(),
                   archive, config, source / "src/soplex.h", compiler_path]
    # All source/header bytes are recorded, including locally generated config.
    input_paths.extend(p for p in (source / "src").rglob("*") if p.is_file())
    inputs = {str(p): identity(p) for p in sorted(set(input_paths))}
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True,
                             text=True, timeout=30).stdout
    output.mkdir(parents=True, exist_ok=False)
    binary = output / "libfamily_lp_v7.so"
    command = [compiler, "-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-shared",
               "-I" + str(source / "src")]
    command += ["-I" + str(p) for p in includes]
    command += [str(own / "bridge.cpp"), str(archive)]
    for path in libraries:
        command += ["-L" + str(path), "-Wl,-rpath," + str(path)]
    command += ["-lgmp", "-lmpfr", "-o", str(binary)]
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    started = time.monotonic()
    code, error, process = None, None, None
    cleanup_errors = []
    try:
        with (output / "build.log").open("xb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            code = process.wait(timeout=600)
    except Exception as exception:
        error = type(exception).__name__ + ": " + str(exception)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as cleanup_error:
                cleanup_errors.append(type(cleanup_error).__name__ + ": " + str(cleanup_error))
            try:
                process.wait(timeout=10)
            except Exception as cleanup_error:
                cleanup_errors.append(type(cleanup_error).__name__ + ": " + str(cleanup_error))
    try:
        unchanged = all(identity(Path(path)) == value for path, value in inputs.items())
    except OSError as exception:
        unchanged = False
        error = (error + "; " if error else "") + "postflight: " + str(exception)
    library_identity = None
    if code == 0:
        try:
            with binary.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise ValueError("Compiler did not produce an ELF shared-library file")
            library_identity = identity(binary)
        except (OSError, ValueError) as exception:
            error = (error + "; " if error else "") + "output: " + str(exception)
    report = {"status": "PASS" if code == 0 and unchanged and error is None and not cleanup_errors and library_identity is not None else "FAIL",
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.monotonic() - started,
              "scope": "Bridge compilation only; no dependency installation or scientific solve",
              "command": command, "compiler_version": version,
              "compiler_binary": {"path": str(compiler_path),
                                  **inputs[str(compiler_path)]},
              "provided_include_directories": [str(p) for p in includes],
              "provided_library_directories": [str(p) for p in libraries],
              "input_identity_scope": "Bridge/helper files, supplied SoPlex source tree, static archive and compiler binary. Transitive external/system headers, linker and shared libraries are not a complete captured toolchain closure. A new build is not a claim of bit-identical or timing-identical reproduction.",
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "exit_code": code,
              "error": error, "cleanup_errors": cleanup_errors,
              "inputs": inputs, "inputs_unchanged": unchanged,
              "library": library_identity}
    (output / "BUILD_RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "library": report["library"]}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
