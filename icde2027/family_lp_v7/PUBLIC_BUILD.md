# Rebuilding the exact family-solver bridge

Certificate replay is the lightweight path: it uses the saved inputs, certificates and fixed standard-library receivers, and does not require SoPlex. Building the optimizer is a separate task. Neither path reproduces the paper's wall-clock measurements by itself.

The recorded optimizer is official SoPlex 8.1.0, commit `a00610534e044f35aa298059eb963312336c4c9c`. Its [official source archive](https://codeload.github.com/scipopt/soplex/tar.gz/a00610534e044f35aa298059eb963312336c4c9c) has SHA-256 `67630bc95203933a3f1145ba7fad512b7f68f59dc18029eee74ac381298f56d8` and 1,234,602 bytes. Verify this identity before extracting it into a new directory. Retain the upstream Apache-2.0 license and notices. The bridge source does not redistribute the account's dependency installation.

The tested environment is Linux x86-64, Python 3.12.3, GCC 11.4.0, Boost 1.73.0, GMP 6.2.1 and MPFR 4.0.2. Provide a compatible C++17 compiler and the existing development headers/libraries. The helpers install nothing. The following Makefile settings are those used in the recorded PIC build; set `QBRIDGE_SOPLEX` to the extracted source directory and `QBRIDGE_DEPS` to the prefix containing the dependencies. These variables are local reproduction settings, not paths built into the source.

```bash
make -C "$QBRIDGE_SOPLEX" -j1 COMP=gnu OPT=opt BOOST=true GMP=true MPFR=true ZLIB=false \
  "GMP_CPPFLAGS=-I$QBRIDGE_DEPS/include" \
  "GMP_LDFLAGS=-L$QBRIDGE_DEPS/lib -Wl,-rpath,$QBRIDGE_DEPS/lib -lgmp" \
  USRCPPFLAGS=-fPIC makelibfile
```

The dependency prefix in these Makefile flags must not contain whitespace, commas or colons. The public Python bridge builder separately supports spaces in its source/output paths. On this Linux/GNU configuration the static archive is `lib/libsoplex-8.1.0.linux.x86_64.gnu.opt.a` under the extracted source.

From the artifact's source root, choose one CPU allocated to your job and a new output directory outside both source trees:

```bash
python -B -I -S icde2027/family_lp_v7/build_portable.py \
  --soplex-source "$QBRIDGE_SOPLEX" \
  --soplex-library "$QBRIDGE_SOPLEX/lib/libsoplex-8.1.0.linux.x86_64.gnu.opt.a" \
  --include-dir "$QBRIDGE_DEPS/include" \
  --library-dir "$QBRIDGE_DEPS/lib" \
  --cpu "$QBRIDGE_CPU" --output "$QBRIDGE_BUILD"
```

`BUILD_RESULT.json` records the actual command, compiler identity, supplied directories, tracked source/archive identities and resulting library hash. `build.log` retains compiler output. Existing output directories are rejected. Compilation has a 600-second limit and bounded cleanup; failed or missing/non-ELF outputs cannot receive PASS. The tracked inputs are not a complete captured toolchain: transitive system/external headers, linker and shared libraries remain environmental dependencies. Different compatible builds need not have identical binary hashes or timings.

A separate recorded rebuild in the original environment produced a library byte-identical to the original (`3c89dda0b8a65db8b16b28a4a988153a4a4c73f01f881102b02a2d68f0742ee0`) and passed small exact-certificate checks. This is one observed rebuild, not a cross-platform guarantee. The historical `build_wrapper.py` is retained as an experiment record; use `build_portable.py` for explicit paths on another machine.

To exercise the newly built library with three exact toy requests, set `QBRIDGE_LIBRARY` to its full path and run the following from the source root. The hash here pins your own fresh build during loading; it is not an assertion that your binary matches the historical one.

```python
import hashlib
import os
from pathlib import Path
from icde2027.family_lp_v7.adapter import ExactLP

root = Path.cwd()
library = Path(os.environ["QBRIDGE_LIBRARY"]).resolve()
engine = ExactLP(
    library,
    expected_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    receiver_path=root / "icde2027/partition_capacity_v6/verify.py",
    general_receiver_path=root / "icde2027/family_lp_v7/verify_general.py",
    expected_general_receiver_sha256=
        "dc7ac103796a946a4a481a03b5abad4167d9a8fbc559134ceff1de062697cf79",
)
program = {"lower": ["3/4", "0/1"], "upper": ["1/1", "1/4"], "rows": []}
args = (program, [2], ["1/1"], [["1/2", "1/2"]])
for result in (engine.solve_family(*args), engine.solve_general_family(*args)):
    assert result["status"] == "verified_optimal"
    assert result["certificate"]["capacity"] == "1/4"
result = engine.solve_fixed_objective(program, ["1/2", "-1/2"])
assert result["status"] == "verified_optimal"
assert result["certificate"]["optimum"] == "1/4"
print("Three toy optima independently checked: 1/4")
```

The adapter's optional native time limit covers optimization only. Formal experiments use the separately reviewed process supervisor for complete-service timeouts, memory limits and failure records. These toy checks establish no speed or scientific improvement.
