# Exact generic family LP prototype

This module implements the SoPlex baseline with exact rational import, export and original-constraint certificates. Development and acceptance here use newly constructed synthetic inputs only. A separate reviewed study harness must authorize real data, bind all sources and enforce the full service timeout and memory limit. This module provides neither a formal study nor evidence of speed or scientific improvement.

`bridge.cpp` / `bridge.h` implement an in-process C ABI. `adapter.py` loads the shared library once with `ctypes`, creates a fresh SoPlex model on each call, and exports exact certificates. It never calls or imports the partition constructor and has no cross-request basis or result cache. `build_wrapper.py` compiles this bridge against the separately built official archive. The independently owned `verify_general.py` supplies acceptance for general family and fixed-policy certificates. Its algorithms are not part of the producer.

## Pinned dependency and build

The dependency is official SoPlex 8.1.0, commit `a00610534e044f35aa298059eb963312336c4c9c`. The parent built the unchanged release with the official Makefile and `USRCPPFLAGS=-fPIC`. Actual dependencies are Boost **1.73.0**, GMP 6.2.1 and MPFR 4.0.2 from the existing account environment, with GCC 11.4.0. No package or system installation is performed. The original non-PIC link failure is preserved under `analysis/family_lp_v7_implementation/build_attempt_01/`.

From the project root, using the parent-built PIC tree:

```bash
taskset -c 10 .venv/bin/python -B -I -S icde2027/family_lp_v7/build_wrapper.py \
  --soplex-source runs/icde2027/analysis/soplex_baseline_dependency_v7/pic_build_02/source/soplex-a00610534e044f35aa298059eb963312336c4c9c \
  --output runs/icde2027/analysis/family_lp_v7_implementation/NEW_BUILD_DIRECTORY
```

The output must be fresh. The build receipt records the archive, source hashes, compiler, actual linked library paths and output hash. The adapter requires an externally supplied raw SHA-256 for the shared library. It loads receiver bytes only from explicit paths with explicit hashes; no package or source-machine fallback is used. The library and receiver paths must remain immutable during the process lifetime. The source repository and its dynamic dependencies remain separate provenance inputs for the study gate.

## Python interface

```python
engine = ExactLP(
    library_path,
    expected_library_sha256=trusted_library_sha,
    receiver_path=trusted_frozen_v6_receiver_path,       # W1 only
    general_receiver_path=trusted_general_receiver_path,
    expected_general_receiver_sha256=trusted_general_sha,
)
result = engine.solve_family(program, sizes, weights, reference)
result = engine.solve_general_family(program, sizes, weights, reference)
result = engine.solve_fixed_objective(program, exact_objective)
result = engine.solve_linear_program(raw_lp)
```

Every method accepts `native_time_limit_ns=None` or a positive unsigned 64-bit integer. The option bounds SoPlex `optimize()` only. It cannot terminate arbitrary Python parsing, export, receiver work, a native crash or an out-of-memory process. An outer process supervisor must implement a complete service budget and memory cap. Initialization is separately recorded as `one_time_setup_ns`.

`program` has exactly `lower`, `upper`, `rows`; each row has exactly `coefficients`, `lower`, `upper`. Family and fixed-objective acceptance use finite boxes in `[0,1]` and finite two-sided moments. Every original row, including zero and redundant rows, retains its original index. Signed, overlapping and cross-context coefficients are supported by the general checker. A mathematically empty feasible set can be a valid request; it cannot receive a successful optimality certificate.

General/fixed trusted input accepts Python `int` excluding `bool`, `Fraction`, and reduced canonical `n/d` strings. The independent receiver requires positive denominators, no leading zeros, no `+`, no `-0`, and `0/1` for zero. It applies its documented 100000-digit per-component and aggregate resource bounds before native import. Weights and each reference block must be nonnegative and sum to exactly one; no normalization occurs. `sizes` must partition the coordinate list. Fixed objectives are arbitrary rational vectors and retain their scale. The original v6 W1 receiver keeps its existing supported-partition domain and parser contract; it is never weakened to admit overlaps.

The raw LP API accepts `objective`, `lower`, `upper`, `rows`, where `None` denotes a free bound side. Its internal exact parser also permits integer strings and noncanonical rational strings, which it canonicalizes before native import. It returns `status="exact_optimal_pending_independent_checker"` and `receiver_receipt=None`. This low-level raw LP is outside the general family/fixed receiver API and its native self-check is explicitly not independent acceptance. Raw dimensions must fit a native signed integer. Explicit finite bounds whose magnitude meets the SoPlex infinity sentinel are rejected; they are never silently made free. Outer resource limits remain the caller's responsibility.

For accepted family/fixed calls, status is `verified_optimal`. The return includes the versioned certificate, the independent receiver receipt, external input digest, native exact solution, phase timings and linked-library identity. The receiver is called with the **caller's original program and objective/probability objects**, plus a digest derived from that request before model construction. A certificate's data cannot replace the external request. Frozen v6 certificates additionally carry canonical input for compatibility, with empty `contexts` and `internal_check` summaries: no cap, root or price claims are fabricated.

General family uses `family-capacity-input/1` and `family-capacity-certificate/1`; fixed objective uses `fixed-policy-optimum-input/1` and `fixed-policy-optimum-certificate/1`. Certificate scalars are exact reduced `n/d` strings. The general/fixed minimal schemas contain no embedded request. See the independently owned `analysis/family_lp_v7_receiver_review/API.md` for the exact checker contract.

## Mathematical model and checks

The family model minimizes `sum_x w_x (t_x - sum_a p_xa m_xa)` on all original boxes and moment rows, with one row `m_xa - t_x <= 0` per action and **free** `t_x`. No original block is omitted when its weight is zero. SoPlex is configured with rational read/solve/check modes, automatic synchronization, zero feasibility/optimality tolerances, and minimization. Other solver choices retain the official defaults; these synthetic tests do not select parameters by observed study outcomes.

Before optimizing, native getters read back every rational coefficient, objective, bound and row dimension from the loaded SoPlex model and compare them exactly with the request. The resulting canonical wire crosses the C ABI and Python compares it byte for byte to its external model before retaining its SHA. A rounded floating-point model cannot pass this comparison.

A native optimum is exported only if SoPlex returns `OPTIMAL`, exact primal/dual exports succeed, every original constraint holds, reduced costs equal `c - A^T pi`, and the signed-bound dual lower bound equals both `c*m` and SoPlex's rational objective. This is a producer self-check. In particular, a floating-point status alone is insufficient.

For moment row dual `pi`, the original lower and upper multipliers are `eta=max(pi,0)` and `lambda=max(-pi,0)`. The synthetic lower/upper-bound examples explicitly test both signs. For epigraph rows, `gamma=-pi_epi >= 0`. Free `t` requires exactly `sum_a gamma_xa=w_x`. Positive-weight policy is `gamma/w`; a zero-weight block must have all gamma zero and returns its original reference policy. The adapter checks original weak duality closure `LB=c(q)*m=F(m)=R` before the independent receiver repeats its own original-program calculations.

For W2, `solve_fixed_objective` proves `min c*m` on the original program for the exact supplied objective. It does not claim that an old stored LP failed to attain its own optimum. Any later transfer-slack decomposition must separately bind the normalized ideal objective and original stored dual semantics in the study protocol.

## C ABI and timing

`flp_solve_v1(request, request_size, solver_limit_ns, result)` accepts ASCII whitespace tokens:

```text
FAMILY_LP_EXACT/1
number_of_columns number_of_rows
objective lower upper                 # once per column
row_lower row_upper nnz i coefficient ...  # once per original row
```

The comments above are explanatory and are not wire syntax. Rational tokens must be reduced canonical `n/d`; `*` marks a free side. Sparse indices are zero-based, increasing, unique and nonzero-valued. All zero rows still appear with `nnz=0`. Invalid syntax, trailing tokens, reversed bounds and noncanonical fractions are refused. Boost 1.73's `mpq_set_str` does not reduce automatically; the bridge explicitly invokes `mpq_canonicalize` before checking canonical bytes. The regression and the original failed test receipt are retained.

The caller supplies a valid zero-initialized `flp_result_v1`, reads the owned length-delimited JSON payload, then calls `flp_release_v1` exactly once. Outcome codes are `0` exact optimum, `1` invalid request, `2` solver refusal and `3` caught native exception. Non-optimal status, infeasibility, unboundedness or native time limit are refusals to issue an optimality certificate, not independently proved infeasibility claims. Invalid C pointers and process termination are outside the C ABI recovery contract.

Timing partitions the structured Python call into model, solve, export, receiver and residual overhead nanoseconds. `total_call_ns` begins with already-decoded Python arguments and ends just before the result returns. JSON transport before or after this boundary, worker startup/shutdown, external timeouts and crashes must be measured by the outer harness. Native model and exact verification/export costs are included, along with failed paths. A caught failure raises `SolveError` with a JSON-safe `diagnostics` object and no accepted certificate. `native_total_ns` is also retained separately, not added a second time.

## Synthetic verification

Run only this author's test file; the independent receiver tests are separately owned:

```bash
taskset -c 10 .venv/bin/python -B -I -S icde2027/family_lp_v7/tests/test_adapter.py
```

Tests bind the named shared-library SHA and both receiver versions, exercise non-dyadic and nearly binary64-indistinguishable inputs, signed/overlapping/cross-context constraints, equalities and zero rows, zero-width boxes, zero probabilities, ties, exact objective scaling, invalid input, infeasible/unbounded refusals, request binding, model roundtrip, failure timing and 30 generated fixed-point family cases. These known-point examples are synthetic checks, not an independent LP oracle or a performance sample. The final author receipt and source hashes live under `analysis/family_lp_v7_implementation/`. Consequential code still requires an independent review before formal execution.
