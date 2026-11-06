# Partition capacity: exact construction, checker and fixed-data study

This standard-library `Fraction` implementation constructs a policy, feasible
mean vector and original-row duals with exactly matching family-capacity bounds.
`verify.py` independently checks those objects against a caller-supplied request;
`study.py` adapts the fixed study inputs and records all outcomes. The complete
study and independent result audit are available in `run_02` and `audit_run_02`
below. Mathematical validity, implementation correctness, novelty and empirical
performance remain separate questions. This module makes no new minimax theorem
or speed claim. It is distinct from the existing binary64 certificate wire format.

## Interface and input domain

```python
from icde2027.partition_capacity_v6 import solve
from icde2027.partition_capacity_v6.verify import input_digest, verify_certificate

request = dict(
    program={
        "lower": [0, 0],
        "upper": ["1/3", 1],
        "rows": [{"coefficients": [1, 1], "lower": 1, "upper": 1}],
    },
    context_sizes=[2],
    exact_weights=[1],
    exact_reference=[["1/2", "1/2"]],
)
expected_digest = input_digest(**request)
certificate = solve(**request)
assert certificate["capacity"] == "1/6"
receipt = verify_certificate(
    **request, certificate=certificate, expected_input_sha256=expected_digest
)
```

`solve(program, context_sizes, exact_weights, exact_reference)` is a pure function
returning a JSON-serializable dictionary. It performs no file, network, process,
or GPU operations, imports no project solver, and does not modify its arguments.
There are no added dependencies. Coordinate indices are zero-based and contexts
occupy contiguous slices in `context_sizes` order. All original rows, including
validated zero rows, retain their exact coefficients, bounds, order, and indices.

The `program` mapping requires exactly `lower`, `upper`, and `rows`. Every row
requires exactly `coefficients`, `lower`, and `upper`; additional metadata is
rejected rather than omitted from the input binding. Vectors are lists or tuples.
Scalar entries accept Python `int` (not `bool`), `fractions.Fraction`, or strings
matching `[+-]?[0-9]+(?:/[1-9][0-9]*)?`. Denominators must be positive; whitespace,
decimal/exponent strings, floats, Decimal, bool, and implicit conversions are
rejected. Equivalent fractions reduce exactly; no probability is normalized.
Every output rational is the reduced string `numerator/positive_denominator`,
including `0/1` and `1/1`; no numerical output passes through float.

Boxes must satisfy `0 <= lower <= upper <= 1`. Context sizes are nonempty positive
integers, summing to the number of coordinates. Weights and each reference row
must be nonnegative and sum to exactly one. Zero weights, zero probabilities,
fixed boxes, equal lower/upper moment bounds, unused coordinates, tied prices,
and non-dyadic rational coefficients are supported. Each nonzero moment row must
have nonnegative coefficients, lie in one context, and have positive support
disjoint from every other row. An empty row must have `lower <= 0 <= upper`.

Errors are explicit: `InvalidInputError` covers malformed requests and invalid
probabilities; `InfeasibleProgramError` is its subclass for an exactly detected
empty supported program; `UnsupportedStructureError` covers signed, overlapping,
or mixed-context rows. For a request containing both malformed and unsupported
parts, no precedence guarantee is given. `CertificateConstructionError` denotes
an internal exact invariant failure and must never be treated as a capacity.

## Construction and certificate

For each context, the algorithm finds the first feasible maximum coordinate cap
by scanning rational upper-box breakpoints. It maximizes the reference mean at
that cap with exact fractional knapsack, ordering equal ratios by original
coordinate index. If the row budget is already exhausted, its price is the
maximum ratio, including fixed coordinates; if it ends exactly after a fill,
the price is the last strictly positive fill's ratio. If the row upper bound is
at least its capped-box mass, its price is zero. Unused coordinates take their
capped upper bounds.

The initial policy uses positive residuals on coordinates with `upper >= cap`.
The remaining probability goes either to the first coordinate whose lower box
bound equals the cap (case A), or to the first row attaining the maximal root,
distributed proportionally to that row's coefficients on `upper >= cap` (case B).
Original moment multipliers and the objective are multiplied by the same exact
context weight, including zero weights.

The versioned certificate includes:

- `canonical_input`, the complete rational input, and `input_sha256`;
- `point` and per-context `policy`;
- original-row lower and upper duals, including zero entries for zero rows;
- original row values, weighted objective coefficients, and box reduced costs;
- per-context cap, first-root witnesses, prices, deficit completion witness,
  capped-box dual value, and weighted/unweighted capacity;
- the identical exact capacity, common-point upper bound, attained policy
  objective, and original-program weak-duality lower bound.

The input digest is SHA-256 of ASCII JSON with keys sorted, separators `,` and
`:`, no whitespace, and `ensure_ascii=True, allow_nan=False`. The input schema
version is `partition-capacity-input/1`; the certificate schema version is
`partition-capacity-v6/1`. Arrays preserve original ordering. Rational inputs are
canonicalized by exact reduction, so equivalent literal fractions bind to the
same mathematical request. Adding or removing a zero row changes the input
identity; reordering distinct rows also changes it. Identical rows are not
distinguished by an external label in this interface.

**Trust boundary:** embedded input is a transport convenience, not a trusted
source of statistical constraints. A receiver must bind the certificate to the
caller's separately trusted program, context sizes, weights, reference, and
expected digest. Verifying only the certificate's self-contained input and hash
does not establish that it answers the requested question, or that its program
has valid statistical provenance. The producer's internal checks compare against
the original caller request and recompute all original feasibility constraints,
policy probability sums, root/price witnesses, context scaling, and the exact
equality

```
a^T eta - b^T lambda
  + sum_i min(z_i lower_i, z_i upper_i)
  = sum_x w_x (policy_x - reference_x)^T point_x
  = sum_x w_x (max(point_x) - reference_x^T point_x),
z = objective + T^T (lambda - eta).
```

The producer self-check and the independently authored `verify.py` are separate.
The latter recomputes original feasibility, policy legality and lower/attained/
upper equality. It checks `contexts.*` and `internal_check` only for JSON shape;
successful receipts explicitly list them as mathematically unchecked. Neither
checker establishes the source assumptions, statistical quantile accuracy, data
authenticity or freshness. No same-capability solver speed comparison is supplied
by the capacity study.

## Completed study and reproduction

The fixed study covers 3,168 distinct programs and 5,760 dependent configuration
occurrences from 48 synthetic worlds. It retains both radii, both randomized
budgets, all three source methods, finite/population views, and adverse source
comparisons. These are retrospective analyses, not new statistical confirmation.

- Complete production output:
  `runs/icde2027/analysis/partition_capacity_study_v6/run_02/`.
- Complete independent result audit:
  `runs/icde2027/analysis/partition_capacity_study_result_audit_v6/audit_run_02/`.
  Its `report.json` SHA256 is
  `9b9223ca70e92768ba9844358e3827a9685dbe4634ad04f4ef49abdbb7b7a417`.
- Portable tools and their explicit input closure:
  `runs/icde2027/analysis/partition_capacity_public_replay_v6/`.
  See its `README.md` for the manifest/admission schema, and
  `production_01/REPORT.md` for the completed actual replay. That replay used
  another existing Python and a new directory on the same host; its four core
  result files matched the original independent audit byte for byte. It reuses
  the same mathematical audit implementation and is not a third independent
  implementation or a cross-host test.
- Full figure inputs and all source-pair counts:
  `runs/icde2027/analysis/partition_capacity_assets_v6/generated_01/`.

The first producer attempt failed during large-integer summary serialization;
the first result-audit attempt rejected an omitted historical progress-file
inventory entry before new numerical checking. Both failed records and the
scoped corrections remain preserved. Only the complete second attempts support
the reported study results. Do not restart or overwrite them to inspect results;
use the portable replay route with a fresh output directory.

For archived rational records, Python's default integer-string limit can be too
small. The study and replay entrypoints set `sys.set_int_max_str_digits(100000)`
inside their process. A custom reader must do likewise before parsing these
records; do not disable the limit or modify system Python. Exact summary output
uses `Decimal(int)` formatting to preserve arbitrarily large result integers
without binary floating conversion. Historical interpreter/stdlib identities
are provenance metadata; portable replay records its actual runtime separately.

These local paths describe the completed author workspace. A public release
must explicitly include the corresponding inputs, tools and receipts; the
existence of the earlier v5.2 GitHub release alone does not publish this study.

## Validation scope

Tests use hand-checkable rational examples, deterministic synthetic programs,
input rejection cases, and certificate-invariant checks. They do not import the
mathematics review's epigraph enumerator or claim a second independent oracle.
The original proposal is
`runs/icde2027/analysis/revision6_review/PARTITION_CAPACITY_PROPOSAL.md`, SHA-256
`35f09a1e20d641a6213d663ac9d95dc77e5184e94f1459ada04052eeec20056b`.
Its scoped mathematical review is stored separately under
`runs/icde2027/analysis/partition_capacity_math_review_v6/`.

From the project root, run the isolated tests using its existing environment:

```bash
taskset -c 10 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s icde2027/partition_capacity_v6/tests -v
```

CPU affinity should be adjusted only if that CPU is occupied. Test logs and
source hashes belong in a fresh attempt directory below
`runs/icde2027/analysis/partition_capacity_implementation_v6/`; failed attempts
must remain visible. No real study outcomes or performance files are required.
