# Dated environment and public-dependency check

On 2026-11-05, an independent check of the corrected public dependency subset
passed all **576 shipped scientific tests**, with **0 failures, 0 errors and
0 skips**. The test process ran from 19:34:42 to 19:34:54 UTC. This is a dated
record of the identified inputs, not a claim that every paper experiment was
rerun. The adjacent `CLEAN_ENVIRONMENT_VERIFICATION.json` is the independent
verifier's unchanged public summary.

| Check | Actual outcome |
|---|---|
| Initial fresh virtual environment | All 17 exact lock versions installed as public PyPI wheels; imports and `pip check` passed |
| Initial public test subset | 573 passed, 3 failed because the configured design-document path was absent |
| Corrected public test subset | 576 passed; all three previously failing cases passed |
| Input integrity | 200 copied inputs unchanged; 197 original public sources and 231 earlier sealed records rechecked; no private `04_Internal_Review` subtree |
| Earlier proof and constructed example | Passed; the example contains five constructed queries and five valid proofs |
| Earlier paper build | 23 source dependencies; 13 pages; byte-identical to the selected PDF |

The installation and follow-up used the same Linux host and Python 3.12.3 base
interpreter. The virtual environment had no system or user site packages; the
checks used CPU only and one computational thread. The follow-up reused the
newly installed environment and did not download or change dependency versions.
The install lock is `requirements-lock.txt`, SHA-256
`af1e35c28bb4a690b9a0b81be42d7c886d3dac29f06d1324e8f48da6b2f78458`.

The original design bytes were already in
`runs/icde2027/checking_schedule_v4/design.md`. The packaging repair provides
those exact bytes at the path required by the current configuration,
`runs/icde2027/CHECKING_SCHEDULE_V4_DESIGN.md`. It also explicitly binds the
historical environment record at `.deployment/requirements-installed.txt`.
That historical record is not a new installation lock.
`CURRENT_SCHEDULE_DEPENDENCIES.json` gives the sources, destinations, sizes and
hashes. No scientific implementation, test expectation, registration, approval,
source snapshot or result was changed to obtain the passing tests.

The proof, five-query example and paper build were not repeated in the follow-up:
their 193 original inputs were byte-identical, and the prior result and manifest
hashes are retained in the adjacent JSON. The paper build used existing TinyTeX,
not a fresh TeX installation, and produced PDF SHA-256
`33fe4c7b5fdd6ecae4b944ae69e4bad8f5ffa78e6cdd6698b0e18ba08c0ba834`.

All failed attempts remain recorded. The initial 573/3 result is unchanged.
The follow-up verifier initially used `fonttools` instead of the import name
`fontTools`; that setup failure occurred before scientific tests started. Its
checker was corrected in a separate attempt, without changing the tested
inputs, implementation, tests or environment.

The public JSON binds the original result, complete test log, JUnit record and
input inventories by SHA-256. Its own SHA-256 is
`a114f88eb113bc8308aa662a3a1afee55da8d8d1ba84689170aed69912236f0a`.
In the complete local handoff, the raw follow-up records are retained under
`04_Internal_Review/reviews/package_v5_public_closure_repair_evidence/independent_followup/`;
the initial records are under
`04_Internal_Review/verification/clean_environment_v5_original/`.
Those records are provenance, not runtime dependencies or publication attachments.

This check did not rebuild a new machine or operating system, rerun formal
statistical or timing grids, reproduce every figure, or create a new complete
submission archive. It establishes the stated installation and test outcomes
for the identified public subset; it does not establish scientific assumptions,
generalization, performance gains or paper acceptance.
