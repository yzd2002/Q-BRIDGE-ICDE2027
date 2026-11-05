# Q-BRIDGE

**Reusable Policy Certificates across Observational and Randomized Logs**

Q-BRIDGE studies repeated policy-threshold queries over a fixed, uncertainty-aware
view built from observational and randomized logs. It separates producing a
candidate answer from checking its arithmetic: the receiver checks a bound or
feasible witness and returns **true**, **false**, or **unknown** for the represented
linear program. These answers do not independently validate the scientific
assumptions used to construct that program.

This repository contains submission materials prepared for ICDE 2027, by
Zida Yang, Weiqing Zeng, Guquan You, Cong Wang, and Jiexin Zheng. It does not imply
acceptance or an artifact-evaluation badge.

## Start here

| Task | Entry point |
|---|---|
| Inspect the implementation | [`icde2027/`](icde2027/) |
| Check one saved certificate without installing scientific packages | The standard-library command below |
| Run the tests and a five-query example | The Python 3.12 quick start below |
| Complete historical results archive | Release in preparation; not yet downloadable |
| Verify the environment-check scope | [`CLEAN_ENVIRONMENT_VERIFICATION.md`](CLEAN_ENVIRONMENT_VERIFICATION.md) |
| Understand code and data reuse terms | [`LICENSE`](LICENSE), [`CODE_LICENSE_SCOPE.json`](CODE_LICENSE_SCOPE.json), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |

## A certificate check in seconds

After cloning this repository, run from its root:

```bash
python3 -B -S icde2027/proof_check.py \
  --view examples/view.json \
  --proof examples/proof.json \
  --query examples/query.json
```

This uses Python's standard library only. It checks the saved teaching example's
view, query, threshold and represented arithmetic. The example is constructed;
it is not a new empirical result.

## Quick start

The tested scientific environment is **Linux, Python 3.12.3, CPU only**.
No GPU, account credential, commercial optimizer or complete legacy installation
is needed. The exact dependency versions are in `requirements-lock.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
./run.sh -m pytest icde2027 -q -p no:cacheprovider
./run.sh tools/small_example.py --out reruns/small_example
```

The corresponding independent environment check passed **576 tests**. It used a
new virtual environment on the same host, with all 17 locked versions installed
from public PyPI. Its scope and the earlier missing-file failure are recorded in
the verification files. This check did not rerun the paper's formal statistical
or timing grids. Use a fresh process and a new output directory for each run.

## Complete reproducibility material

This repository currently provides the source, tests and quick-start examples.
The complete results archive is being prepared for a separate GitHub Release.
It will include all successful, unknown, refused and unsupported-target results,
registered configurations, saved certificates, historical source, and figure
inputs. It is not yet available for download. The public download and verification
instructions will be added here when the complete archive is published.

`SOURCE_MANIFEST.json` binds the files currently provided in this repository.
No acceptance claim or full-archive availability is implied by this source release.

## Reading the results

A **false** answer concerns the minimum over a represented view; it is not an
upper bound on the real-world policy effect. **Unknown** means the supplied
evidence did not resolve the requested threshold. Retaining and checking
certificates can save repeated producer work, while filtering also has a cost;
the artifact retains the comparisons where a proposed engineering target was
not supported. Source assumptions and statistical-interval kernels remain
explicit parts of the trust boundary.

## Licenses

Q-BRIDGE-owned Python and shell source is available under the MIT license within
the scope listed in `CODE_LICENSE_SCOPE.json`. Coat data and data-derived materials
retain their separate **CC BY-NC 4.0** terms and attribution. The unchanged
IEEEtran template retains its own LPPL terms. The code license does not relicense
the datasets, manuscript, figures or third-party material. See the accompanying
notices before redistributing those materials.
