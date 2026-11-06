# Check the concrete Coat policy case

This small example verifies **seven saved optimum certificates**: two solver
outputs for the positive 16-row jacket-type view, four zero-capacity controls,
and the positive 31-row parent view. It checks the exact policy and the two terms
reported in the manuscript's Coat example. It requires Python 3.11+ and only the
standard library; it does not call an optimizer, download data or use a GPU.

From the repository root:

```bash
python3 -B -I -S - <<'PY'
import csv, hashlib, importlib.util, json
from fractions import Fraction as F
from pathlib import Path

root = Path.cwd()
manifest_path = root / "examples/coat_policy_case/MANIFEST.json"
raw = manifest_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "04fd51b79ead079f05cfeb4c9a54d3d59bd141d5427b4819c664e3647c1e56f1"
manifest = json.loads(raw)
for name, expected in manifest["files"].items():
    data = (root / name).read_bytes()
    assert len(data) == expected["bytes"]
    assert hashlib.sha256(data).hexdigest() == expected["sha256"], name

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, root / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

part = load("case_partition", "icde2027/partition_capacity_v6/verify.py")
general = load("case_general", "icde2027/family_lp_v7/verify_general.py")
saved = {}
for case in manifest["cases"]:
    request = json.loads((root / case["input"]).read_text())["request"]
    call = json.loads((root / case["call"]).read_text())
    certificate = call["result"]["response"]["certificate"]
    args = [request[k] for k in
            ("program", "context_sizes", "exact_weights", "exact_reference")]
    verify = (general.verify_family_certificate if case["method"] == "G_general"
              else part.verify_certificate)
    checked = verify(*args, certificate,
                     expected_input_sha256=case["expected_input_sha256"])
    assert checked["verified"] and checked["capacity"] == case["expected_capacity"]
    saved[case["label"]] = (request, certificate)
    print(case["label"], "capacity:", checked["capacity"])

request, certificate = saved["Fusion_Lambda1p5_jacket_S"]
policy = list(map(F, certificate["policy"][0]))
assert policy == list(map(F, saved["Fusion_Lambda1p5_jacket_G"][1]["policy"][0]))
assert policy[0] == F(29, 300)
assert policy.count(F(0)) == 28 and policy.count(F(1, 300)) == 271
features = root / "amber/data/processed/coat/coat/user_item_features"
names = (features / "item_features_map.txt").read_text().splitlines()
items = [list(map(int, line.split())) for line in
         (features / "item_features.ascii").read_text().splitlines()]
vests = {i for i, row in enumerate(items) if row[names.index("jackettype:vests")]}
assert vests == {i for i, value in enumerate(policy) if value == 0}
assert [names[j] for j, value in enumerate(items[0]) if value] == [
    "gender:men", "jackettype:bomber", "color:other", "onfrontpage:yes"]
program = request["program"]
box_terms = [min(F(r)*F(lo), F(r)*F(hi)) for r, lo, hi in zip(
    certificate["box_reduced_costs"], program["lower"], program["upper"])]
moment_terms = [F(eta)*F(row["lower"])-F(lam)*F(row["upper"])
    for eta, lam, row in zip(certificate["original_row_lower_dual"],
                            certificate["original_row_upper_dual"], program["rows"])]
assert sum(value != 0 for value in box_terms) == 1
assert sum(value != 0 for value in moment_terms) == 1
capacity = F(certificate["capacity"])
assert sum(box_terms) + sum(moment_terms) == capacity
with (root / "assets/coat_policy_case/compact_guarantee.csv").open() as handle:
    rows = list(csv.DictReader(handle))
assert list(map(F, [r["exact"] for r in rows])) == [sum(box_terms), sum(moment_terms), capacity]
print("Seven saved optimum certificates verified; policy and two-term table agree.")
print("Guaranteed high-rating improvement (percentage points):", float(100*capacity))
PY
```

Expected output ends with `Seven saved optimum certificates verified; policy
and two-term table agree.` The displayed gain is about **0.246355 percentage
points**. In the certified policy, item 0 receives `29/300`, all 28 vests receive
zero, and 271 other actions retain `1/300`. The metadata describes item 0 as a
men's bomber jacket; no product name or brand is inferred.

The independent mathematical receivers consume a separately supplied request
and its expected digest. They verify original-constraint feasibility, a legal
policy, and equality of lower, attaining-policy and family-wide upper bounds.
They do not validate the statistical intervals, source assumptions, or the
partition constructor's auxiliary `contexts` and `internal_check` fields. The
manifest hash in the command binds this selected example; hashes are not
signatures, so obtain the repository version through a trusted channel.

## Evidence and scope

- `inputs/` holds six **byte-for-byte original JSONL lines**, saved as standalone
  JSON files. `MANIFEST.json` records each original source hash and line number.
- The seven files under `runs/icde2027/analysis/verified_family_lp_v7/run_02/calls/`
  are unchanged original call records. The two positive certificates are also
  exactly equal, after canonical JSON encoding, to the later Coat workflow's
  saved certificates used in `assets/coat_policy_case/ANALYSIS.json`.
- `assets/coat_policy_case/` contains the exact analysis and readable tables.
  Its `SOURCE_MANIFEST.json` lists the broader original analysis inputs, including
  records absent from this small example. Those original absolute paths are
  provenance, not fallback dependencies for the command above.
- This example checks one retrospectively selected case and its controls. It
  does not reproduce all study results, recompute confidence intervals, rerun
  serving times, or establish a joint guarantee across sensitivity declarations.
  Its high-rating guarantee is conditional on the fixed assumptions and is not
  observed purchase lift or evidence that every vest is worse.

## Data attribution

The unchanged item features and data-derived records come from the public
[Coat benchmark](https://www.cs.cornell.edu/~schnabts/mnar/), introduced by
Schnabel et al., [Recommendations as Treatments: Debiasing Learning and
Evaluation](https://proceedings.mlr.press/v48/schnabel16.html), ICML 2016.
Retain the benchmark's **CC BY-NC 4.0** attribution and conditions. The root MIT
license applies only to the exact first-party code listed in the corresponding
code-scope files; it does not relicense these data, tables or manuscript.
