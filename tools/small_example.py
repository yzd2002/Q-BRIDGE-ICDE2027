"""An eight-action constructed smoke test; never a paper measurement."""
import argparse
from pathlib import Path
import json
from icde2027.query_workload import fixture_view, query_stream, measure_method

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
document, elapsed, shape = fixture_view(8, "connected")
queries = query_stream(8, 5, "nearby_nonproportional", [2027110403, 999])
summary, view = measure_method(args.out, "prepared_cache", queries, .001, 16,
                              view=document["view"], common_seconds=elapsed, prefixes=[1, 5])
assert summary["replay"]["proofs_valid"] == 5
print(json.dumps({"scope": "constructed smoke test only", "queries": 5,
                  "proofs_valid": summary["replay"]["proofs_valid"]}))
