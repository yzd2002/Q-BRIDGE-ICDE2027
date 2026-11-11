# Threshold readout from saved exact capacities

The final manuscript reports a post hoc sensitivity readout of the same primary 107 previously unresolved views, at thresholds 0.5 and 1.0 percentage points. The counts are 71 and 42. No view, outcome, policy solver or timing experiment was rerun. This is a descriptive readout of synthetic guarantees, not measured deployment uplift.

`capacity_records.json` contains the exact rational capacities and original-candidate optima for all 240 primary finite projected-Fusion views (100d randomized events, conservative radius Lambda=2), with original occurrence IDs and the fixed 107-view subset flag. Results for thresholds 0.1, 0.5, 1.0 and 2.0 pp are included. The original candidate best value (including the reference) is zero in all 107 views. Therefore all four larger-threshold counts for those candidates are zero.

Run `python verify.py` to recompute counts using exact rational comparisons. To additionally verify every copied record against the full local archive, run:

```bash
python verify.py --source ../assets/family_v7/auditor_outputs/decomposition_occurrences.jsonl
```

The supplied source must match the recorded SHA256. The compact public records verify this readout; they do not replace the full experiment archive or reproduce timing measurements.
