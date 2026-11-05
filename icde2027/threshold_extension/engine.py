"""Five threshold plans sharing one persistent native HiGHS implementation.

Only exact lower/upper evidence authorizes a three-valued threshold decision.
The native objective cutoff is a work proposal, never an acceptance condition.
The original query executor and its registered measurements are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import sys
import time

import highspy
import numpy as np

from .proofs import ThresholdChecker, _floats, bundle_proof


PLANS = ("full_hot", "threshold_hot", "primal_pool", "dual_pool", "both_pools")


@dataclass(frozen=True)
class DualCandidate:
    lower: tuple
    upper: tuple
    constant: float
    residual: np.ndarray
    prepared: object


class ThresholdEngine:
    """Stateful threshold executor; configured strategy 0/1/4 is explicit.

Strategy 0 lets HiGHS choose, 1 requests serial dual simplex, and 4 requests
primal simplex. objective_bound is supplied to threshold plans, but its native
early-stop behavior only applies when HiGHS actually uses dual simplex.

Every threshold plan retains the most recent accepted native primal and dual.
A history-enabled side retains at most bank_size unique native candidates,
including that last candidate; bank_size=0 disables history, not common last
reuse. Native duplicate candidates can change the last pointer, but neither
duplicates nor cache hits reorder FIFO history. The fixed initial witness and
zero dual are separate from these capacities.
    """
    def __init__(self, view, plan="threshold_hot", bank_size=16, *, simplex_strategy=1):
        begin = time.perf_counter()
        if plan not in PLANS or type(bank_size) is not int or not 0 <= bank_size <= 16:
            raise ValueError("known plan and bank_size between 0 and 16 required")
        if type(simplex_strategy) is not int or simplex_strategy not in (0, 1, 4):
            raise ValueError("simplex_strategy must be choose=0, serial dual=1, or primal=4")
        self._view = json.loads(json.dumps(view, allow_nan=False))
        # Candidate banks own all retained arithmetic states. In particular a
        # no-history baseline must not hide another 15 entries in this cache.
        # External receivers independently use their common FIFO cache policy.
        self.checker = ThresholdChecker(self._view, cache_size=0)
        self.digest = self.checker.digest
        self.d, self.j = self.checker.d, self.checker.j
        self.plan, self.bank_size = plan, bank_size
        self.simplex_strategy = simplex_strategy
        self._use_primal_pool = plan in ("primal_pool", "both_pools")
        self._use_dual_pool = plan in ("dual_pool", "both_pools")
        self.primal_retention_capacity = (max(1, bank_size) if self._use_primal_pool else
                                          0 if plan == "full_hot" else 1)
        self.dual_retention_capacity = (max(1, bank_size) if self._use_dual_pool else
                                        0 if plan == "full_hot" else 1)
        self._initial_raw = tuple(self._view["feasible_point"])
        self.initial_primal = self.checker.primal_state(self._initial_raw)
        self.last_primal = None
        self.last_dual = None
        self.last_primal_generation = 0
        self.last_dual_generation = 0
        self.primal_bank = []
        self.dual_bank = []
        self.maximum_primal_bank_entries = 0
        self.maximum_dual_bank_entries = 0
        self._zeros = (0.,) * self.j
        self._zero_dual = self.checker.dual_state(self._zeros, self._zeros)
        self._lower = np.array(self._view["lower"], dtype=float)
        self._upper = np.array(self._view["upper"], dtype=float)
        self.indices = np.arange(self.d, dtype=np.int32)
        self.h = highspy.Highs()
        self.options = {"output_flag": False, "threads": 1, "solver": "simplex",
                        "simplex_strategy": simplex_strategy,
                        "primal_feasibility_tolerance": 1e-10,
                        "dual_feasibility_tolerance": 1e-10}
        for key, value in self.options.items():
            if self.h.setOptionValue(key, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f"HiGHS rejected option {key}")
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.d, self.j
        lp.col_cost_ = np.zeros(self.d)
        lp.col_lower_, lp.col_upper_ = self._lower, self._upper
        lp.row_lower_ = np.array(self._view["rhs_lower"], dtype=float)
        lp.row_upper_ = np.array(self._view["rhs_upper"], dtype=float)
        source = self._view["matrix"]
        matrix = highspy.HighsSparseMatrix()
        matrix.num_col_, matrix.num_row_ = self.d, self.j
        matrix.format_ = highspy.MatrixFormat.kRowwise
        matrix.start_ = np.asarray(source["indptr"], dtype=np.int32)
        matrix.index_ = np.asarray(source["indices"], dtype=np.int32)
        matrix.value_ = np.asarray(source["data"], dtype=float)
        lp.a_matrix_ = matrix
        if self.h.passModel(lp) != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS rejected trusted confidence view")
        self.native_version = self.h.version()
        self.initialization_seconds = time.perf_counter() - begin

    def _new_metrics(self):
        return {"screen_seconds": 0., "solve_seconds": 0., "verify_seconds": 0.,
                "preparation_seconds": 0., "primal_preparation_seconds": 0.,
                "dual_candidates_screened": 0, "dual_candidates_exact_checked": 0,
                "primal_candidates_screened": 0, "primal_candidates_exact_checked": 0,
                "rejected_primal_proposals": 0, "invalid_bank_entries": 0,
                "native_primal_candidates": 0, "native_dual_candidates": 0,
                "native_primal_reuses": 0, "native_dual_reuses": 0,
                "last_primal_updates": 0, "last_dual_updates": 0,
                "rejected_dual_proposals": 0,
                "native_calls": [], "cutoff_retries": 0}

    def _lower_proof(self, c, delta, lo, hi, metrics, scale=None, prepared=None):
        begin = time.perf_counter()
        invalid_before = self.checker.stats["invalid_cache_entries"]
        try:
            return self.checker.lower_proof(c, lo, hi, threshold=delta, scale=scale,
                                            prepared=prepared)
        finally:
            metrics["verify_seconds"] += time.perf_counter() - begin
            metrics["dual_candidates_exact_checked"] += 1
            metrics["invalid_bank_entries"] += self.checker.stats["invalid_cache_entries"] - invalid_before

    def _upper_proof(self, c, state, metrics):
        begin = time.perf_counter()
        try:
            return self.checker.upper_proof(c, state)
        finally:
            metrics["verify_seconds"] += time.perf_counter() - begin
            metrics["primal_candidates_exact_checked"] += 1

    def _ensure_initial(self, metrics):
        if not self.checker.owns_primal(self.initial_primal, self._initial_raw):
            metrics["invalid_bank_entries"] += 1
            begin = time.perf_counter()
            self.initial_primal = self.checker.primal_state(self._initial_raw)
            metrics["verify_seconds"] += time.perf_counter() - begin
        return self.initial_primal

    def _checked_point(self, raw, metrics):
        begin = time.perf_counter()
        metrics["native_primal_candidates"] += 1
        try:
            raw = _floats(raw)
            # The native proposal may repeat a retained exact witness. Reuse
            # only explicit, capacity-bounded states; there is no hidden cache.
            states = [self.initial_primal, self.last_primal]
            if self._use_primal_pool and self.bank_size > 1:
                states.extend(p for p in self.primal_bank if p is not self.last_primal)
            for state in states:
                if state is not None and getattr(state, "raw", None) == raw:
                    if self.checker.owns_primal(state, raw):
                        metrics["native_primal_reuses"] += 1
                        return state
                    metrics["invalid_bank_entries"] += 1
            return self.checker.primal_state(raw)
        except (TypeError, ValueError, ArithmeticError):
            metrics["rejected_primal_proposals"] += 1
            return None
        finally:
            elapsed = time.perf_counter() - begin
            metrics["verify_seconds"] += elapsed
            metrics["primal_preparation_seconds"] += elapsed

    def _save_point(self, state, metrics):
        """Accept a native-solve candidate; this is never called on a hit."""
        begin = time.perf_counter()
        try:
            if not self.primal_retention_capacity or not self.checker.owns_primal(state):
                raise ValueError("native point must be locally checked before retention")
            for i, old in enumerate(self.primal_bank):
                if getattr(old, "raw", None) == state.raw:
                    # Replace in place, repairing a stale/poisoned entry while
                    # preserving the insertion order of this unique point.
                    self.primal_bank[i] = state
                    break
            else:
                self.primal_bank.append(state)
            if len(self.primal_bank) > self.primal_retention_capacity:
                self.primal_bank.pop(0)
            self.last_primal = state
            self.last_primal_generation += 1
            metrics["last_primal_updates"] += 1
            self.maximum_primal_bank_entries = max(self.maximum_primal_bank_entries, len(self.primal_bank))
        finally:
            metrics["preparation_seconds"] += time.perf_counter() - begin

    def _save_dual(self, lo, hi, norm, metrics):
        if not self.dual_retention_capacity:
            return
        begin = time.perf_counter()
        metrics["native_dual_candidates"] += 1
        try:
            if norm <= 0:
                raise ValueError("zero-norm objective has no normalized dual candidate")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                lower = tuple((np.asarray(lo, dtype=float) / norm).tolist())
                upper = tuple((np.asarray(hi, dtype=float) / norm).tolist())
            position, state = None, None
            for i, old in enumerate(self.dual_bank):
                if getattr(old, "lower", None) == lower and getattr(old, "upper", None) == upper:
                    position = i
                    if self.checker.owns_dual(old.prepared, lower, upper):
                        state = old.prepared
                        metrics["native_dual_reuses"] += 1
                    break
            if state is None:
                state = self.checker.dual_state(lower, upper)
            residual = np.array([float(x) for x in state.offset], dtype=float)
            constant = float(state.constant)
            if not np.all(np.isfinite(residual)) or not math.isfinite(constant):
                raise ValueError("normalized dual screening representation is not finite")
            residual.setflags(write=False)
            candidate = DualCandidate(lower, upper, constant, residual, state)
            if position is None:
                self.dual_bank.append(candidate)
            else:
                self.dual_bank[position] = candidate
            if len(self.dual_bank) > self.dual_retention_capacity:
                self.dual_bank.pop(0)
            self.last_dual = candidate
            self.last_dual_generation += 1
            metrics["last_dual_updates"] += 1
            self.maximum_dual_bank_entries = max(self.maximum_dual_bank_entries, len(self.dual_bank))
        except (AttributeError, TypeError, ValueError, ArithmeticError):
            metrics["rejected_dual_proposals"] += 1
        finally:
            metrics["preparation_seconds"] += time.perf_counter() - begin

    def _native_run(self, c, bound, metrics):
        begin = time.perf_counter()
        before = bool(self.h.getBasis().valid)
        if self.h.setOptionValue("objective_bound", math.inf if bound is None else bound) != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS rejected objective_bound")
        if self.h.changeColsCost(self.d, self.indices, np.asarray(c)) != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS rejected query objective")
        at_run = bool(self.h.getBasis().valid)
        run_status = self.h.run()
        info, solution = self.h.getInfo(), self.h.getSolution()
        model_status = self.h.getModelStatus()
        elapsed = time.perf_counter() - begin
        metrics["solve_seconds"] += elapsed
        record = {"run_status": str(run_status), "model_status": self.h.modelStatusToString(model_status),
                  "model_status_code": int(model_status), "objective_bound": bound,
                  "simplex_iterations": int(info.simplex_iteration_count), "seconds": elapsed,
                  "basis_valid_before": before, "basis_valid_at_run": at_run,
                  "basis_valid_after": bool(self.h.getBasis().valid),
                  "dual_valid": bool(solution.dual_valid), "primal_values_valid": bool(solution.value_valid),
                  "configured_simplex_strategy": self.simplex_strategy}
        metrics["native_calls"].append(record)
        if run_status == highspy.HighsStatus.kError:
            raise RuntimeError("HiGHS execution error: " + record["model_status"])
        dual = np.asarray(solution.row_dual, dtype=float)
        lo, hi = None, None
        if solution.dual_valid and dual.shape == (self.j,) and np.all(np.isfinite(dual)):
            lo, hi = np.maximum(dual, 0.).tolist(), np.maximum(-dual, 0.).tolist()
        primal = list(solution.col_value) if solution.value_valid else None
        return {"lower": lo, "upper": hi, "primal": primal, "status": model_status}

    def _screen_duals(self, c, norm, delta, bank, metrics):
        """Rank only within the supplied tier; screening never authorizes."""
        begin = time.perf_counter()
        candidates = []
        try:
            for i, candidate in enumerate(bank):
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        residual = np.asarray(c) + norm * candidate.residual
                        value = norm * candidate.constant + float(
                            np.minimum(residual * self._lower, residual * self._upper).sum())
                    metrics["dual_candidates_screened"] += 1
                    if math.isfinite(value) and value > delta:
                        candidates.append((value, i, candidate))
                except (AttributeError, TypeError, ValueError, ArithmeticError):
                    metrics["invalid_bank_entries"] += 1
            candidates.sort(key=lambda row: (-row[0], row[1]))
            return [row[2] for row in candidates]
        finally:
            metrics["screen_seconds"] += time.perf_counter() - begin

    def _retention_metrics(self):
        """Actual candidate state, excluding the fixed initial/zero witnesses.

        Generations count accepted native proposals, including duplicates. A
        response with no native call cannot change either generation. History
        is the retained FIFO list minus its current last candidate, which may
        occupy any FIFO position after a repeated native proposal.
        """
        primal_last, dual_last = self.last_primal is not None, self.last_dual is not None
        return {"last_primal_present": primal_last, "last_dual_present": dual_last,
                "last_primal_generation": self.last_primal_generation,
                "last_dual_generation": self.last_dual_generation,
                "primal_history_entries": len(self.primal_bank) - int(primal_last),
                "dual_history_entries": len(self.dual_bank) - int(dual_last),
                "primal_total_retained_entries": len(self.primal_bank),
                "dual_total_retained_entries": len(self.dual_bank),
                "primal_retention_capacity": self.primal_retention_capacity,
                "dual_retention_capacity": self.dual_retention_capacity,
                "configured_bank_size": self.bank_size,
                "primal_history_enabled": self._use_primal_pool and self.bank_size > 1,
                "dual_history_enabled": self._use_dual_pool and self.bank_size > 1}

    def _finish(self, c, delta, lower, upper, route, metrics, begin):
        proof = bundle_proof(self.digest, c, delta, lower, upper)
        decision = proof["robust_threshold_decision"]
        result = {**metrics, **self._retention_metrics(),
                  "status": "certified" if decision is True else "unresolved",
                  "route": route, "lower_bound": lower["claimed_lower"],
                  "upper_bound": None if upper is None else upper["claimed_upper"],
                  "certificate": proof, "robust_threshold_decision": decision,
                  "decision_reason": "checked_dual_lower" if decision is True else
                      "checked_primal_counterexample" if decision is False else "checked_bounds_straddle_threshold",
                  "negative_scope": "minimum over this represented confidence view only",
                  "bank_entries": len(self.dual_bank), "primal_bank_entries": len(self.primal_bank),
                  "maximum_dual_bank_entries": self.maximum_dual_bank_entries,
                  "maximum_primal_bank_entries": self.maximum_primal_bank_entries,
                  "native_highs_version": self.native_version, "simplex_strategy": self.simplex_strategy,
                  "simplex_iterations": sum(r["simplex_iterations"] for r in metrics["native_calls"]),
                  "basis_valid_after": bool(self.h.getBasis().valid),
                  "checker_cache": self.checker.cache_info()}
        result["seconds"] = time.perf_counter() - begin
        return result

    def query(self, objective, threshold=.001):
        begin = time.perf_counter()
        c = _floats(objective)
        delta = _floats([threshold])[0]
        if len(c) != self.d:
            raise ValueError("objective dimension differs from the trusted view")
        with np.errstate(over="ignore", invalid="ignore"):
            norm = float(np.sum(np.abs(c)))
        if not math.isfinite(norm):
            raise ValueError("objective L1 norm exceeds finite implementation range")
        metrics = self._new_metrics()
        lower, best_upper = None, None
        initial = self._ensure_initial(metrics)
        if self.plan != "full_hot":
            # Every threshold plan has the same interval-only positive shortcut.
            lower = self._lower_proof(c, delta, self._zeros, self._zeros, metrics,
                                      prepared=self._zero_dual)
            if lower["claimed_lower"] > delta:
                return self._finish(c, delta, lower, None, "box_shortcut", metrics, begin)
            # The common last dual always has priority over older candidates.
            # Last is already one of the retained FIFO entries, never a 17th.
            tiers = [("last_dual", [] if self.last_dual is None else [self.last_dual])]
            if self._use_dual_pool and self.bank_size > 1:
                tiers.append(("historical_dual", [p for p in self.dual_bank if p is not self.last_dual]))
            for route, bank in tiers:
                for candidate in self._screen_duals(c, norm, delta, bank, metrics):
                    try:
                        checked = self._lower_proof(c, delta, candidate.lower, candidate.upper,
                                                    metrics, scale=norm, prepared=candidate.prepared)
                    except (AttributeError, TypeError, ValueError, ArithmeticError):
                        metrics["invalid_bank_entries"] += 1
                        continue
                    if checked["claimed_lower"] > lower["claimed_lower"]:
                        lower = checked
                    if lower["claimed_lower"] > delta:
                        return self._finish(c, delta, lower, None, route, metrics, begin)
            # Initial, then last, then historical FIFO points. Approximate
            # dots can skip checks, but only the exact factory may decide false.
            points = [("initial_primal", initial)]
            if self.last_primal is not None:
                points.append(("last_primal", self.last_primal))
            if self._use_primal_pool and self.bank_size > 1:
                points.extend(("historical_primal", p) for p in self.primal_bank if p is not self.last_primal)
            seen = set()
            for route, state in points:
                if not self.checker.owns_primal(state):
                    metrics["invalid_bank_entries"] += 1
                    continue
                if state.raw in seen:
                    continue
                seen.add(state.raw)
                screen_start = time.perf_counter()
                value = float(np.asarray(c) @ np.asarray(state.raw))
                metrics["primal_candidates_screened"] += 1
                metrics["screen_seconds"] += time.perf_counter() - screen_start
                # Always retain the initial upper, including near equality.
                if route != "initial_primal" and value > delta:
                    continue
                upper = self._upper_proof(c, state, metrics)
                if best_upper is None or upper["claimed_upper"] < best_upper["claimed_upper"]:
                    best_upper = upper
                if upper["claimed_upper"] <= delta:
                    return self._finish(c, delta, lower, upper, route, metrics, begin)

        cutoff = None if self.plan == "full_hot" else delta
        for attempt in range(2):
            proposal = self._native_run(c, cutoff, metrics)
            if proposal["lower"] is not None:
                checked = self._lower_proof(c, delta, proposal["lower"], proposal["upper"], metrics)
                if lower is None or checked["claimed_lower"] > lower["claimed_lower"]:
                    lower = checked
                self._save_dual(proposal["lower"], proposal["upper"], norm, metrics)
            if lower is None:
                lower = self._lower_proof(c, delta, self._zeros, self._zeros, metrics,
                                          prepared=self._zero_dual)
            positive = lower["claimed_lower"] > delta
            # Full optimization needs no upper evidence for an already proved
            # positive answer. Other plans may pay to retain a future shortcut.
            if proposal["primal"] is not None and (not positive or self.plan != "full_hot"):
                state = self._checked_point(proposal["primal"], metrics)
                if state is not None:
                    if self.plan != "full_hot":
                        self._save_point(state, metrics)
                    if not positive:
                        upper = self._upper_proof(c, state, metrics)
                        if best_upper is None or upper["claimed_upper"] < best_upper["claimed_upper"]:
                            best_upper = upper
            if positive:
                route = "solver_cutoff" if proposal["status"] == highspy.HighsModelStatus.kObjectiveBound else "solver"
                return self._finish(c, delta, lower, None, route, metrics, begin)
            if best_upper is None:
                best_upper = self._upper_proof(c, initial, metrics)
            if best_upper["claimed_upper"] <= delta:
                return self._finish(c, delta, lower, best_upper, "solver", metrics, begin)
            # A native cutoff that did not produce exact threshold evidence
            # resumes the same model/basis with the cutoff disabled.
            if cutoff is not None and proposal["status"] != highspy.HighsModelStatus.kOptimal:
                metrics["cutoff_retries"] += 1
                cutoff = None
                continue
            break
        return self._finish(c, delta, lower, best_upper, "solver_unknown", metrics, begin)

    def memory_info(self):
        """Python retained-state payload; not native solver RSS."""
        dual_arrays = sum(sys.getsizeof(x) + sys.getsizeof(x.lower) + sys.getsizeof(x.upper)
                          + x.residual.nbytes for x in self.dual_bank if isinstance(x, DualCandidate))
        primal_refs = sys.getsizeof(self.primal_bank)
        return {"checker_cache_payload_bytes": self.checker.cache_payload_bytes(),
                "retained_exact_state_payload_bytes": self.checker.cache_payload_bytes(
                    [self.initial_primal, self._zero_dual, *self.primal_bank,
                     *(x.prepared for x in self.dual_bank if isinstance(x, DualCandidate))]),
                "dual_screen_payload_bytes": dual_arrays, "primal_bank_reference_bytes": primal_refs,
                **self._retention_metrics(),
                "primal_bank_entries": len(self.primal_bank), "dual_bank_entries": len(self.dual_bank),
                "maximum_primal_bank_entries": self.maximum_primal_bank_entries,
                "maximum_dual_bank_entries": self.maximum_dual_bank_entries,
                "scope": "Approximate retained Python payload; shared view and native solver RSS excluded."}
