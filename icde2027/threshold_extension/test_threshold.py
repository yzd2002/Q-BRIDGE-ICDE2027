"""Correctness gates, not timing experiments, for two-sided threshold answers."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
import math
import gc
from pathlib import Path
import subprocess
import sys
import weakref

import highspy
import numpy as np
import pytest

from ..proof_check import CheckedView
from ..test_confidence_view import teaching_view
from .engine import PLANS, ThresholdEngine
from .proofs import CheckedPrimal, ThresholdChecker, bundle_proof, upper_float


THRESHOLD_CONFIGS = [(plan, 1) for plan in PLANS if plan != "full_hot"] + [("threshold_hot", 0)]
ALL_CONFIGS = [("full_hot", 1), *THRESHOLD_CONFIGS]


def view(lower, upper, *, rows=(), rhs_lower=(), rhs_upper=(), point=None):
    ids, vals, ptr = [], [], [0]
    for row in rows:
        for i, value in enumerate(row):
            if value:
                ids.append(i); vals.append(float(value))
        ptr.append(len(ids))
    return {"format": "qbridge-confidence-view-v1", "lower": list(lower), "upper": list(upper),
            "rhs_lower": list(rhs_lower), "rhs_upper": list(rhs_upper),
            "matrix": {"indptr": ptr, "indices": ids, "data": vals},
            "feasible_point": list(lower if point is None else point),
            "metadata": {"kind": "correctness_test_not_measurement"}}


def make_proof(checker, c, delta, point=None, lo=None, hi=None):
    lo = [0.] * checker.j if lo is None else lo
    hi = [0.] * checker.j if hi is None else hi
    lower = checker.lower_proof(c, lo, hi, threshold=delta)
    upper = None if point is None else checker.upper_proof(c, checker.primal_state(point))
    return bundle_proof(checker.digest, c, delta, lower, upper)


def verify_answer(v, c, delta, answer):
    checked = ThresholdChecker(v).verify(answer["certificate"], expected_objective=c, expected_threshold=delta)
    assert checked["lower_bound"] == answer["lower_bound"]
    assert checked["upper_bound"] == answer["upper_bound"]
    assert checked["robust_threshold_decision"] is answer["robust_threshold_decision"]
    assert answer["status"] == ("certified" if checked["robust_threshold_decision"] is True else "unresolved")
    assert checked["valid"] and checked["valid_lp_lower_bound"]
    # The embedded lower retains compatibility, but the outer two-sided
    # format must not be silently mistaken for an old one-sided certificate.
    CheckedView(v).verify(answer["certificate"]["lower_proof"], expected_objective=c, expected_threshold=delta)
    with pytest.raises(ValueError):
        CheckedView(v).verify(answer["certificate"], expected_objective=c, expected_threshold=delta)
    return checked


def test_equality_proves_only_robust_noncertification_not_small_true_effect():
    v = view([.5, 0.], [.75, 0.])
    c, delta = [.25, -.25], .125
    checker = ThresholdChecker(v)
    proof = make_proof(checker, c, delta, point=[.5, 0.])
    result = checker.verify(proof, expected_objective=c, expected_threshold=delta)
    assert result["robust_threshold_decision"] is False
    assert result["upper_bound"] == delta
    assert Fraction(3, 4) * Fraction(1, 4) > Fraction(1, 8)
    assert result["scientific_assumptions_verified"] is False


def test_one_ulp_positive_minimum_can_remain_unknown_with_legacy_lower_rounding():
    delta = .001
    value = math.nextafter(delta, math.inf)
    v = view([1., 0.], [1., 0.])
    c = [value, -value]
    checker = ThresholdChecker(v)
    proof = make_proof(checker, c, delta, point=[1., 0.])
    assert proof["lower_proof"]["claimed_lower"] == delta
    assert proof["upper_proof"]["claimed_upper"] == value
    assert checker.verify(proof, expected_objective=c, expected_threshold=delta)["robust_threshold_decision"] is None
    # The implementation must not reclassify this unknown as a proved false.
    for plan, strategy in ALL_CONFIGS:
        result = ThresholdEngine(v, plan, simplex_strategy=strategy).query(c, delta)
        assert result["robust_threshold_decision"] is None
        verify_answer(v, c, delta, result)


def test_almost_feasible_point_cannot_authorize_negative_decision():
    v = view([.5, 0.], [.5, 0.])
    checker = ThresholdChecker(v)
    c, delta = [.25, -.25], .125 - 2. ** -53
    bad_point = [.5 - 2. ** -50, 0.]
    assert sum(x * y for x, y in zip(c, bad_point)) <= delta
    with pytest.raises(ValueError, match="exactly satisfy"):
        checker.primal_state(bad_point)
    proof = make_proof(checker, c, delta)
    proof["upper_proof"] = {"point": bad_point, "claimed_upper": sum(x*y for x, y in zip(c, bad_point))}
    proof["robust_threshold_decision"] = False
    with pytest.raises(ValueError):
        checker.verify(proof, expected_objective=c, expected_threshold=delta)


def test_upper_rounding_preserves_representable_equality_and_is_outward():
    for value in [Fraction(1, 8), Fraction(0), Fraction(-1, 8), Fraction(1, 10),
                  Fraction(-1, 10), Fraction(1, 2 ** 1075)]:
        rounded = upper_float(value)
        assert Fraction.from_float(rounded) >= value
        before = math.nextafter(rounded, -math.inf)
        assert Fraction.from_float(before) < value
    assert upper_float(Fraction(1, 8)) == .125


@pytest.mark.parametrize("cache_size", [0, 1, 16])
def test_receiver_cache_policy_and_stateless_equivalence(cache_size):
    v = view([0., 0.], [1., 1.])
    issuer = ThresholdChecker(v)
    receiver = ThresholdChecker(v, cache_size=cache_size)
    for i in range(33):
        c, delta, point = [.25, -.25], .3, [i/32, .5]
        proof = make_proof(issuer, c, delta, point=point)
        assert receiver.verify(proof, expected_objective=c, expected_threshold=delta)["robust_threshold_decision"] is False
        assert receiver.cache_info()["primal_cache_entries"] <= cache_size
        assert receiver.cache_info()["dual_cache_entries"] <= cache_size


def test_unissued_and_mutated_primal_tokens_are_not_trusted():
    v = view([0., 0.], [1., 1.])
    checker = ThresholdChecker(v)
    good = checker.primal_state([.5, .5])
    foreign = ThresholdChecker(v).primal_state([.5, .5])
    forged = CheckedPrimal(checker.digest, (.5, .5), (Fraction(-100), Fraction(100)))
    for bad in [foreign, forged, replace(good, point=(Fraction(-100), Fraction(100)))]:
        with pytest.raises(ValueError):
            checker.upper_proof([.25, -.25], bad)
        checker._point_cache[(.5, .5)] = bad
        restored = checker.primal_state([.5, .5])
        assert checker.upper_proof([.25, -.25], restored)["claimed_upper"] == 0.
    good = checker.primal_state([.5, .5])
    object.__setattr__(good, "point", (Fraction(-100), Fraction(100)))
    assert not checker.owns_primal(good)
    restored = checker.primal_state([.5, .5])
    assert checker.upper_proof([.25, -.25], restored)["claimed_upper"] == 0.


def test_poisoned_prepared_dual_is_rebuilt_from_bound_multipliers():
    v = view([0.], [1.], rows=[[1.]], rhs_lower=[.25], rhs_upper=[.75], point=[.5])
    checker = ThresholdChecker(v)
    state = checker.dual_state([1.], [0.])
    object.__setattr__(state, "constant", Fraction(1000))
    proof = checker.lower_proof([1.], [1.], [0.], threshold=.1)
    assert proof["claimed_lower"] < .25
    assert proof["claimed_lower"] > .249999
    CheckedView(v).verify(proof, expected_objective=[1.], expected_threshold=.1)
    assert checker.cache_info()["invalid_cache_entries"] == 1


@pytest.mark.parametrize("kind", ["view", "objective", "threshold", "decision", "upper", "lower", "bool_decision"])
def test_tampered_query_or_bound_rejected(kind):
    v = view([.5, 0.], [.75, 0.])
    c, delta = [.25, -.25], .125
    checker = ThresholdChecker(v)
    proof = make_proof(checker, c, delta, point=[.5, 0.])
    target = deepcopy(v)
    if kind == "view":
        target["metadata"]["new"] = True
    elif kind == "objective":
        proof["objective"][0] += .125
    elif kind == "threshold":
        proof["decision_threshold"] += .125
    elif kind == "decision":
        proof["robust_threshold_decision"] = True
    elif kind == "upper":
        proof["upper_proof"]["claimed_upper"] -= .125
    elif kind == "lower":
        proof["lower_proof"]["claimed_lower"] += .5
    else:
        proof["robust_threshold_decision"] = 0
    with pytest.raises(ValueError):
        ThresholdChecker(target).verify(proof, expected_objective=c, expected_threshold=delta)


def test_nested_proof_binding_and_global_feasibility():
    v = view([0., 0.], [1., 1.], rows=[[.5, .5]], rhs_lower=[.5], rhs_upper=[.75], point=[.5, .5])
    checker = ThresholdChecker(v)
    proof = make_proof(checker, [.25, -.25], .1, point=[.5, .5])
    proof["lower_proof"]["decision_threshold"] = .2
    with pytest.raises(ValueError):
        checker.verify(proof, expected_objective=[.25, -.25], expected_threshold=.1)
    with pytest.raises(ValueError):
        checker.primal_state([.125, .125])
    bad_view = deepcopy(v); bad_view["feasible_point"] = [.125, .125]
    with pytest.raises(ValueError):
        ThresholdChecker(bad_view)
    with pytest.raises(ValueError):
        ThresholdEngine(bad_view)


@pytest.mark.parametrize("field,malformed", [("lower_proof", None), ("lower_proof", []), ("upper_proof", [])])
def test_malformed_nested_proof_structure_is_explicitly_rejected(field, malformed):
    v = view([.5, 0.], [.75, 0.])
    checker = ThresholdChecker(v)
    proof = make_proof(checker, [.25, -.25], .125, point=[.5, 0.])
    proof[field] = malformed
    with pytest.raises(ValueError):
        checker.verify(proof, expected_objective=[.25, -.25], expected_threshold=.125)


@pytest.mark.parametrize("plan,strategy", ALL_CONFIGS)
def test_independent_exact_vertices_and_every_returned_proof(plan, strategy):
    v = view([0., 0.], [1., 1.], rows=[[.5, .5]], rhs_lower=[.25], rhs_upper=[.75], point=[.75, .75])
    vertices = [(0, Fraction(1, 2)), (0, 1), (Fraction(1, 2), 0), (1, 0),
                (1, Fraction(1, 2)), (Fraction(1, 2), 1)]
    engine = ThresholdEngine(v, plan, simplex_strategy=strategy)
    for numerator in range(-8, 9):
        c = [numerator / 16, -numerator / 16]
        rational_c = list(map(Fraction.from_float, c))
        optimum = min(sum(a*b for a, b in zip(rational_c, p)) for p in vertices)
        for delta in [float(optimum), math.nextafter(float(optimum), -math.inf), .001]:
            answer = engine.query(c, delta)
            verified = verify_answer(v, c, delta, answer)
            assert Fraction.from_float(answer["lower_bound"]) <= optimum
            if answer["upper_bound"] is not None:
                assert Fraction.from_float(answer["upper_bound"]) >= optimum
            if verified["robust_threshold_decision"] is True:
                assert optimum > Fraction.from_float(delta)
            elif verified["robust_threshold_decision"] is False:
                assert optimum <= Fraction.from_float(delta)


def test_primal_history_and_last_are_distinguished_from_initial_witness():
    v = view([0., 0.], [1., 1.], rows=[[.5, .5]], rhs_lower=[.25], rhs_upper=[.75], point=[.75, .75])
    c1, c2, delta = [.25, -.25], [-.25, .25], -.05
    engine = ThresholdEngine(v, "primal_pool")
    first = engine.query(c1, delta)
    previous = engine.query(c1, delta)
    second = engine.query(c2, delta)
    historical = engine.query(c1, delta)
    assert first["route"] == second["route"] == "solver"
    assert previous["route"] == "last_primal"
    assert historical["route"] == "historical_primal"
    assert previous["last_primal_generation"] == first["last_primal_generation"] == 1
    assert historical["last_primal_generation"] == second["last_primal_generation"] == 2
    assert historical["last_dual_generation"] == second["last_dual_generation"]
    for c, answer in [(c1, first), (c1, previous), (c2, second), (c1, historical)]:
        assert answer["robust_threshold_decision"] is False
        verify_answer(v, c, delta, answer)


@pytest.mark.parametrize("plan,strategy", THRESHOLD_CONFIGS)
def test_common_last_dual_hit_and_positive_proofs_do_not_ship_unneeded_upper(plan, strategy):
    v = teaching_view()
    engine = ThresholdEngine(v, plan, simplex_strategy=strategy)
    first = engine.query([-.1, -.1, .1, .1], .001)
    c = [-.105, -.09, .105, .09]
    second = engine.query(c, .001)
    assert first["route"] == "solver"
    assert second["route"] == "last_dual"
    assert second["native_calls"] == []
    assert second["upper_bound"] is None
    assert second["last_dual_generation"] == first["last_dual_generation"] == 1
    assert second["last_primal_generation"] == first["last_primal_generation"]
    assert second["last_dual_updates"] == second["last_primal_updates"] == 0
    assert second["dual_history_entries"] == 0
    assert second["dual_total_retained_entries"] == 1
    assert second["checker_cache"]["cache_size"] == 0
    assert second["checker_cache"]["dual_preparations"] == first["checker_cache"]["dual_preparations"]
    assert second["checker_cache"]["primal_exact_checks"] == first["checker_cache"]["primal_exact_checks"]
    verify_answer(v, c, .001, second)
    full = ThresholdEngine(v, "full_hot").query([-.1, -.1, .1, .1], .001)
    assert full["upper_bound"] is None
    assert full["primal_candidates_exact_checked"] == 0
    assert full["primal_preparation_seconds"] == 0.


def identity_row_view(d):
    return view([0.] * d, [1.] * d, rows=np.eye(d).tolist(),
                rhs_lower=[.25] * d, rhs_upper=[.75] * d, point=[.5] * d)


@pytest.mark.parametrize("plan,strategy", ALL_CONFIGS)
@pytest.mark.parametrize("bank_size", [0, 1, 16])
def test_native_candidates_obey_total_capacity_and_evicted_states_are_released(plan, strategy, bank_size):
    # Twenty distinct sign patterns have twenty distinct, exactly represented
    # minimizers and duals. Equality thresholds require these native proposals
    # without letting earlier points prove the new negative answer.
    d, v = 5, identity_row_view(5)
    engine = ThresholdEngine(v, plan, bank_size=bank_size, simplex_strategy=strategy)
    first_point_ref = first_dual_ref = None
    for n in range(20):
        c = [(1. if n & (1 << i) else -1.) / 8 for i in range(d)]
        optimum = sum(ci * (.25 if ci > 0 else .75) for ci in c)
        result = engine.query(c, optimum)
        assert result["native_calls"]
        assert result["robust_threshold_decision"] is False
        verify_answer(v, c, optimum, result)
        for side in ("primal", "dual"):
            cap = result[f"{side}_retention_capacity"]
            total = result[f"{side}_total_retained_entries"]
            history = result[f"{side}_history_entries"]
            present = result[f"last_{side}_present"]
            assert history + int(present) == total <= cap <= 16
            assert history <= 15
            assert total == (min(n + 1, cap) if cap else 0)
            assert result[f"last_{side}_generation"] == (n + 1 if cap else 0)
            assert result[f"last_{side}_updates"] == (1 if cap else 0)
        assert result["primal_bank_entries"] == result["primal_total_retained_entries"]
        assert result["bank_entries"] == result["dual_total_retained_entries"]
        assert result["checker_cache"]["primal_cache_entries"] == 0
        assert result["checker_cache"]["dual_cache_entries"] == 0
        if n == 0 and plan != "full_hot":
            first_point_ref = weakref.ref(engine.last_primal)
            first_dual_ref = weakref.ref(engine.last_dual.prepared)
    gc.collect()
    if first_point_ref is not None:
        assert first_point_ref() is None
        assert first_dual_ref() is None
    # Weak ownership registries retain no evicted native candidates. The one
    # extra entry is the fixed initial witness or fixed zero prepared dual.
    assert len(engine.checker._issued_points) == len(engine.primal_bank) + 1
    assert len(engine.checker._issued_duals) == len(engine.dual_bank) + 1
    assert len(engine.checker.view_checker._issued_prepared) == len(engine.dual_bank) + 1
    memory = engine.memory_info()
    assert memory["retained_exact_state_payload_bytes"] > memory["checker_cache_payload_bytes"]


@pytest.mark.parametrize("plan,strategy", THRESHOLD_CONFIGS)
def test_bank_size_zero_disables_history_but_keeps_common_last(plan, strategy):
    v = teaching_view()
    engine = ThresholdEngine(v, plan, bank_size=0, simplex_strategy=strategy)
    first = engine.query([-.1, -.1, .1, .1], .001)
    result = engine.query([-.105, -.09, .105, .09], .001)
    assert first["native_calls"] and result["native_calls"] == []
    assert result["route"] == "last_dual"
    assert result["last_dual_generation"] == first["last_dual_generation"]
    assert result["last_primal_generation"] == first["last_primal_generation"]
    assert not result["primal_history_enabled"] and not result["dual_history_enabled"]
    assert result["primal_retention_capacity"] == result["dual_retention_capacity"] == 1
    assert result["primal_history_entries"] == result["dual_history_entries"] == 0
    verify_answer(v, [-.105, -.09, .105, .09], .001, result)


@pytest.mark.parametrize("plan", ["dual_pool", "both_pools"])
def test_historical_dual_hit_keeps_last_and_fifo_unchanged_and_last_tier_wins(plan):
    v = identity_row_view(2)
    engine = ThresholdEngine(v, plan)
    c1, c2, delta = [.25, 0.], [0., .25], .001
    first, second = engine.query(c1, delta), engine.query(c2, delta)
    assert first["native_calls"] and second["native_calls"]
    last = engine.last_dual
    fifo = tuple(id(x) for x in engine.dual_bank)
    old_hit = engine.query(c1, delta)
    assert old_hit["route"] == "historical_dual"
    assert old_hit["native_calls"] == []
    assert old_hit["dual_candidates_screened"] == 2  # last is not screened twice
    assert engine.last_dual is last
    assert tuple(id(x) for x in engine.dual_bank) == fifo
    for side in ("primal", "dual"):
        assert old_hit[f"last_{side}_generation"] == second[f"last_{side}_generation"]
        assert old_hit[f"last_{side}_updates"] == 0
    # Last is tested before the older pool even when both are available.
    last_hit = engine.query(c2, delta)
    assert last_hit["route"] == "last_dual"
    assert last_hit["dual_candidates_screened"] == 1
    verify_answer(v, c1, delta, old_hit)
    verify_answer(v, c2, delta, last_hit)


def test_native_duplicate_changes_last_without_promoting_fifo_and_reuses_exact_states():
    v = identity_row_view(2)
    engine = ThresholdEngine(v, "both_pools", bank_size=2)
    c1, c2, c3 = [-.25, .25], [.25, -.25], [.25, .25]
    delta = math.nextafter(-.125, -math.inf)
    first, second = engine.query(c1, delta), engine.query(c2, delta)
    assert first["robust_threshold_decision"] is second["robust_threshold_decision"] is None
    primal_order = [p.raw for p in engine.primal_bank]
    dual_order = [(p.lower, p.upper) for p in engine.dual_bank]
    repeat = engine.query(c1, delta)
    assert repeat["native_calls"] and repeat["robust_threshold_decision"] is None
    assert [p.raw for p in engine.primal_bank] == primal_order
    assert [(p.lower, p.upper) for p in engine.dual_bank] == dual_order
    assert engine.last_primal.raw == primal_order[0]
    assert (engine.last_dual.lower, engine.last_dual.upper) == dual_order[0]
    assert repeat["last_primal_generation"] > second["last_primal_generation"]
    assert repeat["last_dual_generation"] > second["last_dual_generation"]
    assert repeat["native_primal_reuses"] == repeat["native_primal_candidates"]
    assert repeat["native_dual_reuses"] == repeat["native_dual_candidates"]
    assert repeat["checker_cache"]["primal_exact_checks"] == second["checker_cache"]["primal_exact_checks"]
    last = engine.query(c3, math.nextafter(.125, -math.inf))
    assert last["native_calls"] and last["robust_threshold_decision"] is None
    assert engine.primal_bank[0].raw == primal_order[1]
    assert (engine.dual_bank[0].lower, engine.dual_bank[0].upper) == dual_order[1]
    for c, threshold, answer in [(c1, delta, repeat), (c3, math.nextafter(.125, -math.inf), last)]:
        verify_answer(v, c, threshold, answer)
        assert all(answer[key] >= 0. for key in
                   ("screen_seconds", "solve_seconds", "verify_seconds", "preparation_seconds"))
        assert answer["primal_preparation_seconds"] <= answer["verify_seconds"]


@pytest.mark.parametrize("plan,strategy", THRESHOLD_CONFIGS)
def test_common_last_primal_returns_checked_false_and_does_not_update_generations(plan, strategy):
    v = identity_row_view(2)
    engine = ThresholdEngine(v, plan, simplex_strategy=strategy)
    c, delta = [-.25, .25], -.05
    first = engine.query(c, delta)
    last = engine.query(c, delta)
    assert first["native_calls"] and last["native_calls"] == []
    assert last["route"] == "last_primal"
    assert last["robust_threshold_decision"] is False
    for side in ("primal", "dual"):
        assert last[f"last_{side}_generation"] == first[f"last_{side}_generation"]
        assert last[f"last_{side}_updates"] == 0
    verify_answer(v, c, delta, last)


def test_last_dual_poison_cannot_authorize_and_does_not_create_hidden_history():
    v = identity_row_view(2)
    engine = ThresholdEngine(v)
    first = engine.query([.25, 0.], .001)
    candidate = engine.last_dual
    # Poison both the approximate screen and the exact prepared state. The
    # exact factory must rebuild from the bound multipliers before deciding.
    object.__setattr__(candidate, "constant", 1000.)
    object.__setattr__(candidate.prepared, "constant", Fraction(1000))
    c, delta = [0., .25], .1
    result = engine.query(c, delta)
    assert result["robust_threshold_decision"] is False
    assert result["checker_cache"]["invalid_cache_entries"] >= 1
    assert result["dual_history_entries"] == 0
    verify_answer(v, c, delta, result)
    assert result["last_dual_generation"] >= first["last_dual_generation"]


def test_last_primal_poison_is_rejected_and_new_native_point_is_rechecked():
    v = identity_row_view(2)
    engine = ThresholdEngine(v)
    c, delta = [-.25, .25], -.05
    first = engine.query(c, delta)
    object.__setattr__(engine.last_primal, "point", (Fraction(100), Fraction(-100)))
    result = engine.query(c, delta)
    assert result["native_calls"]
    assert result["invalid_bank_entries"] >= 1
    assert result["last_primal_generation"] > first["last_primal_generation"]
    assert result["checker_cache"]["primal_exact_checks"] > first["checker_cache"]["primal_exact_checks"]
    assert result["primal_history_entries"] == 0
    verify_answer(v, c, delta, result)


def test_explicit_prepared_state_reuse_rejects_foreign_modified_and_bool_inputs():
    v = identity_row_view(1)
    checker = ThresholdChecker(v, cache_size=0)
    state = checker.dual_state([1.], [0.])
    foreign = ThresholdChecker(v, cache_size=0).dual_state([1.], [0.])
    for candidate in [state, foreign]:
        proof = checker.lower_proof([1.], [1.], [0.], threshold=.1, prepared=candidate)
        CheckedView(v).verify(proof, expected_objective=[1.], expected_threshold=.1)
    object.__setattr__(state, "offset", (Fraction(1000),))
    proof = checker.lower_proof([1.], [1.], [0.], threshold=.1, prepared=state)
    assert proof["claimed_lower"] == math.nextafter(.25, -math.inf)
    local = checker.dual_state([1.], [0.])
    for candidate in [local, foreign]:
        for bad in [True, math.nan, math.inf, -1.]:
            with pytest.raises(ValueError):
                checker.lower_proof([1.], [bad], [0.], threshold=.1, prepared=candidate)
    assert checker.cache_info()["dual_cache_entries"] == 0


@pytest.mark.parametrize("plan,strategy", ALL_CONFIGS)
def test_zero_objective_true_false_unknown_and_unscalable_native_dual_costs(plan, strategy):
    v = identity_row_view(2)
    engine = ThresholdEngine(v, plan, simplex_strategy=strategy)
    seed = engine.query([.25, 0.], .001)
    for delta, expected in [(-1., True), (0., False), (-math.ulp(0.), None)]:
        before_primal, before_dual = engine.last_primal_generation, engine.last_dual_generation
        answer = engine.query([0., 0.], delta)
        assert answer["robust_threshold_decision"] is expected
        verify_answer(v, [0., 0.], delta, answer)
        if plan == "full_hot":
            assert answer["native_calls"]
            assert answer["last_primal_generation"] == answer["last_dual_generation"] == 0
        elif expected is None:
            assert answer["native_calls"]
            assert answer["rejected_dual_proposals"] == answer["native_dual_candidates"] >= 1
            assert answer["last_dual_generation"] == before_dual
            assert answer["last_primal_generation"] > before_primal
            assert answer["preparation_seconds"] > 0.
        else:
            assert not answer["native_calls"]
            assert answer["last_dual_generation"] == before_dual
            assert answer["last_primal_generation"] == before_primal
    assert seed["native_calls"]


def cutoff_fixture():
    d = 128
    rows = []
    for i in range(0, d, 2):
        row = [0.] * d; row[i:i+2] = [.5, .5]; rows.append(row)
    v = view([0.] * d, [1.] * d, rows=rows, rhs_lower=[.5] * (d//2),
             rhs_upper=[1.] * (d//2), point=[1.] * d)
    c = [(1 + i / d) / d for i in range(d)]
    return v, c


def test_native_dual_cutoff_actually_triggers_and_exact_evidence_authorizes_it():
    v, c = cutoff_fixture()
    engine = ThresholdEngine(v)
    # This correctness probe isolates simplex from presolve; it is not a
    # registered measurement or a change to the execution default.
    engine.h.setOptionValue("presolve", "off")
    answer = engine.query(c, .1)
    assert answer["native_calls"][0]["model_status_code"] == int(highspy.HighsModelStatus.kObjectiveBound)
    assert answer["route"] == "solver_cutoff"
    assert answer["robust_threshold_decision"] is True
    assert len(answer["native_calls"]) == 1
    verify_answer(v, c, .1, answer)


def test_actual_cutoff_without_strict_exported_lower_retries_to_completion():
    v, c = cutoff_fixture()
    # At the first native bound check, this deterministic LP has dual value
    # .556640625. Its exported lower is one ULP down, exactly this threshold.
    delta = math.nextafter(.556640625, -math.inf)
    engine = ThresholdEngine(v)
    engine.h.setOptionValue("presolve", "off")
    answer = engine.query(c, delta)
    assert answer["native_calls"][0]["model_status_code"] == int(highspy.HighsModelStatus.kObjectiveBound)
    assert answer["cutoff_retries"] == 1
    assert len(answer["native_calls"]) == 2
    assert answer["native_calls"][1]["objective_bound"] is None
    assert answer["native_calls"][1]["model_status_code"] == int(highspy.HighsModelStatus.kOptimal)
    assert answer["rejected_primal_proposals"] >= 1
    assert answer["last_primal_updates"] + answer["rejected_primal_proposals"] == answer["native_primal_candidates"]
    assert answer["last_primal_generation"] == answer["last_primal_updates"] == 1
    assert answer["last_dual_generation"] == answer["last_dual_updates"] == 2
    assert answer["native_dual_candidates"] == 2
    assert answer["robust_threshold_decision"] is True
    assert answer["simplex_iterations"] == sum(x["simplex_iterations"] for x in answer["native_calls"])
    verify_answer(v, c, delta, answer)


def test_choose_strategy_observed_primal_on_retained_feasible_basis(tmp_path):
    d = 16
    v = view([0.] * d, [1.] * d, rows=[[1/d] * d], rhs_lower=[.75], rhs_upper=[1.], point=[1.] * d)
    engine = ThresholdEngine(v, simplex_strategy=0)
    log = tmp_path / "auto_correctness_probe.log"
    engine.h.setOptionValue("output_flag", True)
    engine.h.setOptionValue("log_to_console", False)
    engine.h.setOptionValue("log_file", str(log))
    for c in [[(1+i/d)/d for i in range(d)], [(2-i/d)/d for i in range(d)]]:
        # Use the analytically known minimum for this small correctness case
        # so the newly shared last dual cannot preempt this solver-path probe.
        delta = sum(sorted(c)[:12])
        answer = engine.query(c, delta)
        assert answer["simplex_strategy"] == 0
        assert answer["native_calls"]
        verify_answer(v, c, delta, answer)
    actual_log = log.read_text()
    assert "Using primal simplex solver" in actual_log
    assert "Solving LP with useful basis" in actual_log
    # This demonstrates the observed path for this probe only; the option 0
    # does not mean every future query uses the same simplex variant.


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("delta", [-1e200, -1., 0., 1., 1e200])
def test_large_negative_and_equal_thresholds_preserve_three_value_logic(plan, delta):
    v = view([0., 0.], [1., 1.], point=[.5, .5])
    c = [.25, -.25]
    result = ThresholdEngine(v, plan).query(c, delta)
    verify_answer(v, c, delta, result)
    expected = -.25 > delta
    assert result["robust_threshold_decision"] is expected


def test_invalid_numbers_shapes_and_cache_sizes_rejected():
    v = view([0., 0.], [1., 1.])
    engine = ThresholdEngine(v)
    for c, delta in [([True, 0.], 0.), ([math.nan, 0.], 0.), ([0., 0.], math.inf),
                     ([0.], 0.), ([0., 0.], True), ([1e308, 1e308], 0.)]:
        with pytest.raises(ValueError):
            engine.query(c, delta)
    for size in [-1, 17, True, .5]:
        with pytest.raises(ValueError):
            ThresholdChecker(v, size)
        with pytest.raises(ValueError):
            ThresholdEngine(v, bank_size=size)
    with pytest.raises(ValueError):
        ThresholdEngine(v, simplex_strategy=2)


def test_external_input_mutation_does_not_change_engine_or_checker_view():
    v = view([.5, 0.], [.75, 0.])
    saved = deepcopy(v)
    engine = ThresholdEngine(v)
    v["lower"][0] = .9
    c, delta = [.25, -.25], .125
    result = engine.query(c, delta)
    verify_answer(saved, c, delta, result)
    assert result["robust_threshold_decision"] is False


def test_standard_library_only_receiver_cli(tmp_path):
    v = view([.5, 0.], [.75, 0.])
    c, delta = [.25, -.25], .125
    proof = make_proof(ThresholdChecker(v), c, delta, point=[.5, 0.])
    for name, data in [("view", v), ("proof", proof), ("query", {"objective": c, "threshold": delta})]:
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    command = [sys.executable, "-B", "-S", "-m", "icde2027.threshold_extension.proofs",
               "--view", str(tmp_path/"view.json"), "--proof", str(tmp_path/"proof.json"),
               "--query", str(tmp_path/"query.json")]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["robust_threshold_decision"] is False
    proof["robust_threshold_decision"] = True
    (tmp_path / "proof.json").write_text(json.dumps(proof))
    invalid = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert invalid.returncode == 2

