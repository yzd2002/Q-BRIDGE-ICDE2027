"""Metadata-only policy sequences and a common checked objective cache.

No real-data run or timing occurs on import. New workflow inputs must be bound
to independently audited represented-view artifacts by the separate driver.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import itertools
import json
import math
import sys
import weakref

import numpy as np

from .threshold_extension.proofs import ThresholdChecker, _floats, bundle_proof


FEATURES = ('onfrontpage:yes', 'gender:men', 'jackettype:fleece',
            'jackettype:waterproof', 'color:black')
MULTIPLIERS = (.5, 1., 2.)
THRESHOLDS = (0., .001, .01)
VIEWS = ('observation', 'randomized_group_score', 'randomized_cell_projection',
         'fusion_group_score', 'fusion_cell_projection')
METHODS = ('current_prepared', 'optimized_last', 'native_full_hot')
FIELDS = ('status', 'route', 'lower_bound', 'upper_bound',
          'robust_threshold_decision', 'certificate', 'reason')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value):
    return Fraction.from_float(_floats([value])[0])


def pair(value):
    return [str(value.numerator), str(value.denominator)]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def row_basis(rows):
    """Exact forward echelon basis; no tolerance-based rank classifications."""
    basis = {}
    for source in rows:
        row = list(source)
        for pivot, prior in sorted(basis.items()):
            if row[pivot]:
                factor = row[pivot]
                row = [a-factor*b for a, b in zip(row, prior)]
        pivot = next((i for i, v in enumerate(row) if v), None)
        if pivot is not None:
            factor = row[pivot]
            basis[pivot] = tuple(v/factor for v in row)
    return basis


def in_span(row, basis):
    residual = list(row)
    for pivot, prior in sorted(basis.items()):
        if residual[pivot]:
            factor = residual[pivot]
            residual = [a-factor*b for a, b in zip(residual, prior)]
    return not any(residual)


def catalogue_queries(features, names, rows, *, feature_names=FEATURES):
    """Construct all prespecified positions using metadata only, never outcomes."""
    features = np.asarray(features)
    require(features.ndim == 2 and features.shape[1] == len(names) and len(features) > 0,
            'Metadata dimensions differ')
    require(len(set(names)) == len(names) and all(f in names for f in feature_names),
            'Distinct named metadata columns required')
    require(np.all((features == 0) | (features == 1)), 'Binary catalogue features required')
    d = len(features)
    require(rows.shape[1] == d, 'Moment matrix and catalogue differ')
    dense = [[exact(x) for x in row] for row in rows.toarray()]
    basis = row_basis(dense)
    policies = []
    reference = np.full(d, 1/d)
    for f1, f2 in itertools.combinations(feature_names, 2):
        h1, h2 = features[:, names.index(f1)], features[:, names.index(f2)]
        for row_index, m1 in enumerate(MULTIPLIERS):
            for m2 in (MULTIPLIERS if row_index % 2 == 0 else tuple(reversed(MULTIPLIERS))):
                if m1 == m2 == 1.:
                    continue
                weights = np.power(m1, h1)*np.power(m2, h2)
                policy = weights/weights.sum()
                objective = (policy-reference).tolist()
                exact_weights = [Fraction(m1)**int(a)*Fraction(m2)**int(b) for a, b in zip(h1, h2)]
                total = sum(exact_weights, Fraction())
                ideal = [w/total-Fraction(1, d) for w in exact_weights]
                represented = [exact(x) for x in objective]
                error = [a-b for a, b in zip(ideal, represented)]
                policies.append({'policy_index': len(policies), 'features': [f1, f2],
                                 'multipliers': [m1, m2], 'policy': policy.tolist(),
                                 'reference': reference.tolist(), 'objective': objective,
                                 'ideal_objective_exact': [pair(x) for x in ideal],
                                 'ideal_in_moment_row_span': in_span(ideal, basis),
                                 'represented_in_moment_row_span': in_span(represented, basis),
                                 'represented_sum_exact': pair(sum(represented, Fraction())),
                                 'ideal_transfer_lower_exact': pair(sum((min(x, Fraction()) for x in error), Fraction())),
                                 'ideal_transfer_upper_exact': pair(sum((max(x, Fraction()) for x in error), Fraction())),
                                 'representation_l1_error_exact': pair(sum((abs(a-b) for a, b in zip(ideal, represented)), Fraction()))})
    requests = [{'query_index': i*len(policies)+p['policy_index'],
                 'policy_index': p['policy_index'], 'objective': p['objective'],
                 'threshold': delta, 'features': p['features'], 'multipliers': p['multipliers']}
                for i, delta in enumerate(THRESHOLDS) for p in policies]
    unique = {}
    for p in policies:
        unique.setdefault(tuple(p['objective']), p)
    represented_rows = [[exact(x) for x in p['objective']] for p in unique.values()]
    ideal_rows = [[Fraction(int(a), int(b)) for a, b in p['ideal_objective_exact']] for p in unique.values()]
    return {'format': 'qbridge-coat-metadata-workflow-v5', 'policies': policies, 'requests': requests,
            'statistics': {'items': d, 'feature_pairs': math.comb(len(feature_names), 2),
                           'policy_positions': len(policies), 'requests_per_view': len(requests),
                           'unique_objectives': len(unique),
                           'unique_objective_threshold_pairs': len({(tuple(r['objective']), r['threshold']) for r in requests}),
                           'moment_rank': len(basis), 'represented_objective_rank': len(row_basis(represented_rows)),
                           'ideal_objective_rank': len(row_basis(ideal_rows)),
                           'unique_represented_in_span': sum(p['represented_in_moment_row_span'] for p in unique.values()),
                           'unique_ideal_in_span': sum(p['ideal_in_moment_row_span'] for p in unique.values())},
            'source': 'Researcher-defined complete metadata workflow; no deployed query trace or new statistical samples.'}


def check_answer(checker, answer, objective, threshold):
    require(set(answer) == set(FIELDS), 'Common response fields required')
    decision = answer['robust_threshold_decision']
    require(decision is None or type(decision) is bool, 'Three-valued decision required')
    proof = answer['certificate']
    if proof is None:
        require(answer['status'] == 'uncertain' and answer['lower_bound'] is None
                and answer['upper_bound'] is None and decision is None
                and isinstance(answer['reason'], str) and answer['reason'].strip(), 'Malformed refusal')
        return False
    require(answer['reason'] is None, 'Proof-bearing answer cannot claim a numerical refusal')
    checked = checker.verify(proof, expected_objective=objective, expected_threshold=threshold)
    require(checked.get('valid') is True and answer['lower_bound'] == checked['lower_bound']
            and answer['upper_bound'] == checked['upper_bound']
            and decision is checked['robust_threshold_decision']
            and answer['status'] == ('certified' if decision is True else 'unresolved'),
            'Outer answer disagrees with checked evidence')
    return True


def service(answer):
    return {name: answer.get(name) for name in FIELDS}


def from_bundle(proof, route):
    decision = proof['robust_threshold_decision']
    return {'status': 'certified' if decision is True else 'unresolved', 'route': route,
            'lower_bound': proof['lower_proof']['claimed_lower'],
            'upper_bound': None if proof['upper_proof'] is None else proof['upper_proof']['claimed_upper'],
            'robust_threshold_decision': decision, 'certificate': proof, 'reason': None}


def ideal_transfer(answer, policy, threshold):
    """Conservative arithmetic transfer to the exact intended policy contrast."""
    delta = exact(threshold)
    errors = [Fraction(int(a), int(b)) for a, b in
              (policy['ideal_transfer_lower_exact'], policy['ideal_transfer_upper_exact'])]
    require(errors[0] <= 0 <= errors[1], 'Invalid box error transfer')
    lower = None if answer['lower_bound'] is None else exact(answer['lower_bound'])+errors[0]
    upper = None if answer['upper_bound'] is None else exact(answer['upper_bound'])+errors[1]
    require(lower is None or upper is None or lower <= upper, 'Contradictory transferred evidence')
    verdict = True if lower is not None and lower > delta else False if upper is not None and upper <= delta else None
    return {'lower_exact': None if lower is None else pair(lower),
            'upper_exact': None if upper is None else pair(upper),
            'ideal_robust_threshold_decision': verdict,
            'represented_robust_threshold_decision': answer['robust_threshold_decision']}


@dataclass(frozen=True)
class ObjectiveEvidence:
    objective: tuple
    wire: str


class ExactObjectiveCache:
    """Common full-capacity cache: locally issued immutable evidence only.

    A separate common checker validates every inserted/merged certificate.
    It is not a token accepted from the caller. Rebinding a threshold changes
    no mathematical lower/upper value, but always re-evaluates the decision.
    """
    def __init__(self, engine, view, *, checker_class=ThresholdChecker):
        self.engine = engine
        self.checker = checker_class(view, cache_size=16)
        require(engine.digest == self.checker.digest, 'Engine and cache view differ')
        self.digest = self.checker.digest
        self._cache, self._issued = {}, {}
        self.stats = Counter()

    def _remember(self, key, proof):
        state = ObjectiveEvidence(key, encoded(proof))
        owner = weakref.ref(self); identity = id(state)
        def cleanup(ref):
            obj = owner()
            if obj is not None and obj._issued.get(identity, (None,))[0] is ref:
                del obj._issued[identity]
        self._issued[identity] = (weakref.ref(state, cleanup), state.objective, state.wire)
        self._cache[key] = state

    def _owns(self, state, key):
        if not isinstance(state, ObjectiveEvidence):
            return False
        saved = self._issued.get(id(state))
        return (saved is not None and saved[0]() is state and state.objective is saved[1]
                and state.wire is saved[2] and state.objective == key)

    def _rebind(self, proof, c, delta):
        require(proof['view_digest'] == self.digest and _floats(proof['objective']) == c,
                'Cached evidence differs from view/objective')
        lower = dict(proof['lower_proof']); lower['decision_threshold'] = delta
        return bundle_proof(self.digest, c, delta, lower, proof['upper_proof'])

    def query(self, objective, threshold):
        c, delta = _floats(objective), _floats([threshold])[0]
        require(len(c) == self.checker.d, 'Objective differs from view dimension')
        self.stats['requests'] += 1
        saved = self._cache.get(c); old = None
        if saved is not None:
            if not self._owns(saved, c):
                self.stats['invalid_cached_entries'] += 1
                del self._cache[c]
            else:
                old = self._rebind(json.loads(saved.wire), c, delta)
                if old['robust_threshold_decision'] is not None:
                    self.stats['decisive_cache_hits'] += 1
                    return from_bundle(old, 'objective_cache'), {'engine_invoked': False, 'cache': dict(self.stats)}
                self.stats['nondecisive_cache_hits'] += 1
        self.stats['engine_calls'] += 1
        native = self.engine.query(c, delta)
        fresh = service(native)
        valid = check_answer(self.checker, fresh, c, delta)
        if not valid and old is None:
            return fresh, {'engine_invoked': True, 'native': native, 'cache': dict(self.stats)}
        proof = fresh['certificate'] if valid else old
        if old is not None and valid:
            lower = max((old['lower_proof'], proof['lower_proof']), key=lambda x: x['claimed_lower'])
            uppers = [p for p in (old['upper_proof'], proof['upper_proof']) if p is not None]
            upper = min(uppers, key=lambda x: x['claimed_upper']) if uppers else None
            proof = bundle_proof(self.digest, c, delta, lower, upper)
        answer = from_bundle(proof, 'objective_merge' if old is not None else fresh['route'])
        # Merge may use two independently valid candidates; verify the complete
        # resulting contract before issuing immutable local evidence.
        if old is not None:
            check_answer(self.checker, answer, c, delta)
        self._remember(c, proof)
        self.stats['maximum_objectives'] = max(self.stats['maximum_objectives'], len(self._cache))
        return answer, {'engine_invoked': True, 'native': native, 'cache': dict(self.stats)}

    def memory_info(self):
        # Exact encoded payload plus Python-owned objective/container footprint;
        # whole-process RSS is recorded by the execution driver separately.
        payload = sum(sys.getsizeof(s)+sys.getsizeof(s.wire)+sys.getsizeof(s.objective)
                      +sum(sys.getsizeof(x) for x in s.objective) for s in self._cache.values())
        return {'cached_objectives': len(self._cache), 'objective_cache_payload_bytes': payload,
                'cache_checker_payload_bytes': self.checker.cache_payload_bytes(),
                'counter': dict(self.stats),
                'scope': 'Object/encoded payload estimate; excludes shared view and native solver.'}
