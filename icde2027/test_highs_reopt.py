import numpy as np
import pytest
from .confidence_view import QueryEngine,create_view
from .highs_reopt import HighsReoptimization
from .proof_check import verify_certificate
from .test_confidence_view import teaching_view


@pytest.mark.parametrize('warm',[False,True])
def test_persistent_comparators_check_same_bounds(warm):
    view=teaching_view();engine=HighsReoptimization(view,retain_basis=warm)
    full=QueryEngine(view,reuse=False)
    for c in [[-.1,-.1,.1,.1],[-.105,-.09,.105,.09],[.1,.1,-.1,-.1],[0.,0.,0.,0.]]:
        answer=engine.query(c);expected=full.query(c)
        assert answer['lower_bound']==pytest.approx(expected['lower_bound'],abs=1e-12)
        assert verify_certificate(view,answer['certificate'],expected_objective=c,expected_threshold=.001)['valid_lp_lower_bound']
        assert answer['basis_valid_after']


def test_row_dual_sign_convention_both_active_sides():
    view=create_view([0.],[1.],[[1.]],[.3],[.7])['view']
    engine=HighsReoptimization(view)
    lower=engine.query([1.],0);upper=engine.query([-1.],0)
    assert lower['lower_bound']==pytest.approx(.3)
    assert upper['lower_bound']==pytest.approx(-.7)
    assert lower['certificate']['lower_multipliers'][0]>0
    assert upper['certificate']['upper_multipliers'][0]>0
