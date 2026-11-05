"""Run the package tests with the registered single-thread solver policy.

HiGHS owns a process-global scheduler. Initializing reference LPs with its
automatic thread count and later requesting one thread can reject an otherwise
feasible program. Establish the shared policy before any test LP; never reset a
scheduler that may belong to another caller or modify the production solver.
"""
import warnings

import pytest
from scipy.optimize import OptimizeWarning, linprog


@pytest.fixture(scope="session", autouse=True)
def fixed_highs_thread_policy():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Unrecognized options detected: \{'threads': 1\}.*",
            category=OptimizeWarning,
        )
        probe = linprog([1.], bounds=[(0., 1.)], options={"threads": 1})
    if not probe.success:
        pytest.fail(
            "Cannot establish the single-thread HiGHS test policy. Run the "
            "icde2027 tests in a fresh Python process, before any caller "
            "initializes HiGHS with a different explicit thread count. "
            f"Solver report: {probe.message}"
        )
