"""Drinfeld package — Phase 6.B: the Poisson Lie-bialgebroid
compatibility conditions (drinfeld eqs 4.36-4.38 on (TM, T*M_π))."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, vector_fields
from jacopy.packages.poisson import poisson_structure
from jacopy.packages.drinfeld.poisson_bialgebroid import (
    prove_compat_condition_one,
    prove_compat_condition_three_p1,
    prove_compat_condition_two,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    eta, mu = forms("η μ", degree=1)
    U, Y = vector_fields("U Y")
    P = poisson_structure()
    return reg, eta, mu, U, Y, P


class TestCompatibilityConditions:
    def test_condition_two(self, setup):
        """(4.37): ℒ_{d̃ι̃_ηU}μ = −[dι_Uη, μ]_π — bare."""
        reg, eta, mu, U, Y, P = setup
        assert prove_compat_condition_two(
            P, eta, mu, U, Y, registry=reg
        ).steps

    def test_condition_one_derivator(self, setup):
        """(4.36), the DERIVATOR condition (PDF item 10f): cited
        VF-Jacobi + bridge instances."""
        reg, eta, mu, U, Y, P = setup
        chain = prove_compat_condition_one(
            P, eta, mu, U, Y, registry=reg
        )
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_condition_three_p1(self, setup):
        """(4.38) at p = 1 (the 𝔻_Z g_Z side vanishes by
        alternation)."""
        reg, eta, mu, U, Y, P = setup
        assert prove_compat_condition_three_p1(
            P, eta, mu, U, Y, registry=reg
        ).steps
