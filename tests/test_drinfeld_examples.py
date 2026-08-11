"""Drinfeld package — Phase 6.H: example/regression suites from
[2409.11973 §8]: B_n-generalized geometry, the exceptional Courant
bracket, and the Atiyah bridge to the Phase 4 Bianchi identity."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.poisson.nambu import nambu_structure
import jacopy.packages.drinfeld.examples as ex


@pytest.fixture()
def bn_setup():
    reg = PropertyRegistry()
    f, g, s = functions("fb gb sb", registry=reg)
    U, V, Y = vector_fields("U V Y")
    om, et = forms("ωb ηb", degree=1)
    return reg, f, g, s, U, V, Y, om, et


class TestBnGeneralizedGeometry:
    """(8.15)-(8.16): TM ⊕ C∞M ⊕ T*M with the SO(n+1,n) pairing."""

    def test_symmetric_part_is_d_pairing(self, bn_setup):
        reg, f, g, s, U, V, Y, om, et = bn_setup
        c1, c2, c3 = ex.prove_bn_symmetric_part(
            U, f, om, V, g, et, Y, registry=reg
        )
        assert c1.steps and c2.steps and c3.steps

    def test_right_leibniz(self, bn_setup):
        reg, f, g, s, U, V, Y, om, et = bn_setup
        c1, c2, c3 = ex.prove_bn_right_leibniz(
            U, f, om, V, g, et, s, Y, registry=reg
        )
        assert c1.steps and c2.steps and c3.steps


@pytest.fixture()
def exc_setup():
    reg = PropertyRegistry()
    (f,) = functions("fe", registry=reg)
    U, V = vector_fields("U V")
    om2, et2 = forms("Ωe2 He2", degree=2)
    om5, et5 = forms("Ωe5 He5", degree=5)
    slots2 = list(vector_fields("Ye1 Ye2"))
    slots5 = list(vector_fields("Ze1 Ze2 Ze3 Ze4 Ze5"))
    return reg, f, U, V, om2, et2, om5, et5, slots2, slots5


class TestExceptionalCourant:
    """(8.1): TM ⊕ Λ² ⊕ Λ⁵ with the M-theory cross-term η₂∧dω₂ —
    the user's X+α+β question, now a package theorem suite (5-slot
    mixed-degree wedge evaluations)."""

    def test_symmetric_part_closes_on_exceptional_pairing(
        self, exc_setup
    ):
        reg, f, U, V, om2, et2, om5, et5, s2, s5 = exc_setup
        N = nambu_structure(p=2)
        c2, c5 = ex.prove_exceptional_symmetric_part(
            N, U, om2, om5, V, et2, et5, s2, s5, registry=reg
        )
        assert c2.steps and c5.steps

    def test_right_leibniz(self, exc_setup):
        reg, f, U, V, om2, et2, om5, et5, s2, s5 = exc_setup
        N = nambu_structure(p=2)
        c2, c5 = ex.prove_exceptional_right_leibniz(
            N, U, om2, om5, V, et2, et5, f, s2, s5, registry=reg
        )
        assert c2.steps and c5.steps


class TestAtiyahBridge:
    """(8.10)-(8.14): for an Atiyah algebroid the (5.13) H-closure
    condition IS the Bianchi identity d^∇F = 0 — the concrete
    mechanical content lives in the Phase 4 metric-affine package
    (curvature F of ∇, cited VF-Jacobi instances)."""

    def test_h_closure_is_bianchi_second(self):
        from jacopy.central.objects import connection, vector_fields
        from jacopy.packages.metric_affine import (
            prove_bianchi_second,
        )

        reg = PropertyRegistry()
        (f,) = functions("fa", registry=reg)
        X, Y, Z, W = vector_fields("Xa Ya Za Wa")
        nabla = connection("∇")
        chain, used = prove_bianchi_second(
            nabla, X, Y, Z, W, f, registry=reg
        )
        assert chain.steps and used
