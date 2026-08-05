"""Drinfeld package — Phase 6.C: the double bracket on TM ⊕ T*M.

The STANDARD double (zero tilde side) IS the Dorfman bracket; its
Leibniz-algebroid properties close mechanically. The Poisson double
(LWX/triangular) construction is exercised structurally."""

import pytest

from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.poisson import poisson_structure
from jacopy.packages.drinfeld.double import (
    canonical_pairing,
    dorfman_double,
    poisson_double,
    prove_dorfman_jacobi_form,
    prove_dorfman_right_leibniz_form,
    prove_dorfman_symmetric_part,
    prove_poisson_double_right_leibniz_form,
    prove_poisson_double_right_leibniz_vec,
    prove_poisson_double_symmetric_part_form,
    prove_poisson_double_symmetric_part_vec,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    U, V, W, Y = vector_fields("U V W Y")
    om, et, mu = forms("ω η μ", degree=1)
    return reg, f, U, V, W, Y, om, et, mu


class TestStandardDouble:
    def test_is_dorfman(self, setup):
        """The standard double's components ARE the Dorfman
        formula."""
        from jacopy.algebra.derivation import Act
        from jacopy.central.objects.interior import Interior
        from jacopy.central.tangent.exterior import CARTAN_TM, d

        _, _, U, V, _, _, om, et, _ = setup
        vec, form = dorfman_double(U, om, V, et)
        assert vec == lie_bracket(U, V)
        expected_form = Sum(
            Act(CARTAN_TM.lie(U), et),
            Neg(Act(CARTAN_TM.lie(V), om)),
            d(Act(Interior(V), om)),
        )
        assert form == expected_form

    def test_right_leibniz_form(self, setup):
        reg, f, U, V, _, Y, om, et, _ = setup
        chain, _ = prove_dorfman_right_leibniz_form(
            U, om, V, et, f, [Y], registry=reg
        )
        assert chain.steps

    def test_symmetric_part_is_exact(self, setup):
        """[e₁,e₂] + [e₂,e₁] = (0, d⟨e₁,e₂⟩₊)."""
        reg, f, U, V, _, Y, om, et, _ = setup
        chain, _ = prove_dorfman_symmetric_part(
            U, om, V, et, f, [Y], registry=reg
        )
        assert chain.steps

    def test_leibniz_jacobi_form(self, setup):
        """The flagship: the standard Dorfman bracket's
        Leibniz-Jacobi, form component (92 steps, cited
        VF-Jacobi)."""
        reg, f, U, V, W, Y, om, et, mu = setup
        chain, used = prove_dorfman_jacobi_form(
            U, om, V, et, W, mu, f, [Y], registry=reg
        )
        assert chain.steps
        assert used  # VF-Jacobi instances entered as citations


class TestPoissonDouble:
    def test_reduces_to_dorfman_plus_tilde(self, setup):
        """Structural: the Poisson double = standard Dorfman +
        π-tilde corrections; with the Koszul term removed and sharp
        terms dropped it matches the standard double."""
        reg, _, U, V, _, _, om, et, _ = setup
        P = poisson_structure()
        vec, form = poisson_double(P, U, om, V, et)
        std_vec, std_form = dorfman_double(U, om, V, et)
        # The standard parts are literally contained:
        assert std_vec in vec.children
        for t in std_form.children:
            assert t in form.children


class TestHigherDegreeDouble:
    """User probe 2026-08-04: the double on TM ⊕ Λᵖ (higher
    Dorfman) — degree-general; p = 2 in-suite, p = 7 verified
    offline (sym 157 / rL 144 / Jacobi 2483 steps)."""

    def test_p2_symmetric_part(self, setup):
        reg, f, U, V, _, _, _, _, _ = setup
        Y1, Y2 = vector_fields("Z1 Z2")
        om2, et2 = forms("Ω2 H2", degree=2)
        chain, _ = prove_dorfman_symmetric_part(
            U, om2, V, et2, f, [Y1, Y2], registry=reg
        )
        assert chain.steps

    def test_p2_right_leibniz(self, setup):
        reg, f, U, V, _, _, _, _, _ = setup
        Y1, Y2 = vector_fields("Z1 Z2")
        om2, et2 = forms("Ω2 H2", degree=2)
        chain, _ = prove_dorfman_right_leibniz_form(
            U, om2, V, et2, f, [Y1, Y2], registry=reg
        )
        assert chain.steps

    def test_p2_jacobi(self, setup):
        reg, f, U, V, W, _, _, _, _ = setup
        Y1, Y2 = vector_fields("Z1 Z2")
        om2, et2, mu2 = forms("Ω2 H2 M2", degree=2)
        chain, used = prove_dorfman_jacobi_form(
            U, om2, V, et2, W, mu2, f, [Y1, Y2], registry=reg
        )
        assert chain.steps


class TestPoissonDoubleTheorems:
    """Phase 6.D: the LWX/triangular double's Leibniz-algebroid
    properties — anchor ρ(U+ω) = U + π♯ω, generalized exterior
    derivative 𝒟 = d + d̃ with d̃ = −π♯d. All hold for ANY bivector
    (no [π,π] declaration in the engine)."""

    def test_symmetric_part_form(self, setup):
        reg, f, U, V, W, Y, om, et, mu = setup
        P = poisson_structure()
        chain = prove_poisson_double_symmetric_part_form(
            P, U, om, V, et, [Y], registry=reg
        )
        assert chain.steps

    def test_symmetric_part_vec(self, setup):
        reg, f, U, V, W, Y, om, et, mu = setup
        (ga,) = forms("γp", degree=1)
        P = poisson_structure()
        chain = prove_poisson_double_symmetric_part_vec(
            P, U, om, V, et, ga, registry=reg
        )
        assert chain.steps

    def test_right_leibniz_form(self, setup):
        reg, f, U, V, W, Y, om, et, mu = setup
        P = poisson_structure()
        chain = prove_poisson_double_right_leibniz_form(
            P, U, om, V, et, f, [Y], registry=reg
        )
        assert chain.steps

    def test_right_leibniz_vec(self, setup):
        """Needed the interior vector-slot linearity rule
        (ι_{f·V}(dω) = f·ι_V(dω)) — the 6.D residual that taught
        InteriorVectorLinearityDefinition."""
        reg, f, U, V, W, Y, om, et, mu = setup
        (ga,) = forms("γp", degree=1)
        P = poisson_structure()
        chain = prove_poisson_double_right_leibniz_vec(
            P, U, om, V, et, f, ga, registry=reg
        )
        assert chain.steps

    def test_anchor_is_sum_of_anchors(self, setup):
        from jacopy.algebra.derivation import Act
        from jacopy.packages.poisson.core import SharpVF
        from jacopy.packages.drinfeld.double import (
            poisson_double_anchor_action,
        )

        reg, f, U, V, W, Y, om, et, mu = setup
        P = poisson_structure()
        assert poisson_double_anchor_action(P, U, om, f) == Sum(
            Act(U, f), Act(SharpVF(P.pi, om), f)
        )
