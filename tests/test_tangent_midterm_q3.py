"""TM case — the Midterm Question 3 usual-calculus identity suite
(Phase 2.G; PDF item 9h).

Q3's tilde-calculus half (d̃/L̃ from the Koszul bracket), the duality
conditions (K/K̃, derivator) and the triangular-bialgebroid Courant
bracket are Phase 5/6 deliverables — mapped in the ROADMAP.
"""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent import L, d, lie_bracket, tangent_engine
from jacopy.central.tangent.cartan import (
    prove_L_commutes_with_d_on_one_forms,
    prove_L_d_iota_commutation,
    prove_L_d_iota_exact,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    U, W, X, Y, Z = vector_fields("U W X Y Z")
    (f,) = functions("f", registry=reg)
    (omega,) = forms("ω", degree=1)
    (eta,) = forms("η", degree=1)
    return reg, U, W, X, Y, Z, f, omega, eta


class TestCartanZeroOneFormCompleteness:
    """Q3 first sentence: ALL Cartan relations for 0- and 1-forms.
    Most were closed in 2.D; this class pins the stragglers."""

    def test_L_d_commute_on_one_forms(self, setup):
        """[L_X, d] = 0 on 1-forms — the last missing relation."""
        reg, _, _, X, Y, Z, f, omega, _ = setup
        chain, used = prove_L_commutes_with_d_on_one_forms(
            omega, X, Y, Z, f, registry=reg
        )
        assert chain.steps
        assert len(used) >= 1  # needs a vector-level bracket identity

    def test_iota_anticommute_trivial_on_one_forms(self, setup):
        """ι_Xι_Yω + ι_Yι_Xω = 0 is trivial on 1-forms (both terms
        are ι of a scalar, hence 0)."""
        reg, _, _, X, Y, _, _, omega, _ = setup
        lhs = Sum(
            Act(Interior(X), Act(Interior(Y), omega)),
            Act(Interior(Y), Act(Interior(X), omega)),
        )
        chain = ExpandAndSimplify().prove(
            lhs, Integer(0), registry=reg, engine=tangent_engine(registry=reg)
        )
        assert chain.steps

    def test_L_iota_commutator_trivial_on_functions(self, setup):
        """[L_X, ι_Y] f = ι_[X,Y] f — every term vanishes on 0-forms."""
        reg, _, _, X, Y, _, f, _, _ = setup
        lhs = Sum(
            L(X, Act(Interior(Y), f)),
            Neg(Act(Interior(Y), L(X, f))),
        )
        rhs = Act(Interior(lie_bracket(X, Y)), f)
        chain = ExpandAndSimplify().prove(
            lhs, rhs, registry=reg, engine=tangent_engine(registry=reg)
        )
        assert chain.steps


class TestQ3CalculusIdentities:
    """The three displayed identities, usual-calculus half."""

    def test_L_L_commutator(self, setup):
        """L_U L_V μ − L_V L_U μ − L_[U,V] μ = 0 — already a 2.D
        theorem; re-pinned here as part of the Q3 suite."""
        from jacopy.central.tangent import prove_L_L_commutator_on_functions

        reg, U, W, _, _, _, f, _, _ = setup
        assert prove_L_L_commutator_on_functions(U, W, f, registry=reg).steps

    def test_L_d_iota_commutation(self, setup):
        """L_U dι_W η − dι_[U,W] η − dι_W L_U η = 0 (1-form η)."""
        reg, U, W, _, Y, _, f, _, eta = setup
        chain, used = prove_L_d_iota_commutation(
            eta, U, W, Y, f, registry=reg
        )
        assert chain.steps

    def test_L_d_iota_exact(self, setup):
        """L_W dι_V ω − dι_W dι_V ω = 0 (1-form ω) — Cartan magic on
        the exact form dι_V ω."""
        reg, _, W, _, Y, Z, f, omega, _ = setup
        chain, used = prove_L_d_iota_exact(
            omega, Z, W, Y, f, registry=reg
        )
        assert chain.steps

    def test_L_d_iota_commutation_wrong_sign_fails(self, setup):
        """Sanity: flipping the middle term's sign must not close."""
        reg, U, W, _, Y, _, f, _, eta = setup
        lhs = Sum(
            Pairing(L(U, d(Act(Interior(W), eta))), Y),
            Pairing(d(Act(Interior(lie_bracket(U, W)), eta)), Y),
            Neg(Pairing(d(Act(Interior(W), L(U, eta))), Y)),
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs, Integer(0), registry=reg,
                engine=tangent_engine(registry=reg),
            )


class TestHeadSumLinearity:
    """The structural rule Q3 exposed: sums in the head slot split."""

    def test_pairing_head_sum(self, setup):
        reg, _, _, X, _, _, _, omega, eta = setup
        node = Pairing(Sum(omega, eta), X)
        out, steps = tangent_engine(registry=reg).expand(node)
        assert steps
        assert out == Sum(Pairing(omega, X), Pairing(eta, X))
