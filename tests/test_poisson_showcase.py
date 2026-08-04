"""Poisson package — Phase 5.C: the cotangent-algebroid showcase.

(T*M, π♯, [·,·]_π) is a Lie algebroid — PROVED, not declared: the
Phase 3 axioms close mechanically for the concrete Koszul structure."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.poisson.core import poisson_structure
from jacopy.packages.poisson.koszul import koszul_bracket
from jacopy.packages.poisson.showcase import (
    _cyclic_jacobi_sum,
    prove_hamiltonian_morphism,
    prove_koszul_right_leibniz,
    prove_poisson_jacobi,
    showcase_engine,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, g, h = functions("f g h", registry=reg)
    (Y,) = vector_fields("Y")
    alpha, beta = forms("α β", degree=1)
    P = poisson_structure()
    return reg, f, g, h, Y, alpha, beta, P


class TestRightLeibniz:
    def test_closes_with_no_assumption(self, setup):
        """The algebroid right-Leibniz axiom is a THEOREM here —
        closes with no declaration at all."""
        reg, f, _, _, Y, alpha, beta, P = setup
        chain = prove_koszul_right_leibniz(
            P, alpha, beta, f, Y, registry=reg
        )
        assert chain.steps

    def test_wrong_anchor_term_fails(self, setup):
        """Dropping the (π♯α)(f)·β term must fail."""
        reg, f, _, _, Y, alpha, beta, P = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Pairing(
                    koszul_bracket(P, alpha, Product(f, beta)), Y
                ),
                Pairing(
                    Product(f, koszul_bracket(P, alpha, beta)), Y
                ),
                registry=reg,
                engine=showcase_engine(P, registry=reg),
            )


class TestPoissonJacobi:
    def test_closes_under_declaration(self, setup):
        reg, f, g, h, _, _, _, P = setup
        chain, thm = prove_poisson_jacobi(P, f, g, h, registry=reg)
        assert len(chain.steps) == 3
        assert "Poisson (" in thm.from_axioms[0]

    def test_two_legs_recorded(self, setup):
        reg, f, g, h, _, _, _, P = setup
        chain, _ = prove_poisson_jacobi(P, f, g, h, registry=reg)
        rules = [s.rule for s in chain.steps]
        assert "SN evaluation on exact forms" in rules
        assert "declared Poisson structure" in rules

    def test_honest_without_declaration(self, setup):
        """The cyclic sum is NOT provable without [π,π]_SN = 0."""
        reg, f, g, h, _, _, _, P = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                _cyclic_jacobi_sum(P, f, g, h),
                Integer(0),
                registry=reg,
                engine=showcase_engine(
                    P, registry=reg, declare_poisson=False
                ),
            )


class TestHamiltonianMorphism:
    def test_anchor_property_on_generators(self, setup):
        """X_{{f,g}} = [X_f, X_g] acting on h — the anchor-morphism
        axiom on exact generators, via the cited Jacobi instance."""
        reg, f, g, h, _, _, _, P = setup
        chain = prove_hamiltonian_morphism(P, f, g, h, registry=reg)
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_wrong_sign_fails(self, setup):
        reg, f, g, h, _, _, _, P = setup
        from jacopy.proof.theorems import TheoremBook, cite

        _, thm = prove_poisson_jacobi(P, f, g, h, registry=reg)
        book = TheoremBook()
        book.add(thm)
        engine = showcase_engine(P, registry=reg)
        cite(engine, book, thm.name)
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Act(P.hamiltonian(P.bracket(f, g)), h),
                Neg(
                    Act(
                        LieBracketVF(
                            P.hamiltonian(f), P.hamiltonian(g)
                        ),
                        h,
                    )
                ),
                registry=reg,
                engine=engine,
            )

    def test_arbitrary_names(self, setup):
        reg, *_ = setup
        f, g, h = functions("H K W", registry=reg)
        P = poisson_structure("Π")
        chain = prove_hamiltonian_morphism(P, f, g, h, registry=reg)
        assert chain.steps
