"""Metric-affine package — Phase 4.C.2: the torsionful Bianchi
identities and the Ricci identity on functions.

The only non-definitional input is the section-level Lie Jacobi
identity, entering as cited instance theorems DERIVED from the
commutator definition (agreement on generators)."""

import pytest

from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import connection, functions, vector_fields
from jacopy.packages.metric_affine import (
    covariant_torsion_derivative,
    curvature,
    metric_affine_engine,
    prove_bianchi_first,
    prove_bianchi_second,
    prove_ricci_identity_on_functions,
    torsion,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    X, Y, Z, W = vector_fields("X Y Z W")
    nabla = connection()
    return reg, f, X, Y, Z, W, nabla


class TestRicciIdentity:
    def test_curvature_kills_scalars(self, setup):
        reg, f, X, Y, _, _, nabla = setup
        chain = prove_ricci_identity_on_functions(
            nabla, X, Y, f, registry=reg
        )
        assert chain.steps

    def test_wrong_sign_fails(self, setup):
        reg, f, X, Y, _, _, nabla = setup
        from jacopy.central.tangent.lie_bracket import lie_bracket

        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Sum(
                    nabla(X, nabla(Y, f)),
                    Neg(nabla(Y, nabla(X, f))),
                    nabla(lie_bracket(X, Y), f),
                ),
                Integer(0),
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )


class TestBianchiFirst:
    def test_closes_with_cited_jacobi(self, setup):
        reg, f, X, Y, Z, _, nabla = setup
        chain, used = prove_bianchi_first(nabla, X, Y, Z, f, registry=reg)
        assert chain.steps and used
        assert any(s.provenance_tag == "theorem" for s in chain.steps)
        assert all(
            "Lie bracket definition" in " ".join(t.from_axioms)
            for t in used
        )

    def test_honest_without_jacobi(self, setup):
        """The raw statement stalls on exactly the Jacobi combination
        without the citations."""
        reg, f, X, Y, Z, _, nabla = setup
        cyc = [(X, Y, Z), (Y, Z, X), (Z, X, Y)]
        lhs = Sum(*(curvature(nabla, a, b, c) for a, b, c in cyc))
        rhs = Sum(
            *(
                Sum(
                    torsion(nabla, torsion(nabla, a, b), c),
                    covariant_torsion_derivative(nabla, a, b, c),
                )
                for a, b, c in cyc
            )
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                rhs,
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )

    def test_false_variant_fails(self, setup):
        """Dropping the (∇T) block breaks the identity."""
        reg, f, X, Y, Z, _, nabla = setup
        cyc = [(X, Y, Z), (Y, Z, X), (Z, X, Y)]
        lhs = Sum(*(curvature(nabla, a, b, c) for a, b, c in cyc))
        rhs = Sum(
            *(
                torsion(nabla, torsion(nabla, a, b), c)
                for a, b, c in cyc
            )
        )
        from jacopy.packages.metric_affine.bianchi import _jacobi_engine

        engine, _ = _jacobi_engine([(X, Y, Z)], f, reg)
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs, rhs, registry=reg, engine=engine
            )

    def test_arbitrary_names(self, setup):
        reg, f, *_ = setup
        A, B, C = vector_fields("ξ η ζ")
        nabla = connection("∇̃")
        chain, _ = prove_bianchi_first(nabla, A, B, C, f, registry=reg)
        assert chain.steps


class TestBianchiSecond:
    def test_closes_with_cited_instances(self, setup):
        reg, f, X, Y, Z, W, nabla = setup
        chain, used = prove_bianchi_second(
            nabla, X, Y, Z, W, f, registry=reg
        )
        assert chain.steps
        assert any(s.provenance_tag == "theorem" for s in chain.steps)
        # both instance families cited: section-level Jacobi and the
        # in-direction variant
        names = " ".join(t.name for t in used)
        assert "lie_jacobi" in names and "covariant_jacobi" in names
