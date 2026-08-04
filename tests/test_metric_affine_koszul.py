"""Metric-affine package — Phase 4.C.1: the metric layer, the
non-metricity tensor, and the Koszul formula family (generalized =
Schouten identity form / classical / Levi-Civita uniqueness)."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import connection, functions, vector_fields
from jacopy.packages.metric_affine import (
    metric,
    metric_affine_engine,
    nonmetricity,
    prove_generalized_koszul,
    prove_koszul_formula,
    prove_levi_civita_unique,
    torsion,
)
from jacopy.packages.metric_affine.koszul import _koszul_rhs


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    X, Y, Z = vector_fields("X Y Z")
    nabla = connection()
    g = metric()
    return reg, f, X, Y, Z, nabla, g


class TestMetricLayer:
    def test_symmetry_and_bilinearity(self, setup):
        reg, f, X, Y, _, _, g = setup
        eng = metric_affine_engine(registry=reg)
        out, _ = eng.expand(g(Y, X))
        assert out == g(X, Y)
        out, _ = eng.expand(g(Product(f, X), Y))
        assert out == Product(f, g(X, Y))
        out, _ = eng.expand(g(X, Integer(0)))
        assert out == Integer(0)

    def test_metric_value_is_scalar(self, setup):
        from jacopy.central.calculus.scalars import is_scalar_function

        reg, _, X, Y, _, _, g = setup
        assert is_scalar_function(g(X, Y), reg)

    def test_distinct_metrics(self, setup):
        _, _, X, Y, _, _, g = setup
        h = metric("h")
        assert g(X, Y) != h(X, Y)


class TestNonMetricity:
    def test_expansion(self, setup):
        reg, _, X, Y, Z, nabla, g = setup
        eng = metric_affine_engine(registry=reg)
        out, steps = eng.expand(nonmetricity(nabla, g, X, Y, Z))
        assert any("non-metricity definition" in s.rule for s in steps)

    def test_tensorial_all_slots(self, setup):
        reg, f, X, Y, Z, nabla, g = setup
        target = Product(f, nonmetricity(nabla, g, X, Y, Z))
        for lhs in (
            nonmetricity(nabla, g, Product(f, X), Y, Z),
            nonmetricity(nabla, g, X, Product(f, Y), Z),
            nonmetricity(nabla, g, X, Y, Product(f, Z)),
        ):
            chain = ExpandAndSimplify().prove(
                lhs,
                target,
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )
            assert chain.steps

    def test_symmetric_in_last_two_slots(self, setup):
        reg, _, X, Y, Z, nabla, g = setup
        chain = ExpandAndSimplify().prove(
            nonmetricity(nabla, g, X, Y, Z),
            nonmetricity(nabla, g, X, Z, Y),
            registry=reg,
            engine=metric_affine_engine(registry=reg),
        )
        assert chain.steps

    def test_vanishes_under_compatibility(self, setup):
        reg, _, X, Y, Z, nabla, g = setup
        eng = metric_affine_engine(
            registry=reg, compatible=((nabla, g),)
        )
        chain = ExpandAndSimplify().prove(
            nonmetricity(nabla, g, X, Y, Z),
            Integer(0),
            registry=reg,
            engine=eng,
        )
        assert chain.steps

    def test_open_without_compatibility(self, setup):
        reg, _, X, Y, Z, nabla, g = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                nonmetricity(nabla, g, X, Y, Z),
                Integer(0),
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )


class TestTorsionFree:
    def test_torsion_vanishes(self, setup):
        reg, _, X, Y, _, nabla, _ = setup
        eng = metric_affine_engine(registry=reg, torsion_free=(nabla,))
        chain = ExpandAndSimplify().prove(
            torsion(nabla, X, Y),
            Integer(0),
            registry=reg,
            engine=eng,
        )
        assert chain.steps

    def test_scoped_to_connection(self, setup):
        """Only the declared connection's torsion vanishes."""
        reg, _, X, Y, _, nabla, _ = setup
        other = connection("D")
        eng = metric_affine_engine(registry=reg, torsion_free=(nabla,))
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                torsion(other, X, Y),
                Integer(0),
                registry=reg,
                engine=eng,
            )


class TestKoszulFamily:
    def test_generalized_koszul_is_definitional(self, setup):
        """The Schouten decomposition in identity form — closes with
        NO declarations."""
        reg, _, X, Y, Z, nabla, g = setup
        chain = prove_generalized_koszul(nabla, g, X, Y, Z, registry=reg)
        assert chain.steps

    def test_classical_koszul_under_declarations(self, setup):
        reg, _, X, Y, Z, nabla, g = setup
        chain = prove_koszul_formula(nabla, g, X, Y, Z, registry=reg)
        assert chain.steps

    def test_classical_koszul_honest_without_declarations(self, setup):
        reg, _, X, Y, Z, nabla, g = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Product(Integer(2), g(nabla(X, Y), Z)),
                _koszul_rhs(nabla, g, X, Y, Z, with_defects=False),
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )

    def test_wrong_sign_generalized_koszul_fails(self, setup):
        """Flipping the sign of the Q-block breaks the identity."""
        reg, _, X, Y, Z, nabla, g = setup
        wrong = Sum(
            _koszul_rhs(nabla, g, X, Y, Z, with_defects=False),
            nonmetricity(nabla, g, X, Y, Z),
            nonmetricity(nabla, g, Y, X, Z),
            Neg(nonmetricity(nabla, g, Z, X, Y)),
            g(torsion(nabla, X, Y), Z),
            Neg(g(torsion(nabla, X, Z), Y)),
            Neg(g(torsion(nabla, Y, Z), X)),
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Product(Integer(2), g(nabla(X, Y), Z)),
                wrong,
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )

    def test_levi_civita_unique(self, setup):
        reg, _, X, Y, Z, nabla, g = setup
        chain, thm = prove_levi_civita_unique(
            nabla, connection("D"), g, X, Y, Z, registry=reg
        )
        assert len(chain.steps) == 4
        assert thm.lhs == nabla(X, Y)
        assert "non-degeneracy" in " ".join(thm.from_axioms)
        rules = [s.rule for s in chain.steps]
        assert any("agreement on generic section" in r for r in rules)

    def test_levi_civita_unique_needs_two_contexts(self, setup):
        reg, _, X, Y, Z, nabla, g = setup
        with pytest.raises(ValueError):
            prove_levi_civita_unique(
                nabla, nabla, g, X, Y, Z, registry=reg
            )
