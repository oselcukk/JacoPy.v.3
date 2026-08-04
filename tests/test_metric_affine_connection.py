"""Metric-affine package — Phase 4.A: connection structure rules.

Direction-slot C∞-linearity is the package's own definitional rule;
the argument-side Leibniz is INHERITED (product_rule + the Phase 1/2
rule ∇_X f = X(f)) and only verified here."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import connection, functions, vector_fields
from jacopy.packages.metric_affine import metric_affine_engine


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, g = functions("f g", registry=reg)
    X, Y, Z = vector_fields("X Y Z")
    nabla = connection()
    return reg, f, g, X, Y, Z, nabla


class TestDirectionLinearity:
    def test_scalar_and_sum(self, setup):
        reg, f, _, X, Y, Z, nabla = setup
        eng = metric_affine_engine(registry=reg)
        out, steps = eng.expand(nabla(Sum(Product(f, X), Y), Z))
        assert out == Sum(Product(f, nabla(X, Z)), nabla(Y, Z))
        assert any("direction linearity" in s.rule for s in steps)

    def test_neg_and_zero(self, setup):
        reg, _, _, X, _, Z, nabla = setup
        eng = metric_affine_engine(registry=reg)
        out, _ = eng.expand(nabla(Neg(X), Z))
        assert out == Neg(nabla(X, Z))
        out, _ = eng.expand(nabla(Integer(0), Z))
        assert out == Integer(0)

    def test_plain_direction_inert(self, setup):
        reg, _, _, X, _, Z, nabla = setup
        eng = metric_affine_engine(registry=reg)
        node = nabla(X, Z)
        out, steps = eng.expand(node)
        assert out == node and not steps

    def test_composite_scalar_coefficient(self, setup):
        """∇_{fg·X} Z = f·g·∇_X Z (nested scalar product)."""
        reg, f, g, X, _, Z, nabla = setup
        chain = ExpandAndSimplify().prove(
            nabla(Product(f, Product(g, X)), Z),
            Product(f, g, nabla(X, Z)),
            registry=reg,
            engine=metric_affine_engine(registry=reg),
        )
        assert chain.steps


class TestInheritedArgumentLeibniz:
    def test_leibniz_via_product_rule(self, setup):
        """∇_X(fY) = X(f)·Y + f·∇_X Y — no package rule involved:
        product_rule + ∇_X f = X(f)."""
        reg, f, _, X, Y, _, nabla = setup
        chain = ExpandAndSimplify().prove(
            nabla(X, Product(f, Y)),
            Sum(Product(Act(X, f), Y), Product(f, nabla(X, Y))),
            registry=reg,
            engine=metric_affine_engine(registry=reg),
        )
        assert chain.steps

    def test_additivity(self, setup):
        reg, _, _, X, Y, Z, nabla = setup
        chain = ExpandAndSimplify().prove(
            nabla(X, Sum(Y, Z)),
            Sum(nabla(X, Y), nabla(X, Z)),
            registry=reg,
            engine=metric_affine_engine(registry=reg),
        )
        assert chain.steps

    def test_scalar_action(self, setup):
        reg, f, _, X, *_ = setup
        nabla = connection()
        chain = ExpandAndSimplify().prove(
            nabla(X, f),
            Act(X, f),
            registry=reg,
            engine=metric_affine_engine(registry=reg),
        )
        assert chain.steps


class TestHonesty:
    def test_distinct_connections_do_not_mix(self, setup):
        reg, _, _, X, Y, _, nabla = setup
        other = connection("D")
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                nabla(X, Y),
                other(X, Y),
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )

    def test_direction_argument_asymmetry(self, setup):
        """∇_{fX} Z ≠ ∇_X(fZ) — the two slots behave differently and
        the engine must NOT conflate them."""
        reg, f, _, X, _, Z, nabla = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                nabla(Product(f, X), Z),
                nabla(X, Product(f, Z)),
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )

    def test_arbitrary_names(self, setup):
        reg, f, *_ = setup
        A, B = vector_fields("Kedi Fare")
        nabla = connection("∇̃")
        chain = ExpandAndSimplify().prove(
            nabla(Product(f, A), B),
            Product(f, nabla(A, B)),
            registry=reg,
            engine=metric_affine_engine(registry=reg),
        )
        assert chain.steps
