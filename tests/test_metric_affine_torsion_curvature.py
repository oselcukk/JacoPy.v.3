"""Metric-affine package — Phase 4.B: torsion, curvature, the
difference tensor, and their classical theorems (tensoriality,
antisymmetry) — all PROVED from the canonical definitions."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import connection, functions, vector_fields
from jacopy.central.tangent.engine import tangent_engine
from jacopy.packages.metric_affine import (
    metric_affine_engine,
    curvature,
    prove_connection_difference_tensorial,
    prove_curvature_antisymmetry,
    prove_curvature_tensorial,
    prove_torsion_antisymmetry,
    prove_torsion_tensorial,
    torsion,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    X, Y, Z = vector_fields("X Y Z")
    nabla = connection()
    return reg, f, X, Y, Z, nabla


class TestDefinitions:
    def test_torsion_expansion(self, setup):
        reg, _, X, Y, _, nabla = setup
        eng = metric_affine_engine(registry=reg)
        out, steps = eng.expand(torsion(nabla, X, Y))
        assert out == Sum(
            nabla(X, Y), Neg(nabla(Y, X)), Neg(LieBracketVF(X, Y))
        )
        assert any("torsion definition" in s.rule for s in steps)

    def test_curvature_expansion(self, setup):
        reg, _, X, Y, Z, nabla = setup
        eng = metric_affine_engine(registry=reg)
        out, steps = eng.expand(curvature(nabla, X, Y, Z))
        assert out == Sum(
            nabla(X, nabla(Y, Z)),
            Neg(nabla(Y, nabla(X, Z))),
            Neg(nabla(LieBracketVF(X, Y), Z)),
        )
        assert any("curvature definition" in s.rule for s in steps)

    def test_nodes_carry_connection_identity(self, setup):
        _, _, X, Y, Z, nabla = setup
        other = connection("D")
        assert torsion(nabla, X, Y) != torsion(other, X, Y)
        assert curvature(nabla, X, Y, Z) != curvature(other, X, Y, Z)


class TestTensoriality:
    def test_torsion_both_slots(self, setup):
        reg, f, X, Y, _, nabla = setup
        first, second = prove_torsion_tensorial(
            nabla, X, Y, f, registry=reg
        )
        assert first.steps and second.steps

    def test_curvature_all_three_slots(self, setup):
        """The third slot is the classical double-Leibniz argument:
        X(Y(f)) terms cancel against [X,Y](f)."""
        reg, f, X, Y, Z, nabla = setup
        r1, r2, r3 = prove_curvature_tensorial(
            nabla, X, Y, Z, f, registry=reg
        )
        assert r1.steps and r2.steps and r3.steps

    def test_torsion_antisymmetry(self, setup):
        reg, _, X, Y, _, nabla = setup
        assert prove_torsion_antisymmetry(nabla, X, Y, registry=reg).steps

    def test_curvature_antisymmetry(self, setup):
        reg, _, X, Y, Z, nabla = setup
        assert prove_curvature_antisymmetry(
            nabla, X, Y, Z, registry=reg
        ).steps

    def test_difference_tensor_bilinear(self, setup):
        reg, f, X, Y, _, nabla = setup
        d1, d2 = prove_connection_difference_tensorial(
            nabla, connection("D"), X, Y, f, registry=reg
        )
        assert d1.steps and d2.steps

    def test_arbitrary_names(self, setup):
        reg, f, *_ = setup
        A, B = vector_fields("ξ ζ")
        nabla = connection("∇̂")
        first, second = prove_torsion_tensorial(
            nabla, A, B, f, registry=reg
        )
        assert first.steps and second.steps


class TestBracketModuleStructure:
    def test_rule_is_theorem_classified(self, setup):
        """[fX, Y] rewrites with provenance 'theorem'; foundational
        mode attaches the generic-function derivation, which must not
        use the rule itself."""
        reg, f, X, Y, *_ = setup
        eng = metric_affine_engine(registry=reg, mode="foundational")
        out, steps = eng.expand(LieBracketVF(Product(f, X), Y))
        assert out == Sum(
            Product(f, LieBracketVF(X, Y)),
            Neg(Product(Act(Y, f), X)),
        )
        step = steps[0]
        assert step.provenance_tag == "theorem"

        def walk(s):
            for c in s.children:
                yield c
                yield from walk(c)

        assert step.children
        assert not any(
            "module structure" in sub.rule for sub in walk(step)
        )

    def test_tangent_engine_does_not_have_it(self, setup):
        """Phase 2 isolation: the tangent engine leaves [fX, Y] inert
        at section level."""
        reg, f, X, Y, *_ = setup
        eng = tangent_engine(registry=reg)
        node = LieBracketVF(Product(f, X), Y)
        out, steps = eng.expand(node)
        assert out == node and not steps


class TestHonesty:
    def test_false_tensoriality_fails(self, setup):
        reg, f, X, Y, _, nabla = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                torsion(nabla, Product(f, X), Y),
                Product(f, f, torsion(nabla, X, Y)),
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )

    def test_distinct_connections_do_not_mix(self, setup):
        reg, _, X, Y, _, nabla = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                torsion(nabla, X, Y),
                torsion(connection("D"), X, Y),
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )

    def test_wrong_antisymmetry_sign_fails(self, setup):
        reg, _, X, Y, _, nabla = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                torsion(nabla, X, Y),
                torsion(nabla, Y, X),
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )
