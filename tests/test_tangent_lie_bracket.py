"""TM case — Lie bracket: canonical definition + proved properties
(Phase 2.A)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Integer, Product, Symbol
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import functions, vector_fields
from jacopy.central.tangent import (
    LieBracketActionDefinition,
    ScalarActAsMultiplicationDefinition,
    lie_bracket,
    tangent_engine,
    prove_antisymmetry,
    prove_jacobi,
    prove_leibniz_second_slot,
    prove_first_slot_function_linearity,
    prove_additivity_second_slot,
    prove_scalar_homogeneity,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    X, Y, Z = vector_fields("X Y Z")
    f, g = functions("f g", registry=reg)
    return reg, X, Y, Z, f, g


# --------------------------------------------------------------------- #
# The bracket node                                                       #
# --------------------------------------------------------------------- #


class TestLieBracketNode:
    def test_is_derivation_degree_zero(self, setup):
        """[X, Y] is again a vector field: a degree-0 derivation."""
        _, X, Y, *_ = setup
        br = lie_bracket(X, Y)
        assert isinstance(br, LieBracketVF)
        assert isinstance(br, Derivation)
        assert br.degree == Degree.const(0)

    def test_structural_equality(self, setup):
        _, X, Y, *_ = setup
        assert lie_bracket(X, Y) == lie_bracket(X, Y)
        assert lie_bracket(X, Y) != lie_bracket(Y, X)

    def test_can_act_on_functions(self, setup):
        reg, X, Y, _, f, _ = setup
        expr = lie_bracket(X, Y)(f)
        assert isinstance(expr, Act)
        assert degree_of(expr, reg) == Degree.const(0)

    def test_nests(self, setup):
        _, X, Y, Z, *_ = setup
        nested = lie_bracket(X, lie_bracket(Y, Z))
        assert isinstance(nested, LieBracketVF)
        assert isinstance(nested.Y, LieBracketVF)

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            lie_bracket("X", "Y")


# --------------------------------------------------------------------- #
# Definitional expansion                                                 #
# --------------------------------------------------------------------- #


class TestDefinitionalExpansion:
    def test_action_definition_fires(self, setup):
        """[X, Y](f) → X(Y(f)) − Y(X(f)) under the tangent engine."""
        reg, X, Y, _, f, _ = setup
        expr = Act(lie_bracket(X, Y), f)
        expanded, steps = tangent_engine(registry=reg).expand(expr)
        assert steps, "definitional rule should fire"
        target_pos = Act(X, Act(Y, f))
        assert any(target_pos == n for n in expanded.walk())

    def test_scalar_act_as_multiplication(self, setup):
        """f(x) → f·x for a registered degree-0 scalar f."""
        reg, X, _, _, f, g = setup
        rule = ScalarActAsMultiplicationDefinition(reg)
        expr = Act(f, Act(X, g))
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Product(f, Act(X, g))

    def test_scalar_rule_ignores_derivations(self, setup):
        reg, X, _, _, f, _ = setup
        rule = ScalarActAsMultiplicationDefinition(reg)
        assert not rule.matches(Act(X, f))

    def test_scalar_rule_ignores_unregistered(self):
        rule = ScalarActAsMultiplicationDefinition(None)
        assert not rule.matches(Act(Symbol("s"), Symbol("x")))


# --------------------------------------------------------------------- #
# Theorems (PDF 9c)                                                      #
# --------------------------------------------------------------------- #


class TestLieBracketProperties:
    def test_antisymmetry_closes(self, setup):
        reg, X, Y, _, f, _ = setup
        chain = prove_antisymmetry(X, Y, f, registry=reg)
        assert chain.steps

    def test_self_bracket_zero(self, setup):
        """[X, X](f) = 0 — degenerate case of antisymmetry."""
        reg, X, _, _, f, _ = setup
        chain = ExpandAndSimplify().prove(
            Act(lie_bracket(X, X), f),
            Integer(0),
            registry=reg,
            engine=tangent_engine(registry=reg),
        )
        assert chain.steps

    def test_jacobi_closes(self, setup):
        reg, X, Y, Z, f, _ = setup
        chain = prove_jacobi(X, Y, Z, f, registry=reg)
        assert chain.steps

    def test_leibniz_second_slot_closes(self, setup):
        reg, X, Y, _, f, g = setup
        chain = prove_leibniz_second_slot(X, f, Y, g, registry=reg)
        assert chain.steps

    def test_first_slot_function_linearity_closes(self, setup):
        reg, X, Y, _, f, g = setup
        chain = prove_first_slot_function_linearity(f, X, Y, g, registry=reg)
        assert chain.steps

    def test_wrong_identity_fails(self, setup):
        """Sanity: the prover must NOT close a false identity."""
        reg, X, Y, _, f, _ = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Act(lie_bracket(X, Y), f),
                Act(lie_bracket(Y, X), f),  # missing the minus sign
                registry=reg,
                engine=tangent_engine(registry=reg),
            )

    def test_additivity_second_slot_closes(self, setup):
        """[X, Y + Z] = [X, Y] + [X, Z] (R-linearity, additive part)."""
        reg, X, Y, Z, f, _ = setup
        assert prove_additivity_second_slot(X, Y, Z, f, registry=reg).steps

    def test_scalar_homogeneity_closes(self, setup):
        """[c·X, Y] = c·[X, Y] for numeric constants (both signs)."""
        reg, X, Y, _, f, _ = setup
        assert prove_scalar_homogeneity(2, X, Y, f, registry=reg).steps
        assert prove_scalar_homogeneity(-3, X, Y, f, registry=reg).steps

    def test_derivation_kills_constants(self, setup):
        """D(c) = 0 — the product_rule fact behind homogeneity."""
        from jacopy.algorithms.product_rule import product_rule
        from jacopy.core.expr import Integer as _Int
        reg, X, *_ = setup
        assert product_rule(Act(X, _Int(5)), reg) == _Int(0)

    def test_jacobi_all_orderings(self, setup):
        """Jacobi closes for every cyclic labelling of the three fields."""
        reg, X, Y, Z, f, _ = setup
        for a, b, c in [(X, Y, Z), (Y, Z, X), (Z, X, Y)]:
            assert prove_jacobi(a, b, c, f, registry=reg).steps


class TestNonCircularity:
    """Audit regression (2026-07-30): theorem-backed rules must not
    use themselves in their foundational derivations."""

    def test_orientation_derivation_avoids_itself(self):
        """The orientation rule is backed by the antisymmetry theorem;
        its derivation must come from the commutator definition alone
        (the original builder used the full engine and the rule fired
        on [Y,X] inside its own proof)."""
        from jacopy.core.registry import PropertyRegistry
        from jacopy.algebra.lie_bracket_vf import LieBracketVF
        from jacopy.central.objects import vector_fields
        from jacopy.central.tangent.engine import tangent_engine

        reg = PropertyRegistry()
        X, Y = vector_fields("X Y")
        eng = tangent_engine(registry=reg, mode="foundational")
        _, steps = eng.expand(LieBracketVF(Y, X))
        assert steps and "orientation" in steps[0].rule
        assert steps[0].children  # derivation attached

        def walk(s):
            for c in s.children:
                yield c
                yield from walk(c)

        assert not any(
            "orientation" in sub.rule for sub in walk(steps[0])
        )


class TestScalarCoefficientComposition:
    """A scalar factor in operator position acts by MULTIPLICATION,
    never by Leibniz: (g·X)(f·h) = g·X(f)·h + g·f·X(h). The old
    composition unfold Leibniz-split the scalar layer too —
    g(ab) → g(a)b + a·g(b) = 2g·ab — doubling every term (caught by
    the 5.D.2 two-scalar Koszul gap)."""

    def test_scalar_times_derivation_leibniz(self):
        from jacopy.algorithms.product_rule import product_rule
        from jacopy.algorithms.simplify import simplify
        from jacopy.core.expr import Sum

        reg = PropertyRegistry()
        f, g, h = functions("f g h", registry=reg)
        (X,) = vector_fields("X")
        out = simplify(
            product_rule(
                Act(Product(g, X), Product(f, h)), reg
            ),
            reg,
        )
        expected = simplify(
            Sum(
                Product(g, Act(X, f), h),
                Product(g, f, Act(X, h)),
            ),
            reg,
        )
        assert out == expected

    def test_scalar_op_multiplies(self):
        """Act(g, x) with g a plain scalar is g·x (no split)."""
        from jacopy.algorithms.product_rule import product_rule
        from jacopy.algorithms.canonicalize import canonicalize

        reg = PropertyRegistry()
        f, g, h = functions("f g h", registry=reg)
        out = product_rule(Act(g, Product(f, h)), reg)
        assert canonicalize(out) == canonicalize(Product(g, f, h))
