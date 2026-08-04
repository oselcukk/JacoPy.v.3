"""Algebroid case — the declaration system: hierarchy axioms as
opt-in, instance-scoped rules (Phase 3.B)."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import Bundle, functions
from jacopy.central.algebroid import (
    DECLARATIONS,
    LEVELS,
    Algebroid,
    algebroid,
    algebroid_engine,
    jacobiator,
    locality_term,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    return reg, f


# --------------------------------------------------------------------- #
# Declaration bookkeeping                                                #
# --------------------------------------------------------------------- #


class TestDeclarationBookkeeping:
    def test_levels_expand(self, setup):
        E = algebroid("E", declare=("leibniz",))
        assert E.declarations == frozenset({"right-leibniz", "jacobi"})

    def test_leibniz_level_does_not_grant_morphism(self, setup):
        """The 3.D theorem's honesty: 'leibniz' does NOT silently
        include anchor-morphism — it is provable, hence never assumed."""
        E = algebroid("E", declare=("leibniz",))
        assert not E.declares("anchor-morphism")

    def test_atomic_declarations(self, setup):
        E = algebroid("E", declare=("right-leibniz", "anchor-morphism"))
        assert E.declares("right-leibniz")
        assert not E.declares("jacobi")

    def test_unknown_declaration_rejected(self, setup):
        with pytest.raises(ValueError):
            algebroid("E", declare=("courant-ish",))

    def test_with_declarations_extends(self, setup):
        E = algebroid("E", declare=("almost-leibniz",))
        E2 = E.with_declarations("jacobi")
        assert E2.declarations == frozenset({"right-leibniz", "jacobi"})
        assert E != E2  # declarations are part of the context identity

    def test_all_levels_are_valid(self, setup):
        for level in LEVELS:
            assert algebroid("E", declare=(level,)).declarations


# --------------------------------------------------------------------- #
# The rules, one per axiom                                               #
# --------------------------------------------------------------------- #


class TestDeclaredRules:
    def test_right_leibniz(self, setup):
        reg, f = setup
        E = algebroid("E", declare=("right-leibniz",))
        u, v = E.sections("u v")
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(E.bracket(u, Product(f, v)))
        assert out == Sum(
            Product(Act(E.anchor(u), f), v),
            Product(f, E.bracket(u, v)),
        )
        assert "right-Leibniz (E)" in steps[0].rule

    def test_right_leibniz_leaves_first_slot(self, setup):
        """[fu, v] needs LEFT-Leibniz; right alone must not touch it."""
        reg, f = setup
        E = algebroid("E", declare=("right-leibniz",))
        u, v = E.sections("u v")
        eng = algebroid_engine(E, registry=reg)
        node = E.bracket(Product(f, u), v)
        out, steps = eng.expand(node)
        assert out == node and not steps

    def test_left_leibniz_with_locality_term(self, setup):
        reg, f = setup
        E = algebroid("E", declare=("local",))
        u, v = E.sections("u v")
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(E.bracket(Product(f, u), v))
        assert out == Sum(
            Neg(Product(Act(E.anchor(v), f), u)),
            Product(f, E.bracket(u, v)),
            locality_term(E, f, u, v),
        )

    def test_anchor_morphism(self, setup):
        reg, _ = setup
        E = algebroid("E", declare=("pre-leibniz",))
        u, v = E.sections("u v")
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(E.anchor(E.bracket(u, v)))
        assert out == LieBracketVF(E.anchor(u), E.anchor(v))

    def test_anchor_morphism_inert_without_declaration(self, setup):
        reg, _ = setup
        E = algebroid("E", declare=("right-leibniz",))
        u, v = E.sections("u v")
        eng = algebroid_engine(E, registry=reg)
        node = E.anchor(E.bracket(u, v))
        out, steps = eng.expand(node)
        assert out == node and not steps

    def test_jacobi_via_jacobiator_node(self, setup):
        reg, _ = setup
        E = algebroid("E", declare=("leibniz",))
        u, v, w = E.sections("u v w")
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(jacobiator(E, u, v, w))
        assert out == Integer(0)
        assert "Leibniz-Jacobi (E)" in steps[0].rule

    def test_jacobiator_expands_definitionally_without_jacobi(self, setup):
        reg, _ = setup
        E = algebroid("E")
        u, v, w = E.sections("u v w")
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(jacobiator(E, u, v, w))
        assert steps  # the definitional expansion fired
        expected = Sum(
            E.bracket(u, E.bracket(v, w)),
            Neg(E.bracket(E.bracket(u, v), w)),
            Neg(E.bracket(v, E.bracket(u, w))),
        )
        assert out == expected


# --------------------------------------------------------------------- #
# Scoping + assumption honesty                                           #
# --------------------------------------------------------------------- #


class TestScopingAndHonesty:
    def test_declaration_scoped_to_its_algebroid(self, setup):
        """E's right-Leibniz must not fire on F's bracket."""
        reg, f = setup
        E = algebroid("E", declare=("right-leibniz",))
        F = algebroid("F")
        u, v = F.sections("u v")
        eng = algebroid_engine(E, registry=reg)
        node = F.bracket(u, Product(f, v))
        out, steps = eng.expand(node)
        assert out == node and not steps

    def test_proof_closes_with_assumption_fails_without(self, setup):
        """The same statement: provable under the declaration, honest
        ProofFailure without it."""
        reg, f = setup
        stmt = lambda alg, u, v: (
            alg.bracket(u, Product(f, v)),
            Sum(
                Product(Act(alg.anchor(u), f), v),
                Product(f, alg.bracket(u, v)),
            ),
        )
        E_yes = algebroid("E", declare=("right-leibniz",))
        u, v = E_yes.sections("u v")
        lhs, rhs = stmt(E_yes, u, v)
        chain = ExpandAndSimplify().prove(
            lhs, rhs, registry=reg,
            engine=algebroid_engine(E_yes, registry=reg),
        )
        assert chain.steps

        E_no = algebroid("E")
        lhs, rhs = stmt(E_no, u, v)
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs, rhs, registry=reg,
                engine=algebroid_engine(E_no, registry=reg),
            )

    def test_proof_records_assumptions(self, setup):
        """A closed proof's steps name the axioms used — the exact
        assumption record."""
        reg, f = setup
        E = algebroid("E", declare=("right-leibniz",))
        u, v = E.sections("u v")
        chain = ExpandAndSimplify().prove(
            E.bracket(u, Product(f, v)),
            Sum(
                Product(Act(E.anchor(u), f), v),
                Product(f, E.bracket(u, v)),
            ),
            registry=reg,
            engine=algebroid_engine(E, registry=reg),
        )
        assert any("right-Leibniz (E)" in s.rule for s in chain.steps)
