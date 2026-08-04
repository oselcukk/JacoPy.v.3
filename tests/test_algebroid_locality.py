"""Algebroid case — Phase 3.E.1: the locality operator's structure
(C∞-multilinearity [MC Def 3.5]) and the coboundary's derived
linearity + Leibniz (theorems from the defining pairing)."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.objects import Bundle, functions
from jacopy.central.algebroid import (
    LocalityOperator,
    algebroid,
    algebroid_engine,
    locality_term,
    tangent_algebroid,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, g = functions("f g", registry=reg)
    E = algebroid("E", Bundle("E"), declare=("local",))
    u, v, w = E.sections("u v w")
    return reg, f, g, E, u, v, w


# --------------------------------------------------------------------- #
# The generalized locality node                                          #
# --------------------------------------------------------------------- #


class TestLocalityNode:
    def test_locality_term_builds_coboundary_slot(self, setup):
        _, f, _, E, u, v, _ = setup
        node = locality_term(E, f, u, v)
        assert isinstance(node, LocalityOperator)
        assert node.form == E.D(f)
        assert node._repr_inner() == "L(Df,u,v)"

    def test_general_form_slot(self, setup):
        """The first slot takes ANY E-1-form, not just Df."""
        _, f, g, E, u, v, _ = setup
        node = LocalityOperator("E", Sum(E.D(f), E.D(g)), u, v)
        assert node.rewritable_slots[0] == Sum(E.D(f), E.D(g))

    def test_tangent_algebroid_has_no_locality(self, setup):
        _, f, *_ = setup
        TMalg = tangent_algebroid()
        X, Y = TMalg.sections("X Y")
        with pytest.raises(ValueError):
            locality_term(TMalg, f, X, Y)

    def test_left_leibniz_emits_generalized_node(self, setup):
        reg, f, _, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(E.bracket(Product(f, u), v))
        assert out == Sum(
            Neg(Product(Act(E.anchor(v), f), u)),
            Product(f, E.bracket(u, v)),
            locality_term(E, f, u, v),
        )


# --------------------------------------------------------------------- #
# C∞-multilinearity (definitional, MC Def 3.5)                           #
# --------------------------------------------------------------------- #


class TestLocalityMultilinearity:
    def test_form_slot_scalar_and_sum(self, setup):
        reg, f, g, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(
            LocalityOperator("E", Product(f, Sum(E.D(f), E.D(g))), u, v)
        )
        assert out == Product(
            f,
            Sum(
                LocalityOperator("E", E.D(f), u, v),
                LocalityOperator("E", E.D(g), u, v),
            ),
        )

    def test_section_slots(self, setup):
        reg, f, _, E, u, v, w = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(
            LocalityOperator("E", E.D(f), Sum(u, w), Product(f, v))
        )
        assert out == Sum(
            Product(f, LocalityOperator("E", E.D(f), u, v)),
            Product(f, LocalityOperator("E", E.D(f), w, v)),
        )

    def test_neg_and_zero(self, setup):
        reg, f, _, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(LocalityOperator("E", E.D(f), Neg(u), v))
        assert out == Neg(LocalityOperator("E", E.D(f), u, v))
        out, _ = eng.expand(LocalityOperator("E", E.D(f), u, Integer(0)))
        assert out == Integer(0)

    def test_opaque_slots_stay_inert(self, setup):
        reg, f, _, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        node = locality_term(E, f, u, v)
        out, steps = eng.expand(node)
        assert out == node and not steps


# --------------------------------------------------------------------- #
# Coboundary linearity + Leibniz (theorems from the pairing)             #
# --------------------------------------------------------------------- #


class TestCoboundaryLinearity:
    def test_leibniz(self, setup):
        reg, f, g, E, *_ = setup
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(E.D(Product(f, g)))
        assert out == Sum(Product(f, E.D(g)), Product(g, E.D(f)))
        assert steps[0].provenance_tag == "theorem"

    def test_additivity_neg_constants(self, setup):
        reg, f, g, E, *_ = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(E.D(Sum(f, Neg(g))))
        assert out == Sum(E.D(f), Neg(E.D(g)))
        out, _ = eng.expand(E.D(Product(Integer(3), f)))
        assert out == Product(Integer(3), E.D(f))
        out, _ = eng.expand(E.D(Integer(5)))
        assert out == Integer(0)

    def test_foundational_subproof_is_noncircular(self, setup):
        """The attached derivation must use only the pairing definition
        — never the linearity rule being derived."""
        reg, f, g, E, *_ = setup
        eng = algebroid_engine(E, registry=reg, mode="foundational")
        _, steps = eng.expand(E.D(Product(f, g)))
        agreement = steps[0].children[0]
        assert agreement.rule == "agreement on generators (pairing section)"
        assert agreement.children  # the mechanical pairing leg

        def walk(s):
            for c in s.children:
                yield c
                yield from walk(c)

        # the DERIVATION (everything below the root step) must not use
        # the rule it derives
        assert not any(
            "coboundary linearity" in sub.rule for sub in walk(steps[0])
        )

    def test_pairing_consistency(self, setup):
        """⟨D(fg), u⟩ must equal ρ(u)(f·g) whichever rule fires first."""
        reg, f, g, E, u, *_ = setup
        chain = ExpandAndSimplify().prove(
            Pairing(E.D(Product(f, g)), u),
            Act(E.anchor(u), Product(f, g)),
            registry=reg,
            engine=algebroid_engine(E, registry=reg),
        )
        assert chain.steps

    def test_scoped_to_algebroid(self, setup):
        reg, f, g, E, *_ = setup
        F = algebroid("F", Bundle("F"))
        eng = algebroid_engine(E, registry=reg)
        node = F.D(Product(f, g))
        out, steps = eng.expand(node)
        assert out == node and not steps


# --------------------------------------------------------------------- #
# Composite: everything together under "local"                           #
# --------------------------------------------------------------------- #


class TestComposite:
    def test_bracket_with_product_coefficient(self, setup):
        """[f·g·u, v] — left-Leibniz with a product coefficient; the
        locality slot then splits by D-Leibniz + multilinearity."""
        reg, f, g, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        lhs = E.bracket(Product(f, Product(g, u)), v)
        rhs = Sum(
            Neg(Product(Act(E.anchor(v), Product(f, g)), u)),
            Product(f, g, E.bracket(u, v)),
            Product(f, locality_term(E, g, u, v)),
            Product(g, locality_term(E, f, u, v)),
        )
        chain = ExpandAndSimplify().prove(
            lhs, rhs, registry=reg, engine=eng
        )
        assert chain.steps
