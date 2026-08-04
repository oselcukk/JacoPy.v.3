"""Algebroid case — the anchored-bundle context, definitional rules,
and the literal E = TM reduction (Phase 3.A)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import Bundle, functions, vector_fields
from jacopy.central.algebroid import (
    Algebroid,
    AlgebroidBracket,
    AnchoredVF,
    BracketBilinearityDefinition,
    CoboundaryForm,
    algebroid,
    algebroid_engine,
    tangent_algebroid,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    E = algebroid("E", Bundle("E"))
    u, v, w = E.sections("u v w")
    (f,) = functions("f", registry=reg)
    return reg, E, u, v, w, f


# --------------------------------------------------------------------- #
# Context objects                                                        #
# --------------------------------------------------------------------- #


class TestAlgebroidContext:
    def test_sections_live_on_bundle(self, setup):
        _, E, u, *_ = setup
        assert u.bundle == E.bundle

    def test_anchor_is_degree_zero_derivation(self, setup):
        reg, E, u, *_ = setup
        rho_u = E.anchor(u)
        assert isinstance(rho_u, AnchoredVF)
        assert isinstance(rho_u, Derivation)
        assert rho_u.degree == Degree.const(0)
        assert rho_u.section is u

    def test_bracket_is_section_again(self, setup):
        _, E, u, v, *_ = setup
        br = E.bracket(u, v)
        assert isinstance(br, AlgebroidBracket)
        assert br._repr_inner() == "[u,v]_E"

    def test_anchored_action_is_scalar(self, setup):
        reg, E, u, _, _, f = setup
        expr = E.act(u, f)
        assert isinstance(expr, Act)
        assert degree_of(expr, reg) == Degree.const(0)

    def test_coboundary_is_e_one_form(self, setup):
        reg, E, _, _, _, f = setup
        Df = E.D(f)
        assert isinstance(Df, CoboundaryForm)
        assert degree_of(Df, reg) == Degree.const(1)
        assert Df.bundle == E.bundle

    def test_distinct_algebroids_distinct_nodes(self, setup):
        _, E, u, v, *_ = setup
        other = algebroid("F", Bundle("E"))
        assert E.bracket(u, v) != other.bracket(u, v)
        assert E.anchor(u) != other.anchor(u)

    def test_anchor_name_parametric(self):
        E = algebroid("A", Bundle("A"), anchor_name="a")
        (u,) = E.sections("u")
        assert E.anchor(u)._repr_inner() == "a(u)"


# --------------------------------------------------------------------- #
# Definitional rules                                                     #
# --------------------------------------------------------------------- #


class TestDefinitionalRules:
    def test_anchor_additivity(self, setup):
        reg, E, u, v, *_ = setup
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(E.anchor(Sum(u, v)))
        assert steps
        assert out == Sum(E.anchor(u), E.anchor(v))

    def test_anchor_smooth_linearity(self, setup):
        """ρ(fu) = f·ρ(u) — C∞-linearity holds BY DEFINITION (bundle
        morphism), unlike the bracket's Leibniz rules."""
        reg, E, u, _, _, f = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(E.anchor(Product(f, u)))
        assert out == Product(f, E.anchor(u))

    def test_bracket_real_bilinearity(self, setup):
        reg, E, u, v, w, _ = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(E.bracket(Sum(u, v), w))
        assert out == Sum(E.bracket(u, w), E.bracket(v, w))
        out, _ = eng.expand(E.bracket(u, Product(Integer(2), v)))
        assert out == Product(Integer(2), E.bracket(u, v))
        out, _ = eng.expand(E.bracket(Neg(u), v))
        assert out == Neg(E.bracket(u, v))

    def test_bracket_smooth_factor_stays_inert(self, setup):
        """THE layer split: [u, fv] must NOT expand — right-Leibniz is
        a declared axiom (3.B), not a definition."""
        reg, E, u, v, _, f = setup
        eng = algebroid_engine(E, registry=reg)
        node = E.bracket(u, Product(f, v))
        out, steps = eng.expand(node)
        assert out == node and not steps
        node = E.bracket(Product(f, u), v)
        out, steps = eng.expand(node)
        assert out == node and not steps

    def test_bilinearity_rule_scoped_to_algebroid_brackets(self, setup):
        _, E, u, v, *_ = setup
        rule = BracketBilinearityDefinition()
        from jacopy.central.tangent import lie_bracket

        assert not rule.matches(lie_bracket(Sum(u, v), u))

    def test_coboundary_evaluation(self, setup):
        """⟨Df, u⟩ = ρ(u)(f) [MC Def 3.5]."""
        reg, E, u, _, _, f = setup
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(Pairing(E.D(f), u))
        assert steps
        assert out == Act(E.anchor(u), f)

    def test_coboundary_bilinear_evaluation_proof(self, setup):
        """⟨Df, u + v⟩ = ⟨Df, u⟩ + ⟨Df, v⟩ closes (collect + anchor
        linearity + coboundary together)."""
        from jacopy.proof.strategies import ExpandAndSimplify

        reg, E, u, v, _, f = setup
        lhs = Pairing(E.D(f), Sum(u, v))
        rhs = Sum(Pairing(E.D(f), u), Pairing(E.D(f), v))
        chain = ExpandAndSimplify().prove(
            lhs, rhs, registry=reg, engine=algebroid_engine(E, registry=reg)
        )
        assert chain.steps

    def test_anchored_leibniz_on_functions(self, setup):
        """ρ(u)(f·g) = ρ(u)(f)·g + f·ρ(u)(g) — ρ(u) is a derivation,
        so the graded Leibniz pass applies to it for free."""
        from jacopy.proof.strategies import ExpandAndSimplify

        reg, E, u, _, _, f = setup
        (g,) = functions("g", registry=reg)
        lhs = Act(E.anchor(u), Product(f, g))
        rhs = Sum(
            Product(Act(E.anchor(u), f), g),
            Product(f, Act(E.anchor(u), g)),
        )
        chain = ExpandAndSimplify().prove(
            lhs, rhs, registry=reg, engine=algebroid_engine(E, registry=reg)
        )
        assert chain.steps


# --------------------------------------------------------------------- #
# The literal E = TM reduction (PDF item 8a)                             #
# --------------------------------------------------------------------- #


class TestTangentReduction:
    def test_anchor_is_identity(self):
        TMa = tangent_algebroid()
        (X,) = vector_fields("X")
        assert TMa.anchor(X) is X

    def test_bracket_is_lie(self):
        from jacopy.central.tangent import lie_bracket

        TMa = tangent_algebroid()
        X, Y = vector_fields("X Y")
        assert TMa.bracket(X, Y) == lie_bracket(X, Y)

    def test_coboundary_is_usual_d(self):
        from jacopy.central.tangent import d

        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        assert tangent_algebroid().D(f) == d(f)

    def test_action_is_direct(self):
        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        (X,) = vector_fields("X")
        assert tangent_algebroid().act(X, f) == Act(X, f)

    def test_engine_delegates_to_tangent(self):
        """⟨Df, X⟩ = X(f) closes via the Palais rule — the algebroid
        engine on TM IS the tangent engine."""
        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        (X,) = vector_fields("X")
        TMa = tangent_algebroid()
        eng = algebroid_engine(TMa, registry=reg)
        out, _ = eng.expand(Pairing(TMa.D(f), X))
        assert out == Act(X, f)

    def test_full_proof_through_algebroid_api(self):
        """A Phase 2 theorem re-proved entirely through the algebroid
        API: antisymmetry of the TM bracket."""
        from jacopy.proof.strategies import ExpandAndSimplify

        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        TMa = tangent_algebroid()
        X, Y = vector_fields("X Y")
        lhs = Act(TMa.bracket(X, Y), f)
        rhs = Neg(Act(TMa.bracket(Y, X), f))
        chain = ExpandAndSimplify().prove(
            lhs, rhs, registry=reg,
            engine=algebroid_engine(TMa, registry=reg),
        )
        assert chain.steps
