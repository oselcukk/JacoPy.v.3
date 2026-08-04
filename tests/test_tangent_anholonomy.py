"""TM case — anholonomy coefficients γ^c_ab + frame writing of the
bracket properties (Phase 2.E)."""

import pytest

from jacopy.algebra.derivation import Act, degree_of
from jacopy.core.expr import Integer, Neg
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import Bundle, frame, functions, vector_fields
from jacopy.central.tangent import (
    AnholonomyCoefficient,
    FrameBracketCoefficientDefinition,
    HolonomicFrameDefinition,
    anholonomy_coefficient,
    lie_bracket,
    tangent_engine,
    prove_gamma_antisymmetry,
    prove_coframe_differential,
    prove_frame_jacobi,
    prove_holonomic_gamma_vanishes,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    e = frame()
    return reg, e, f


# --------------------------------------------------------------------- #
# The coefficient atom                                                   #
# --------------------------------------------------------------------- #


class TestAnholonomyCoefficient:
    def test_canonical_order(self, setup):
        _, e, _ = setup
        g = anholonomy_coefficient(e, "c", "a", "b")
        assert isinstance(g, AnholonomyCoefficient)
        assert g._repr_inner() == "γ^c_ab"

    def test_antisymmetry_normalization(self, setup):
        _, e, _ = setup
        flipped = anholonomy_coefficient(e, "c", "b", "a")
        assert flipped == Neg(anholonomy_coefficient(e, "c", "a", "b"))

    def test_repeated_lower_is_zero(self, setup):
        _, e, _ = setup
        assert anholonomy_coefficient(e, "c", "a", "a") == Integer(0)

    def test_is_a_function_not_a_constant(self, setup):
        """γ is degree 0 but NOT constant: e_d(γ) survives."""
        from jacopy.algorithms.product_rule import product_rule

        reg, e, _ = setup
        g = anholonomy_coefficient(e, "c", "a", "b")
        assert degree_of(g, reg) == Degree.const(0)
        acted = product_rule(Act(e.field("d"), g), reg)
        assert acted != Integer(0)

    def test_distinct_frames_distinct(self, setup):
        _, e, _ = setup
        other = frame("θ")
        assert anholonomy_coefficient(e, "c", "a", "b") != (
            anholonomy_coefficient(other, "c", "a", "b")
        )


# --------------------------------------------------------------------- #
# The definitional rule                                                  #
# --------------------------------------------------------------------- #


class TestCoefficientRule:
    def test_extraction_fires(self, setup):
        """⟨e^c, [e_a, e_b]⟩ → γ^c_ab."""
        reg, e, _ = setup
        node = Pairing(
            e.dual().field("c"), lie_bracket(e.field("a"), e.field("b"))
        )
        out, steps = tangent_engine(registry=reg).expand(node)
        assert out == anholonomy_coefficient(e, "c", "a", "b")
        assert steps

    def test_cross_frame_inert(self, setup):
        """A coframe of another frame must not extract a coefficient."""
        reg, e, _ = setup
        other = frame("θ")
        node = Pairing(
            other.dual().field("c"),
            lie_bracket(e.field("a"), e.field("b")),
        )
        rule = FrameBracketCoefficientDefinition()
        assert not rule.matches(node)

    def test_non_frame_bracket_inert(self, setup):
        reg, e, _ = setup
        X, Y = vector_fields("X Y")
        node = Pairing(e.dual().field("c"), lie_bracket(X, Y))
        assert not FrameBracketCoefficientDefinition().matches(node)

    def test_nested_bracket_inert(self, setup):
        """⟨e^e, [e_a, [e_b, e_c]]⟩ is NOT a plain coefficient."""
        _, e, _ = setup
        node = Pairing(
            e.dual().field("e"),
            lie_bracket(
                e.field("a"), lie_bracket(e.field("b"), e.field("c"))
            ),
        )
        assert not FrameBracketCoefficientDefinition().matches(node)


# --------------------------------------------------------------------- #
# Holonomic mode                                                         #
# --------------------------------------------------------------------- #


class TestHolonomicFrame:
    def test_bracket_vanishes(self, setup):
        """γ extraction collapses to zero through the bracket rewrite."""
        reg, e, _ = setup
        assert prove_holonomic_gamma_vanishes(
            e, "c", "a", "b", registry=reg
        ).steps

    def test_only_declared_frame_affected(self, setup):
        reg, e, _ = setup
        other = frame("θ")
        rule = HolonomicFrameDefinition(other)
        assert not rule.matches(lie_bracket(e.field("a"), e.field("b")))


# --------------------------------------------------------------------- #
# Frame writing of the bracket properties (item 9c)                      #
# --------------------------------------------------------------------- #


class TestFrameProperties:
    def test_gamma_antisymmetry(self, setup):
        reg, e, _ = setup
        assert prove_gamma_antisymmetry(e, "c", "a", "b", registry=reg).steps

    def test_coframe_differential(self, setup):
        """de^c(e_a, e_b) = −γ^c_ab — Palais + duality + is_constant +
        coefficient extraction, all in one proof."""
        reg, e, _ = setup
        chain = prove_coframe_differential(e, "c", "a", "b", registry=reg)
        assert chain.steps

    def test_coframe_differential_wrong_sign_fails(self, setup):
        """Sanity: de^c(e_a, e_b) = +γ^c_ab must NOT close."""
        from jacopy.core.multi_eval import MultiEval
        from jacopy.central.tangent.exterior import d

        reg, e, _ = setup
        e_up = e.dual().field("c")
        lhs = MultiEval(
            d(e_up), e.field("a"), e.field("b"),
            alternating=True, slot_kind="vector",
        )
        rhs = anholonomy_coefficient(e, "c", "a", "b")
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs, rhs, registry=reg, engine=tangent_engine(registry=reg)
            )

    def test_frame_jacobi(self, setup):
        reg, e, f = setup
        chain, used = prove_frame_jacobi(
            e, "e", "a", "b", "c", f, registry=reg
        )
        assert chain.steps
        assert len(used) == 1
