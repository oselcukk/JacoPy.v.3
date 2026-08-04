"""TM case — Lie derivative L + Cartan relations, all as theorems
(Phase 2.D). Includes the repair-loop closure of d² = 0 for p ≥ 1."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import TheoremBook
from jacopy.central.calculus import LieDerivative
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent import (
    CARTAN_TM,
    L,
    d,
    lie_bracket,
    lie_derivative,
    tangent_engine,
    prove_with_bracket_identities,
    prove_d_squared_zero_on_one_forms,
    prove_cartan_magic_on_functions,
    prove_cartan_magic_on_one_forms,
    prove_L_commutes_with_d_on_functions,
    prove_L_iota_commutator,
    prove_L_L_commutator_on_functions,
    prove_L_L_commutator_on_one_forms,
    prove_iota_anticommute,
    register_cartan_theorems,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    X, Y, Z = vector_fields("X Y Z")
    f, g = functions("f g", registry=reg)
    return reg, X, Y, Z, f, g


# --------------------------------------------------------------------- #
# The L operator                                                         #
# --------------------------------------------------------------------- #


class TestLieDerivative:
    def test_degree_zero_derivation(self, setup):
        _, X, *_ = setup
        op = lie_derivative(X)
        assert isinstance(op, LieDerivative)
        assert isinstance(op, Derivation)
        assert op.degree == Degree.const(0)

    def test_carries_calculus_and_direction(self, setup):
        _, X, *_ = setup
        op = lie_derivative(X)
        assert op.calculus_name == "Cartan-TM"
        assert op.vector is X

    def test_L_preserves_degree(self, setup):
        reg, X, *_ = setup
        (w,) = forms("ω", degree=2)
        assert degree_of(L(X, w), reg) == Degree.const(2)

    def test_L_on_function_expands_to_directional(self, setup):
        """L_X f → X(f) (core case of the canonical definition)."""
        reg, X, _, _, f, _ = setup
        out, steps = tangent_engine(registry=reg).expand(L(X, f))
        assert out == Act(X, f)
        assert steps

    def test_L_on_vector_field_expands_to_bracket(self, setup):
        """L_X Y → [X, Y] (core case of the canonical definition)."""
        reg, X, Y, *_ = setup
        out, steps = tangent_engine(registry=reg).expand(L(X, Y))
        assert out == lie_bracket(X, Y)
        assert steps

    def test_L_evaluation_formula_on_one_form(self, setup):
        """⟨L_X ω, Y⟩ = X(⟨ω,Y⟩) − ⟨ω, [X,Y]⟩."""
        reg, X, Y, *_ = setup
        (w,) = forms("ω", degree=1)
        rhs = Sum(
            Act(X, Pairing(w, Y)),
            Neg(Pairing(w, lie_bracket(X, Y))),
        )
        chain = ExpandAndSimplify().prove(
            Pairing(L(X, w), Y), rhs,
            registry=reg, engine=tangent_engine(registry=reg),
        )
        assert chain.steps


# --------------------------------------------------------------------- #
# Interior slot insertion (the Phase 1 deferral, now live)               #
# --------------------------------------------------------------------- #


class TestInteriorRules:
    def test_iota_kills_functions(self, setup):
        reg, X, _, _, f, _ = setup
        out, _ = tangent_engine(registry=reg).expand(Act(Interior(X), f))
        assert out == Integer(0)

    def test_iota_on_one_form_is_pairing(self, setup):
        reg, X, *_ = setup
        (w,) = forms("ω", degree=1)
        out, _ = tangent_engine(registry=reg).expand(Act(Interior(X), w))
        assert out == Pairing(w, X)

    def test_iota_evaluation_inserts_slot(self, setup):
        """⟨ι_X ω, Y⟩ → ω(X, Y) for a 2-form."""
        reg, X, Y, *_ = setup
        (w,) = forms("ω", degree=2)
        node = Pairing(Act(Interior(X), w), Y)
        out, _ = tangent_engine(registry=reg).expand(node)
        assert out == MultiEval(w, X, Y, alternating=True, slot_kind="vector")


# --------------------------------------------------------------------- #
# Cartan relations — all theorems                                        #
# --------------------------------------------------------------------- #


class TestCartanRelations:
    def test_magic_on_functions(self, setup):
        reg, X, _, _, f, _ = setup
        assert prove_cartan_magic_on_functions(X, f, registry=reg).steps

    def test_magic_on_one_forms(self, setup):
        """The centerpiece: v2's axiom, proved here from definitions."""
        reg, X, Y, *_ = setup
        (w,) = forms("ω", degree=1)
        assert prove_cartan_magic_on_one_forms(w, X, Y, registry=reg).steps

    def test_L_commutes_with_d(self, setup):
        reg, X, Y, _, f, _ = setup
        assert prove_L_commutes_with_d_on_functions(
            X, f, Y, registry=reg
        ).steps

    def test_L_iota_commutator(self, setup):
        reg, X, Y, *_ = setup
        (w,) = forms("ω", degree=1)
        assert prove_L_iota_commutator(w, X, Y, registry=reg).steps

    def test_L_L_commutator_on_functions(self, setup):
        reg, X, Y, _, f, _ = setup
        assert prove_L_L_commutator_on_functions(X, Y, f, registry=reg).steps

    def test_L_L_commutator_on_one_forms(self, setup):
        reg, X, Y, Z, f, _ = setup
        (w,) = forms("ω", degree=1)
        chain, used = prove_L_L_commutator_on_one_forms(
            w, X, Y, Z, f, registry=reg
        )
        assert chain.steps
        assert len(used) >= 1  # needed a vector-level bracket identity

    def test_iota_anticommute(self, setup):
        reg, X, Y, *_ = setup
        (w2,) = forms("η", degree=2)
        assert prove_iota_anticommute(w2, X, Y, registry=reg).steps

    def test_wrong_magic_fails(self, setup):
        """Sanity: magic with a dropped term must not close."""
        reg, X, Y, *_ = setup
        (w,) = forms("ω", degree=1)
        lhs = Pairing(L(X, w), Y)
        rhs = Pairing(Act(Interior(X), d(w)), Y)  # missing d(ι_X ω)
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs, rhs, registry=reg, engine=tangent_engine(registry=reg)
            )


# --------------------------------------------------------------------- #
# d² = 0 beyond functions (the 2.C frontier, closed)                     #
# --------------------------------------------------------------------- #


class TestDSquaredZeroHigher:
    def test_one_forms_close_with_one_identity(self, setup):
        reg, X, Y, Z, f, _ = setup
        (w,) = forms("ω", degree=1)
        chain, used = prove_d_squared_zero_on_one_forms(
            w, X, Y, Z, f, registry=reg
        )
        assert chain.steps
        assert len(used) == 1  # one rearranged-Jacobi instance
        assert used[0].rhs == Integer(0)
        assert used[0].generality == "generic-function"

    def test_two_forms_close_with_four_identities(self, setup):
        reg, X, Y, Z, f, _ = setup
        (W,) = vector_fields("W")
        (w2,) = forms("η", degree=2)
        ddw2 = MultiEval(
            d(d(w2)), X, Y, Z, W, alternating=True, slot_kind="vector"
        )
        chain, used = prove_with_bracket_identities(
            ddw2, Integer(0), f, registry=reg, max_repairs=8
        )
        assert chain.steps
        assert len(used) == 4  # one Jacobi instance per fixed slot

    def test_repair_guard_rejects_non_sections(self, setup):
        """Soundness guard (audit 2): the repair loop must refuse to
        conclude V = 0 from V(f) = 0 when V is not a section — e.g. an
        interior product kills every function without being zero."""
        from jacopy.central.objects.interior import Interior
        from jacopy.central.tangent.cartan import _is_section_combination

        reg, X, Y, _, f, _ = setup
        assert not _is_section_combination(Interior(X))
        assert _is_section_combination(
            Sum(lie_bracket(X, Y), Neg(X))
        )

    def test_repair_loop_rejects_false_identity(self, setup):
        """The repair loop must NOT close a genuinely false equation."""
        reg, X, Y, _, f, _ = setup
        (w,) = forms("ω", degree=1)
        lhs = Pairing(w, lie_bracket(X, Y))
        with pytest.raises(ProofFailure):
            prove_with_bracket_identities(
                lhs, Integer(0), f, registry=reg
            )


# --------------------------------------------------------------------- #
# Book registration                                                      #
# --------------------------------------------------------------------- #


class TestRegistration:
    def test_register_cartan_theorems(self, setup):
        reg, X, Y, _, f, _ = setup
        (w,) = forms("ω", degree=1)
        book = TheoremBook()
        thms = register_cartan_theorems(book, w, X, Y, f, registry=reg)
        assert {t.name for t in thms} == {
            "cartan_magic_on_functions",
            "cartan_magic_on_one_forms",
            "L_iota_commutator",
            "iota_anticommute",
        }
        assert all(t.name in book for t in thms)
