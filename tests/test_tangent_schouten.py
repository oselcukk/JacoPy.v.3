"""TM case — Schouten-Nijenhuis bracket, Wedge lift, Sym/Alt
unfolding, and the 8w generalization (Phase 2.F)."""

import pytest

from jacopy.algebra.derivation import Act, degree_of
from jacopy.brackets.base import BracketApply
from jacopy.core.expr import Integer, Neg, Product, Rational, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.core.wedge import Wedge
from jacopy.algorithms.simplify import simplify
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import (
    antisymmetrize,
    bivector,
    forms,
    functions,
    symmetrize,
    tensors,
    vector_fields,
)
from jacopy.central.objects.interior import contract_all
from jacopy.central.tangent import lie_bracket, tangent_engine
from jacopy.central.tangent.lie_bracket import BracketOrientationDefinition
from jacopy.central.tangent.schouten import (
    SN,
    SNExpansionDefinition,
    multivector_degree,
    prove_reduces_to_lie,
    prove_sn_graded_antisymmetry,
    prove_sn_vector_function,
    prove_sn_wedge_leibniz,
    sn_bracket,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    X, Y, Z = vector_fields("X Y Z")
    f, g = functions("f g", registry=reg)
    return reg, X, Y, Z, f, g


# --------------------------------------------------------------------- #
# Wedge lift + wedge canonical form                                      #
# --------------------------------------------------------------------- #


class TestWedgeLift:
    def test_wedge_of_vectors_is_bivector(self, setup):
        _, X, Y, *_ = setup
        assert degree_of(Wedge(X, Y)) == Degree.const(2)

    def test_wedge_with_bracket_child(self, setup):
        _, X, Y, Z, *_ = setup
        assert degree_of(Wedge(X, lie_bracket(Y, Z))) == Degree.const(2)

    def test_wedge_repeated_vector_zero(self, setup):
        _, X, Y, *_ = setup
        assert simplify(Wedge(X, X)) == Integer(0)

    def test_wedge_sort_sign(self, setup):
        """Y ∧ X normalizes to −(X ∧ Y) (odd-degree factors)."""
        _, X, Y, *_ = setup
        assert simplify(Wedge(Y, X)) == Neg(Wedge(X, Y))

    def test_wedge_neg_pullout(self, setup):
        _, X, Y, *_ = setup
        assert simplify(Wedge(X, Neg(Y))) == Neg(Wedge(X, Y))


# --------------------------------------------------------------------- #
# SN bracket — node + degree helper                                      #
# --------------------------------------------------------------------- #


class TestSNNode:
    def test_multivector_degrees(self, setup):
        reg, X, Y, _, f, _ = setup
        assert multivector_degree(f, reg) == 0
        assert multivector_degree(X, reg) == 1
        assert multivector_degree(Wedge(X, Y), reg) == 2
        assert multivector_degree(bivector("π"), reg) == 2

    def test_opaque_for_atomic_bivector(self, setup):
        """[π, X] with an opaque bivector stays inert (Phase 5 will
        reason about it symbolically)."""
        reg, X, *_ = setup
        node = sn_bracket(bivector("π"), X)
        assert not SNExpansionDefinition(reg).matches(node)

    def test_inert_for_symbolic_degree(self, setup):
        reg, X, *_ = setup
        from jacopy.central.objects import p_vectors

        p = Degree.var("p")
        (pi,) = p_vectors("π", degree=p)
        assert multivector_degree(pi, reg) is None
        assert not SNExpansionDefinition(reg).matches(sn_bracket(pi, X))


# --------------------------------------------------------------------- #
# SN theorems (PDF 9f)                                                   #
# --------------------------------------------------------------------- #


class TestSNTheorems:
    def test_reduces_to_lie(self, setup):
        reg, X, Y, _, f, _ = setup
        assert prove_reduces_to_lie(X, Y, f, registry=reg).steps

    def test_vector_function_shifted_antisymmetry(self, setup):
        reg, X, _, _, f, _ = setup
        assert prove_sn_vector_function(X, f, registry=reg).steps

    def test_functions_bracket_vanishes(self, setup):
        reg, _, _, _, f, g = setup
        chain = ExpandAndSimplify().prove(
            sn_bracket(f, g), Integer(0),
            registry=reg, engine=tangent_engine(registry=reg),
        )
        assert chain.steps

    def test_wedge_leibniz(self, setup):
        reg, X, Y, Z, *_ = setup
        assert prove_sn_wedge_leibniz(X, Y, Z, registry=reg).steps

    def test_graded_antisymmetry(self, setup):
        """[X∧Y, Z] = −[Z, X∧Y] — closes through the theorem-backed
        bracket orientation canonical form."""
        reg, X, Y, Z, *_ = setup
        assert prove_sn_graded_antisymmetry(X, Y, Z, registry=reg).steps

    def test_wrong_sign_fails(self, setup):
        """Sanity: [X∧Y, Z] = +[Z, X∧Y] must NOT close."""
        reg, X, Y, Z, *_ = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                sn_bracket(Wedge(X, Y), Z),
                sn_bracket(Z, Wedge(X, Y)),
                registry=reg,
                engine=tangent_engine(registry=reg),
            )


# --------------------------------------------------------------------- #
# Bracket orientation canonical form                                     #
# --------------------------------------------------------------------- #


class TestSNFunctionOperands:
    """Audit family (2026-07-23): the m=0 operand corner that exposed
    the inherited sign bug — [X∧Y, f] must alternate under X ↔ Y."""

    def test_wedge_function_alternates(self, setup):
        reg, X, Y, _, f, _ = setup
        alt = Sum(
            sn_bracket(Wedge(X, Y), f), sn_bracket(Wedge(Y, X), f)
        )
        chain = ExpandAndSimplify().prove(
            alt, Integer(0), registry=reg, engine=tangent_engine(registry=reg)
        )
        assert chain.steps

    def test_graded_antisymmetry_with_function(self, setup):
        """[f, X∧Y] = +[X∧Y, f] (sign (−1)^{1+s_f·s_{X∧Y}} = +1)."""
        reg, X, Y, _, f, _ = setup
        lhs = Sum(
            sn_bracket(f, Wedge(X, Y)),
            Neg(sn_bracket(Wedge(X, Y), f)),
        )
        chain = ExpandAndSimplify().prove(
            lhs, Integer(0), registry=reg, engine=tangent_engine(registry=reg)
        )
        assert chain.steps

    def test_scalar_factor_leaves_wedge(self, setup):
        """f ∧ X normalizes to the same canonical form as f·X
        (degree-0 factors commute out of the wedge)."""
        reg, X, _, _, f, _ = setup
        from jacopy.core.expr import Product

        assert simplify(Wedge(f, X), reg) == simplify(Product(f, X), reg)
        assert not isinstance(simplify(Wedge(f, X), reg), Wedge)

    def test_sn_degree_hook(self, setup):
        """degree_of on SN nodes uses the MULTIVECTOR grading
        (audit fix: the operator-grading default gave m([π,π]) = 4)."""
        from jacopy.core.symbolic_degree import Degree

        reg, X, Y, *_ = setup
        pi = bivector("π")
        assert degree_of(sn_bracket(pi, pi), reg) == Degree.const(3)
        assert degree_of(sn_bracket(X, Y), reg) == Degree.const(1)
        assert degree_of(sn_bracket(Wedge(X, Y), X), reg) == Degree.const(2)


class TestNamingAudit:
    """Audit 3 (2026-07-23): naming/display/identity family — free
    symbol choice must never affect identity, and display patterns
    must follow the carried names."""

    def test_type_sensitive_equality(self, setup):
        """Same key, different type -> distinct (FrameField vs plain
        VectorField named identically)."""
        from jacopy.central.objects import frame, forms as _forms
        from jacopy.central.objects import vector_fields as _vf

        e = frame()
        (user_vf,) = _vf("e_a")
        assert e.field("a") != user_vf
        assert len({e.field("a"), user_vf}) == 2
        (user_form,) = _forms("e^a", degree=1)
        assert e.dual().field("a") != user_form

    def test_lie_derivative_name_parametric(self, setup):
        """The tilde calculus can display its Lie derivative as L̃
        (same bug family as the fixed nabla/D connection display)."""
        from jacopy.central.calculus import BracketCalculus

        _, X, *_ = setup
        tilde = BracketCalculus(
            "tilde-test", anchor=lambda v: v, bracket=lie_bracket,
            d_name="d̃", lie_name="L̃",
        )
        assert tilde.lie(X)._repr_inner() == "L̃_X"
        assert tilde.d._repr_inner() == "d̃"
        # identity stays calculus-scoped, not display-scoped
        assert tilde.lie(X) != __import__(
            "jacopy.central.tangent", fromlist=["CARTAN_TM"]
        ).CARTAN_TM.lie(X)

    def test_gamma_display_carries_frame_name(self, setup):
        from jacopy.central.objects import frame
        from jacopy.central.tangent.anholonomy import (
            anholonomy_coefficient,
        )

        g_default = anholonomy_coefficient(frame("e"), "c", "a", "b")
        g_theta = anholonomy_coefficient(frame("θ"), "c", "a", "b")
        assert g_default != g_theta
        assert g_default._repr_inner() != g_theta._repr_inner()

    def test_connection_display_uses_own_name(self, setup):
        from jacopy.central.objects import connection

        _, X, Y, *_ = setup
        assert "D_" in connection("D")(X, Y)._repr_inner()

    def test_proofs_name_independent(self, setup):
        """A full theorem closes with arbitrary unicode names."""
        from jacopy.central.objects import vector_fields as _vf
        from jacopy.central.tangent import prove_antisymmetry

        reg, *_ = setup
        ksi, zeta = _vf("ξ ζ")
        (phi,) = __import__(
            "jacopy.central.objects", fromlist=["functions"]
        ).functions("φ", registry=reg)
        assert prove_antisymmetry(ksi, zeta, phi, registry=reg).steps


class TestBracketOrientation:
    def test_flip(self, setup):
        _, X, Y, *_ = setup
        rule = BracketOrientationDefinition()
        flipped = lie_bracket(Y, X)  # "Y" > "X"
        assert rule.matches(flipped)
        assert rule.rewrite(flipped) == Neg(lie_bracket(X, Y))

    def test_canonical_untouched(self, setup):
        _, X, Y, *_ = setup
        assert not BracketOrientationDefinition().matches(lie_bracket(X, Y))

    def test_self_bracket_zero(self, setup):
        _, X, *_ = setup
        rule = BracketOrientationDefinition()
        assert rule.matches(lie_bracket(X, X))
        assert rule.rewrite(lie_bracket(X, X)) == Integer(0)

    def test_is_theorem_classified(self):
        assert BracketOrientationDefinition().is_theorem


# --------------------------------------------------------------------- #
# Sym/Alt permutation unfolding (item 8x deferral)                       #
# --------------------------------------------------------------------- #


class TestSymAltUnfolding:
    def test_alt_two_slots(self, setup):
        reg, X, Y, *_ = setup
        (T,) = tensors("T", upper=0, lower=2)
        node = MultiEval(
            antisymmetrize(T), X, Y, alternating=False, slot_kind="vector"
        )
        out, steps = tangent_engine(registry=reg).expand(node)
        assert steps
        expected = Product(
            Rational(1, 2),
            Sum(
                MultiEval(T, X, Y, alternating=False, slot_kind="vector"),
                Neg(MultiEval(T, Y, X, alternating=False, slot_kind="vector")),
            ),
        )
        assert out == expected

    def test_alt_antisymmetry_theorem(self, setup):
        """Alt(T)(X,Y) + Alt(T)(Y,X) = 0."""
        reg, X, Y, *_ = setup
        (T,) = tensors("T", upper=0, lower=2)
        both = Sum(
            MultiEval(antisymmetrize(T), X, Y, alternating=False, slot_kind="vector"),
            MultiEval(antisymmetrize(T), Y, X, alternating=False, slot_kind="vector"),
        )
        chain = ExpandAndSimplify().prove(
            both, Integer(0), registry=reg, engine=tangent_engine(registry=reg)
        )
        assert chain.steps

    def test_sym_symmetry_theorem(self, setup):
        """Sym(T)(X,Y) − Sym(T)(Y,X) = 0."""
        reg, X, Y, *_ = setup
        (T,) = tensors("T", upper=0, lower=2)
        diff = Sum(
            MultiEval(symmetrize(T), X, Y, alternating=False, slot_kind="vector"),
            Neg(MultiEval(symmetrize(T), Y, X, alternating=False, slot_kind="vector")),
        )
        chain = ExpandAndSimplify().prove(
            diff, Integer(0), registry=reg, engine=tangent_engine(registry=reg)
        )
        assert chain.steps

    def test_single_slot_identity(self, setup):
        reg, X, *_ = setup
        (w,) = forms("ω", degree=1)
        node = Pairing(symmetrize(w), X)
        out, _ = tangent_engine(registry=reg).expand(node)
        assert out == Pairing(w, X)


# --------------------------------------------------------------------- #
# 8w generalization — iterated interior = partial evaluation             #
# --------------------------------------------------------------------- #


class TestItem8wGeneralization:
    def test_iterated_interior_equals_full_evaluation(self, setup):
        """⟨ι_Y ι_X σ, Z⟩ = σ(X, Y, Z) for a 3-form — fixing slots of
        an alternating k-linear map IS the iterated interior product."""
        reg, X, Y, Z, *_ = setup
        (s3,) = forms("σ", degree=3)
        lhs = Pairing(contract_all(s3, X, Y), Z)
        rhs = MultiEval(s3, X, Y, Z, alternating=True, slot_kind="vector")
        chain = ExpandAndSimplify().prove(
            lhs, rhs, registry=reg, engine=tangent_engine(registry=reg)
        )
        assert chain.steps

    def test_contract_all_validation(self, setup):
        _, X, *_ = setup
        (w,) = forms("ω", degree=2)
        with pytest.raises(ValueError):
            contract_all(w)
        with pytest.raises(TypeError):
            contract_all(w, "not expr")
