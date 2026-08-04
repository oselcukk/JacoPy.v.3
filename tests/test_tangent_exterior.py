"""TM case — exterior derivative d derived from the Lie bracket
(Phase 2.C): Palais formula, d² = 0 on functions, TheoremBook entry."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import TheoremBook, cite
from jacopy.proof.expansion import ExpansionEngine
from jacopy.central.calculus import (
    BracketCalculus,
    ExteriorDerivative,
    IntrinsicDDefinition,
)
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.tangent import tangent_engine
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.central.tangent.exterior import (
    CARTAN_TM,
    cartan_calculus,
    d,
    exterior_d,
    prove_df_on_vector,
    prove_one_form_intrinsic,
    prove_d_squared_zero_on_functions,
    register_d_squared_zero,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    X, Y, Z = vector_fields("X Y Z")
    f, g = functions("f g", registry=reg)
    return reg, X, Y, Z, f, g


# --------------------------------------------------------------------- #
# The operator                                                           #
# --------------------------------------------------------------------- #


class TestExteriorDerivative:
    def test_is_degree_plus_one_derivation(self):
        op = exterior_d()
        assert isinstance(op, ExteriorDerivative)
        assert isinstance(op, Derivation)
        assert op.degree == Degree.const(1)

    def test_carries_calculus_identity(self):
        assert exterior_d().calculus_name == "Cartan-TM"

    def test_distinct_calculi_distinct_d(self):
        other = BracketCalculus(
            "other", anchor=lambda X: X, bracket=lie_bracket
        )
        assert other.d != exterior_d()

    def test_degree_bookkeeping(self, setup):
        """|dω| = |ω| + 1, automatically via Act."""
        reg, *_ = setup
        (w,) = forms("ω", degree=1)
        assert degree_of(d(w), reg) == Degree.const(2)
        p = Degree.var("p")
        (eta,) = forms("η", degree=p)
        assert degree_of(d(eta), reg) == p + Degree.const(1)

    def test_tm_is_an_instantiation(self):
        """PDF 8a: the usual d IS the (id, Lie) instantiation."""
        calc = cartan_calculus()
        assert calc is CARTAN_TM
        (X,) = vector_fields("X")
        assert calc.anchor(X) is X


# --------------------------------------------------------------------- #
# The Palais definitional rule                                           #
# --------------------------------------------------------------------- #


class TestPalaisRule:
    def test_zero_form_case(self, setup):
        """⟨df, X⟩ = X(f) closes definitionally."""
        reg, X, _, _, f, _ = setup
        assert len(prove_df_on_vector(X, f, registry=reg).steps) == 2

    def test_one_form_case(self, setup):
        """dω(X,Y) = X(ω(Y)) − Y(ω(X)) − ω([X,Y])."""
        reg, X, Y, *_ = setup
        (w,) = forms("ω", degree=1)
        assert prove_one_form_intrinsic(w, X, Y, registry=reg).steps

    def test_two_form_unroll_shape(self, setup):
        """dω(X,Y,Z) for a 2-form: 3 anchor terms + 3 bracket terms."""
        reg, X, Y, Z, *_ = setup
        (w,) = forms("ω", degree=2)
        node = MultiEval(d(w), X, Y, Z, alternating=True, slot_kind="vector")
        rule = IntrinsicDDefinition(CARTAN_TM, reg)
        assert rule.matches(node)
        out = rule.rewrite(node)
        assert isinstance(out, Sum)
        assert len(out.children) == 6

    def test_arity_mismatch_left_inert(self, setup):
        """A 1-form's d evaluated on 3 slots must NOT fire (diagnostic)."""
        reg, X, Y, Z, *_ = setup
        (w,) = forms("ω", degree=1)
        bad = MultiEval(d(w), X, Y, Z, alternating=True, slot_kind="vector")
        assert not IntrinsicDDefinition(CARTAN_TM, reg).matches(bad)

    def test_pairing_case_requires_zero_form(self, setup):
        reg, X, *_ = setup
        (w,) = forms("ω", degree=1)
        pairing_of_dw = Pairing(d(w), X)  # dω is a 2-form: wrong arity
        assert not IntrinsicDDefinition(CARTAN_TM, reg).matches(pairing_of_dw)

    def test_foreign_d_untouched(self, setup):
        """Another calculus' d is not expanded by the TM rule."""
        reg, X, Y, _, f, _ = setup
        other = BracketCalculus(
            "other", anchor=lambda v: v, bracket=lie_bracket
        )
        node = MultiEval(
            Act(other.d, Act(other.d, f)),
            X, Y, alternating=True, slot_kind="vector",
        )
        assert not IntrinsicDDefinition(CARTAN_TM, reg).matches(node)


# --------------------------------------------------------------------- #
# d² = 0                                                                 #
# --------------------------------------------------------------------- #


class TestDSquaredZero:
    def test_closes_on_functions(self, setup):
        reg, X, Y, _, f, _ = setup
        chain = prove_d_squared_zero_on_functions(f, X, Y, registry=reg)
        assert chain.steps
        assert chain.steps[-1].after == Integer(0)

    def test_registered_and_citable(self, setup):
        """Prove once, register, cite: one theorem-tagged rewrite."""
        reg, X, Y, _, f, _ = setup
        book = TheoremBook()
        thm = register_d_squared_zero(book, f, X, Y, registry=reg)
        assert thm.generality == "generic-function"
        eng = cite(ExpansionEngine([]), book, thm.name)
        node = MultiEval(d(d(f)), X, Y, alternating=True, slot_kind="vector")
        out, steps = eng.expand(node)
        assert out == Integer(0)
        assert steps[0].provenance_tag == "theorem"

    def test_one_form_residual_is_jacobi(self, setup):
        """On 1-forms, plain ExpandAndSimplify (no cited identities)
        cancels everything except ⟨ω, ·⟩ over the vector-level Jacobi
        sum. The closing proof lives in 2.D
        (test_tangent_cartan.TestDSquaredZeroHigher) via the repair
        loop; this test keeps pinning what the *definitional* rules
        alone can and cannot do."""
        reg, X, Y, Z, *_ = setup
        (w,) = forms("ω", degree=1)
        ddw = MultiEval(d(d(w)), X, Y, Z, alternating=True, slot_kind="vector")
        with pytest.raises(ProofFailure) as exc:
            ExpandAndSimplify().prove(
                ddw, Integer(0), registry=reg,
                engine=tangent_engine(registry=reg),
            )
        msg = str(exc.value)
        assert "⟨ω," in msg and "_VF" in msg  # pairing-over-bracket residual
