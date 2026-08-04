"""Central code — p-form evaluation (MultiEval) + interior ι_X (Phase 1.E.1)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Expr
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import (
    forms,
    vector_fields,
    Interior,
    interior,
    contract,
)


# --------------------------------------------------------------------- #
# p-form evaluation ω(X₁,…,X_p)                                         #
# --------------------------------------------------------------------- #


class TestFormEvaluation:
    def test_one_form_is_pairing(self):
        (w,) = forms("ω", degree=1)
        (U,) = vector_fields("U")
        assert isinstance(w(U), Pairing)

    def test_p_form_is_multi_eval(self):
        (sigma,) = forms("σ", degree=3)
        U, V, W = vector_fields("U V W")
        result = sigma(U, V, W)
        assert isinstance(result, MultiEval)
        assert result.head is sigma
        assert result.args == (U, V, W)

    def test_multi_eval_alternating(self):
        (sigma,) = forms("σ", degree=2)
        U, V = vector_fields("U V")
        result = sigma(U, V)
        assert result.alternating is True
        assert result.slot_kind == "vector"

    def test_full_evaluation_scalar(self):
        """ω(X₁,…,X_p) full evaluation → scalar (degree 0)."""
        (sigma,) = forms("σ", degree=2)
        U, V = vector_fields("U V")
        assert degree_of(sigma(U, V)) == Degree.const(0)

    def test_two_form_repr(self):
        (sigma,) = forms("σ", degree=2)
        U, V = vector_fields("U V")
        assert "σ" in sigma(U, V)._repr_inner()

    def test_call_requires_arg(self):
        (w,) = forms("ω", degree=1)
        with pytest.raises(TypeError):
            w()

    def test_call_rejects_non_expr(self):
        (sigma,) = forms("σ", degree=2)
        (U,) = vector_fields("U")
        with pytest.raises(TypeError):
            sigma(U, "not expr")


# --------------------------------------------------------------------- #
# Interior operator ι_X                                                 #
# --------------------------------------------------------------------- #


class TestInterior:
    def test_interior_is_derivation(self):
        (U,) = vector_fields("U")
        iota = interior(U)
        assert isinstance(iota, Interior)
        assert isinstance(iota, Derivation)

    def test_interior_degree_minus_one(self):
        (U,) = vector_fields("U")
        assert interior(U).degree == Degree.const(-1)

    def test_interior_carries_vector(self):
        (U,) = vector_fields("U")
        assert interior(U).vector is U

    def test_interior_repr(self):
        (U,) = vector_fields("U")
        assert interior(U)._repr_inner() == "ι_U"

    def test_interior_custom_name(self):
        (U,) = vector_fields("U")
        assert interior(U, name="ι")._repr_inner() == "ι"

    def test_distinct_vectors_distinct_iota(self):
        U, V = vector_fields("U V")
        assert interior(U) != interior(V)

    def test_same_vector_equal(self):
        (U,) = vector_fields("U")
        assert interior(U) == interior(U)

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            Interior("not expr")


# --------------------------------------------------------------------- #
# ι_X ω — contraction                                                   #
# --------------------------------------------------------------------- #


class TestContraction:
    def test_contract_is_act(self):
        (U,) = vector_fields("U")
        (sigma,) = forms("σ", degree=3)
        result = contract(U, sigma)
        assert isinstance(result, Act)
        assert isinstance(result.op, Interior)
        assert result.arg is sigma

    def test_contraction_lowers_degree(self):
        """ι_X ω: degree |ω| − 1 (interior lowers the degree by 1)."""
        (U,) = vector_fields("U")
        (sigma,) = forms("σ", degree=3)
        assert degree_of(contract(U, sigma)) == Degree.const(2)

    def test_contraction_one_form_to_scalar(self):
        """ι_X ω: 1-form → 0-form (scalar)."""
        (U,) = vector_fields("U")
        (w,) = forms("ω", degree=1)
        assert degree_of(contract(U, w)) == Degree.const(0)

    def test_contraction_symbolic_degree(self):
        """ι_X ω: p-form (symbolic p) → (p−1)-form."""
        (U,) = vector_fields("U")
        p = Degree.var("p")
        (w,) = forms("ω", degree=p)
        expected = p + Degree.const(-1)
        assert degree_of(contract(U, w)) == expected

    def test_call_form_equals_contract(self):
        """interior(X)(ω) ≡ contract(X, ω)."""
        (U,) = vector_fields("U")
        (sigma,) = forms("σ", degree=2)
        assert interior(U)(sigma) == contract(U, sigma)

    def test_contract_repr(self):
        (U,) = vector_fields("U")
        (sigma,) = forms("σ", degree=2)
        assert contract(U, sigma)._repr_inner() == "ι_U(σ)"

    def test_contract_rejects_non_expr(self):
        (U,) = vector_fields("U")
        with pytest.raises(TypeError):
            contract(U, "not expr")
