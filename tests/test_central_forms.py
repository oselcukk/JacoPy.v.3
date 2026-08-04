"""Central code — form + pairing tests (Phase 1.B).

Form (1-form, p-form) + ω(U) = ⟨ω, U⟩ pairing.
"""

import pytest

from jacopy.algebra.derivation import degree_of
from jacopy.core.expr import Expr
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import (
    Bundle,
    TM,
    Form,
    forms,
    form_on_vector,
    vector_on_form,
    vector_fields,
    functions,
)


# --------------------------------------------------------------------- #
# Form structure                                                       #
# --------------------------------------------------------------------- #


class TestForm:
    def test_forms_returns_forms(self):
        w, e = forms("ω η", degree=1)
        assert isinstance(w, Form) and isinstance(e, Form)

    def test_single_is_tuple(self):
        result = forms("ω", degree=1)
        assert isinstance(result, tuple) and len(result) == 1

    def test_one_form_degree(self):
        (w,) = forms("ω", degree=1)
        assert w.degree == Degree.const(1)

    def test_p_form_degree(self):
        (sigma,) = forms("σ", degree=3)
        assert sigma.degree == Degree.const(3)

    def test_degree_of_no_registry(self):
        """A Form carries its own degree — degree_of needs no registry."""
        (w,) = forms("ω", degree=2)
        assert degree_of(w) == Degree.const(2)

    def test_symbolic_degree(self):
        p = Degree.var("p")
        (w,) = forms("ω", degree=p)
        assert w.degree == p

    def test_default_bundle_tm(self):
        (w,) = forms("ω", degree=1)
        assert w.bundle == TM

    def test_custom_bundle(self):
        E = Bundle("E", dim=6)
        (w,) = forms("ω", degree=1, bundle=E)
        assert w.bundle == E

    def test_repr(self):
        (w,) = forms("ω", degree=1)
        assert w._repr_inner() == "ω"

    def test_same_name_diff_degree_distinct(self):
        (w1,) = forms("ω", degree=1)
        (w2,) = forms("ω", degree=2)
        assert w1 != w2

    def test_same_name_diff_bundle_distinct(self):
        E = Bundle("E")
        (w_tm,) = forms("ω", degree=1)
        (w_e,) = forms("ω", degree=1, bundle=E)
        assert w_tm != w_e

    def test_equal_forms(self):
        (w1,) = forms("ω", degree=1)
        (w2,) = forms("ω", degree=1)
        assert w1 == w2 and hash(w1) == hash(w2)

    def test_rejects_non_bundle(self):
        with pytest.raises(TypeError):
            Form("ω", degree=1, bundle="nope")

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            Form("", degree=1)


# --------------------------------------------------------------------- #
# Pairing ω(U) = ⟨ω, U⟩                                                 #
# --------------------------------------------------------------------- #


class TestPairing:
    def test_form_call_is_pairing(self):
        (w,) = forms("ω", degree=1)
        (U,) = vector_fields("U")
        expr = w(U)
        assert isinstance(expr, Pairing)
        assert expr.alpha is w
        assert expr.X is U

    def test_pairing_repr(self):
        (w,) = forms("ω", degree=1)
        (U,) = vector_fields("U")
        assert w(U)._repr_inner() == "⟨ω, U⟩"

    def test_pairing_is_scalar(self):
        """⟨ω, U⟩ has degree 0 (scalar) — no registry needed."""
        (w,) = forms("ω", degree=1)
        (U,) = vector_fields("U")
        assert degree_of(w(U)) == Degree.const(0)

    def test_form_on_vector_helper(self):
        (w,) = forms("ω", degree=1)
        (U,) = vector_fields("U")
        assert form_on_vector(w, U) == w(U)

    def test_vector_on_form_same_scalar(self):
        """Both directions yield the same scalar (PDF item 8t)."""
        (w,) = forms("ω", degree=1)
        (U,) = vector_fields("U")
        assert vector_on_form(U, w) == form_on_vector(w, U)

    def test_form_call_rejects_non_expr(self):
        (w,) = forms("ω", degree=1)
        with pytest.raises(TypeError):
            w("not an expr")

    def test_pairing_with_function_scalar(self):
        """⟨ω, U⟩ is a scalar; its product with a function is well defined."""
        from jacopy.core.expr import Product
        reg = PropertyRegistry()
        (w,) = forms("ω", degree=1)
        (U,) = vector_fields("U")
        (f,) = functions("f", registry=reg)
        expr = Product(f, w(U))       # f·⟨ω, U⟩
        assert degree_of(expr, reg) == Degree.const(0)
