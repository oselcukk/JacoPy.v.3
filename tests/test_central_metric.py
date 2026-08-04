"""Central code — metric, inverse metric, Hodge star (Phase 1.F)."""

import pytest

from jacopy.algebra.derivation import degree_of
from jacopy.core.expr import Atom
from jacopy.core.multi_eval import MultiEval
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import (
    Bundle,
    tangent_bundle,
    Metric,
    InverseMetric,
    metric,
    hodge,
    HodgeStar,
    Flat,
    Sharp,
    forms,
    vector_fields,
)


# --------------------------------------------------------------------- #
# Metric g                                                             #
# --------------------------------------------------------------------- #


class TestMetric:
    def test_default_name(self):
        g = metric()
        assert g.name == "g"

    def test_signature_0_2(self):
        assert metric().signature == (0, 2)

    def test_g_of_two_vectors(self):
        g = metric()
        X, Y = vector_fields("X Y")
        result = g(X, Y)
        assert isinstance(result, MultiEval)
        assert result.head is g

    def test_g_symmetric_not_alternating(self):
        """g(X,Y) is symmetric → alternating=False."""
        g = metric()
        X, Y = vector_fields("X Y")
        assert g(X, Y).alternating is False

    def test_g_of_vectors_scalar(self):
        g = metric()
        X, Y = vector_fields("X Y")
        assert degree_of(g(X, Y)) == Degree.const(0)

    def test_flat_from_metric(self):
        g = metric()
        assert isinstance(g.flat(), Flat)

    def test_inverse(self):
        g = metric()
        g_inv = g.inverse()
        assert isinstance(g_inv, InverseMetric)
        assert g_inv.signature == (2, 0)

    def test_inverse_name(self):
        g = metric("g")
        assert g.inverse().name == "g⁻¹"

    def test_metric_repr(self):
        assert metric("g")._repr_inner() == "g"

    def test_rejects_bad_bundle(self):
        with pytest.raises(TypeError):
            Metric("g", bundle="nope")


# --------------------------------------------------------------------- #
# Inverse metric g⁻¹                                                   #
# --------------------------------------------------------------------- #


class TestInverseMetric:
    def test_g_inv_of_two_forms(self):
        g_inv = metric().inverse()
        a, b = forms("α β", degree=1)
        result = g_inv(a, b)
        assert isinstance(result, MultiEval)
        assert result.slot_kind == "covector"

    def test_g_inv_scalar(self):
        g_inv = metric().inverse()
        a, b = forms("α β", degree=1)
        assert degree_of(g_inv(a, b)) == Degree.const(0)

    def test_sharp_from_inverse(self):
        g_inv = metric().inverse()
        assert isinstance(g_inv.sharp(), Sharp)


# --------------------------------------------------------------------- #
# Hodge star ⋆_g                                                       #
# --------------------------------------------------------------------- #


class TestHodge:
    def test_hodge_is_hodgestar(self):
        g = metric(bundle=tangent_bundle(dim=4))
        (w,) = forms("ω", degree=1)
        assert isinstance(hodge(w, g), HodgeStar)

    def test_hodge_degree_n_minus_p_concrete(self):
        """⋆ω: n=4, p=1 → 4−1=3-form."""
        g = metric(bundle=tangent_bundle(dim=4))
        (w,) = forms("ω", degree=1)
        assert degree_of(hodge(w, g)) == Degree.const(3)

    def test_hodge_degree_n_minus_p_two_form(self):
        """⋆ω: n=4, p=2 → 2-form."""
        g = metric(bundle=tangent_bundle(dim=4))
        (w,) = forms("ω", degree=2)
        assert degree_of(hodge(w, g)) == Degree.const(2)

    def test_hodge_symbolic_dim(self):
        """Symbolic dimension (dim=None) → ⋆ω has degree n − p."""
        g = metric()  # TM, dim=None → symbolic n
        (w,) = forms("ω", degree=2)
        expected = Degree.var("n") - Degree.const(2)
        assert degree_of(hodge(w, g)) == expected

    def test_hodge_symbolic_form_and_dim(self):
        """Both the dimension and the form degree symbolic: ⋆ω = n − p."""
        g = metric()
        p = Degree.var("p")
        (w,) = forms("ω", degree=p)
        assert degree_of(hodge(w, g)) == Degree.var("n") - p

    def test_hodge_repr(self):
        g = metric(bundle=tangent_bundle(dim=3))
        (w,) = forms("ω", degree=1)
        assert hodge(w, g)._repr_inner() == "⋆ω"

    def test_hodge_rejects_non_metric(self):
        (w,) = forms("ω", degree=1)
        with pytest.raises(TypeError):
            hodge(w, "not a metric")

    def test_double_hodge_grading(self):
        """⋆⋆ω: n=4, p=1 → ⋆(3-form) → 4−3=1-form (back to the start)."""
        g = metric(bundle=tangent_bundle(dim=4))
        (w,) = forms("ω", degree=1)
        once = hodge(w, g)
        twice = hodge(once, g)
        assert degree_of(twice) == Degree.const(1)  # returned to the start
