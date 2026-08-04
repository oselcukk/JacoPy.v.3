"""Central code — wedge ∧, tensor product ⊗, (anti)symmetrization (Phase 1.D)."""

import pytest

from jacopy.algebra.derivation import degree_of
from jacopy.core.expr import Symbol, Zero
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import (
    forms,
    p_vectors,
    Wedge,
    wedge,
    TensorProduct,
    tensor_product,
    Symmetrization,
    Antisymmetrization,
    symmetrize,
    antisymmetrize,
)


# --------------------------------------------------------------------- #
# Wedge ∧                                                              #
# --------------------------------------------------------------------- #


class TestWedge:
    def test_wedge_of_two_forms(self):
        a, b = forms("α β", degree=1)
        w = wedge(a, b)
        assert isinstance(w, Wedge)

    def test_wedge_degree_law(self):
        """|α∧β| = |α| + |β|."""
        a, b = forms("α β", degree=1)
        assert degree_of(wedge(a, b)) == Degree.const(2)

    def test_wedge_mixed_degree(self):
        (a,) = forms("α", degree=2)
        (b,) = forms("β", degree=3)
        assert degree_of(wedge(a, b)) == Degree.const(5)

    def test_wedge_flattens(self):
        a, b, c = forms("α β γ", degree=1)
        w = wedge(wedge(a, b), c)
        assert isinstance(w, Wedge)
        assert len(w.children) == 3

    def test_wedge_absorbs_zero(self):
        (a,) = forms("α", degree=1)
        assert wedge(a, Zero) == Zero

    def test_wedge_repr(self):
        a, b = forms("α β", degree=1)
        assert wedge(a, b)._repr_inner() == "(α ∧ β)"


# --------------------------------------------------------------------- #
# Tensor product ⊗                                                     #
# --------------------------------------------------------------------- #


class TestTensorProduct:
    def test_tp_of_two(self):
        a, b = forms("α β", degree=1)
        t = tensor_product(a, b)
        assert isinstance(t, TensorProduct)

    def test_tp_degree_law(self):
        """|a⊗b| = |a| + |b|."""
        a, b = forms("α β", degree=1)
        assert degree_of(tensor_product(a, b)) == Degree.const(2)

    def test_tp_mixed(self):
        (a,) = forms("α", degree=2)
        (pi,) = p_vectors("π", degree=3)
        assert degree_of(tensor_product(a, pi)) == Degree.const(5)

    def test_tp_flattens(self):
        a, b, c = forms("α β γ", degree=1)
        t = tensor_product(tensor_product(a, b), c)
        assert isinstance(t, TensorProduct)
        assert len(t.children) == 3

    def test_tp_single_collapses(self):
        (a,) = forms("α", degree=1)
        assert tensor_product(a) == a

    def test_tp_repr(self):
        a, b = forms("α β", degree=1)
        assert tensor_product(a, b)._repr_inner() == "α ⊗ β"

    def test_tp_not_commutative_structurally(self):
        a, b = forms("α β", degree=1)
        assert tensor_product(a, b) != tensor_product(b, a)

    def test_tp_requires_two(self):
        with pytest.raises(ValueError):
            TensorProduct(Symbol("a"))


# --------------------------------------------------------------------- #
# (Anti)symmetrization                                                 #
# --------------------------------------------------------------------- #


class TestSymmetrization:
    def test_symmetrize_node(self):
        (T,) = forms("T", degree=2)
        s = symmetrize(T)
        assert isinstance(s, Symmetrization)
        assert s.arg is T

    def test_antisymmetrize_node(self):
        (T,) = forms("T", degree=2)
        a = antisymmetrize(T)
        assert isinstance(a, Antisymmetrization)
        assert a.arg is T

    def test_symmetrize_preserves_degree(self):
        (T,) = forms("T", degree=3)
        assert degree_of(symmetrize(T)) == Degree.const(3)

    def test_antisymmetrize_preserves_degree(self):
        (T,) = forms("T", degree=3)
        assert degree_of(antisymmetrize(T)) == Degree.const(3)

    def test_sym_repr(self):
        (T,) = forms("T", degree=2)
        assert symmetrize(T)._repr_inner() == "Sym(T)"

    def test_alt_repr(self):
        (T,) = forms("T", degree=2)
        assert antisymmetrize(T)._repr_inner() == "Alt(T)"

    def test_sym_vs_alt_distinct(self):
        (T,) = forms("T", degree=2)
        assert symmetrize(T) != antisymmetrize(T)

    def test_sym_rejects_non_expr(self):
        with pytest.raises(TypeError):
            symmetrize("not an expr")
