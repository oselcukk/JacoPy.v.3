"""Central code — p-vector + (q,r)-tensor + fT tests (Phase 1.C)."""

import pytest

from jacopy.algebra.derivation import degree_of
from jacopy.core.expr import Product
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import (
    Bundle,
    TM,
    PVector,
    p_vectors,
    bivector,
    Tensor,
    tensors,
    scale,
    functions,
)


# --------------------------------------------------------------------- #
# PVector                                                              #
# --------------------------------------------------------------------- #


class TestPVector:
    def test_p_vectors_returns(self):
        a, b = p_vectors("π ρ", degree=2)
        assert isinstance(a, PVector) and isinstance(b, PVector)

    def test_degree(self):
        (pi,) = p_vectors("π", degree=2)
        assert pi.degree == Degree.const(2)

    def test_degree_of_no_registry(self):
        """A p-vector carries its own degree (generic Atom+degree)."""
        (pi,) = p_vectors("π", degree=3)
        assert degree_of(pi) == Degree.const(3)

    def test_symbolic_degree(self):
        p = Degree.var("p")
        (pi,) = p_vectors("π", degree=p)
        assert pi.degree == p

    def test_default_bundle(self):
        (pi,) = p_vectors("π", degree=2)
        assert pi.bundle == TM

    def test_custom_bundle(self):
        E = Bundle("E", dim=6)
        (pi,) = p_vectors("π", degree=2, bundle=E)
        assert pi.bundle == E

    def test_bivector_shortcut(self):
        pi = bivector("π")
        assert isinstance(pi, PVector)
        assert pi.degree == Degree.const(2)

    def test_bivector_single_name(self):
        with pytest.raises(ValueError):
            bivector("π ρ")

    def test_equality(self):
        (a,) = p_vectors("π", degree=2)
        (b,) = p_vectors("π", degree=2)
        assert a == b and hash(a) == hash(b)

    def test_diff_degree_distinct(self):
        (a,) = p_vectors("π", degree=2)
        (b,) = p_vectors("π", degree=3)
        assert a != b


# --------------------------------------------------------------------- #
# Tensor (q, r)                                                        #
# --------------------------------------------------------------------- #


class TestTensor:
    def test_tensors_returns(self):
        S, T = tensors("S T", upper=1, lower=2)
        assert isinstance(S, Tensor) and isinstance(T, Tensor)

    def test_signature(self):
        (T,) = tensors("T", upper=1, lower=2)
        assert T.upper == 1
        assert T.lower == 2
        assert T.signature == (1, 2)

    def test_default_bundle(self):
        (T,) = tensors("T", upper=2, lower=0)
        assert T.bundle == TM

    def test_custom_bundle(self):
        E = Bundle("E")
        (T,) = tensors("T", upper=1, lower=1, bundle=E)
        assert T.bundle == E

    def test_equality(self):
        (a,) = tensors("T", upper=1, lower=2)
        (b,) = tensors("T", upper=1, lower=2)
        assert a == b and hash(a) == hash(b)

    def test_diff_signature_distinct(self):
        (a,) = tensors("T", upper=1, lower=2)
        (b,) = tensors("T", upper=2, lower=1)
        assert a != b

    def test_rejects_negative_upper(self):
        with pytest.raises(ValueError):
            Tensor("T", upper=-1, lower=0)

    def test_rejects_negative_lower(self):
        with pytest.raises(ValueError):
            Tensor("T", upper=0, lower=-1)

    def test_zero_tensor_is_function_like(self):
        """A (0,0)-tensor is like a scalar field — valid."""
        (T,) = tensors("T", upper=0, lower=0)
        assert T.signature == (0, 0)


# --------------------------------------------------------------------- #
# fT — scaling by a function (item 8q)                                 #
# --------------------------------------------------------------------- #


class TestScale:
    def test_scale_is_product(self):
        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        (T,) = tensors("T", upper=1, lower=1)
        fT = scale(f, T)
        assert isinstance(fT, Product)
        assert fT.children == (f, T)

    def test_scale_form(self):
        from jacopy.central.objects import forms
        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        (w,) = forms("ω", degree=1)
        fw = scale(f, w)
        assert isinstance(fw, Product)

    def test_scale_pvector(self):
        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        (pi,) = p_vectors("π", degree=2)
        assert isinstance(scale(f, pi), Product)

    def test_scale_repr(self):
        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        (T,) = tensors("T", upper=1, lower=1)
        assert scale(f, T)._repr_inner() == "(f * T)"

    def test_scale_rejects_non_expr(self):
        (T,) = tensors("T", upper=1, lower=1)
        with pytest.raises(TypeError):
            scale("f", T)
