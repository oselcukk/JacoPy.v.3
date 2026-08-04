"""Central code — basic object tests (Phase 1.A).

Bundle + Function + VectorField + ``U(f)``.
"""

import pytest

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import (
    Bundle,
    TangentBundle,
    tangent_bundle,
    TM,
    functions,
    VectorField,
    vector_fields,
    apply_to_function,
)


# --------------------------------------------------------------------- #
# Bundle                                                                #
# --------------------------------------------------------------------- #


class TestBundle:
    def test_tm_is_tangent(self):
        assert TM.is_tangent
        assert TM.name == "TM"

    def test_tm_symbolic_dim_default(self):
        assert TM.dim is None

    def test_tangent_bundle_with_dim(self):
        b = tangent_bundle(dim=4)
        assert b.dim == 4
        assert b.is_tangent

    def test_bundle_value_equality(self):
        assert tangent_bundle() == tangent_bundle()
        assert tangent_bundle(dim=4) == tangent_bundle(dim=4)
        assert tangent_bundle(dim=4) != tangent_bundle(dim=3)

    def test_bundle_hashable(self):
        assert hash(tangent_bundle()) == hash(tangent_bundle())

    def test_generic_bundle(self):
        E = Bundle("E", dim=6)
        assert not E.is_tangent
        assert E.name == "E"
        assert E.dim == 6

    def test_bundle_rejects_bad_name(self):
        with pytest.raises(ValueError):
            Bundle("")

    def test_bundle_rejects_bad_dim(self):
        with pytest.raises(ValueError):
            Bundle("E", dim=0)

    def test_generic_vs_tangent_distinct(self):
        assert Bundle("TM") != tangent_bundle()  # different kind


# --------------------------------------------------------------------- #
# Functions                                                            #
# --------------------------------------------------------------------- #


class TestFunctions:
    def test_returns_symbols(self):
        reg = PropertyRegistry()
        f, g = functions("f g", registry=reg)
        assert isinstance(f, Symbol)
        assert isinstance(g, Symbol)

    def test_single_is_tuple(self):
        reg = PropertyRegistry()
        result = functions("f", registry=reg)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_degree_zero_default(self):
        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        assert degree_of(f, reg) == Degree.const(0)

    def test_registry_declares_graded(self):
        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        assert reg.get(f, Graded).degree == Degree.const(0)

    def test_rejects_empty(self):
        reg = PropertyRegistry()
        with pytest.raises(ValueError):
            functions("", registry=reg)


# --------------------------------------------------------------------- #
# VectorField                                                          #
# --------------------------------------------------------------------- #


class TestVectorField:
    def test_returns_vector_fields(self):
        U, V, W = vector_fields("U V W")
        assert all(isinstance(x, VectorField) for x in (U, V, W))

    def test_is_derivation(self):
        (U,) = vector_fields("U")
        assert isinstance(U, Derivation)

    def test_degree_zero(self):
        (U,) = vector_fields("U")
        assert U.degree == Degree.const(0)

    def test_degree_of_no_registry(self):
        """Since VectorField is a Derivation, degree_of needs no registry."""
        (U,) = vector_fields("U")
        assert degree_of(U) == Degree.const(0)

    def test_default_bundle_is_tm(self):
        (U,) = vector_fields("U")
        assert U.bundle == TM
        assert U.bundle.is_tangent

    def test_custom_bundle(self):
        E = Bundle("E", dim=6)
        (u,) = vector_fields("u", bundle=E)
        assert u.bundle == E

    def test_same_name_different_bundle_distinct(self):
        E = Bundle("E")
        (U_tm,) = vector_fields("U")
        (U_e,) = vector_fields("U", bundle=E)
        assert U_tm != U_e

    def test_same_name_same_bundle_equal(self):
        (U1,) = vector_fields("U")
        (U2,) = vector_fields("U")
        assert U1 == U2
        assert hash(U1) == hash(U2)

    def test_single_is_tuple(self):
        result = vector_fields("U")
        assert isinstance(result, tuple) and len(result) == 1

    def test_rejects_non_bundle(self):
        with pytest.raises(TypeError):
            VectorField("U", bundle="not a bundle")


# --------------------------------------------------------------------- #
# U(f) — action of a vector field on a function                         #
# --------------------------------------------------------------------- #


class TestVectorFieldAction:
    def test_U_of_f_is_act(self):
        reg = PropertyRegistry()
        (U,) = vector_fields("U")
        (f,) = functions("f", registry=reg)
        expr = U(f)
        assert isinstance(expr, Act)
        assert expr.op is U
        assert expr.arg is f

    def test_U_of_f_repr(self):
        reg = PropertyRegistry()
        (U,) = vector_fields("U")
        (f,) = functions("f", registry=reg)
        assert U(f)._repr_inner() == "U(f)"

    def test_apply_to_function_helper(self):
        reg = PropertyRegistry()
        (U,) = vector_fields("U")
        (f,) = functions("f", registry=reg)
        assert apply_to_function(U, f) == U(f)

    def test_nested_action(self):
        """U(V(f)) — nested directional derivative."""
        reg = PropertyRegistry()
        U, V = vector_fields("U V")
        (f,) = functions("f", registry=reg)
        expr = U(V(f))
        assert isinstance(expr, Act)
        assert expr.op is U
        assert isinstance(expr.arg, Act)
        assert expr.arg.op is V
        assert expr.arg._repr_inner() == "V(f)"

    def test_U_of_f_scalar_degree(self):
        """U(f): degree 0 (op) + degree 0 (f) = 0 — still a scalar."""
        reg = PropertyRegistry()
        (U,) = vector_fields("U")
        (f,) = functions("f", registry=reg)
        assert degree_of(U(f), reg) == Degree.const(0)

    def test_apply_rejects_non_vector_field(self):
        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        with pytest.raises(TypeError):
            apply_to_function(f, f)
