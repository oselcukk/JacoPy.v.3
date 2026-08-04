"""Central code — connection ∇ + covariant derivative (Phase 1.G)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import (
    Bundle,
    TM,
    Connection,
    CovariantOp,
    connection,
    covariant_derivative,
    vector_fields,
    forms,
    functions,
    tensors,
)
from jacopy.core.registry import PropertyRegistry


# --------------------------------------------------------------------- #
# Connection object                                                     #
# --------------------------------------------------------------------- #


class TestConnection:
    def test_default_name(self):
        assert connection().name == "∇"

    def test_default_bundle(self):
        assert connection().bundle == TM

    def test_custom_bundle(self):
        E = Bundle("E")
        assert connection(bundle=E).bundle == E

    def test_equality(self):
        assert connection() == connection()

    def test_diff_bundle_distinct(self):
        E = Bundle("E")
        assert connection() != connection(bundle=E)

    def test_rejects_bad_bundle(self):
        with pytest.raises(TypeError):
            Connection("∇", bundle="nope")


# --------------------------------------------------------------------- #
# The ∇_X operator                                                      #
# --------------------------------------------------------------------- #


class TestCovariantOp:
    def test_op_is_derivation(self):
        (X,) = vector_fields("X")
        op = connection().op(X)
        assert isinstance(op, CovariantOp)
        assert isinstance(op, Derivation)

    def test_op_degree_zero(self):
        """∇_X has degree 0 (it preserves the degree)."""
        (X,) = vector_fields("X")
        assert connection().op(X).degree == Degree.const(0)

    def test_op_repr(self):
        (X,) = vector_fields("X")
        assert connection().op(X)._repr_inner() == "∇_X"

    def test_op_carries_vector(self):
        (X,) = vector_fields("X")
        assert connection().op(X).vector is X

    def test_distinct_directions_distinct(self):
        X, Y = vector_fields("X Y")
        assert connection().op(X) != connection().op(Y)


# --------------------------------------------------------------------- #
# ∇_X T — covariant derivative, degree preserved                       #
# --------------------------------------------------------------------- #


class TestCovariantDerivative:
    def test_nabla_is_act(self):
        (X,) = vector_fields("X")
        (Y,) = vector_fields("Y")
        result = connection()(X, Y)
        assert isinstance(result, Act)
        assert isinstance(result.op, CovariantOp)

    def test_nabla_vector_is_vector(self):
        """∇_X Y: vector → vector (degree 0)."""
        X, Y = vector_fields("X Y")
        assert degree_of(connection()(X, Y)) == Degree.const(0)

    def test_nabla_one_form_preserves_degree(self):
        """∇_X ω: 1-form → 1-form (degree 1)."""
        (X,) = vector_fields("X")
        (w,) = forms("ω", degree=1)
        assert degree_of(connection()(X, w)) == Degree.const(1)

    def test_nabla_p_form_preserves_degree(self):
        """∇_X ω: p-form → p-form (symbolic p is preserved)."""
        (X,) = vector_fields("X")
        p = Degree.var("p")
        (w,) = forms("ω", degree=p)
        assert degree_of(connection()(X, w)) == p

    def test_nabla_function_scalar(self):
        """∇_X f: function → function (degree 0)."""
        reg = PropertyRegistry()
        (X,) = vector_fields("X")
        (f,) = functions("f", registry=reg)
        assert degree_of(connection()(X, f), reg) == Degree.const(0)

    def test_nabla_repr(self):
        X, Y = vector_fields("X Y")
        assert connection()(X, Y)._repr_inner() == "∇_X(Y)"

    def test_covariant_derivative_helper(self):
        X, Y = vector_fields("X Y")
        nabla = connection()
        assert covariant_derivative(nabla, X, Y) == nabla(X, Y)

    def test_nested_covariant(self):
        """∇_X ∇_Y Z — nested covariant derivative."""
        X, Y, Z = vector_fields("X Y Z")
        nabla = connection()
        inner = nabla(Y, Z)         # ∇_Y Z
        outer = nabla(X, inner)     # ∇_X(∇_Y Z)
        assert isinstance(outer, Act)
        assert degree_of(outer) == Degree.const(0)

    def test_helper_rejects_non_connection(self):
        X, Y = vector_fields("X Y")
        with pytest.raises(TypeError):
            covariant_derivative("not conn", X, Y)

    def test_nabla_rejects_non_expr(self):
        (X,) = vector_fields("X")
        with pytest.raises(TypeError):
            connection()(X, "not expr")
