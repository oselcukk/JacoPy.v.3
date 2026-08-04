"""Central code — frame (e_a) + coframe (e^a) + duality (Phase 1.H)."""

import pytest

from jacopy.algebra.derivation import degree_of
from jacopy.core.expr import One, Symbol
from jacopy.core.pairing import Pairing
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import (
    Bundle,
    TM,
    Frame,
    Coframe,
    FrameField,
    CoframeField,
    VectorField,
    Form,
    frame,
    coframe,
    kronecker_delta,
    tangent_bundle,
)


# --------------------------------------------------------------------- #
# Frame (e_a)                                                          #
# --------------------------------------------------------------------- #


class TestFrame:
    def test_default_name(self):
        assert frame().name == "e"

    def test_default_bundle(self):
        assert frame().bundle == TM

    def test_field_is_vector_field(self):
        e_a = frame().field("a")
        assert isinstance(e_a, FrameField)
        assert isinstance(e_a, VectorField)

    def test_field_degree_zero(self):
        assert frame().field("a").degree == Degree.const(0)

    def test_field_repr(self):
        assert frame().field("a")._repr_inner() == "e_a"

    def test_field_carries_index(self):
        assert frame().field("a").index == "a"

    def test_field_integer_index(self):
        assert frame().field(1)._repr_inner() == "e_1"

    def test_distinct_indices_distinct(self):
        e = frame()
        assert e.field("a") != e.field("b")

    def test_fields_multiple(self):
        e_a, e_b, e_c = frame().fields("a b c")
        assert [f.index for f in (e_a, e_b, e_c)] == ["a", "b", "c"]

    def test_custom_bundle(self):
        E = Bundle("E")
        assert frame(bundle=E).field("a").bundle == E

    def test_rejects_empty_index(self):
        with pytest.raises(ValueError):
            frame().field("")

    def test_rejects_bad_index_type(self):
        with pytest.raises(TypeError):
            frame().field(True)


# --------------------------------------------------------------------- #
# Coframe (e^a)                                                        #
# --------------------------------------------------------------------- #


class TestCoframe:
    def test_field_is_form(self):
        e_up = coframe().field("a")
        assert isinstance(e_up, CoframeField)
        assert isinstance(e_up, Form)

    def test_coframe_field_degree_one(self):
        assert coframe().field("a").degree == Degree.const(1)

    def test_coframe_field_repr(self):
        assert coframe().field("a")._repr_inner() == "e^a"

    def test_coframe_field_index(self):
        assert coframe().field("a").index == "a"

    def test_coframe_fields_multiple(self):
        a, b = coframe().fields("a b")
        assert (a.index, b.index) == ("a", "b")


# --------------------------------------------------------------------- #
# Frame ↔ Coframe duality                                              #
# --------------------------------------------------------------------- #


class TestDuality:
    def test_frame_dual_is_coframe(self):
        assert isinstance(frame().dual(), Coframe)

    def test_coframe_dual_is_frame(self):
        assert isinstance(coframe().dual(), Frame)

    def test_dual_roundtrip(self):
        e = frame("e", bundle=tangent_bundle(dim=4))
        assert e.dual().dual() == e

    def test_pairing_is_pairing_node(self):
        p = frame().pairing("a", "b")
        assert isinstance(p, Pairing)

    def test_pairing_scalar(self):
        """⟨e^a, e_b⟩ is a scalar (degree 0)."""
        assert degree_of(frame().pairing("a", "b")) == Degree.const(0)

    def test_pairing_uses_dual_field(self):
        p = frame().pairing("a", "b")
        # upper-index coframe field + lower-index frame field
        assert isinstance(p._alpha, CoframeField)
        assert isinstance(p._X, FrameField)

    def test_kronecker_same_index_is_one(self):
        assert kronecker_delta("a", "a") is One

    def test_kronecker_distinct_is_delta_atom(self):
        from jacopy.central.objects.frame import KroneckerDelta

        d = kronecker_delta("a", "b")
        assert isinstance(d, KroneckerDelta)
        assert d == kronecker_delta("a", "b")
        assert d._repr_inner() == "δ^a_b"

    def test_kronecker_is_constant_scalar(self):
        """δ carries degree 0 and the is_constant marker: X(δ) = 0."""
        from jacopy.algebra.derivation import Act, degree_of
        from jacopy.algorithms.product_rule import product_rule
        from jacopy.core.expr import Integer as _Int
        from jacopy.core.symbolic_degree import Degree
        from jacopy.central.objects import vector_fields

        d = kronecker_delta("a", "b")
        assert degree_of(d) == Degree.const(0)
        (X,) = vector_fields("X")
        assert product_rule(Act(X, d)) == _Int(0)
