"""TM case — Phase 1 deferral rules: ∇_X f = X(f) and ⟨e^a, e_b⟩ = δ^a_b
(Phase 2.B)."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import One, Symbol
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ProofFailure
from jacopy.central.objects import (
    Bundle,
    connection,
    forms,
    frame,
    functions,
    kronecker_delta,
    tensors,
    vector_fields,
)
from jacopy.central.tangent import (
    CovariantScalarActionDefinition,
    FrameDualityDefinition,
    is_scalar_function,
    prove_covariant_scalar_action,
    prove_covariant_leibniz,
    prove_frame_duality,
    tangent_engine,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    X, Y = vector_fields("X Y")
    f, g = functions("f g", registry=reg)
    return reg, X, Y, f, g


# --------------------------------------------------------------------- #
# Scalar-function recognition                                            #
# --------------------------------------------------------------------- #


class TestIsScalarFunction:
    def test_registered_symbol(self, setup):
        reg, _, _, f, _ = setup
        assert is_scalar_function(f, reg)

    def test_unregistered_symbol(self):
        assert not is_scalar_function(Symbol("s"), None)

    def test_vector_field_is_not(self, setup):
        reg, X, *_ = setup
        assert not is_scalar_function(X, reg)

    def test_directional_derivative_is(self, setup):
        reg, X, _, f, _ = setup
        assert is_scalar_function(Act(X, f), reg)

    def test_pairing_is(self, setup):
        reg, X, *_ = setup
        (w,) = forms("ω", degree=1)
        assert is_scalar_function(w(X), reg)

    def test_covariant_derivative_of_vector_is_not(self, setup):
        reg, X, Y, *_ = setup
        assert not is_scalar_function(connection()(X, Y), reg)


# --------------------------------------------------------------------- #
# ∇_X f = X(f)                                                          #
# --------------------------------------------------------------------- #


class TestCovariantScalarAction:
    def test_closes(self, setup):
        reg, X, _, f, _ = setup
        assert prove_covariant_scalar_action(
            connection(), X, f, registry=reg
        ).steps

    def test_nabla_of_vector_stays_inert(self, setup):
        """∇_X Y must NOT rewrite to X(Y)."""
        reg, X, Y, *_ = setup
        expr = connection()(X, Y)
        out, steps = tangent_engine(registry=reg).expand(expr)
        assert out == expr and not steps

    def test_nabla_of_form_stays_inert(self, setup):
        reg, X, *_ = setup
        (w,) = forms("ω", degree=2)
        expr = connection()(X, w)
        out, steps = tangent_engine(registry=reg).expand(expr)
        assert out == expr and not steps

    def test_non_tangent_bundle_inert(self, setup):
        """On a general bundle E the rule waits for the anchor (Phase 3)."""
        reg, X, _, f, _ = setup
        E = Bundle("E")
        rule = CovariantScalarActionDefinition(reg)
        expr = connection(bundle=E)(X, f)
        assert not rule.matches(expr)

    def test_leibniz_closes_for_vector(self, setup):
        reg, X, Y, f, _ = setup
        assert prove_covariant_leibniz(
            connection(), X, f, Y, registry=reg
        ).steps

    def test_leibniz_closes_for_form_and_tensor(self, setup):
        """∇_X(f·T) = X(f)·T + f·∇_X T for a 2-form and a (1,1)-tensor."""
        reg, X, _, f, _ = setup
        (w,) = forms("ω", degree=2)
        (T,) = tensors("T", upper=1, lower=1)
        nabla = connection()
        assert prove_covariant_leibniz(nabla, X, f, w, registry=reg).steps
        assert prove_covariant_leibniz(nabla, X, f, T, registry=reg).steps

    def test_wrong_leibniz_fails(self, setup):
        """Sanity: dropping the X(f)·Y term must not close."""
        from jacopy.core.expr import Product
        from jacopy.proof.strategies import ExpandAndSimplify

        reg, X, Y, f, _ = setup
        nabla = connection()
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                nabla(X, Product(f, Y)),
                Product(f, nabla(X, Y)),
                registry=reg,
                engine=tangent_engine(registry=reg),
            )


# --------------------------------------------------------------------- #
# ⟨e^a, e_b⟩ = δ^a_b                                                    #
# --------------------------------------------------------------------- #


class TestFrameDuality:
    def test_distinct_labels_close_to_delta(self):
        assert prove_frame_duality(frame(), "a", "b").steps

    def test_equal_labels_close_to_one(self):
        assert prove_frame_duality(frame(), "a", "a").steps

    def test_engine_rewrites_pairing(self):
        e = frame()
        out, steps = tangent_engine().expand(e.pairing("a", "b"))
        assert out == kronecker_delta("a", "b")
        assert steps

    def test_equal_label_rewrites_to_one(self):
        e = frame()
        out, _ = tangent_engine().expand(e.pairing("c", "c"))
        assert out is One

    def test_mismatched_frames_inert(self):
        """⟨e^a, f_b⟩ for different frames must not fire."""
        from jacopy.core.pairing import Pairing

        e, other = frame("e"), frame("f")
        rule = FrameDualityDefinition()
        cross = Pairing(e.dual().field("a"), other.field("b"))
        assert not rule.matches(cross)

    def test_mismatched_bundles_inert(self):
        from jacopy.core.pairing import Pairing

        E = Bundle("E")
        e_tm, e_bundle = frame("e"), frame("e", bundle=E)
        rule = FrameDualityDefinition()
        cross = Pairing(e_tm.dual().field("a"), e_bundle.field("b"))
        assert not rule.matches(cross)

    def test_base_name_recovery(self):
        e = frame("θ")
        assert e.field("a").base_name == "θ"
        assert e.dual().field("a").base_name == "θ"
