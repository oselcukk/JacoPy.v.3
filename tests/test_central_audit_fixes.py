"""Central code — Phase 1 audit fixes.

Pins the corrections from the PDF-conformance / mathematical audit:

1. ``Form.__call__`` arity guard (a p-form with concrete ``p`` needs
   exactly ``p`` arguments; partial contraction is ``ι_X ω``).
2. ``signature_of`` — (q, r) tracking through ``∇_X``, ``fT``, Neg,
   Sym/Alt (PDF item 8n type preservation).
3. ``CovariantOp`` carries the connection's bundle in its identity.
4. ``hodge`` rejects out-of-range degrees on concrete dimensions.
5. ``tilde_contract`` routes a plain vector field to the canonical
   pairing (grading contact point ``ι̃_ω X = ⟨ω, X⟩``).
6. ``PVector.__call__`` / ``Tensor.__call__`` full evaluations.
"""

import pytest

from jacopy.algebra.derivation import Act, degree_of
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import (
    Bundle,
    Connection,
    Tensor,
    connection,
    contract,
    forms,
    hodge,
    kronecker_delta,
    metric,
    p_vectors,
    bivector,
    scale,
    signature_of,
    symmetrize,
    tangent_bundle,
    tensors,
    tilde_contract,
    vector_fields,
)
from jacopy.core.expr import Neg, One


# --------------------------------------------------------------------- #
# 1. Form arity guard                                                   #
# --------------------------------------------------------------------- #


class TestFormArityGuard:
    def test_one_form_single_arg_is_pairing(self):
        (w,) = forms("ω", degree=1)
        (X,) = vector_fields("X")
        assert isinstance(w(X), Pairing)

    def test_two_form_single_arg_rejected(self):
        """ω(X) for a 2-form is NOT a scalar pairing — must raise."""
        (w,) = forms("ω", degree=2)
        (X,) = vector_fields("X")
        with pytest.raises(TypeError):
            w(X)

    def test_two_form_partial_contraction_via_interior(self):
        """The correct partial contraction ι_X ω has degree p − 1."""
        (w,) = forms("ω", degree=2)
        (X,) = vector_fields("X")
        assert degree_of(contract(X, w)) == Degree.const(1)

    def test_two_form_full_evaluation(self):
        (w,) = forms("ω", degree=2)
        X, Y = vector_fields("X Y")
        result = w(X, Y)
        assert isinstance(result, MultiEval)
        assert degree_of(result) == Degree.const(0)

    def test_two_form_three_args_rejected(self):
        (w,) = forms("ω", degree=2)
        X, Y, Z = vector_fields("X Y Z")
        with pytest.raises(TypeError):
            w(X, Y, Z)

    def test_symbolic_degree_unenforced(self):
        """Symbolic p: arity unenforced, alternating MultiEval built."""
        p = Degree.var("p")
        (w,) = forms("ω", degree=p)
        X, Y = vector_fields("X Y")
        result = w(X, Y)
        assert isinstance(result, MultiEval)
        assert result.alternating is True


# --------------------------------------------------------------------- #
# 2. signature_of — (q, r) tracking (PDF item 8n)                       #
# --------------------------------------------------------------------- #


class TestSignatureOf:
    def test_tensor(self):
        (T,) = tensors("T", upper=1, lower=2)
        assert signature_of(T) == (1, 2)

    def test_vector_field(self):
        (X,) = vector_fields("X")
        assert signature_of(X) == (1, 0)

    def test_form_concrete(self):
        (w,) = forms("ω", degree=3)
        assert signature_of(w) == (0, 3)

    def test_pvector_concrete(self):
        pi = bivector("π")
        assert signature_of(pi) == (2, 0)

    def test_metric_and_inverse(self):
        g = metric()
        assert signature_of(g) == (0, 2)
        assert signature_of(g.inverse()) == (2, 0)

    def test_symbolic_degree_is_none(self):
        p = Degree.var("p")
        (w,) = forms("ω", degree=p)
        assert signature_of(w) is None

    def test_covariant_derivative_preserves_signature(self):
        """PDF 8n: ∇_X maps a (q, r)-tensor to a (q, r)-tensor."""
        (T,) = tensors("T", upper=1, lower=2)
        (X,) = vector_fields("X")
        assert signature_of(connection()(X, T)) == (1, 2)

    def test_nested_covariant_preserves_signature(self):
        (T,) = tensors("T", upper=2, lower=1)
        X, Y = vector_fields("X Y")
        nabla = connection()
        assert signature_of(nabla(X, nabla(Y, T))) == (2, 1)

    def test_scale_preserves_signature(self):
        """fT (item 8q) keeps the (q, r) shape."""
        from jacopy.core.expr import Symbol
        (T,) = tensors("T", upper=0, lower=2)
        f = Symbol("f")
        assert signature_of(scale(f, T)) == (0, 2)

    def test_neg_and_symmetrize_preserve(self):
        (T,) = tensors("T", upper=1, lower=1)
        assert signature_of(Neg(T)) == (1, 1)
        assert signature_of(symmetrize(T)) == (1, 1)

    def test_ambiguous_product_is_none(self):
        from jacopy.core.expr import Product
        S, T = tensors("S T", upper=1, lower=1)
        assert signature_of(Product(S, T)) is None

    def test_unknown_is_none(self):
        from jacopy.core.expr import Symbol
        assert signature_of(Symbol("f")) is None


# --------------------------------------------------------------------- #
# 3. CovariantOp bundle identity                                        #
# --------------------------------------------------------------------- #


class TestCovariantOpBundle:
    def test_same_name_different_bundle_distinct(self):
        (X,) = vector_fields("X")
        E = Bundle("E")
        assert Connection("∇").op(X) != Connection("∇", bundle=E).op(X)

    def test_same_bundle_equal(self):
        (X,) = vector_fields("X")
        assert Connection("∇").op(X) == Connection("∇").op(X)

    def test_op_carries_bundle(self):
        (X,) = vector_fields("X")
        E = Bundle("E")
        assert Connection("∇", bundle=E).op(X).bundle == E


# --------------------------------------------------------------------- #
# 4. Hodge degree-range guard                                           #
# --------------------------------------------------------------------- #


class TestHodgeGuard:
    def test_in_range_ok(self):
        g = metric(bundle=tangent_bundle(dim=4))
        (w,) = forms("ω", degree=4)
        assert degree_of(hodge(w, g)) == Degree.const(0)

    def test_over_dimension_rejected(self):
        g = metric(bundle=tangent_bundle(dim=2))
        (w,) = forms("η", degree=5)
        with pytest.raises(ValueError):
            hodge(w, g)

    def test_symbolic_dim_unchecked(self):
        """Symbolic n: no range check (abstract proof mode)."""
        g = metric()
        (w,) = forms("η", degree=5)
        assert degree_of(hodge(w, g)) == Degree.var("n") - Degree.const(5)

    def test_undeterminable_degree_unchecked(self):
        from jacopy.core.expr import Symbol
        g = metric(bundle=tangent_bundle(dim=3))
        # Symbol has no determinable degree: guard skipped, node built.
        assert hodge(Symbol("f"), g) is not None


# --------------------------------------------------------------------- #
# 5. tilde_contract grading contact point                               #
# --------------------------------------------------------------------- #


class TestTildeContractVectorField:
    def test_vector_field_routes_to_pairing(self):
        """ι̃_α X = ⟨α, X⟩ — a scalar, not a degree −1 Act."""
        (a,) = forms("α", degree=1)
        (X,) = vector_fields("X")
        result = tilde_contract(a, X)
        assert isinstance(result, Pairing)
        assert degree_of(result) == Degree.const(0)

    def test_pvector_stays_act(self):
        (a,) = forms("α", degree=1)
        (pi,) = p_vectors("π", degree=3)
        result = tilde_contract(a, pi)
        assert isinstance(result, Act)
        assert degree_of(result) == Degree.const(2)


# --------------------------------------------------------------------- #
# 6. PVector / Tensor full evaluation                                   #
# --------------------------------------------------------------------- #


class TestPVectorCall:
    def test_bivector_two_covectors(self):
        pi = bivector("π")
        a, b = forms("α β", degree=1)
        result = pi(a, b)
        assert isinstance(result, MultiEval)
        assert result.slot_kind == "covector"
        assert result.alternating is True
        assert degree_of(result) == Degree.const(0)

    def test_bivector_single_arg_rejected(self):
        pi = bivector("π")
        (a,) = forms("α", degree=1)
        with pytest.raises(TypeError):
            pi(a)

    def test_one_vector_single_arg_is_pairing(self):
        (v,) = p_vectors("v", degree=1)
        (a,) = forms("α", degree=1)
        assert isinstance(v(a), Pairing)


class TestTensorCall:
    def test_full_evaluation(self):
        """T(α, X, Y) for a (1, 2)-tensor — mixed slots, scalar."""
        (T,) = tensors("T", upper=1, lower=2)
        (a,) = forms("α", degree=1)
        X, Y = vector_fields("X Y")
        result = T(a, X, Y)
        assert isinstance(result, MultiEval)
        assert result.slot_kind == "mixed"
        assert result.alternating is False
        assert degree_of(result) == Degree.const(0)

    def test_wrong_arity_rejected(self):
        (T,) = tensors("T", upper=1, lower=2)
        (a,) = forms("α", degree=1)
        with pytest.raises(TypeError):
            T(a)

    def test_kronecker_same_label_is_one(self):
        # Fixed-index reading: equal labels denote the same index.
        assert kronecker_delta("a", "a") is One
