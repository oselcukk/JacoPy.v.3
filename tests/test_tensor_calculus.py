"""PDF items 8n / 9e — the evaluation laws of ℒ and ∇ on general
(q, r)-tensors (2026-09-08 audit, compliance finding 2): the mixed
identities close, the non-alternating soundness pin stays red."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import (
    connection,
    forms,
    tensors,
    vector_fields,
)
from jacopy.central.tangent import tangent_engine
from jacopy.central.tangent.cartan import L
from jacopy.packages.metric_affine import metric_affine_engine
from jacopy.proof import show_equal
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    X, Y = vector_fields("X Y")
    (alpha,) = forms("alpha", degree=1)
    return reg, X, Y, alpha


def test_lie_derivative_of_mixed_tensor(setup):
    # (ℒ_X T)(α, Y) = X(T(α,Y)) − T(ℒ_Xα, Y) − T(α, ℒ_XY)
    reg, X, Y, alpha = setup
    (T,) = tensors("T", upper=1, lower=1)
    lhs = MultiEval(
        L(X, T), alpha, Y, alternating=False, slot_kind="mixed"
    )
    rhs = (
        Act(X, T(alpha, Y))
        - T(L(X, alpha), Y)
        - T(alpha, L(X, Y))
    )
    assert show_equal(
        lhs, rhs, registry=reg, engine=tangent_engine(registry=reg)
    ).steps


def test_covariant_derivative_of_mixed_tensor(setup):
    # (∇_X T)(α, Y) = X(T(α,Y)) − T(∇_Xα, Y) − T(α, ∇_XY)
    reg, X, Y, alpha = setup
    (T,) = tensors("T", upper=1, lower=1)
    conn = connection()
    lhs = MultiEval(
        conn(X, T),
        alpha,
        Y,
        alternating=False,
        slot_kind="mixed",
    )
    rhs = (
        Act(X, T(alpha, Y))
        - T(conn(X, alpha), Y)
        - T(alpha, conn(X, Y))
    )
    assert show_equal(
        lhs,
        rhs,
        registry=reg,
        engine=metric_affine_engine(registry=reg),
    ).steps


def test_general_covariant_tensor_still_not_alternating(setup):
    # The reliability-pass pin must SURVIVE the new unroll:
    # (ℒ_X T)(Y, Y) ≠ 0 for a general (0,2) tensor.
    reg, X, Y, alpha = setup
    (T,) = tensors("T", upper=0, lower=2)
    lhs = MultiEval(
        L(X, T), Y, Y, alternating=False, slot_kind="mixed"
    )
    with pytest.raises(ProofFailure):
        show_equal(
            lhs,
            Integer(0),
            registry=reg,
            engine=tangent_engine(registry=reg),
        )


def test_higher_mixed_tensor_unrolls(setup):
    # a (1,2)-tensor: three slot corrections.
    reg, X, Y, alpha = setup
    (Z,) = vector_fields("Z")
    (T,) = tensors("T", upper=1, lower=2)
    lhs = MultiEval(
        L(X, T),
        alpha,
        Y,
        Z,
        alternating=False,
        slot_kind="mixed",
    )
    rhs = (
        Act(X, T(alpha, Y, Z))
        - T(L(X, alpha), Y, Z)
        - T(alpha, L(X, Y), Z)
        - T(alpha, Y, L(X, Z))
    )
    assert show_equal(
        lhs, rhs, registry=reg, engine=tangent_engine(registry=reg)
    ).steps
