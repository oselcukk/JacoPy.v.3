"""Phase 7.B.2b — Poisson generalized geometry (PDF item 14f,
Watamura): the tilde-Dorfman double (TM)₀ ⊕ (T*M)_θ, its
tilde-Courant companion, and the axiom suite at p = 1."""

import pytest

from jacopy.core.expr import Integer
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.generalized.poisson_generalized import (
    d_operator_theta,
    poisson_base,
    prove_theta_anchor_morphism,
    prove_theta_courant_skew,
    prove_theta_d_pairing_value,
    prove_theta_invariance,
    theta_anchor,
    theta_courant,
    theta_dorfman,
)
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, h = functions("f h", registry=reg)
    om, et, ze = forms("ω η ζ", degree=1)
    U, V, W = vector_fields("U V W")
    (X,) = vector_fields("X")
    return reg, f, h, om, et, ze, U, V, W, X, poisson_base()


def test_theta_anchor_morphism_c2(setup):
    # THE 6.I.4 capstone at p = 1, Watamura framing.
    reg, f, h, om, et, ze, U, V, W, X, N = setup
    chain = prove_theta_anchor_morphism(
        N, U, om, V, et, h, registry=reg
    )
    assert len(chain.steps) > 20


def test_theta_anchor_morphism_needs_poisson(setup):
    reg, f, h, om, et, ze, U, V, W, X, N = setup
    with pytest.raises(ProofFailure):
        prove_theta_anchor_morphism(
            N, U, om, V, et, h,
            registry=reg,
            declare_poisson=False,
        )


def test_theta_courant_skew(setup):
    reg, f, h, om, et, ze, U, V, W, X, N = setup
    chain, thm = prove_theta_courant_skew(
        N, U, om, V, et, registry=reg
    )
    assert len(chain.steps) == 2


def test_theta_d_pairing_is_half_anchor(setup):
    reg, f, h, om, et, ze, U, V, W, X, N = setup
    chain, thm = prove_theta_d_pairing_value(
        N, U, om, h, registry=reg
    )
    assert "½ρ(x)(h)" in thm.statement


def test_theta_right_leibniz_c3_at_p1(setup):
    # [C'3] — the 6.D theorem instantiated at p = 1 (any bivector).
    from jacopy.packages.drinfeld.double import (
        prove_nambu_double_right_leibniz_form,
        prove_nambu_double_right_leibniz_vec,
    )

    reg, f, h, om, et, ze, U, V, W, X, N = setup
    assert prove_nambu_double_right_leibniz_form(
        N, U, om, V, et, f, (X,), registry=reg
    ).steps
    assert prove_nambu_double_right_leibniz_vec(
        N, U, om, V, et, f, ze, registry=reg
    ).steps


def test_theta_symmetric_part_c4_at_p1(setup):
    # [C'4] — the 6.D symmetric part (𝒟 = d + d̃) at p = 1.
    from jacopy.packages.drinfeld.double import (
        prove_nambu_double_symmetric_part_form,
        prove_nambu_double_symmetric_part_vec,
    )

    reg, f, h, om, et, ze, U, V, W, X, N = setup
    assert prove_nambu_double_symmetric_part_form(
        N, U, om, V, et, (X,), registry=reg
    ).steps
    assert prove_nambu_double_symmetric_part_vec(
        N, U, om, V, et, ze, registry=reg
    ).steps


def test_theta_invariance_is_a_diagnosed_open_slice(setup):
    # [C'5] honest-fail pin (2026-09-08): the identity is verified
    # on paper but the mixed-language residual sits at a stuck
    # rewriting fixpoint — see the prover docstring and the ROADMAP
    # 14f.i deferral. This pin flips when the confluence work lands.
    reg, f, h, om, et, ze, U, V, W, X, N = setup
    with pytest.raises(ProofFailure, match="residual"):
        prove_theta_invariance(
            N, U, om, V, et, W, ze, registry=reg
        )


def test_theta_structures_reject_higher_order():
    from jacopy.packages.poisson.nambu import nambu_structure

    N2 = nambu_structure(p=2)
    U, V = vector_fields("U V")
    om, et = forms("ω η", degree=2)
    with pytest.raises(ValueError, match="order-1"):
        theta_dorfman(N2, U, om, V, et)
