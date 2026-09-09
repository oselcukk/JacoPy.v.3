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
    # [C'3] for the (2.6)-(2.7) structure itself, with
    # ρ(x)f = (θ♯ω)(f) (2026-09-09 audit, finding 2).
    from jacopy.packages.generalized.poisson_generalized import (
        prove_theta_right_leibniz,
    )

    reg, f, h, om, et, ze, U, V, W, X, N = setup
    chain, thm = prove_theta_right_leibniz(
        N, U, om, V, et, f, registry=reg
    )
    assert "[C'3]" in thm.statement
    assert len(chain.steps) == 2


def test_theta_symmetric_part_c4_at_p1(setup):
    # [C'4] for the (2.6)-(2.7) structure: sym = 2·D_θ⟨x,y⟩₊ with
    # D_θ = ½(−θ♯dh, 0).
    from jacopy.packages.generalized.poisson_generalized import (
        prove_theta_symmetric_part,
    )

    reg, f, h, om, et, ze, U, V, W, X, N = setup
    chain, thm = prove_theta_symmetric_part(
        N, U, om, V, et, registry=reg
    )
    assert "[C'4]" in thm.statement
    assert len(chain.steps) == 2


def test_theta_structure_is_watamura_not_triangular(setup):
    # The audit's defining checks (arXiv:1408.2649 (2.6)-(2.7)):
    # the anchor KILLS pure vectors and the bracket of two pure
    # vectors is ZERO.
    from jacopy.packages.drinfeld.tilde_calculus import (
        _tilde_engine,
    )
    from jacopy.packages.poisson.tilde import _normalized_by

    reg, f, h, om, et, ze, U, V, W, X, N = setup
    eng = _tilde_engine(N, reg, declare_fi=False)
    assert _normalized_by(
        eng, theta_anchor(N, U, Integer(0)), reg
    ) == Integer(0)
    vec, form = theta_dorfman(
        N, U, Integer(0), V, Integer(0)
    )
    assert _normalized_by(eng, vec, reg) == Integer(0)
    assert _normalized_by(eng, form, reg) == Integer(0)


def test_theta_invariance_c5_closes(setup):
    # [C'5] — CLOSED 2026-09-08 (second pass): the former
    # honest-fail pin flipped once the sharp-pairing scalar got a
    # consistent canonical representative (θ(a,b) = ⟨b,θ♯a⟩ =
    # −⟨a,θ♯b⟩, exact-oriented). The earlier 'stuck fixpoint' was a
    # wrongly-oriented side rule, not a confluence obstacle.
    reg, f, h, om, et, ze, U, V, W, X, N = setup
    chain, thm = prove_theta_invariance(
        N, U, om, V, et, W, ze, registry=reg
    )
    assert "[C'5]" in thm.statement
    assert any(
        "canonicalization" in a for a in thm.from_axioms
    )


def test_theta_structures_reject_higher_order():
    from jacopy.packages.poisson.nambu import nambu_structure

    N2 = nambu_structure(p=2)
    U, V = vector_fields("U V")
    om, et = forms("ω η", degree=2)
    with pytest.raises(ValueError, match="order-1"):
        theta_dorfman(N2, U, om, V, et)
