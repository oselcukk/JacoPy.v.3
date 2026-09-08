"""Phase 7.B.2 — the H-twisted standard Courant algebroid on
TM ⊕ T*M (PDF item 14e): the full Courant-axiom suite of the
H-twisted Dorfman bracket, the twisted Courant bracket and relation,
the dH-defect Jacobi, and the Ševera B-transform corollary."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.generalized.twisted_courant import (
    courant_bracket_h,
    prove_severa_twist_is_courant,
    prove_twisted_anchor_morphism,
    prove_twisted_courant_dorfman_relation,
    prove_twisted_invariance,
    prove_twisted_jacobi_dh_defect,
    prove_twisted_right_leibniz,
    prove_twisted_symmetric_part,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    om, et, ze, mu = forms("ω η ζ μ", degree=1)
    (H,) = forms("H", degree=3)
    (B,) = forms("B", degree=2)
    U, V, W = vector_fields("U V W")
    (X,) = vector_fields("X")
    return reg, f, om, et, ze, mu, H, B, U, V, W, X


def test_twisted_anchor_morphism(setup):
    reg, f, om, et, ze, mu, H, B, U, V, W, X = setup
    chain, thm = prove_twisted_anchor_morphism(
        H, U, om, V, et, registry=reg
    )
    assert "[C'2]" in thm.statement


def test_twisted_right_leibniz(setup):
    reg, f, om, et, ze, mu, H, B, U, V, W, X = setup
    chain, thm = prove_twisted_right_leibniz(
        H, U, om, V, et, f, registry=reg
    )
    assert "[C'3]" in thm.statement
    assert len(chain.steps) == 2


def test_twisted_symmetric_part_has_untwisted_d(setup):
    reg, f, om, et, ze, mu, H, B, U, V, W, X = setup
    chain, thm = prove_twisted_symmetric_part(
        H, U, om, V, et, registry=reg
    )
    # the H-term drops out: the twist does not change D
    assert "does not change D" in thm.statement


def test_twisted_invariance(setup):
    reg, f, om, et, ze, mu, H, B, U, V, W, X = setup
    chain, thm = prove_twisted_invariance(
        H, U, om, V, et, W, ze, registry=reg
    )
    assert "[C'5]" in thm.statement


def test_twisted_courant_dorfman_relation(setup):
    reg, f, om, et, ze, mu, H, B, U, V, W, X = setup
    chain, thm = prove_twisted_courant_dorfman_relation(
        H, U, om, V, et, registry=reg
    )
    assert len(chain.steps) == 2


def test_twisted_jacobi_dh_defect(setup):
    reg, f, om, et, ze, mu, H, B, U, V, W, X = setup
    chain, thms = prove_twisted_jacobi_dh_defect(
        H, U, om, V, et, W, mu, f, (X,), registry=reg
    )
    assert len(thms) >= 1
    assert len(chain.steps) > 50


def test_severa_b_transform_is_courant(setup):
    reg, f, om, et, ze, mu, H, B, U, V, W, X = setup
    chain, thms = prove_severa_twist_is_courant(
        B, U, om, V, et, W, mu, f, (X,), registry=reg
    )
    assert len(chain.steps) > 50


def test_twisted_courant_bracket_shape(setup):
    reg, f, om, et, ze, mu, H, B, U, V, W, X = setup
    from jacopy.packages.generalized.standard_courant import (
        courant_bracket_std,
    )
    from jacopy.packages.drinfeld.twist import h_term
    from jacopy.core.expr import Sum

    vec_h, form_h = courant_bracket_h(H, U, om, V, et)
    vec0, form0 = courant_bracket_std(U, om, V, et)
    assert vec_h == vec0
    assert form_h == Sum(form0, h_term(H, U, V))
