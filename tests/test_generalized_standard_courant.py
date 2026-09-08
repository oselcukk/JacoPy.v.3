"""Phase 7.B — the standard Courant algebroid on TM ⊕ T*M (PDF item
14b-d): the Dorfman bracket's remaining Courant axioms, the Courant
(skew) bracket, the Dorfman↔Courant relation, and the full LWX list
concretely (the [C1] DT-Jacobi included)."""

import pytest

from jacopy.core.expr import Integer, Product, Rational
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.tangent.exterior import d
from jacopy.packages.generalized.standard_courant import (
    courant_bracket_std,
    d_operator_std,
    prove_anchor_morphism_std,
    prove_courant_c3_lwx_std,
    prove_courant_dorfman_relation,
    prove_courant_jacobi_dt_std,
    prove_courant_skew_std,
    prove_dorfman_invariance_std,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    om, et, ze = forms("ω η ζ", degree=1)
    U, V, W = vector_fields("U V W")
    (X,) = vector_fields("X")
    return reg, f, om, et, ze, U, V, W, X


def test_d_operator_is_half_d(setup):
    reg, f, *_ = setup
    vec, form = d_operator_std(f)
    assert vec == Integer(0)
    assert form == Product(Rational(1, 2), d(f))


def test_anchor_morphism_c2(setup):
    reg, f, om, et, ze, U, V, W, X = setup
    chain, thm = prove_anchor_morphism_std(
        U, om, V, et, registry=reg
    )
    assert len(chain.steps) == 1


def test_dorfman_invariance_c5(setup):
    reg, f, om, et, ze, U, V, W, X = setup
    chain, thm = prove_dorfman_invariance_std(
        U, om, V, et, W, ze, registry=reg
    )
    assert "[C'5]" in thm.statement


def test_courant_dorfman_relation(setup):
    reg, f, om, et, ze, U, V, W, X = setup
    chain, thm = prove_courant_dorfman_relation(
        U, om, V, et, registry=reg
    )
    assert len(chain.steps) == 2  # vector + form components


def test_courant_skew(setup):
    reg, f, om, et, ze, U, V, W, X = setup
    chain, _ = prove_courant_skew_std(U, om, V, et, registry=reg)
    assert len(chain.steps) == 2


def test_courant_c3_lwx(setup):
    reg, f, om, et, ze, U, V, W, X = setup
    chain, thm = prove_courant_c3_lwx_std(
        U, om, V, et, f, registry=reg
    )
    assert "LWX [C3]" in thm.statement


def test_courant_jacobi_dt(setup):
    reg, f, om, et, ze, U, V, W, X = setup
    chain, thms = prove_courant_jacobi_dt_std(
        U, om, V, et, W, ze, f, (X,), registry=reg
    )
    # closes through cited VF-Jacobi identities (the repair loop)
    assert len(thms) >= 1
    assert len(chain.steps) > 100
