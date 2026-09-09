"""Phase 7.B.2d — Roytenberg → Courant identifications (PDF item
14g): each Roytenberg switch configuration IS one of the verified
Courant structures; the coboundary D_θ survives every twist."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.generalized.poisson_generalized import (
    poisson_base,
)
from jacopy.packages.generalized.roytenberg_courant import (
    prove_full_roytenberg_symmetric_part,
    prove_roytenberg_h_is_twisted_dorfman,
    prove_roytenberg_is_theta_double_under_poisson,
    prove_roytenberg_reduces_to_dorfman,
)
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    om, et = forms("ω η", degree=1)
    (H,) = forms("H", degree=3)
    U, V = vector_fields("U V")
    return reg, f, om, et, H, U, V, poisson_base()


def test_roytenberg_all_off_is_standard_dorfman(setup):
    reg, f, om, et, H, U, V, N = setup
    chain, thm = prove_roytenberg_reduces_to_dorfman(
        N, U, om, V, et, registry=reg
    )
    assert "standard Dorfman" in thm.statement


def test_roytenberg_tilde_r_is_theta_double(setup):
    # THE 14g statement: the Poisson-case Roytenberg bracket yields
    # the verified Courant structure — the TRIANGULAR θ-double
    # (2026-09-09 audit, finding 2: not Watamura's trivial-TM one).
    reg, f, om, et, H, U, V, N = setup
    chain, thm = prove_roytenberg_is_theta_double_under_poisson(
        N, U, om, V, et, registry=reg
    )
    assert "TRIANGULAR θ-double" in thm.statement


def test_roytenberg_theta_identification_needs_poisson(setup):
    # honest-fail: the difference IS the derived R-twist.
    reg, f, om, et, H, U, V, N = setup
    with pytest.raises(ProofFailure, match="R"):
        prove_roytenberg_is_theta_double_under_poisson(
            N, U, om, V, et,
            registry=reg,
            declare_poisson=False,
        )


def test_roytenberg_h_is_twisted_dorfman(setup):
    reg, f, om, et, H, U, V, N = setup
    chain, thm = prove_roytenberg_h_is_twisted_dorfman(
        N, H, U, om, V, et, registry=reg
    )
    assert "H-twisted" in thm.statement


def test_full_roytenberg_symmetric_part_is_d_theta(setup):
    reg, f, om, et, H, U, V, N = setup
    chain, thm = prove_full_roytenberg_symmetric_part(
        N, H, U, om, V, et, registry=reg
    )
    assert "triangular coboundary" in thm.statement
    assert len(chain.steps) == 2
