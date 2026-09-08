"""Phase 7.B.2e — frame/coframe evaluation of the generalized
structures (PDF item 14h): pairing, Dorfman components, θ-double
Koszul components and anchor components on basis sections."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import frame
from jacopy.packages.poisson import poisson_structure
from jacopy.packages.generalized.frame_evaluation import (
    prove_anchor_components,
    prove_basis_pairing_components,
    prove_dorfman_form_components,
    prove_dorfman_vector_components,
    prove_theta_koszul_components,
)


@pytest.fixture()
def setup():
    return PropertyRegistry(), frame(), poisson_structure()


def test_basis_pairing_is_kronecker(setup):
    reg, fr, P = setup
    chain, thm = prove_basis_pairing_components(
        fr, "a", "b", "c", "d", registry=reg
    )
    assert "δ^b_c + δ^d_a" in thm.statement


def test_dorfman_vector_components_are_anholonomy(setup):
    reg, fr, P = setup
    chain, thm = prove_dorfman_vector_components(
        fr, "u", "a", "c", registry=reg
    )
    assert "γ^u_ac" in thm.statement


def test_dorfman_form_components_are_minus_anholonomy(setup):
    reg, fr, P = setup
    chain, thm = prove_dorfman_form_components(
        fr, "a", "d", "b", registry=reg
    )
    assert "−γ^d_ab" in thm.statement


def test_theta_koszul_components_are_tilde_anholonomy(setup):
    reg, fr, P = setup
    chain, thm = prove_theta_koszul_components(
        P, fr, "a", "b", "c", registry=reg
    )
    assert "γ̃_c^{ab}" in thm.statement


def test_anchor_components_are_bivector_entries(setup):
    reg, fr, P = setup
    chain, thm = prove_anchor_components(
        P, fr, "b", "c", registry=reg
    )
    assert "θ(e^b, e^c)" in thm.statement
