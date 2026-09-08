"""Phase 7.B.2f.4 — Thm 9.2 + Cor 9.1 Ševera-lite classification
(primary paper §9): the bracket-leg decomposition, the (9.21) twist
laws, im Δ ⊂ im χ, and the antisymmetry of H for isotropic φ."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import functions, vector_fields
from jacopy.packages.generalized.bourbaki_exact import (
    exact_bourbaki_context,
)
from jacopy.packages.generalized.bourbaki_precalculus import (
    BourbakiPreCalculus,
)
from jacopy.packages.generalized.bourbaki_severa import (
    prove_chi_chi_bracket_vanishes,
    prove_chi_phi_bracket,
    prove_delta_first_slot_anomaly,
    prove_delta_lands_in_chi,
    prove_delta_tensorial_second_slot,
    prove_h_antisymmetric_for_isotropic_phi,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, h = functions("f h", registry=reg)
    E = exact_bourbaki_context()
    B = BourbakiPreCalculus()
    U, V = vector_fields("U V")
    z, y = B.z_sections("z y")
    return reg, f, h, E, B, U, V, z, y


def test_chi_phi_leg(setup):
    reg, f, h, E, B, U, V, z, y = setup
    chain, thm = prove_chi_phi_bracket(E, V, z, registry=reg)
    assert "χ(−ℒ^Z_V z + dι_V z)" in thm.statement


def test_chi_chi_leg_vanishes(setup):
    reg, f, h, E, B, U, V, z, y = setup
    chain, thm = prove_chi_chi_bracket_vanishes(
        E, z, y, registry=reg
    )
    assert "[χz, χy]_E = 0" in thm.statement
    rules = [s.rule for s in chain.steps]
    assert any("non-degeneracy" in r for r in rules)


def test_delta_921_laws(setup):
    reg, f, h, E, B, U, V, z, y = setup
    chain1, thm1 = prove_delta_tensorial_second_slot(
        E, U, V, f, registry=reg
    )
    assert "H(U, fV) = fH(U,V)" in thm1.statement
    chain2, thm2 = prove_delta_first_slot_anomaly(
        E, U, V, f, registry=reg
    )
    assert "l_df F(U,V)" in thm2.statement


def test_delta_lands_in_image_of_chi(setup):
    reg, f, h, E, B, U, V, z, y = setup
    chain, thm = prove_delta_lands_in_chi(
        E, U, V, h, registry=reg
    )
    assert "im Δ ⊂ im χ" in thm.statement


def test_cor_91_h_is_a_two_form_for_isotropic_phi(setup):
    reg, f, h, E, B, U, V, z, y = setup
    # general form: Δ(U,V) + Δ(V,U) = χ(dF)
    chain, thm = prove_h_antisymmetric_for_isotropic_phi(
        E, U, V, registry=reg, isotropic_phi=False
    )
    assert "χ(d F(U,V))" in thm.statement
    # isotropic φ: F = 0 declared, H antisymmetric
    chain2, thm2 = prove_h_antisymmetric_for_isotropic_phi(
        E, U, V, registry=reg, isotropic_phi=True
    )
    assert "Z-valued 2-form" in thm2.statement
    assert any(
        "g-isotropic splitting" in a
        for a in thm2.from_axioms
    )
