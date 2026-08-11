"""Drinfeld package — Phase 6.I.2: bracket-morphism compatibility
(paper (4.18)-(4.20)) for φ = id ⊕ Π, i.e. the total anchor of the
Nambu double."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.poisson.nambu import nambu_structure
from jacopy.proof.strategies import ProofFailure
import jacopy.packages.drinfeld.bracket_morphism as bm


def _setup(p):
    reg = PropertyRegistry()
    (h,) = functions("hm", registry=reg)
    U, V = vector_fields("Um Vm")
    om, et = forms("ωm ηm", degree=p)
    return reg, h, U, V, om, et, nambu_structure(p=p)


@pytest.mark.parametrize("p", [1, 2])
class TestBracketMorphismCompat:
    def test_compat_condition(self, p):
        """(4.20): 𝒦̃_ηU + Π(ℒ_Uη) = [U,Πη] — declaration-free."""
        reg, h, U, V, om, et, N = _setup(p)
        assert bm.prove_morphism_compat_condition(
            N, U, et, h, registry=reg
        ).steps

    def test_compat_condition_dual(self, p):
        """(4.20)-dual: ℒ̃_ωV + Π(𝒦_Vω) = [Πω,V] — magic."""
        reg, h, U, V, om, et, N = _setup(p)
        assert bm.prove_morphism_compat_condition_dual(
            N, om, V, h, registry=reg
        ).steps

    def test_phi_z_is_morphism_under_fi(self, p):
        reg, h, U, V, om, et, N = _setup(p)
        assert bm.prove_phi_z_is_morphism(
            N, om, et, h, registry=reg
        ).steps

    def test_total_anchor_is_bracket_morphism(self, p):
        """(4.19): ρ([e₁,e₂]_double) = [ρe₁, ρe₂]_Lie under FI —
        the double-level Hamiltonian anchor-morphism (p = 3 offline:
        47 steps)."""
        reg, h, U, V, om, et, N = _setup(p)
        assert bm.prove_total_anchor_is_bracket_morphism(
            N, U, om, V, et, h, registry=reg
        ).steps


class TestHonestFails:
    def test_phi_z_needs_fi(self):
        reg, h, U, V, om, et, N = _setup(2)
        with pytest.raises(ProofFailure):
            bm.prove_phi_z_is_morphism(
                N, om, et, h, registry=reg, declare_fi=False
            )

    def test_capstone_needs_fi(self):
        reg, h, U, V, om, et, N = _setup(2)
        with pytest.raises(ProofFailure):
            bm.prove_total_anchor_is_bracket_morphism(
                N, U, om, V, et, h, registry=reg,
                declare_fi=False,
            )
