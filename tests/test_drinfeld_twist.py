"""Drinfeld package — Phase 6.E: twist automorphisms [2409.11973 §7].

The Ψ_Π-twisted standard Dorfman bracket reproduces the Nambu tilde
calculus (form side exactly, vec side modulo the R-twist) and the
Ψ_B twist carries the Ševera H = dB remnant — all declaration-free.
p = 1 and p = 2 in-suite (the theorems are degree-general)."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.poisson.nambu import nambu_structure
from jacopy.packages.drinfeld.twist import (
    prove_b_twist_severa,
    prove_b_twist_vec_unchanged,
    prove_pi_twist_form_is_nambu,
    prove_pi_twist_vec_is_nambu_plus_r,
    prove_twisted_kappa_tilde,
    prove_twisted_lie_tilde,
    prove_twisted_z_bracket_is_koszul,
)


def _setup(p):
    reg = PropertyRegistry()
    f, h = functions("f h", registry=reg)
    U, V = vector_fields("U V")
    om, et = forms("ωt ηt", degree=p)
    (B,) = forms("Bt", degree=p + 1)
    slots = list(
        vector_fields(" ".join(f"Yt{i}" for i in range(1, p + 1)))
    )
    return reg, h, U, V, om, et, B, slots, nambu_structure(p=p)


@pytest.mark.parametrize("p", [1, 2])
class TestPiTwist:
    """Ψ_Π = [[1,Π],[0,1]]: the tilde calculus is BORN from the
    twist — the exam scenario's mechanical content."""

    def test_form_component_is_nambu_double(self, p):
        reg, h, U, V, om, et, B, slots, N = _setup(p)
        chain = prove_pi_twist_form_is_nambu(
            N, U, om, V, et, slots, registry=reg
        )
        assert chain.steps

    def test_vec_component_is_nambu_plus_r_twist(self, p):
        """Declaration-free: the difference is EXACTLY R′(ω,η) =
        [Πω,Πη] − Π[ω,η]_Kos for any (p+1)-vector."""
        reg, h, U, V, om, et, B, slots, N = _setup(p)
        chain = prove_pi_twist_vec_is_nambu_plus_r(
            N, U, om, V, et, h, registry=reg
        )
        assert chain.steps

    def test_twisted_lie_tilde_matches(self, p):
        """(7.25) ℒ̃′_ω = [Πω,·] − Π𝒦_· equals 6.D's ℒ̃ via magic."""
        reg, h, U, V, om, et, B, slots, N = _setup(p)
        chain = prove_twisted_lie_tilde(N, om, V, h, registry=reg)
        assert chain.steps

    def test_twisted_kappa_tilde_matches(self, p):
        reg, h, U, V, om, et, B, slots, N = _setup(p)
        chain = prove_twisted_kappa_tilde(N, et, U, h, registry=reg)
        assert chain.steps

    def test_twisted_z_bracket_is_koszul(self, p):
        """(7.25) [ω,η]′_Z = ℒ_Πωη + 𝒦_Πηω equals the higher Koszul
        bracket (magic formula)."""
        reg, h, U, V, om, et, B, slots, N = _setup(p)
        chain = prove_twisted_z_bracket_is_koszul(
            N, om, et, slots, registry=reg
        )
        assert chain.steps


@pytest.mark.parametrize("p", [1, 2])
class TestBTwist:
    """Ψ_B = [[1,0],[B,1]]: the Ševera remnant H = dB."""

    def test_vec_component_unchanged(self, p):
        reg, h, U, V, om, et, B, slots, N = _setup(p)
        chain = prove_b_twist_vec_unchanged(
            B, U, om, V, et, h, registry=reg
        )
        assert chain.steps

    def test_severa_h_equals_db(self, p):
        """form([e₁,e₂]_Ψ) − form([e₁,e₂]) = ι_V ι_U dB — when dB=0
        the Dorfman bracket is preserved."""
        reg, h, U, V, om, et, B, slots, N = _setup(p)
        chain = prove_b_twist_severa(
            B, U, om, V, et, slots, registry=reg
        )
        assert chain.steps
