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


@pytest.mark.parametrize("p", [1, 2])
class TestRTwistLinearity:
    """(7.26): R′ is C∞-linear in the second entry; the first entry
    carries the ``−Π(df ∧ g_Z)`` anomaly (TM case: L_A = 0,
    σ_d(f,·) = df ∧ ·)."""

    def test_second_entry_linear(self, p):
        from jacopy.packages.drinfeld.twist import (
            prove_r_twist_second_entry_linear,
        )

        reg, h, U, V, om, et, B, slots, N = _setup(p)
        (f,) = functions("ft", registry=reg)
        chain = prove_r_twist_second_entry_linear(
            N, om, et, f, h, registry=reg
        )
        assert chain.steps

    def test_first_entry_obstruction(self, p):
        from jacopy.packages.drinfeld.twist import (
            prove_r_twist_first_entry_obstruction,
        )

        reg, h, U, V, om, et, B, slots, N = _setup(p)
        (f,) = functions("ft", registry=reg)
        chain = prove_r_twist_first_entry_obstruction(
            N, om, et, f, h, registry=reg
        )
        assert chain.steps


@pytest.mark.parametrize("p", [1, 2])
class TestHTwistedInitialBracket:
    """Twists applied to the H-TWISTED Dorfman bracket."""

    def test_severa_full(self, p):
        """Ψ_B on H-Dorfman = (H+dB)-Dorfman [eq (7.19)-(7.20)]."""
        from jacopy.packages.drinfeld.twist import (
            prove_b_twist_severa_h,
        )

        reg, h, U, V, om, et, B, slots, N = _setup(p)
        (H,) = forms("Ht", degree=p + 2)
        chain = prove_b_twist_severa_h(
            H, B, U, om, V, et, slots, registry=reg
        )
        assert chain.steps

    def test_pi_twist_h_form_shift(self, p):
        """form = Nambu form + H(U+Πω, V+Πη)."""
        from jacopy.packages.drinfeld.twist import (
            prove_pi_twist_h_form,
        )

        reg, h, U, V, om, et, B, slots, N = _setup(p)
        (H,) = forms("Ht", degree=p + 2)
        chain = prove_pi_twist_h_form(
            N, H, U, om, V, et, slots, registry=reg
        )
        assert chain.steps

    def test_pi_twist_h_vec_shift(self, p):
        """vec = Nambu vec + R′ − Π(H(U+Πω, V+Πη))."""
        from jacopy.packages.drinfeld.twist import (
            prove_pi_twist_h_vec,
        )

        reg, h, U, V, om, et, B, slots, N = _setup(p)
        (H,) = forms("Ht", degree=p + 2)
        chain = prove_pi_twist_h_vec(
            N, H, U, om, V, et, h, registry=reg
        )
        assert chain.steps


@pytest.mark.parametrize("p", [1, 2])
class TestMTwist:
    """Ψ_m = [[m,0],[0,m̃]] frame-change twist [eq (7.13)-(7.16)]:
    conjugation by an OPAQUE invertible endomorphism pair (EndoVF /
    EndoForm primitives + linearity/inverse laws). No new bracket
    terms are generated — the mechanical content is that the
    conjugated bracket is again a Leibniz-algebroid bracket with the
    twisted anchor ρ′(U+ω) = m(U)."""

    def test_m_is_bracket_morphism(self, p):
        from jacopy.packages.drinfeld.twist import (
            prove_m_is_bracket_morphism,
        )

        reg, h, U, V, om, et, B, slots, N = _setup(p)
        chain = prove_m_is_bracket_morphism(
            "m", U, V, h, registry=reg
        )
        assert chain.steps

    def test_right_leibniz_vec_with_twisted_anchor(self, p):
        from jacopy.packages.drinfeld.twist import (
            prove_m_twist_right_leibniz_vec,
        )

        reg, h, U, V, om, et, B, slots, N = _setup(p)
        (f,) = functions("fm", registry=reg)
        chain = prove_m_twist_right_leibniz_vec(
            "m", U, om, V, et, f, h, registry=reg
        )
        assert chain.steps

    def test_right_leibniz_form_raw_level(self, p):
        from jacopy.packages.drinfeld.twist import (
            prove_m_twist_right_leibniz_form,
        )

        reg, h, U, V, om, et, B, slots, N = _setup(p)
        (f,) = functions("fm", registry=reg)
        chain = prove_m_twist_right_leibniz_form(
            "m", U, om, V, et, f, registry=reg
        )
        assert chain.steps

    def test_symmetric_part_conjugated_exact(self, p):
        from jacopy.packages.drinfeld.twist import (
            prove_m_twist_symmetric_part_form,
        )

        reg, h, U, V, om, et, B, slots, N = _setup(p)
        chain = prove_m_twist_symmetric_part_form(
            "m", U, om, V, et, registry=reg
        )
        assert chain.steps

    def test_components_are_conjugation_formulas(self, p):
        """(7.14)/(7.16) structural: the twisted components ARE the
        conjugation formulas verbatim."""
        from jacopy.central.objects.endomorphism import (
            EndoForm,
            EndoVF,
        )
        from jacopy.central.tangent.lie_bracket import lie_bracket
        from jacopy.packages.drinfeld.double import dorfman_double
        from jacopy.packages.drinfeld.twist import twisted_dorfman_m

        reg, h, U, V, om, et, B, slots, N = _setup(p)
        vec, form = twisted_dorfman_m("m", U, om, V, et)
        mU, mV = EndoVF("m", U), EndoVF("m", V)
        assert vec == EndoVF(
            "m", lie_bracket(mU, mV), inverted=True
        )
        base_vec, base_form = dorfman_double(
            mU, EndoForm("m", om), mV, EndoForm("m", et)
        )
        assert form == EndoForm("m", base_form, inverted=True)


class TestEndomorphismLaws:
    """Unit tests for the 6.E.3 primitive."""

    def test_inverse_law_both_ways(self):
        from jacopy.central.objects.endomorphism import (
            EndoForm,
            EndoInverseDefinition,
            EndoVF,
        )

        (U,) = vector_fields("U")
        (om,) = forms("ωe", degree=2)
        rule = EndoInverseDefinition()
        n1 = EndoVF("m", EndoVF("m", U, inverted=True))
        n2 = EndoForm("m", EndoForm("m", om), inverted=True)
        assert rule.matches(n1) and rule.rewrite(n1) == U
        assert rule.matches(n2) and rule.rewrite(n2) == om

    def test_different_names_do_not_cancel(self):
        from jacopy.central.objects.endomorphism import (
            EndoInverseDefinition,
            EndoVF,
        )

        (U,) = vector_fields("U")
        n = EndoVF("m", EndoVF("k", U, inverted=True))
        assert not EndoInverseDefinition().matches(n)

    def test_linearity_scalar_and_sum(self):
        from jacopy.core.expr import Product, Sum
        from jacopy.central.objects.endomorphism import (
            EndoLinearityDefinition,
            EndoVF,
        )

        reg = PropertyRegistry()
        (f,) = functions("fe", registry=reg)
        U, V = vector_fields("U V")
        rule = EndoLinearityDefinition(reg)
        n = EndoVF("m", Product(f, U))
        assert rule.matches(n)
        assert rule.rewrite(n) == Product(f, EndoVF("m", U))
        n2 = EndoVF("m", Sum(U, V))
        assert rule.rewrite(n2) == Sum(
            EndoVF("m", U), EndoVF("m", V)
        )
