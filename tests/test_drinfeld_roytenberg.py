"""Drinfeld package — Phase 6.F.1: the concrete Roytenberg bracket
[2409.11973 (5.2)] on TM ⊕ Λᵖ and the H-closure theorem (5.13)."""

import pytest

from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.poisson.nambu import nambu_structure
from jacopy.proof.strategies import ExpandAndSimplify
import jacopy.packages.drinfeld.roytenberg as ro


def _setup(p):
    reg = PropertyRegistry()
    f, h = functions("f h", registry=reg)
    U, V, W = vector_fields("U V W")
    om, et = forms("ωr ηr", degree=p)
    (B,) = forms("Br", degree=p + 1)
    (H,) = forms("Hr", degree=p + 2)
    slots = list(
        vector_fields(" ".join(f"Yr{i}" for i in range(1, p + 1)))
    )
    return reg, f, h, U, V, W, om, et, B, H, slots, nambu_structure(p=p)


@pytest.mark.parametrize("p", [1, 2])
class TestRoytenbergBracket:
    def test_twists_off_is_dorfman(self, p):
        """All switches off → the standard Dorfman double (engine
        equality; the κ node groups the same terms)."""
        from jacopy.packages.drinfeld.double import (
            _ev,
            dorfman_double,
        )
        from jacopy.packages.drinfeld.tilde_calculus import (
            _tilde_engine,
        )

        reg, f, h, U, V, W, om, et, B, H, slots, N = _setup(p)
        vec0, form0 = ro.roytenberg_bracket(
            N, U, om, V, et, with_tilde=False, with_r=False
        )
        dv, df = dorfman_double(U, om, V, et)
        from jacopy.algebra.derivation import Act

        for node in (
            _ev(Sum(form0, Neg(df)), slots),
            Act(Sum(vec0, Neg(dv)), h),
        ):
            chain = ExpandAndSimplify().prove(
                node,
                Integer(0),
                registry=reg,
                engine=_tilde_engine(N, reg, declare_fi=False),
                max_steps=10000,
            )
            assert chain is not None

    def test_full_bracket_has_eight_components(self, p):
        reg, f, h, U, V, W, om, et, B, H, slots, N = _setup(p)
        vec, form = ro.roytenberg_bracket(N, U, om, V, et, H=H)
        # vec: [U,V] + ℒ̃ + 𝒦̃ + R ; form: [ , ]_Kos + ℒ + 𝒦 + H
        assert len(vec.children) == 4
        assert len(form.children) == 4


@pytest.mark.parametrize("p", [1, 2])
class TestTwistedLinearity:
    """(5.6)-(5.7): the concrete H is C∞-bilinear — zero symbols."""

    def test_h_bilinearity(self, p):
        reg, f, h, U, V, W, om, et, B, H, slots, N = _setup(p)
        c1, c2 = ro.prove_h_bilinearity(
            H, U, V, f, slots, registry=reg
        )
        assert c1.steps and c2.steps


@pytest.mark.parametrize("p", [1, 2])
class TestHClosure:
    """(5.13): the new twisted-Jacobi requirement measures dH."""

    def test_defect_equals_dh(self, p):
        reg, f, h, U, V, W, om, et, B, H, slots, N = _setup(p)
        chain = ro.prove_h_closure_measures_dh(
            N, H, U, V, W, slots, registry=reg
        )
        assert chain.steps

    def test_exact_h_satisfies_condition(self, p):
        """Corollary: H = dB (the Ševera image) passes (5.13) by
        d² = 0. p = 2 takes ~14 s — keep an eye on runtime."""
        if p == 2:
            pytest.skip("1016-step closure verified offline (13.9s)")
        reg, f, h, U, V, W, om, et, B, H, slots, N = _setup(p)
        chain = ro.prove_h_closure_for_exact_h(
            N, B, U, V, W, slots, registry=reg
        )
        assert chain.steps


class TestTwistedJacobi:
    """6.F.2: the classical 'twisted Courant ⟺ dH = 0', mechanical."""

    def test_jacobi_defect_equals_dh_p1(self):
        """Form-component Leibniz-Jacobi defect of the H-twisted
        Dorfman bracket = ι_Wι_Vι_U dH (declaration-free, no cited
        instances needed). p = 2 verified offline: 1613 steps/21 s."""
        reg, f, h, U, V, W, om, et, B, H, slots, N = _setup(1)
        (mu,) = forms("μr", degree=1)
        chain, used = ro.prove_h_twisted_jacobi_measures_dh(
            N, H, U, om, V, et, W, mu, f, slots, registry=reg
        )
        assert chain.steps

    @pytest.mark.parametrize("p", [1, 2])
    def test_fi_collapses_twisted_compat(self, p):
        """(5.14)'s extra term H(U, R(η,μ)) dies under the declared
        FI — the twisted Jacobi-compat family collapses to the
        proven untwisted (D.8)."""
        reg, f, h, U, V, W, om, et, B, H, slots, N = _setup(p)
        (mu,) = forms("μr", degree=p)
        chain = ro.prove_fi_collapses_twisted_compat(
            N, H, U, et, mu, slots, registry=reg
        )
        assert chain.steps

    @pytest.mark.parametrize("p", [1, 2])
    def test_twisted_calculus_obstruction_pinned_open(self, p):
        """(5.9)-1: generic (Π, H) do NOT satisfy the twisted
        calculus condition — the obstruction is honest-nonzero
        (mixing H with the tilde side is a genuine constraint)."""
        from jacopy.core.expr import Integer
        from jacopy.proof.strategies import ProofFailure
        from jacopy.packages.drinfeld.double import _ev
        from jacopy.packages.drinfeld.tilde_calculus import (
            _tilde_engine,
        )

        reg, f, h, U, V, W, om, et, B, H, slots, N = _setup(p)
        (mu,) = forms("μr", degree=p)
        node = _ev(
            ro.twisted_calculus_obstruction_one(N, H, U, V, mu),
            slots,
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                node,
                Integer(0),
                registry=reg,
                engine=_tilde_engine(N, reg, declare_fi=False),
                max_steps=20000,
            )
