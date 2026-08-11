"""Drinfeld package — Phase 6.I.1: metric-invariance compatibility
(App D (D.14)-(D.19) / the (4.44)-(4.50) family). Claim 6: the
double's metric-invariance operator is ℒ along the TOTAL anchor."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, vector_fields
from jacopy.packages.poisson.nambu import nambu_structure
from jacopy.proof.strategies import ProofFailure
import jacopy.packages.drinfeld.metric_invariance as mi


def _setup(p):
    reg = PropertyRegistry()
    U, V, W = vector_fields("Ui Vi Wi")
    om, et, mu = forms("ωi ηi μi", degree=p)
    om2, om3 = forms("ωi2 ωi3", degree=p)
    slots = (
        list(
            vector_fields(
                " ".join(f"Yi{k}" for k in range(1, p))
            )
        )
        if p > 1
        else []
    )
    return reg, U, V, W, om, et, mu, om2, om3, slots, nambu_structure(p=p)


@pytest.mark.parametrize("p", [1, 2])
class TestMetricInvariance:
    def test_d14_z_invariance(self, p):
        """(D.14): the Z-side invariance operator is ℒ_{Πω} — under
        the declared FI (p = 3 verified offline: 209 steps)."""
        reg, U, V, W, om, et, mu, om2, om3, slots, N = _setup(p)
        assert mi.prove_z_metric_invariance(
            N, om, et, mu, slots, registry=reg
        ).steps

    def test_d16_a_mixing(self, p):
        reg, U, V, W, om, et, mu, om2, om3, slots, N = _setup(p)
        assert mi.prove_a_mixing_condition(
            N, U, V, mu, slots, registry=reg
        ).steps

    def test_d17_a_invariance_of_gz(self, p):
        """(D.17): exact 𝒦̃ corrections, declaration-free."""
        reg, U, V, W, om, et, mu, om2, om3, slots, N = _setup(p)
        assert mi.prove_a_invariance_of_gz(
            N, U, et, mu, slots, registry=reg
        ).steps

    def test_d18_dual_mixing(self, p):
        reg, U, V, W, om, et, mu, om2, om3, slots, N = _setup(p)
        assert mi.prove_dual_mixing_condition(
            N, om, et, W, slots, registry=reg
        ).steps

    def test_d19_claim6_double_invariance(self, p):
        """Claim 6: ℒ_{U+Πω} g_E(e₂,e₃) = g_E([e₁,e₂],e₃) +
        g_E(e₂,[e₁,e₃]) — under the declared FI (p = 3 offline: 905
        steps / 4.4 s)."""
        reg, U, V, W, om, et, mu, om2, om3, slots, N = _setup(p)
        assert mi.prove_double_metric_invariance(
            N, U, om, V, om2, W, om3, slots, registry=reg
        ).steps


class TestHonestFails:
    """(D.14) and Claim 6 genuinely NEED the FI — pinned."""

    def test_d14_without_fi(self):
        reg, U, V, W, om, et, mu, om2, om3, slots, N = _setup(2)
        with pytest.raises(ProofFailure):
            mi.prove_z_metric_invariance(
                N, om, et, mu, slots, registry=reg,
                declare_fi=False,
            )

    def test_claim6_without_fi(self):
        reg, U, V, W, om, et, mu, om2, om3, slots, N = _setup(2)
        with pytest.raises(ProofFailure):
            mi.prove_double_metric_invariance(
                N, U, om, V, om2, W, om3, slots, registry=reg,
                declare_fi=False,
            )
