"""Drinfeld package — App D instantiation (tilde calculus).

Closed here, declaration-free (d²=0 / Cartan relations / Lie
Jacobi): the Jacobi compatibility conditions (D.9), (D.11), (D.12),
(D.13) at p = 1, 2.

DIAGNOSED-OPEN (recorded, honest): the tilde CALCULUS conditions
(D.5)-(D.7). The declared-FI machinery (node face + composed-action
instance citations) reduces them to two known residual families —
p = 1: the 5.E.2b bracket-pairing family; p ≥ 2: a g_Z-shaped family
``Π(d g_Z(ω, ι_W dη))`` that the bracket-morphism face of the FI
alone does not kill. The honest-fail tests below PIN this state."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.poisson.nambu import nambu_structure
from jacopy.proof.strategies import ProofFailure
import jacopy.packages.drinfeld.tilde_calculus as tc


def _setup(p):
    reg = PropertyRegistry()
    f, h = functions("f h", registry=reg)
    U, V, W = vector_fields("U V W")
    om, et, mu = forms("ωc ηc μc", degree=p)
    slots = list(
        vector_fields(" ".join(f"Yc{i}" for i in range(1, p + 1)))
    )
    return reg, f, h, U, V, W, om, et, mu, slots, nambu_structure(p=p)


@pytest.mark.parametrize("p", [1, 2])
class TestJacobiCompatibility:
    """(D.9)/(D.11)-(D.13): declaration-free closures."""

    def test_d9(self, p):
        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(p)
        chain = tc.prove_jacobi_compat_d9(
            N, U, et, mu, slots, registry=reg
        )
        assert chain.steps

    def test_d11(self, p):
        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(p)
        chain, _ = tc.prove_jacobi_compat_d11(
            N, om, V, W, f, h, registry=reg
        )
        assert chain.steps

    def test_d12(self, p):
        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(p)
        chain = tc.prove_jacobi_compat_d12(
            N, om, V, W, h, registry=reg
        )
        assert chain.steps

    def test_d13(self, p):
        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(p)
        chain = tc.prove_jacobi_compat_d13(
            N, mu, U, V, h, registry=reg
        )
        assert chain.steps


class TestCalculusConditionsClosed:
    """(D.5)-(D.7) at p ≥ 2: CLOSED 2026-08-05 — the "g_Z family"
    residual turned out to be a DERIVABLE consequence of the FI
    itself (both orientations + Lie antisymmetry force
    Π(d g_Z) = 0, the FISharpPairingSwap rule), discovered via the
    twisted-FI consistency residual in 6.F.3."""

    @pytest.mark.parametrize("cond", [1, 2, 3])
    def test_d5_d6_d7_close_at_p2(self, cond):
        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(2)
        prover = {
            1: tc.prove_tilde_calculus_condition_one,
            2: tc.prove_tilde_calculus_condition_two,
            3: tc.prove_tilde_calculus_condition_three,
        }[cond]
        chain, _ = prover(N, om, et, W, f, h, registry=reg)
        assert chain.steps


class TestCalculusConditionsP1Closed:
    """The p = 1 face of (D.5)-(D.7): CLOSED 2026-09-07 by the 6.J
    stall-time difference-test citation — the s-bridge (the 5.E.2b
    bracket-pairing family) and the FI-on-exacts family enter as
    ±/d/W/Π-paired instance lifts subtracted at stall points under a
    strictly-decreasing node-size metric. No new axioms: the FI
    declaration (as before) plus congruence lifts of proven/declared
    instances. FI withheld still honest-fails."""

    @pytest.mark.parametrize("cond", [1, 2, 3])
    def test_p1_closes(self, cond):
        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(1)
        prover = {
            1: tc.prove_tilde_calculus_condition_one,
            2: tc.prove_tilde_calculus_condition_two,
            3: tc.prove_tilde_calculus_condition_three,
        }[cond]
        chain, _ = prover(N, om, et, W, f, h, registry=reg)
        assert chain.steps

    def test_p1_without_fi_still_fails(self):
        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(1)
        with pytest.raises(ProofFailure):
            tc.prove_tilde_calculus_condition_one(
                N, om, et, W, f, h, registry=reg,
                declare_fi=False,
            )

    def test_d5_without_fi_fails(self):
        """No FI declaration → honest fail (R-twist residual)."""
        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(2)
        with pytest.raises(ProofFailure):
            tc.prove_tilde_calculus_condition_one(
                N, om, et, W, f, h, registry=reg, declare_fi=False
            )


class TestTildeBuildingBlocks:
    def test_kappa_tilde_shape(self):
        from jacopy.core.expr import Neg, Sum

        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(2)
        node = tc.kappa_tilde_nambu(N, et, W)
        assert isinstance(node, Sum) and len(node.children) == 2

    def test_morphism_declaration_fires_on_two_sharps(self):
        from jacopy.central.tangent.lie_bracket import lie_bracket

        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(2)
        rule = tc.NambuMorphismDeclaration(N)
        node = lie_bracket(N.sharp_vf(om), N.sharp_vf(et))
        assert rule.matches(node)
        out = rule.rewrite(node)
        assert out == N.sharp_vf(
            tc.nambu_koszul_bracket(N, om, et)
        )


@pytest.mark.parametrize("p", [1, 2])
class TestJacobiCompatibilityHeavy:
    """(D.8)/(D.10): the 'heavy' pair — closes declaration-free at
    the RAW expression level with the 6.E rule set (operator-level
    magic + lie-iota + interior anticommutation). The p = 1 Poisson
    counterparts (6.B, eqs 4.36/4.38) needed bridge-instance craft;
    the tilde instantiation does not."""

    def test_d8(self, p):
        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(p)
        chain = tc.prove_jacobi_compat_d8(
            N, U, et, mu, registry=reg
        )
        assert chain.steps

    def test_d10(self, p):
        reg, f, h, U, V, W, om, et, mu, slots, N = _setup(p)
        chain = tc.prove_jacobi_compat_d10(
            N, om, et, W, registry=reg
        )
        assert chain.steps
