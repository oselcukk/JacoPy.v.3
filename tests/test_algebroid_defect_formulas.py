"""Phase 6.G — exact Cartan defect formulas on almost-Lie algebroids
[2409.11973 App A (A.5)/(A.7)/(A.8)]: the Phase 3 diagnosis nodes
(predator 𝒫, Jacobiator 𝒥) ARE the curvature of the calculus.

Finding: 7 of the 8 formulas close on a BARE algebroid (no
declarations at all — stronger than the paper's almost-Lie setting);
only (A.5)-2 (d²μ) genuinely needs the antisymmetric bracket, pinned
by an honest-fail test."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import Bundle, forms, functions
from jacopy.central.algebroid import algebroid
from jacopy.proof.strategies import ProofFailure
import jacopy.central.algebroid.defect_formulas as df


def _setup(decls=("antisymmetric",)):
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid("Ed", Bundle("Ed"), declare=decls)
    u, v, w = E.sections("u v w")
    mu, et, om = forms("μd ηd ωd", degree=1, bundle=E.bundle)
    return reg, f, E, u, v, w, mu, et, om


class TestFunctionDefects:
    """(A.7): all three defects equal 𝒫(u,v)(f) — bare algebroid."""

    def test_lie_lie(self):
        reg, f, E, u, v, w, mu, et, om = _setup(())
        assert df.prove_defect_lie_lie_on_functions(
            E, u, v, f, registry=reg
        ).steps

    def test_d_squared(self):
        reg, f, E, u, v, w, mu, et, om = _setup(())
        assert df.prove_defect_d_squared_on_functions(
            E, u, v, f, registry=reg
        ).steps

    def test_lie_d(self):
        reg, f, E, u, v, w, mu, et, om = _setup(())
        assert df.prove_defect_lie_d_on_functions(
            E, u, v, f, registry=reg
        ).steps


class TestOneFormDefects:
    """(A.5): 𝒫 and μ(𝒥) quantify the broken relations."""

    def test_lie_lie(self):
        reg, f, E, u, v, w, mu, et, om = _setup(())
        assert df.prove_defect_lie_lie_on_one_forms(
            E, mu, u, v, w, registry=reg
        ).steps

    def test_d_squared_needs_antisymmetry(self):
        """(A.5)-2 closes in the paper's almost-Lie setting…"""
        reg, f, E, u, v, w, mu, et, om = _setup()
        assert df.prove_defect_d_squared_on_one_forms(
            E, mu, u, v, w, registry=reg
        ).steps

    def test_d_squared_honest_fail_without_antisymmetry(self):
        """…and honestly fails on a bare algebroid: the residual is
        exactly the antisymmetry-needing bracket combination."""
        reg, f, E, u, v, w, mu, et, om = _setup(())
        with pytest.raises(ProofFailure):
            df.prove_defect_d_squared_on_one_forms(
                E, mu, u, v, w, registry=reg
            )

    def test_lie_d(self):
        reg, f, E, u, v, w, mu, et, om = _setup(())
        assert df.prove_defect_lie_d_on_one_forms(
            E, mu, u, v, w, registry=reg
        ).steps


class TestCalculusConditionDefects:
    """(A.8): the calculus-condition defects are pure predator."""

    def test_condition_two(self):
        reg, f, E, u, v, w, mu, et, om = _setup(())
        assert df.prove_defect_calculus_condition_two(
            E, et, u, v, w, registry=reg
        ).steps

    def test_condition_three(self):
        reg, f, E, u, v, w, mu, et, om = _setup(())
        assert df.prove_defect_calculus_condition_three(
            E, om, u, v, w, registry=reg
        ).steps
