"""Phase 7.B.2f.2 — the Bourbaki STRUCTURE layer (primary paper
§6-7): R-valued metric, 𝔻/𝕃, ℒ^R with anchor symbol; Cor 6.1, the
(7.2) generalized B 4.11, and the left-Leibniz corollary."""

import pytest

from jacopy.core.expr import Product
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import functions
from jacopy.packages.generalized.bourbaki_structure import (
    BLieRE,
    BMetric,
    bourbaki_structure_context,
    prove_left_leibniz_bourbaki,
    prove_locality_is_symbol,
    prove_right_leibniz_from_r_invariance,
)
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = bourbaki_structure_context()
    u, v = E.sections("u v")
    return reg, f, E, u, v


def test_cor_61_locality_is_the_symbol(setup):
    reg, f, E, u, v = setup
    chain, thm = prove_locality_is_symbol(
        E, u, v, f, registry=reg
    )
    assert "local almost-Leibniz" in thm.statement
    assert "𝕃_df(g(u,v))" in thm.statement


def test_72_right_leibniz_from_r_valued_invariance(setup):
    # The generalized B 4.11: R-VALUED invariance + anchor-symbol
    # derivation + non-degeneracy ⟹ defect-free right-Leibniz.
    reg, f, E, u, v = setup
    chain, thm = prove_right_leibniz_from_r_invariance(
        E, u, v, f, registry=reg
    )
    assert "(7.2)" in thm.statement
    rules = [s.rule for s in chain.steps]
    assert any("non-degeneracy" in r for r in rules)
    assert any("R-VALUED" in r or "R-valued" in r for r in rules)


def test_left_leibniz_with_symbol_locality(setup):
    reg, f, E, u, v = setup
    chain, thm = prove_left_leibniz_bourbaki(
        E, u, v, f, registry=reg
    )
    assert "𝕃_df(g(u,v))" in thm.statement
    assert "ℚ-linear" in chain.steps[0].rule


def test_r_metric_is_not_scalar(setup):
    # The audit's 10h point: g is R-VALUED — it must not register
    # as a scalar function.
    from jacopy.central.calculus.scalars import (
        is_scalar_function,
    )

    reg, f, E, u, v = setup
    assert not is_scalar_function(BMetric(u, v), reg)


def test_lie_re_stays_inert_without_invariance(setup):
    # Without the (7.8) declaration ℒ^R g(v,w) must NOT open —
    # honest inertness of the abstract layer.
    from jacopy.packages.generalized.bourbaki_structure import (
        bourbaki_structure_engine,
    )
    from jacopy.packages.poisson.tilde import _normalized_by

    reg, f, E, u, v = setup
    w = E.sections("w")[0]
    eng = bourbaki_structure_engine(
        E, reg, invariance=False
    )
    node = BLieRE(u, BMetric(v, w))
    assert _normalized_by(eng, node, reg) == node
