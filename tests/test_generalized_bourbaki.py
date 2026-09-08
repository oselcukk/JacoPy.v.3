"""Phase 7.B.2f.1 — the Bourbaki pre-calculus quintet from the
PRIMARY paper (pre-metric-bourbaki.pdf Def 9.1, Thm 9.1, Def 9.2
remark): abstract (R, Z) with R-valued structure, the standard
construction's checks, and the Cartan instantiation."""

import pytest

from jacopy.core.expr import Integer, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.generalized.bourbaki_precalculus import (
    BD,
    BIota,
    BLieR,
    BLieRAdditivityDefinition,
    BLieZ,
    BSymbol,
    BourbakiPreCalculus,
    bourbaki_engine,
    prove_cartan_satisfies_914,
    prove_cartan_satisfies_915,
    prove_standard_metric_invariance,
    prove_standard_symmetric_part,
    standard_bracket,
    standard_metric,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    B = BourbakiPreCalculus()
    U, V, W = vector_fields("U V W")
    z, y, x = B.z_sections("z y x")
    return reg, f, B, U, V, W, z, y, x


def test_standard_symmetric_part(setup):
    reg, f, B, U, V, W, z, y, x = setup
    chain, thm = prove_standard_symmetric_part(
        U, z, V, y, registry=reg
    )
    assert "d g_S" in thm.statement
    assert len(chain.steps) == 2


def test_standard_metric_invariance(setup):
    reg, f, B, U, V, W, z, y, x = setup
    chain, thm = prove_standard_metric_invariance(
        U, z, V, y, W, x, registry=reg
    )
    assert "(9.14)" in " ".join(thm.from_axioms)
    assert "(9.15)" in " ".join(thm.from_axioms)


def test_d_is_first_order_with_symbol(setup):
    # d(f·r) = f·dr + l_df r — the defining first-order law.
    reg, f, B, U, V, W, z, y, x = setup
    from jacopy.packages.poisson.tilde import _normalized_by

    (r,) = B.r_sections("r")
    eng = bourbaki_engine(reg)
    nf = _normalized_by(eng, BD(Product(f, r)), reg)
    expected = _normalized_by(
        eng,
        Sum(Product(f, BD(r)), BSymbol(f, r)),
        reg,
    )
    assert nf == expected


def test_lie_r_has_no_smooth_law(setup):
    # ℒ^R is ℝ-bilinear ONLY (Def 9.1) — a scalar product in its
    # section slot must stay INERT, never pull out.
    reg, f, B, U, V, W, z, y, x = setup
    (r,) = B.r_sections("r")
    rule = BLieRAdditivityDefinition(reg)
    assert not rule.matches(BLieR(U, Product(f, r)))


def test_cartan_is_a_bourbaki_precalculus(setup):
    # Def 9.2 remark: the usual Cartan calculus satisfies (9.14)
    # ([ℒ,ι] commutator) and (9.15) (magic + ι² = 0).
    reg, f, B, U, V, W, z, y, x = setup
    (om,) = forms("ω", degree=2)
    (X,) = vector_fields("X")
    assert prove_cartan_satisfies_914(
        U, V, om, (X,), registry=reg
    ).steps
    assert prove_cartan_satisfies_915(
        U, V, om, (X,), registry=reg
    ).steps


def test_standard_bracket_shape(setup):
    reg, f, B, U, V, W, z, y, x = setup
    vec, zc = standard_bracket(U, z, V, y)
    from jacopy.central.tangent.lie_bracket import lie_bracket
    from jacopy.core.expr import Neg

    assert vec == lie_bracket(U, V)
    assert zc == Sum(
        BLieZ(U, y), Neg(BLieZ(V, z)), BD(BIota(V, z))
    )
    assert standard_metric(U, z, V, y) == Sum(
        BIota(U, y), BIota(V, z)
    )
