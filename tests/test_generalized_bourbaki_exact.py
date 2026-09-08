"""Phase 7.B.2f.3 — exact pre-metric-Bourbaki algebroids and the
INDUCED pre-calculus (primary paper §8-9): Lemma 9.1 and Prop 9.1
(9.11)/(9.12) — the circle with Thm 9.1 closes."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import functions, vector_fields
from jacopy.packages.generalized.bourbaki_exact import (
    ChiSec,
    PhiSec,
    exact_bourbaki_context,
    exact_bourbaki_engine,
    prove_911,
    prove_912,
    prove_lemma_91,
)
from jacopy.packages.generalized.bourbaki_precalculus import (
    BourbakiPreCalculus,
)
from jacopy.packages.generalized.bourbaki_structure import (
    BMetric,
)
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (h,) = functions("h", registry=reg)
    E = exact_bourbaki_context()
    B = BourbakiPreCalculus()
    U, V = vector_fields("U V")
    (z,) = B.z_sections("z")
    (u_sec,) = E.sections("u")
    return reg, h, E, B, U, V, z, u_sec


def test_lemma_91_bracket_lands_in_image_of_chi(setup):
    reg, h, E, B, U, V, z, u_sec = setup
    chain, thm = prove_lemma_91(E, u_sec, z, h, registry=reg)
    assert "image of χ" in thm.statement


def test_prop_911_induced_axiom_915(setup):
    reg, h, E, B, U, V, z, u_sec = setup
    chain, thm = prove_911(E, U, V, z, registry=reg)
    assert "(9.15)" in thm.statement
    assert any(
        "exactness" in a for a in thm.from_axioms
    )


def test_prop_912_induced_axiom_914_needs_isotropy(setup):
    reg, h, E, B, U, V, z, u_sec = setup
    chain, thm = prove_912(E, U, V, z, registry=reg)
    assert "(9.14)" in thm.statement
    assert any("g-isotropy" in a for a in thm.from_axioms)


def test_912_honest_fails_without_isotropy(setup):
    # Without the declared isotropy the g(χz, [φU,φV]) leg stays
    # inert and the citation cannot close — honest-fail.
    reg, h, E, B, U, V, z, u_sec = setup
    from jacopy.packages.generalized.bourbaki_exact import (
        _cited_zero_proof,
        _invariance_instance,
    )
    from jacopy.packages.generalized.bourbaki_precalculus import (
        BIota,
        BLieZ,
    )
    from jacopy.packages.generalized.bourbaki_structure import (
        BLieRE,
    )
    from jacopy.central.tangent.lie_bracket import lie_bracket
    from jacopy.core.expr import Neg, Sum

    engine = exact_bourbaki_engine(
        E, reg, isotropic=False
    )
    target = Sum(
        BLieRE(PhiSec(U), BIota(V, z)),
        Neg(BIota(lie_bracket(U, V), z)),
        Neg(BIota(V, BLieZ(U, z))),
    )
    instance = _invariance_instance(
        E, PhiSec(U), PhiSec(V), ChiSec(z)
    )
    with pytest.raises(ProofFailure):
        _cited_zero_proof(
            "no_isotropy",
            "s",
            target,
            instance,
            "instance",
            engine,
            reg,
            from_axioms=(),
            notes="",
        )


def test_splitting_anchors(setup):
    # ρ(φU) = U, ρ(χz) = 0 — the (8.5) exactness data.
    from jacopy.algebra.derivation import Act
    from jacopy.core.expr import Integer
    from jacopy.packages.poisson.tilde import _normalized_by

    reg, h, E, B, U, V, z, u_sec = setup
    eng = exact_bourbaki_engine(E, reg)
    assert (
        _normalized_by(
            eng, Act(E.anchor(PhiSec(U)), h), reg
        )
        == Act(U, h)
    )
    assert (
        _normalized_by(
            eng, Act(E.anchor(ChiSec(z)), h), reg
        )
        == Integer(0)
    )
