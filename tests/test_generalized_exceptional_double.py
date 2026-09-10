"""The exceptional Drinfel'd double identification (2026-09-10):
the Ψ_Π-rotated exceptional Courant bracket equals the double of
TM and Λ²⊕Λ⁵ built from the block map Π = (Π₃, Π₆ + Π₃⊛Π₃) up to
the derived twist R′, whose bidegree pieces are the mixed
Π₃/Π₆/⊛ conditions."""

import pytest

from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.algebra.derivation import Act
from jacopy.packages.poisson.nambu import nambu_structure
from jacopy.packages.generalized.exceptional_double import (
    _engine,
    _normalize,
    exceptional_double,
    exceptional_r_twist,
    exceptional_r_twist_components,
    exceptional_z_bracket,
    prove_exceptional_r_twist_bidegree_split,
    prove_exceptional_r_twist_symmetric_part,
    prove_r22_reduces_to_lifted_cross_term,
    prove_r25_is_lie_tilde_equivariance_defect,
    prove_rotated_is_exceptional_double_plus_r,
)
from jacopy.packages.generalized.exceptional_rotation import (
    rotated_exceptional_bracket,
)
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def exc():
    reg = PropertyRegistry()
    (h,) = functions("h", registry=reg)
    U, V = vector_fields("U V")
    om2, et2 = forms("ω₂ η₂", degree=2)
    om5, et5 = forms("ω₅ η₅", degree=5)
    N3 = nambu_structure("Π₃", p=2)
    N6 = nambu_structure("Π₆", p=5)
    return reg, h, U, V, om2, et2, om5, et5, N3, N6


def test_rotated_bracket_is_the_double_plus_the_derived_twist(exc):
    reg, h, U, V, om2, et2, om5, et5, N3, N6 = exc
    chain, thm = prove_rotated_is_exceptional_double_plus_r(
        N3, N6, U, om2, om5, V, et2, et5, h, registry=reg
    )
    assert len(chain.steps) == 3
    assert "declaration-free" in thm.statement
    assert not any(a.startswith("declared") for a in thm.from_axioms)


def test_derived_twist_is_the_rotated_vector_part_on_pure_forms(exc):
    # R′(ω,η) is literally vec(Ψ⁻¹[Ψω, Ψη]) for U = V = 0.
    reg, h, U, V, om2, et2, om5, et5, N3, N6 = exc
    rv, r2, r5 = rotated_exceptional_bracket(
        N3, N6, Integer(0), om2, om5, Integer(0), et2, et5
    )
    z2, z5 = exceptional_z_bracket(N3, N6, om2, om5, et2, et5)
    eng = _engine(N3, N6, reg)
    R = exceptional_r_twist(N3, N6, om2, om5, et2, et5)
    assert _normalize(eng, Act(Sum(rv, Neg(R)), h), reg) == Integer(0)
    assert _normalize(eng, Sum(r2, Neg(z2)), reg) == Integer(0)
    assert _normalize(eng, Sum(r5, Neg(z5)), reg) == Integer(0)


def test_derived_twist_splits_by_bidegree(exc):
    reg, h, U, V, om2, et2, om5, et5, N3, N6 = exc
    chain, thm = prove_exceptional_r_twist_bidegree_split(
        N3, N6, om2, om5, et2, et5, h, registry=reg
    )
    assert len(chain.steps) == 1
    parts = exceptional_r_twist_components(
        N3, N6, om2, om5, et2, et5
    )
    assert set(parts) == {"22", "25", "52", "55"}


def test_r22_reduces_to_the_lifted_cross_term_under_declared_fi(exc):
    reg, h, U, V, om2, et2, om5, et5, N3, N6 = exc
    chain, thm = prove_r22_reduces_to_lifted_cross_term(
        N3, N6, om2, et2, h, registry=reg
    )
    assert "Π₆ + Π₃⊛Π₃" in thm.statement
    assert any("declared" in a for a in thm.from_axioms)


def test_r22_reduction_needs_the_declared_fi(exc):
    reg, h, U, V, om2, et2, om5, et5, N3, N6 = exc
    with pytest.raises(ProofFailure, match="declared"):
        prove_r22_reduces_to_lifted_cross_term(
            N3, N6, om2, et2, h, registry=reg, declare_fi=False
        )
    # and the honest residual really is non-zero without it
    from jacopy.core.wedge import Wedge
    from jacopy.central.tangent.exterior import d
    from jacopy.packages.generalized.exceptional_double import pi_hat

    parts = exceptional_r_twist_components(
        N3, N6, om2, Integer(0), et2, Integer(0)
    )
    diff = Sum(parts["22"], Neg(pi_hat(N3, N6, Wedge(et2, d(om2)))))
    nf = _normalize(_engine(N3, N6, reg), Act(diff, h), reg)
    assert nf != Integer(0)


def test_r25_is_the_lie_tilde_equivariance_defect(exc):
    reg, h, U, V, om2, et2, om5, et5, N3, N6 = exc
    chain, thm = prove_r25_is_lie_tilde_equivariance_defect(
        N3, N6, om2, et5, h, registry=reg
    )
    assert "ℒ̃-equivariance" in thm.statement


def test_symmetric_part_of_the_derived_twist_is_the_exact_pairing(exc):
    reg, h, U, V, om2, et2, om5, et5, N3, N6 = exc
    chain, thm = prove_exceptional_r_twist_symmetric_part(
        N3, N6, om2, om5, et2, et5, h, registry=reg
    )
    assert "transported pairing" in thm.statement
    assert not any(a.startswith("declared") for a in thm.from_axioms)


def test_double_reduces_to_the_exceptional_bracket_when_pi_is_absent(exc):
    # With both form slots of one argument zero, the block map does
    # nothing and the double's form slots are the exceptional
    # bracket's (structural sanity of the construction).
    from jacopy.packages.drinfeld.examples import (
        exceptional_courant_bracket,
    )

    reg, h, U, V, om2, et2, om5, et5, N3, N6 = exc
    dv, d2, d5 = exceptional_double(
        N3, N6, U, Integer(0), Integer(0), V, Integer(0), Integer(0)
    )
    ev, e2, e5 = exceptional_courant_bracket(
        U, Integer(0), Integer(0), V, Integer(0), Integer(0)
    )
    eng = _engine(N3, N6, reg)
    assert _normalize(eng, Act(Sum(dv, Neg(ev)), h), reg) == Integer(0)
    assert _normalize(eng, Sum(d2, Neg(e2)), reg) == Integer(0)
    assert _normalize(eng, Sum(d5, Neg(e5)), reg) == Integer(0)
