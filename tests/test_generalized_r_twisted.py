"""Phase 7.B.2c — the R-twisted tilde-Dorfman bracket (PDF item
14f.iii, Watamura R-flux): the dual of the H-twist on the Poisson
generalized double, with the anchor-defect structural theorem."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import (
    PVector,
    forms,
    functions,
    vector_fields,
)
from jacopy.packages.generalized.poisson_generalized import (
    poisson_base,
)
from jacopy.packages.generalized.r_twisted import (
    prove_derived_r_vanishes_under_poisson,
    prove_r_twisted_anchor_morphism,
    prove_r_twisted_courant_relation,
    prove_r_twisted_right_leibniz,
    prove_r_twisted_symmetric_part,
    r_term,
    r_twisted_theta_dorfman,
)
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, h = functions("f h", registry=reg)
    om, et = forms("ω η", degree=1)
    U, V = vector_fields("U V")
    return reg, f, h, om, et, U, V, poisson_base(), PVector(
        "R", degree=3
    )


def test_r_term_lands_in_vector_component(setup):
    reg, f, h, om, et, U, V, N, R = setup
    from jacopy.packages.generalized.poisson_generalized import (
        theta_dorfman,
    )

    vec_r, form_r = r_twisted_theta_dorfman(N, R, U, om, V, et)
    vec0, form0 = theta_dorfman(N, U, om, V, et)
    assert form_r == form0
    assert vec_r == Sum(vec0, r_term(R, om, et))


def test_r_twisted_right_leibniz_c3(setup):
    reg, f, h, om, et, U, V, N, R = setup
    chain, thm = prove_r_twisted_right_leibniz(
        N, R, U, om, V, et, f, registry=reg
    )
    assert "[C'3]" in thm.statement


def test_r_twisted_symmetric_part_c4(setup):
    reg, f, h, om, et, U, V, N, R = setup
    chain, thm = prove_r_twisted_symmetric_part(
        N, R, U, om, V, et, registry=reg
    )
    assert "does not change D_θ" in thm.statement


def test_r_term_is_in_the_anchor_kernel(setup):
    # Corrected structural theorem (2026-09-09 audit, finding 2):
    # the vector-valued R-term lies in ker ρ (the (2.6) anchor
    # reads only the form slot), so the anchor morphism SURVIVES
    # the R-twist — as in Watamura §3.
    reg, f, h, om, et, U, V, N, R = setup
    chain = prove_r_twisted_anchor_morphism(
        N, R, U, om, V, et, h, registry=reg
    )
    assert chain.steps


def test_r_twisted_anchor_morphism_needs_poisson(setup):
    reg, f, h, om, et, U, V, N, R = setup
    with pytest.raises(ProofFailure):
        prove_r_twisted_anchor_morphism(
            N, R, U, om, V, et, h,
            registry=reg,
            declare_poisson=False,
        )


def test_derived_r_vanishes_under_poisson(setup):
    reg, f, h, om, et, U, V, N, R = setup
    chain, thm = prove_derived_r_vanishes_under_poisson(
        N, om, et, registry=reg
    )
    assert "R′" in thm.statement


def test_r_twisted_courant_relation(setup):
    reg, f, h, om, et, U, V, N, R = setup
    chain, thm = prove_r_twisted_courant_relation(
        N, R, U, om, V, et, registry=reg
    )
    assert len(chain.steps) == 2


def test_r_twisted_jacobi_defect_is_dtheta_r(setup):
    # The R-side Ševera theorem (2026-09-10): J_R − J_θ = (ι̃ι̃ι̃ d_θR, 0),
    # declaration-free — any bivector, any trivector.
    from jacopy.central.objects import vector_fields as _vf
    from jacopy.packages.generalized.r_twisted import (
        prove_r_twisted_jacobi_defect_is_dtheta_r,
    )

    reg, f, h, om, et, U, V, N, R = setup
    (W,) = _vf("W")
    (ze,) = forms("ζ", degree=1)
    chain, thm = prove_r_twisted_jacobi_defect_is_dtheta_r(
        N, R, U, om, V, et, W, ze, h, registry=reg
    )
    assert len(chain.steps) == 2
    assert "d_θR = 0" in thm.statement
    assert not any(a.startswith("declared") for a in thm.from_axioms)


def test_r_twisted_jacobi_under_declared_closure(setup):
    from jacopy.central.objects import vector_fields as _vf
    from jacopy.packages.generalized.r_twisted import prove_r_twisted_jacobi

    reg, f, h, om, et, U, V, N, R = setup
    W, X = _vf("W X")
    (ze,) = forms("ζ", degree=1)
    chain, thm = prove_r_twisted_jacobi(
        N, R, U, om, V, et, W, ze, h, X, registry=reg
    )
    rules = [s.rule for s in chain.steps]
    assert any("declared closure d_θR = 0" in r for r in rules)
    assert any("CITED" in a for a in thm.from_axioms)


def test_r_twisted_jacobi_records_the_jacobiator_as_target(setup):
    # F5 (2026-09-10 audit): lhs is J_R(h) itself, rhs 0; the chain
    # carries the defect theorem, the declared closure instance and
    # the cited [C'1] components with their real before/after.
    from jacopy.central.objects import vector_fields as _vf
    from jacopy.packages.generalized.r_twisted import prove_r_twisted_jacobi

    reg, f, h, om, et, U, V, N, R = setup
    W, X = _vf("W X")
    (ze,) = forms("ζ", degree=1)
    chain, thm = prove_r_twisted_jacobi(
        N, R, U, om, V, et, W, ze, h, X, registry=reg
    )
    x, y, z = (U, om), (V, et), (W, ze)
    B = lambda a, b: r_twisted_theta_dorfman(N, R, *a, *b)  # noqa: E731
    target = Act(
        Sum(B(x, B(y, z))[0], Neg(B(B(x, y), z)[0]), Neg(B(y, B(x, z))[0])),
        h,
    )
    assert thm.lhs == target and thm.rhs == Integer(0)
    assert len(chain.steps) == 5
    assert chain.steps[0].before == target
    assert chain.steps[0].children  # the defect sub-proof is attached
    assert chain.steps[2].before != Integer(0)  # [C'1] cited on J_θ(h), not 0→0
