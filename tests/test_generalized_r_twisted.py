"""Phase 7.B.2c — the R-twisted tilde-Dorfman bracket (PDF item
14f.iii, Watamura R-flux): the dual of the H-twist on the Poisson
generalized double, with the anchor-defect structural theorem."""

import pytest

from jacopy.core.expr import Integer, Sum
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
    prove_r_anchor_defect,
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


def test_r_anchor_defect_is_exactly_the_r_term(setup):
    # The structural theorem: the R-flux breaks the anchor morphism
    # by exactly ι̃_η ι̃_ω R (quasi-Courant picture).
    reg, f, h, om, et, U, V, N, R = setup
    chain = prove_r_anchor_defect(
        N, R, U, om, V, et, h, registry=reg
    )
    assert len(chain.steps) > 20


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
