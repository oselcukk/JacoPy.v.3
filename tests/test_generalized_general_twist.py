"""PDF items 13j-13k — the GENERAL Ψ-twist procedure, the
independent-A/Z standard construction, and the exceptional Ψ_Π
rotation with every term separated."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.algebroid.context import algebroid
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.poisson.nambu import nambu_structure
from jacopy.packages.generalized.general_twist import (
    prove_general_twist_bilinearity,
    prove_general_twist_invariance,
    prove_general_twist_jacobi,
    prove_general_twist_right_leibniz,
    prove_general_twist_symmetric_part,
)
from jacopy.packages.generalized.exceptional_rotation import (
    prove_rotated_form_decomposition,
    prove_rotated_vector_decomposition,
)
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    # The transports CITE initial-bracket axioms, so the context
    # must genuinely declare them (2026-09-09 audit, finding 1).
    E = algebroid(
        "E",
        declare=(
            "right-leibniz",
            "jacobi",
            "symmetric-part",
            "metric-invariance",
        ),
    )
    u, v, w = E.sections("u v w")
    return reg, f, E, u, v, w


# ---- 13j: the general Ψ procedure, property transport ------------- #


def test_twist_transports_bilinearity(setup):
    reg, f, E, u, v, w = setup
    chain, thm = prove_general_twist_bilinearity(
        E, u, v, w, registry=reg
    )
    assert "invertible Ψ" in thm.statement


def test_twist_transports_right_leibniz_with_primed_anchor(setup):
    reg, f, E, u, v, w = setup
    chain, thm = prove_general_twist_right_leibniz(
        E, u, v, f, registry=reg
    )
    assert "ρ' = ρ∘Ψ" in thm.statement


def test_twist_transports_jacobi(setup):
    from jacopy.core.expr import Integer, Neg, Sum
    from jacopy.packages.generalized.general_twist import (
        twisted_bracket,
    )

    reg, f, E, u, v, w = setup
    chain, thm = prove_general_twist_jacobi(E, u, v, w, registry=reg)
    assert "Leibniz-Jacobi transports" in thm.statement
    # The RECORDED equation is the advertised twisted Jacobiator
    # itself (finding 6), reusable by TheoremDefinition.
    br = lambda a, b: twisted_bracket(E, a, b)
    assert thm.lhs == Sum(
        br(u, br(v, w)),
        Neg(br(br(u, v), w)),
        Neg(br(v, br(u, w))),
    )
    assert thm.rhs == Integer(0)
    assert len(chain.steps) == 2


def test_jacobi_without_declaration_is_conditional(setup):
    from jacopy.core.expr import Integer

    reg, f, E, u, v, w = setup
    E2 = algebroid("E_rl", declare=("right-leibniz",))
    a, b, c = E2.sections("a b c")
    chain, thm = prove_general_twist_jacobi(
        E2, a, b, c, registry=reg
    )
    # No declared jacobi: only the structural identity is
    # recorded — rhs is the Ψ⁻¹-instance, NOT zero.
    assert thm.rhs != Integer(0)
    assert "STRUCTURAL identity only" in thm.statement
    assert "NOT established here" in thm.statement


def test_twist_transports_bourbaki_data(setup):
    from jacopy.core.expr import Integer

    reg, f, E, u, v, w = setup
    # Full theorems need the EXPLICIT R-valued declaration: the
    # central scalar flags certify EMetric, not BMetric/BLieRE
    # (2026-09-09 recheck, finding A).
    chain, thm = prove_general_twist_symmetric_part(
        E, u, v, registry=reg, declare_r_axioms=True
    )
    assert "𝔻' = Ψ⁻¹∘𝔻" in thm.statement
    assert thm.rhs == Integer(0)
    assert any("R-VALUED" in a for a in thm.from_axioms)
    chain2, thm2 = prove_general_twist_invariance(
        E, u, v, w, registry=reg, declare_r_axioms=True
    )
    assert "g' = g(Ψ·,Ψ·)" in " ".join(
        (thm2.statement,) + thm2.from_axioms + (thm2.notes,)
    )
    assert thm2.rhs == Integer(0)


def test_bourbaki_transports_without_r_declaration_are_conditional(
    setup,
):
    from jacopy.core.expr import Integer

    reg, f, E, u, v, w = setup
    # Without declare_r_axioms only the structural identity is
    # recorded: rhs is the Ψ-carried instance, NOT zero.
    for chain, thm in (
        prove_general_twist_symmetric_part(E, u, v, registry=reg),
        prove_general_twist_invariance(E, u, v, w, registry=reg),
    ):
        assert thm.rhs != Integer(0)
        assert "STRUCTURAL identity only" in thm.statement
        assert "NOT established here" in thm.statement
        assert len(chain.steps) == 1


def test_right_leibniz_transport_needs_the_initial_axiom(setup):
    reg, f, E, u, v, w = setup
    bare = algebroid("E2")
    a, b = bare.sections("a b")
    with pytest.raises(ProofFailure):
        prove_general_twist_right_leibniz(bare, a, b, f, registry=reg)


def test_cited_transports_need_their_initial_axioms(setup):
    reg, f, E, u, v, w = setup
    bare = algebroid("bare_E")
    a, b, c = bare.sections("a b c")
    with pytest.raises(ProofFailure):
        prove_general_twist_jacobi(bare, a, b, c, registry=reg)
    with pytest.raises(ProofFailure):
        prove_general_twist_symmetric_part(
            bare, a, b, registry=reg
        )
    with pytest.raises(ProofFailure):
        prove_general_twist_invariance(
            bare, a, b, c, registry=reg
        )


# ---- 13k: independent A/Z data ------------------------------------ #


def test_standard_construction_on_independent_a(setup):
    from jacopy.packages.generalized.bourbaki_precalculus import (
        BourbakiPreCalculus,
        prove_standard_metric_invariance,
        prove_standard_symmetric_part,
    )

    reg, f, E, u, v, w = setup
    A = algebroid("A", declare=("antisymmetric",))
    B = BourbakiPreCalculus()
    a, b, c = A.sections("a b c")
    z, y, x = B.z_sections("z y x")
    be = algebroid_engine(A, registry=reg)
    assert prove_standard_symmetric_part(
        a, z, b, y,
        registry=reg,
        bracket=A.bracket,
        base_engine=be,
    )[0].steps
    assert prove_standard_metric_invariance(
        a, z, b, y, c, x,
        registry=reg,
        bracket=A.bracket,
        base_engine=be,
    )[0].steps


# ---- 13k: the exceptional Ψ_Π rotation ---------------------------- #


@pytest.fixture()
def exc():
    reg = PropertyRegistry()
    U, V = vector_fields("U V")
    om2, et2 = forms("ω₂ η₂", degree=2)
    om5, et5 = forms("ω₅ η₅", degree=5)
    N3 = nambu_structure("Π₃", p=2)
    N6 = nambu_structure("Π₆", p=5)
    return reg, U, V, om2, et2, om5, et5, N3, N6


def test_exceptional_rotation_form_slots_separate(exc):
    reg, U, V, om2, et2, om5, et5, N3, N6 = exc
    chain, thm = prove_rotated_form_decomposition(
        N3, N6, U, om2, om5, V, et2, et5, registry=reg
    )
    assert "Ψ_Π-invariant" in thm.statement
    assert len(chain.steps) == 2


def test_exceptional_rotation_vector_slot_separates(exc):
    reg, U, V, om2, et2, om5, et5, N3, N6 = exc
    chain, thm = prove_rotated_vector_decomposition(
        N3, N6, U, om2, om5, V, et2, et5, registry=reg
    )
    assert "⊛(form₅')" in thm.statement
