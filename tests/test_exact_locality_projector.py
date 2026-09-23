"""MC Prop 4.2 on the exact Courant algebroid (ledger K3.b,
2026-09-23): the projector is the identity on ker ρ = T*M (the paper's
argument, mechanical), and the B-field projector shows the TM part
is not determined — a certified non-uniqueness."""

import pytest

from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.objects.endomorphism import EndoForm
from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.packages.generalized.exact_locality_projector import (
    _engine,
    _norm,
    b_field_projector,
    locally_exact_form,
    prove_b_field_projector_is_a_locality_projector,
    prove_projector_fixes_locally_exact_forms,
)
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def cast():
    reg = PropertyRegistry()
    f1, f2, g1, g2, h = functions("f₁ f₂ g₁ g₂ h", registry=reg)
    (U,) = vector_fields("U")
    (om,) = forms("ω", degree=1)
    (B,) = forms("B", degree=2)
    return reg, f1, f2, g1, g2, h, U, om, B


def test_projector_is_identity_on_locally_exact_forms(cast):
    reg, f1, f2, g1, g2, h, U, om, B = cast
    chain, thm = prove_projector_fixes_locally_exact_forms(
        (f1, f2), (g1, g2), registry=reg
    )
    assert len(chain.steps) == 1
    assert "ker ρ" in thm.statement
    assert any(a.startswith("declared") for a in thm.from_axioms)


def test_absorption_hypothesis_is_essential(cast):
    # without the declared C(dg) = dg the map is an arbitrary
    # C∞-linear endomorphism and the identity is false
    reg, f1, f2, g1, g2, h, U, om, B = cast
    from jacopy.central.tangent.exterior import d

    node = EndoForm("𝒫", d(g1))
    eng = _engine(reg)  # no projector declaration: 𝒫(dg) stays inert
    assert eng.expand(node)[0] == node
    eng_decl = _engine(reg, projector_name="𝒫")
    assert eng_decl.expand(node)[0] == d(g1)
    with pytest.raises(ValueError):
        locally_exact_form((f1,), (g1, g2))


def test_b_field_projector_satisfies_every_requirement_but_is_not_pr2(cast):
    reg, f1, f2, g1, g2, h, U, om, B = cast
    chain, thm = prove_b_field_projector_is_a_locality_projector(
        B, U, om, h, g1, registry=reg
    )
    assert len(chain.steps) == 7
    last = chain.steps[-1]
    assert last.after != Integer(0) and "ι_U(B)" in last.after._repr_inner()
    assert "holds on ker ρ only" in thm.statement


def test_b_field_finding_collapses_for_b_zero(cast):
    # with B = 0 the deformation is pr₂ itself: the prover refuses to
    # certify a non-identity (honesty of the finding)
    reg, f1, f2, g1, g2, h, U, om, B = cast
    with pytest.raises(ProofFailure, match="collapsed"):
        prove_b_field_projector_is_a_locality_projector(
            Integer(0), U, om, h, g1, registry=reg
        )


def test_b_field_projector_shape(cast):
    reg, f1, f2, g1, g2, h, U, om, B = cast
    v, f = b_field_projector(B, U, om)
    assert v == Integer(0) and f._repr_inner() == "(ω + ι_U(B))"
