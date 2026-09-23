"""MC Prop 4.2 on the exact Courant algebroid (ledger K3.b, 2026-09-23;
audit 0058df0 folded in): the projector is the identity on ker ρ (the
paper's argument), every bundle map C: TM → T*M gives a locality
projector P_C (B-fields are the antisymmetric subfamily), and the
non-uniqueness is certified by a frame witness, not by a residual."""

import pytest

from jacopy.central.objects import Tensor, forms, frame, functions, vector_fields
from jacopy.central.objects.endomorphism import EndoForm
from jacopy.central.tangent.exterior import d
from jacopy.core.expr import Integer
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wedge import Wedge
from jacopy.packages.generalized.exact_locality_projector import (
    _engine,
    b_field_projector,
    linear_projector,
    locally_exact_form,
    prove_b_field_projector_is_a_locality_projector,
    prove_linear_projector_is_a_locality_projector,
    prove_projector_differs_from_pr2_on_a_frame,
    prove_projector_fixes_locally_exact_forms,
)
from jacopy.packages.poisson.nambu import nambu_structure
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def cast():
    reg = PropertyRegistry()
    f1, f2, g1, g2, h = functions("f₁ f₂ g₁ g₂ h", registry=reg)
    U, V = vector_fields("U V")
    om, et = forms("ω η", degree=1)
    (B,) = forms("B", degree=2)
    return reg, f1, f2, g1, g2, h, U, V, om, et, B


def test_projector_is_identity_on_locally_exact_forms(cast):
    reg, f1, f2, g1, g2, h, U, V, om, et, B = cast
    chain, thm = prove_projector_fixes_locally_exact_forms((f1, f2), (g1, g2), registry=reg)
    assert len(chain.steps) == 1
    assert "ker ρ" in thm.statement
    assert any(a.startswith("declared") for a in thm.from_axioms)


def test_absorption_hypothesis_is_essential(cast):
    reg, f1, f2, g1, g2, h, U, V, om, et, B = cast
    node = EndoForm("𝒫", d(g1))
    assert _engine(reg).expand(node)[0] == node  # inert without the declaration
    assert _engine(reg, projector_name="𝒫").expand(node)[0] == d(g1)
    with pytest.raises(ValueError):
        locally_exact_form((f1,), (g1, g2))
    with pytest.raises(TypeError):
        prove_projector_fixes_locally_exact_forms((U,), (g1,), registry=reg)


def test_general_bundle_map_gives_a_locality_projector(cast):
    # F3: the family is P_C for ANY (0,2)-tensor C, not only B-fields
    reg, f1, f2, g1, g2, h, U, V, om, et, B = cast
    C = Tensor("C", upper=0, lower=2)
    chain, thm = prove_linear_projector_is_a_locality_projector(C, U, om, V, et, h, g1, registry=reg)
    assert len(chain.steps) == 6
    assert "every bundle map" in thm.statement
    assert "DIFFERENT" in chain.steps[1].rule
    # a 2-form IS a (0,2)-tensor: the B-field subfamily sits inside P_C
    assert linear_projector(B, U, om)[0] == Integer(0)
    with pytest.raises(TypeError):
        linear_projector(om, U, om)  # a 1-form is not a bundle map TM → T*M


def test_b_field_subfamily_is_a_locality_projector(cast):
    reg, f1, f2, g1, g2, h, U, V, om, et, B = cast
    chain, thm = prove_b_field_projector_is_a_locality_projector(B, U, om, h, g1, V=V, eta=et, registry=reg)
    assert len(chain.steps) == 6
    assert "membership only" in thm.notes
    # B = 0 is pr₂ itself and is (correctly) a locality projector too
    chain0, _ = prove_b_field_projector_is_a_locality_projector(Integer(0), U, om, h, g1, registry=reg)
    assert len(chain0.steps) == 6


@pytest.mark.parametrize("degree", [1, 3])
def test_wrong_b_degree_is_refused(cast, degree):
    # F2: ω + ι_U B is not a 1-form unless B is a 2-form
    reg, f1, f2, g1, g2, h, U, V, om, et, B = cast
    (bad,) = forms("Bbad", degree=degree)
    with pytest.raises(TypeError):
        prove_b_field_projector_is_a_locality_projector(bad, U, om, h, g1, registry=reg)
    with pytest.raises(TypeError):
        b_field_projector(bad, U, om, registry=reg)
    with pytest.raises(TypeError):
        prove_b_field_projector_is_a_locality_projector(B, om, om, h, g1, registry=reg)  # U not a vector


def test_non_uniqueness_is_witnessed_not_residual(cast):
    # F1: the certificate is a literal 1 on a frame, not an unreduced residual
    reg, f1, f2, g1, g2, h, U, V, om, et, B = cast
    chain, thm = prove_projector_differs_from_pr2_on_a_frame(frame(), dim=2, registry=reg)
    assert chain.steps[1].after == Integer(1) and thm.rhs == Integer(1)
    assert "ker ρ only" in thm.statement
    with pytest.raises(ValueError):
        prove_projector_differs_from_pr2_on_a_frame(frame(), dim=1, registry=reg)


def test_zero_gap_from_alternation_is_not_certified(cast):
    # the auditor's counterexample: B = df∧dg, U = Π(B) — the gap
    # U(f)dg − U(g)df vanishes by alternation; the membership theorem
    # still holds (it claims nothing about the gap) and no function
    # certifies P_B ≠ pr₂ for this U
    reg, f1, f2, g1, g2, h, U, V, om, et, B = cast
    N = nambu_structure("Π", p=2)
    Bfg = Wedge(d(f1), d(g1))
    UB = N.sharp_vf(Bfg)
    chain, thm = prove_b_field_projector_is_a_locality_projector(Bfg, UB, om, h, g2, registry=reg)
    assert all("non-identity" not in s.justification for s in chain.steps)
    assert "≠" not in thm.statement


def test_inputs_are_typed_by_kind_not_degree(cast):
    # audit 7a11162 R1/R2/R4/R5
    from jacopy.central.objects import Bundle, Form, PVector, frame
    from jacopy.central.objects.bundle import TangentBundle
    from jacopy.core.expr import Product, Sum

    reg, f1, f2, g1, g2, h, U, V, om, et, B = cast
    with pytest.raises(TypeError):
        prove_b_field_projector_is_a_locality_projector(PVector("Π", degree=2), U, om, h, g1, registry=reg)
    with pytest.raises(TypeError):
        prove_b_field_projector_is_a_locality_projector(Form("B_F", degree=2, bundle=Bundle("F")), U, om, h, g1, registry=reg)
    for good in (Integer(0), Product(h, U), Sum(U, V)):
        chain, _ = prove_b_field_projector_is_a_locality_projector(B, good, om, h, g1, registry=reg)
        assert len(chain.steps) == 6
    chain, _ = prove_linear_projector_is_a_locality_projector(Integer(0), U, om, V, et, h, g1, registry=reg)
    assert len(chain.steps) == 6
    with pytest.raises(ValueError):
        prove_projector_differs_from_pr2_on_a_frame(frame(bundle=TangentBundle(dim=1)), dim=2, registry=reg)
    chain, thm = prove_projector_differs_from_pr2_on_a_frame(frame(bundle=TangentBundle(dim=3)), dim=3, registry=reg)
    assert thm.rhs == Integer(1)
