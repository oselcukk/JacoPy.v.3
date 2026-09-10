"""The component face of the multivector interior (2026-09-10):
ι_P on a decomposable form, its sign consistency with the
decomposable-multivector rule, the sharp's pairing face at p ≥ 2,
and the (4.13) ↔ ⊛ frame-component identity."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wedge import Wedge
from jacopy.central.objects import forms, frame, functions, vector_fields
from jacopy.central.objects.multivector_interior import (
    MultivectorInterior,
    MultivectorInteriorDecomposableDefinition,
    MultivectorInteriorDecomposableFormDefinition,
    MultivectorInteriorLinearityDefinition,
    MultivectorWedgeEvalDefinition,
)
from jacopy.packages.drinfeld.tilde_calculus import (
    NambuSharpPairingEvalDefinition,
    _tilde_engine,
)
from jacopy.packages.generalized.frame_evaluation import (
    prove_boxtimes_components_are_413,
)
from jacopy.packages.poisson.nambu import nambu_structure
from jacopy.packages.poisson.tilde import _normalized_by
from jacopy.proof.strategies import ProofFailure


@pytest.mark.parametrize("p,q", [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)])
def test_interior_on_decomposable_form_agrees_with_decomposable_multivector(p, q):
    # ι_{X₁∧…∧X_p}(α₁∧…∧α_q): route A (nested ordinary interiors)
    # and route B (the Σ_I sgn(I,Iᶜ) P(α_I) α_{Iᶜ} component face)
    # must give the same (q−p)-form.
    reg = PropertyRegistry()
    N = nambu_structure(p=2)
    Xs = vector_fields(" ".join(f"X{i}" for i in range(p)))
    als = forms(" ".join(f"α{i}" for i in range(q)), degree=1)
    slots = vector_fields(" ".join(f"Y{i}" for i in range(q - p))) if q > p else ()
    eng_a = _tilde_engine(N, reg, declare_fi=False)
    eng_a.register(MultivectorInteriorLinearityDefinition(reg))
    eng_a.register(MultivectorInteriorDecomposableDefinition())
    eng_b = _tilde_engine(N, reg, declare_fi=False)
    eng_b.register(MultivectorInteriorLinearityDefinition(reg))
    eng_b.register(MultivectorInteriorDecomposableFormDefinition(reg))
    eng_b.register(MultivectorWedgeEvalDefinition())
    node = Act(MultivectorInterior(Wedge(*Xs)), Wedge(*als))
    ev = (
        (lambda x: MultiEval(x, *slots, alternating=True, slot_kind="vector"))
        if slots
        else (lambda x: x)
    )
    a = _normalized_by(eng_a, ev(node), reg)
    b = _normalized_by(eng_b, ev(node), reg)
    assert _normalized_by(eng_b, Sum(a, Neg(b)), reg) == Integer(0)


def test_form_rule_stays_inert_on_opaque_forms_and_at_p1():
    reg = PropertyRegistry()
    rule = MultivectorInteriorDecomposableFormDefinition(reg)
    N3 = nambu_structure(p=2)
    (om5,) = forms("Ω5", degree=5)
    assert not rule.matches(Act(MultivectorInterior(N3.pi), om5))
    X, = vector_fields("Xv")
    a, b = forms("a b", degree=1)
    assert not rule.matches(Act(MultivectorInterior(X), Wedge(a, b)))


def test_form_rule_gives_the_scalar_evaluation_when_degrees_match():
    reg = PropertyRegistry()
    rule = MultivectorInteriorDecomposableFormDefinition(reg)
    N3 = nambu_structure(p=2)
    a, b, c = forms("a b c", degree=1)
    out = rule.rewrite(Act(MultivectorInterior(N3.pi), Wedge(a, b, c)))
    assert out == MultiEval(N3.pi, a, b, c, alternating=True, slot_kind="covector")


def test_sharp_pairing_face_at_p2_gives_components():
    reg = PropertyRegistry()
    N3 = nambu_structure("Π₃", p=2)
    fr = frame()
    co = fr.dual()
    rule = NambuSharpPairingEvalDefinition(N3, reg)
    node = Pairing(co.field("c"), N3.sharp_vf(Wedge(co.field("a"), co.field("b"))))
    assert rule.matches(node)
    assert rule.rewrite(node) == MultiEval(
        N3.pi, co.field("a"), co.field("b"), co.field("c"),
        alternating=True, slot_kind="covector",
    )
    (om2,) = forms("Ω2", degree=2)
    assert not rule.matches(Pairing(co.field("c"), N3.sharp_vf(om2)))


def test_boxtimes_components_are_the_413_formula():
    reg = PropertyRegistry()
    N3 = nambu_structure("Π₃", p=2)
    chain, thm = prove_boxtimes_components_are_413(
        N3, frame(), ("a1", "a2", "a3", "a4", "a5"), "c", registry=reg
    )
    assert len(chain.steps) == 3
    assert "5Π^{[a₁a₂a₃}Π^{a₄a₅]c}" in thm.statement
    # the canonical form is the 10 ½-weighted splits, not 120 terms
    assert "1/2" in chain.steps[0].after._repr_inner()


def test_boxtimes_413_rejects_wrong_data():
    reg = PropertyRegistry()
    with pytest.raises(ValueError):
        prove_boxtimes_components_are_413(
            nambu_structure(p=5), frame(), ("a1", "a2", "a3", "a4", "a5"), "c", registry=reg
        )
    with pytest.raises(ValueError):
        prove_boxtimes_components_are_413(
            nambu_structure(p=2), frame(), ("a1", "a2"), "c", registry=reg
        )
