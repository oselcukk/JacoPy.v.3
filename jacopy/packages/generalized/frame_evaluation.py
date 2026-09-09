"""
Frame/coframe evaluation of the generalized structures (Phase
7.B.2e; PDF item 14h): the canonical pairing, the Dorfman bracket
and the total anchor evaluated on BASIS sections ``e_a ⊕ e^b`` —
every scalar output lands on the named component functions of the
existing frame layer (Kronecker deltas ``δ``, anholonomy ``γ^c_ab``,
tilde anholonomy ``γ̃_c^{ab}``, bivector components ``θ(e^a, e^b)``).

Component dictionary produced here:

* ``⟨e_a ⊕ e^b, e_c ⊕ e^d⟩₊ = δ^b_c + δ^d_a`` — the pairing is the
  natural duality pairing in the split basis;
* ``⟨e^u, vec[e_a⊕0, e_c⊕0]⟩ = γ^u_ac`` — the vector side of the
  standard Dorfman bracket reproduces the anholonomy (Phase 2.E);
* ``form[e_a⊕0, 0⊕e^d](e_b) = −γ^d_ab`` — the ``ℒ`` side of the
  Dorfman form component is the Maurer-Cartan/coframe-differential
  face of the same coefficients;
* ``⟨form[0⊕e^a, 0⊕e^b]_θ, e_c⟩ = γ̃_c^{ab}`` — the Koszul side of
  the Poisson generalized double reproduces the TILDE anholonomy
  (PDF 12i, now read as a 14h component);
* ``⟨e^c, ρ(0 ⊕ e^b)⟩ = θ(e^b, e^c)`` — the (2.6) anchor
  ``ρ(U+ω) = θ♯ω`` decomposes into the bivector components on the
  coframe (pure frame vectors are killed: ``ρ(e_a ⊕ 0) = 0``).

Everything is engine-normalized; honest-fail throughout.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.objects.frame import Frame, KroneckerDelta
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.anholonomy import (
    anholonomy_coefficient,
)
from jacopy.central.tangent.cartan import L as _L
from jacopy.central.tangent.engine import tangent_engine
from jacopy.central.tangent.exterior import d
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.poisson.core import (
    PoissonStructure,
    SharpVF,
)
from jacopy.packages.poisson.koszul import KoszulBracket
from jacopy.packages.poisson.tilde import (
    TildeAnholonomyDefinition,
    tilde_anholonomy_coefficient,
)


def _iota(X: Expr, x: Expr) -> Expr:
    return Act(Interior(X), x)


def basis_pairing(
    fr: Frame, a, b_up, c, d_up
) -> Tuple[Expr, Expr]:
    """``(⟨e_a ⊕ e^b, e_c ⊕ e^d⟩₊, δ^b_c + δ^d_a)`` — the
    evaluated pairing and its component value."""
    co = fr.dual()
    lhs = Sum(
        _iota(fr.field(a), co.field(d_up)),
        _iota(fr.field(c), co.field(b_up)),
    )
    rhs = Sum(
        KroneckerDelta(str(b_up), str(c)),
        KroneckerDelta(str(d_up), str(a)),
    )
    return lhs, rhs


def _zero_theorem(
    name,
    statement,
    diff,
    engine,
    registry,
    *,
    from_axioms,
    notes,
    label,
) -> Tuple[ProofChain, Theorem]:
    from jacopy.packages.poisson.tilde import _normalized_by

    nf = _normalized_by(engine, diff, registry)
    if nf != Integer(0):
        raise ProofFailure(
            f"{name}: {label} FAILS — residual "
            + nf._repr_inner()[:160]
        )
    step = ProofStep(
        diff,
        Integer(0),
        rule=label,
        justification="engine normal form of the difference",
    )
    chain = ProofChain([step])
    theorem = Theorem(
        name=name,
        statement=statement,
        lhs=diff,
        rhs=Integer(0),
        proof=chain,
        generality="instance",
        from_axioms=from_axioms,
        notes=notes,
    )
    return chain, theorem


def prove_basis_pairing_components(
    fr: Frame,
    a,
    b_up,
    c,
    d_up,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``⟨e_a ⊕ e^b, e_c ⊕ e^d⟩₊ = δ^b_c + δ^d_a`` — frame duality
    evaluates the canonical pairing to Kronecker components."""
    lhs, rhs = basis_pairing(fr, a, b_up, c, d_up)
    return _zero_theorem(
        "generalized_basis_pairing",
        "⟨e_a ⊕ e^b, e_c ⊕ e^d⟩₊ = δ^b_c + δ^d_a (PDF 14h)",
        Sum(lhs, Neg(rhs)),
        tangent_engine(registry=registry),
        registry,
        from_axioms=("frame duality (definitional)",),
        notes="the split-basis component of the pairing",
        label="pairing components normalize to 0",
    )


def prove_dorfman_vector_components(
    fr: Frame,
    upper,
    a,
    c,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``⟨e^u, vec[e_a⊕0, e_c⊕0]_D⟩ = γ^u_ac`` — the vector side of
    the standard Dorfman bracket on pure frame sections IS the Lie
    bracket, whose frame components are the anholonomy
    coefficients."""
    co = fr.dual()
    lhs = Pairing(
        co.field(upper), lie_bracket(fr.field(a), fr.field(c))
    )
    rhs = anholonomy_coefficient(fr, upper, a, c)
    return _zero_theorem(
        "generalized_dorfman_vector_components",
        "⟨e^u, vec[e_a⊕0, e_c⊕0]_D⟩ = γ^u_ac (PDF 14h; the "
        "Dorfman vector side on the frame is the anholonomy)",
        Sum(lhs, Neg(rhs)),
        tangent_engine(registry=registry),
        registry,
        from_axioms=(
            "Dorfman definition (vector = Lie bracket)",
            "anholonomy coefficient extraction (2.E)",
        ),
        notes="PDF 9b read as a 14h component",
        label="vector components normalize to 0",
    )


def prove_dorfman_form_components(
    fr: Frame,
    a,
    d_up,
    b,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``form[e_a⊕0, 0⊕e^d]_D (e_b) = −γ^d_ab`` — the Dorfman form
    component on a mixed basis pair is ``ℒ_{e_a} e^d``, whose frame
    evaluation is the Maurer-Cartan face of the anholonomy."""
    co = fr.dual()
    # form component of [e_a ⊕ 0, 0 ⊕ e^d]: ℒ_{e_a} e^d (the ℒ_V-
    # and dι_V-terms vanish for a zero second vector leg).
    form = _L(fr.field(a), co.field(d_up))
    lhs = Pairing(form, fr.field(b))
    rhs = Neg(anholonomy_coefficient(fr, d_up, a, b))
    return _zero_theorem(
        "generalized_dorfman_form_components",
        "form[e_a⊕0, 0⊕e^d]_D (e_b) = −γ^d_ab (PDF 14h; the "
        "ℒ-side of the Dorfman form component on the basis)",
        Sum(lhs, Neg(rhs)),
        tangent_engine(registry=registry),
        registry,
        from_axioms=(
            "Dorfman definition (form ℒ-term)",
            "Cartan magic + frame duality + anholonomy (2.E)",
        ),
        notes="the coframe-differential face de^d(e_a,e_b) = −γ^d_ab",
        label="form components normalize to 0",
    )


def prove_theta_koszul_components(
    P: PoissonStructure,
    fr: Frame,
    a_up,
    b_up,
    c,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``⟨form[0⊕e^a, 0⊕e^b]_θ, e_c⟩ = γ̃_c^{ab}`` — the Koszul side
    of the Poisson generalized double on pure coframe sections
    reproduces the TILDE anholonomy coefficients (PDF 12i as a 14h
    component)."""
    co = fr.dual()
    lhs = Pairing(
        KoszulBracket(P.pi, co.field(a_up), co.field(b_up)),
        fr.field(c),
    )
    rhs = tilde_anholonomy_coefficient(P, fr, a_up, b_up, c)
    engine = tangent_engine(registry=registry)
    engine.register(TildeAnholonomyDefinition(P, fr))
    return _zero_theorem(
        "generalized_theta_koszul_components",
        "⟨form[0⊕e^a, 0⊕e^b]_θ, e_c⟩ = γ̃_c^{ab} (PDF 14h; the "
        "θ-double's Koszul side on the coframe is the tilde "
        "anholonomy)",
        Sum(lhs, Neg(rhs)),
        engine,
        registry,
        from_axioms=(
            "θ-double definition (form = Koszul bracket)",
            "tilde anholonomy extraction (12i)",
        ),
        notes="Watamura components on the coframe",
        label="tilde components normalize to 0",
    )


def prove_anchor_components(
    P: PoissonStructure,
    fr: Frame,
    b_up,
    c_up,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``⟨e^c, ρ(0 ⊕ e^b)⟩ = θ(e^b, e^c)`` — the (2.6) anchor
    ``ρ(U+ω) = θ♯ω`` decomposes into the BIVECTOR COMPONENTS on
    the coframe."""
    co = fr.dual()
    lhs = Pairing(
        co.field(c_up), SharpVF(P.pi, co.field(b_up))
    )
    rhs = MultiEval(
        P.pi,
        co.field(b_up),
        co.field(c_up),
        alternating=True,
        slot_kind="covector",
    )
    from jacopy.packages.poisson.core import (
        SharpEvaluationDefinition,
    )

    engine = tangent_engine(registry=registry)
    engine.register(SharpEvaluationDefinition(P))
    return _zero_theorem(
        "generalized_anchor_components",
        "⟨e^c, ρ(0 ⊕ e^b)⟩ = θ(e^b, e^c) (PDF 14h; the total "
        "anchor's form leg = bivector components)",
        Sum(lhs, Neg(rhs)),
        engine,
        registry,
        from_axioms=(
            "total anchor + sharp evaluation definitions",
        ),
        notes="ρ(0 ⊕ e^b) = θ♯e^b on the basis (Watamura (2.6))",
        label="anchor components normalize to 0",
    )
