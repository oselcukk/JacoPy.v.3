"""
The GENERAL Ψ-twist procedure (PDF item 13j — "This is really
important!"): given ANY bracket on ``E`` satisfying a set of
algebroid properties and ANY invertible bundle morphism
``Ψ : E → E``, the twisted bracket

    [u, v]' := Ψ⁻¹ [Ψu, Ψv]_E

inherits every property with the PRIMED operators separated out:

    ρ'  = ρ ∘ Ψ,           g'(u,v)  = g(Ψu, Ψv),
    𝔻'  = Ψ⁻¹ ∘ 𝔻,         𝕃'_{df} = Ψ⁻¹ ∘ 𝕃_{df},

mechanized here at FULL generality — ``Ψ`` is an opaque C∞-linear
morphism with an opaque inverse (only ``Ψ⁻¹Ψ = ΨΨ⁻¹ = id`` and
linearity are used), the bracket is the opaque Phase 3 algebroid
bracket, and the R-VALUED metric layer is the 7.B.2f structure.
Property-transport theorems:

* ℝ-bilinearity of ``[·,·]'`` (structural);
* right-Leibniz transports with anchor ``ρ' = ρ∘Ψ``;
* Leibniz-Jacobi transports (the twisted Jacobiator is
  ``Ψ⁻¹`` of the original's, cited as a declared instance);
* the symmetric part transports as ``[u,v]' + [v,u]' =
  𝔻'g'(u,v)`` — the primed Bourbaki data;
* metric invariance transports with ``ℒ^R`` unchanged (its
  direction reads through ``ρ'``).

The 6.E/7.B concrete twists (Ψ_Π, Ψ_B, Ψ_m, H/R) are instances;
this module is the missing GENERAL procedure of 13j, with each
primed operator separated exactly as the PDF asks.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    Product,
    Sum,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.algebroid.context import (
    Algebroid,
    algebroid,
)
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.packages.generalized.bourbaki_precalculus import (
    _SlotAtom,
    _split_scalar,
)
from jacopy.packages.generalized.bourbaki_structure import (
    BDop,
    BMetric,
)


class PsiSec(_SlotAtom):
    """``Ψ(u) ∈ 𝔛(E)`` — the opaque twist morphism applied to a
    section."""

    def __init__(self, u: Expr) -> None:
        super().__init__(f"Ψ({u._repr_inner()})", u)

    @property
    def u(self) -> Expr:
        return self._slots[0]

    def with_slots(self, u: Expr) -> "PsiSec":
        return PsiSec(u)


class PsiInvSec(_SlotAtom):
    """``Ψ⁻¹(u) ∈ 𝔛(E)`` — the opaque inverse."""

    def __init__(self, u: Expr) -> None:
        super().__init__(f"Ψ⁻¹({u._repr_inner()})", u)

    @property
    def u(self) -> Expr:
        return self._slots[0]

    def with_slots(self, u: Expr) -> "PsiInvSec":
        return PsiInvSec(u)


class PsiLinearityDefinition(Definition):
    """C∞-linearity of ``Ψ`` and ``Ψ⁻¹`` (bundle morphisms): sums,
    negations, zeros and scalar factors pull out."""

    name = "Ψ/Ψ⁻¹ C∞-linearity (bundle morphisms)"
    anchor = (PsiSec, PsiInvSec)

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, (PsiSec, PsiInvSec)):
            return False
        (s,) = expr.rewritable_slots
        return (
            isinstance(s, (Sum, Neg))
            or s == Integer(0)
            or _split_scalar(s, self._registry) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        (s,) = expr.rewritable_slots
        if s == Integer(0):
            return Integer(0)
        if isinstance(s, Sum):
            return Sum(
                *(expr.with_slots(c) for c in s.children)
            )
        if isinstance(s, Neg):
            return Neg(expr.with_slots(s.arg))
        f, rest = _split_scalar(s, self._registry)
        return Product(f, expr.with_slots(rest))


class PsiInverseDefinition(Definition):
    """``Ψ⁻¹(Ψ(u)) → u`` and ``Ψ(Ψ⁻¹(u)) → u`` — the ONLY structural
    assumption on the twist: invertibility."""

    name = "Ψ invertibility: Ψ⁻¹Ψ = ΨΨ⁻¹ = id"
    anchor = (PsiSec, PsiInvSec)

    def matches(self, expr: Expr) -> bool:
        if isinstance(expr, PsiInvSec):
            return isinstance(expr.u, PsiSec)
        if isinstance(expr, PsiSec):
            return isinstance(expr.u, PsiInvSec)
        return False

    def rewrite(self, expr: Expr) -> Expr:
        return expr.u.u


def twisted_bracket(
    alg: Algebroid, u: Expr, v: Expr
) -> Expr:
    """(13j): ``[u, v]' := Ψ⁻¹ [Ψu, Ψv]_E``."""
    return PsiInvSec(
        alg.bracket(PsiSec(u), PsiSec(v))
    )


def twisted_anchor(alg: Algebroid, u: Expr) -> Expr:
    """``ρ'(u) = ρ(Ψu)`` — the primed anchor."""
    return alg.anchor(PsiSec(u))


def twisted_metric(u: Expr, v: Expr) -> Expr:
    """``g'(u, v) = g(Ψu, Ψv)`` — the primed R-valued metric."""
    return BMetric(PsiSec(u), PsiSec(v))


def twisted_d(r: Expr) -> Expr:
    """``𝔻'(r) = Ψ⁻¹(𝔻 r)`` — the primed coboundary."""
    return PsiInvSec(BDop(r))


def general_twist_engine(
    alg: Algebroid,
    registry: Optional[PropertyRegistry] = None,
    *,
    structure_rules: bool = True,
) -> ExpansionEngine:
    """Ψ rules over the Phase 3 declaration engine (and optionally
    the 7.B.2f R-valued structure rules)."""
    rules: List[Definition] = [
        PsiInverseDefinition(),
        PsiLinearityDefinition(registry),
    ]
    eng = ExpansionEngine(rules)
    if structure_rules:
        from jacopy.packages.generalized.bourbaki_structure import (
            bourbaki_structure_engine,
        )

        base = bourbaki_structure_engine(
            alg, registry, invariance=False
        )
    else:
        base = algebroid_engine(alg, registry=registry)
    for d_ in base.definitions:
        eng.register(d_)
    return eng


def _normalize(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


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
    extra_steps=(),
) -> Tuple[ProofChain, Theorem]:
    nf = _normalize(engine, diff, registry)
    if nf != Integer(0):
        raise ProofFailure(
            f"{name}: {label} FAILS — residual "
            + nf._repr_inner()[:160]
        )
    steps = list(extra_steps) + [
        ProofStep(
            diff,
            Integer(0),
            rule=label,
            justification="engine normal form",
        )
    ]
    chain = ProofChain(steps)
    theorem = Theorem(
        name=name,
        statement=statement,
        lhs=diff,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=from_axioms,
        notes=notes,
    )
    return chain, theorem


def prove_general_twist_bilinearity(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``[u + w, v]' = [u,v]' + [w,v]'`` — ℝ-bilinearity transports
    through any invertible C∞-linear ``Ψ`` (structural)."""
    engine = general_twist_engine(
        alg, registry, structure_rules=False
    )
    diff = Sum(
        twisted_bracket(alg, Sum(u, w), v),
        Neg(twisted_bracket(alg, u, v)),
        Neg(twisted_bracket(alg, w, v)),
    )
    return _zero_theorem(
        f"general_twist_bilinearity_{alg.name}",
        "[u + w, v]' = [u,v]' + [w,v]' — bilinearity transports "
        "through any invertible Ψ (13j)",
        diff,
        engine,
        registry,
        from_axioms=(
            "Ψ/Ψ⁻¹ C∞-linearity",
            "bracket ℝ-bilinearity (definitional)",
        ),
        notes="PDF 13j general procedure",
        label="bilinearity defect normalizes to 0",
    )


def prove_general_twist_right_leibniz(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Right-Leibniz transports with the PRIMED anchor:

    ``[u, f·v]' = f·[u,v]' + ρ'(u)(f)·v``,  ``ρ' = ρ∘Ψ``

    — requires the INITIAL bracket's declared right-Leibniz."""
    if not alg.declares("right-leibniz"):
        raise ProofFailure(
            "the transport needs the initial bracket's declared "
            "right-Leibniz"
        )
    engine = general_twist_engine(
        alg, registry, structure_rules=False
    )
    diff = Sum(
        twisted_bracket(alg, u, Product(f, v)),
        Neg(Product(f, twisted_bracket(alg, u, v))),
        Neg(
            Product(Act(twisted_anchor(alg, u), f), v)
        ),
    )
    return _zero_theorem(
        f"general_twist_right_leibniz_{alg.name}",
        "[u, f·v]' = f·[u,v]' + ρ'(u)(f)·v with ρ' = ρ∘Ψ — "
        "right-Leibniz transports through Ψ with the primed "
        "anchor separated out (13j)",
        diff,
        engine,
        registry,
        from_axioms=(
            f"right-Leibniz ({alg.name}, declared on the "
            "initial bracket)",
            "Ψ/Ψ⁻¹ C∞-linearity + invertibility",
        ),
        notes="PDF 13j: the primed operator ρ' = ρ∘Ψ",
        label="right-Leibniz defect normalizes to 0",
    )


def prove_general_twist_jacobi(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Leibniz-Jacobi transports: the twisted Jacobiator is
    ``Ψ⁻¹`` of the initial one at ``(Ψu, Ψv, Ψw)``, cited as a
    declared-zero instance,

    ``[u,[v,w]']' = [[u,v]',w]' + [v,[u,w]']'``."""
    engine = general_twist_engine(
        alg, registry, structure_rules=False
    )
    br = lambda a, b: twisted_bracket(alg, a, b)
    target = Sum(
        br(u, br(v, w)),
        Neg(br(br(u, v), w)),
        Neg(br(v, br(u, w))),
    )
    B = alg.bracket
    pu, pv, pw = PsiSec(u), PsiSec(v), PsiSec(w)
    instance = PsiInvSec(
        Sum(
            B(pu, B(pv, pw)),
            Neg(B(B(pu, pv), pw)),
            Neg(B(pv, B(pu, pw))),
        )
    )
    cite = ProofStep(
        instance,
        Integer(0),
        rule=(
            "cite the initial bracket's Leibniz-Jacobi at "
            "(Ψu, Ψv, Ψw), carried through Ψ⁻¹"
        ),
        justification=(
            "declared axiom instance; Ψ⁻¹(0) = 0"
        ),
        provenance_tag="axiom",
    )
    return _zero_theorem(
        f"general_twist_jacobi_{alg.name}",
        "[u,[v,w]']' = [[u,v]',w]' + [v,[u,w]']' — "
        "Leibniz-Jacobi transports through any invertible Ψ "
        "(13j)",
        Sum(target, Neg(instance)),
        engine,
        registry,
        from_axioms=(
            f"Leibniz-Jacobi ({alg.name}, declared instance at "
            "(Ψu,Ψv,Ψw))",
            "Ψ/Ψ⁻¹ invertibility",
        ),
        notes="PDF 13j general procedure",
        label=(
            "twisted Jacobiator minus the cited instance "
            "normalizes to 0"
        ),
        extra_steps=(cite,),
    )


def prove_general_twist_symmetric_part(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The symmetric part transports as the PRIMED Bourbaki data:

    ``[u,v]' + [v,u]' = 𝔻'g'(u,v)``,
    ``𝔻' = Ψ⁻¹∘𝔻``, ``g'(u,v) = g(Ψu, Ψv)``

    — cited from the initial bracket's declared (6.11) instance at
    ``(Ψu, Ψv)``."""
    engine = general_twist_engine(alg, registry)
    B = alg.bracket
    pu, pv = PsiSec(u), PsiSec(v)
    instance = PsiInvSec(
        Sum(
            B(pu, pv),
            B(pv, pu),
            Neg(BDop(BMetric(pu, pv))),
        )
    )
    cite = ProofStep(
        instance,
        Integer(0),
        rule=(
            "cite the initial (6.11) symmetric part at "
            "(Ψu, Ψv), carried through Ψ⁻¹"
        ),
        justification="declared axiom instance; Ψ⁻¹(0) = 0",
        provenance_tag="axiom",
    )
    target = Sum(
        twisted_bracket(alg, u, v),
        twisted_bracket(alg, v, u),
        Neg(twisted_d(twisted_metric(u, v))),
    )
    return _zero_theorem(
        f"general_twist_symmetric_part_{alg.name}",
        "[u,v]' + [v,u]' = 𝔻'g'(u,v) with 𝔻' = Ψ⁻¹∘𝔻 and "
        "g' = g(Ψ·,Ψ·) — the Bourbaki data transports with the "
        "primed operators separated out (13j)",
        Sum(target, Neg(instance)),
        engine,
        registry,
        from_axioms=(
            "(6.11) symmetric part (declared instance at "
            "(Ψu, Ψv))",
            "Ψ/Ψ⁻¹ invertibility + linearity",
        ),
        notes="PDF 13j: the primed operators 𝔻', g'",
        label=(
            "twisted symmetric part minus the cited instance "
            "normalizes to 0"
        ),
        extra_steps=(cite,),
    )


def prove_general_twist_invariance(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Metric invariance transports with the primed data:

    ``ℒ^R_{ρ'(u)} g'(v,w) = g'([u,v]', w) + g'(v, [u,w]')``

    — cited from the initial (7.8) instance at ``(Ψu, Ψv, Ψw)``
    (the ℒ^R operator itself is unchanged: it reads only the
    anchor direction, and ``ρ'(u) = ρ(Ψu)``)."""
    from jacopy.packages.generalized.bourbaki_structure import (
        BLieRE,
    )

    engine = general_twist_engine(alg, registry)
    pu, pv, pw = PsiSec(u), PsiSec(v), PsiSec(w)
    B = alg.bracket
    instance = Sum(
        BLieRE(pu, BMetric(pv, pw)),
        Neg(BMetric(B(pu, pv), pw)),
        Neg(BMetric(pv, B(pu, pw))),
    )
    cite = ProofStep(
        instance,
        Integer(0),
        rule=(
            "cite the initial (7.8) invariance at (Ψu, Ψv, Ψw)"
        ),
        justification="declared axiom instance",
        provenance_tag="axiom",
    )
    target = Sum(
        BLieRE(pu, twisted_metric(v, w)),
        Neg(
            BMetric(
                PsiSec(twisted_bracket(alg, u, v)), pw
            )
        ),
        Neg(
            BMetric(
                pv, PsiSec(twisted_bracket(alg, u, w))
            )
        ),
    )
    return _zero_theorem(
        f"general_twist_invariance_{alg.name}",
        "ℒ^R_{ρ'(u)} g'(v,w) = g'([u,v]', w) + g'(v, [u,w]') — "
        "metric invariance transports through Ψ with the primed "
        "metric and anchor (13j)",
        Sum(target, Neg(instance)),
        engine,
        registry,
        from_axioms=(
            "(7.8) metric invariance (declared instance at "
            "(Ψu, Ψv, Ψw))",
            "Ψ/Ψ⁻¹ invertibility",
        ),
        notes="PDF 13j: g' = g(Ψ·,Ψ·), ρ' = ρ∘Ψ",
        label=(
            "twisted invariance minus the cited instance "
            "normalizes to 0"
        ),
        extra_steps=(cite,),
    )
