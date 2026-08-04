"""
The derived algebroid and the two exterior derivatives
(Phase 4.F.6) [ADM Defs 3.14-3.16, eqs (3.41)-(3.43)].

DESIGN: ``d(∇)`` / ``d̂(∇)`` are the exterior derivatives keyed to
the (projected) modified bracket. Instead of parallel operator code,
a **derived algebroid context** ``E^∇`` is created on the SAME
bundle, whose bracket and anchor BRIDGE definitionally to the base:

    [u,v]_{E^∇} → [u,v]_E − Λ(u,v)      (or − Λ̂ when projected),
    ρ_{E^∇}(u)  → ρ_E(u),

and the whole Phase 3.F Cartan machinery (Palais ``d``, intrinsic
``L``, ``ι``) applies to ``E^∇`` verbatim — legitimized by the
4.F.1 theorem that the modified pair is an almost-dull algebroid.

Mechanical theorems:

* **Cartan magic on functions for d(∇)** — declaration-free (the
  3.F generic magic, inherited by the derived context).
* **[ADM eq (3.43)]** ``d̂(∇)²f = 0`` evaluated on ``(u,v)`` — needs
  the base ``anchor-morphism`` declaration plus the projector's
  kernel rule (``ρ(𝒫L) = 0``): the projected modified bracket has
  the SAME anchor image as the base bracket, so the function-level
  ``d²`` obstruction is inherited from the base. Honest fail without
  ``anchor-morphism`` (tested). For the UNPROJECTED ``d(∇)`` the
  same statement honestly FAILS (``ρ(Λ) ≠ 0`` in general) — also
  tested; this asymmetry is exactly why ADM introduces ``d̂``.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.calculus import (
    IntrinsicDDefinition,
    IntrinsicLDefinition,
)
from jacopy.central.objects.connection import Connection
from jacopy.central.objects.frame import Frame
from jacopy.central.algebroid.context import (
    Algebroid,
    AlgebroidBracket,
    AnchoredVF,
    algebroid,
)
from jacopy.central.algebroid.calculus import algebroid_calculus
from jacopy.central.algebroid.locality import LocalityProjector
from jacopy.packages.metric_affine.e_bianchi import (
    _lam,
    _lam_hat,
)
from jacopy.packages.metric_affine.modified import _engine


class DerivedBracketBridgeDefinition(Definition):
    """``[u,v]_{E^∇} → [u,v]_E − Λ(u,v)`` (or ``− Λ̂`` when
    projected) — the derived context's bracket is DEFINED by the
    (projected) modified bracket of the base."""

    anchor = AlgebroidBracket

    def __init__(
        self,
        base: Algebroid,
        derived: Algebroid,
        conn: Connection,
        fr: Frame,
        projector: Optional[LocalityProjector] = None,
    ) -> None:
        self._base = base
        self._derived = derived
        self._conn = conn
        self._fr = fr
        self._projector = projector
        hat = "̂" if projector is not None else ""
        self.name = (
            f"derived bracket ({derived.name}): "
            f"[u,v] = [u,v]_{base.name} − Λ{hat}(u,v)"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, AlgebroidBracket)
            and expr.algebroid_name == self._derived.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        u, v = expr.u, expr.v
        if self._projector is not None:
            correction = _lam_hat(
                self._base,
                self._conn,
                self._fr,
                self._projector,
                u,
                v,
            )
        else:
            correction = _lam(self._base, self._conn, self._fr, u, v)
        return Sum(self._base.bracket(u, v), Neg(correction))


class DerivedAnchorBridgeDefinition(Definition):
    """``ρ_{E^∇}(u) → ρ_E(u)`` — same bundle, same anchor."""

    anchor = AnchoredVF

    def __init__(self, base: Algebroid, derived: Algebroid) -> None:
        self._base = base
        self._derived = derived
        self.name = (
            f"derived anchor ({derived.name}): ρ = ρ_{base.name}"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, AnchoredVF)
            and expr.algebroid_name == self._derived.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        return self._base.anchor(expr.section)


def derived_algebroid(
    base: Algebroid,
    *,
    projected: bool = False,
    name: Optional[str] = None,
) -> Algebroid:
    """The derived context ``E^∇`` (or ``E^∇̂``) on the base's
    bundle. Carries NO hierarchy declarations of its own — every
    property must be bridged or proven."""
    suffix = "∇̂" if projected else "∇"
    return algebroid(
        name if name is not None else f"{base.name}^{suffix}",
        base.bundle,
    )


def derived_engine(
    base: Algebroid,
    derived: Algebroid,
    conn: Connection,
    fr: Frame,
    *,
    registry: Optional[PropertyRegistry] = None,
    projector: Optional[LocalityProjector] = None,
):
    """The base metric-affine engine + the derived context's Cartan
    rules (``IntrinsicD/L`` of ``calc_{E^∇}``) + the two bridges."""
    projectors = (projector,) if projector is not None else ()
    engine = _engine(base, fr, registry, projectors=projectors)
    calc = algebroid_calculus(derived)
    engine.register(IntrinsicDDefinition(calc, registry))
    engine.register(IntrinsicLDefinition(calc, registry))
    engine.register(
        DerivedBracketBridgeDefinition(
            base, derived, conn, fr, projector
        )
    )
    engine.register(DerivedAnchorBridgeDefinition(base, derived))
    return engine


def d_derived(derived: Algebroid, omega: Expr) -> Act:
    """``d(∇) ω`` (or ``d̂(∇) ω``) — inert node, Palais-expanded by
    the derived engine."""
    return Act(algebroid_calculus(derived).d, omega)


def lie_derived(derived: Algebroid, u: Expr):
    """``ℒ^∇_u`` — the derived Lie derivative operator."""
    return algebroid_calculus(derived).lie(u)


def prove_derived_magic_on_functions(
    base: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    projector: Optional[LocalityProjector] = None,
) -> ProofChain:
    """Cartan magic on functions for the derived calculus,
    ``ℒ^∇_u f = (d(∇) ι_u + ι_u d(∇)) f`` — declaration-free (the
    3.F generic magic, inherited through the bridges)."""
    from jacopy.central.objects.interior import Interior

    derived = derived_algebroid(
        base, projected=projector is not None
    )
    iota = Interior(u)
    lhs = Act(lie_derived(derived, u), f)
    rhs = Sum(
        Act(
            algebroid_calculus(derived).d,
            Act(iota, f),
        ),
        Act(iota, d_derived(derived, f)),
    )
    return ExpandAndSimplify().prove(
        lhs,
        rhs,
        registry=registry,
        engine=derived_engine(
            base,
            derived,
            conn,
            fr,
            registry=registry,
            projector=projector,
        ),
    )


def prove_projected_d_squared_on_functions(
    base: Algebroid,
    conn: Connection,
    fr: Frame,
    projector: LocalityProjector,
    f: Expr,
    u: Expr,
    v: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[ADM eq (3.43)]: ``(d̂(∇)² f)(u, v) = 0`` — the projected
    modified bracket's anchor image equals the base bracket's
    (kernel rule ``ρ(𝒫L) = 0``), so the base ``anchor-morphism``
    declaration closes the function-level ``d²``. The UNPROJECTED
    ``d(∇)`` version honestly fails (``ρ(Λ) ≠ 0`` in general)."""
    derived = derived_algebroid(base, projected=True)
    node = MultiEval(
        d_derived(derived, d_derived(derived, f)),
        u,
        v,
        alternating=True,
        slot_kind="vector",
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=derived_engine(
            base,
            derived,
            conn,
            fr,
            registry=registry,
            projector=projector,
        ),
    )


def prove_derived_magic_on_forms(
    base: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    omega: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    projector: Optional[LocalityProjector] = None,
) -> ProofChain:
    """Cartan magic for the derived calculus on a p-FORM, evaluated
    against the given section slots (``len(slots) == p``):

        (ℒ^∇_u ω)(v₁,…,v_p) = ((d(∇) ι_u + ι_u d(∇)) ω)(v₁,…,v_p)

    — declaration-free at every degree (the 3.F generic magic; the
    p ≥ 2 cases exercise the arity-general Palais/interior rules
    through the bridges)."""
    from jacopy.core.pairing import Pairing
    from jacopy.central.objects.interior import Interior

    derived = derived_algebroid(
        base, projected=projector is not None
    )
    calc = algebroid_calculus(derived)
    iota = Interior(u)

    def ev(expr: Expr) -> Expr:
        if len(slots) == 1:
            return Pairing(expr, slots[0])
        return MultiEval(
            expr, *slots, alternating=True, slot_kind="vector"
        )

    lhs = ev(Act(lie_derived(derived, u), omega))
    rhs = Sum(
        ev(Act(calc.d, Act(iota, omega))),
        ev(Act(iota, d_derived(derived, omega))),
    )
    return ExpandAndSimplify().prove(
        lhs,
        rhs,
        registry=registry,
        engine=derived_engine(
            base,
            derived,
            conn,
            fr,
            registry=registry,
            projector=projector,
        ),
    )


def prove_projected_d_squared_associator(
    base: Algebroid,
    conn: Connection,
    fr: Frame,
    projector: LocalityProjector,
    alpha: Expr,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[ADM eqs (3.44)-(3.45)]: on E-1-forms the projected ``d̂²``
    does NOT vanish — it measures the ASSOCIATOR (Jacobiator
    combination, in the calculus orientation) of the projected
    modified bracket:

        (d̂(∇)² α)(u,v,w)
            = ⟨α, b̂(b̂(u,v),w) − b̂(b̂(u,w),v) + b̂(b̂(v,w),u)⟩

    (under the base ``anchor-morphism`` declaration, which kills the
    predator part). Identity form — ARBITRARY connection; the
    combination is exactly the Phase 3.F ``d²α`` obstruction shape,
    now for ``b̂``."""
    from jacopy.core.pairing import Pairing
    from jacopy.packages.metric_affine.e_bianchi import (
        projected_modified_bracket,
    )

    derived = derived_algebroid(base, projected=True)

    def bhat(p, q):
        return projected_modified_bracket(
            base, conn, fr, projector, p, q
        )

    node = MultiEval(
        d_derived(derived, d_derived(derived, alpha)),
        u,
        v,
        w,
        alternating=True,
        slot_kind="vector",
    )
    assoc = Sum(
        bhat(bhat(u, v), w),
        Neg(bhat(bhat(u, w), v)),
        bhat(bhat(v, w), u),
    )
    return ExpandAndSimplify().prove(
        node,
        Pairing(alpha, assoc),
        registry=registry,
        engine=derived_engine(
            base,
            derived,
            conn,
            fr,
            registry=registry,
            projector=projector,
        ),
    )
