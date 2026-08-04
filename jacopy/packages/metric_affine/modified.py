"""
Modified E-torsion and projected E-curvature (Phase 4.E.2)
[MC Prop 3.2, Def 3.7, Prop 3.3-3.4, Cor 3.3, Def 3.11].

The pseudo tensors of 4.E.1 carry locality defects; the MC corrections
repair them:

    ᴸT(u, v)   := T⁽⁰⁾(u, v) + Σ_s L(e^s, ∇_{e_s} u, v),
    ᴸ̂R(u, v, w) := R⁽⁰⁾(u, v)w + Σ_s ∇_{𝒫(L(e^s, ∇_{e_s} u, v))} w,

with ``(e_s)`` a frame of ``E`` and ``𝒫`` a locality projector
(Phase 3.E.4 — this module is its main consumer).

The proofs turn on ONE new rule, the **coframe recombination**

    Σ_s ρ(e_s)(f) · E(e^s)  →  E(Df)

(collection direction ONLY — its inverse, a ``Df``-decomposition
axiom, would ping-pong against it; same design decision as the C2
collection of Phase 3.E.2). It is exactly the coframe-completeness
identity ``Df = Σ_s ⟨Df, e_s⟩ e^s = Σ_s ρ(e_s)(f)·e^s`` read
backwards, applied inside any slot-linear expression ``E`` via the
slot-walking substitution protocol.

* ``ᴸT`` is then ``C^∞``-tensorial in BOTH slots on a local algebroid
  (the recombined ``L(Df,u,v)`` cancels the 4.E.1 defect);
* ``ᴸ̂R``'s first slot closes the same way — the recombined
  ``∇_{𝒫(L(Df,u,v))}w`` meets the projector's ABSORPTION rule
  (``𝒫`` acts as identity on coboundary values) and cancels;
* ``ᴸ̂R``'s last slot uses the projector's KERNEL rule
  (``ρ(𝒫(L(…))) = 0``) plus the anchor morphism — MC Prop 3.3-3.4
  composed.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Neg, Product, Sum
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.calculus.indexed_rules import contains_index
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.objects.connection import Connection, CovariantOp
from jacopy.central.objects.frame import (
    CoframeField,
    Frame,
    FrameField,
    FrameIndex,
)
from jacopy.central.algebroid.context import (
    Algebroid,
    AnchoredVF,
    LocalityOperator,
)
from jacopy.central.algebroid.locality import LocalityProjector
from jacopy.packages.metric_affine.torsion_curvature import (
    curvature,
    torsion,
)


class CoframeRecombinationDefinition(Definition):
    """``Σ_s ρ(e_s)(f)·E(e^s) → E(Df)`` — coframe completeness in the
    collection direction (the sole direction; the decomposition
    inverse would ping-pong)."""

    anchor = IndexedSum

    def __init__(
        self,
        alg: Algebroid,
        fr: Frame,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(alg, Algebroid):
            raise TypeError(
                "CoframeRecombinationDefinition expects an Algebroid"
            )
        if not isinstance(fr, Frame):
            raise TypeError(
                "CoframeRecombinationDefinition expects a Frame"
            )
        self._alg = alg
        self._frame = fr
        self._registry = registry
        self.name = (
            f"coframe recombination ({alg.name}, {fr.name}): "
            "Σ_s ρ(e_s)(f)·E(e^s) = E(Df)"
        )

    def _split(self, expr):
        if not isinstance(expr, IndexedSum):
            return None
        dummy = expr.dummy._repr_inner()
        body = expr.body
        sign = False
        if isinstance(body, Neg):
            sign = True
            body = body.arg
        factors = (
            list(body.children) if isinstance(body, Product) else [body]
        )
        for k, c in enumerate(factors):
            if not (
                isinstance(c, Act)
                and isinstance(c.op, AnchoredVF)
                and c.op.algebroid_name == self._alg.name
                and isinstance(c.op.section, FrameField)
                and c.op.section.base_name == self._frame.name
                and c.op.section.index == dummy
                and is_scalar_function(c.arg, self._registry)
            ):
                continue
            rest = factors[:k] + factors[k + 1 :]
            if not rest:
                continue
            coframe_atom = self._frame.dual().field(dummy)
            if not any(
                contains_index(r, dummy) for r in rest
            ):
                continue
            return sign, c.arg, rest, coframe_atom
        return None

    def matches(self, expr: Expr) -> bool:
        split = self._split(expr)
        if split is None:
            return False
        sign, f, rest, coframe_atom = split
        dummy = expr.dummy._repr_inner()
        substituted = [
            r.substitute_atom(coframe_atom, self._alg.D(f))
            for r in rest
        ]
        # Sound only if the substitution consumed EVERY occurrence of
        # the bound index.
        return not any(
            contains_index(r, dummy) for r in substituted
        )

    def rewrite(self, expr: Expr) -> Expr:
        sign, f, rest, coframe_atom = self._split(expr)
        substituted = [
            r.substitute_atom(coframe_atom, self._alg.D(f))
            for r in rest
        ]
        out: Expr = (
            substituted[0]
            if len(substituted) == 1
            else Product(*substituted)
        )
        return Neg(out) if sign else out


def modified_torsion(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    *,
    bound: str = "s",
) -> Expr:
    """``ᴸT(u,v) := T⁽⁰⁾(u,v) + Σ_s L(e^s, ∇_{e_s}u, v)``
    [MC Def 3.7 with the canonical representative]."""
    if fr.bundle != alg.bundle:
        raise ValueError(
            "the frame must live on the algebroid's bundle"
        )
    dummy = FrameIndex(bound)
    return Sum(
        torsion(conn, u, v),
        IndexedSum(
            dummy,
            fr,
            LocalityOperator(
                alg.name,
                fr.dual().field(bound),
                conn(fr.field(bound), u),
                v,
            ),
        ),
    )


def projected_curvature(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    projector: LocalityProjector,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    bound: str = "s",
) -> Expr:
    """``ᴸ̂R(u,v,w) := R⁽⁰⁾(u,v)w + Σ_s ∇_{𝒫(L(e^s, ∇_{e_s}u, v))} w``
    [MC Def 3.11 / Cor 3.3]."""
    if not isinstance(projector, LocalityProjector):
        raise TypeError(
            "projected_curvature expects a LocalityProjector"
        )
    if fr.bundle != alg.bundle:
        raise ValueError(
            "the frame must live on the algebroid's bundle"
        )
    dummy = FrameIndex(bound)
    correction_body = Act(
        CovariantOp(
            conn.name,
            projector(
                LocalityOperator(
                    alg.name,
                    fr.dual().field(bound),
                    conn(fr.field(bound), u),
                    v,
                )
            ),
            bundle=conn.bundle,
        ),
        w,
    )
    return Sum(
        curvature(conn, u, v, w),
        IndexedSum(dummy, fr, correction_body),
    )


def _engine(
    alg: Algebroid,
    fr: Frame,
    registry,
    *,
    projectors=(),
):
    from jacopy.central.calculus.indexed_rules import (
        IndexedSumEvalPushInDefinition,
        IndexedSumLinearityDefinition,
        KroneckerContractionDefinition,
    )
    from jacopy.packages.metric_affine.decomposition import (
        IndexedSumOperatorPushDefinition,
    )
    from jacopy.packages.metric_affine.e_connection import (
        e_connection_engine,
    )

    engine = e_connection_engine(
        alg, registry=registry, projectors=projectors
    )
    engine.register(CoframeRecombinationDefinition(alg, fr, registry))
    engine.register(IndexedSumLinearityDefinition())
    engine.register(IndexedSumEvalPushInDefinition())
    engine.register(KroneckerContractionDefinition())
    engine.register(IndexedSumOperatorPushDefinition())
    return engine


def prove_modified_torsion_tensorial(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> tuple:
    """``ᴸT(f·u, v) = f·ᴸT(u, v)`` AND ``ᴸT(u, f·v) = f·ᴸT(u, v)`` —
    the modified torsion is a genuine tensor on a local algebroid
    [MC Prop 3.2/Def 3.7]: the recombined ``L(Df,u,v)`` cancels the
    pseudo defect."""
    first = ExpandAndSimplify().prove(
        modified_torsion(alg, conn, fr, Product(f, u), v),
        Product(f, modified_torsion(alg, conn, fr, u, v)),
        registry=registry,
        engine=_engine(alg, fr, registry),
    )
    second = ExpandAndSimplify().prove(
        modified_torsion(alg, conn, fr, u, Product(f, v)),
        Product(f, modified_torsion(alg, conn, fr, u, v)),
        registry=registry,
        engine=_engine(alg, fr, registry),
    )
    return first, second


def prove_projected_curvature_first_slot(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    projector: LocalityProjector,
    u: Expr,
    v: Expr,
    w: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``ᴸ̂R(f·u, v, w) = f·ᴸ̂R(u, v, w)`` [MC Cor 3.3, first slot]:
    the recombined ``∇_{𝒫(L(Df,u,v))}w`` meets the projector's
    absorption rule (``L̂ ∈ [L̃]``) and cancels the pseudo defect."""
    return ExpandAndSimplify().prove(
        projected_curvature(
            alg, conn, fr, projector, Product(f, u), v, w
        ),
        Product(
            f, projected_curvature(alg, conn, fr, projector, u, v, w)
        ),
        registry=registry,
        engine=_engine(
            alg, fr, registry, projectors=(projector,)
        ),
    )


def prove_projected_curvature_last_slot(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    projector: LocalityProjector,
    u: Expr,
    v: Expr,
    w: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``ᴸ̂R(u, v, f·w) = f·ᴸ̂R(u, v, w)`` [MC Prop 3.3-3.4 composed]:
    the ``R⁽⁰⁾`` part needs the anchor morphism, the correction's
    Leibniz term dies by the projector's KERNEL rule
    (``ρ(𝒫(L(…))) = 0`` — ``im(L̂) ⊂ ker ρ``)."""
    return ExpandAndSimplify().prove(
        projected_curvature(
            alg, conn, fr, projector, u, v, Product(f, w)
        ),
        Product(
            f, projected_curvature(alg, conn, fr, projector, u, v, w)
        ),
        registry=registry,
        engine=_engine(
            alg, fr, registry, projectors=(projector,)
        ),
    )


def generalized_ricci_anomaly(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    projector: LocalityProjector,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    bound: str = "s",
) -> Expr:
    """``Σ_s [∇_{L(e^s,∇_{e_s}u,v)}w − ∇_{𝒫L(e^s,∇_{e_s}u,v)}w]`` —
    the ``(1−𝒫)``-anomaly of the generalized Ricci identity
    [ADM eq (3.36)]."""
    dummy = FrameIndex(bound)
    L_node = LocalityOperator(
        alg.name,
        fr.dual().field(bound),
        conn(fr.field(bound), u),
        v,
    )
    return IndexedSum(
        dummy,
        fr,
        Sum(
            Act(
                CovariantOp(conn.name, L_node, bundle=conn.bundle), w
            ),
            Neg(
                Act(
                    CovariantOp(
                        conn.name,
                        projector(L_node),
                        bundle=conn.bundle,
                    ),
                    w,
                )
            ),
        ),
    )


def prove_generalized_ricci_identity(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    projector: LocalityProjector,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
):
    """[ADM Prop 3.8]: the generalized Ricci identity on a local
    algebroid, for an ARBITRARY E-connection (purely definitional —
    no declarations):

        ∇²_{u,v}w − ∇²_{v,u}w
          = ᴸ̂R(u,v,w) − ∇_{ᴸT(u,v)}w
            + Σ_s ∇_{(1−𝒫)L(e^s,∇_{e_s}u,v)}w,

    with ``∇²_{u,v}w := ∇_u∇_v w − ∇_{∇_u v}w`` the second covariant
    derivative. The ``(1−𝒫)`` anomaly is exactly the gap between the
    modified torsion's ``L``-correction and the projected
    curvature's ``𝒫L``-correction."""
    from jacopy.proof.strategies import ExpandAndSimplify

    def cov(direction: Expr, arg: Expr) -> Expr:
        return Act(
            CovariantOp(conn.name, direction, bundle=conn.bundle),
            arg,
        )

    lhs = Sum(
        conn(u, conn(v, w)),
        Neg(conn(v, conn(u, w))),
        Neg(cov(conn(u, v), w)),
        cov(conn(v, u), w),
    )
    rhs = Sum(
        projected_curvature(alg, conn, fr, projector, u, v, w),
        Neg(cov(modified_torsion(alg, conn, fr, u, v), w)),
        generalized_ricci_anomaly(
            alg, conn, fr, projector, u, v, w
        ),
    )
    return ExpandAndSimplify().prove(
        lhs,
        rhs,
        registry=registry,
        engine=_engine(alg, fr, registry, projectors=(projector,)),
    )
