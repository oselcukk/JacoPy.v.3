"""
Linear E-connections and E-torsion/E-curvature (Phase 4.E.1)
[MC Def 3.2-3.3, Prop 3.1-3.3].

A **linear E-connection** on an anchored bundle ``(E, ρ)`` is a map
``∇: 𝔛(E) × 𝔛(E) → 𝔛(E)`` that is ``C^∞``-linear in the direction and
satisfies the ANCHORED Leibniz rule

    ∇_u (f·v) = ρ(u)(f)·v + f·∇_u v.

The context reuses :class:`~jacopy.central.objects.connection.Connection`
on the algebroid's bundle; the only genuinely new definitional rule is
the anchored scalar action ``∇_u f = ρ(u)(f)``
(:class:`EConnectionScalarActionDefinition`) — direction linearity and
the argument Leibniz come from the Phase 4.A rules + ``product_rule``
unchanged.

**E-torsion / E-curvature** are the 4.B nodes with the ALGEBROID
bracket in their expansions:

    T⁽⁰⁾(u,v)  = ∇_u v − ∇_v u − [u,v]_E,
    R⁽⁰⁾(u,v)w = ∇_u ∇_v w − ∇_v ∇_u w − ∇_{[u,v]_E} w.

The superscript ⁰ is deliberate [MC §3]: on a general local algebroid
these are only PSEUDO-tensors —

* ``T⁽⁰⁾(u, f·v) = f·T⁽⁰⁾(u,v)`` holds (right-Leibniz), but
* ``T⁽⁰⁾(f·u, v) = f·T⁽⁰⁾(u,v) − L(Df, u, v)`` — the locality term
  survives as the tensoriality DEFECT
  (:func:`prove_pseudo_torsion_first_slot_defect`), which is exactly
  why the MODIFIED torsion ``ᴸT`` exists (Phase 4.E.2);
* ``R⁽⁰⁾(u,v)(f·w) = f·R⁽⁰⁾(u,v)w`` holds IFF the anchor is a bracket
  morphism — MC Prop 3.3, mechanized as the conditional theorem
  :func:`prove_e_curvature_last_slot_conditional` (honest failure
  without ``anchor-morphism``).
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.objects.connection import Connection, CovariantOp
from jacopy.central.algebroid.context import Algebroid, locality_term
from jacopy.packages.metric_affine.torsion_curvature import (
    Curvature,
    Torsion,
    curvature,
    torsion,
)


def e_connection(alg: Algebroid, name: str = "∇") -> Connection:
    """A linear E-connection context on the algebroid's bundle
    [MC Def 3.2]. On the tangent algebroid this IS an ordinary affine
    connection (the reduction is literal)."""
    if not isinstance(alg, Algebroid):
        raise TypeError("e_connection expects an Algebroid")
    return Connection(name, bundle=alg.bundle)


class EConnectionScalarActionDefinition(Definition):
    """``∇_u f → ρ(u)(f)`` — the anchored scalar action of an
    E-connection (the algebroid case deferred by the Phase 1 tangent
    rule's docstring, now delivered)."""

    anchor = Act

    def __init__(
        self,
        alg: Algebroid,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(alg, Algebroid):
            raise TypeError(
                "EConnectionScalarActionDefinition expects an Algebroid"
            )
        self._alg = alg
        self._registry = registry
        self.name = (
            f"E-connection scalar action ({alg.name}): ∇_u f = ρ(u)(f)"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, CovariantOp)
            and expr.op.bundle == self._alg.bundle
            and is_scalar_function(expr.arg, self._registry)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Act(self._alg.anchor(expr.op.vector), expr.arg)


class ETorsionExpansionDefinition(Definition):
    """``T⁽⁰⁾(u,v) → ∇_u v − ∇_v u − [u,v]_E`` — the E-torsion
    definition (the algebroid bracket in the correction slot)."""

    anchor = Torsion

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"E-torsion definition ({alg.name}): "
            "T⁰(u,v) = ∇_u v − ∇_v u − [u,v]_E"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Torsion)
            and expr.bundle == self._alg.bundle
        )

    def rewrite(self, expr: Expr) -> Expr:
        u, v = expr.arguments

        def nabla(a: Expr, b: Expr) -> Expr:
            return Act(
                CovariantOp(
                    expr.connection_name, a, bundle=expr.bundle
                ),
                b,
            )

        return Sum(
            nabla(u, v),
            Neg(nabla(v, u)),
            Neg(self._alg.bracket(u, v)),
        )


class ECurvatureExpansionDefinition(Definition):
    """``R⁽⁰⁾(u,v)w → ∇_u∇_v w − ∇_v∇_u w − ∇_{[u,v]_E} w`` — the
    E-curvature definition."""

    anchor = Curvature

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"E-curvature definition ({alg.name}): "
            "R⁰(u,v)w = ∇_u∇_v w − ∇_v∇_u w − ∇_{[u,v]_E} w"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Curvature)
            and expr.bundle == self._alg.bundle
        )

    def rewrite(self, expr: Expr) -> Expr:
        u, v, w = expr.arguments

        def nabla(a: Expr, b: Expr) -> Expr:
            return Act(
                CovariantOp(
                    expr.connection_name, a, bundle=expr.bundle
                ),
                b,
            )

        return Sum(
            nabla(u, nabla(v, w)),
            Neg(nabla(v, nabla(u, w))),
            Neg(nabla(self._alg.bracket(u, v), w)),
        )


def e_connection_engine(
    alg: Algebroid,
    *,
    registry: Optional[PropertyRegistry] = None,
    mode: str = "efficient",
    projectors=(),
) -> ExpansionEngine:
    """The algebroid engine extended with the E-connection layer: the
    4.A connection structure rules plus the anchored scalar action and
    the E-torsion/E-curvature expansions."""
    from jacopy.central.algebroid.engine import algebroid_engine
    from jacopy.packages.metric_affine.rules import (
        ConnectionArgumentLeibnizDefinition,
        ConnectionDirectionLinearityDefinition,
    )

    engine = algebroid_engine(
        alg, registry=registry, mode=mode, projectors=projectors
    )
    engine.register(EConnectionScalarActionDefinition(alg, registry))
    engine.register(ConnectionDirectionLinearityDefinition(registry))
    engine.register(ConnectionArgumentLeibnizDefinition(registry))
    engine.register(ETorsionExpansionDefinition(alg))
    engine.register(ECurvatureExpansionDefinition(alg))
    return engine


# --------------------------------------------------------------------- #
# Theorems [MC Prop 3.1-3.3 territory]                                   #
# --------------------------------------------------------------------- #


def prove_e_connection_anchored_leibniz(
    alg: Algebroid,
    conn: Connection,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``∇_u(f·v) = ρ(u)(f)·v + f·∇_u v`` — the anchored Leibniz rule,
    inherited from ``product_rule`` + the anchored scalar action (no
    new axiom)."""
    return ExpandAndSimplify().prove(
        conn(u, Product(f, v)),
        Sum(
            Product(Act(alg.anchor(u), f), v),
            Product(f, conn(u, v)),
        ),
        registry=registry,
        engine=e_connection_engine(alg, registry=registry),
    )


def prove_e_torsion_second_slot_tensorial(
    alg: Algebroid,
    conn: Connection,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``T⁽⁰⁾(u, f·v) = f·T⁽⁰⁾(u, v)`` — the second slot IS tensorial
    (the anchored Leibniz cancels against the declared right-Leibniz
    of the bracket)."""
    return ExpandAndSimplify().prove(
        torsion(conn, u, Product(f, v)),
        Product(f, torsion(conn, u, v)),
        registry=registry,
        engine=e_connection_engine(alg, registry=registry),
    )


def prove_pseudo_torsion_first_slot_defect(
    alg: Algebroid,
    conn: Connection,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``T⁽⁰⁾(f·u, v) = f·T⁽⁰⁾(u, v) − L(Df, u, v)`` — the PSEUDO
    tensor's first-slot defect on a local algebroid: the bracket's
    left-Leibniz locality term survives. This is exactly why the
    modified torsion ``ᴸT`` exists (Phase 4.E.2)."""
    return ExpandAndSimplify().prove(
        torsion(conn, Product(f, u), v),
        Sum(
            Product(f, torsion(conn, u, v)),
            Neg(locality_term(alg, f, u, v)),
        ),
        registry=registry,
        engine=e_connection_engine(alg, registry=registry),
    )


def prove_e_curvature_last_slot_conditional(
    alg: Algebroid,
    conn: Connection,
    u: Expr,
    v: Expr,
    w: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[MC Prop 3.3] ``R⁽⁰⁾(u,v)(f·w) = f·R⁽⁰⁾(u,v)w`` — holds exactly
    when the anchor is a bracket morphism: the obstruction is
    ``(ρ(u)ρ(v) − ρ(v)ρ(u) − ρ([u,v]))(f)·w``. Closes under a
    declared ``anchor-morphism``; fails honestly otherwise."""
    return ExpandAndSimplify().prove(
        curvature(conn, u, v, Product(f, w)),
        Product(f, curvature(conn, u, v, w)),
        registry=registry,
        engine=e_connection_engine(alg, registry=registry),
    )
