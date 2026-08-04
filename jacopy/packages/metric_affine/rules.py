"""
Connection structure rules (Phase 4.A).

An affine connection is, BY DEFINITION, ``C^∞``-linear in its
direction slot and a derivation in its argument:

* ``∇_{fX + Y} T = f·∇_X T + ∇_Y T`` — the direction slot lives inside
  the :class:`~jacopy.central.objects.connection.CovariantOp` atom, so
  it needs its own rule (:class:`ConnectionDirectionLinearityDefinition`).
* ``∇_X (f·T) = X(f)·T + f·∇_X T`` — at TREE positions the argument
  side needs no new rule: ``∇_X`` is a degree-0 :class:`Derivation`,
  so the generic ``product_rule`` pass supplies the Leibniz split and
  the Phase 1/2 rule ``∇_X f = X(f)`` finishes (verified in the 4.A
  tests). INSIDE atom slots, however, ``product_rule`` cannot reach
  (it walks children, and slot contents are not children) — the
  engine's slot protocol can, so the same Leibniz is ALSO provided as
  the engine rule :class:`ConnectionArgumentLeibnizDefinition`
  (Phase 4.C surfaced this: ``g(Z, ∇_X(f·Y))`` stalls without it).
  Both express the one canonical definition; they are confluent.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.objects.connection import CovariantOp


class ConnectionDirectionLinearityDefinition(Definition):
    """``∇_{fX + Y} T → f·∇_X T + ∇_Y T`` (definitional: a connection
    is ``C^∞``-linear in the direction). Also ``∇_{−X} T → −∇_X T``
    and ``∇_0 T → 0``."""

    name = "connection direction linearity: ∇_{fX+Y}T = f·∇_X T + ∇_Y T"
    anchor = Act

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            if is_scalar_function(head, self._registry):
                rest = expr.children[1:]
                return head, (rest[0] if len(rest) == 1 else Product(*rest))
        return None

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, CovariantOp)):
            return False
        X = expr.op.vector
        if isinstance(X, (Sum, Neg)) or X == Integer(0):
            return True
        return self._scalar_split(X) is not None

    def rewrite(self, expr: Expr) -> Expr:
        op = expr.op
        T = expr.arg

        def nabla(direction: Expr) -> Expr:
            return Act(
                CovariantOp(
                    op.connection_name, direction, bundle=op.bundle
                ),
                T,
            )

        X = op.vector
        if X == Integer(0):
            return Integer(0)
        if isinstance(X, Sum):
            return Sum(*(nabla(c) for c in X.children))
        if isinstance(X, Neg):
            return Neg(nabla(X.arg))
        scalar, rest = self._scalar_split(X)
        return Product(scalar, nabla(rest))


class ConnectionArgumentLeibnizDefinition(Definition):
    """``∇_X (f·T) → X(f)·T + f·∇_X T`` plus additivity/negation/
    constants in the argument — the canonical connection Leibniz as an
    ENGINE rule, so it also fires inside operator-atom slots where the
    ``product_rule`` pass cannot reach (e.g. ``g(Z, ∇_X(f·Y))``).
    Confluent with the tree-position ``product_rule`` route."""

    name = "connection argument Leibniz: ∇_X(fT) = X(f)·T + f·∇_X T"
    anchor = Act

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            if is_scalar_function(head, self._registry):
                rest = expr.children[1:]
                return head, (rest[0] if len(rest) == 1 else Product(*rest))
        return None

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, CovariantOp)):
            return False
        T = expr.arg
        if isinstance(T, (Sum, Neg, Integer, Rational)):
            return True
        return self._scalar_split(T) is not None

    def rewrite(self, expr: Expr) -> Expr:
        op = expr.op
        T = expr.arg
        if isinstance(T, (Integer, Rational)):
            return Integer(0)
        if isinstance(T, Sum):
            return Sum(*(Act(op, c) for c in T.children))
        if isinstance(T, Neg):
            return Neg(Act(op, T.arg))
        f, rest = self._scalar_split(T)
        return Sum(
            Product(Act(op, f), rest),
            Product(f, Act(op, rest)),
        )
