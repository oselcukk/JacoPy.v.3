"""
Definitional engine rules for the Phase 1 deferrals (Phase 2.B).

Two expansion rules that Phase 1 built the *nodes* for but deliberately
left inert:

* :class:`CovariantScalarActionDefinition` — ``∇_X f = X(f)`` on the
  tangent bundle (PDF item 8m/9): the covariant derivative of a scalar
  function is the directional derivative. Combined with the graded
  Leibniz pass this closes the module property
  ``∇_X(f·T) = X(f)·T + f·∇_X T`` as a theorem.
* :class:`FrameDualityDefinition` — ``⟨e^a, e_b⟩ = δ^a_b`` (PDF item
  8p): the duality pairing of a coframe field against a frame field of
  the *same* frame evaluates to the Kronecker delta (``1`` on equal
  labels, symbolic ``δ^a_b`` otherwise).

Both are **definitional** rules in the sense of the Phase 2 definition
policy: they unfold canonical definitions and are always on in the
tangent engine.

Scalar recognition
------------------

``∇_X f = X(f)`` must fire on *functions only* — ``∇_X Y`` (a vector)
and ``∇_X ω`` (a form) share degree bookkeeping but are genuine
covariant derivatives. Degree 0 alone cannot distinguish a function
from a vector field result, so :func:`is_scalar_function` implements a
conservative structural check: registered scalar symbols, numeric
literals, pairings/multi-evaluations (always scalars), degree-0
derivations applied to scalar functions, and sums/products/negations
thereof. Anything unrecognized is left inert — soundness over
completeness.

On a general algebroid the rule becomes ``∇_u f = ρ_E(u)(f)``; the
anchor wiring arrives in Phase 3, so this rule only fires for
connections living on a tangent bundle.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum, Symbol
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition
from jacopy.central.calculus.scalars import is_scalar_function  # noqa: F401  (re-export)
from jacopy.central.objects.connection import CovariantOp
from jacopy.central.objects.frame import (
    CoframeField,
    FrameField,
    kronecker_delta,
)


class CovariantScalarActionDefinition(Definition):
    """``∇_X f → X(f)`` for a scalar function ``f`` (tangent bundle).

    The canonical definition of the connection's action on functions:
    on ``TM`` the anchor is the identity, so the covariant derivative
    of a function is the directional derivative. Fires only when the
    operand is certainly a scalar function (see
    :func:`is_scalar_function`) and the connection lives on a tangent
    bundle (the algebroid case ``∇_u f = ρ_E(u)(f)`` arrives in
    Phase 3).
    """

    name = "connection on functions: ∇_X f = X(f)"
    anchor = Act

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, CovariantOp)
            and expr.op.bundle.is_tangent
            and is_scalar_function(expr.arg, self._registry)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Act(expr.op.vector, expr.arg)


class FrameDualityDefinition(Definition):
    """``⟨e^a, e_b⟩ → δ^a_b`` for a coframe/frame pair of the same
    frame (PDF item 8p).

    Fires on a :class:`Pairing` whose form slot is a
    :class:`CoframeField` and whose vector slot is a
    :class:`FrameField` sharing the same base name and bundle. Equal
    index labels give ``1``; distinct labels give the symbolic
    ``δ^a_b`` (fixed-index reading — see
    :func:`~jacopy.central.objects.frame.kronecker_delta`).
    """

    name = "frame duality: ⟨e^a, e_b⟩ = δ^a_b"
    anchor = Pairing

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Pairing)
            and isinstance(expr.alpha, CoframeField)
            and isinstance(expr.X, FrameField)
            and expr.alpha.base_name == expr.X.base_name
            and expr.alpha.bundle == expr.X.bundle
        )

    def rewrite(self, expr: Expr) -> Expr:
        return kronecker_delta(expr.alpha.index, expr.X.index)


# --------------------------------------------------------------------- #
# Theorems closed by the 2.B rules                                      #
# --------------------------------------------------------------------- #


def _engine(registry: Optional[PropertyRegistry]):
    # Late import: engine.py aggregates rules from this module.
    from jacopy.central.tangent.engine import tangent_engine

    return tangent_engine(registry=registry)


def prove_covariant_scalar_action(conn, X: Expr, f: Expr, *, registry=None):
    """Prove ``∇_X f = X(f)`` (definitional; closes in one rewrite)."""
    from jacopy.proof.strategies import ExpandAndSimplify

    return ExpandAndSimplify().prove(
        conn(X, f), Act(X, f), registry=registry, engine=_engine(registry)
    )


def prove_covariant_leibniz(conn, X: Expr, f: Expr, T: Expr, *, registry=None):
    """Prove the module property ``∇_X(f·T) = X(f)·T + f·∇_X T``.

    A theorem: the graded Leibniz pass (``∇_X`` is a degree-0
    derivation) splits the product, and the definitional rule turns
    ``∇_X f`` into ``X(f)``. Works for any tensor-like ``T`` — the
    inert ``∇_X T`` survives untouched on both sides.
    """
    from jacopy.proof.strategies import ExpandAndSimplify

    lhs = conn(X, Product(f, T))
    rhs = Sum(Product(Act(X, f), T), Product(f, conn(X, T)))
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_frame_duality(fr, upper, lower, *, registry=None):
    """Prove ``⟨e^a, e_b⟩ = δ^a_b`` for a frame (definitional)."""
    from jacopy.proof.strategies import ExpandAndSimplify

    return ExpandAndSimplify().prove(
        fr.pairing(upper, lower),
        kronecker_delta(upper, lower),
        registry=registry,
        engine=_engine(registry),
    )
