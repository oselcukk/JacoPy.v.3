"""
Slot-insertion rules for the interior product ``ι_X`` (the Phase 1
deferral; PDF item 8r).

The canonical definition ``(ι_X ω)(Y_1, …, Y_{p−1}) := ω(X, Y_1, …)``
as engine rules — calculus-independent (no anchor or bracket is
involved, so these are NOT scoped to a :class:`BracketCalculus`):

* :class:`InteriorActDefinition` — the degenerate degrees, directly on
  the ``Act`` node: ``ι_X f = 0`` for a 0-form and ``ι_X ω = ⟨ω, X⟩``
  for a 1-form (the contraction of a 1-form *is* the pairing).
* :class:`InteriorEvalDefinition` — the evaluation shapes:
  ``⟨ι_X ω, Y⟩ → ω(X, Y)`` (2-forms) and
  ``(ι_X ω)(Y_1, …, Y_k) → ω(X, Y_1, …, Y_k)`` in general. Concrete
  degrees must match the slot count; a symbolic degree is
  caller-asserted (same arity policy as the intrinsic ``d``).
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act, degree_of
from jacopy.core.expr import Expr, Integer
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition
from jacopy.central.calculus.bracket_calculus import _evaluate


def _interior_cls():
    # Late import: the objects layer must stay importable without the
    # calculus layer.
    from jacopy.central.objects.interior import Interior

    return Interior


class InteriorActDefinition(Definition):
    """``ι_X f → 0`` (0-forms) and ``ι_X ω → ⟨ω, X⟩`` (1-forms)."""

    name = "interior product: ι_X f = 0, ι_X ω = ⟨ω, X⟩ (deg 0 / 1)"
    anchor = Act

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _arg_degree(self, omega: Expr) -> Optional[int]:
        try:
            return degree_of(omega, self._registry).as_int()
        except ValueError:
            return None

    def matches(self, expr: Expr) -> bool:
        Interior = _interior_cls()
        if not (isinstance(expr, Act) and isinstance(expr.op, Interior)):
            return False
        return self._arg_degree(expr.arg) in (0, 1)

    def rewrite(self, expr: Expr) -> Expr:
        p = self._arg_degree(expr.arg)
        if p == 0:
            return Integer(0)
        return Pairing(expr.arg, expr.op.vector)


class InteriorEvalDefinition(Definition):
    """Slot insertion under evaluation: ``(ι_X ω)(Y…) → ω(X, Y…)``."""

    name = "interior product evaluation: (ι_X ω)(Y_1,…,Y_k) = ω(X, Y_1,…,Y_k)"
    anchor = (Pairing, MultiEval)

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _is_interior_act(self, head: Expr) -> bool:
        Interior = _interior_cls()
        return isinstance(head, Act) and isinstance(head.op, Interior)

    def _arg_degree(self, omega: Expr) -> Optional[int]:
        try:
            return degree_of(omega, self._registry).as_int()
        except ValueError:
            return None

    def matches(self, expr: Expr) -> bool:
        if isinstance(expr, Pairing):
            if not self._is_interior_act(expr.alpha):
                return False
            return self._arg_degree(expr.alpha.arg) == 2
        if isinstance(expr, MultiEval):
            if not self._is_interior_act(expr.head):
                return False
            p = self._arg_degree(expr.head.arg)
            return p is None or p == expr.arity + 1
        return False

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr, Pairing):
            omega = expr.alpha.arg
            X = expr.alpha.op.vector
            return _evaluate(omega, (X, expr.X))
        omega = expr.head.arg
        X = expr.head.op.vector
        return _evaluate(omega, (X,) + expr.args)
