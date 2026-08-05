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


class InteriorVectorLinearityDefinition(Definition):
    """``ι_{f·X + Y} → f·ι_X + ι_Y`` at the ``Act`` level — the
    interior product is tensorial in its vector slot (Phase 6.D: the
    Poisson double's ``ι_{fV}(dω)`` shapes sit unevaluated inside
    π-slots and need the operator-level split)."""

    anchor = Act

    def __init__(self, registry=None) -> None:
        self._registry = registry
        self.name = (
            "interior vector linearity: ι_{fX+Y} = f·ι_X + ι_Y"
        )

    def _split(self, vec):
        from jacopy.core.expr import Integer, Neg, Product, Sum
        from jacopy.central.calculus.scalars import (
            is_scalar_function,
        )

        if isinstance(vec, (Sum, Neg)) or vec == Integer(0):
            return True
        if isinstance(vec, Product) and len(vec.children) >= 2:
            return any(
                is_scalar_function(c, self._registry)
                for c in vec.children
            )
        return False

    def matches(self, expr) -> bool:
        from jacopy.central.objects.interior import Interior

        return (
            isinstance(expr, Act)
            and isinstance(expr.op, Interior)
            and self._split(expr.op.vector)
        )

    def rewrite(self, expr):
        from jacopy.core.expr import Integer, Neg, Product, Sum
        from jacopy.central.calculus.scalars import (
            is_scalar_function,
        )
        from jacopy.central.objects.interior import Interior

        vec = expr.op.vector
        arg = expr.arg
        if vec == Integer(0):
            return Integer(0)
        if isinstance(vec, Sum):
            return Sum(
                *(
                    Act(Interior(c), arg)
                    for c in vec.children
                )
            )
        if isinstance(vec, Neg):
            return Neg(Act(Interior(vec.arg), arg))
        scalars = [
            c
            for c in vec.children
            if is_scalar_function(c, self._registry)
        ]
        rest = [
            c
            for c in vec.children
            if not is_scalar_function(c, self._registry)
        ]
        core = rest[0] if len(rest) == 1 else Product(*rest)
        return Product(*scalars, Act(Interior(core), arg))
