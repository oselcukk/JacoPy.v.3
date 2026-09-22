"""
Engine semantics of :class:`~jacopy.central.objects.partial_eval.PartialEval`
(PDF item 8w generalization; ledger K2, 2026-09-22).

* :class:`PartialEvalCollapseDefinition` — filling the open slots is
  the full evaluation: ``T(a, ·)(b) → T(a, b)`` on a
  :class:`~jacopy.core.multi_eval.MultiEval` whose head is a partial
  map, and ``⟨T(a, ·), b⟩ → T(a, b)`` / ``⟨β, π(α, ·)⟩ → π(α, β)``
  through a :class:`~jacopy.core.pairing.Pairing` (one open slot).
* :class:`PartialEvalLinearityDefinition` — multilinearity in the
  FIXED slots: sums split, certainly-scalar factors pull out, a zero
  slot kills the map (definitional — the head is multilinear).
* :class:`AlternatingPartialEvalAsInteriorDefinition` — for an
  alternating form on vector slots the partial map IS the iterated
  interior product, ``ω(X₁, …, X_j, ·, …) = ι_{X_j}⋯ι_{X_1} ω``
  (the convention of :func:`~jacopy.central.objects.interior.contract_all`,
  proved on the 3-form instance in Phase 2.F); fixed slots that are
  not leading are moved to the front with the permutation sign.
* :class:`MusicalAsPartialEvalDefinition` — the bilinear musical atoms
  are the ``j = 1`` case: ``g^♭(X) → g(X, ·)``, ``π^♯(α) → π(α, ·)``.

All four are calculus-independent and ride with the tangent engine.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition


def _pe_cls():
    from jacopy.central.objects.partial_eval import PartialEval

    return PartialEval


def _permutation_sign(perm) -> int:
    sign = 1
    seq = list(perm)
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                sign = -sign
    return sign


class PartialEvalCollapseDefinition(Definition):
    """``T(a, ·)(b) → T(a, b)`` and the pairing shapes."""

    name = "partial evaluation: filling the open slots is the full evaluation"
    anchor = (MultiEval, Pairing)

    def matches(self, expr: Expr) -> bool:
        PE = _pe_cls()
        if isinstance(expr, MultiEval):
            return isinstance(expr.head, PE) and expr.arity == expr.head.n_open
        if isinstance(expr, Pairing):
            return (isinstance(expr.alpha, PE) and expr.alpha.n_open == 1) or (
                isinstance(expr.X, PE) and expr.X.n_open == 1
            )
        return False

    def rewrite(self, expr: Expr) -> Expr:
        PE = _pe_cls()
        if isinstance(expr, MultiEval):
            return expr.head(*expr.args)
        if isinstance(expr.alpha, PE):
            return expr.alpha(expr.X)
        return expr.X(expr.alpha)


class PartialEvalLinearityDefinition(Definition):
    """Multilinearity in the fixed slots of a partial map."""

    name = "partial evaluation: multilinearity in the fixed slots"
    anchor = None  # PartialEval (set in __init__)

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry
        self.anchor = _pe_cls()

    def _scalar_split(self, value: Expr):
        from jacopy.central.calculus.scalars import is_scalar_function

        if isinstance(value, Product) and len(value.children) >= 2:
            scalars = [c for c in value.children if is_scalar_function(c, self._registry)]
            rest = [c for c in value.children if not is_scalar_function(c, self._registry)]
            if scalars and len(rest) == 1:
                return (scalars[0] if len(scalars) == 1 else Product(*scalars)), rest[0]
        return None

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, _pe_cls()) and any(
            isinstance(v, (Sum, Neg)) or v == Integer(0) or self._scalar_split(v) is not None
            for _, v in expr.fixed
        )

    def rewrite(self, expr: Expr) -> Expr:
        PE = _pe_cls()

        def rebuilt(pos: int, value: Expr) -> Expr:
            fixed = dict(expr.fixed)
            fixed[pos] = value
            return PE(
                expr.head, expr.arity, fixed,
                alternating=expr.alternating, slot_kind=expr.slot_kind,
            )

        for pos, value in expr.fixed:
            if value == Integer(0):
                return Integer(0)
            if isinstance(value, Sum):
                return Sum(*(rebuilt(pos, t) for t in value.children))
            if isinstance(value, Neg):
                return Neg(rebuilt(pos, value.arg))
            split = self._scalar_split(value)
            if split is not None:
                scalar, core = split
                return Product(scalar, rebuilt(pos, core))
        return expr  # pragma: no cover


class AlternatingPartialEvalAsInteriorDefinition(Definition):
    """``ω(X₁, …, X_j, ·, …) → ι_{X_j}⋯ι_{X_1} ω`` (alternating, vector
    slots); non-leading fixed positions carry the sign of the
    permutation that moves them to the front."""

    name = (
        "partial evaluation of an alternating form = iterated interior "
        "product (with the slot-permutation sign)"
    )
    anchor = None

    def __init__(self) -> None:
        self.anchor = _pe_cls()

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, _pe_cls())
            and expr.alternating
            and expr.slot_kind == "vector"
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.objects.interior import contract_all

        fixed = expr.fixed_positions
        # permutation taking (fixed…, open…) to (0, 1, …, k−1)
        perm = list(fixed) + list(expr.open_positions)
        sign = _permutation_sign(perm)
        out = contract_all(expr.head, *expr.fixed_args)
        return out if sign > 0 else Neg(out)


class MusicalAsPartialEvalDefinition(Definition):
    """``g^♭(X) → g(X, ·)`` and ``π^♯(α) → π(α, ·)`` — the bilinear
    musical atoms as one-slot partial evaluations (alternating iff the
    carrier is a form / multivector)."""

    name = "musical map = one-slot partial evaluation: g♭(X) = g(X,·), π♯(α) = π(α,·)"
    anchor = Act

    def matches(self, expr: Expr) -> bool:
        from jacopy.central.objects.musical import Flat, Sharp

        return isinstance(expr, Act) and isinstance(expr.op, (Flat, Sharp))

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.objects.musical import Flat
        from jacopy.central.objects.partial_eval import partial_eval

        if isinstance(expr.op, Flat):
            return partial_eval(expr.op.metric, expr.arg, None, slot_kind="vector")
        return partial_eval(expr.op.bivector, expr.arg, None, slot_kind="covector")
