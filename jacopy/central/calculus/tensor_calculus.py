"""
Evaluation laws of ℒ and ∇ on general (q, r)-TENSORS (PDF items
9e / 8n; 2026-09-08 audit, compliance finding 2).

The canonical definitions — the Lie and covariant derivatives are
derivations of the full contraction algebra, so on a (q, r)-tensor
``T`` evaluated on ``q`` covectors and ``r`` vectors:

    (ℒ_X T)(a₁,…,a_{q+r}) = X(T(a₁,…)) − Σᵢ T(…, ℒ_X aᵢ, …),
    (∇_X T)(a₁,…,a_{q+r}) = X(T(a₁,…)) − Σᵢ T(…, ∇_X aᵢ, …).

These fire ONLY on the mixed, non-alternating evaluation of a
REGISTERED :class:`~jacopy.central.objects.tensor.Tensor` head with
the exact arity — the form/multivector calculus keeps its own
alternating rules (whose flags are inherited, never overwritten:
the reliability-pass contract).
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition
from jacopy.central.objects.tensor import Tensor
from jacopy.central.objects.connection import CovariantOp
from jacopy.central.calculus.bracket_calculus import (
    LieDerivative,
)


def _tensor_head(expr: Expr, op_type) -> Optional[tuple]:
    """``(operator, T)`` when ``expr`` is a mixed MultiEval of an
    ``op``-applied registered Tensor with matching arity."""
    if not isinstance(expr, MultiEval):
        return None
    if expr.alternating or expr.slot_kind != "mixed":
        return None
    head = expr.head
    if not (
        isinstance(head, Act) and isinstance(head.op, op_type)
    ):
        return None
    T = head.arg
    if not isinstance(T, Tensor):
        return None
    if T.upper + T.lower != expr.arity:
        return None
    return head.op, T


class TensorLieEvalDefinition(Definition):
    """``(ℒ_X T)(a₁,…) → X(T(a₁,…)) − Σᵢ T(…, ℒ_X aᵢ, …)`` — the
    canonical evaluation law of the Lie derivative on a (q,r)-tensor
    (PDF 9e). Slot-agnostic: the same ``ℒ_X`` acts in covector and
    vector slots (the intrinsic rules then turn ``ℒ_X Y`` into
    ``[X, Y]``)."""

    name = (
        "tensor Lie evaluation: (ℒ_X T)(a…) = X(T(a…)) − "
        "Σ T(…, ℒ_X aᵢ, …)"
    )
    anchor = MultiEval

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        return (
            _tensor_head(expr, LieDerivative) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        op, T = _tensor_head(expr, LieDerivative)
        X = op.vector
        args = expr.args
        terms = [
            Act(
                X,
                MultiEval(
                    T,
                    *args,
                    alternating=False,
                    slot_kind="mixed",
                ),
            )
        ]
        for i, a in enumerate(args):
            replaced = (
                args[:i] + (Act(op, a),) + args[i + 1 :]
            )
            terms.append(
                Neg(
                    MultiEval(
                        T,
                        *replaced,
                        alternating=False,
                        slot_kind="mixed",
                    )
                )
            )
        return Sum(*terms)


class TensorCovariantEvalDefinition(Definition):
    """``(∇_X T)(a₁,…) → X(T(a₁,…)) − Σᵢ T(…, ∇_X aᵢ, …)`` — the
    canonical evaluation law of the covariant derivative on a
    (q,r)-tensor (PDF 8n)."""

    name = (
        "tensor covariant evaluation: (∇_X T)(a…) = X(T(a…)) − "
        "Σ T(…, ∇_X aᵢ, …)"
    )
    anchor = MultiEval

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        return _tensor_head(expr, CovariantOp) is not None

    def rewrite(self, expr: Expr) -> Expr:
        op, T = _tensor_head(expr, CovariantOp)
        X = op.vector
        args = expr.args
        terms = [
            Act(
                X,
                MultiEval(
                    T,
                    *args,
                    alternating=False,
                    slot_kind="mixed",
                ),
            )
        ]
        for i, a in enumerate(args):
            replaced = (
                args[:i] + (Act(op, a),) + args[i + 1 :]
            )
            terms.append(
                Neg(
                    MultiEval(
                        T,
                        *replaced,
                        alternating=False,
                        slot_kind="mixed",
                    )
                )
            )
        return Sum(*terms)
