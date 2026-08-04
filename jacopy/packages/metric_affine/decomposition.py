"""
Frame-decomposition axioms and IndexedSum operator pushes
(Phase 4.D.2b).

**Decompositions** (opt-in via ``metric_affine_engine(...,
decompositions=((conn, fr), ...))``) — the completeness of a frame,
stated on the two section-producing shapes the component layer meets:

    ∇_{e_b} e_c → Σ_s Γ^s_{bc} e_s,
    [e_a, e_b]  → Σ_s γ^s_{ab} e_s.

Both pick a bound-index name FRESH for the matched node (cycling
``s, t, u, v, w, x``), so nested decompositions cannot capture an
outer bound index; α-equivalence of :class:`IndexedSum` makes the
choice irrelevant to equality.

**Pushes** (:class:`IndexedSumOperatorPushDefinition`, always-on with
the indexed layer): linear operators commute with the finite sum —

    D(Σ_s t_s)        → Σ_s D(t_s)          (operator argument),
    ∇_{Σ_s V_s} T     → Σ_s ∇_{V_s} T       (connection direction),
    ⟨α, Σ_s V_s⟩      → Σ_s ⟨α, V_s⟩        (pairing argument slot),
    g(Σ_s V_s, W)-style slots are already covered by the bilinearity
    rules firing inside the sum body.

All pushes are capture-guarded (the other operand must be free of the
bound index).
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Expr, Product
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.pairing import Pairing
from jacopy.proof.expansion import Definition
from jacopy.central.calculus.indexed_rules import contains_index
from jacopy.central.objects.connection import Connection, CovariantOp
from jacopy.central.objects.frame import Frame, FrameField, FrameIndex
from jacopy.central.tangent.anholonomy import AnholonomyCoefficient
from jacopy.packages.metric_affine.frame_components import (
    ConnectionCoefficient,
)

_BOUND_CANDIDATES = ("s", "t", "u", "v", "w", "x")


def _fresh_bound(*taken: str) -> str:
    for name in _BOUND_CANDIDATES:
        if name not in taken:
            return name
    raise ValueError(
        "no fresh bound-index name available; rename the free indices"
    )


class ConnectionFrameDecompositionDefinition(Definition):
    """``∇_{e_b} e_c → Σ_s Γ^s_{bc} e_s`` — completeness of the frame
    for the declared ``(∇, frame)`` pair (opt-in axiom)."""

    anchor = Act

    def __init__(self, conn: Connection, fr: Frame) -> None:
        if not isinstance(conn, Connection):
            raise TypeError(
                "ConnectionFrameDecompositionDefinition expects a Connection"
            )
        if not isinstance(fr, Frame):
            raise TypeError(
                "ConnectionFrameDecompositionDefinition expects a Frame"
            )
        self._conn = conn
        self._frame = fr
        self.name = (
            f"frame decomposition ({conn.name}, {fr.name}): "
            "∇_{e_b} e_c = Σ_s Γ^s_bc e_s"
        )

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, CovariantOp)):
            return False
        if expr.op.connection_name != self._conn.name:
            return False
        direction, arg = expr.op.vector, expr.arg
        return (
            isinstance(direction, FrameField)
            and isinstance(arg, FrameField)
            and all(
                leg.base_name == self._frame.name
                and leg.bundle == self._frame.bundle
                for leg in (direction, arg)
            )
        )

    def rewrite(self, expr: Expr) -> Expr:
        b = expr.op.vector.index
        c = expr.arg.index
        s = _fresh_bound(b, c)
        return IndexedSum(
            FrameIndex(s),
            self._frame,
            Product(
                ConnectionCoefficient(
                    self._conn.name, self._frame.name, s, b, c
                ),
                self._frame.field(s),
            ),
        )


class BracketFrameDecompositionDefinition(Definition):
    """``[e_a, e_b] → Σ_s γ^s_{ab} e_s`` — completeness of the frame
    for the bracket (opt-in axiom; the summed counterpart of the 2.E
    coefficient extraction)."""

    anchor = LieBracketVF

    def __init__(self, fr: Frame) -> None:
        if not isinstance(fr, Frame):
            raise TypeError(
                "BracketFrameDecompositionDefinition expects a Frame"
            )
        self._frame = fr
        self.name = (
            f"frame decomposition ({fr.name}): [e_a, e_b] = Σ_s γ^s_ab e_s"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, LieBracketVF):
            return False
        legs = (expr.X, expr.Y)
        return all(
            isinstance(leg, FrameField)
            and leg.base_name == self._frame.name
            and leg.bundle == self._frame.bundle
            for leg in legs
        )

    def rewrite(self, expr: Expr) -> Expr:
        a = expr.X.index
        b = expr.Y.index
        s = _fresh_bound(a, b)
        return IndexedSum(
            FrameIndex(s),
            self._frame,
            Product(
                AnholonomyCoefficient(self._frame.name, s, a, b),
                self._frame.field(s),
            ),
        )


class IndexedSumOperatorPushDefinition(Definition):
    """Linear operators commute with the finite indexed sum:
    operator arguments, connection directions and pairing argument
    slots all push inside (capture-guarded)."""

    name = (
        "indexed sum push: D(Σ t_s) = Σ D(t_s), ∇_{Σ V_s} = Σ ∇_{V_s}, "
        "⟨α, Σ V_s⟩ = Σ ⟨α, V_s⟩"
    )
    anchor = (Act, Pairing)

    def matches(self, expr: Expr) -> bool:
        from jacopy.central.algebroid.context import AnchoredVF

        if isinstance(expr, Pairing):
            return isinstance(expr.X, IndexedSum)
        if isinstance(expr, Act):
            if isinstance(expr.arg, IndexedSum):
                return True
            op = expr.op
            if isinstance(op, CovariantOp) and isinstance(
                op.vector, IndexedSum
            ):
                return True
            # ρ(Σ_s V_s)(f) = Σ_s ρ(V_s)(f) — the anchor's section
            # slot (Phase 4.F.6: d̂² needs ρ(Σ𝒫L) to meet the
            # projector's kernel rule termwise).
            return isinstance(op, AnchoredVF) and isinstance(
                op.section, IndexedSum
            )
        return False

    @staticmethod
    def _renamed_on_capture(inner: IndexedSum, *others: Expr):
        """α-convert ``inner`` when the surrounding operands mention
        its bound name; ``None`` when no capture."""
        from jacopy.central.calculus.indexed_rules import (
            contains_index,
            fresh_bound_name,
        )

        dummy = inner.dummy._repr_inner()
        if not any(contains_index(o, dummy) for o in others):
            return None
        fresh = fresh_bound_name(inner.body, *others)
        return inner.with_dummy(FrameIndex(fresh))

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr, Pairing):
            inner = expr.X
            renamed = self._renamed_on_capture(inner, expr.alpha)
            if renamed is not None:
                return Pairing(expr.alpha, renamed)
            return IndexedSum(
                inner.dummy,
                inner.range_,
                Pairing(expr.alpha, inner.body),
            )
        if isinstance(expr.arg, IndexedSum):
            inner = expr.arg
            renamed = self._renamed_on_capture(inner, expr.op)
            if renamed is not None:
                return Act(expr.op, renamed)
            return IndexedSum(
                inner.dummy,
                inner.range_,
                Act(expr.op, inner.body),
            )
        from jacopy.central.algebroid.context import AnchoredVF

        op = expr.op
        if isinstance(op, AnchoredVF):
            inner = op.section
            renamed = self._renamed_on_capture(inner, expr.arg)
            if renamed is not None:
                return Act(
                    AnchoredVF(op.algebroid_name, renamed), expr.arg
                )
            return IndexedSum(
                inner.dummy,
                inner.range_,
                Act(
                    AnchoredVF(op.algebroid_name, inner.body),
                    expr.arg,
                ),
            )
        inner = op.vector
        renamed = self._renamed_on_capture(inner, expr.arg)
        if renamed is not None:
            return Act(
                CovariantOp(
                    op.connection_name, renamed, bundle=op.bundle
                ),
                expr.arg,
            )
        return IndexedSum(
            inner.dummy,
            inner.range_,
            Act(
                CovariantOp(
                    op.connection_name, inner.body, bundle=op.bundle
                ),
                expr.arg,
            ),
        )
