"""
The GIVEN-INPUT evaluation workflow (PDF items 11e / 12f; 2026-09-08
audit, compliance finding "girdi-akışı"): the user supplies concrete
COMPONENT DATA — connection coefficients ``Γ^a_{bc}``, metric
components ``g_{ab}``, anholonomy coefficients ``γ^c_{ab}``, Poisson
bivector components ``θ^{ab}`` — and the engine EVALUATES the
package's symbolic component expressions against them.

Usage sketch::

    fr = frame()
    data = ComponentInput(fr, dim=2, connection=conn, metric=g)
    data.set_gamma("0", "0", "1", f)        # Γ^0_{01} = f
    data.set_metric("0", "0", Integer(1))   # g_00 = 1
    data.set_metric("1", "1", h)            # g_11 = h
    value = data.evaluate(some_component_expr, registry=reg)

Rules of the game:

* every coefficient NOT set defaults to ``0`` **only when the input
  is declared** ``total=True`` (the user asserts the data is
  complete); otherwise unset coefficients stay SYMBOLIC —
  soundness over convenience;
* the frame is declared over ``dim`` concrete positions, so
  Kronecker deltas with distinct concrete labels evaluate to ``0``
  and equal labels to ``1`` (the audit's "fixed basis positions");
* the values may be numbers or registered scalar functions —
  derivatives of given functions (``e_0(h)``) remain symbolic
  first-class output, expressed purely in the given data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from jacopy.core.expr import Expr, Integer
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.central.objects.frame import (
    CoframeField,
    Frame,
    KroneckerDelta,
)
from jacopy.central.tangent.anholonomy import (
    AnholonomyCoefficient,
)
from jacopy.packages.metric_affine.frame_components import (
    ConnectionCoefficient,
)
from jacopy.packages.metric_affine.metric import MetricValue


class ComponentInput:
    """A bundle of user-supplied component data on a frame."""

    def __init__(
        self,
        fr: Frame,
        *,
        dim: Optional[int] = None,
        connection=None,
        metric=None,
        pi=None,
        total: bool = False,
        holonomic: bool = False,
    ) -> None:
        self._frame = fr
        self._dim = dim
        self._connection = connection
        self._metric = metric
        self._pi = pi
        self._total = bool(total)
        self._holonomic = bool(holonomic)
        self._indices: Tuple[str, ...] = (
            tuple(str(i) for i in range(dim))
            if dim is not None
            else ()
        )
        self._gamma: Dict[Tuple[str, str, str], Expr] = {}
        self._anholonomy: Dict[
            Tuple[str, str, str], Expr
        ] = {}
        self._metric_components: Dict[
            Tuple[str, str], Expr
        ] = {}
        self._pi_components: Dict[Tuple[str, str], Expr] = {}

    # ---- data entry ------------------------------------------------ #

    @staticmethod
    def _value(v) -> Expr:
        if isinstance(v, int):
            return Integer(v)
        if not isinstance(v, Expr):
            raise TypeError(
                "component values must be Expr or int"
            )
        return v

    def set_gamma(self, upper, first, second, value) -> None:
        """``Γ^upper_{first second} := value``."""
        self._gamma[
            (str(upper), str(first), str(second))
        ] = self._value(value)

    def set_anholonomy(
        self, upper, first, second, value
    ) -> None:
        """``γ^upper_{first second} := value`` (and the
        antisymmetric partner is implied by the caller)."""
        self._anholonomy[
            (str(upper), str(first), str(second))
        ] = self._value(value)

    def set_metric(self, a, b, value) -> None:
        """``g_{ab} := value`` (symmetric: both orders stored)."""
        v = self._value(value)
        self._metric_components[(str(a), str(b))] = v
        self._metric_components[(str(b), str(a))] = v

    def set_pi(self, a, b, value) -> None:
        """``θ^{ab} := value`` (antisymmetric: the swapped order
        stores the negated value)."""
        from jacopy.core.expr import Neg

        v = self._value(value)
        self._pi_components[(str(a), str(b))] = v
        self._pi_components[(str(b), str(a))] = (
            Integer(0) if v == Integer(0) else Neg(v)
        )

    # ---- lookups (None = stay symbolic) ---------------------------- #

    def _concrete(self, key) -> bool:
        """True when every label denotes a declared basis position —
        bound (dummy) labels stay symbolic so IndexedSum bodies are
        unrolled BEFORE substitution, never zeroed under them."""
        if not self._indices:
            return True
        return all(k in self._indices for k in key)

    def gamma(self, key) -> Optional[Expr]:
        if key in self._gamma:
            return self._gamma[key]
        if self._total and self._concrete(key):
            return Integer(0)
        return None

    def anholonomy_value(self, key) -> Optional[Expr]:
        if key in self._anholonomy:
            return self._anholonomy[key]
        if (
            self._holonomic or self._total
        ) and self._concrete(key):
            return Integer(0)
        return None

    def metric_component(self, key) -> Optional[Expr]:
        if key in self._metric_components:
            return self._metric_components[key]
        if self._total and self._concrete(key):
            return Integer(0)
        return None

    def pi_component(self, key) -> Optional[Expr]:
        if key in self._pi_components:
            return self._pi_components[key]
        if self._total and self._concrete(key):
            return Integer(0)
        return None

    @property
    def frame(self) -> Frame:
        return self._frame

    # ---- evaluation ------------------------------------------------ #

    def evaluate(
        self,
        expr: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
        engine: Optional[ExpansionEngine] = None,
    ) -> Expr:
        """Normalize ``expr`` with the package engine PLUS the
        substitution rules for this data set.

        The caller's ``engine`` is NEVER mutated: a fresh engine is
        layered per call, so data-set rules cannot accumulate in a
        shared engine and leak between evaluations (2026-09-09
        audit, finding 4 — the result must depend on the expression
        and THIS data set only, not on prior calls)."""
        from jacopy.proof.expansion import ExpansionEngine
        from jacopy.packages.metric_affine.engine import (
            metric_affine_engine,
        )
        from jacopy.packages.poisson.tilde import (
            _normalized_by,
        )

        base = (
            engine
            if engine is not None
            else metric_affine_engine(registry=registry)
        )
        eng = ExpansionEngine(
            [
                IndexedSumUnrollDefinition(self),
                SubstituteComponentsDefinition(self),
                ConcreteDeltaDefinition(self),
            ]
        )
        for d_ in base.definitions:
            eng.register(d_)
        return _normalized_by(eng, expr, registry)


class SubstituteComponentsDefinition(Definition):
    """Rewrite the coefficient atoms of THIS input's frame (and
    connection/metric/bivector names) to their given values; unset
    coefficients stay symbolic unless the input is total."""

    anchor = (
        ConnectionCoefficient,
        AnholonomyCoefficient,
        MetricValue,
        MultiEval,
    )

    def __init__(self, data: ComponentInput) -> None:
        self._d = data
        self.name = (
            f"substitute given components ({data.frame.name})"
        )

    def _lookup(self, expr: Expr) -> Optional[Expr]:
        d = self._d
        fr = d.frame
        if isinstance(expr, ConnectionCoefficient):
            if expr.frame_name != fr.name:
                return None
            if (
                d._connection is not None
                and expr.connection_name
                != d._connection.name
            ):
                return None
            return d.gamma((expr.upper,) + expr.lower)
        if isinstance(expr, AnholonomyCoefficient):
            if expr.frame_name != fr.name:
                return None
            return d.anholonomy_value(
                (expr.upper,) + expr.lower
            )
        if isinstance(expr, MetricValue):
            if d._metric is None:
                return None
            if expr.metric_name != d._metric.name:
                return None
            X, Y = expr.X, expr.Y
            from jacopy.central.objects.frame import (
                FrameField,
            )

            # identity includes the BUNDLE: a same-named frame on
            # another bundle must not receive this data set's
            # values (2026-09-09 recheck, finding B)
            if not (
                isinstance(X, FrameField)
                and isinstance(Y, FrameField)
                and X.base_name == fr.name
                and Y.base_name == fr.name
                and X.bundle == fr.bundle
                and Y.bundle == fr.bundle
            ):
                return None
            return d.metric_component(
                (X.index, Y.index)
            )
        if isinstance(expr, MultiEval):
            if d._pi is None or expr.arity != 2:
                return None
            if expr.head != d._pi:
                return None
            a, b = expr.args
            if not (
                isinstance(a, CoframeField)
                and isinstance(b, CoframeField)
                and a.base_name == fr.name
                and b.base_name == fr.name
                and a.bundle == fr.bundle
                and b.bundle == fr.bundle
            ):
                return None
            return d.pi_component(
                (a.index, b.index)
            )
        return None

    def matches(self, expr: Expr) -> bool:
        return self._lookup(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        return self._lookup(expr)


class ConcreteDeltaDefinition(Definition):
    """On a DECLARED finite frame the index labels denote distinct
    basis positions: ``δ^a_b → 0`` when BOTH labels are declared
    concrete positions (equal labels already collapse to ``1`` at
    construction). A BOUND (dummy) label is structurally distinct
    from a concrete one without denoting a different position, so
    it stays symbolic until the enclosing sum is unrolled —
    zeroing it under the sum falsifies ``Σ_s δ^s_0 = 1``
    (2026-09-09 audit, finding 3)."""

    anchor = KroneckerDelta

    def __init__(self, data: ComponentInput) -> None:
        self._d = data
        self.name = (
            "concrete basis positions: δ^a_b = [a = b] "
            "(both labels concrete)"
        )

    def matches(self, expr: Expr) -> bool:
        if self._d._dim is None:
            return False
        if not isinstance(expr, KroneckerDelta):
            return False
        idx = self._d._indices
        return expr.upper in idx and expr.lower in idx

    def rewrite(self, expr: Expr) -> Expr:
        return (
            Integer(1)
            if expr.upper == expr.lower
            else Integer(0)
        )


class IndexedSumUnrollDefinition(Definition):
    """Unroll ``Σ_s body`` over the DECLARED finite frame:
    the bound index takes every concrete basis label. Fires only
    when the input carries a dimension; substitution rules then see
    concrete labels only (bound labels are guarded out of the
    zero-defaults)."""

    def __init__(self, data: ComponentInput) -> None:
        from jacopy.core.indexed_sum import IndexedSum

        self._d = data
        self.anchor = IndexedSum
        self.name = (
            f"unroll Σ over the {len(data._indices)}-dim frame "
            f"({data.frame.name})"
        )

    def matches(self, expr: Expr) -> bool:
        from jacopy.core.indexed_sum import IndexedSum

        if not self._d._indices:
            return False
        if not isinstance(expr, IndexedSum):
            return False
        # FULL frame identity (name + bundle), never name alone: a
        # same-named frame on another bundle keeps its own rank and
        # stays symbolic here (2026-09-09 recheck, finding B).
        return expr._range is self._d.frame or (
            expr._range == self._d.frame
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.objects.frame import FrameIndex
        from jacopy.core.expr import Sum as _Sum

        terms = [
            expr._body.substitute_atom(
                expr._dummy, FrameIndex(label)
            )
            for label in self._d._indices
        ]
        if not terms:
            return Integer(0)
        return terms[0] if len(terms) == 1 else _Sum(*terms)
