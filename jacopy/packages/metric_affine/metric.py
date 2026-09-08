"""
The metric layer of the metric-affine package (Phase 4.C).

* :class:`Metric` — a pseudo-Riemannian metric context ``g`` on TM;
  ``g(X, Y)`` is a scalar :class:`MetricValue` atom. Symmetry and
  ``C^∞``-bilinearity are definitional (g is a symmetric (0,2)-tensor
  by definition); non-degeneracy is definitional too and licenses the
  generic-section agreement steps of the tactics (never a rewrite).
* :class:`NonMetricity` — the canonical definition
  ``Q(X, Y, Z) := X(g(Y,Z)) − g(∇_X Y, Z) − g(Y, ∇_X Z)``.
* Two OPT-IN declared rules (the package analogue of the algebroid
  declaration system; passed to ``metric_affine_engine``):

  - :class:`MetricCompatibilityDefinition` — ``∇ g = 0`` as the
    rewrite ``X(g(Y,Z)) → g(∇_X Y, Z) + g(Y, ∇_X Z)``;
  - :class:`TorsionFreeDefinition` — ``T(∇) = 0`` as the rewrite
    ``[X, Y] → ∇_X Y − ∇_Y X``.

* Theorems: ``Q`` is tensorial and ``(Y,Z)``-symmetric (definitional);
  ``Q = 0`` under compatibility and ``T = 0`` under torsion-freeness
  (consistency); the **generalized Koszul formula** (the Schouten
  decomposition in identity form — holds for ANY connection, purely
  definitionally); the **classical Koszul formula** under the two
  declarations; and **Levi-Civita uniqueness** (a tactic: both
  connections satisfy the same Koszul right-hand side, then
  non-degeneracy cancels the generic section).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Atom, Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.objects.bundle import Bundle, TM
from jacopy.central.objects.connection import Connection, CovariantOp
from jacopy.algebra.lie_bracket_vf import LieBracketVF


class MetricValue(Atom):
    """``g(X, Y)`` — a metric evaluation, a scalar ``C^∞`` function
    (degree-0 atom, NOT a derivation; slot protocol on ``X, Y``)."""

    __slots__ = ("_metric_name", "_X", "_Y")

    def __init__(self, metric_name: str, X: Expr, Y: Expr) -> None:
        for s in (X, Y):
            if not isinstance(s, Expr):
                raise TypeError("MetricValue requires Expr arguments")
        self._metric_name = metric_name
        self._X = X
        self._Y = Y

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def X(self) -> Expr:
        return self._X

    @property
    def Y(self) -> Expr:
        return self._Y

    @property
    def degree(self) -> Degree:
        """A metric evaluation is a scalar function."""
        return Degree.const(0)

    @property
    def rewritable_slots(self):
        return (self._X, self._Y)

    def with_slots(self, X: Expr, Y: Expr) -> "MetricValue":
        return MetricValue(self._metric_name, X, Y)

    def _key(self) -> Any:
        return (self._metric_name, self._X, self._Y)

    def _repr_inner(self) -> str:
        return (
            f"{self._metric_name}({self._X._repr_inner()},"
            f"{self._Y._repr_inner()})"
        )

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        """Slot-walking substitution (the base Atom walk sees no
        children; the metric's arguments live in slots — needed for
        index renaming under an IndexedSum, Phase 4.D.2c)."""
        if self == dummy:
            return target
        new_X = self._X.substitute_atom(dummy, target)
        new_Y = self._Y.substitute_atom(dummy, target)
        if new_X is self._X and new_Y is self._Y:
            return self
        return MetricValue(self._metric_name, new_X, new_Y)


class Metric:
    """``g`` — a (pseudo-)Riemannian metric context on TM (symmetric,
    non-degenerate (0,2)-tensor BY DEFINITION)."""

    __slots__ = ("_name",)

    def __init__(self, name: str = "g") -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Metric name must be a non-empty str")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, X: Expr, Y: Expr) -> MetricValue:
        if not isinstance(X, Expr) or not isinstance(Y, Expr):
            raise TypeError("Metric expects Expr arguments")
        return MetricValue(self._name, X, Y)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Metric) and self._name == other._name

    def __hash__(self) -> int:
        return hash(("tm-metric", self._name))

    def __repr__(self) -> str:
        return f"Metric({self._name!r})"


def metric(name: str = "g") -> Metric:
    """Create a metric context."""
    return Metric(name)


def as_metric_context(g) -> Metric:
    """COERCE a metric object into this package's context (the PDF
    item 5 compatibility bridge; 2026-09-08 audit, compliance
    finding 3).

    Accepts either this package's :class:`Metric` context or the
    central :class:`jacopy.central.objects.metric.Metric` ATOM — the
    two carry the same defining semantics (a symmetric,
    non-degenerate (0,2)-tensor BY DEFINITION), differing only in
    their role (Expr atom for Hodge/musical vs evaluation context
    here), so the bridge is the shared name. Anything else raises.
    """
    if isinstance(g, Metric):
        return g
    from jacopy.central.objects.metric import (
        Metric as CentralMetric,
    )

    if isinstance(g, CentralMetric):
        return Metric(g.name)
    raise TypeError(
        "expected a metric (metric_affine context or the central "
        f"Metric atom), got {type(g).__name__}"
    )


# --------------------------------------------------------------------- #
# Definitional rules for g                                               #
# --------------------------------------------------------------------- #


class MetricValueSymmetryDefinition(Definition):
    """``g(Y, X) → g(X, Y)`` — canonical slot order (symmetric by
    definition; canonical-form family)."""

    name = "metric symmetry: g(Y, X) = g(X, Y) (canonical order)"
    anchor = MetricValue

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MetricValue)
            and expr.X._repr_inner() > expr.Y._repr_inner()
        )

    def rewrite(self, expr: Expr) -> Expr:
        return expr.with_slots(expr.Y, expr.X)


class MetricValueBilinearityDefinition(Definition):
    """``C^∞``-bilinearity of ``g`` (a (0,2)-tensor by definition)."""

    name = "metric bilinearity: g(fX + W, Y) = f·g(X,Y) + g(W,Y) (both slots)"
    anchor = MetricValue

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            if is_scalar_function(head, self._registry):
                rest = expr.children[1:]
                return head, (rest[0] if len(rest) == 1 else Product(*rest))
        return None

    def _splittable(self, slot: Expr) -> bool:
        if isinstance(slot, (Sum, Neg)) or slot == Integer(0):
            return True
        return self._scalar_split(slot) is not None

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, MetricValue) and any(
            self._splittable(s) for s in expr.rewritable_slots
        )

    def rewrite(self, expr: Expr) -> Expr:
        slots = tuple(expr.rewritable_slots)
        for i, slot in enumerate(slots):
            if not self._splittable(slot):
                continue

            def rebuild(sub: Expr) -> Expr:
                new = slots[:i] + (sub,) + slots[i + 1 :]
                return expr.with_slots(*new)

            if slot == Integer(0):
                return Integer(0)
            if isinstance(slot, Sum):
                return Sum(*(rebuild(c) for c in slot.children))
            if isinstance(slot, Neg):
                return Neg(rebuild(slot.arg))
            scalar, rest = self._scalar_split(slot)
            return Product(scalar, rebuild(rest))
        raise AssertionError("rewrite called without a splittable slot")


# --------------------------------------------------------------------- #
# Non-metricity                                                          #
# --------------------------------------------------------------------- #


class InverseMetricComponent(Atom):
    """``g^{ab}`` — a component of the inverse metric on a frame; a
    scalar function. Symmetric: the factory
    :func:`inverse_metric_component` canonicalizes the index order.
    Its DEFINING property is the contraction
    ``Σ_s g^{as}·g(e_s, e_c) = δ^a_c``
    (:class:`MetricInverseContractionDefinition`), well-posed by the
    metric's definitional non-degeneracy."""

    __slots__ = ("_metric_name", "_first", "_second")

    def __init__(self, metric_name: str, first: str, second: str) -> None:
        self._metric_name = metric_name
        self._first = first
        self._second = second

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def upper(self) -> Tuple[str, str]:
        return (self._first, self._second)

    @property
    def degree(self) -> Degree:
        return Degree.const(0)

    @property
    def index_names(self) -> Tuple[str, ...]:
        return (self._first, self._second)

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        from jacopy.central.objects.frame import substituted_index_names

        if self == dummy:
            return target
        renamed = substituted_index_names(
            self.index_names, dummy, target
        )
        if renamed is None:
            return self
        return inverse_metric_component_named(
            self._metric_name, *renamed
        )

    def _key(self) -> Any:
        return (self._metric_name, self._first, self._second)

    def _repr_inner(self) -> str:
        return f"{self._metric_name}^{self._first}{self._second}"


def inverse_metric_component_named(
    metric_name: str, first: str, second: str
) -> InverseMetricComponent:
    """``g^{ab}`` in canonical (sorted) index order — the symmetry of
    the inverse metric as a canonical form."""
    if first > second:
        first, second = second, first
    return InverseMetricComponent(metric_name, first, second)


def inverse_metric_component(
    g: Metric, first, second
) -> InverseMetricComponent:
    """``g^{ab}`` of the metric ``g`` (canonical index order)."""
    from jacopy.central.objects.frame import _index_str

    g = as_metric_context(g)
    return inverse_metric_component_named(
        g.name, _index_str(first), _index_str(second)
    )


class MetricInverseContractionDefinition(Definition):
    """``Σ_s g^{as}·g(e_s, e_c)·rest → δ^a_c·rest`` — the defining
    contraction of the inverse metric (rest must be free of the bound
    index)."""

    name = "inverse metric contraction: Σ_s g^{as} g(e_s, e_c) = δ^a_c"

    def __init__(self) -> None:
        from jacopy.core.indexed_sum import IndexedSum

        self.anchor = IndexedSum

    def _split(self, expr):
        from jacopy.core.indexed_sum import IndexedSum
        from jacopy.central.calculus.indexed_rules import contains_index
        from jacopy.central.objects.frame import FrameField

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
        inv = None
        val = None
        for k, c in enumerate(factors):
            if (
                inv is None
                and isinstance(c, InverseMetricComponent)
                and dummy in c.index_names
            ):
                inv = k
            elif (
                val is None
                and isinstance(c, MetricValue)
                and isinstance(c.X, FrameField)
                and isinstance(c.Y, FrameField)
                and dummy in (c.X.index, c.Y.index)
            ):
                val = k
        if inv is None or val is None:
            return None
        g_inv = factors[inv]
        g_val = factors[val]
        if g_inv.metric_name != g_val.metric_name:
            return None
        up = [n for n in g_inv.index_names if n != dummy]
        low = [
            leg.index
            for leg in (g_val.X, g_val.Y)
            if leg.index != dummy
        ]
        if len(up) != 1 or len(low) != 1:
            return None
        rest = [
            c for k, c in enumerate(factors) if k not in (inv, val)
        ]
        if any(contains_index(c, dummy) for c in rest):
            return None
        return sign, rest, up[0], low[0]

    def matches(self, expr: Expr) -> bool:
        return self._split(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.objects.frame import kronecker_delta

        sign, rest, up, low = self._split(expr)
        delta = kronecker_delta(up, low)
        out: Expr = (
            delta if not rest else Product(delta, *rest)
        )
        return Neg(out) if sign else out


class NonMetricity(Atom):
    """``Q(X, Y, Z)`` — the non-metricity of ``(∇, g)``, a scalar
    ``C^∞`` function (canonical definition
    ``Q(X,Y,Z) := X(g(Y,Z)) − g(∇_X Y, Z) − g(Y, ∇_X Z)``)."""

    __slots__ = ("_connection_name", "_bundle", "_metric_name", "_X", "_Y", "_Z")

    def __init__(
        self,
        connection_name: str,
        metric_name: str,
        X: Expr,
        Y: Expr,
        Z: Expr,
        *,
        bundle: Optional[Bundle] = None,
    ) -> None:
        for s in (X, Y, Z):
            if not isinstance(s, Expr):
                raise TypeError("NonMetricity requires Expr arguments")
        self._connection_name = connection_name
        self._bundle = bundle if bundle is not None else TM
        self._metric_name = metric_name
        self._X = X
        self._Y = Y
        self._Z = Z

    @property
    def connection_name(self) -> str:
        return self._connection_name

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    @property
    def arguments(self) -> Tuple[Expr, Expr, Expr]:
        return (self._X, self._Y, self._Z)

    @property
    def rewritable_slots(self):
        return (self._X, self._Y, self._Z)

    def with_slots(self, X: Expr, Y: Expr, Z: Expr) -> "NonMetricity":
        return NonMetricity(
            self._connection_name,
            self._metric_name,
            X,
            Y,
            Z,
            bundle=self._bundle,
        )

    @property
    def degree(self) -> Degree:
        """Non-metricity evaluations are scalar functions."""
        return Degree.const(0)

    def _key(self) -> Any:
        return (
            self._connection_name,
            self._bundle,
            self._metric_name,
            self._X,
            self._Y,
            self._Z,
        )

    def _repr_inner(self) -> str:
        return (
            f"Q({self._X._repr_inner()},{self._Y._repr_inner()},"
            f"{self._Z._repr_inner()})"
        )


def nonmetricity(
    conn: Connection, g: Metric, X: Expr, Y: Expr, Z: Expr
) -> NonMetricity:
    """``Q(X, Y, Z)`` of the pair ``(∇, g)``."""
    if not isinstance(conn, Connection):
        raise TypeError("nonmetricity expects a Connection")
    g = as_metric_context(g)
    return NonMetricity(
        conn.name, g.name, X, Y, Z, bundle=conn.bundle
    )


class NonMetricityExpansionDefinition(Definition):
    """``Q(X,Y,Z) → X(g(Y,Z)) − g(∇_X Y, Z) − g(Y, ∇_X Z)``
    (canonical definition)."""

    name = "non-metricity definition: Q(X,Y,Z) = X(g(Y,Z)) − g(∇_X Y,Z) − g(Y,∇_X Z)"
    anchor = NonMetricity

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, NonMetricity)

    def rewrite(self, expr: Expr) -> Expr:
        X, Y, Z = expr.arguments
        g = expr.metric_name

        def nabla(a: Expr, b: Expr) -> Expr:
            return Act(
                CovariantOp(expr.connection_name, a, bundle=expr.bundle), b
            )

        return Sum(
            Act(X, MetricValue(g, Y, Z)),
            Neg(MetricValue(g, nabla(X, Y), Z)),
            Neg(MetricValue(g, Y, nabla(X, Z))),
        )


# --------------------------------------------------------------------- #
# Opt-in declared rules (compatibility, torsion-freeness)                #
# --------------------------------------------------------------------- #


def _is_vector_direction(op: Expr) -> bool:
    """A section acting as a derivation (wedge-degree-1 protocol) —
    NOT a composite operator like ``∇_X`` or ``L_X``."""
    return (
        isinstance(op, Derivation)
        and getattr(op, "wedge_degree", None) == Degree.const(1)
    )


class MetricCompatibilityDefinition(Definition):
    """Declared ``∇g = 0``:
    ``X(g(Y,Z)) → g(∇_X Y, Z) + g(Y, ∇_X Z)`` for the declared
    ``(∇, g)`` pair (the tangent analogue of the algebroid C1)."""

    anchor = Act

    def __init__(self, conn: Connection, g: Metric) -> None:
        if not isinstance(conn, Connection):
            raise TypeError("MetricCompatibilityDefinition expects a Connection")
        g = as_metric_context(g)
        self._conn = conn
        self._g = g
        self.name = (
            f"metric compatibility ({conn.name}, {g.name}): "
            "X(g(Y,Z)) = g(∇_X Y, Z) + g(Y, ∇_X Z)"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and _is_vector_direction(expr.op)
            and isinstance(expr.arg, MetricValue)
            and expr.arg.metric_name == self._g.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        X = expr.op
        Y, Z = expr.arg.X, expr.arg.Y
        g = self._g
        nabla = self._conn
        return Sum(
            g(nabla(X, Y), Z),
            g(Y, nabla(X, Z)),
        )


class TorsionFreeDefinition(Definition):
    """Declared ``T(∇) = 0``:
    ``[X, Y] → ∇_X Y − ∇_Y X`` (the bracket expressed through the
    declared torsion-free connection)."""

    anchor = LieBracketVF

    def __init__(self, conn: Connection) -> None:
        if not isinstance(conn, Connection):
            raise TypeError("TorsionFreeDefinition expects a Connection")
        self._conn = conn
        self.name = (
            f"torsion-free ({conn.name}): [X, Y] = ∇_X Y − ∇_Y X"
        )

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, LieBracketVF)

    def rewrite(self, expr: Expr) -> Expr:
        nabla = self._conn
        return Sum(
            nabla(expr.X, expr.Y), Neg(nabla(expr.Y, expr.X))
        )
