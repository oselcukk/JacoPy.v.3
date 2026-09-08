"""
Frame components of the connection (Phase 4.D.1).

The 2.E coefficient-extraction pattern applied to the connection:

    Γ^a_{bc} := ⟨e^a, ∇_{e_b} e_c⟩

(:class:`ConnectionCoefficient` — no index symmetry is imposed: a
general connection has none). On top of it:

* the **connection 1-form** ``ω^a_b`` with its defining evaluation
  ``ω^a_b(X) := ⟨e^a, ∇_X e_b⟩`` (so ``ω^a_b(e_c) = Γ^a_{cb}``),
* the **torsion components theorem**
  ``⟨e^a, T(e_b, e_c)⟩ = Γ^a_{bc} − Γ^a_{cb} − γ^a_{bc}``,
* the **first Cartan structure equation in frame-evaluated form**
  ``T^a(e_b, e_c) = de^a(e_b, e_c) + ω^a_c(e_b) − ω^a_b(e_c)``
  (the wedge term evaluated against the frame; the form-VALUED
  equations with genuine ``Σ_d ω^a_d ∧ e^d`` need the
  IndexedSum/Wedge layer — Phase 4.D.2).

Splitting a coframe pairing over a sum is GUARDED
(:class:`FramePairingSplitDefinition`): it fires only when a summand
is extraction-eligible, so it cannot ping-pong against the
pairing-collection phase of ``simplify`` (which regroups what the
split leaves behind — with no eligible summand left, the split stays
quiet).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Atom, Expr, Integer, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.objects.connection import Connection, CovariantOp
from jacopy.central.objects.frame import (
    CoframeField,
    Frame,
    FrameField,
    IndexLike,
    _index_str,
)
from jacopy.central.tangent.anholonomy import anholonomy_coefficient
from jacopy.central.tangent.exterior import d
from jacopy.packages.metric_affine.torsion_curvature import (
    curvature,
    torsion,
)


class ConnectionCoefficient(Atom):
    """``Γ^a_{bc}`` — a connection coefficient on a frame; a genuine
    function on the base (degree 0, not constant). No index symmetry
    is imposed."""

    __slots__ = ("_connection_name", "_frame_name", "_upper", "_first", "_second")

    def __init__(
        self,
        connection_name: str,
        frame_name: str,
        upper: str,
        first: str,
        second: str,
    ) -> None:
        self._connection_name = connection_name
        self._frame_name = frame_name
        self._upper = upper
        self._first = first
        self._second = second

    @property
    def connection_name(self) -> str:
        return self._connection_name

    @property
    def frame_name(self) -> str:
        return self._frame_name

    @property
    def upper(self) -> str:
        return self._upper

    @property
    def lower(self) -> Tuple[str, str]:
        return (self._first, self._second)

    @property
    def degree(self) -> Degree:
        return Degree.const(0)

    def _key(self) -> Any:
        return (
            self._connection_name,
            self._frame_name,
            self._upper,
            self._first,
            self._second,
        )

    def _repr_inner(self) -> str:
        base = f"Γ^{self._upper}_{self._first}{self._second}"
        if self._connection_name == "∇" and self._frame_name == "e":
            return base
        return f"Γ({self._connection_name},{self._frame_name})" + base[1:]

    @property
    def index_names(self) -> Tuple[str, ...]:
        return (self._upper, self._first, self._second)

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        from jacopy.central.objects.frame import substituted_index_names

        if self == dummy:
            return target
        renamed = substituted_index_names(
            self.index_names, dummy, target
        )
        if renamed is None:
            return self
        return ConnectionCoefficient(
            self._connection_name, self._frame_name, *renamed
        )


def connection_coefficient(
    conn: Connection,
    fr: Frame,
    upper: IndexLike,
    first: IndexLike,
    second: IndexLike,
) -> ConnectionCoefficient:
    """``Γ^a_{bc}`` of the connection on the frame."""
    if not isinstance(conn, Connection):
        raise TypeError("connection_coefficient expects a Connection")
    if not isinstance(fr, Frame):
        raise TypeError("connection_coefficient expects a Frame")
    return ConnectionCoefficient(
        conn.name,
        fr.name,
        _index_str(upper),
        _index_str(first),
        _index_str(second),
    )


class ConnectionForm(Atom):
    """``ω^a_b`` — the connection 1-form on a frame (degree 1), defined
    by its evaluation ``ω^a_b(X) := ⟨e^a, ∇_X e_b⟩``."""

    __slots__ = ("_connection_name", "_frame_name", "_upper", "_lower")

    def __init__(
        self,
        connection_name: str,
        frame_name: str,
        upper: str,
        lower: str,
    ) -> None:
        self._connection_name = connection_name
        self._frame_name = frame_name
        self._upper = upper
        self._lower = lower

    @property
    def connection_name(self) -> str:
        return self._connection_name

    @property
    def frame_name(self) -> str:
        return self._frame_name

    @property
    def upper(self) -> str:
        return self._upper

    @property
    def lower(self) -> str:
        return self._lower

    @property
    def degree(self) -> Degree:
        return Degree.const(1)

    def _key(self) -> Any:
        return (
            self._connection_name,
            self._frame_name,
            self._upper,
            self._lower,
        )

    def _repr_inner(self) -> str:
        return f"ω^{self._upper}_{self._lower}"

    @property
    def index_names(self) -> Tuple[str, ...]:
        return (self._upper, self._lower)

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        from jacopy.central.objects.frame import substituted_index_names

        if self == dummy:
            return target
        renamed = substituted_index_names(
            self.index_names, dummy, target
        )
        if renamed is None:
            return self
        return ConnectionForm(
            self._connection_name, self._frame_name, *renamed
        )


def connection_form(
    conn: Connection, fr: Frame, upper: IndexLike, lower: IndexLike
) -> ConnectionForm:
    """``ω^a_b`` of the connection on the frame."""
    if not isinstance(conn, Connection):
        raise TypeError("connection_form expects a Connection")
    if not isinstance(fr, Frame):
        raise TypeError("connection_form expects a Frame")
    return ConnectionForm(
        conn.name, fr.name, _index_str(upper), _index_str(lower)
    )


# --------------------------------------------------------------------- #
# Definitional rules                                                     #
# --------------------------------------------------------------------- #


def _is_frame_covariant(node: Expr, alpha: CoframeField):
    """``(conn_name, b, c)`` for ``∇_{e_b} e_c`` matching the coframe's
    frame, else ``None``."""
    if not (isinstance(node, Act) and isinstance(node.op, CovariantOp)):
        return None
    direction = node.op.vector
    arg = node.arg
    if not (
        isinstance(direction, FrameField) and isinstance(arg, FrameField)
    ):
        return None
    if not all(
        leg.base_name == alpha.base_name and leg.bundle == alpha.bundle
        for leg in (direction, arg)
    ):
        return None
    return node.op.connection_name, direction.index, arg.index


class FrameConnectionCoefficientDefinition(Definition):
    """``⟨e^a, ∇_{e_b} e_c⟩ → Γ^a_{bc}`` — the definition of the
    connection coefficient (the 2.E extraction pattern)."""

    name = "connection coefficient: ⟨e^a, ∇_{e_b} e_c⟩ = Γ^a_bc"
    anchor = Pairing

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        if not isinstance(expr.alpha, CoframeField):
            return False
        return _is_frame_covariant(expr.X, expr.alpha) is not None

    def rewrite(self, expr: Expr) -> Expr:
        alpha = expr.alpha
        conn_name, b, c = _is_frame_covariant(expr.X, alpha)
        return ConnectionCoefficient(
            conn_name, alpha.base_name, alpha.index, b, c
        )


class ConnectionFormEvaluationDefinition(Definition):
    """``⟨ω^a_b, X⟩ → ⟨e^a, ∇_X e_b⟩`` — the defining evaluation of the
    connection 1-form (constructed per connection/frame pair so the
    rewrite can rebuild ``∇`` and the coframe)."""

    anchor = Pairing

    def __init__(self, conn: Connection, fr: Frame) -> None:
        if not isinstance(conn, Connection):
            raise TypeError(
                "ConnectionFormEvaluationDefinition expects a Connection"
            )
        if not isinstance(fr, Frame):
            raise TypeError(
                "ConnectionFormEvaluationDefinition expects a Frame"
            )
        self._conn = conn
        self._frame = fr
        self.name = (
            f"connection form ({conn.name}, {fr.name}): "
            "⟨ω^a_b, X⟩ = ⟨e^a, ∇_X e_b⟩"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Pairing)
            and isinstance(expr.alpha, ConnectionForm)
            and expr.alpha.connection_name == self._conn.name
            and expr.alpha.frame_name == self._frame.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        omega = expr.alpha
        return Pairing(
            self._frame.dual().field(omega.upper),
            self._conn(expr.X, self._frame.field(omega.lower)),
        )


#: Shapes a coframe pairing may be split over (extraction-eligible).
def _scalar_core_split(term: Expr, registry=None):
    """``(sign, scalars, core)`` — strip a sign and the certainly-
    scalar factors; ``core`` is the single non-scalar factor (or the
    whole term)."""
    from jacopy.central.calculus.scalars import is_scalar_function

    sign = False
    core = term
    if isinstance(core, Neg):
        sign = True
        core = core.arg
    if isinstance(core, Product) and len(core.children) >= 2:
        scalars = [
            c for c in core.children if is_scalar_function(c, registry)
        ]
        rest = [
            c
            for c in core.children
            if not is_scalar_function(c, registry)
        ]
        if len(rest) == 1:
            return sign, scalars, rest[0]
        return sign, [], core
    return sign, [], core


def _eligible(term: Expr, alpha: CoframeField) -> bool:
    from jacopy.core.indexed_sum import IndexedSum

    _, _, core = _scalar_core_split(term)
    if isinstance(core, IndexedSum):
        return True
    if isinstance(core, FrameField):
        return core.base_name == alpha.base_name
    if isinstance(core, LieBracketVF):
        return isinstance(core.X, FrameField) and isinstance(
            core.Y, FrameField
        )
    return _is_frame_covariant(core, alpha) is not None


class FramePairingScalarDefinition(Definition):
    """``⟨e^a, f·V⟩ → f·⟨e^a, V⟩`` — ``C^∞``-linearity of the pairing
    in the vector slot, GUARDED to extraction-eligible cores (so it
    cannot ping-pong with the pairing-collection phase: after pulling,
    the pairing extracts to an atom and the collector has nothing to
    regroup)."""

    name = "frame pairing scalar: ⟨e^a, f·V⟩ = f·⟨e^a, V⟩ (eligible core)"
    anchor = Pairing

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        if not isinstance(expr.alpha, CoframeField):
            return False
        sign, scalars, core = _scalar_core_split(expr.X, self._registry)
        return bool(scalars or sign) and _eligible(core, expr.alpha)

    def rewrite(self, expr: Expr) -> Expr:
        sign, scalars, core = _scalar_core_split(expr.X, self._registry)
        out: Expr = Pairing(expr.alpha, core)
        if scalars:
            out = Product(*scalars, out)
        return Neg(out) if sign else out


class FramePairingSplitDefinition(Definition):
    """``⟨e^a, Σ tᵢ⟩ → Σ ⟨e^a, tᵢ⟩`` — GUARDED: only when a summand is
    extraction-eligible (frame field, frame bracket, or frame
    covariant derivative). The guard is what keeps this from
    ping-ponging with the pairing-collection phase of ``simplify``:
    once the eligible summands are extracted, the leftovers regroup
    and the split stays quiet."""

    name = "frame pairing split: ⟨e^a, Σtᵢ⟩ = Σ⟨e^a, tᵢ⟩ (eligible summand)"
    anchor = Pairing

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        if not isinstance(expr.alpha, CoframeField):
            return False
        if not isinstance(expr.X, Sum):
            return False
        return any(_eligible(t, expr.alpha) for t in expr.X.children)

    def rewrite(self, expr: Expr) -> Expr:
        return Sum(
            *(Pairing(expr.alpha, t) for t in expr.X.children)
        )


# --------------------------------------------------------------------- #
# Theorems                                                               #
# --------------------------------------------------------------------- #


def _engine(registry: Optional[PropertyRegistry], conn=None, fr=None):
    from jacopy.packages.metric_affine.engine import metric_affine_engine

    engine = metric_affine_engine(registry=registry)
    if conn is not None and fr is not None:
        engine.register(ConnectionFormEvaluationDefinition(conn, fr))
    return engine


def prove_torsion_components(
    conn: Connection,
    fr: Frame,
    upper: IndexLike,
    b: IndexLike,
    c: IndexLike,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨e^a, T(e_b, e_c)⟩ = Γ^a_{bc} − Γ^a_{cb} − γ^a_{bc}`` — the
    torsion components on the frame."""
    e_up = fr.dual().field(upper)
    lhs = Pairing(e_up, torsion(conn, fr.field(b), fr.field(c)))
    rhs = Sum(
        connection_coefficient(conn, fr, upper, b, c),
        Neg(connection_coefficient(conn, fr, upper, c, b)),
        Neg(anholonomy_coefficient(fr, upper, b, c)),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_connection_form_on_frame(
    conn: Connection,
    fr: Frame,
    upper: IndexLike,
    lower: IndexLike,
    at: IndexLike,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``ω^a_b(e_c) = Γ^a_{cb}`` — the connection form evaluated on
    the frame yields the coefficients."""
    lhs = Pairing(connection_form(conn, fr, upper, lower), fr.field(at))
    rhs = connection_coefficient(conn, fr, upper, at, lower)
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry, conn, fr)
    )


def prove_cartan_first_structure(
    conn: Connection,
    fr: Frame,
    upper: IndexLike,
    b: IndexLike,
    c: IndexLike,
    *,
    bound: str = "s",
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """The first Cartan structure equation with the GENUINE indexed
    wedge sum, evaluated on frame arguments:

    ``⟨e^a, T(e_b,e_c)⟩ = de^a(e_b,e_c) + (Σ_s ω^a_s ∧ e^s)(e_b,e_c)``

    The proof drives the whole 4.D.2 layer: evaluation pushes into the
    indexed sum, the wedge expands by the determinant convention, the
    coframe pairings become Kronecker deltas, and the bound index
    contracts onto the connection coefficients — meeting the torsion
    components from the left-hand side. ``bound`` names the bound
    index (must not collide with ``a, b, c``).
    """
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.core.wedge import Wedge
    from jacopy.central.objects.frame import FrameIndex

    for label in (upper, b, c):
        if _index_str(label) == bound:
            raise ValueError(
                f"bound index {bound!r} collides with a free index; "
                "pass a different `bound` name"
            )
    e_up = fr.dual().field(upper)
    eb, ec = fr.field(b), fr.field(c)
    dummy = FrameIndex(bound)
    wedge_sum = IndexedSum(
        dummy,
        fr,
        Wedge(
            connection_form(conn, fr, upper, bound),
            fr.dual().field(bound),
        ),
    )
    lhs = Pairing(e_up, torsion(conn, eb, ec))
    rhs = Sum(
        MultiEval(d(e_up), eb, ec, alternating=True, slot_kind="vector"),
        MultiEval(wedge_sum, eb, ec, alternating=True, slot_kind="vector"),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry, conn, fr)
    )


def _decomposed_engine(conn: Connection, fr: Frame, registry):
    from jacopy.packages.metric_affine.engine import metric_affine_engine

    engine = metric_affine_engine(
        registry=registry, decompositions=((conn, fr),)
    )
    engine.register(ConnectionFormEvaluationDefinition(conn, fr))
    return engine


def curvature_component_formula(
    conn: Connection,
    fr: Frame,
    a: IndexLike,
    b: IndexLike,
    c: IndexLike,
    d: IndexLike,
    *,
    bound: str = "s",
) -> Expr:
    """The classical component formula

    ``R^a_{bcd} = e_c(Γ^a_db) − e_d(Γ^a_cb)
                + Σ_s Γ^a_cs Γ^s_db − Σ_s Γ^a_ds Γ^s_cb
                − Σ_s Γ^a_sb γ^s_cd``

    (anholonomic frame; the γ term is the frame correction)."""
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.central.objects.frame import FrameIndex

    G = lambda u, m, n: ConnectionCoefficient(
        conn.name, fr.name, _index_str(u), _index_str(m), _index_str(n)
    )
    s = bound
    dummy = FrameIndex(s)
    return Sum(
        Act(fr.field(c), G(a, d, b)),
        Neg(Act(fr.field(d), G(a, c, b))),
        IndexedSum(dummy, fr, Product(G(a, c, s), G(s, d, b))),
        Neg(IndexedSum(dummy, fr, Product(G(a, d, s), G(s, c, b)))),
        Neg(
            IndexedSum(
                dummy,
                fr,
                Product(
                    G(a, s, b),
                    anholonomy_coefficient(fr, s, c, d),
                ),
            )
        ),
    )


def prove_curvature_components(
    conn: Connection,
    fr: Frame,
    a: IndexLike,
    b: IndexLike,
    c: IndexLike,
    d: IndexLike,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨e^a, R(e_c, e_d) e_b⟩ =`` the classical component formula —
    requires the frame-decomposition axioms (the ΓΓ terms come from
    decomposing the inner covariant derivative)."""
    lhs = Pairing(
        fr.dual().field(a),
        curvature(conn, fr.field(c), fr.field(d), fr.field(b)),
    )
    return ExpandAndSimplify().prove(
        lhs,
        curvature_component_formula(conn, fr, a, b, c, d),
        registry=registry,
        engine=_decomposed_engine(conn, fr, registry),
    )


def prove_cartan_second_structure(
    conn: Connection,
    fr: Frame,
    a: IndexLike,
    b: IndexLike,
    c: IndexLike,
    d: IndexLike,
    *,
    bound: str = "s",
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """The second Cartan structure equation, frame-evaluated with the
    genuine indexed wedge sum:

    ``⟨e^a, R(e_c,e_d)e_b⟩ = (dω^a_b + Σ_s ω^a_s ∧ ω^s_b)(e_c, e_d)``

    Needs the frame-decomposition axioms (the ω∧ω evaluation and the
    curvature expansion meet in the ΓΓ component form); nested sums
    α-convert on capture automatically.
    """
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.core.wedge import Wedge
    from jacopy.central.objects.frame import FrameIndex
    from jacopy.central.tangent.exterior import d as exterior_d

    for label in (a, b, c, d):
        if _index_str(label) == bound:
            raise ValueError(
                f"bound index {bound!r} collides with a free index; "
                "pass a different `bound` name"
            )
    lhs = Pairing(
        fr.dual().field(a),
        curvature(conn, fr.field(c), fr.field(d), fr.field(b)),
    )
    dummy = FrameIndex(bound)
    rhs = Sum(
        MultiEval(
            exterior_d(connection_form(conn, fr, a, b)),
            fr.field(c),
            fr.field(d),
            alternating=True,
            slot_kind="vector",
        ),
        MultiEval(
            IndexedSum(
                dummy,
                fr,
                Wedge(
                    connection_form(conn, fr, a, bound),
                    connection_form(conn, fr, bound, b),
                ),
            ),
            fr.field(c),
            fr.field(d),
            alternating=True,
            slot_kind="vector",
        ),
    )
    return ExpandAndSimplify().prove(
        lhs,
        rhs,
        registry=registry,
        engine=_decomposed_engine(conn, fr, registry),
    )


def ricci_component(
    conn: Connection,
    fr: Frame,
    b: IndexLike,
    d: IndexLike,
    *,
    bound: str = "s",
) -> Expr:
    """``Ric_{bd} := Σ_s ⟨e^s, R(e_s, e_b) e_d⟩`` — the trace of the
    curvature over its first slot (the canonical definition)."""
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.central.objects.frame import FrameIndex

    for label in (b, d):
        if _index_str(label) == bound:
            raise ValueError(
                f"bound index {bound!r} collides with a free index"
            )
    dummy = FrameIndex(bound)
    return IndexedSum(
        dummy,
        fr,
        Pairing(
            fr.dual().field(bound),
            curvature(
                conn, fr.field(bound), fr.field(b), fr.field(d)
            ),
        ),
    )


def prove_ricci_components(
    conn: Connection,
    fr: Frame,
    b: IndexLike,
    d: IndexLike,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``Ric_{bd} = Σ_s [ e_s(Γ^s_bd) − e_b(Γ^s_sd) + Σ_t Γ^s_st Γ^t_bd
    − Σ_t Γ^s_bt Γ^t_sd − Σ_t Γ^s_td γ^t_sb ]`` — the trace of the
    component formula (nested sums; shadowing and capture handled by
    the α-machinery)."""
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.central.objects.frame import FrameIndex

    s = "s"
    dummy = FrameIndex(s)
    body = curvature_component_formula(
        conn, fr, s, d, s, b, bound="t"
    )
    rhs = IndexedSum(dummy, fr, body)
    return ExpandAndSimplify().prove(
        ricci_component(conn, fr, b, d),
        rhs,
        registry=registry,
        engine=_decomposed_engine(conn, fr, registry),
    )


def ricci_scalar(
    conn: Connection,
    fr: Frame,
    g,
    *,
    bounds: Tuple[str, str, str] = ("s", "t", "u"),
) -> Expr:
    """``R := Σ_s Σ_t g^{st} · Ric_{st}`` — the Ricci scalar as its
    canonical contraction (``bounds``: the three bound names — outer
    two for the metric contraction, inner one for the Ricci trace)."""
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.central.objects.frame import FrameIndex
    from jacopy.packages.metric_affine.metric import (
        Metric,
        inverse_metric_component,
    )

    from jacopy.packages.metric_affine.metric import as_metric_context

    g = as_metric_context(g)
    s, t, u = bounds
    if len({s, t, u}) != 3:
        raise ValueError("the three bound names must be distinct")
    return IndexedSum(
        FrameIndex(s),
        fr,
        IndexedSum(
            FrameIndex(t),
            fr,
            Product(
                inverse_metric_component(g, s, t),
                ricci_component(conn, fr, s, t, bound=u),
            ),
        ),
    )


def einstein_component(
    conn: Connection,
    fr: Frame,
    g,
    b: IndexLike,
    d: IndexLike,
) -> Expr:
    """``G_{bd} := Ric_{bd} − ½·R·g(e_b, e_d)`` — the Einstein tensor
    components (canonical definition; trace identities like
    ``G^a_a = (1 − n/2)·R`` need the symbolic dimension ``n`` and are
    deliberately out of scope here)."""
    from jacopy.core.expr import Rational
    from jacopy.packages.metric_affine.metric import Metric

    from jacopy.packages.metric_affine.metric import as_metric_context

    g = as_metric_context(g)
    return Sum(
        ricci_component(conn, fr, b, d),
        Neg(
            Product(
                Rational(1, 2),
                ricci_scalar(conn, fr, g),
                g(fr.field(b), fr.field(d)),
            )
        ),
    )


def prove_ricci_scalar_components(
    conn: Connection,
    fr: Frame,
    g,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """The Ricci scalar unfolds to the ``Γ``-component form:
    ``R = Σ_s Σ_t g^{st} · Σ_u [component-formula trace]`` — the
    triple-sum computation, mechanical end to end."""
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.central.objects.frame import FrameIndex
    from jacopy.packages.metric_affine.metric import (
        inverse_metric_component,
    )

    rhs = IndexedSum(
        FrameIndex("s"),
        fr,
        IndexedSum(
            FrameIndex("t"),
            fr,
            Product(
                inverse_metric_component(g, "s", "t"),
                IndexedSum(
                    FrameIndex("u"),
                    fr,
                    curvature_component_formula(
                        conn, fr, "u", "t", "u", "s", bound="v"
                    ),
                ),
            ),
        ),
    )
    return ExpandAndSimplify().prove(
        ricci_scalar(conn, fr, g),
        rhs,
        registry=registry,
        engine=_decomposed_engine(conn, fr, registry),
    )


def prove_cartan_first_evaluated(
    conn: Connection,
    fr: Frame,
    upper: IndexLike,
    b: IndexLike,
    c: IndexLike,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """The first Cartan structure equation, frame-evaluated:

    ``⟨e^a, T(e_b,e_c)⟩ = de^a(e_b,e_c) + ω^a_c(e_b) − ω^a_b(e_c)``

    (the ``Σ_d ω^a_d ∧ e^d`` wedge term evaluated against the frame
    collapses to the two ω-values; the form-valued equation itself is
    Phase 4.D.2). Everything mechanical: torsion expansion, Palais on
    ``de^a``, the coefficient extractions, γ-cancellation.
    """
    e_up = fr.dual().field(upper)
    eb, ec = fr.field(b), fr.field(c)
    lhs = Pairing(e_up, torsion(conn, eb, ec))
    rhs = Sum(
        MultiEval(d(e_up), eb, ec, alternating=True, slot_kind="vector"),
        Pairing(connection_form(conn, fr, upper, c), eb),
        Neg(Pairing(connection_form(conn, fr, upper, b), ec)),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry, conn, fr)
    )
