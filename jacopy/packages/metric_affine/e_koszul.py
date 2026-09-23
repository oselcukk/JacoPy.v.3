"""
The E-Koszul formula family (Phase 4.F.3) [MC Prop 3.7-3.8,
ADM Prop 3.12 one direction].

The algebroid analogue of the Phase 4.C Koszul module, with the
locality correction ``K`` [MC eq (3.36)]:

    K(∇,g,L)(u,v,w) := − g(Σ_s L(e^s,∇_{e_s}v,w), u)
                       − g(Σ_s L(e^s,∇_{e_s}u,w), v)
                       + g(Σ_s L(e^s,∇_{e_s}u,v), w).

* :func:`prove_e_generalized_koszul` — the **E-generalized Koszul
  identity**, valid for an ARBITRARY E-connection and E-metric on a
  local algebroid, purely definitionally (no declarations):

      2·g(∇_u v, w) + K(u,v,w)
        = ρ(u)g(v,w) + ρ(v)g(u,w) − ρ(w)g(u,v)
        + g([u,v],w) − g([u,w],v) − g([v,w],u)
        − Q(u,v,w) − Q(v,u,w) + Q(w,u,v)
        + g(ᴸT(u,v),w) − g(ᴸT(u,w),v) − g(ᴸT(v,w),u),

  with ``Q`` the E-nonmetricity and ``ᴸT`` the modified torsion —
  the defect blocks are exactly the price of an arbitrary
  connection.
* :func:`prove_e_koszul_formula` — the clean modified Koszul formula
  under the declared ``ᴸT = 0`` and ``Q = 0``: the defining property
  of the E-Koszul connection [MC Def 3.14]. Together with the 4.F.1
  theorem ``ᴸT-free ⟹ admissible``, this closes one direction of
  [ADM Prop 3.12]: an E-Levi-Civita connection is admissible AND
  E-Koszul. The converse (E-Koszul + admissible ⟹ E-LC, via
  non-degeneracy stripping) is 4.F.3b.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.objects.connection import Connection
from jacopy.central.objects.frame import Frame
from jacopy.central.algebroid.context import Algebroid
from jacopy.packages.metric_affine.admissible import (
    MetricCompatibilityCollection,
    ModifiedTorsionFreeDeclaration,
    _locality_sum,
)
from jacopy.packages.metric_affine.modified import (
    _engine,
    modified_torsion,
)


def e_nonmetricity(
    alg: Algebroid, conn: Connection, u: Expr, v: Expr, w: Expr
) -> Expr:
    """``Q(∇,g)(u,v,w) := ρ(u)(g(v,w)) − g(∇_u v, w) − g(v, ∇_u w)``
    — the E-nonmetricity, as an explicit expression."""
    return Sum(
        Act(alg.anchor(u), alg.metric(v, w)),
        Neg(alg.metric(conn(u, v), w)),
        Neg(alg.metric(v, conn(u, w))),
    )


def koszul_correction(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    w: Expr,
) -> Expr:
    """``K(∇,g,L)(u,v,w)`` [MC eq (3.36)]."""
    return Sum(
        Neg(alg.metric(_locality_sum(alg, conn, fr, v, w, "s"), u)),
        Neg(alg.metric(_locality_sum(alg, conn, fr, u, w, "s"), v)),
        alg.metric(_locality_sum(alg, conn, fr, u, v, "s"), w),
    )


def _koszul_lhs(alg, conn, fr, u, v, w) -> Expr:
    return Sum(
        Product(Integer(2), alg.metric(conn(u, v), w)),
        koszul_correction(alg, conn, fr, u, v, w),
    )


def _koszul_rhs_clean(alg, u, v, w) -> Expr:
    """The six declaration-free Koszul terms."""
    g = alg.metric
    rho = alg.anchor
    br = alg.bracket
    return Sum(
        Act(rho(u), g(v, w)),
        Act(rho(v), g(u, w)),
        Neg(Act(rho(w), g(u, v))),
        g(br(u, v), w),
        Neg(g(br(u, w), v)),
        Neg(g(br(v, w), u)),
    )


class EKoszulDeclaration(Definition):
    """``∇`` DECLARED E-Koszul [MC Def 3.14]: every metric-wrapped
    covariant derivative is solved through the modified Koszul
    formula,

        g(∇_x a, b) → ½·(R(x,a,b) − K(x,a,b)),

    with ``R`` the six-term Koszul right-hand side. Terminates: ``R``
    holds no ``∇``-in-metric-slot at all, and ``K``'s ``∇``'s sit
    inside ``L``-slots (inert to this rule)."""

    def __init__(
        self,
        alg: Algebroid,
        conn: Connection,
        fr: Frame,
    ) -> None:
        from jacopy.central.algebroid.context import EMetric

        self.anchor = EMetric
        self._alg = alg
        self.owner = alg
        self._conn = conn
        self._fr = fr
        self.name = (
            f"declared E-Koszul ({alg.name}, {conn.name}): "
            "g(∇_x a, b) = ½(R(x,a,b) − K(x,a,b))"
        )

    def _split(self, expr: Expr):
        from jacopy.central.algebroid.context import EMetric
        from jacopy.central.objects.connection import CovariantOp

        if not (
            isinstance(expr, EMetric)
            and expr.algebroid_name == self._alg.name
        ):
            return None
        for slot, other in ((expr.u, expr.v), (expr.v, expr.u)):
            if (
                isinstance(slot, Act)
                and isinstance(slot.op, CovariantOp)
                and slot.op.connection_name == self._conn.name
            ):
                return slot.op.vector, slot.arg, other
        return None

    def matches(self, expr: Expr) -> bool:
        return self._split(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.expr import Rational

        x, a, b = self._split(expr)
        return Product(
            Rational(1, 2),
            Sum(
                _koszul_rhs_clean(self._alg, x, a, b),
                Neg(
                    koszul_correction(
                        self._alg, self._conn, self._fr, x, a, b
                    )
                ),
            ),
        )


class AdmissibilityMetricPairCollection(Definition):
    """Declared admissibility, collected THROUGH a metric wrapper:
    a same-signed transposed bracket pair inside E-metric slots,

        ±c·g([a,b], x) + ±c·g([b,a], x)
            → ±c·(g(Σ_s L(e^s,∇_{e_s}a,b), x)
                 + g(Σ_s L(e^s,∇_{e_s}b,a), x)),

    the metric-level companion of the section-level
    :class:`~jacopy.packages.metric_affine.admissible.
    AdmissibilityDeclaration` (which cannot see brackets inside
    metric slots). Collection-only direction, same cycle
    rationale."""

    anchor = Sum

    def __init__(
        self,
        alg: Algebroid,
        conn: Connection,
        fr: Frame,
        *,
        bound: str = "s",
    ) -> None:
        self._alg = alg
        self.owner = alg
        self._conn = conn
        self._fr = fr
        self._bound = bound
        self.name = (
            f"admissible connection, metric-wrapped ({alg.name}, "
            f"{conn.name}): g([a,b],x) + g([b,a],x) = g(ΣL,x) + …"
        )

    def _bracket_metric(self, term: Expr):
        """``(sign, cofactors, a, b, x)`` for ``±c·g([a,b], x)``."""
        from jacopy.central.algebroid.context import (
            AlgebroidBracket,
            EMetric,
        )

        sign = 1
        core = term
        if isinstance(core, Neg):
            sign = -1
            core = core.arg
        factors = (
            list(core.children)
            if isinstance(core, Product)
            else [core]
        )
        metrics = [
            (k, c)
            for k, c in enumerate(factors)
            if isinstance(c, EMetric)
            and c.algebroid_name == self._alg.name
        ]
        if len(metrics) != 1:
            return None
        k, m = metrics[0]
        cofactors = tuple(
            c for j, c in enumerate(factors) if j != k
        )
        for slot, other in ((m.u, m.v), (m.v, m.u)):
            if (
                isinstance(slot, AlgebroidBracket)
                and slot.algebroid_name == self._alg.name
            ):
                return sign, cofactors, slot.u, slot.v, other
        return None

    @staticmethod
    def _cofactor_key(cofactors):
        return tuple(sorted(c._repr_inner() for c in cofactors))

    def _find_pair(self, expr: Sum):
        terms = expr.children
        for i, ti in enumerate(terms):
            si = self._bracket_metric(ti)
            if si is None:
                continue
            for j in range(i + 1, len(terms)):
                sj = self._bracket_metric(terms[j])
                if sj is None:
                    continue
                if (
                    si[0] == sj[0]
                    and self._cofactor_key(si[1])
                    == self._cofactor_key(sj[1])
                    and si[2] == sj[3]
                    and si[3] == sj[2]
                    and si[4] == sj[4]
                ):
                    return (i, j), si
        return None

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Sum)
            and self._find_pair(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        (i, j), (sign, cofactors, a, b, x) = self._find_pair(expr)
        if a._repr_inner() > b._repr_inner():
            a, b = b, a
        g = self._alg.metric
        collected: Expr = Sum(
            g(
                _locality_sum(
                    self._alg, self._conn, self._fr, a, b, self._bound
                ),
                x,
            ),
            g(
                _locality_sum(
                    self._alg, self._conn, self._fr, b, a, self._bound
                ),
                x,
            ),
        )
        if cofactors:
            collected = Product(*cofactors, collected)
        if sign < 0:
            collected = Neg(collected)
        rest = [
            t for k, t in enumerate(expr.children) if k not in (i, j)
        ]
        if not rest:
            return collected
        return Sum(*rest, collected)


def prove_e_generalized_koszul(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """The E-generalized Koszul identity (see module docstring) —
    ARBITRARY connection, no declarations; the Q- and ᴸT-blocks
    carry the whole defect."""
    g = alg.metric
    lhs = _koszul_lhs(alg, conn, fr, u, v, w)
    rhs = Sum(
        _koszul_rhs_clean(alg, u, v, w),
        Neg(e_nonmetricity(alg, conn, u, v, w)),
        Neg(e_nonmetricity(alg, conn, v, u, w)),
        e_nonmetricity(alg, conn, w, u, v),
        g(modified_torsion(alg, conn, fr, u, v), w),
        Neg(g(modified_torsion(alg, conn, fr, u, w), v)),
        Neg(g(modified_torsion(alg, conn, fr, v, w), u)),
    )
    return ExpandAndSimplify().prove(
        lhs,
        rhs,
        registry=registry,
        engine=_engine(alg, fr, registry),
    )


def prove_e_koszul_formula(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """The clean modified Koszul formula [MC Prop 3.7-3.8 /
    Def 3.14] under the DECLARED ``ᴸT = 0`` + ``Q = 0``:

        2·g(∇_u v, w) + K(u,v,w)
          = ρ(u)g(v,w) + ρ(v)g(u,w) − ρ(w)g(u,v)
          + g([u,v],w) − g([u,w],v) − g([v,w],u).

    With 4.F.1's ``ᴸT-free ⟹ admissible`` this closes one direction
    of [ADM Prop 3.12]: E-Levi-Civita ⟹ admissible + E-Koszul.
    Honest fail without the declarations (tested)."""
    engine = _engine(alg, fr, registry)
    engine.register(ModifiedTorsionFreeDeclaration(alg, conn, fr))
    engine.register(MetricCompatibilityCollection(alg, conn))
    return ExpandAndSimplify().prove(
        _koszul_lhs(alg, conn, fr, u, v, w),
        _koszul_rhs_clean(alg, u, v, w),
        registry=registry,
        engine=engine,
    )


def prove_e_koszul_admissible_implies_compatible(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[ADM Prop 3.12, ⟸ half 1]: a connection that is E-Koszul
    (declared) and admissible (declared, metric-wrapped) is
    metric-compatible — ``Q(∇,g)(u,v,w) = 0``. Fully mechanical, NO
    non-degeneracy: the ``Koszul(u,v,w) + Koszul(u,w,v)``
    combination cancels the K-blocks against the admissibility
    collection."""
    engine = _engine(alg, fr, registry)
    engine.register(EKoszulDeclaration(alg, conn, fr))
    engine.register(
        AdmissibilityMetricPairCollection(alg, conn, fr)
    )
    return ExpandAndSimplify().prove(
        e_nonmetricity(alg, conn, u, v, w),
        Integer(0),
        registry=registry,
        engine=engine,
    )


def prove_e_koszul_admissible_implies_torsion_free(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[ADM Prop 3.12, ⟸ half 2]: an E-Koszul admissible connection
    is modified-torsion-free. Mechanical part:
    ``g(ᴸT(u,v), w) = 0`` for the generic section ``w`` (the
    antisymmetrized Koszul combination + the metric-wrapped
    admissibility collection); the conclusion ``ᴸT(u,v) = 0`` is the
    explicitly-labeled non-degeneracy step ([B 4.11] pattern)."""
    engine = _engine(alg, fr, registry)
    engine.register(EKoszulDeclaration(alg, conn, fr))
    engine.register(
        AdmissibilityMetricPairCollection(alg, conn, fr)
    )
    paired = alg.metric(modified_torsion(alg, conn, fr, u, v), w)
    leg = ExpandAndSimplify().prove(
        paired, Integer(0), registry=registry, engine=engine
    )
    step1 = ProofStep(
        paired,
        Integer(0),
        rule="antisymmetrized Koszul + admissibility",
        justification="g(ᴸT(u,v), w) = 0 mechanically",
    )
    for s in leg:
        step1.add_child(s)
    step2 = ProofStep(
        modified_torsion(alg, conn, fr, u, v),
        Integer(0),
        rule="non-degeneracy of g (agreement on generic section w)",
        justification=(
            "g(ᴸT(u,v), w) = 0 for generic w forces ᴸT(u,v) = 0"
        ),
    )
    return ProofChain([step1, step2])


def prove_metric_lie_derivative_identity(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[ADM Prop 3.14]: for an E-Levi-Civita connection (declared
    ``ᴸT = 0`` + ``Q = 0``), the modified-bracket Lie derivative of
    the E-metric is the symmetrized covariant derivative:

        (ℒ^∇_v g)(u,w) := ρ(v)(g(u,w)) − g([v,u]^∇, w)
                          − g(u, [v,w]^∇)
                        = g(∇_u v, w) + g(u, ∇_w v)."""
    from jacopy.packages.metric_affine.admissible import (
        modified_bracket,
    )

    g = alg.metric
    lhs = Sum(
        Act(alg.anchor(v), g(u, w)),
        Neg(g(modified_bracket(alg, conn, fr, v, u), w)),
        Neg(g(u, modified_bracket(alg, conn, fr, v, w))),
    )
    rhs = Sum(g(conn(u, v), w), g(u, conn(w, v)))
    engine = _engine(alg, fr, registry)
    engine.register(ModifiedTorsionFreeDeclaration(alg, conn, fr))
    engine.register(MetricCompatibilityCollection(alg, conn))
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=engine
    )


def prove_koszul_reference_decomposition(
    alg: Algebroid,
    conn: Connection,
    ref: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[ADM Prop 3.13, identity form]: the deviation of an ARBITRARY
    connection ``∇`` from an E-Koszul reference ``∇̄`` (declared) is
    pure defect data:

        2g(∇_u v, w) − 2g(∇̄_u v, w) + K(∇)(u,v,w) − K(∇̄)(u,v,w)
          = − Q(u,v,w) − Q(v,u,w) + Q(w,u,v)
            + g(ᴸT(u,v),w) − g(ᴸT(u,w),v) − g(ᴸT(v,w),u),

    with ``Q``/``ᴸT`` the E-nonmetricity and modified torsion of
    ``∇`` — the E-analogue of the Schouten decomposition (obtained by
    subtracting the clean Koszul formula of ``∇̄`` from the
    E-generalized Koszul identity of ``∇``)."""
    g = alg.metric
    lhs = Sum(
        Product(Integer(2), g(conn(u, v), w)),
        Neg(Product(Integer(2), g(ref(u, v), w))),
        koszul_correction(alg, conn, fr, u, v, w),
        Neg(koszul_correction(alg, ref, fr, u, v, w)),
    )
    rhs = Sum(
        Neg(e_nonmetricity(alg, conn, u, v, w)),
        Neg(e_nonmetricity(alg, conn, v, u, w)),
        e_nonmetricity(alg, conn, w, u, v),
        g(modified_torsion(alg, conn, fr, u, v), w),
        Neg(g(modified_torsion(alg, conn, fr, u, w), v)),
        Neg(g(modified_torsion(alg, conn, fr, v, w), u)),
    )
    engine = _engine(alg, fr, registry)
    engine.register(EKoszulDeclaration(alg, ref, fr))
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=engine
    )


class NullLocalityDefinition(Definition):
    """``L(ω, u, v) → 0`` — the DECLARED null locality operator
    (``[L] = [0]``, e.g. any almost-Lie algebroid [MC Prop 3.10]).
    The setting of the generalized fundamental theorem
    [MC Thm 3.2]."""

    def __init__(self, alg: Algebroid) -> None:
        from jacopy.central.algebroid.context import LocalityOperator

        self.anchor = LocalityOperator
        self._alg = alg
        self.owner = alg
        self.name = f"null locality ({alg.name}): L = 0"

    def matches(self, expr: Expr) -> bool:
        from jacopy.central.algebroid.context import LocalityOperator

        return (
            isinstance(expr, LocalityOperator)
            and expr.algebroid_name == self._alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)


def prove_e_koszul_unique_under_null_locality(
    alg: Algebroid,
    conn1: Connection,
    conn2: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[MC Thm 3.2, abstract half]: with a NULL locality operator the
    E-Koszul connection is UNIQUE — two Koszul-declared connections
    satisfy ``g(∇¹_u v − ∇²_u v, w) = 0`` mechanically (the
    K-corrections vanish with ``L = 0``, both sides collapse to the
    same Koszul right-hand side), and non-degeneracy (the explicitly
    labeled stripping step) gives ``∇¹_u v = ∇²_u v``. The component
    decomposition (3.44) is the deferred component-level half."""
    from jacopy.proof.step import ProofStep

    g = alg.metric
    engine = _engine(alg, fr, registry)
    engine.register(NullLocalityDefinition(alg))
    engine.register(EKoszulDeclaration(alg, conn1, fr))
    engine.register(EKoszulDeclaration(alg, conn2, fr))
    paired = Sum(
        g(conn1(u, v), w), Neg(g(conn2(u, v), w))
    )
    leg = ExpandAndSimplify().prove(
        paired, Integer(0), registry=registry, engine=engine
    )
    step1 = ProofStep(
        paired,
        Integer(0),
        rule="both Koszul formulas collapse (L = 0 ⟹ K = 0)",
        justification="g(∇¹_u v − ∇²_u v, w) = 0 mechanically",
    )
    for s in leg:
        step1.add_child(s)
    step2 = ProofStep(
        Sum(conn1(u, v), Neg(conn2(u, v))),
        Integer(0),
        rule="non-degeneracy of g (agreement on generic section w)",
        justification="∇¹_u v = ∇²_u v — the E-Koszul connection is unique",
    )
    return ProofChain([step1, step2])


def prove_projectors_agree_on_coboundary_values(
    alg: Algebroid,
    fr: Frame,
    f: Expr,
    u: Expr,
    v: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[MC Prop 4.2, mechanizable fragment]: ANY two locality
    projectors agree on coboundary locality values —
    ``𝒫₁(L(Df,u,v)) − 𝒫₂(L(Df,u,v)) = 0`` (both absorption rules
    fire). The FULL uniqueness (𝒫 = pr₂∘τ on exact almost-Courant)
    needs the exact-sequence layer — recorded deferral."""
    from jacopy.central.algebroid.context import LocalityOperator
    from jacopy.central.algebroid.locality import locality_projector

    P1 = locality_projector(alg, "P1")
    P2 = locality_projector(alg, "P2")
    L_node = LocalityOperator(alg.name, alg.D(f), u, v)
    lhs = Sum(P1(L_node), Neg(P2(L_node)))
    engine = _engine(alg, fr, registry, projectors=(P1, P2))
    return ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=engine
    )


# ------------------------------------------------------------------- #
# MC Thm 3.2 — the component decomposition (3.44) (ledger K3.a)        #
# ------------------------------------------------------------------- #


def _pseudo_torsion(alg: Algebroid, conn: Connection, u: Expr, v: Expr) -> Expr:
    """``⁰T(∇)(u,v) = ∇_u v − ∇_v u − [u,v]_E`` [MC eq (3.8)] — with a
    null locality operator this IS the modified torsion."""
    return Sum(conn(u, v), Neg(conn(v, u)), Neg(alg.bracket(u, v)))


def fundamental_theorem_components(
    alg: Algebroid,
    conn: Connection,
    kconn: Connection,
    fr: Frame,
    b,
    c,
    d,
):
    """The two sides of the lowered-index (3.44) on the E-frame
    ``(X_a)``: with ``Γ(∇)_{dbc} := g(∇_{X_b} X_c, X_d)``,

        Γ(∇)_{dbc} = Γ(K∇)_{dbc}
                     + ½[ −Q_{bcd} − Q_{cbd} + Q_{dbc}
                          + ⁰T_{dbc} − ⁰T_{cbd} − ⁰T_{bcd} ],

    where ``Q_{abc} = Q(∇,g)(X_a; X_b, X_c)`` and ``⁰T_{abc} =
    g(⁰T(∇)(X_b, X_c), X_a)``. Returns ``(lhs, rhs)``."""
    Xb, Xc, Xd = fr.field(b), fr.field(c), fr.field(d)
    g = alg.metric

    def Q(a, x, y):
        return e_nonmetricity(alg, conn, a, x, y)

    def T(a, x, y):
        return g(_pseudo_torsion(alg, conn, x, y), a)

    lhs = g(conn(Xb, Xc), Xd)
    rhs = Sum(
        g(kconn(Xb, Xc), Xd),
        Product(
            Rational(1, 2),
            Sum(
                Neg(Q(Xb, Xc, Xd)),
                Neg(Q(Xc, Xb, Xd)),
                Q(Xd, Xb, Xc),
                T(Xd, Xb, Xc),
                Neg(T(Xc, Xb, Xd)),
                Neg(T(Xb, Xc, Xd)),
            ),
        ),
    )
    return lhs, rhs


def prove_fundamental_theorem_components(
    alg: Algebroid,
    conn: Connection,
    kconn: Connection,
    fr: Frame,
    b="b",
    c="c",
    d="d",
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[MC Thm 3.2, component half — eq (3.44)] on a local
    almost-Leibniz algebroid with a NULL locality operator: the
    components of an ARBITRARY E-connection ``∇`` decompose as the
    components of the unique E-Koszul connection ``K∇`` (declared
    E-Koszul) plus the non-metricity and pseudo-torsion blocks of
    ``∇`` — the lowered-index form of (3.44), obtained by pairing
    with ``X_d`` instead of contracting with ``g^{ad}``:

        g(∇_{X_b}X_c, X_d) = g(K∇_{X_b}X_c, X_d)
            + ½[−Q_{bcd} − Q_{cbd} + Q_{dbc} + ⁰T_{dbc} − ⁰T_{cbd} − ⁰T_{bcd}].

    Two engine steps: (1) the decomposition itself (the frame instance
    of the Schouten trick — the E-generalized Koszul identity of ``∇``
    minus the clean Koszul formula of ``K∇``, ``K = 0`` because
    ``L = 0``); (2) the explicit E-Koszul components (3.37) at
    ``L = 0``, ``g(K∇_{X_b}X_c, X_d) = ½[ρ(X_b)g_{cd} + ρ(X_c)g_{bd} −
    ρ(X_d)g_{bc} + g([X_b,X_c],X_d) − g([X_b,X_d],X_c) −
    g([X_c,X_d],X_b)]`` (the paper's ``γ`` terms are the lowered
    E-anholonomy ``g([X_b,X_c],X_d)``).

    Honest scope: the raised-index (3.44) differs by the contraction
    with ``g^{ad}``; the inverse-metric contraction rule of
    :mod:`jacopy.packages.metric_affine.metric` lives on the
    ``MetricValue`` node family, not on the algebroid's ``EMetric``,
    so the raising is not mechanized here (E4 adapter item)."""
    lhs, rhs = fundamental_theorem_components(alg, conn, kconn, fr, b, c, d)
    engine = _engine(alg, fr, registry)
    engine.register(NullLocalityDefinition(alg))
    engine.register(EKoszulDeclaration(alg, kconn, fr))
    leg1 = ExpandAndSimplify().prove(lhs, rhs, registry=registry, engine=engine)
    Xb, Xc, Xd = fr.field(b), fr.field(c), fr.field(d)
    k_lhs = alg.metric(kconn(Xb, Xc), Xd)
    k_rhs = Product(Rational(1, 2), _koszul_rhs_clean(alg, Xb, Xc, Xd))
    leg2 = ExpandAndSimplify().prove(k_lhs, k_rhs, registry=registry, engine=engine)
    step1 = ProofStep(
        lhs,
        rhs,
        rule="(3.44), lowered index: Γ(∇)_{dbc} = Γ(K∇)_{dbc} + ½[Q-block + ⁰T-block]",
        justification="frame instance of the Schouten trick (L = 0 ⟹ K = 0); engine normal form",
    )
    for s in leg1:
        step1.add_child(s)
    # the second step rewrites the Koszul components INSIDE the whole
    # right-hand side, so the chain stays connected (step1.after ==
    # step2.before) and its final expression keeps the Q/⁰T corrections
    # (2026-09-23 audit, F4: two disconnected identities read as one)
    rhs_expanded = rhs.substitute_atom(k_lhs, k_rhs) if hasattr(rhs, "substitute_atom") else _replace(rhs, k_lhs, k_rhs)
    if rhs_expanded == rhs:
        rhs_expanded = _replace(rhs, k_lhs, k_rhs)
    step2 = ProofStep(
        rhs,
        rhs_expanded,
        rule="E-Koszul components at L = 0 (3.37) substituted: Γ(K∇)_{dbc} = ½[ρ(X)g + lowered γ terms]",
        justification="declared E-Koszul connection; engine normal form of the component",
    )
    for s in leg2:
        step2.add_child(s)
    return ProofChain([step1, step2])


def _replace(expr: Expr, target: Expr, value: Expr) -> Expr:
    """Structural replacement of ``target`` by ``value`` (children walk)."""
    if expr == target:
        return value
    kids = getattr(expr, "children", ())
    if not kids:
        return expr
    new_kids = tuple(_replace(k, target, value) for k in kids)
    if new_kids == tuple(kids):
        return expr
    return expr._rebuild(new_kids)
