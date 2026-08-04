"""
The E-Bianchi identities (Phase 4.F.4b) [ADM Prop 3.7].

Notation: ``b(u,v) := [u,v]^∇ = [u,v] − Λ(u,v)`` the modified
bracket, ``Λ(u,v) := Σ_s L(e^s, ∇_{e_s}u, v)``, ``b̂`` the projected
modified bracket (``Λ̂ = 𝒫Λ``); ``ᴸT(u,v) = ∇_uv − ∇_vu − b(u,v)``;
``ᴸ̂R(u,v)w = ∇_u∇_vw − ∇_v∇_uw − ∇_{b̂(u,v)}w``; ``𝔖`` the cyclic
sum over ``(u,v,w)``.

**E-Bianchi I, identity form** (ARBITRARY connection, hand-derived
and machine-verified, purely definitional):

    𝔖 ᴸ̂R(u,v)w = 𝔖 [ (∇_u ᴸT)(v,w) + ᴸT(ᴸT(u,v), w) ]
                  − 𝔖 ∇_{(1−𝒫)Λ(u,v)}w
                  − 𝔖 b(b(u,v), w)
                  − 𝔖 [ b(v, ∇_u w) + b(∇_v u, w) ].

The ``(1−𝒫)Λ`` anomaly is the SAME gap as in the generalized Ricci
identity [ADM Prop 3.8]; ``𝔖 b(b(u,v),w)`` is the modified-bracket
Jacobiator block ([ADM]'s ``[[u,v]^∇,w]^∇ + cycl`` extra term); the
last block is the anti-symmetry defect of ``b``.

**Admissible specialization** [ADM Prop 3.7 proper]: for an
ADMISSIBLE connection ``b`` is anti-symmetric (Phase 4.F.1), so the
last block vanishes under the admissibility declaration:

    𝔖 ᴸ̂R(u,v)w = 𝔖 [ (∇_u ᴸT)(v,w) + ᴸT(ᴸT(u,v), w) ]
                  − 𝔖 ∇_{(1−𝒫)Λ(u,v)}w − 𝔖 b(b(u,v), w).

TM sanity: with ``L = 0`` and ``b`` the Lie bracket this reduces to
the Phase 4.C torsionful Bianchi I (whose ``𝔖 [[u,v],w]`` block dies
by the cited Jacobi instances).
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.objects.connection import Connection, CovariantOp
from jacopy.central.objects.frame import Frame
from jacopy.central.algebroid.context import Algebroid
from jacopy.central.algebroid.locality import LocalityProjector
from jacopy.packages.metric_affine.admissible import (
    AdmissibilityDeclaration,
    modified_bracket,
)
from jacopy.packages.metric_affine.modified import (
    _engine,
    generalized_ricci_anomaly,
    modified_torsion,
    projected_curvature,
)


def _cyc(f, u, v, w):
    """``𝔖 f(u,v,w)`` — the cyclic sum."""
    return Sum(f(u, v, w), f(v, w, u), f(w, u, v))


def _cov(conn: Connection, direction: Expr, arg: Expr) -> Expr:
    return Act(
        CovariantOp(conn.name, direction, bundle=conn.bundle), arg
    )


def covariant_modified_torsion_derivative(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    u: Expr,
    v: Expr,
    w: Expr,
) -> Expr:
    """``(∇_u ᴸT)(v, w)`` written out."""
    return Sum(
        conn(u, modified_torsion(alg, conn, fr, v, w)),
        Neg(modified_torsion(alg, conn, fr, conn(u, v), w)),
        Neg(modified_torsion(alg, conn, fr, v, conn(u, w))),
    )


def prove_e_bianchi_first_identity(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    projector: LocalityProjector,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    admissible: bool = False,
) -> ProofChain:
    """E-Bianchi I (see module docstring). With
    ``admissible=False``: the full identity form, NO declarations.
    With ``admissible=True``: the admissibility declaration is
    registered and the anti-symmetry-defect block is dropped from
    the right-hand side [ADM Prop 3.7 proper]."""

    def LhatR(a, b_, c):
        return projected_curvature(
            alg, conn, fr, projector, a, b_, c
        )

    def nabla_LT(a, b_, c):
        return covariant_modified_torsion_derivative(
            alg, conn, fr, a, b_, c
        )

    def LT_of_LT(a, b_, c):
        return modified_torsion(
            alg,
            conn,
            fr,
            modified_torsion(alg, conn, fr, a, b_),
            c,
        )

    def anomaly(a, b_, c):
        return generalized_ricci_anomaly(
            alg, conn, fr, projector, a, b_, c
        )

    def b_of_b(a, b_, c):
        return modified_bracket(
            alg, conn, fr, modified_bracket(alg, conn, fr, a, b_), c
        )

    def defect_block(a, b_, c):
        return Sum(
            modified_bracket(alg, conn, fr, b_, conn(a, c)),
            modified_bracket(alg, conn, fr, conn(b_, a), c),
        )

    lhs = _cyc(LhatR, u, v, w)
    rhs_terms = [
        _cyc(nabla_LT, u, v, w),
        _cyc(LT_of_LT, u, v, w),
        Neg(_cyc(anomaly, u, v, w)),
        Neg(_cyc(b_of_b, u, v, w)),
    ]
    engine = _engine(alg, fr, registry, projectors=(projector,))
    if admissible:
        engine.register(AdmissibilityDeclaration(alg, conn, fr))
    else:
        rhs_terms.append(Neg(_cyc(defect_block, u, v, w)))
    return ExpandAndSimplify().prove(
        lhs,
        Sum(*rhs_terms),
        registry=registry,
        engine=engine,
    )


# --------------------------------------------------------------------- #
# E-Bianchi II (Phase 4.F.4c)                                            #
# --------------------------------------------------------------------- #


def _lam(alg, conn, fr, a, b, *, bound="s"):
    """``Λ(a,b) = Σ_s L(e^s, ∇_{e_s}a, b)`` as a section."""
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.central.objects.frame import FrameIndex
    from jacopy.central.algebroid.context import LocalityOperator

    return IndexedSum(
        FrameIndex(bound),
        fr,
        LocalityOperator(
            alg.name,
            fr.dual().field(bound),
            conn(fr.field(bound), a),
            b,
        ),
    )


def _lam_hat(alg, conn, fr, projector, a, b, *, bound="s"):
    """``Λ̂(a,b) = Σ_s 𝒫L(e^s, ∇_{e_s}a, b)``."""
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.central.objects.frame import FrameIndex
    from jacopy.central.algebroid.context import LocalityOperator

    return IndexedSum(
        FrameIndex(bound),
        fr,
        projector(
            LocalityOperator(
                alg.name,
                fr.dual().field(bound),
                conn(fr.field(bound), a),
                b,
            )
        ),
    )


def anomaly_direction(alg, conn, fr, projector, a, b) -> Expr:
    """``(1−𝒫)Λ(a,b)`` as a section — the recurring anomaly."""
    return Sum(
        _lam(alg, conn, fr, a, b),
        Neg(_lam_hat(alg, conn, fr, projector, a, b)),
    )


def projected_modified_bracket(alg, conn, fr, projector, p, q) -> Expr:
    """``b̂(p,q) := [p,q]_E − Λ̂(p,q)`` — the projected modified
    bracket, on arbitrary section expressions."""
    return Sum(
        alg.bracket(p, q),
        Neg(_lam_hat(alg, conn, fr, projector, p, q)),
    )


def prove_e_bianchi_second_identity(
    alg: Algebroid,
    conn: Connection,
    fr: Frame,
    projector: LocalityProjector,
    u: Expr,
    v: Expr,
    w: Expr,
    x: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """E-Bianchi II, identity form (ARBITRARY connection, hand-
    derived from the engine's residual and machine-verified —
    purely definitional, no declarations):

        𝔖 [ (∇_u ᴸ̂R)(v,w)x + ᴸ̂R(ᴸT(u,v), w)x ]
          =   𝔖 ∇_{b̂([u,v], w)}x
            + 𝔖 ∇_{b̂(u, ∇_w v)}x + 𝔖 ∇_{b̂(∇_u w, v)}x
            − 𝔖 ∇_{[Λ(u,v), w]}x + 𝔖 ∇_{Λ̂(Λ(u,v), w)}x
            + 𝔖 ∇_{(1−𝒫)Λ(u,v)}(∇_w x)
            − 𝔖 ∇_u(∇_{(1−𝒫)Λ(v,w)}x).

    The first/fourth/fifth blocks combine NOTIONALLY into
    ``𝔖 ∇_{b̂(b(u,v), w)}x`` (the projected mirror of Bianchi I's
    modified-Jacobiator block; the split form is kept because the
    nested expansion exceeds the engine's convergence bound), and
    the ``(1−𝒫)Λ`` anomalies are the same gap as in the generalized
    Ricci identity. TM sanity: with ``L = 0`` all correction blocks
    except the first vanish and the identity reduces to the Phase
    4.C torsionful Bianchi II shape."""
    from jacopy.central.objects.connection import CovariantOp

    def cov(direction, arg):
        return Act(
            CovariantOp(conn.name, direction, bundle=conn.bundle),
            arg,
        )

    def R(a, b, c):
        return projected_curvature(alg, conn, fr, projector, a, b, c)

    def LT(a, b):
        return modified_torsion(alg, conn, fr, a, b)

    def cov_R_deriv(a, b, c, d):
        return Sum(
            conn(a, R(b, c, d)),
            Neg(R(conn(a, b), c, d)),
            Neg(R(b, conn(a, c), d)),
            Neg(R(b, c, conn(a, d))),
        )

    def piece(a, b, c):
        return Sum(cov_R_deriv(a, b, c, x), R(LT(a, b), c, x))

    def bhat(p, q):
        return projected_modified_bracket(
            alg, conn, fr, projector, p, q
        )

    def anom(a, b):
        return anomaly_direction(alg, conn, fr, projector, a, b)

    lhs = _cyc(piece, u, v, w)
    rhs = Sum(
        _cyc(lambda a, b, c: cov(bhat(alg.bracket(a, b), c), x), u, v, w),
        _cyc(lambda a, b, c: cov(bhat(a, conn(c, b)), x), u, v, w),
        _cyc(lambda a, b, c: cov(bhat(conn(a, c), b), x), u, v, w),
        Neg(
            _cyc(
                lambda a, b, c: cov(
                    alg.bracket(_lam(alg, conn, fr, a, b), c), x
                ),
                u,
                v,
                w,
            )
        ),
        _cyc(
            lambda a, b, c: cov(
                _lam_hat(
                    alg,
                    conn,
                    fr,
                    projector,
                    _lam(alg, conn, fr, a, b),
                    c,
                ),
                x,
            ),
            u,
            v,
            w,
        ),
        _cyc(lambda a, b, c: cov(anom(a, b), cov(c, x)), u, v, w),
        Neg(
            _cyc(
                lambda a, b, c: conn(a, cov(anom(b, c), x)), u, v, w
            )
        ),
    )
    return ExpandAndSimplify().prove(
        lhs,
        rhs,
        registry=registry,
        engine=_engine(alg, fr, registry, projectors=(projector,)),
    )
