"""
The Koszul formula family (Phase 4.C).

* :func:`prove_generalized_koszul` — the **generalized Koszul
  identity** (the Schouten decomposition in identity form), valid for
  an ARBITRARY connection and metric, purely definitionally:

      2·g(∇_X Y, Z) = X g(Y,Z) + Y g(X,Z) − Z g(X,Y)
                    + g([X,Y],Z) − g([X,Z],Y) − g([Y,Z],X)
                    − Q(X,Y,Z) − Q(Y,X,Z) + Q(Z,X,Y)
                    + g(T(X,Y),Z) − g(T(X,Z),Y) − g(T(Y,Z),X).

  The Q-block is (twice) the disformation and the T-block (twice) the
  contortion paired against ``Z`` — setting them to zero is exactly
  the Levi-Civita specialization.
* :func:`prove_koszul_formula` — the classical Koszul formula, closing
  under the declared ``metric-compatible`` + ``torsion-free`` rules.
* :func:`prove_levi_civita_unique` — uniqueness: two compatible
  torsion-free connections satisfy the same Koszul right-hand side,
  and non-degeneracy (definitional for a metric) cancels the generic
  third slot — a tactic in the Phase 3.D architecture.
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.objects.connection import Connection
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.metric_affine.metric import Metric, nonmetricity
from jacopy.packages.metric_affine.torsion_curvature import torsion


def _koszul_rhs(
    conn: Connection, g: Metric, X: Expr, Y: Expr, Z: Expr, *, with_defects: bool
) -> Expr:
    """The Koszul right-hand side; ``with_defects`` adds the Q- and
    T-blocks of the generalized identity."""
    terms = [
        Act(X, g(Y, Z)),
        Act(Y, g(X, Z)),
        Neg(Act(Z, g(X, Y))),
        g(lie_bracket(X, Y), Z),
        Neg(g(lie_bracket(X, Z), Y)),
        Neg(g(lie_bracket(Y, Z), X)),
    ]
    if with_defects:
        terms += [
            Neg(nonmetricity(conn, g, X, Y, Z)),
            Neg(nonmetricity(conn, g, Y, X, Z)),
            nonmetricity(conn, g, Z, X, Y),
            g(torsion(conn, X, Y), Z),
            Neg(g(torsion(conn, X, Z), Y)),
            Neg(g(torsion(conn, Y, Z), X)),
        ]
    return Sum(*terms)


def prove_generalized_koszul(
    conn: Connection,
    g: Metric,
    X: Expr,
    Y: Expr,
    Z: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """The generalized Koszul identity — NO declarations needed: every
    Q and T expands definitionally and the derivative terms cancel."""
    from jacopy.packages.metric_affine.engine import metric_affine_engine

    lhs = Product(Integer(2), g(conn(X, Y), Z))
    return ExpandAndSimplify().prove(
        lhs,
        _koszul_rhs(conn, g, X, Y, Z, with_defects=True),
        registry=registry,
        engine=metric_affine_engine(registry=registry),
    )


def prove_koszul_formula(
    conn: Connection,
    g: Metric,
    X: Expr,
    Y: Expr,
    Z: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """The classical Koszul formula — closes under the declared
    ``metric-compatible (∇, g)`` + ``torsion-free ∇`` rules; fails
    honestly without them."""
    from jacopy.packages.metric_affine.engine import metric_affine_engine

    lhs = Product(Integer(2), g(conn(X, Y), Z))
    return ExpandAndSimplify().prove(
        lhs,
        _koszul_rhs(conn, g, X, Y, Z, with_defects=False),
        registry=registry,
        engine=metric_affine_engine(
            registry=registry,
            compatible=((conn, g),),
            torsion_free=(conn,),
        ),
    )


def prove_levi_civita_unique(
    conn1: Connection,
    conn2: Connection,
    g: Metric,
    X: Expr,
    Y: Expr,
    Z: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Uniqueness of the Levi-Civita connection: two metric-compatible
    torsion-free connections agree.

    Both satisfy the classical Koszul formula whose right-hand side
    does not mention the connection; hence
    ``g(∇_X Y, Z) = g(∇'_X Y, Z)`` for the generic ``Z``, and the
    metric's definitional non-degeneracy gives ``∇_X Y = ∇'_X Y``
    (agreement on generators).
    """
    from jacopy.packages.metric_affine.engine import metric_affine_engine

    if conn1 == conn2:
        raise ValueError(
            "Levi-Civita uniqueness needs two distinct connection "
            "contexts"
        )

    rhs = _koszul_rhs(conn1, g, X, Y, Z, with_defects=False)

    def _leg(conn: Connection) -> ProofChain:
        return ExpandAndSimplify().prove(
            Product(Integer(2), g(conn(X, Y), Z)),
            rhs,
            registry=registry,
            engine=metric_affine_engine(
                registry=registry,
                compatible=((conn, g),),
                torsion_free=(conn,),
            ),
        )

    leg1 = _leg(conn1)
    leg2 = _leg(conn2)

    step1 = ProofStep(
        Product(Integer(2), g(conn1(X, Y), Z)),
        rhs,
        rule=f"Koszul formula for {conn1.name}",
        justification="metric compatibility + torsion-freeness",
    )
    for s in leg1:
        step1.add_child(s)
    step2 = ProofStep(
        Product(Integer(2), g(conn2(X, Y), Z)),
        rhs,
        rule=f"Koszul formula for {conn2.name}",
        justification="metric compatibility + torsion-freeness",
    )
    for s in leg2:
        step2.add_child(s)
    step3 = ProofStep(
        g(conn1(X, Y), Z),
        g(conn2(X, Y), Z),
        rule="same Koszul right-hand side",
        justification=(
            "the right-hand side does not mention the connection; "
            "divide by 2"
        ),
    )
    step4 = ProofStep(
        conn1(X, Y),
        conn2(X, Y),
        rule="non-degeneracy of g (agreement on generic section Z)",
        justification=(
            "g(∇_X Y − ∇'_X Y, Z) = 0 for the generic section Z and "
            "the metric is non-degenerate by definition"
        ),
    )
    chain = ProofChain([step1, step2, step3, step4])

    names = (X._repr_inner(), Y._repr_inner())
    theorem = Theorem(
        name=(
            f"levi_civita_unique_{conn1.name}_{conn2.name}_"
            f"{names[0]}_{names[1]}"
        ),
        statement=(
            f"{conn1.name}_{names[0]} {names[1]} = "
            f"{conn2.name}_{names[0]} {names[1]} "
            "(metric-compatible torsion-free connections agree)"
        ),
        lhs=conn1(X, Y),
        rhs=conn2(X, Y),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"metric compatibility ({conn1.name}, {g.name})",
            f"metric compatibility ({conn2.name}, {g.name})",
            f"torsion-free ({conn1.name})",
            f"torsion-free ({conn2.name})",
            "non-degeneracy of the metric (definitional)",
        ),
        notes="Koszul right-hand side is connection-free",
    )
    return chain, theorem
