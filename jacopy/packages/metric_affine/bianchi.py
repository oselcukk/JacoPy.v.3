"""
The Bianchi identities and the Ricci identity (Phase 4.C.2), in their
TORSIONFUL (metric-affine) form:

* **Bianchi I**:
  ``𝔖 R(X,Y)Z = 𝔖 [ T(T(X,Y), Z) + (∇_X T)(Y,Z) ]``,
* **Bianchi II**:
  ``𝔖 [ (∇_X R)(Y,Z)W + R(T(X,Y), Z)W ] = 0``,

with ``𝔖`` the cyclic sum over ``(X, Y, Z)`` and the covariant
derivatives of the tensors written out
(``(∇_X T)(Y,Z) = ∇_X(T(Y,Z)) − T(∇_X Y, Z) − T(Y, ∇_X Z)``, likewise
for ``R`` with three argument slots and the value slot).

Everything expands definitionally; the ONLY external input is the
section-level Lie Jacobi identity, entering as cited instance
theorems (:func:`~jacopy.central.tangent.lie_bracket.jacobi_combination_theorems`
— derived, not assumed). The engine surfaced exactly this: the
Bianchi I obstruction reduces to the bare Jacobi combination.

* **Ricci identity on functions**:
  ``∇_X ∇_Y f − ∇_Y ∇_X f − ∇_{[X,Y]} f = 0`` — curvature acts
  trivially on scalars; closes purely definitionally.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.proof.theorems import Theorem, TheoremBook, cite
from jacopy.central.objects.connection import Connection
from jacopy.central.tangent.lie_bracket import (
    jacobi_combination_theorems,
    lie_bracket,
)
from jacopy.packages.metric_affine.torsion_curvature import (
    curvature,
    torsion,
)


def covariant_torsion_derivative(
    conn: Connection, X: Expr, Y: Expr, Z: Expr
) -> Expr:
    """``(∇_X T)(Y, Z)`` written out."""
    return Sum(
        conn(X, torsion(conn, Y, Z)),
        Neg(torsion(conn, conn(X, Y), Z)),
        Neg(torsion(conn, Y, conn(X, Z))),
    )


def covariant_curvature_derivative(
    conn: Connection, X: Expr, Y: Expr, Z: Expr, W: Expr
) -> Expr:
    """``(∇_X R)(Y, Z)W`` written out."""
    return Sum(
        conn(X, curvature(conn, Y, Z, W)),
        Neg(curvature(conn, conn(X, Y), Z, W)),
        Neg(curvature(conn, Y, conn(X, Z), W)),
        Neg(curvature(conn, Y, Z, conn(X, W))),
    )


def _jacobi_engine(
    triples,
    f: Expr,
    registry: Optional[PropertyRegistry],
) -> Tuple[object, List[Theorem]]:
    """A metric-affine engine with the section-level Jacobi instance
    theorems for the given triples cited (both signs)."""
    from jacopy.packages.metric_affine.engine import metric_affine_engine

    engine = metric_affine_engine(registry=registry)
    book = TheoremBook()
    used: List[Theorem] = []
    for triple in triples:
        thm, thm_neg = jacobi_combination_theorems(
            *triple,
            f,
            registry=registry,
            engine=metric_affine_engine(registry=registry),
        )
        for t in (thm, thm_neg):
            if t.name not in book:
                book.add(t)
                used.append(t)
                cite(engine, book, t.name)
    return engine, used


def prove_ricci_identity_on_functions(
    conn: Connection,
    X: Expr,
    Y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``∇_X ∇_Y f − ∇_Y ∇_X f − ∇_{[X,Y]} f = 0`` — the curvature
    operator kills scalars (purely definitional: ``∇f = Xf`` and the
    bracket's commutator definition)."""
    from jacopy.packages.metric_affine.engine import metric_affine_engine

    lhs = Sum(
        conn(X, conn(Y, f)),
        Neg(conn(Y, conn(X, f))),
        Neg(conn(lie_bracket(X, Y), f)),
    )
    return ExpandAndSimplify().prove(
        lhs,
        Integer(0),
        registry=registry,
        engine=metric_affine_engine(registry=registry),
    )


def prove_bianchi_first(
    conn: Connection,
    X: Expr,
    Y: Expr,
    Z: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """The torsionful first Bianchi identity
    ``𝔖 R(X,Y)Z = 𝔖 [ T(T(X,Y),Z) + (∇_X T)(Y,Z) ]``.

    ``f`` is the generic function used to derive the cited
    section-level Jacobi instances. Returns the chain and the cited
    theorems.
    """
    cyc = [(X, Y, Z), (Y, Z, X), (Z, X, Y)]
    lhs = Sum(*(curvature(conn, a, b, c) for a, b, c in cyc))
    rhs = Sum(
        *(
            Sum(
                torsion(conn, torsion(conn, a, b), c),
                covariant_torsion_derivative(conn, a, b, c),
            )
            for a, b, c in cyc
        )
    )
    engine, used = _jacobi_engine([(X, Y, Z)], f, registry)
    chain = ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=engine
    )
    return chain, used


def _normalize(expr: Expr, engine, registry) -> Expr:
    from jacopy.algorithms.product_rule import product_rule
    from jacopy.algorithms.simplify import simplify

    current = expr
    for _ in range(8):
        for _ in range(64):
            expanded, _steps = engine.expand(current)
            after = product_rule(expanded, registry)
            if after == current:
                break
            current = after
        reduced = simplify(current, registry)
        if reduced == current:
            break
        current = reduced
    return current


def covariant_jacobi_theorems(
    conn: Connection,
    X: Expr,
    Y: Expr,
    Z: Expr,
    W: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[Theorem, Theorem]:
    """The Jacobi combination inside the connection's DIRECTION slot:
    ``Σ ±∇_{[·,[·,·]]} W = 0`` — the instance theorem Bianchi II
    consumes.

    Two mechanical legs on ``∇_V W`` with ``V`` the Jacobi
    combination: (1) direction linearity splits it into the residual
    shape; (2) with the section-level Jacobi instances cited, the slot
    protocol collapses ``V`` to zero. Composing: the residual shape
    vanishes.
    """
    from jacopy.proof.step import ProofStep
    from jacopy.packages.metric_affine.engine import metric_affine_engine

    V = Sum(
        lie_bracket(X, lie_bracket(Y, Z)),
        lie_bracket(Y, lie_bracket(Z, X)),
        lie_bracket(Z, lie_bracket(X, Y)),
    )
    node = conn(V, W)

    # Leg 1: direction linearity splits ∇_V W into the residual shape
    # (mechanical, verified by a closing prove).
    plain = metric_affine_engine(registry=registry)
    shape = _normalize(node, plain, registry)
    leg1 = ExpandAndSimplify().prove(
        node, shape, registry=registry, engine=metric_affine_engine(registry=registry)
    )

    # Leg 2: the direction itself vanishes — the cited section-level
    # Jacobi instance.
    cited_engine, _ = _jacobi_engine([(X, Y, Z)], f, registry)
    leg2 = ExpandAndSimplify().prove(
        V,
        Integer(0),
        registry=registry,
        engine=cited_engine,
    )

    step = ProofStep(
        shape,
        Integer(0),
        rule="Jacobi inside the connection direction",
        justification=(
            "the shape equals ∇_V W by direction linearity (leg 1), "
            "V = 0 by the Jacobi instance (leg 2), and ∇_0 W = 0"
        ),
    )
    for s in list(leg1) + list(leg2):
        step.add_child(s)
    chain = ProofChain([step])

    from jacopy.algorithms.simplify import simplify

    names = (X._repr_inner(), Y._repr_inner(), Z._repr_inner())
    thm = Theorem(
        name=(
            f"covariant_jacobi_{conn.name}_" + "_".join(names)
            + f"_{W._repr_inner()}"
        ),
        statement=(
            f"Σ_cyc ∇_[{names[0]},[{names[1]},{names[2]}]] "
            f"{W._repr_inner()} = 0"
        ),
        lhs=shape,
        rhs=Integer(0),
        proof=chain,
        generality="instance",
        from_axioms=(
            "Lie bracket definition",
            "connection direction linearity",
        ),
    )
    neg_chain = ProofChain(list(chain.steps))
    neg_chain.append(
        ProofStep(
            simplify(Neg(shape), registry),
            Integer(0),
            rule="negate both sides",
            justification="the combination vanishes, so does its negation",
        )
    )
    thm_neg = Theorem(
        name=thm.name + "_neg",
        statement=f"−({thm.statement})",
        lhs=simplify(Neg(shape), registry),
        rhs=Integer(0),
        proof=neg_chain,
        generality="instance",
        from_axioms=thm.from_axioms,
    )
    return thm, thm_neg


def prove_bianchi_second(
    conn: Connection,
    X: Expr,
    Y: Expr,
    Z: Expr,
    W: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """The torsionful second Bianchi identity
    ``𝔖 [ (∇_X R)(Y,Z)W + R(T(X,Y), Z)W ] = 0``."""
    from jacopy.proof.theorems import cite as _cite

    cyc = [(X, Y, Z), (Y, Z, X), (Z, X, Y)]
    lhs = Sum(
        *(
            Sum(
                covariant_curvature_derivative(conn, a, b, c, W),
                curvature(conn, torsion(conn, a, b), c, W),
            )
            for a, b, c in cyc
        )
    )
    engine, used = _jacobi_engine([(X, Y, Z)], f, registry)
    book = TheoremBook()
    for t in covariant_jacobi_theorems(
        conn, X, Y, Z, W, f, registry=registry
    ):
        book.add(t)
        used.append(t)
        _cite(engine, book, t.name)
    chain = ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=engine
    )
    return chain, used
