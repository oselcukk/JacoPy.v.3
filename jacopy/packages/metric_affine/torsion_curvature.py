"""
Torsion and curvature (Phase 4.B).

Canonical definitions (PDF item 11; definition policy §1 — one
definition, everything else a theorem):

    T(X, Y)   := ∇_X Y − ∇_Y X − [X, Y],
    R(X, Y)Z  := ∇_X ∇_Y Z − ∇_Y ∇_X Z − ∇_{[X,Y]} Z.

Both are opaque section-valued atoms with an always-on definitional
expansion; the classical facts about them are THEOREMS proved from
the expansion:

* ``C^∞``-multilinearity in every slot (tensoriality) — the proofs
  drive the Leibniz terms against each other; the section-level
  bracket module structure
  (:class:`~jacopy.central.tangent.lie_bracket.LieBracketLeibnizDefinition`,
  itself a derived rule) supplies the ``[fX, Y]`` splits.
* Antisymmetry ``T(X,Y) = −T(Y,X)``, ``R(X,Y)Z = −R(Y,X)Z``.
* The difference tensor ``Δ(∇,∇')(X,Y) := ∇_X Y − ∇'_X Y`` is
  ``C^∞``-bilinear — no node needed, the statement is about the
  explicit difference (:func:`prove_connection_difference_tensorial`).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Expr, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.objects.bundle import Bundle, TM
from jacopy.central.objects.connection import Connection
from jacopy.central.tangent.lie_bracket import lie_bracket


class Torsion(Derivation):
    """``T(X, Y)`` — the torsion of a connection, itself a section
    (degree-0 atom, wedge degree 1; slot protocol on ``X, Y``)."""

    __slots__ = ("_connection_name", "_bundle", "_X", "_Y")

    def __init__(
        self,
        connection_name: str,
        X: Expr,
        Y: Expr,
        *,
        bundle: Optional[Bundle] = None,
        name: Optional[str] = None,
    ) -> None:
        for s in (X, Y):
            if not isinstance(s, Expr):
                raise TypeError("Torsion requires Expr arguments")
        display = (
            name
            if name is not None
            else f"T({X._repr_inner()},{Y._repr_inner()})"
        )
        super().__init__(display, degree=0)
        self._connection_name = connection_name
        self._bundle = bundle if bundle is not None else TM
        self._X = X
        self._Y = Y

    @property
    def connection_name(self) -> str:
        return self._connection_name

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    @property
    def arguments(self) -> Tuple[Expr, Expr]:
        return (self._X, self._Y)

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._X, self._Y)

    def with_slots(self, X: Expr, Y: Expr) -> "Torsion":
        return Torsion(
            self._connection_name, X, Y, bundle=self._bundle
        )

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._connection_name,
            self._bundle,
            self._X,
            self._Y,
        )


class Curvature(Derivation):
    """``R(X, Y)Z`` — the curvature of a connection applied to ``Z``,
    itself a section (degree-0 atom, wedge degree 1)."""

    __slots__ = ("_connection_name", "_bundle", "_X", "_Y", "_Z")

    def __init__(
        self,
        connection_name: str,
        X: Expr,
        Y: Expr,
        Z: Expr,
        *,
        bundle: Optional[Bundle] = None,
        name: Optional[str] = None,
    ) -> None:
        for s in (X, Y, Z):
            if not isinstance(s, Expr):
                raise TypeError("Curvature requires Expr arguments")
        display = (
            name
            if name is not None
            else (
                f"R({X._repr_inner()},{Y._repr_inner()})"
                f"{Z._repr_inner()}"
            )
        )
        super().__init__(display, degree=0)
        self._connection_name = connection_name
        self._bundle = bundle if bundle is not None else TM
        self._X = X
        self._Y = Y
        self._Z = Z

    @property
    def connection_name(self) -> str:
        return self._connection_name

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    @property
    def arguments(self) -> Tuple[Expr, Expr, Expr]:
        return (self._X, self._Y, self._Z)

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._X, self._Y, self._Z)

    def with_slots(self, X: Expr, Y: Expr, Z: Expr) -> "Curvature":
        return Curvature(
            self._connection_name, X, Y, Z, bundle=self._bundle
        )

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._connection_name,
            self._bundle,
            self._X,
            self._Y,
            self._Z,
        )


def torsion(conn: Connection, X: Expr, Y: Expr) -> Torsion:
    """``T(X, Y)`` of the given connection."""
    if not isinstance(conn, Connection):
        raise TypeError("torsion expects a Connection")
    return Torsion(conn.name, X, Y, bundle=conn.bundle)


def curvature(conn: Connection, X: Expr, Y: Expr, Z: Expr) -> Curvature:
    """``R(X, Y)Z`` of the given connection."""
    if not isinstance(conn, Connection):
        raise TypeError("curvature expects a Connection")
    return Curvature(conn.name, X, Y, Z, bundle=conn.bundle)


def _nabla(node, X: Expr, T: Expr) -> Expr:
    from jacopy.central.objects.connection import CovariantOp

    return Act(
        CovariantOp(node.connection_name, X, bundle=node.bundle), T
    )


class TorsionExpansionDefinition(Definition):
    """``T(X, Y) → ∇_X Y − ∇_Y X − [X, Y]`` (canonical definition —
    the TANGENT case: the bracket emitted is the Lie bracket, so the
    rule guards on the tangent bundle; algebroid torsion expands via
    :class:`~jacopy.packages.metric_affine.e_connection.\
ETorsionExpansionDefinition` with the algebroid bracket)."""

    name = "torsion definition: T(X,Y) = ∇_X Y − ∇_Y X − [X,Y]"
    anchor = Torsion

    def matches(self, expr: Expr) -> bool:
        from jacopy.central.objects.bundle import TangentBundle

        return isinstance(expr, Torsion) and isinstance(
            expr.bundle, TangentBundle
        )

    def rewrite(self, expr: Expr) -> Expr:
        X, Y = expr.arguments
        return Sum(
            _nabla(expr, X, Y),
            Neg(_nabla(expr, Y, X)),
            Neg(lie_bracket(X, Y)),
        )


class CurvatureExpansionDefinition(Definition):
    """``R(X, Y)Z → ∇_X ∇_Y Z − ∇_Y ∇_X Z − ∇_{[X,Y]} Z``
    (canonical definition)."""

    name = (
        "curvature definition: R(X,Y)Z = ∇_X∇_Y Z − ∇_Y∇_X Z − ∇_{[X,Y]}Z"
    )
    anchor = Curvature

    def matches(self, expr: Expr) -> bool:
        from jacopy.central.objects.bundle import TangentBundle

        return isinstance(expr, Curvature) and isinstance(
            expr.bundle, TangentBundle
        )

    def rewrite(self, expr: Expr) -> Expr:
        X, Y, Z = expr.arguments
        return Sum(
            _nabla(expr, X, _nabla(expr, Y, Z)),
            Neg(_nabla(expr, Y, _nabla(expr, X, Z))),
            Neg(_nabla(expr, lie_bracket(X, Y), Z)),
        )


# --------------------------------------------------------------------- #
# Theorems                                                               #
# --------------------------------------------------------------------- #


def _engine(registry: Optional[PropertyRegistry]):
    from jacopy.packages.metric_affine.engine import metric_affine_engine

    return metric_affine_engine(registry=registry)


def prove_torsion_tensorial(
    conn: Connection,
    X: Expr,
    Y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, ProofChain]:
    """``T(fX, Y) = f·T(X,Y)`` and ``T(X, fY) = f·T(X,Y)`` — the
    Leibniz terms of ``∇`` cancel against the bracket's module
    structure (both derived, not assumed)."""
    first = ExpandAndSimplify().prove(
        torsion(conn, Product(f, X), Y),
        Product(f, torsion(conn, X, Y)),
        registry=registry,
        engine=_engine(registry),
    )
    second = ExpandAndSimplify().prove(
        torsion(conn, X, Product(f, Y)),
        Product(f, torsion(conn, X, Y)),
        registry=registry,
        engine=_engine(registry),
    )
    return first, second


def prove_torsion_antisymmetry(
    conn: Connection,
    X: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``T(X, Y) = −T(Y, X)``."""
    return ExpandAndSimplify().prove(
        torsion(conn, X, Y),
        Neg(torsion(conn, Y, X)),
        registry=registry,
        engine=_engine(registry),
    )


def prove_curvature_tensorial(
    conn: Connection,
    X: Expr,
    Y: Expr,
    Z: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, ProofChain, ProofChain]:
    """``R(fX,Y)Z = R(X,fY)Z = R(X,Y)(fZ) = f·R(X,Y)Z`` — slot by
    slot. The third slot is the classical double-Leibniz computation:
    the ``X(Y(f))`` second derivatives cancel against the commutator
    ``[X,Y](f)``."""
    target = Product(f, curvature(conn, X, Y, Z))
    chains = []
    for lhs in (
        curvature(conn, Product(f, X), Y, Z),
        curvature(conn, X, Product(f, Y), Z),
        curvature(conn, X, Y, Product(f, Z)),
    ):
        chains.append(
            ExpandAndSimplify().prove(
                lhs,
                target,
                registry=registry,
                engine=_engine(registry),
            )
        )
    return tuple(chains)


def prove_curvature_antisymmetry(
    conn: Connection,
    X: Expr,
    Y: Expr,
    Z: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``R(X, Y)Z = −R(Y, X)Z``."""
    return ExpandAndSimplify().prove(
        curvature(conn, X, Y, Z),
        Neg(curvature(conn, Y, X, Z)),
        registry=registry,
        engine=_engine(registry),
    )


def prove_connection_difference_tensorial(
    conn1: Connection,
    conn2: Connection,
    X: Expr,
    Y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, ProofChain]:
    """``Δ(∇,∇')(X,Y) := ∇_X Y − ∇'_X Y`` is ``C^∞``-bilinear: the
    direction slot by connection linearity, the argument slot because
    the two Leibniz terms ``X(f)·Y`` cancel between the connections
    [MC Prop 3.1 pattern]."""

    def delta(a: Expr, b: Expr) -> Expr:
        return Sum(conn1(a, b), Neg(conn2(a, b)))

    first = ExpandAndSimplify().prove(
        delta(Product(f, X), Y),
        Product(f, delta(X, Y)),
        registry=registry,
        engine=_engine(registry),
    )
    second = ExpandAndSimplify().prove(
        delta(X, Product(f, Y)),
        Product(f, delta(X, Y)),
        registry=registry,
        engine=_engine(registry),
    )
    return first, second
