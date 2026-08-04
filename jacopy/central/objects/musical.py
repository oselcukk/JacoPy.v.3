"""
Musical isomorphisms ``♯`` / ``♭`` — passing between ``TM`` and
``T*M`` (PDF item 8w).

**The essence of item 8w.** A bilinear map of the form
``TM × TM → C^∞(M)`` (e.g. a metric ``g(X, Y)`` or a bivector
``π(α, β)``) can be turned into a map ``TM → T*M`` (or ``T*M → TM``)
by leaving one slot open. This is the coordinate-free form of index
raising/lowering:

* :class:`Flat` ``g^♭ : TM → T*M`` — ``g^♭(X) := g(X, ·)``. Lowers a
  vector field to a 1-form. Raises the grading by 1 (vector degree 0 →
  1-form degree 1), i.e. degree ``+1``.
* :class:`Sharp` ``π^♯ : T*M → TM`` — ``π^♯(α) := π(α, ·)``. Raises a
  1-form to a vector field. Lowers the grading by 1 (1-form degree 1 →
  vector degree 0), i.e. degree ``−1``.

When ``g`` and ``π`` come from the same geometry and are mutually
inverse (symplectic / Poisson), ``♯`` and ``♭`` are inverse to each
other; that musical compatibility is handled in Phase 5.

**Tensoriality.** Musical maps are ``C^∞(M)``-linear
(``π^♯(fα) = f · π^♯(α)``), i.e. they are *not derivations* — no
Leibniz rule applies over products. They are therefore modelled as
:class:`~jacopy.core.expr.Atom` subclasses (not as a
:class:`~jacopy.algebra.derivation.Derivation`); the degree is carried
on the instance and recognized by ``degree_of``.

The "generalization to arbitrary numbers" of item 8w (musical views of
p-linear maps, ``(E)^k → C^∞`` ↔ ``E^{k-j} → (E^*)^j``) is **deferred**
and tracked in the ROADMAP (Phase 2/4 entry); this module covers the
bilinear case.
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Atom, Expr
from jacopy.core.symbolic_degree import Degree, as_degree


class Flat(Atom):
    """``g^♭ : TM → T*M`` — lowers a vector to a 1-form (degree ``+1``).

    ``g^♭(X) = g(X, ·)``. Tensorial (no Leibniz).
    """

    __slots__ = ("_metric", "_name", "_degree")

    def __init__(self, g: Expr, *, name: Optional[str] = None) -> None:
        if not isinstance(g, Expr):
            raise TypeError("Flat requires an Expr metric/2-form")
        self._metric = g
        self._name = name if name is not None else f"{g._repr_inner()}♭"
        self._degree = as_degree(1)

    @property
    def metric(self) -> Expr:
        return self._metric

    @property
    def degree(self) -> Degree:
        return self._degree

    def _key(self) -> Any:
        return (self._name, self._degree, self._metric)

    def _repr_inner(self) -> str:
        return self._name

    def __call__(self, X: Expr) -> Act:
        """``g^♭(X) = g(X, ·)`` — lower a vector to a 1-form."""
        if not isinstance(X, Expr):
            raise TypeError("Flat call argument must be an Expr")
        return Act(self, X)


class Sharp(Atom):
    """``π^♯ : T*M → TM`` — raises a 1-form to a vector (degree ``−1``).

    ``π^♯(α) = π(α, ·)``. Tensorial (no Leibniz).
    """

    __slots__ = ("_bivector", "_name", "_degree")

    def __init__(self, pi: Expr, *, name: Optional[str] = None) -> None:
        if not isinstance(pi, Expr):
            raise TypeError("Sharp requires an Expr bivector")
        self._bivector = pi
        self._name = name if name is not None else f"{pi._repr_inner()}♯"
        self._degree = as_degree(-1)

    @property
    def bivector(self) -> Expr:
        return self._bivector

    @property
    def degree(self) -> Degree:
        return self._degree

    def _key(self) -> Any:
        return (self._name, self._degree, self._bivector)

    def _repr_inner(self) -> str:
        return self._name

    def __call__(self, alpha: Expr) -> Act:
        """``π^♯(α) = π(α, ·)`` — raise a 1-form to a vector."""
        if not isinstance(alpha, Expr):
            raise TypeError("Sharp call argument must be an Expr")
        return Act(self, alpha)


def flat(g: Expr, *, name: Optional[str] = None) -> Flat:
    """Build the ``g^♭`` musical (flat) operator."""
    return Flat(g, name=name)


def sharp(pi: Expr, *, name: Optional[str] = None) -> Sharp:
    """Build the ``π^♯`` musical (sharp) operator."""
    return Sharp(pi, name=name)
