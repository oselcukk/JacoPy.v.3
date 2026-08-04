"""
Functions — ``C^∞(M)``, smooth functions on the base manifold
(PDF item 8d).

Functions are **bundle-independent**: whatever the bundle ``E`` is,
scalar fields always live on the base manifold ``M`` and carry degree
0. For that reason :func:`functions` takes no ``bundle=`` parameter.

A function is represented in the core expression tree by a
:class:`~jacopy.core.expr.Symbol`; its grading
(``Graded(degree=0)``) is declared on the given
:class:`~jacopy.core.registry.PropertyRegistry`. The action of a
vector field on a function ``U(f)`` is defined in
:mod:`~jacopy.central.objects.vector_field`.
"""

from __future__ import annotations

from typing import Tuple

from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry


def functions(
    names: str,
    *,
    registry: PropertyRegistry,
    degree: int = 0,
) -> Tuple[Symbol, ...]:
    """Declare one or more ``C^∞(M)`` functions.

    ``functions("f g h", registry=reg)`` returns three
    :class:`Symbol`\\ s; each is declared on ``reg`` with
    :class:`Graded(degree=0)`. Even for a single name the return value
    is a tuple: ``(f,) = functions("f", registry=reg)``.

    Parameters
    ----------
    names
        Whitespace-separated name(s).
    registry
        The registry the grading is declared on. An explicit
        ``registry=`` is mandatory; there is no hidden singleton, so
        property state stays local.
    degree
        Grading degree; defaults to 0 (classical ``C^∞(M)``). A
        different degree may occasionally be wanted for a p-form
        generator.
    """
    if not isinstance(names, str):
        raise TypeError("names must be a whitespace-separated str")
    if not isinstance(degree, int):
        raise TypeError("degree must be an int")
    parts = tuple(names.split())
    if not parts:
        raise ValueError("at least one name is required")
    syms = tuple(Symbol(n) for n in parts)
    for s in syms:
        registry.declare(s, Graded(degree=degree))
    return syms
