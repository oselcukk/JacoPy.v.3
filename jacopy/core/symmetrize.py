"""
Symmetrization and anti-symmetrization ``T_(…)`` / ``T_[…]`` — PDF
item 8x.

Single-argument operators that take the symmetric or anti-symmetric
part of an object:

* :class:`Symmetrization` ``Sym(T)`` — the symmetric part
  (e.g. ``T_{(ab)} = ½(T_{ab} + T_{ba})``).
* :class:`Antisymmetrization` ``Alt(T)`` — the anti-symmetric /
  alternating part (e.g. ``T_{[ab]} = ½(T_{ab} − T_{ba})``).

In this skeleton layer the nodes are **inert**: the permutation
expansion over the slots (``½Σ_σ ± σ·T``) will be defined in a later
sub-phase (multi-argument evaluation / frame components). Degree is
preserved: ``|Sym(T)| = |Alt(T)| = |T|``; accordingly
:func:`~jacopy.algebra.derivation.degree_of` recognises them as the
degree of their argument.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.core.expr import Expr


class Symmetrization(Expr):
    """``Sym(T)`` — the symmetric part of an object (inert node)."""

    __slots__ = ("_arg",)

    def __init__(self, arg: Expr) -> None:
        if not isinstance(arg, Expr):
            raise TypeError("Symmetrization argument must be an Expr")
        self._arg = arg

    @property
    def arg(self) -> Expr:
        return self._arg

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._arg,)

    def _key(self) -> Any:
        return (self._arg,)

    def _repr_inner(self) -> str:
        return f"Sym({self._arg._repr_inner()})"


class Antisymmetrization(Expr):
    """``Alt(T)`` — the anti-symmetric (alternating) part of an object."""

    __slots__ = ("_arg",)

    def __init__(self, arg: Expr) -> None:
        if not isinstance(arg, Expr):
            raise TypeError("Antisymmetrization argument must be an Expr")
        self._arg = arg

    @property
    def arg(self) -> Expr:
        return self._arg

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._arg,)

    def _key(self) -> Any:
        return (self._arg,)

    def _repr_inner(self) -> str:
        return f"Alt({self._arg._repr_inner()})"


def symmetrize(arg: Expr) -> Symmetrization:
    """``Sym(T)`` — the symmetric part."""
    return Symmetrization(arg)


def antisymmetrize(arg: Expr) -> Antisymmetrization:
    """``Alt(T)`` — the anti-symmetric (alternating) part."""
    return Antisymmetrization(arg)
